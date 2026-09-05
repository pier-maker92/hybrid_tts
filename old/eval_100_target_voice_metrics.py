#!/usr/bin/env python
import argparse
import gc
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torchaudio
from jiwer import cer as compute_cer
from jiwer import wer as compute_wer

from bench_batch_inference_target_voice import (
    build_phoneme_ids,
    clear_cuda,
    latest_checkpoint,
    load_librispeech_samples,
    run_chunk,
)
from evaluation import WhisperASR
from inference import (
    load_hybrid_model,
    load_kmeans_centroids,
    load_vae,
    load_vocoder,
    resolve_local_research_path,
)
from util import build_tokenizer


class UTMOSAny:
    def __init__(self, device):
        self.device = device
        self.kind = None
        self.model = None
        try:
            import utmosv2

            self.model = utmosv2.create_model(pretrained=True, device=str(device))
            self.kind = "utmosv2"
            return
        except Exception as exc:
            print(f"UTMOSv2 unavailable: {exc}", flush=True)

        try:
            import utmos

            self.model = utmos.Score()
            self.kind = "utmos"
        except Exception as exc:
            print(f"UTMOS unavailable: {exc}", flush=True)

    @torch.no_grad()
    def predict(self, wav_path):
        if self.model is None:
            return None
        if self.kind == "utmosv2":
            return float(self.model.predict(input_path=str(wav_path), device=str(self.device), num_workers=0))
        return float(self.model(str(wav_path)))


def load_models(args, device):
    checkpoint = latest_checkpoint(args.checkpoint_run)
    with open(os.path.join(checkpoint, "config.json"), "r") as f:
        cfg = json.load(f)
    scratch_dir = os.environ.get("SCRATCH", "/scratch/piermel")
    cfg["vae_checkpoint"] = resolve_local_research_path(cfg.get("vae_checkpoint"), scratch_dir)
    cfg["kmeans_path"] = resolve_local_research_path(cfg.get("kmeans_path"), scratch_dir)
    dtype = (
        torch.bfloat16
        if device.type == "cuda"
        and cfg.get("training", {}).get("bf16")
        and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    with open("data/phoneme_vocab.json", "r") as f:
        phoneme_vocab = json.load(f)
    tok = build_tokenizer(cfg, pretrinaed=False)
    kmeans_centroids = load_kmeans_centroids(cfg.get("kmeans_path"), device=device, dtype=dtype)
    hybrid_model = load_hybrid_model(cfg, checkpoint, device, dtype, tokenizer=tok)
    vae = load_vae(cfg["vae_checkpoint"], device, dtype)
    if vae is None:
        raise RuntimeError("Could not load VAE")
    if getattr(vae, "speaker_encoder_type", None) == "wavlm" and getattr(vae, "speaker_encoder", None) is not None:
        vae.speaker_encoder = vae.speaker_encoder.to(device=device, dtype=torch.float32)
    vocoder = load_vocoder(cfg.get("vocoder_checkpoint", "vocos"), device)
    if vocoder is None:
        raise RuntimeError("Could not load vocoder")
    return checkpoint, cfg, dtype, phoneme_vocab, (hybrid_model, vae, vocoder, tok, kmeans_centroids)


def mean_present(values):
    present = [v for v in values if v is not None]
    return None if not present else float(sum(present) / len(present))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default=None)
    p.add_argument("--checkpoint_run", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_ref_seconds", type=float, default=20.0)
    p.add_argument("--max_audio_seconds", type=float, default=60.0)
    p.add_argument("--max_len", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num_steps", type=int, default=4)
    p.add_argument("--diffusion_temperature", type=float, default=0.3)
    p.add_argument("--guidance_scale", type=float, default=1.2)
    p.add_argument("--vae_num_steps", type=int, default=16)
    p.add_argument("--vae_temperature", type=float, default=0.2)
    p.add_argument("--vae_guidance_scale", type=float, default=1.3)
    p.add_argument("--whisper_model", default="openai/whisper-large-v3")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    wav_dir = out_dir / "wavs"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    args.dataset_root = args.dataset_root or os.path.join(
        os.environ["SLURM_TMPDIR"], "datasets", "librispeech-aligned"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, parquet_files = load_librispeech_samples(args.dataset_root, args.num_samples, args.max_ref_seconds)
    checkpoint, cfg, dtype, phoneme_vocab, models = load_models(args, device)
    tok = models[3]
    prompts = build_phoneme_ids([sample["text"] for sample in samples], phoneme_vocab, tok)

    bench_args = SimpleNamespace(**vars(args))
    bench_args.save_wavs_per_run = 100000
    records = []
    gen_start = time.time()
    for start in range(0, len(samples), args.batch_size):
        end = min(start + args.batch_size, len(samples))
        chunk_records = run_chunk(
            samples[start:end],
            prompts[start:end],
            models,
            bench_args,
            device,
            dtype,
            save_dir=wav_dir,
        )
        records.extend(chunk_records)
        partial = {
            "checkpoint": checkpoint,
            "dataset_root": args.dataset_root,
            "parquet_files": parquet_files,
            "progress": {"generated": len(records), "total": len(samples)},
        }
        with open(out_dir / "metrics_report.json", "w") as f:
            json.dump(partial, f, indent=2)
        print(json.dumps(partial["progress"]), flush=True)

    generation_seconds = time.time() - gen_start
    del models
    gc.collect()
    clear_cuda()

    print("Loading ASR/UTMOS metrics models", flush=True)
    asr = WhisperASR(device, model_name=args.whisper_model)
    utmos = UTMOSAny(device)

    metrics = []
    metric_start = time.time()
    for idx, rec in enumerate(records):
        sample = samples[idx]
        wav_path = wav_dir / f"{sample['id'].replace('/', '_')}.wav"
        gen_wav, gen_sr = torchaudio.load(str(wav_path))
        hyp = asr.transcribe(gen_wav.squeeze(0), gen_sr)
        ref = sample["text"].lower().strip()
        utmos_score = utmos.predict(str(wav_path))
        item = {
            **rec,
            "reference_text": ref,
            "asr_text": hyp,
            "WER": float(compute_wer(ref, hyp)),
            "CER": float(compute_cer(ref, hyp)),
            "UTMOS": utmos_score,
            "wav_path": str(wav_path),
        }
        metrics.append(item)
        if (idx + 1) % 10 == 0 or idx + 1 == len(records):
            print(f"metrics {idx + 1}/{len(records)}", flush=True)
            with open(out_dir / "metrics_report.json", "w") as f:
                json.dump({"progress": {"metrics": idx + 1, "total": len(records)}}, f, indent=2)

    summary = {
        "checkpoint": checkpoint,
        "dataset_root": args.dataset_root,
        "num_samples": len(metrics),
        "batch_size": args.batch_size,
        "generation_seconds": generation_seconds,
        "metrics_seconds": time.time() - metric_start,
        "WER_mean": mean_present([m["WER"] for m in metrics]),
        "CER_mean": mean_present([m["CER"] for m in metrics]),
        "UTMOS_mean": mean_present([m["UTMOS"] for m in metrics]),
        "UTMOS_backend": utmos.kind,
        "speaker_cosine_vae_mean": mean_present([m["speaker_cosine_vae"] for m in metrics]),
        "speaker_cosine_vae_min": min(m["speaker_cosine_vae"] for m in metrics),
        "generated_duration_s_mean": mean_present([m["duration_gen_s"] for m in metrics]),
        "metrics": metrics,
    }
    with open(out_dir / "metrics_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
