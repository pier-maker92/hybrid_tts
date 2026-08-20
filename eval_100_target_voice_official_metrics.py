#!/usr/bin/env python
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torchaudio

from bench_batch_inference_target_voice import (
    build_phoneme_ids,
    clear_cuda,
    latest_checkpoint,
    load_librispeech_samples,
    run_chunk,
)
from inference import (
    load_hybrid_model,
    load_kmeans_centroids,
    load_vae,
    load_vocoder,
    resolve_local_research_path,
)
from util import build_tokenizer

SR = 24000


def load_tts_models(args, device):
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


def pad_pair(hyp, ref):
    hyp = hyp.squeeze().float()
    ref = ref.squeeze().float()
    hyp_len = hyp.numel()
    ref_len = ref.numel()
    max_len = max(hyp_len, ref_len)
    if hyp_len < max_len:
        hyp = torch.nn.functional.pad(hyp, (0, max_len - hyp_len))
    if ref_len < max_len:
        ref = torch.nn.functional.pad(ref, (0, max_len - ref_len))
    lens = torch.tensor([hyp_len / max_len], dtype=torch.float32)
    return hyp.unsqueeze(0), ref.unsqueeze(0), lens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default=None)
    p.add_argument("--checkpoint_run", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--audiocodecs_root", default="/scratch/piermel/audiocodecs")
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
    p.add_argument("--dwer_model", default="small")
    p.add_argument("--dwer_device", default="cuda")
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

    checkpoint, cfg, dtype, phoneme_vocab, models = load_tts_models(args, device)
    prompts = build_phoneme_ids([sample["text"] for sample in samples], phoneme_vocab, models[3])

    bench_args = SimpleNamespace(**vars(args))
    bench_args.save_wavs_per_run = 100000
    records = []
    gen_start = time.time()
    for start in range(0, len(samples), args.batch_size):
        end = min(start + args.batch_size, len(samples))
        records.extend(
            run_chunk(
                samples[start:end],
                prompts[start:end],
                models,
                bench_args,
                device,
                dtype,
                save_dir=wav_dir,
            )
        )
        with open(out_dir / "official_metrics_report.json", "w") as f:
            json.dump({"progress": {"generated": len(records), "total": len(samples)}}, f, indent=2)
        print(f"generated {len(records)}/{len(samples)}", flush=True)
    generation_seconds = time.time() - gen_start

    del models
    gc.collect()
    clear_cuda()

    metrics_root = os.path.join(args.audiocodecs_root, "downstream")
    sys.path.insert(0, metrics_root)
    from metrics.dwer import DWER
    from metrics.speaker_similarity import SpkSimWavLM
    from metrics.utmos import UTMOS
    sys.path.pop(0)

    hf_cache = os.path.join(os.environ.get("HF_HOME", "/scratch/piermel/.cache/huggingface"), "hub")
    metric_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading official metrics", flush=True)
    dwer_metric = DWER(args.dwer_model, SR, save_path=hf_cache, device=args.dwer_device)
    utmos_metric = UTMOS(SR)
    spksim_metric = SpkSimWavLM("microsoft/wavlm-base-sv", SR, save_path=hf_cache)

    rows = []
    metric_start = time.time()
    for idx, rec in enumerate(records):
        sample = samples[idx]
        hyp_path = wav_dir / f"{sample['id'].replace('/', '_')}.wav"
        hyp, hyp_sr = torchaudio.load(str(hyp_path))
        ref = sample["wav"].unsqueeze(0)
        if sample["sr"] != SR:
            ref = torchaudio.functional.resample(ref, sample["sr"], SR)
        ref = ref / (ref.abs().max() + 1e-8)
        if hyp_sr != SR:
            hyp = torchaudio.functional.resample(hyp, hyp_sr, SR)

        hyp_pad, ref_pad, lens = pad_pair(hyp, ref)
        hyp_pad = hyp_pad.to(metric_device)
        ref_pad = ref_pad.to(metric_device)
        lens = lens.to(metric_device)
        sid = sample["id"]

        dwer_metric.append([sid], hyp_pad, ref_pad, lens=lens)
        utmos_metric.append([sid], hyp.to(metric_device))
        spksim_metric.append([sid], hyp_pad, ref_pad, lens=lens)
        rows.append(
            {
                **rec,
                "wav_path": str(hyp_path),
                "ref_duration_s": sample["duration"],
                "ref_text": sample["text"],
            }
        )
        if (idx + 1) % 10 == 0 or idx + 1 == len(records):
            progress = {"official_metrics": idx + 1, "total": len(records)}
            with open(out_dir / "official_metrics_report.json", "w") as f:
                json.dump({"progress": progress}, f, indent=2)
            print(json.dumps(progress), flush=True)

    dwer_summary = dwer_metric.summarize()
    summary = {
        "checkpoint": checkpoint,
        "dataset_root": args.dataset_root,
        "parquet_files": parquet_files,
        "num_samples": len(rows),
        "batch_size": args.batch_size,
        "generation_seconds": generation_seconds,
        "metrics_seconds": time.time() - metric_start,
        "official_metrics": {
            "dWER": float(dwer_summary["error_rate"]),
            "dCER": float(dwer_summary["error_rate_char"]),
            "UTMOS": float(utmos_metric.summarize("average")),
            "SpkSimWavLM": float(spksim_metric.summarize("average")),
        },
        "config": {
            "target_voice": "per-sample input reference audio via vae.extract_speaker_embedding",
            "dwer_model": args.dwer_model,
            "dwer_device": args.dwer_device,
            "spksim_model": "microsoft/wavlm-base-sv",
            "max_ref_seconds": args.max_ref_seconds,
            "max_audio_seconds": args.max_audio_seconds,
            "max_len": args.max_len,
            "num_steps": args.num_steps,
            "diffusion_temperature": args.diffusion_temperature,
            "guidance_scale": args.guidance_scale,
            "vae_num_steps": args.vae_num_steps,
            "vae_temperature": args.vae_temperature,
            "vae_guidance_scale": args.vae_guidance_scale,
        },
        "samples": rows,
    }
    with open(out_dir / "official_metrics_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["official_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
