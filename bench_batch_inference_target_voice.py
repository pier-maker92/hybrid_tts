#!/usr/bin/env python
import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from g2p_en import G2p

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import (  # noqa: E402
    align_continuous_tokens,
    load_hybrid_model,
    load_kmeans_centroids,
    load_vae,
    load_vocoder,
    resolve_local_research_path,
    trim_unpaired_discrete_tokens,
)
from util import build_tokenizer  # noqa: E402


def latest_checkpoint(run_dir):
    checkpoints = []
    for path in Path(run_dir).glob("checkpoint-*"):
        match = re.search(r"checkpoint-(\d+)$", path.name)
        if match and path.is_dir():
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* dirs found under {run_dir}")
    return str(sorted(checkpoints)[-1][1])


def audio_to_tensor(audio_obj):
    if isinstance(audio_obj, dict):
        wav = torch.as_tensor(audio_obj["array"]).float()
        sr = int(audio_obj["sampling_rate"])
    elif hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        wav = torch.as_tensor(samples.data).float()
        sr = int(samples.sample_rate)
    elif hasattr(audio_obj, "array"):
        wav = torch.as_tensor(audio_obj.array).float()
        sr = int(audio_obj.sampling_rate)
    else:
        raise TypeError(f"Unsupported audio object: {type(audio_obj)}")

    if wav.ndim == 2:
        if wav.shape[0] <= 8:
            wav = wav.mean(dim=0)
        else:
            wav = wav.mean(dim=-1)
    wav = wav.squeeze().contiguous()
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak
    return wav, sr


def first_existing_text(row):
    for key in (
        "text_normalized",
        "normalized_text",
        "text",
        "transcription",
        "transcript",
        "sentence",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(f"No text field found. Available keys: {sorted(row.keys())}")


def first_existing_audio(row):
    for key in ("audio", "output_audio", "wav", "speech"):
        if key in row:
            return row[key]
    raise KeyError(f"No audio field found. Available keys: {sorted(row.keys())}")


def load_librispeech_samples(dataset_root, num_samples, max_ref_seconds):
    parquet_files = sorted(
        str(path)
        for path in Path(dataset_root).rglob("*.parquet")
        if "test_clean" in str(path) and not path.name.startswith("._")
    )
    if not parquet_files:
        raise FileNotFoundError(f"No test_clean parquet files under {dataset_root}")

    dataset = load_dataset("parquet", data_files=parquet_files, split="train")
    samples = []
    for idx, row in enumerate(dataset):
        wav, sr = audio_to_tensor(first_existing_audio(row))
        duration = float(wav.numel()) / float(sr)
        if max_ref_seconds is not None and duration > max_ref_seconds:
            continue
        samples.append(
            {
                "id": str(row.get("id", row.get("audio_id", row.get("file", idx)))),
                "text": first_existing_text(row),
                "wav": wav,
                "sr": sr,
                "duration": duration,
            }
        )
        if len(samples) >= num_samples:
            break
    if len(samples) < num_samples:
        raise RuntimeError(
            f"Only found {len(samples)} samples after filtering; requested {num_samples}"
        )
    samples.sort(key=lambda item: item["duration"], reverse=True)
    return samples, parquet_files


def build_phoneme_ids(texts, vocab, tok):
    g2p = G2p()
    prompts = []
    for text in texts:
        ids = []
        for phoneme in g2p(text):
            if phoneme in vocab:
                ids.append(vocab[phoneme])
        if not ids:
            raise RuntimeError(f"Empty phoneme sequence for text: {text!r}")
        ids.append(tok.start_audio_id)
        prompts.append(ids)
    return prompts


def make_prompt_batch(prompts, tok, device):
    tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in prompts]
    discrete_sequence = torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=tok.pad_id
    )
    attention_mask = torch.zeros_like(discrete_sequence, dtype=torch.bool)
    for index, ids in enumerate(prompts):
        attention_mask[index, : len(ids)] = True
    return {"discrete_sequence": discrete_sequence, "attention_mask": attention_mask}


def decode_one(
    sample_index,
    sample_out,
    tok,
    kmeans_centroids,
    hybrid_model,
    vae,
    vocoder,
    speaker_embeddings,
    args,
    dtype,
    device,
):
    final_discrete = sample_out["discrete_tokens"]
    z_denorm = sample_out["continuous_tokens"]
    discrete_lengths = sample_out["discrete_lengths"]
    token_len = int(discrete_lengths[sample_index].item())
    sample_discrete = final_discrete[sample_index]
    if sample_discrete.ndim == 2:
        sample_discrete = sample_discrete[:, 0]
    audio_tokens = sample_discrete[:token_len].clamp_min(0).long().tolist()
    if getattr(tok, "audio_bpe", None) is not None:
        audio_tokens = tok.audio_bpe.decode(audio_tokens)
    if not audio_tokens:
        raise RuntimeError(f"No audio tokens generated for sample {sample_index}")

    z_sample = None if z_denorm is None else z_denorm[sample_index : sample_index + 1, :token_len]
    tokens_tensor = torch.tensor(audio_tokens, dtype=torch.long, device=device)

    if kmeans_centroids is not None:
        tokens_tensor = trim_unpaired_discrete_tokens(tokens_tensor, z_sample)
        audio_tokens = tokens_tensor.tolist()
        if tokens_tensor.numel() == 0:
            raise RuntimeError(f"No paired kmeans tokens generated for sample {sample_index}")
        if tokens_tensor.min().item() < 0 or tokens_tensor.max().item() >= kmeans_centroids.shape[0]:
            raise RuntimeError(
                f"Kmeans token out of range: min={tokens_tensor.min().item()} "
                f"max={tokens_tensor.max().item()} clusters={kmeans_centroids.shape[0]}"
            )
        z_semantic = kmeans_centroids.index_select(0, tokens_tensor).unsqueeze(0)
        z_acoustic = align_continuous_tokens(
            z_sample,
            length=len(audio_tokens),
            continuous_dim=hybrid_model.config.continuous_dim,
            dtype=dtype,
            device=device,
        )
        z = torch.cat([z_semantic, z_acoustic], dim=-1)
        padding_mask = torch.zeros((1, len(audio_tokens)), dtype=torch.bool, device=device)
        reconstructed_mel, reconstructed_padding_mask = vae.sample(
            num_steps=args.vae_num_steps,
            temperature=args.vae_temperature,
            guidance_scale=args.vae_guidance_scale,
            z=z,
            padding_mask=padding_mask,
            speaker_embedding=speaker_embeddings[sample_index : sample_index + 1],
        )
    else:
        raise RuntimeError("This benchmark expects kmeans centroids.")

    mel = reconstructed_mel[0]
    mask = reconstructed_padding_mask[0]
    mel = mel[~mask].unsqueeze(0).permute(0, 2, 1).float().to(device)
    wav = vocoder.decode(mel).squeeze()
    wav = wav / (wav.abs().max() + 1e-8)
    max_len = int(args.max_audio_seconds * 24000)
    if wav.numel() > max_len:
        wav = wav[:max_len]
    return wav.detach(), token_len


@torch.no_grad()
def run_chunk(samples, prompts, models, args, device, dtype, save_dir=None):
    hybrid_model, vae, vocoder, tok, kmeans_centroids = models
    speaker_embeddings = vae.extract_speaker_embedding(
        [(sample["wav"].to(device), sample["sr"]) for sample in samples]
    )
    if speaker_embeddings is None:
        raise RuntimeError("VAE returned no speaker embeddings; target_voice conditioning is unavailable.")
    speaker_embeddings = speaker_embeddings.to(device=device, dtype=dtype)

    batch = make_prompt_batch(prompts, tok, device)
    sample_out = hybrid_model.sample(
        batch=batch,
        max_steps=args.max_len,
        temperature=args.temperature,
        num_steps=args.num_steps,
        diffusion_temperature=args.diffusion_temperature,
        guidance_scale=args.guidance_scale,
        vae=vae,
    )

    records = []
    for i, sample in enumerate(samples):
        generated, token_len = decode_one(
            i, sample_out, tok, kmeans_centroids, hybrid_model, vae, vocoder,
            speaker_embeddings, args, dtype, device
        )
        generated_embedding = vae.extract_speaker_embedding([(generated.to(device), 24000)])
        generated_embedding = generated_embedding.to(device=device, dtype=dtype)
        sim = F.cosine_similarity(
            speaker_embeddings[i : i + 1].float(), generated_embedding.float(), dim=-1
        ).mean().item()
        if save_dir is not None and i < args.save_wavs_per_run:
            out_wav = Path(save_dir) / f"{sample['id'].replace('/', '_')}.wav"
            torchaudio.save(str(out_wav), generated.unsqueeze(0).cpu(), 24000)
        records.append(
            {
                "id": sample["id"],
                "duration_ref_s": sample["duration"],
                "duration_gen_s": float(generated.numel()) / 24000.0,
                "tokens": token_len,
                "speaker_cosine_vae": sim,
            }
        )
    return records


def memory_gb():
    if not torch.cuda.is_available():
        return {"peak_allocated_gb": None, "peak_reserved_gb": None}
    return {
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024 ** 3),
    }


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--checkpoint_run", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_sizes", default="1,2,4,8,16,32,64,100")
    parser.add_argument("--max_ref_seconds", type=float, default=20.0)
    parser.add_argument("--max_audio_seconds", type=float, default=60.0)
    parser.add_argument("--max_len", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--diffusion_temperature", type=float, default=0.3)
    parser.add_argument("--guidance_scale", type=float, default=1.2)
    parser.add_argument("--vae_num_steps", type=int, default=16)
    parser.add_argument("--vae_temperature", type=float, default=0.2)
    parser.add_argument("--vae_guidance_scale", type=float, default=1.3)
    parser.add_argument("--save_wavs_per_run", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_root = args.dataset_root or os.path.join(
        os.environ["SLURM_TMPDIR"], "datasets", "librispeech-aligned"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = latest_checkpoint(args.checkpoint_run)
    with open(os.path.join(ckpt, "config.json"), "r") as f:
        cfg = json.load(f)
    scratch_dir = os.environ.get("SCRATCH", "/scratch/piermel")
    cfg["vae_checkpoint"] = resolve_local_research_path(cfg.get("vae_checkpoint"), scratch_dir)
    cfg["kmeans_path"] = resolve_local_research_path(cfg.get("kmeans_path"), scratch_dir)
    dtype = torch.bfloat16 if device.type == "cuda" and cfg.get("training", {}).get("bf16") and torch.cuda.is_bf16_supported() else torch.float32

    samples, parquet_files = load_librispeech_samples(
        args.dataset_root, args.num_samples, args.max_ref_seconds
    )
    with open("data/phoneme_vocab.json", "r") as f:
        phoneme_vocab = json.load(f)
    tok = build_tokenizer(cfg, pretrinaed=False)
    prompts = build_phoneme_ids([sample["text"] for sample in samples], phoneme_vocab, tok)

    kmeans_centroids = load_kmeans_centroids(cfg.get("kmeans_path"), device=device, dtype=dtype)
    hybrid_model = load_hybrid_model(cfg, ckpt, device, dtype, tokenizer=tok)
    vae = load_vae(cfg["vae_checkpoint"], device, dtype)
    if vae is None:
        raise RuntimeError("Could not load VAE")
    if getattr(vae, "speaker_encoder_type", None) == "wavlm" and getattr(
        vae, "speaker_encoder", None
    ) is not None:
        vae.speaker_encoder = vae.speaker_encoder.to(device=device, dtype=torch.float32)
    vocoder = load_vocoder(cfg.get("vocoder_checkpoint", "vocos"), device)
    if vocoder is None:
        raise RuntimeError("Could not load vocoder")
    models = (hybrid_model, vae, vocoder, tok, kmeans_centroids)

    report = {
        "checkpoint": ckpt,
        "dataset_root": args.dataset_root,
        "parquet_files": parquet_files,
        "num_samples": len(samples),
        "max_ref_seconds": args.max_ref_seconds,
        "config": {
            "max_len": args.max_len,
            "num_steps": args.num_steps,
            "diffusion_temperature": args.diffusion_temperature,
            "guidance_scale": args.guidance_scale,
            "vae_num_steps": args.vae_num_steps,
            "vae_temperature": args.vae_temperature,
            "vae_guidance_scale": args.vae_guidance_scale,
            "target_voice": "per-sample input audio via vae.extract_speaker_embedding",
        },
        "batch_probe": [],
    }

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    max_ok = 0
    for batch_size in batch_sizes:
        if batch_size > len(samples):
            continue
        clear_cuda()
        start = time.time()
        entry = {"batch_size": batch_size, "status": "started"}
        try:
            run_chunk(
                samples[:batch_size],
                prompts[:batch_size],
                models,
                args,
                device,
                dtype,
                save_dir=None,
            )
            entry.update({"status": "ok", "seconds": time.time() - start})
            entry.update(memory_gb())
            max_ok = batch_size
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                entry.update({"status": "oom", "error": str(exc)[:1000]})
                entry.update(memory_gb())
                report["batch_probe"].append(entry)
                with open(out_dir / "report.json", "w") as f:
                    json.dump(report, f, indent=2)
                print(json.dumps(entry, indent=2), flush=True)
                clear_cuda()
                break
            entry.update({"status": "error", "error": traceback.format_exc()})
            report["batch_probe"].append(entry)
            with open(out_dir / "report.json", "w") as f:
                json.dump(report, f, indent=2)
            print(json.dumps(entry, indent=2), flush=True)
            raise
        report["batch_probe"].append(entry)
        with open(out_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(entry, indent=2), flush=True)

    if max_ok <= 0:
        raise RuntimeError("No batch size succeeded.")

    clear_cuda()
    full_records = []
    full_start = time.time()
    wav_dir = out_dir / f"wavs_batch{max_ok}"
    wav_dir.mkdir(exist_ok=True)
    for start_idx in range(0, len(samples), max_ok):
        end_idx = min(start_idx + max_ok, len(samples))
        records = run_chunk(
            samples[start_idx:end_idx],
            prompts[start_idx:end_idx],
            models,
            args,
            device,
            dtype,
            save_dir=wav_dir if start_idx == 0 else None,
        )
        full_records.extend(records)
        report["full_100_progress"] = {"done": len(full_records), "total": len(samples)}
        with open(out_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)

    sims = [r["speaker_cosine_vae"] for r in full_records]
    rtfs = [r["duration_gen_s"] for r in full_records]
    report["full_100"] = {
        "batch_size": max_ok,
        "seconds": time.time() - full_start,
        "samples_per_second": len(full_records) / max(time.time() - full_start, 1e-6),
        "speaker_cosine_vae_mean": sum(sims) / len(sims),
        "speaker_cosine_vae_min": min(sims),
        "generated_duration_s_mean": sum(rtfs) / len(rtfs),
    }
    report["samples"] = full_records
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["full_100"], indent=2), flush=True)


if __name__ == "__main__":
    main()
