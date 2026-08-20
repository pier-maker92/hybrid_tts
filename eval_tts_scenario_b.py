#!/usr/bin/env python
import argparse
import glob
import json
import logging
import math
import os
import sys
import time
from typing import Optional
import torch
import torchaudio

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("eval_tts_scenario_b")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import (  # noqa: E402
    align_continuous_tokens,
    clean_text_and_phonemize,
    load_hybrid_model,
    load_kmeans_centroids,
    load_vae,
    load_vocoder,
    trim_unpaired_discrete_tokens,
)
from util import build_tokenizer  # noqa: E402


SR = 24000


def audio_duration_seconds(audio) -> Optional[float]:
    if audio is None:
        return None
    if isinstance(audio, dict):
        arr = audio.get("array")
        sr = audio.get("sampling_rate")
        if arr is not None and sr:
            return len(arr) / float(sr)
        path = audio.get("path")
        if path and os.path.exists(path):
            info = torchaudio.info(path)
            return info.num_frames / float(info.sample_rate)
        return None

    duration = getattr(audio, "duration", None)
    if duration is not None:
        return float(duration)

    metadata = getattr(audio, "metadata", None)
    if metadata is not None:
        num_frames = getattr(metadata, "num_frames", None)
        sample_rate = getattr(metadata, "sample_rate", None)
        if num_frames is not None and sample_rate:
            return float(num_frames) / float(sample_rate)

    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        data = getattr(samples, "data", None)
        sample_rate = getattr(samples, "sample_rate", None)
        if data is not None and sample_rate:
            return data.shape[-1] / float(sample_rate)

    return None


def resolve_checkpoint(path: str) -> str:
    if os.path.basename(path).startswith("checkpoint-"):
        return path
    candidates = []
    for ckpt in glob.glob(os.path.join(path, "checkpoint-*")):
        name = os.path.basename(ckpt)
        if name == "checkpoint-final":
            continue
        try:
            step = int(name.split("-")[-1])
        except ValueError:
            continue
        if os.path.exists(os.path.join(ckpt, "config.json")):
            candidates.append((step, ckpt))
    if not candidates:
        final = os.path.join(path, "checkpoint-final")
        if os.path.exists(os.path.join(final, "config.json")):
            return final
        raise FileNotFoundError(f"No checkpoint found under {path}")
    return sorted(candidates)[-1][1]


def load_test_clean(dataset_root: str, max_duration: float, num_samples: int):
    from datasets import load_dataset

    test_clean = os.path.join(dataset_root, "test_clean")
    files = [
        f
        for f in sorted(glob.glob(os.path.join(test_clean, "*.parquet")))
        if not os.path.basename(f).startswith("._")
    ]
    if not files:
        raise FileNotFoundError(f"No parquet files under {test_clean}")
    ds = load_dataset("parquet", data_files=files, split="train")

    selected = []
    for idx, ex in enumerate(ds):
        audio = ex.get("audio")
        duration = audio_duration_seconds(audio)
        text = ex.get("transcript") or ex.get("text") or ex.get("transcription") or ""
        if duration is None or not text.strip():
            continue
        if duration <= max_duration:
            selected.append((idx, ex, duration))
        if num_samples > 0 and len(selected) >= num_samples:
            break
    return selected


def max_tokens_for_seconds(vae_cfg: dict, max_audio_seconds: float) -> int:
    sample_rate = int(vae_cfg.get("sample_rate", SR))
    mel_cfg = vae_cfg.get("mel_spectrogram_config") or {}
    hop_length = int(mel_cfg.get("hop_length", 256))
    enc_cfg = vae_cfg.get("encoder_config") or {}
    compress = int(enc_cfg.get("compress_factor_C", 4))
    tokens_per_second = sample_rate / float(hop_length * compress)
    return max(1, int(math.ceil(max_audio_seconds * tokens_per_second)))


@torch.no_grad()
def synthesize_one(
    text: str,
    model,
    tokenizer,
    vae,
    vocoder,
    phoneme_vocab,
    kmeans_centroids,
    device,
    dtype,
    max_steps: int,
    max_audio_seconds: float,
    temperature: float,
    diffusion_steps: int,
    diffusion_temperature: float,
    guidance_scale: float,
    vae_num_steps: int,
    vae_temperature: float,
    vae_guidance_scale: float,
):
    prompt_ids = clean_text_and_phonemize(text, phoneme_vocab)
    if not prompt_ids:
        raise ValueError("empty phoneme sequence")
    prompt_ids.append(tokenizer.start_audio_id)

    discrete_sequence = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(discrete_sequence, dtype=torch.bool, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    sample_out = model.sample(
        batch={"discrete_sequence": discrete_sequence, "attention_mask": attention_mask},
        max_steps=max_steps,
        temperature=temperature,
        num_steps=diffusion_steps,
        diffusion_temperature=diffusion_temperature,
        guidance_scale=guidance_scale,
        vae=vae,
    )

    final_discrete = sample_out["discrete_tokens"]
    z_denorm = sample_out["continuous_tokens"]
    audio_tokens = final_discrete.squeeze(-1).squeeze(0).tolist()
    bpe_decoded = False
    if getattr(tokenizer, "audio_bpe", None) is not None:
        audio_tokens = tokenizer.audio_bpe.decode(audio_tokens)
        bpe_decoded = True

    tokens_tensor = torch.tensor(audio_tokens, dtype=torch.long, device=device)
    if kmeans_centroids is not None:
        tokens_tensor = trim_unpaired_discrete_tokens(tokens_tensor, z_denorm)
        audio_tokens = tokens_tensor.tolist()
        if tokens_tensor.numel() == 0:
            raise ValueError("no audio tokens generated")
        if tokens_tensor.min().item() < 0 or tokens_tensor.max().item() >= kmeans_centroids.shape[0]:
            raise ValueError(
                f"kmeans token out of range: min={tokens_tensor.min().item()} "
                f"max={tokens_tensor.max().item()} clusters={kmeans_centroids.shape[0]}"
            )
        z_semantic = kmeans_centroids.index_select(0, tokens_tensor).unsqueeze(0)
        z_acoustic = align_continuous_tokens(
            z_denorm,
            length=len(audio_tokens),
            continuous_dim=model.config.continuous_dim,
            dtype=dtype,
            device=device,
        )
        z = torch.cat([z_semantic, z_acoustic], dim=-1)
        padding_mask = torch.zeros((1, len(audio_tokens)), dtype=torch.bool, device=device)
        mel, mel_mask = vae.sample(
            num_steps=vae_num_steps,
            temperature=vae_temperature,
            guidance_scale=vae_guidance_scale,
            z=z,
            padding_mask=padding_mask,
        )
    else:
        padding_mask = torch.zeros((1, len(audio_tokens)), dtype=torch.bool, device=device)
        vq_emb = vae.encoder.vq.codebook(tokens_tensor).unsqueeze(0)
        z_acoustic = align_continuous_tokens(
            z_denorm,
            length=len(audio_tokens),
            continuous_dim=model.config.continuous_dim,
            dtype=dtype,
            device=device,
        )
        mel, mel_mask = vae.sample(
            num_steps=vae_num_steps,
            temperature=vae_temperature,
            guidance_scale=vae_guidance_scale,
            z_semantic=vq_emb,
            z_acoustic=z_acoustic,
            padding_mask=padding_mask,
        )

    mel = mel[0][~mel_mask[0]].unsqueeze(0).permute(0, 2, 1).float().to(device)
    audio = vocoder.decode(mel).squeeze()
    audio = audio / (audio.abs().max() + 1e-8)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    max_samples = int(max_audio_seconds * SR)
    truncated = False
    if audio.shape[-1] > max_samples:
        audio = audio[..., :max_samples]
        truncated = True
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    duration = audio.shape[-1] / float(SR)
    return audio, {
        "num_raw_audio_tokens": len(audio_tokens),
        "hit_token_cap": len(audio_tokens) >= max_steps,
        "bpe_decoded": bpe_decoded,
        "audio_duration_sec": duration,
        "wall_time_sec": elapsed,
        "rtf": elapsed / duration if duration > 0 else None,
        "truncated_to_max_audio_seconds": truncated,
    }


def main():
    parser = argparse.ArgumentParser(description="Scenario B full-TTS eval on LibriSpeech test-clean")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir or run dir containing checkpoint-*")
    parser.add_argument("--dataset_root", default=None, help="Dataset root containing test_clean")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=0, help="0 means all filtered samples")
    parser.add_argument("--max_ref_seconds", type=float, default=20.0)
    parser.add_argument("--max_audio_seconds", type=float, default=60.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--diffusion_temperature", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--vae_num_steps", type=int, default=16)
    parser.add_argument("--vae_temperature", type=float, default=0.2)
    parser.add_argument("--vae_guidance_scale", type=float, default=1.3)
    parser.add_argument("--vocoder", default="vocos")
    parser.add_argument("--device", default=None)
    args, _ = parser.parse_known_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = resolve_checkpoint(args.checkpoint)
    logger.info(f"Using checkpoint: {checkpoint}")
    logger.info(f"Using device: {device}")

    with open(os.path.join(checkpoint, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(cfg["vae_checkpoint"], "config.json")) as f:
        vae_cfg = json.load(f)

    max_steps = max_tokens_for_seconds(vae_cfg, args.max_audio_seconds)
    logger.info(f"EOS-based generation with hard cap: max_steps={max_steps} (~{args.max_audio_seconds}s)")

    dtype = torch.float32
    if device.type == "cuda" and cfg.get("training", {}).get("bf16") and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "phoneme_vocab.json")
    with open(vocab_path) as f:
        phoneme_vocab = json.load(f)

    tokenizer = build_tokenizer(cfg, pretrinaed=False)
    model = load_hybrid_model(cfg, checkpoint, device, dtype, tokenizer=tokenizer)
    vae = load_vae(cfg["vae_checkpoint"], device, dtype)
    vocoder = load_vocoder(args.vocoder, device)
    if vae is None or vocoder is None:
        raise RuntimeError("failed to load VAE or vocoder")
    kmeans_centroids = load_kmeans_centroids(cfg.get("kmeans_path"), device, dtype)

    dataset_root = args.dataset_root or os.path.join(
        os.environ["SLURM_TMPDIR"], "datasets", "librispeech-aligned"
    )
    selected = load_test_clean(dataset_root, args.max_ref_seconds, args.num_samples)
    if not selected:
        raise RuntimeError("no test-clean samples matched the filter")
    logger.info(f"Selected {len(selected)} samples from test-clean with duration <= {args.max_ref_seconds}s")

    wav_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    records = []
    for out_idx, (dataset_idx, ex, ref_duration) in enumerate(selected):
        sid = ex.get("id") or f"sample_{dataset_idx}"
        text = ex.get("transcript") or ex.get("text") or ex.get("transcription") or ""
        logger.info(f"[{out_idx+1}/{len(selected)}] {sid}: ref={ref_duration:.2f}s text={text[:120]!r}")
        try:
            audio, rec = synthesize_one(
                text=text,
                model=model,
                tokenizer=tokenizer,
                vae=vae,
                vocoder=vocoder,
                phoneme_vocab=phoneme_vocab,
                kmeans_centroids=kmeans_centroids,
                device=device,
                dtype=dtype,
                max_steps=max_steps,
                max_audio_seconds=args.max_audio_seconds,
                temperature=args.temperature,
                diffusion_steps=args.num_steps,
                diffusion_temperature=args.diffusion_temperature,
                guidance_scale=args.guidance_scale,
                vae_num_steps=args.vae_num_steps,
                vae_temperature=args.vae_temperature,
                vae_guidance_scale=args.vae_guidance_scale,
            )
            wav_path = os.path.join(wav_dir, f"sample_{out_idx:02d}_{sid}_scenarioB_tts.wav")
            torchaudio.save(wav_path, audio.cpu(), SR)
            rec.update({
                "sample_index": out_idx,
                "dataset_index": int(dataset_idx),
                "id": sid,
                "text": text,
                "ref_duration_sec": ref_duration,
                "wav_path": wav_path,
                "error": None,
            })
            records.append(rec)
            logger.info(
                f"[{sid}] generated {rec['audio_duration_sec']:.2f}s in "
                f"{rec['wall_time_sec']:.2f}s, RTF={rec['rtf']:.3f}, "
                f"tokens={rec['num_raw_audio_tokens']}, hit_cap={rec['hit_token_cap']}"
            )
        except Exception as exc:
            logger.exception(f"[{sid}] scenario B failed")
            records.append({
                "sample_index": out_idx,
                "dataset_index": int(dataset_idx),
                "id": sid,
                "text": text,
                "ref_duration_sec": ref_duration,
                "error": repr(exc),
            })

    rtfs = [r["rtf"] for r in records if r.get("rtf") is not None]
    report = {
        "scenario": "B_full_tts_rtf_only",
        "checkpoint": checkpoint,
        "dataset_root": dataset_root,
        "split": "librispeech test-clean",
        "filter": {"ref_duration_sec_lte": args.max_ref_seconds},
        "n_samples_requested": args.num_samples,
        "n_samples": len(records),
        "n_success": len(rtfs),
        "rtf_mean": sum(rtfs) / len(rtfs) if rtfs else None,
        "rtf_values": rtfs,
        "max_audio_seconds": args.max_audio_seconds,
        "max_steps": max_steps,
        "generation": {
            "temperature": args.temperature,
            "num_steps": args.num_steps,
            "diffusion_temperature": args.diffusion_temperature,
            "guidance_scale": args.guidance_scale,
            "vae_num_steps": args.vae_num_steps,
            "vae_temperature": args.vae_temperature,
            "vae_guidance_scale": args.vae_guidance_scale,
            "uses_ratio": False,
            "stops_on_eos_until_max_steps": True,
        },
        "samples": records,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "scenario_b_rtf_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Scenario B RTF mean: {report['rtf_mean']} (n={len(rtfs)})")
    logger.info(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
