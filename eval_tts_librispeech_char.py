#!/usr/bin/env python
"""Batched full-TTS evaluation for char-conditioned HybridTTS checkpoints."""
import argparse
import glob
import json
import logging
import math
import os
import re
import sys
import time

import torch
import torchaudio

from inference import (
    align_continuous_tokens,
    combine_semantic_and_acoustic_latents,
    configure_sm_inference,
    discrete_tokens_to_semantic_latents,
    encode_text_prompt,
    load_hybrid_model,
    load_kmeans_centroids,
    load_vae,
    load_vocoder,
    trim_unpaired_discrete_tokens,
)
from util import build_tokenizer

SR = 24000
LOG = logging.getLogger("eval_tts_librispeech_char")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def resolve_scratch(path):
    return path.replace("$SCRATCH", os.environ["SCRATCH"]) if path else path


def checkpoint_label(path):
    return f"{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)}"


def decode_audio_field(audio):
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        return samples.data, int(samples.sample_rate), float(samples.duration_seconds)

    array, sample_rate = audio.get("array"), audio.get("sampling_rate")
    if array is None or not sample_rate:
        return None, None, None
    waveform = torch.as_tensor(array, dtype=torch.float32)
    return waveform, int(sample_rate), waveform.shape[-1] / float(sample_rate)


def audio_duration_seconds(audio):
    _, _, duration = decode_audio_field(audio)
    return duration


def load_test_clean(dataset_root, max_ref_seconds, num_samples):
    from datasets import load_dataset

    files = sorted(glob.glob(os.path.join(dataset_root, "test_clean", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset_root}/test_clean")
    dataset = load_dataset("parquet", data_files=files, split="train")
    selected = []
    for index, example in enumerate(dataset):
        duration = audio_duration_seconds(example.get("audio", {}))
        text = example.get("transcript") or example.get("text") or example.get("transcription") or ""
        if duration is not None and duration <= max_ref_seconds and text.strip():
            selected.append((index, example, duration))
        if num_samples > 0 and len(selected) >= num_samples:
            break
    return selected


def conditioning_text(text):
    """Use the checkpoint's char vocabulary and ensure terminal punctuation."""
    text = re.sub(r"\s+", " ", text.lower()).strip()
    text = "".join(ch for ch in text if ch in "abcdefghijklmnopqrstuvwxyz ,-.!?\"'")
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def max_tokens_for_seconds(vae_cfg, max_audio_seconds):
    sample_rate = int(vae_cfg.get("sample_rate", SR))
    hop_length = int((vae_cfg.get("mel_spectrogram_config") or {}).get("hop_length", 256))
    compress = int((vae_cfg.get("encoder_config") or {}).get("compress_factor_C", 4))
    return max(1, int(math.ceil(max_audio_seconds * sample_rate / (hop_length * compress))))


def decode_one(tokens, z_sample, speaker_embedding, model, vae, vocoder, kmeans_centroids, dtype, device, args, full_z_mode=False):
    tokens = tokens.clamp_min(0).long()
    if full_z_mode:
        if z_sample is None or z_sample.ndim != 3 or z_sample.shape[-1] != vae.config.latent_dim:
            raise ValueError("SM checkpoints require generated full DiCodec z")
        z = z_sample.to(device=device, dtype=dtype)
        if z.shape[1] < 1 or z.shape[1] > len(tokens):
            raise ValueError("Generated full z has invalid length")
        tokens = tokens[:z.shape[1]]
    elif kmeans_centroids is not None:
        tokens = trim_unpaired_discrete_tokens(tokens, z_sample)
        if not len(tokens):
            raise ValueError("no paired audio tokens generated")
        z_semantic = kmeans_centroids.index_select(0, tokens).unsqueeze(0)
    else:
        z_semantic = discrete_tokens_to_semantic_latents(vae, tokens, dtype, device)
    if not full_z_mode:
        z_acoustic = align_continuous_tokens(
            z_sample, length=len(tokens), continuous_dim=model.config.continuous_dim,
            dtype=dtype, device=device,
        )
        z = (torch.cat([z_semantic, z_acoustic], dim=-1) if kmeans_centroids is not None
             else combine_semantic_and_acoustic_latents(z_semantic, z_acoustic, vae))
    padding_mask = torch.zeros((1, len(tokens)), dtype=torch.bool, device=device)
    mel, mel_mask = vae.sample(
        num_steps=args.vae_num_steps,
        temperature=args.vae_temperature,
        guidance_scale=args.vae_guidance_scale,
        z=z,
        padding_mask=padding_mask,
        speaker_embedding=speaker_embedding,
    )
    mel = mel[0][~mel_mask[0]].unsqueeze(0).permute(0, 2, 1).float().to(device)
    audio = vocoder.decode(mel).squeeze()
    if audio.numel() == 0 or not torch.isfinite(audio).all():
        raise ValueError("Vocoder returned empty or non-finite audio")
    audio = audio / (audio.abs().max() + 1e-8)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    max_samples = int(args.max_audio_seconds * SR)
    return audio[..., :max_samples], len(tokens)


def reference_audios_srs(batch, device):
    """Return normalized LibriSpeech waveforms in the inference voice-condition format."""
    references = []
    for _, example, _ in batch:
        waveform, sample_rate, _ = decode_audio_field(example["audio"])
        if waveform is None or sample_rate is None:
            raise ValueError("missing reference audio")
        waveform = waveform.float()
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
        references.append((waveform.to(device), sample_rate))
    return references


def pad_pair(hypothesis, reference):
    hypothesis = hypothesis.squeeze().float()
    reference = reference.squeeze().float()
    hypothesis_length, reference_length = hypothesis.numel(), reference.numel()
    padded_length = max(hypothesis_length, reference_length, 1)
    hypothesis = torch.nn.functional.pad(hypothesis, (0, padded_length - hypothesis_length))
    reference = torch.nn.functional.pad(reference, (0, padded_length - reference_length))
    lengths = torch.tensor([max(hypothesis_length, reference_length) / padded_length], dtype=torch.float32)
    return hypothesis.unsqueeze(0), reference.unsqueeze(0), lengths


def build_metrics(args):
    metrics_root = os.path.join(args.audiocodecs_root, "downstream")
    sys.path.insert(0, metrics_root)
    try:
        from metrics.dwer import DWER
        from metrics.speaker_similarity import SpkSimWavLM
        from metrics.utmos import UTMOS
    finally:
        sys.path.pop(0)
    cache = os.path.join(os.environ.get("HF_HOME", "/scratch/piermel/.cache/huggingface"), "hub")
    dwer = DWER(args.dwer_model, SR, save_path=cache, device=args.dwer_device)
    # Reuse Whisper for transcript WER/CER, avoiding a second ASR model in memory.
    wer_cer = DWER(args.dwer_model, SR, save_path=cache, model=dwer.model, device=args.dwer_device)
    utmos = UTMOS(SR)
    speaker_similarity = SpkSimWavLM("microsoft/wavlm-base-sv", SR, save_path=cache) if args.compute_speaker_similarity else None
    return dwer, wer_cer, utmos, speaker_similarity


@torch.no_grad()
def evaluate_checkpoint(checkpoint, selected, args):
    checkpoint = os.path.abspath(checkpoint)
    with open(os.path.join(checkpoint, "config.json")) as handle:
        cfg = json.load(handle)
    cfg["vae_checkpoint"] = resolve_scratch(cfg["vae_checkpoint"])
    with open(os.path.join(cfg["vae_checkpoint"], "config.json")) as handle:
        vae_cfg = json.load(handle)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = torch.bfloat16 if device.type == "cuda" and cfg.get("training", {}).get("bf16") and torch.cuda.is_bf16_supported() else torch.float32
    vae_dtype = torch.float32
    full_z_mode = configure_sm_inference(cfg, checkpoint)
    tokenizer = build_tokenizer(cfg, pretrinaed=False)
    if getattr(tokenizer, "char_tokenizer", None) is None:
        raise ValueError(f"{checkpoint} is not char-conditioned")
    model = load_hybrid_model(cfg, checkpoint, device, model_dtype, tokenizer=tokenizer)
    vae = load_vae(cfg["vae_checkpoint"], device, vae_dtype, training_cfg=cfg.get("training", {}))
    vocoder = load_vocoder(args.vocoder, device)
    kmeans_centroids = load_kmeans_centroids(cfg.get("kmeans_path"), device, vae_dtype)
    if vae is None or vocoder is None:
        raise RuntimeError("failed to load VAE or vocoder")
    if full_z_mode and model.config.continuous_dim != vae.config.latent_dim:
        raise ValueError("HybridTTS and DiCodec latent dimensions do not match")
    LOG.info("Decoding mode: %s; VAE: %s", "full_z" if full_z_mode else "semantic/acoustic", cfg["vae_checkpoint"])

    max_steps = max_tokens_for_seconds(vae_cfg, args.max_audio_seconds)
    output_dir = os.path.join(args.output_dir, checkpoint_label(checkpoint))
    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    records, total_wall_time, total_audio_seconds = [], 0.0, 0.0
    dwer_metric, wer_cer_metric, utmos_metric, speaker_similarity_metric = build_metrics(args)

    for start in range(0, len(selected), args.batch_size):
        batch = selected[start : start + args.batch_size]
        texts = [conditioning_text(ex.get("transcript") or ex.get("text") or ex.get("transcription") or "") for _, ex, _ in batch]
        prompt_ids = [encode_text_prompt(text, tokenizer) + [tokenizer.start_audio_id] for text in texts]
        if any(not ids[:-1] for ids in prompt_ids):
            raise ValueError("empty char prompt after normalization")
        prompts = [torch.tensor(ids, dtype=torch.long, device=device) for ids in prompt_ids]
        discrete_sequence = torch.nn.utils.rnn.pad_sequence(prompts, batch_first=True, padding_value=tokenizer.pad_id)
        attention_mask = torch.zeros_like(discrete_sequence, dtype=torch.bool)
        for row, ids in enumerate(prompt_ids):
            attention_mask[row, : len(ids)] = True
        speaker_embeddings = vae.extract_speaker_embedding(reference_audios_srs(batch, device))
        if speaker_embeddings is None:
            raise RuntimeError("The VAE does not provide speaker embeddings required for voice conditioning")
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.time()
        sample_out = model.sample(
            batch={"discrete_sequence": discrete_sequence, "attention_mask": attention_mask},
            max_steps=max_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_steps=args.num_steps,
            diffusion_temperature=args.diffusion_temperature,
            guidance_scale=args.guidance_scale,
            vae=vae,
        )
        discrete_lengths = sample_out.get("discrete_lengths")
        if discrete_lengths is None:
            discrete_lengths = (sample_out["discrete_tokens"].squeeze(-1) >= 0).sum(dim=1)
        for row, (dataset_index, example, ref_duration) in enumerate(batch):
            sid = str(example.get("id") or f"sample_{dataset_index}")
            try:
                token_count = int(discrete_lengths[row].item())
                tokens = sample_out["discrete_tokens"][row, :token_count].squeeze(-1)
                continuous_tokens = sample_out["continuous_tokens"]
                z_sample = None if continuous_tokens is None else continuous_tokens[row : row + 1, :token_count]
                audio, token_count = decode_one(
                    tokens,
                    z_sample,
                    speaker_embeddings[row : row + 1],
                    model,
                    vae,
                    vocoder,
                    kmeans_centroids,
                    vae_dtype,
                    device,
                    args,
                    full_z_mode=full_z_mode,
                )
                duration = audio.shape[-1] / float(SR)
                total_audio_seconds += duration
                safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sid)
                wav_path = os.path.join(wav_dir, f"sample_{start + row:04d}_{safe_id}.wav")
                torchaudio.save(wav_path, audio.cpu(), SR)
                reference, reference_sr = reference_audios_srs([batch[row]], device)[0]
                reference = reference.unsqueeze(0)
                if reference_sr != SR:
                    reference = torchaudio.functional.resample(reference, reference_sr, SR)
                hypothesis, metric_reference, lengths = pad_pair(audio, reference)
                hypothesis, metric_reference, lengths = hypothesis.to(device), metric_reference.to(device), lengths.to(device)
                reference_text = batch[row][1].get("transcript") or batch[row][1].get("text") or batch[row][1].get("transcription") or ""
                dwer_metric.append([sid], hypothesis, metric_reference, lens=lengths)
                wer_cer_metric.append([sid], hypothesis, metric_reference, lens=lengths, ref_text=[reference_text])
                utmos_metric.append([sid], audio)
                if speaker_similarity_metric is not None:
                    speaker_similarity_metric.append([sid], hypothesis, metric_reference, lens=lengths)
                records.append({"dataset_index": dataset_index, "id": sid, "conditioning_text": texts[row], "reference_text": reference_text, "ref_duration_sec": ref_duration, "audio_duration_sec": duration, "token_count": token_count, "wav_path": wav_path, "error": None})
            except Exception as exc:
                LOG.exception("[%s] evaluation failed", sid)
                records.append({"dataset_index": dataset_index, "id": sid, "conditioning_text": texts[row], "ref_duration_sec": ref_duration, "error": repr(exc)})
                with open(os.path.join(output_dir, "failure.json"), "w") as handle:
                    json.dump({"checkpoint": checkpoint, "samples": records}, handle, indent=2)
                raise RuntimeError(f"Evaluation stopped at {sid}; see {output_dir}/failure.json") from exc
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_wall_time += time.time() - started
        LOG.info("%s: processed %d/%d", checkpoint_label(checkpoint), min(start + len(batch), len(selected)), len(selected))

    dwer_summary, wer_cer_summary = dwer_metric.summarize(), wer_cer_metric.summarize()
    metrics = {"dWER": float(dwer_summary["error_rate"]), "WER": float(wer_cer_summary["error_rate"]), "CER": float(wer_cer_summary["CER"]), "UTMOS": float(utmos_metric.summarize("average"))}
    if speaker_similarity_metric is not None:
        metrics["SpkSimWavLM"] = float(speaker_similarity_metric.summarize("average"))
    report = {"checkpoint": checkpoint, "vae_checkpoint": cfg["vae_checkpoint"], "text_tokenizer": "char", "voice_conditioning": "paired_librispeech_reference_audio", "split": "librispeech test-clean", "filter": {"ref_duration_sec_lte": args.max_ref_seconds}, "batch_size": args.batch_size, "n_samples": len(records), "n_success": sum(r.get("error") is None for r in records), "rtf": total_wall_time / total_audio_seconds if total_audio_seconds else None, "metrics": metrics, "generation": {"temperature": args.temperature, "num_steps": args.num_steps, "diffusion_temperature": args.diffusion_temperature, "guidance_scale": args.guidance_scale, "vae_num_steps": args.vae_num_steps, "vae_temperature": args.vae_temperature, "vae_guidance_scale": args.vae_guidance_scale, "max_audio_seconds": args.max_audio_seconds, "max_steps": max_steps}, "samples": records}
    with open(os.path.join(output_dir, "report.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    LOG.info("%s complete: RTF=%s", checkpoint_label(checkpoint), report["rtf"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate char-conditioned TTS checkpoints on LibriSpeech test-clean")
    parser.add_argument("--checkpoint", action="append", required=True, help="Checkpoint directory; repeat for each model")
    parser.add_argument("--output_dir", default="/scratch/piermel/agente/eval_tts_librispeech_test_clean_char_20s")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--num_samples", type=int, default=0, help="0 evaluates every filtered sample")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_ref_seconds", type=float, default=20.0)
    parser.add_argument("--max_audio_seconds", type=float, default=60.0)
    # Defaults copied from inference.py's standard generation path.
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--diffusion_temperature", type=float, default=0.2)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--vae_num_steps", type=int, default=8)
    parser.add_argument("--vae_temperature", type=float, default=0.2)
    parser.add_argument("--vae_guidance_scale", type=float, default=1.3)
    parser.add_argument("--vocoder", default="vocos")
    parser.add_argument("--device", default=None)
    parser.add_argument("--audiocodecs_root", default="/scratch/piermel/audiocodecs")
    parser.add_argument("--dwer_model", default="small")
    parser.add_argument("--dwer_device", default="cuda")
    parser.add_argument("--compute_speaker_similarity", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    dataset_root = args.dataset_root or os.path.join(os.environ["SLURM_TMPDIR"], "datasets", "librispeech-aligned")
    selected = load_test_clean(dataset_root, args.max_ref_seconds, args.num_samples)
    if not selected:
        raise RuntimeError("no test-clean samples matched the duration/text filter")
    LOG.info("Selected %d samples with reference duration <= %.1fs", len(selected), args.max_ref_seconds)
    for checkpoint in args.checkpoint:
        evaluate_checkpoint(checkpoint, selected, args)


if __name__ == "__main__":
    main()
