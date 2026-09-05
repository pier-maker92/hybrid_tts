#!/usr/bin/env python
"""End-to-end TTS evaluation on LibriSpeech test-clean, two scenarios in one pass.

Scenario A - resynthesis (metrics: UTMOS / dWER / SpkSimWavLM):
    original audio --VAE.encode--> discrete VQ tokens
                   --diffusion--> acoustic features (z_acoustic)
                   --VAE.sample + vocoder--> generated audio (hyp)
    hyp is scored against the original audio (ref).

Scenario B - full TTS (report RTF, no metrics):
    text --hybrid model--> discrete tokens (BPE-decoded to raw VAE tokens)
         --diffusion--> acoustic features --VAE.sample + vocoder--> generated audio

For every sample we save the original and both generated waveforms.
Metrics use the exact audiocodecs configs (see evaluation/audiocodecs_metrics.py).
"""

import os
import sys
import json
import glob
import time
import argparse
import logging

import torch
import torchaudio

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("eval_tts")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import (
    load_hybrid_model,
    load_vae,
    load_vocoder,
    clean_text_and_phonemize,
)
from inference_diffusion import load_diffusion_model
from util import build_tokenizer

SR = 24000
LIBRISPEECH_TEST_CLEAN = "/scratch/piermel/datasets/librispeech-aligned/test_clean"


def _resolve_scratch(cfg):
    scratch = os.environ.get("SCRATCH", "")
    if cfg.get("vae_checkpoint"):
        cfg["vae_checkpoint"] = cfg["vae_checkpoint"].replace("$SCRATCH", scratch)
    return cfg


def load_test_clean(num_samples):
    from datasets import load_dataset

    files = sorted(glob.glob(os.path.join(LIBRISPEECH_TEST_CLEAN, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files under {LIBRISPEECH_TEST_CLEAN}")
    ds = load_dataset("parquet", data_files=files, split="train")
    if num_samples and num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))
    return ds


@torch.no_grad()
def synth_from_tokens(raw_tokens, diffusion_model, vae, vocoder, device, dtype,
                      diff_steps, diff_temp, diff_guid, speaker_embedding=None):
    """raw VAE discrete tokens -> diffusion acoustics -> VAE.sample -> vocoder -> audio [1, T]."""
    discrete_sequence = torch.tensor([raw_tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(discrete_sequence, dtype=torch.bool, device=device)

    context_vector = diffusion_model.embed(discrete_sequence)
    if diff_guid > 1.0:
        uncond = torch.full_like(discrete_sequence, diffusion_model.config.pad_token_id)
        context_vector = (context_vector, diffusion_model.embed(uncond))

    diffusion_out = diffusion_model.diffusion_head.generate(
        num_steps=diff_steps,
        context_vector=context_vector,
        temperature=diff_temp,
        guidance_scale=diff_guid,
        padding_mask=~attention_mask,
        speaker_embedding=speaker_embedding,
    )
    z_denorm = diffusion_model.dynamic_normalizer.denormalize(diffusion_out.audio_features)

    padding_mask = torch.zeros((1, len(raw_tokens)), dtype=torch.bool, device=device)
    tokens_tensor = torch.tensor(raw_tokens, dtype=torch.long, device=device)
    vq_emb = vae.encoder.vq.codebook(tokens_tensor).unsqueeze(0)

    mel, mel_mask = vae.sample(
        num_steps=16,
        temperature=0.2,
        guidance_scale=1.4,
        z_semantic=vq_emb,
        z_acoustic=z_denorm,
        padding_mask=padding_mask,
    )
    mel = mel[0][~mel_mask[0]].unsqueeze(0).permute(0, 2, 1).float().to(device)
    audio = vocoder.decode(mel).squeeze()
    audio = audio / (audio.abs().max() + 1e-8)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    return audio  # [1, T]


@torch.no_grad()
def vae_encode_tokens(vae, audio, sr, device, dtype):
    """original audio -> VAE discrete VQ token indices (raw, list)."""
    audios_srs = [(audio.to(device=device, dtype=dtype), sr)]
    feats = vae.extract_features(audios_srs)
    enc_features, enc_padding_mask = feats[0], feats[1]
    enc_out = vae.encode(enc_features, enc_padding_mask)
    indices = enc_out.indices
    return indices.squeeze(0).tolist()


def main():
    p = argparse.ArgumentParser(description="TTS eval: resynthesis (A) + full TTS (B)")
    p.add_argument("--hybrid_checkpoint", type=str, required=True)
    p.add_argument("--diffusion_checkpoint", type=str, required=True)
    p.add_argument("--output_dir", "-o", type=str, default="eval_tts_out")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--vocoder", type=str, default="vocos")
    p.add_argument("--device", type=str, default=None)
    # hybrid (discrete token) generation
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_len", type=int, default=1000)
    # diffusion generation
    p.add_argument("--num_steps", "-n", type=int, default=8)
    p.add_argument("--diffusion_temperature", type=float, default=1.0)
    p.add_argument("--guidance_scale", "-dg", type=float, default=1.0)
    p.add_argument("--audiocodecs_dwer_device", type=str, default="cuda")
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    wav_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    with open(os.path.join(args.hybrid_checkpoint, "config.json")) as f:
        hybrid_cfg = _resolve_scratch(json.load(f))
    with open(os.path.join(args.diffusion_checkpoint, "config.json")) as f:
        diffusion_cfg = _resolve_scratch(json.load(f))

    dtype = torch.float32
    if device.type == "cuda":
        tcfg = hybrid_cfg.get("training", {})
        if tcfg.get("bf16") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

    vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "phoneme_vocab.json")
    with open(vocab_path) as f:
        phoneme_vocab = json.load(f)

    logger.info("Building tokenizers + loading models...")
    hybrid_tok = build_tokenizer(hybrid_cfg, pretrinaed=False)
    diffusion_tok = build_tokenizer(diffusion_cfg, pretrinaed=False)
    hybrid_model = load_hybrid_model(hybrid_cfg, args.hybrid_checkpoint, device, dtype, tokenizer=hybrid_tok)
    diffusion_model = load_diffusion_model(diffusion_cfg, args.diffusion_checkpoint, device, dtype, tokenizer=diffusion_tok)
    n_diff = sum(p.numel() for p in diffusion_model.parameters())
    n_diff_head = sum(p.numel() for p in diffusion_model.diffusion_head.parameters())
    logger.info(f"DiffusionOnlyModel parameters: {n_diff:,} ({n_diff/1e6:.2f}M); "
                f"diffusion_head: {n_diff_head:,} ({n_diff_head/1e6:.2f}M)")
    vae = load_vae(diffusion_cfg.get("vae_checkpoint"), device, dtype)
    vocoder = load_vocoder(args.vocoder, device)
    if vae is None or vocoder is None:
        logger.error("Failed to load VAE or vocoder.")
        sys.exit(1)

    # audiocodecs metrics
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation"))
    import audiocodecs_metrics
    sys.path.pop(0)
    ac_metrics_a = audiocodecs_metrics.AudioCodecsMetrics(
        sample_rate=SR, device=device, dwer_device=args.audiocodecs_dwer_device,
        compute_sim=False,  # SIM not wanted in any scenario
    )
    ac_metrics_b = audiocodecs_metrics.AudioCodecsMetrics(
        sample_rate=SR, device=device, dwer_device=args.audiocodecs_dwer_device,
        compute_sim=False,
    )

    ds = load_test_clean(args.num_samples)
    logger.info(f"Loaded {len(ds)} LibriSpeech test-clean samples.")

    rtf_list = []
    diff_kwargs = dict(
        diff_steps=args.num_steps,
        diff_temp=args.diffusion_temperature,
        diff_guid=args.guidance_scale,
    )

    for i, ex in enumerate(ds):
        sid = ex.get("id") or f"{ex.get('speaker', 'spk')}_{i}"
        text = ex.get("transcript") or ex.get("transcription") or ""
        audio = torch.tensor(ex["audio"]["array"], dtype=torch.float32)
        sr = int(ex["audio"]["sampling_rate"])

        # original (ref), resampled to SR to match generated
        orig = torchaudio.functional.resample(audio, sr, SR) if sr != SR else audio
        orig = orig / (orig.abs().max() + 1e-8)
        torchaudio.save(os.path.join(wav_dir, f"sample_{i}_{sid}_original.wav"), orig.unsqueeze(0), SR)

        speaker_embedding = None
        if hasattr(vae, "extract_speaker_embedding"):
            speaker_embedding = vae.extract_speaker_embedding(
                [(audio.to(device=device, dtype=dtype), sr)]
            )
            if speaker_embedding is not None:
                speaker_embedding = speaker_embedding.to(device=device, dtype=dtype)

        # ---- Scenario A: resynthesis (metrics) ----
        try:
            tokens_a = vae_encode_tokens(vae, audio, sr, device, dtype)
            gen_a = synth_from_tokens(
                tokens_a,
                diffusion_model,
                vae,
                vocoder,
                device,
                dtype,
                speaker_embedding=speaker_embedding,
                **diff_kwargs,
            )
            torchaudio.save(os.path.join(wav_dir, f"sample_{i}_{sid}_scenarioA_gen.wav"), gen_a.cpu(), SR)
            ac_metrics_a.append(sid, gen_a[0].cpu(), orig)
        except Exception:
            logger.exception(f"[sample {i}] scenario A failed")

        # ---- Scenario B: full TTS (RTF, no metrics) ----
        try:
            if not text.strip():
                raise ValueError("empty transcript")
            t0 = time.time()
            prompt_ids = clean_text_and_phonemize(text, phoneme_vocab)
            prompt_ids.append(hybrid_tok.start_audio_id)
            discrete_sequence = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(discrete_sequence, dtype=torch.bool, device=device)
            sample_out = hybrid_model.sample(
                batch={
                    "discrete_sequence": discrete_sequence,
                    "attention_mask": attention_mask,
                },
                max_steps=args.max_len,
                temperature=args.temperature,
                num_steps=1,
                vae=vae,
                reference_audios_srs=[(audio.to(device=device, dtype=dtype), sr)],
                voice_conditioner=vae,
            )
            audio_tokens = sample_out["discrete_tokens"].squeeze(-1).squeeze(0).tolist()
            if getattr(hybrid_tok, "audio_bpe", None) is not None:
                audio_tokens = hybrid_tok.audio_bpe.decode(audio_tokens)
            gen_b = synth_from_tokens(
                audio_tokens,
                diffusion_model,
                vae,
                vocoder,
                device,
                dtype,
                speaker_embedding=speaker_embedding,
                **diff_kwargs,
            )
            torch.cuda.synchronize() if device.type == "cuda" else None
            elapsed = time.time() - t0
            dur = gen_b.shape[-1] / float(SR)
            rtf = elapsed / dur if dur > 0 else 0.0
            rtf_list.append(rtf)
            torchaudio.save(os.path.join(wav_dir, f"sample_{i}_{sid}_scenarioB_tts.wav"), gen_b.cpu(), SR)
            ac_metrics_b.append(sid, gen_b[0].cpu(), orig)
        except Exception:
            logger.exception(f"[sample {i}] scenario B failed")

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(ds)}")

    # ---- summary ----
    metrics_a = ac_metrics_a.summarize()
    metrics_b = ac_metrics_b.summarize()
    rtf_mean = sum(rtf_list) / len(rtf_list) if rtf_list else None
    report = {
        "n_samples": len(ds),
        "scenarioA_resynthesis_metrics": metrics_a,
        "scenarioB_tts_metrics": metrics_b,
        "scenarioB_tts_RTF_mean": rtf_mean,
        "scenarioB_RTF_n": len(rtf_list),
        "config": {
            "hybrid_checkpoint": args.hybrid_checkpoint,
            "diffusion_checkpoint": args.diffusion_checkpoint,
            "diffusion_num_steps": args.num_steps,
            "diffusion_temperature": args.diffusion_temperature,
            "guidance_scale": args.guidance_scale,
            "hybrid_temperature": args.temperature,
        },
    }
    out_json = os.path.join(args.output_dir, "eval_tts_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Scenario A metrics: {metrics_a}")
    logger.info(f"Scenario B metrics: {metrics_b}")
    logger.info(f"Scenario B RTF mean: {rtf_mean} (n={len(rtf_list)})")
    logger.info(f"Saved report to {out_json}")


if __name__ == "__main__":
    main()
