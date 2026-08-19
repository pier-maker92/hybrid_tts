#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchaudio


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if requested == "fp32":
        return torch.float32
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def load_kmeans(path: Path) -> torch.Tensor:
    ckpt_path = path / "encoder_kmeans.pt" if path.is_dir() else path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"kmeans checkpoint non trovato: {ckpt_path}")
    codebook = torch.load(ckpt_path, map_location="cpu")
    if "centroids" not in codebook:
        raise ValueError(f"{ckpt_path} non contiene la chiave 'centroids'")
    centroids = codebook["centroids"].float()
    if centroids.ndim != 2:
        raise ValueError(f"centroids deve essere [K, D], trovato {tuple(centroids.shape)}")
    return centroids


def align_continuous(z_cont: torch.Tensor | None, length: int, dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if z_cont is None:
        return torch.zeros((1, length, dim), dtype=dtype, device=device)
    z_cont = z_cont.to(device=device, dtype=dtype)
    if z_cont.shape[1] == length:
        return z_cont
    if z_cont.shape[1] < length:
        missing = length - z_cont.shape[1]
        pad = torch.zeros((z_cont.shape[0], missing, z_cont.shape[2]), dtype=z_cont.dtype, device=device)
        return torch.cat([z_cont, pad], dim=1)
    return z_cont[:, :length]



def trim_unpaired_discrete_tokens(tokens_tensor: torch.Tensor, z_cont: torch.Tensor | None):
    if z_cont is None:
        return tokens_tensor
    continuous_len = z_cont.shape[1]
    discrete_len = tokens_tensor.numel()
    if continuous_len < discrete_len:
        return tokens_tensor[:continuous_len]
    return tokens_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/Users/software/Research/hybrid_tts/checkpoints/dicodec-kmeans-512")
    parser.add_argument("--vae_checkpoint", default="/Users/software/Research/MelCausalVAE/checkpoints/paper/disent/dicodec-18")
    parser.add_argument("--kmeans_path", default="/Users/software/Research/MelCausalVAE/kmeans/512")
    parser.add_argument("--text", default="Hello, this is a small inference test.")
    parser.add_argument("--output", default="/Users/software/Research/hybrid_tts/out_dicodec_kmeans.wav")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--max_len", type=int, default=260)
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--diffusion_temperature", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--vocoder", default="vocos")
    parser.add_argument("--decode_only_token", action="store_true")
    parser.add_argument(
        "--voice_condition",
        default=None,
        help="Reference audio file used to extract a speaker embedding for DiCodec decoder FiLM conditioning.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo))

    from inference import (
        clean_text_and_phonemize,
        load_hybrid_model,
        load_vae,
        load_vocoder,
        load_voice_condition,
    )
    from util import build_tokenizer

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    with (checkpoint / "config.json").open() as f:
        cfg = json.load(f)

    cfg["vae_checkpoint"] = str(Path(args.vae_checkpoint).expanduser())
    cfg["kmeans_path"] = str(Path(args.kmeans_path).expanduser())
    cfg["continuous_start"] = int(cfg.get("continuous_start", 4))

    for required in [Path(cfg["vae_checkpoint"]), Path(cfg["kmeans_path"])]:
        if not required.exists():
            raise FileNotFoundError(f"path richiesto non trovato: {required}")

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    centroids = load_kmeans(Path(cfg["kmeans_path"])).to(device=device, dtype=dtype)
    if centroids.shape[1] != cfg["continuous_start"]:
        raise ValueError(
            f"kmeans dim={centroids.shape[1]}, ma continuous_start={cfg['continuous_start']}"
        )

    tokenizer = build_tokenizer(cfg)
    model = load_hybrid_model(cfg, str(checkpoint), device, dtype, tokenizer)
    vae = load_vae(cfg["vae_checkpoint"], device, dtype)
    if vae is None:
        raise RuntimeError("load VAE fallito")
    speaker_embedding = None
    if args.voice_condition:
        speaker_embedding = load_voice_condition(args.voice_condition, vae, device)
    vocoder = load_vocoder(args.vocoder, device)
    if vocoder is None:
        raise RuntimeError("load vocoder fallito")

    with (repo / "data" / "phoneme_vocab.json").open() as f:
        phoneme_vocab = json.load(f)

    prompt_ids = clean_text_and_phonemize(args.text, phoneme_vocab)
    if not prompt_ids:
        raise RuntimeError("testo vuoto dopo phonemization")
    prompt_ids.append(tokenizer.start_audio_id)

    batch = {
        "discrete_sequence": torch.tensor([prompt_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.bool, device=device),
    }

    with torch.no_grad():
        sample = model.sample(
            batch=batch,
            max_steps=args.max_len,
            temperature=args.temperature,
            num_steps=args.num_steps,
            diffusion_temperature=args.diffusion_temperature,
            guidance_scale=args.guidance_scale,
            vae=vae,
        )

        discrete = sample["discrete_tokens"].reshape(-1).long()
        if discrete.numel() == 0:
            raise RuntimeError("il modello ha prodotto EOS subito: nessun token audio")
        if discrete.min().item() < 0 or discrete.max().item() >= centroids.shape[0]:
            raise RuntimeError(
                f"token fuori range kmeans: min={discrete.min().item()}, max={discrete.max().item()}, K={centroids.shape[0]}"
            )

        z_cont = None if args.decode_only_token else sample["continuous_tokens"]
        discrete = trim_unpaired_discrete_tokens(discrete, z_cont)
        z_sem = centroids.index_select(0, discrete).unsqueeze(0)
        z_cont = align_continuous(
            z_cont,
            length=discrete.numel(),
            dim=model.config.continuous_dim,
            dtype=dtype,
            device=device,
        )
        z = torch.cat([z_sem, z_cont], dim=-1)
        padding_mask = torch.zeros(z.shape[:2], dtype=torch.bool, device=device)

        mel, mel_mask = vae.sample(
            num_steps=16,
            temperature=0.2,
            guidance_scale=1.0,
            z=z,
            padding_mask=padding_mask,
            speaker_embedding=speaker_embedding,
        )

        mel = mel[0][~mel_mask[0]].unsqueeze(0).permute(0, 2, 1).float().to(device)
        wav = vocoder.decode(mel).squeeze()
        wav = wav / (wav.abs().max() + 1e-8)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output), wav.cpu(), 24000)
    print(f"OK: salvato {output}")


if __name__ == "__main__":
    main()
