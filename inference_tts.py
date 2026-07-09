#!/usr/bin/env python
import os
import sys
import json
import torch
import argparse
import logging
import torchaudio

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("inference_tts")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import load_hybrid_model, load_vae, load_vocoder, clean_text_and_phonemize
from inference_diffusion import load_diffusion_model
from util import build_tokenizer

def main():
    parser = argparse.ArgumentParser(
        description="End-to-End TTS Inference Script (Hybrid Token-Only + Diffusion-Only)"
    )
    parser.add_argument(
        "--hybrid_checkpoint",
        type=str,
        required=True,
        help="Path to the HybridTTS token-only checkpoint directory",
    )
    parser.add_argument(
        "--diffusion_checkpoint",
        type=str,
        required=True,
        help="Path to the DiffusionOnlyModel checkpoint directory",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        required=True,
        help="Input text string to synthesize (requires g2p_en package)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output path for the generated wav audio (default: output.wav)",
    )
    parser.add_argument(
        "--vocoder",
        type=str,
        default="vocos",
        help="Vocoder type or HuggingFace checkpoint name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on ('cuda', 'mps', or 'cpu')",
    )
    # Hybrid Model generation hyperparams
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for discrete autoregressive token sampling (default: 0.0)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1000,
        help="Maximum generation length of discrete tokens (default: 1000)",
    )
    # Diffusion generation hyperparams
    parser.add_argument(
        "--diffusion_temperature",
        type=float,
        default=1.0,
        help="Temperature for the CFM diffusion head (default: 1.0)",
    )
    parser.add_argument(
        "-dg",
        "--guidance_scale",
        type=float,
        default=1.3,
        help="CFG guidance scale for the diffusion head (default: 1.3)",
    )
    parser.add_argument(
        "-n",
        "--num_steps",
        type=int,
        default=16,
        help="Number of diffusion steps (default: 16)",
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Load configurations
    hybrid_config_path = os.path.join(args.hybrid_checkpoint, "config.json")
    if not os.path.exists(hybrid_config_path):
        logger.error(f"Config file not found at {hybrid_config_path}")
        sys.exit(1)
    with open(hybrid_config_path, "r") as f:
        hybrid_cfg = json.load(f)

    diffusion_config_path = os.path.join(args.diffusion_checkpoint, "config.json")
    if not os.path.exists(diffusion_config_path):
        logger.error(f"Config file not found at {diffusion_config_path}")
        sys.exit(1)
    with open(diffusion_config_path, "r") as f:
        diffusion_cfg = json.load(f)

    scratch_dir = os.environ.get("SCRATCH", "/Users/software/Research")
    if diffusion_cfg.get("vae_checkpoint"):
        diffusion_cfg["vae_checkpoint"] = diffusion_cfg["vae_checkpoint"].replace(
            "$SCRATCH", scratch_dir
        )
    if hybrid_cfg.get("vae_checkpoint"):
        hybrid_cfg["vae_checkpoint"] = hybrid_cfg["vae_checkpoint"].replace(
            "$SCRATCH", scratch_dir
        )

    dtype = torch.float32
    if device.type == "cuda":
        training_cfg = hybrid_cfg.get("training", {})
        if training_cfg.get("bf16") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

    # Load phoneme vocabulary
    vocab_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "phoneme_vocab.json"
    )
    if not os.path.exists(vocab_path):
        logger.error(f"Phoneme vocabulary not found at {vocab_path}!")
        sys.exit(1)
    with open(vocab_path, "r") as f:
        phoneme_vocab = json.load(f)

    logger.info("Building tokenizers...")
    # Tokenizer from hybrid model configuration
    hybrid_tok = build_tokenizer(hybrid_cfg, pretrinaed=False)
    # Tokenizer from diffusion model configuration
    diffusion_tok = build_tokenizer(diffusion_cfg, pretrinaed=False)

    logger.info("Loading models...")
    hybrid_model = load_hybrid_model(
        hybrid_cfg, args.hybrid_checkpoint, device, dtype, tokenizer=hybrid_tok
    )
    
    diffusion_model = load_diffusion_model(
        diffusion_cfg, args.diffusion_checkpoint, device, dtype, tokenizer=diffusion_tok
    )

    vae = load_vae(diffusion_cfg.get("vae_checkpoint"), device, dtype)
    if vae is None:
        logger.error("Could not load VAE model.")
        sys.exit(1)

    vocoder = load_vocoder(args.vocoder, device)
    if vocoder is None:
        logger.error("Could not load Vocoder.")
        sys.exit(1)

    # 1. Prepare Text Input
    prompt_ids = clean_text_and_phonemize(args.text, phoneme_vocab)
    if not prompt_ids:
        logger.error("Empty phoneme input. Nothing to synthesize.")
        sys.exit(1)

    prompt_ids.append(hybrid_tok.start_audio_id)
    
    # 2. Hybrid Model Inference (Discrete Token Generation)
    with torch.no_grad():
        logger.info("Generating discrete tokens with HybridModel...")
        discrete_sequence = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(
            discrete_sequence, dtype=torch.bool, device=device
        )
        batch = {
            "discrete_sequence": discrete_sequence,
            "attention_mask": attention_mask,
        }

        # Hybrid model sample
        sample_out = hybrid_model.sample(
            batch=batch,
            max_steps=args.max_len,
            temperature=args.temperature,
            num_steps=1, # Diffusion not used here
            vae=None,
        )

        final_discrete = sample_out["discrete_tokens"]
        audio_tokens = final_discrete.squeeze(-1).squeeze(0).tolist()

        logger.info(f"Generated {len(audio_tokens)} discrete tokens from HybridModel.")

        # Decode BPE if applicable. The Diffusion model ALWAYS expects raw VAE tokens.
        if getattr(hybrid_tok, "audio_bpe", None) is not None:
            logger.info("Hybrid uses BPE. Decoding BPE audio tokens to raw VAE tokens before diffusion...")
            audio_tokens = hybrid_tok.audio_bpe.decode(audio_tokens)
            logger.info(f"Decoded into {len(audio_tokens)} raw VAE tokens.")

        # 3. Diffusion Model Inference (Continuous Feature Generation)
        logger.info("Generating continuous features with DiffusionOnlyModel...")
        
        # Prepare input for diffusion model
        diffusion_discrete_sequence = torch.tensor([audio_tokens], dtype=torch.long, device=device)
        diffusion_attention_mask = torch.ones_like(diffusion_discrete_sequence, dtype=torch.bool, device=device)
        
        context_vector = diffusion_model.embed(diffusion_discrete_sequence)
        
        if args.guidance_scale > 1.0:
            uncond_discrete = torch.full_like(diffusion_discrete_sequence, diffusion_model.config.pad_token_id)
            uncond_context_vector = diffusion_model.embed(uncond_discrete)
            context_vector = (context_vector, uncond_context_vector)

        upsampled_padding_mask = ~diffusion_attention_mask

        diffusion_out = diffusion_model.diffusion_head.generate(
            num_steps=args.num_steps,
            context_vector=context_vector,
            temperature=args.diffusion_temperature,
            guidance_scale=args.guidance_scale,
            padding_mask=upsampled_padding_mask,
        )

        z_denorm = diffusion_out.audio_features
        z_denorm = diffusion_model.dynamic_normalizer.denormalize(z_denorm)

        # 4. Synthesize Audio (VAE + Vocoder)
        logger.info("Synthesizing waveform using VAE and Vocoder...")
        padding_mask = torch.zeros((1, len(audio_tokens)), dtype=torch.bool, device=device)
        
        tokens_tensor = torch.tensor(audio_tokens, dtype=torch.long, device=device)
        vq_emb = vae.encoder.vq.codebook(tokens_tensor)
        vq_emb = vq_emb.unsqueeze(0)
        
        reconstructed_mel, reconstructed_padding_mask = vae.sample(
            num_steps=16,
            temperature=0.2,
            guidance_scale=1.4,
            z_semantic=vq_emb,
            z_acoustic=z_denorm,
            padding_mask=padding_mask,
        )

        mel = reconstructed_mel[0]
        mask = reconstructed_padding_mask[0]
        mel = mel[~mask].unsqueeze(0).permute(0, 2, 1).float().to(device)
        recon_audio = vocoder.decode(mel).squeeze()
        
        recon_audio = recon_audio / (recon_audio.abs().max() + 1e-8)
        if recon_audio.dim() == 1:
            recon_audio = recon_audio.unsqueeze(0)
            
        torchaudio.save(args.output, recon_audio.cpu(), 24000)
        logger.info(f"SUCCESS: Audio generated and saved to '{args.output}'!")

if __name__ == "__main__":
    main()
