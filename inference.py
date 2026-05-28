#!/usr/bin/env python
# Script updated to use the built-in sample method
import os
import sys
import json
import torch
import argparse
import logging
import torchaudio
from tqdm import tqdm
from omegaconf import OmegaConf
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("inference")

# Add the root directory to path to allow absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.builder import build_model as build_hybrid_model
from modules.submodules.MelCausalVAE.modules.builder import build_model as build_vae
from util import build_tokenizer


def load_vae(
    checkpoint_dir: str, device: torch.device, dtype: torch.dtype
) -> Optional[torch.nn.Module]:
    """Loads the MelCausalVAE model from checkpoint."""
    try:
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
        vae = build_vae(cfg_dict)

        checkpoint_path = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.exists(checkpoint_path):
            vae.from_pretrained(checkpoint_path)
        else:
            checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
            if os.path.exists(checkpoint_path):
                vae.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        vae.eval()
        vae.to(device=device, dtype=dtype)
        logger.info(f"Successfully loaded VAE from {checkpoint_dir}")
        return vae
    except Exception as e:
        logger.error(f"Failed to load VAE from {checkpoint_dir}: {e}")
        return None


def load_vocoder(vocoder_name_or_path: str, device: torch.device):
    """Loads the Vocos vocoder."""
    try:
        from vocos import Vocos

        if vocoder_name_or_path == "bigvgan" or vocoder_name_or_path == "vocos":
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        else:
            vocoder = Vocos.from_pretrained(vocoder_name_or_path).to(device)
        vocoder.eval()
        logger.info(f"Successfully loaded Vocoder: {vocoder_name_or_path}")
        return vocoder
    except Exception as e:
        logger.error(f"Failed to load Vocoder {vocoder_name_or_path}: {e}")
        return None


def load_hybrid_model(
    cfg_dict: Dict[str, Any],
    checkpoint_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    tokenizer,
) -> torch.nn.Module:
    """Builds HybridTTS and loads its weights from safetensors or pt file."""
    model = build_hybrid_model(cfg_dict, tokenizer=tokenizer)

    # Load model weights
    if os.path.isdir(checkpoint_dir):
        safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
        pt_path = os.path.join(checkpoint_dir, "model.pt")
        if os.path.exists(safetensors_path):
            checkpoint_file = safetensors_path
        elif os.path.exists(pt_path):
            checkpoint_file = pt_path
        else:
            raise FileNotFoundError(
                f"No model weight file (model.safetensors or model.pt) found in {checkpoint_dir}"
            )
    else:
        checkpoint_file = checkpoint_dir

    if checkpoint_file.endswith(".safetensors"):
        from safetensors.torch import load_file as load_safetensors

        state_dict = load_safetensors(checkpoint_file, device="cpu")
    else:
        state_dict = torch.load(checkpoint_file, map_location="cpu")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device=device, dtype=dtype)
    logger.info(f"Successfully loaded HybridTTS model from {checkpoint_file}")
    return model


def clean_text_and_phonemize(text: str, vocab: Dict[str, int]) -> List[int]:
    """Converts a standard English text into phoneme IDs using g2p_en or prompts input phonemes."""
    try:
        from g2p_en import G2p

        logger.info("Initializing G2P model...")
        g2p = G2p()
        phonemes = g2p(text)
        logger.info(f"G2P output: {phonemes}")
    except ImportError:
        logger.error(
            "Error: g2p_en is not installed!\n"
            "Please run: pip install g2p_en\n"
            "Alternatively, run this script using the --phonemes flag to specify ARPAbet phonemes directly."
        )
        sys.exit(1)

    phoneme_ids = []
    skipped_tokens = []
    for p in phonemes:
        # Match against phoneme vocabulary (case-sensitive)
        if p in vocab:
            phoneme_ids.append(vocab[p])
        else:
            # Accumulate skipped tokens (e.g. spaces, punctuations) for debugging
            if p.strip():
                skipped_tokens.append(p)

    if skipped_tokens:
        logger.debug(f"Skipped unknown tokens: {skipped_tokens}")

    logger.info(f"Mapped {len(phoneme_ids)} phonemes to vocabulary IDs: {phoneme_ids}")
    return phoneme_ids


def main():
    parser = argparse.ArgumentParser(
        description="Simple TTS Inference Script for HybridTTS Model"
    )
    parser.add_argument(
        "--hybrid_checkpoint",
        type=str,
        required=True,
        help="Path to the HybridTTS checkpoint directory",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text",
        "-t",
        type=str,
        help="Input text string to synthesize (requires g2p_en package)",
    )
    group.add_argument(
        "--phonemes",
        "-p",
        type=str,
        help="Direct space-separated ARPAbet phoneme sequence (e.g., 'HH AH0 L OW1 W ER1 L D')",
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
        help="Vocoder type or HuggingFace checkpoint name (default: vocos -> charactr/vocos-mel-24khz)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on ('cuda', 'mps', or 'cpu'). Auto-selects if not provided.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=4,
        help="Number of diffusion steps for latents generation and VAE decoding (default: 16)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for discrete autoregressive token sampling (default: 0.0)",
    )
    parser.add_argument(
        "--diffusion_temperature",
        type=float,
        default=1.0,
        help="Temperature for the CFM diffusion head (default: 0.2)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="CFG guidance scale for the diffusion head (default: 1.3)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k filtering for discrete autoregressive sampling (default: 50, 0 to disable)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p nucleus filtering for discrete autoregressive sampling (default: 0.95, 1.0 to disable)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=500,
        help="Maximum generation length of discrete tokens (default: 500)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=2.2,
        help="Ratio of generated VQ frames per phoneme (default: 2.2)",
    )
    parser.add_argument(
        "--token_first",
        action="store_true",
        help="Ablation mode: generate all discrete tokens first, and then run diffusion once on the sequence (default: False)",
    )
    parser.add_argument(
        "--decode_only_token",
        action="store_true",
        help="Use only the generated quantized tokens in the VAE (continuous features set to zero) (default: False)",
    )
    args = parser.parse_args()

    # Device selection
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

    # Resolve SCRATCH environment variable
    scratch_dir = os.environ.get("SCRATCH", "/Users/software/Research")

    # Load configuration
    config_path = os.path.join(args.hybrid_checkpoint, "config.json")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        sys.exit(1)

    logger.info(f"Loading HybridTTS configuration from {config_path}...")
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    # Resolve vae_checkpoint path
    if cfg_dict.get("vae_checkpoint"):
        cfg_dict["vae_checkpoint"] = cfg_dict["vae_checkpoint"].replace(
            "$SCRATCH", scratch_dir
        )
    # Determine appropriate precision (dtype) for stability on MPS/CPU
    if device.type == "mps":
        # Force float32 on MPS because bfloat16/float16 support is unstable/incomplete in many MPS kernels
        dtype = torch.float32
        logger.info("Using float32 precision for MPS device compatibility.")
    elif device.type == "cuda":
        # Check if the training config specified bf16 and check GPU capability
        training_cfg = cfg_dict.get("training")
        use_bf16 = training_cfg.get("bf16") if training_cfg is not None else None
        if use_bf16 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            logger.info("Using bfloat16 precision on CUDA.")
        else:
            dtype = torch.float32
            logger.info("Using float32 precision on CUDA.")
    else:
        dtype = torch.float32
        logger.info("Using float32 precision on CPU.")

    # Load phoneme vocabulary
    vocab_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "phoneme_vocab.json"
    )
    if not os.path.exists(vocab_path):
        logger.error(f"Phoneme vocabulary not found at {vocab_path}!")
        sys.exit(1)
    with open(vocab_path, "r") as f:
        phoneme_vocab = json.load(f)

    # Build tokenizer first as the single source of truth
    logger.info("Building tokenizer...")
    tok = build_tokenizer(cfg_dict, pretrinaed=False)

    logger.info("Loading models...")
    hybrid_model = load_hybrid_model(
        cfg_dict, args.hybrid_checkpoint, device, dtype, tokenizer=tok
    )
    vae = load_vae(cfg_dict["vae_checkpoint"], device, dtype)
    if vae is None:
        logger.error("Could not load VAE model.")
        sys.exit(1)

    vocoder = load_vocoder(args.vocoder, device)
    if vocoder is None:
        logger.error("Could not load Vocoder.")
        sys.exit(1)

    # Parse inputs to phoneme IDs
    if args.phonemes:
        input_phonemes = args.phonemes.split()
        prompt_ids = []
        for p in input_phonemes:
            if p in phoneme_vocab:
                prompt_ids.append(phoneme_vocab[p])
            else:
                logger.warning(f"Phoneme '{p}' not found in vocabulary, skipping.")
        logger.info(f"Parsed direct phoneme sequence: {input_phonemes} -> {prompt_ids}")
    else:
        prompt_ids = clean_text_and_phonemize(args.text, phoneme_vocab)

    if not prompt_ids:
        logger.error("Empty phoneme input. Nothing to synthesize.")
        sys.exit(1)

    # Map prompt_ids to unified vocab and append <start_audio>

    prompt_ids.append(tok.start_audio_id)

    # Calculate target length based on prompt length if default max_len is used
    if args.max_len == 500:
        target_len = int(len(prompt_ids) * args.ratio)
        logger.info(
            f"Target generation length calculated from prompt length: {target_len} frames (ratio={args.ratio})"
        )
    else:
        target_len = args.max_len
        logger.info(f"Using explicitly specified max_len: {target_len} frames")

    with torch.no_grad():
        logger.info("Running built-in sample function...")
        discrete_sequence = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(
            discrete_sequence, dtype=torch.bool, device=device
        )

        batch = {
            "discrete_sequence": discrete_sequence,
            "attention_mask": attention_mask,
        }

        sample_out = hybrid_model.sample(
            batch=batch,
            max_steps=target_len,
            temperature=args.temperature,
            num_steps=args.num_steps,
            diffusion_temperature=args.diffusion_temperature,
            guidance_scale=args.guidance_scale,
        )

        final_discrete = sample_out["discrete_tokens"]
        z_denorm = sample_out["continuous_tokens"]

        audio_tokens = final_discrete.squeeze(-1).squeeze(0).tolist()

        logger.info("Decoding continuous features using VAE...")
        if z_denorm is not None:
            padding_mask = torch.zeros(
                z_denorm.shape[:2], dtype=torch.bool, device=device
            )
        else:
            padding_mask = torch.zeros(
                (1, len(audio_tokens)), dtype=torch.bool, device=device
            )

        tokens_tensor = torch.tensor(audio_tokens, dtype=torch.long, device=device)
        vq_emb = vae.encoder.vq.codebook(tokens_tensor)  # [L, 32]
        vq_emb = vq_emb.unsqueeze(0)  # [1, L, 32]

        if args.decode_only_token or z_denorm is None:
            logger.info(
                "Using only generated quantized tokens (continuous features zeroed out)."
            )
            z_denorm = torch.zeros(
                (1, len(audio_tokens), hybrid_model.config.continuous_dim),
                dtype=dtype,
                device=device,
            )

        z_vae = torch.cat([vq_emb, z_denorm], dim=-1)

        # Using hardcoded parameters for VAE sample as before
        reconstructed_mel, reconstructed_padding_mask = vae.sample(
            num_steps=16,
            temperature=0.2,
            guidance_scale=1.0,
            z=z_vae,
            padding_mask=padding_mask,
        )

        # Step 5: Synthesize waveform using Vocoder
        logger.info("Synthesizing waveform using Vocoder...")
        mel = reconstructed_mel[0]
        mask = reconstructed_padding_mask[0]
        # Squeeze the padding mask if applicable
        mel = mel[~mask].unsqueeze(0).permute(0, 2, 1).float().to(device)

        recon_audio = vocoder.decode(mel).squeeze()

        # Normalize audio and save to disk
        recon_audio = recon_audio / (recon_audio.abs().max() + 1e-8)

        # Ensure correct shape [1, num_samples]
        if recon_audio.dim() == 1:
            recon_audio = recon_audio.unsqueeze(0)

        torchaudio.save(args.output, recon_audio.cpu(), 24000)
        logger.info(f"SUCCESS: Audio generated and saved to '{args.output}'!")


if __name__ == "__main__":
    main()
