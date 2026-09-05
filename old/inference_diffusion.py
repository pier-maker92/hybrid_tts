#!/usr/bin/env python
import os
import sys
import json
import torch
import argparse
import logging
import torchaudio
from tqdm import tqdm
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("inference_diffusion")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_diffusion import DiffusionOnlyModel
from util import build_tokenizer
from data.audio_dataset import DiffusionDataCollator, TrainDatasetWrapper
from inference import load_vae, load_vocoder

def load_diffusion_model(
    cfg_dict: dict,
    checkpoint_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    tokenizer
) -> torch.nn.Module:
    """Builds DiffusionOnlyModel and loads its weights from safetensors, pt or bin file."""
    from modules.builder import load_codebook_config_from_cfg
    from modules.configs import HybridTTSConfig, DiTConfig
    
    scratch_dir = os.environ.get("SCRATCH", "/Users/software/Research")
    vae_checkpoint = cfg_dict.get("vae_checkpoint")
    if vae_checkpoint:
        vae_checkpoint = vae_checkpoint.replace("$SCRATCH", scratch_dir)
        cfg_dict["vae_checkpoint"] = vae_checkpoint
        
    continuous_dim, _ = load_codebook_config_from_cfg(cfg_dict)
    
    diffusion_head_cfg = cfg_dict.get("diffusion_head", cfg_dict.get("diffusion_head_config", {}))
    diffusion_config = DiTConfig(**diffusion_head_cfg)
    
    # Override with actual dataset/model dimensions
    diffusion_config.audio_latent_dim = continuous_dim
    diffusion_config.backbone_dim = diffusion_config.net_dim

    config = HybridTTSConfig(
        backbone_config=None,
        diffusion_head_config=diffusion_config,
        continuous_adapter_config=None,
        prompt_vocab_size=tokenizer.prompt_vocab_size,
        discrete_token_vocab_size=tokenizer.discrete_token_vocab_size,
        continuous_dim=continuous_dim,
        pad_token_id=tokenizer.pad_id,
        start_audio_id=tokenizer.start_audio_id,
        end_audio_id=tokenizer.end_audio_id,
        shift_audio_offset=cfg_dict.get("training", {}).get("shift_audio_offset", 1),
    )
    
    model = DiffusionOnlyModel(config, tokenizer=tokenizer)

    # Load model weights
    if os.path.isdir(checkpoint_dir):
        safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
        pt_path = os.path.join(checkpoint_dir, "model.pt")
        bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path):
            checkpoint_file = safetensors_path
        elif os.path.exists(bin_path):
            checkpoint_file = bin_path
        elif os.path.exists(pt_path):
            checkpoint_file = pt_path
        else:
            raise FileNotFoundError(
                f"No model weight file found in {checkpoint_dir}"
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
    logger.info(f"Successfully loaded DiffusionOnlyModel from {checkpoint_file}")
    return model

def main():
    parser = argparse.ArgumentParser(
        description="Inference Script for DiffusionOnlyModel (Dataset loading)"
    )
    parser.add_argument(
        "-c", "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the DiffusionOnlyModel checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="diffusion_outputs",
        help="Output directory for generated wav audio files",
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
        help="Device to run inference on",
    )
    parser.add_argument(
        "-n", "--num_steps",
        type=int,
        default=16,
        help="Number of diffusion steps (default: 4)",
    )
    parser.add_argument(
        "--diffusion_temperature",
        type=float,
        default=1.0,
        help="Temperature for the CFM diffusion head (default: 1.0)",
    )
    parser.add_argument(
        "-dg", "--guidance_scale",
        type=float,
        default=1.0,
        help="CFG guidance scale for the diffusion head (default: 1.0)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate from dataset",
    )
    parser.add_argument(
        "--save_original",
        action="store_true",
        help="If set, also decodes and saves the original ground-truth features for comparison",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to load (overrides the one in config.json)",
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
    
    os.makedirs(args.output_dir, exist_ok=True)

    config_path = os.path.join(args.checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    scratch_dir = os.environ.get("SCRATCH")
    if cfg_dict.get("vae_checkpoint"):
        cfg_dict["vae_checkpoint"] = cfg_dict["vae_checkpoint"].replace(
            "$SCRATCH", scratch_dir
        )
        
    dtype = torch.float32
    if device.type == "cuda":
        training_cfg = cfg_dict.get("training", {})
        if training_cfg.get("bf16") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

    logger.info("Building tokenizer...")
    tok = build_tokenizer(cfg_dict, pretrinaed=False)

    logger.info("Building dataset...")
    training_cfg = cfg_dict.get("training", {})
    dataset_name = args.dataset if args.dataset else training_cfg.get("dataset_name")
    logger.info(f"Using dataset: {dataset_name}")
    
    force_vocab_build = training_cfg.get("force_vocab_build", False)
    discrete_only = training_cfg.get("discrete_only", False)
    
    if dataset_name == "ljspeech-prepared":
        from data.lj_speech_prepared import LJSpeechDataset
        base_dataset = LJSpeechDataset(force_vocab_build=force_vocab_build)
        split = "train"  # LJSpeech only has a train split
    elif dataset_name in ["libritts-r-prepared", "libritts_r_prepared"]:
        from data.libri_tts_r_prepared import LibriTTSRPrepared
        base_dataset = LibriTTSRPrepared()
        split = "test"
    else:
        raise ValueError(f"Dataset {dataset_name} is not fully supported for direct loading in this script yet. Please add it.")
        
    test_dataset = TrainDatasetWrapper(base_dataset, split, discrete_only=discrete_only)

    logger.info("Loading models...")
    model = load_diffusion_model(cfg_dict, args.checkpoint_dir, device, dtype, tok)
    
    vae = load_vae(cfg_dict.get("vae_checkpoint"), device, dtype)
    if vae is None:
        logger.error("Could not load VAE model.")
        sys.exit(1)

    vocoder = load_vocoder(args.vocoder, device)
    if vocoder is None:
        logger.error("Could not load Vocoder.")
        sys.exit(1)

    data_collator = DiffusionDataCollator(pad_id=tok.discrete_token_vocab_size, tokenizer=tok)
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=data_collator)

    logger.info(f"Starting generation of {args.num_samples} samples...")
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= args.num_samples:
                break
                
            logger.info(f"Processing sample {i+1}/{args.num_samples}")
            
            discrete_sequence = batch["discrete_sequence"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            context_vector = model.embed(discrete_sequence)
            
            if args.guidance_scale > 1.0:
                uncond_discrete = torch.full_like(discrete_sequence, model.config.pad_token_id)
                uncond_context_vector = model.embed(uncond_discrete)
                context_vector = (context_vector, uncond_context_vector)

            upsampled_padding_mask = ~attention_mask

            sample_out = model.diffusion_head.generate(
                num_steps=args.num_steps,
                context_vector=context_vector,
                temperature=args.diffusion_temperature,
                guidance_scale=args.guidance_scale,
                padding_mask=upsampled_padding_mask,
            )

            z_denorm = sample_out.audio_features
            z_denorm = model.dynamic_normalizer.denormalize(z_denorm)
            
            final_discrete = discrete_sequence.squeeze(0).tolist()
            if getattr(tok, "audio_bpe", None) is not None:
                # Remove padding tokens
                valid_len = attention_mask.sum().item()
                final_discrete = final_discrete[:valid_len]
                final_discrete = tok.audio_bpe.decode(final_discrete)
                
            padding_mask = torch.zeros((1, len(final_discrete)), dtype=torch.bool, device=device)
            
            tokens_tensor = torch.tensor(final_discrete, dtype=torch.long, device=device)
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
                
            out_path = os.path.join(args.output_dir, f"sample_{i}_generated.wav")
            torchaudio.save(out_path, recon_audio.cpu(), 24000)
            logger.info(f"Saved generated audio to {out_path}")
            
            # Save original if requested
            if args.save_original and batch.get("continuous_sequence") is not None:
                orig_c_tokens = batch["continuous_sequence"].to(device=device, dtype=dtype)
                
                orig_mel, orig_mask = vae.sample(
                    num_steps=16,
                    temperature=0.2,
                    guidance_scale=1.4,
                    z_semantic=vq_emb,
                    z_acoustic=orig_c_tokens,
                    padding_mask=padding_mask,
                )
                
                omel = orig_mel[0]
                omask = orig_mask[0]
                omel = omel[~omask].unsqueeze(0).permute(0, 2, 1).float().to(device)
                orig_audio = vocoder.decode(omel).squeeze()
                
                orig_audio = orig_audio / (orig_audio.abs().max() + 1e-8)
                if orig_audio.dim() == 1:
                    orig_audio = orig_audio.unsqueeze(0)
                    
                orig_out_path = os.path.join(args.output_dir, f"sample_{i}_original.wav")
                torchaudio.save(orig_out_path, orig_audio.cpu(), 24000)
                logger.info(f"Saved original audio to {orig_out_path}")

if __name__ == "__main__":
    main()
