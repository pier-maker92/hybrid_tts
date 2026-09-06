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
from modules.modalities import resolve_modalities
from modules.submodules.MelCausalVAE.dicodec.modules.builder import (
    build_model as build_vae,
    load_external_semantic_quantizer,
)
from util import build_tokenizer


def load_vae(
    checkpoint_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    training_cfg: Optional[Dict[str, Any]] = None,
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
        training_cfg = training_cfg or {}
        discrete, _ = resolve_modalities(training_cfg)
        if discrete and training_cfg.get("semantic_quantizer_checkpoint"):
            load_external_semantic_quantizer(
                vae,
                checkpoint_path=training_cfg["semantic_quantizer_checkpoint"],
                quantizer_type=training_cfg.get("semantic_quantizer_type", "std_vq"),
                codebook_size=training_cfg.get("semantic_codebook_size"),
                target_source=training_cfg.get("audio_quantizer_source"),
            )
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
        bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            checkpoint_file = safetensors_path
        elif os.path.exists(bin_path):
            checkpoint_file = bin_path
        elif os.path.exists(pt_path):
            checkpoint_file = pt_path
        else:
            raise FileNotFoundError(
                f"No model weight file (model.safetensors, pytorch_model.bin or model.pt) found in {checkpoint_dir}"
            )
    else:
        checkpoint_file = checkpoint_dir

    if checkpoint_file.endswith(".safetensors"):
        from safetensors.torch import load_file as load_safetensors

        state_dict = load_safetensors(checkpoint_file, device="cpu")
    else:
        state_dict = torch.load(checkpoint_file, map_location="cpu")

    # Retrocompatibility for CausalLMWrapper refactoring #FIXME this is supposed to be removed in the future
    model_state_keys = set(model.state_dict().keys())
    new_state_dict = {}
    for k, v in state_dict.items():
        if k not in model_state_keys and k.startswith("backbone."):
            alt_k = k.replace("backbone.", "backbone.model.", 1)
            if alt_k in model_state_keys:
                new_state_dict[alt_k] = v
                continue
        new_state_dict[k] = v
    state_dict = new_state_dict

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


def load_phoneme_vocab() -> Dict[str, int]:
    vocab_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "phoneme_vocab.json"
    )
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Phoneme vocabulary not found at {vocab_path}!")
    with open(vocab_path, "r") as f:
        return json.load(f)


def encode_text_prompt(
    text: str, tokenizer, phoneme_vocab: Optional[Dict[str, int]] = None
) -> List[int]:
    if getattr(tokenizer, "char_tokenizer", None) is not None:
        prompt_ids = tokenizer.encode_text(text)
        logger.info(f"Mapped {len(prompt_ids)} chars to vocabulary IDs: {prompt_ids}")
        return prompt_ids

    if phoneme_vocab is None:
        phoneme_vocab = load_phoneme_vocab()
    return clean_text_and_phonemize(text, phoneme_vocab)


def resolve_local_research_path(path: Optional[str], scratch_dir: str) -> Optional[str]:
    if not path:
        return path
    resolved = path.replace("$SCRATCH", scratch_dir)
    if os.path.exists(resolved):
        return resolved

    scratch_prefix = "/scratch/piermel/"
    if resolved.startswith(scratch_prefix):
        candidate = os.path.join(scratch_dir, resolved[len(scratch_prefix) :])
        if os.path.exists(candidate):
            return candidate

    return resolved


def load_kmeans_centroids(
    path: Optional[str], device: torch.device, dtype: torch.dtype
):
    if not path:
        return None
    kmeans_file = (
        os.path.join(path, "encoder_kmeans.pt") if os.path.isdir(path) else path
    )
    if not os.path.exists(kmeans_file):
        raise FileNotFoundError(f"kmeans checkpoint not found: {kmeans_file}")
    codebook = torch.load(kmeans_file, map_location="cpu")
    if "centroids" not in codebook:
        raise ValueError(f"kmeans checkpoint has no 'centroids': {kmeans_file}")
    centroids = codebook["centroids"].to(device=device, dtype=dtype)
    logger.info(
        f"Loaded kmeans centroids from {kmeans_file}: shape={tuple(centroids.shape)}"
    )
    return centroids


def load_voice_condition(path: str, vae: torch.nn.Module, device: torch.device):
    if not hasattr(vae, "extract_speaker_embedding"):
        raise RuntimeError("Loaded VAE does not support speaker embedding extraction.")
    audios_srs = load_voice_reference_audio(path, device)
    speaker_embedding = vae.extract_speaker_embedding(audios_srs)
    if speaker_embedding is None:
        raise RuntimeError(
            "Voice conditioning requires a VAE checkpoint with speaker_encoder_config."
        )
    return speaker_embedding.to(device=device, dtype=next(vae.parameters()).dtype)


def load_voice_reference_audio(path: str, device: torch.device):
    path = os.path.expandvars(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Voice condition audio not found: {path}")

    wav, sample_rate = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)
    max_abs = wav.abs().max()
    if max_abs > 0:
        wav = wav / max_abs

    return [(wav.to(device), sample_rate)]


def align_continuous_tokens(
    z_denorm: Optional[torch.Tensor],
    length: int,
    continuous_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    if z_denorm is None:
        return torch.zeros((1, length, continuous_dim), dtype=dtype, device=device)
    z_denorm = z_denorm.to(device=device, dtype=dtype)
    if z_denorm.shape[1] == length:
        return z_denorm
    if z_denorm.shape[1] < length:
        missing = length - z_denorm.shape[1]
        pad = torch.zeros(
            (z_denorm.shape[0], missing, z_denorm.shape[2]),
            dtype=z_denorm.dtype,
            device=device,
        )
        logger.info(
            f"Continuous tokens shorter than discrete tokens; appending {missing} zero frame(s)."
        )
        return torch.cat([z_denorm, pad], dim=1)
    logger.info(
        f"Continuous tokens longer than discrete tokens; trimming {z_denorm.shape[1]} -> {length}."
    )
    return z_denorm[:, :length]


def discrete_tokens_to_semantic_latents(
    vae: torch.nn.Module,
    tokens_tensor: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    external_quantizer = getattr(vae, "external_semantic_quantizer", None)
    if external_quantizer is None:
        if hasattr(vae.encoder, "vq"):
            return vae.encoder.vq.codebook(tokens_tensor).unsqueeze(0)
        raise RuntimeError(
            "Cannot decode discrete tokens: VAE has neither encoder.vq nor an "
            "external semantic quantizer."
        )

    logger.info("Decoding discrete tokens with the external semantic quantizer.")

    quantizer = getattr(external_quantizer, "quantizer", None)
    quantizer_module = getattr(quantizer, "quantizer_module", None)
    if quantizer_module is None:
        raise RuntimeError(
            "Cannot decode discrete tokens: external semantic quantizer does not "
            "expose quantizer.quantizer_module."
        )

    embedding = getattr(quantizer_module, "embedding", None)
    if embedding is None:
        codebook = getattr(quantizer_module, "codebook", None)
        if codebook is None:
            raise RuntimeError(
                "Cannot decode discrete tokens: missing quantizer codebook."
            )
        codes = torch.nn.functional.embedding(tokens_tensor, codebook)
    elif isinstance(embedding, torch.nn.Embedding):
        codes = embedding(tokens_tensor)
    else:
        codes = torch.nn.functional.embedding(tokens_tensor, embedding)

    codes = codes.unsqueeze(0).to(device=device, dtype=dtype)
    valid_mask = torch.ones(codes.shape[:2], dtype=torch.bool, device=device)
    return external_quantizer.decoder(codes, valid_mask=valid_mask.unsqueeze(-1))


def vae_context_dim(vae: torch.nn.Module) -> Optional[int]:
    context_proj = getattr(getattr(vae, "decoder", None), "context_vector_proj", None)
    if context_proj is None:
        return None
    for module in context_proj.modules():
        if isinstance(module, torch.nn.Linear):
            return module.in_features
    return None


def combine_semantic_and_acoustic_latents(
    z_semantic: torch.Tensor,
    z_acoustic: torch.Tensor,
    vae: torch.nn.Module,
) -> torch.Tensor:
    expected_dim = vae_context_dim(vae)
    concat_dim = z_semantic.shape[-1] + z_acoustic.shape[-1]
    if expected_dim == concat_dim:
        return torch.cat([z_semantic, z_acoustic], dim=-1)
    if expected_dim == z_semantic.shape[-1] == z_acoustic.shape[-1]:
        return z_semantic + z_acoustic
    if expected_dim == z_semantic.shape[-1]:
        logger.warning(
            "VAE decoder expects %s dims; ignoring acoustic latents with dim %s.",
            expected_dim,
            z_acoustic.shape[-1],
        )
        return z_semantic
    raise RuntimeError(
        "Cannot combine semantic/acoustic latents for VAE decoder: "
        f"semantic_dim={z_semantic.shape[-1]}, acoustic_dim={z_acoustic.shape[-1]}, "
        f"decoder_context_dim={expected_dim}."
    )


def trim_unpaired_discrete_tokens(
    tokens_tensor: torch.Tensor, z_denorm: Optional[torch.Tensor]
):
    if z_denorm is None:
        return tokens_tensor
    continuous_len = z_denorm.shape[1]
    discrete_len = tokens_tensor.numel()
    if continuous_len < discrete_len:
        logger.info(
            f"Dropping {discrete_len - continuous_len} trailing discrete token(s) without matching continuous frame."
        )
        return tokens_tensor[:continuous_len]
    return tokens_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Simple TTS Inference Script for HybridTTS Model"
    )
    parser.add_argument(
        "-c",
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
        action="append",
        help="Input text string to synthesize (requires g2p_en package)",
    )
    group.add_argument(
        "--phonemes",
        "-p",
        type=str,
        action="append",
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
        "-n",
        "--num_steps",
        type=int,
        default=4,
        help="Number of diffusion steps for latents generation and VAE decoding (default: 4)",
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
        help="Temperature for the CFM diffusion head (default: 1.0)",
    )
    parser.add_argument(
        "-dg",
        "--guidance_scale",
        type=float,
        default=1.0,
        help="CFG guidance scale for the diffusion head (default: 1.0)",
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
        "--no_ratio",
        action="store_true",
        help="Use max_len only as an EOS generation cap instead of deriving length from prompt ratio.",
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
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default=None,
        help="Override VAE checkpoint path from the HybridTTS config.",
    )
    parser.add_argument(
        "--kmeans_path",
        type=str,
        default=None,
        help="Override kmeans path from the HybridTTS config.",
    )
    parser.add_argument(
        "--voice_condition",
        type=str,
        default=None,
        help="Reference audio file used to extract a speaker embedding for DiCodec decoder FiLM conditioning.",
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

    if args.vae_checkpoint:
        cfg_dict["vae_checkpoint"] = args.vae_checkpoint
    if args.kmeans_path:
        cfg_dict["kmeans_path"] = args.kmeans_path

    cfg_dict["vae_checkpoint"] = resolve_local_research_path(
        cfg_dict.get("vae_checkpoint"), scratch_dir
    )
    cfg_dict["kmeans_path"] = resolve_local_research_path(
        cfg_dict.get("kmeans_path"), scratch_dir
    )
    training_cfg = cfg_dict.get("training", {}) or {}
    if training_cfg.get("semantic_quantizer_checkpoint"):
        training_cfg["semantic_quantizer_checkpoint"] = resolve_local_research_path(
            training_cfg.get("semantic_quantizer_checkpoint"), scratch_dir
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

    # Build tokenizer first as the single source of truth
    logger.info("Building tokenizer...")
    tok = build_tokenizer(cfg_dict, pretrinaed=False)
    phoneme_vocab = None
    if getattr(tok, "char_tokenizer", None) is None:
        try:
            phoneme_vocab = load_phoneme_vocab()
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
    kmeans_centroids = load_kmeans_centroids(
        cfg_dict.get("kmeans_path"), device=device, dtype=dtype
    )

    logger.info("Loading models...")
    hybrid_model = load_hybrid_model(
        cfg_dict, args.hybrid_checkpoint, device, dtype, tokenizer=tok
    )
    requires_voice_condition = bool(
        getattr(hybrid_model, "backbone_voice_condition", False)
        or getattr(hybrid_model, "diffusion_voice_condition", False)
    )
    if requires_voice_condition and not args.voice_condition:
        logger.error(
            "This checkpoint was trained with voice_condition=true. "
            "Pass --voice_condition with a reference audio file so the loaded VAE can "
            "extract the speaker embedding."
        )
        sys.exit(1)
    vae = load_vae(cfg_dict["vae_checkpoint"], device, dtype, training_cfg=training_cfg)
    if vae is None:
        logger.error("Could not load VAE model.")
        sys.exit(1)

    speaker_embedding = None
    voice_reference_audios_srs = None
    if args.voice_condition:
        logger.info(f"Extracting speaker embedding from {args.voice_condition}...")
        try:
            voice_reference_audios_srs = load_voice_reference_audio(
                args.voice_condition,
                device,
            )
            speaker_embedding = load_voice_condition(args.voice_condition, vae, device)
        except Exception as e:
            logger.error(f"Could not load voice condition: {e}")
            sys.exit(1)
        logger.info(
            f"Voice conditioning enabled: speaker_embedding shape={tuple(speaker_embedding.shape)}"
        )

    vocoder = load_vocoder(args.vocoder, device)
    if vocoder is None:
        logger.error("Could not load Vocoder.")
        sys.exit(1)

    # Parse inputs to phoneme IDs
    prompt_batches = []
    input_labels = []
    if args.phonemes:
        if getattr(tok, "char_tokenizer", None) is not None:
            logger.error(
                "--phonemes cannot be used with text_tokenizer='char'. Use --text."
            )
            sys.exit(1)
        for phoneme_index, phoneme_text in enumerate(args.phonemes):
            input_phonemes = phoneme_text.split()
            prompt_ids = []
            for p in input_phonemes:
                if p in phoneme_vocab:
                    prompt_ids.append(phoneme_vocab[p])
                else:
                    logger.warning(f"Phoneme '{p}' not found in vocabulary, skipping.")
            logger.info(
                f"Parsed direct phoneme sequence {phoneme_index}: {input_phonemes} -> {prompt_ids}"
            )
            if not prompt_ids:
                logger.error(f"Empty phoneme input for phoneme index {phoneme_index}.")
                sys.exit(1)
            prompt_batches.append(prompt_ids)
            input_labels.append(f"phonemes_{phoneme_index}")
    else:
        for text_index, text in enumerate(args.text):
            prompt_ids = encode_text_prompt(text, tok, phoneme_vocab)
            if not prompt_ids:
                logger.error(
                    f"Empty text input after tokenization for text index {text_index}."
                )
                sys.exit(1)
            prompt_batches.append(prompt_ids)
            input_labels.append(f"text_{text_index}")

    if not prompt_batches:
        logger.error("Empty input. Nothing to synthesize.")
        sys.exit(1)

    # Map prompt_ids to unified vocab and append <start_audio>
    for prompt_ids in prompt_batches:
        prompt_ids.append(tok.start_audio_id)

    # Calculate target length based on prompt length if requested by legacy default.
    if args.max_len == 500 and not args.no_ratio:
        target_len = max(
            int(len(prompt_ids) * args.ratio) for prompt_ids in prompt_batches
        )
        logger.info(
            f"Target generation length calculated from prompt length: {target_len} frames (ratio={args.ratio})"
        )
    else:
        target_len = args.max_len
        logger.info(f"Using max_len as EOS cap: {target_len} frames")

    with torch.no_grad():
        logger.info(
            f"Running built-in sample function for batch_size={len(prompt_batches)}..."
        )
        prompt_tensors = [
            torch.tensor(prompt_ids, dtype=torch.long, device=device)
            for prompt_ids in prompt_batches
        ]
        discrete_sequence = torch.nn.utils.rnn.pad_sequence(
            prompt_tensors,
            batch_first=True,
            padding_value=tok.pad_id,
        )
        attention_mask = torch.zeros_like(discrete_sequence, dtype=torch.bool)
        for sample_index, prompt_ids in enumerate(prompt_batches):
            attention_mask[sample_index, : len(prompt_ids)] = True

        batch = {
            "discrete_sequence": discrete_sequence,
            "attention_mask": attention_mask,
        }

        # intatiate genertor with seed 42
        generator = torch.Generator(device=device)
        generator.manual_seed(42)

        sample_out = hybrid_model.sample(
            batch=batch,
            max_steps=target_len,
            temperature=args.temperature,
            num_steps=args.num_steps,
            diffusion_temperature=args.diffusion_temperature,
            guidance_scale=args.guidance_scale,
            vae=vae,
            generator=generator,
            reference_audios_srs=voice_reference_audios_srs,
            voice_conditioner=vae,
        )

        final_discrete = sample_out["discrete_tokens"]
        z_denorm = sample_out["continuous_tokens"]
        discrete_lengths = sample_out.get("discrete_lengths")
        if discrete_lengths is None:
            discrete_lengths = (final_discrete.squeeze(-1) >= 0).sum(dim=1)

        output_root, output_ext = os.path.splitext(args.output)
        if output_ext == "":
            output_ext = ".wav"
        multi_output = len(prompt_batches) > 1

        for sample_index in range(len(prompt_batches)):
            token_len = int(discrete_lengths[sample_index].item())
            sample_discrete = final_discrete[sample_index]
            if sample_discrete.ndim == 2:
                sample_discrete = sample_discrete[:, 0]
            audio_tokens = sample_discrete[:token_len].clamp_min(0).long().tolist()

            if getattr(tok, "audio_bpe", None) is not None:
                logger.info("Decoding BPE audio tokens to VAE tokens...")
                audio_tokens = tok.audio_bpe.decode(audio_tokens)

            if not audio_tokens:
                logger.error(
                    f"No audio tokens were generated for sample {sample_index}."
                )
                sys.exit(1)

            z_sample = (
                None
                if z_denorm is None
                else z_denorm[sample_index : sample_index + 1, :token_len]
            )

            tokens_tensor = torch.tensor(audio_tokens, dtype=torch.long, device=device)
            padding_mask = torch.zeros(
                (1, len(audio_tokens)), dtype=torch.bool, device=device
            )

            if kmeans_centroids is not None:
                logger.info(
                    "Decoding kmeans discrete tokens plus continuous features using VAE..."
                )
                if not args.decode_only_token:
                    tokens_tensor = trim_unpaired_discrete_tokens(
                        tokens_tensor, z_sample
                    )
                    audio_tokens = tokens_tensor.tolist()
                    padding_mask = torch.zeros(
                        (1, len(audio_tokens)), dtype=torch.bool, device=device
                    )
                if tokens_tensor.numel() == 0:
                    logger.error(
                        f"No audio tokens were generated for sample {sample_index}."
                    )
                    sys.exit(1)
                if (
                    tokens_tensor.min().item() < 0
                    or tokens_tensor.max().item() >= kmeans_centroids.shape[0]
                ):
                    logger.error(
                        f"Generated token out of kmeans range for sample {sample_index}: "
                        f"min={tokens_tensor.min().item()}, max={tokens_tensor.max().item()}, "
                        f"clusters={kmeans_centroids.shape[0]}"
                    )
                    sys.exit(1)
                z_semantic = kmeans_centroids.index_select(0, tokens_tensor).unsqueeze(
                    0
                )
                if args.decode_only_token or z_sample is None:
                    logger.info(
                        "Using only generated kmeans tokens; continuous features zeroed out."
                    )
                    z_sample = None
                z_acoustic = align_continuous_tokens(
                    z_sample,
                    length=len(audio_tokens),
                    continuous_dim=hybrid_model.config.continuous_dim,
                    dtype=dtype,
                    device=device,
                )
                z = torch.cat([z_semantic, z_acoustic], dim=-1)
                reconstructed_mel, reconstructed_padding_mask = vae.sample(
                    num_steps=8,
                    temperature=0.2,
                    guidance_scale=1.3,
                    z=z,
                    padding_mask=padding_mask,
                    speaker_embedding=speaker_embedding,
                )
            else:
                logger.info(
                    "Decoding VQ discrete tokens plus continuous features using VAE..."
                )
                vq_emb = discrete_tokens_to_semantic_latents(
                    vae,
                    tokens_tensor,
                    dtype=dtype,
                    device=device,
                )
                if args.decode_only_token or z_sample is None:
                    logger.info(
                        "Using only generated quantized tokens (continuous features zeroed out)."
                    )
                    z_sample = torch.zeros(
                        (1, len(audio_tokens), hybrid_model.config.continuous_dim),
                        dtype=dtype,
                        device=device,
                    )
                else:
                    z_sample = align_continuous_tokens(
                        z_sample,
                        length=len(audio_tokens),
                        continuous_dim=hybrid_model.config.continuous_dim,
                        dtype=dtype,
                        device=device,
                    )
                z = combine_semantic_and_acoustic_latents(vq_emb, z_sample, vae)
                reconstructed_mel, reconstructed_padding_mask = vae.sample(
                    num_steps=8,
                    temperature=0.2,
                    guidance_scale=1.3,
                    z=z,
                    padding_mask=padding_mask,
                    speaker_embedding=speaker_embedding,
                )

            logger.info("Synthesizing waveform using Vocoder...")
            mel = reconstructed_mel[0]
            mask = reconstructed_padding_mask[0]
            mel = mel[~mask].unsqueeze(0).permute(0, 2, 1).float().to(device)

            recon_audio = vocoder.decode(mel).squeeze()
            recon_audio = recon_audio / (recon_audio.abs().max() + 1e-8)
            if recon_audio.dim() == 1:
                recon_audio = recon_audio.unsqueeze(0)

            output_path = (
                f"{output_root}_{sample_index}{output_ext}"
                if multi_output
                else args.output
            )
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            torchaudio.save(output_path, recon_audio.cpu(), 24000)
            logger.info(
                f"SUCCESS: Audio generated for {input_labels[sample_index]} "
                f"({token_len} tokens) and saved to '{output_path}'!"
            )


if __name__ == "__main__":
    main()


# # inferenza di prova (testo d'esempio, output in outputs/tmp_test.wav)
# python inference.py \
#   -c checkpoints/tmp \
#   --text "I tripped over my own shoelaces and landed face-first in a pie. At least dessert was served" \
#   --voice_condition /Users/software/Research/MelCausalVAE/ablations/female.wav \
#   -o output.wav \
#  --num_steps 6 \
#  --diffusion_temperature 0.3 \
#  --guidance_scale 1.8

# python inference.py \
#   -c checkpoints/tmp \
#   --text "I stood frozen, my heart pounding in my chest, as I witnessed the horrifying moment my father was taken from us, desperate tears streaming down my face as I screamed for someone, anyone, to help." \
#   --voice_condition /Users/software/Research/MelCausalVAE/ablations/female.wav \
#   -o output.wav \
#  --num_steps 6 \
#  --diffusion_temperature 0.3 \
#  --guidance_scale 1.8

# batch inference
# python inference.py \
#   -c checkpoints/tmp \
#   --text "I tripped over my own shoelaces and landed face-first in a pie. At least dessert was served" \
#   --text "I stood frozen, my heart pounding in my chest, as I witnessed the horrifying moment my father was taken from us, desperate tears streaming down my face as I screamed for someone, anyone, to help." \
#   --voice_condition /Users/software/Research/MelCausalVAE/ablations/female.wav \
#   -o output.wav \
#  --num_steps 6 \
#  --diffusion_temperature 0.3 \
#  --guidance_scale 1.8
