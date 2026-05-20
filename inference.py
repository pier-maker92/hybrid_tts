#!/usr/bin/env python
import os
import sys
import json
import torch
import torchaudio
import argparse
import logging
from typing import List, Dict, Any, Optional
from omegaconf import OmegaConf

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


def load_hybrid_config(
    config_dir: str = "configs", setting_name: str = "setting-1"
) -> Dict[str, Any]:
    """Loads and merges the default and setting-specific yaml configurations."""
    main_path = os.path.join(config_dir, "main.yaml")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"Main config not found at {main_path}")

    main_cfg = OmegaConf.load(main_path)

    # Load default components manually to simulate Hydra composition in standard python
    train_defaults = OmegaConf.load(os.path.join(config_dir, "defaults", "train.yaml"))
    backbone_defaults = OmegaConf.load(
        os.path.join(config_dir, "defaults", "backbone.yaml")
    )
    diff_defaults = OmegaConf.load(
        os.path.join(config_dir, "defaults", "diffusion_head.yaml")
    )

    merged_cfg = OmegaConf.create()
    merged_cfg = OmegaConf.merge(merged_cfg, train_defaults)
    merged_cfg = OmegaConf.merge(merged_cfg, backbone_defaults)
    merged_cfg = OmegaConf.merge(merged_cfg, diff_defaults)
    merged_cfg = OmegaConf.merge(merged_cfg, main_cfg)

    # Load setting specific experiment config if it exists
    if setting_name:
        setting_path = os.path.join(
            config_dir, "settings", "exps", f"{setting_name}.yaml"
        )
        if os.path.exists(setting_path):
            logger.info(f"Merging experiment settings from {setting_path}")
            setting_cfg = OmegaConf.load(setting_path)
            merged_cfg = OmegaConf.merge(merged_cfg, setting_cfg)
        else:
            logger.warning(
                f"Experiment settings file {setting_path} not found. Using defaults."
            )

    return OmegaConf.to_container(merged_cfg, resolve=True)


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
) -> torch.nn.Module:
    """Builds HybridTTS and loads its weights from safetensors or pt file."""
    # Read vocab size to dynamically set config parameters if needed
    vocab_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "phoneme_vocab.json"
    )
    if os.path.exists(vocab_path):
        with open(vocab_path, "r") as f:
            phoneme_vocab = json.load(f)
        vocab_size = len(phoneme_vocab)
    else:
        vocab_size = 256

    backbone_cfg = cfg_dict.get("backbone_config")
    if backbone_cfg is None:
        backbone_cfg = cfg_dict.get("backbone")
    is_pretrained = backbone_cfg.get("pretrained")

    if not is_pretrained:
        cfg_dict["prompt_vocab_size"] = vocab_size + 3  # Phonemes + Special Tokens
        cfg_dict["prompt_offset"] = 0
        cfg_dict["pad_token_id"] = vocab_size
        cfg_dict["start_audio_id"] = vocab_size + 1
        cfg_dict["end_audio_id"] = vocab_size + 2
        logger.info(
            f"Configured scratch HybridTTS: prompt_vocab_size={cfg_dict['prompt_vocab_size']}, start_audio_id={cfg_dict['start_audio_id']}"
        )

    model = build_hybrid_model(cfg_dict)

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


def sample_next_token(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95
) -> int:
    """Samples next token from logits with temperature, top_k, and nucleus (top_p) filtering."""
    if temperature == 0.0:
        return torch.argmax(logits).item()

    logits = logits / temperature

    # Top-K filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    # Top-P (Nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token


@torch.no_grad()
def generate_discrete_tokens(
    model: torch.nn.Module,
    prompt_ids: List[int],
    max_len: int = 500,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
) -> List[int]:
    """Generates discrete tokens autoregressively from phoneme prompt_ids."""
    model.eval()
    device = model.device

    # Convert list of prompt ids to tensor of shape [1, L_prompt]
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # We will generate discrete tokens autoregressively.
    # Start with a dummy token at index 0.
    generated_tokens = [0]

    logger.info("Starting autoregressive generation of discrete audio tokens...")

    for step in range(max_len):
        t = len(generated_tokens)
        discrete_input = torch.tensor(
            [generated_tokens], dtype=torch.long, device=device
        )

        # continuous input: shape [1, t, continuous_dim] filled with zeros
        continuous_input = torch.zeros(
            (1, t, model.config.continuous_dim), dtype=model.dtype, device=device
        )

        # Run forward pass of HybridTTS
        outputs = model(
            prompt_ids=prompt_tensor,
            discrete_tokens=discrete_input,
            continuous_tokens=continuous_input,
        )

        # logits shape: [1, t, discrete_token_vocab_size]
        # We need the last logit (corresponding to the dummy token) to predict the next token
        logits = outputs.token_logits[0, -1, :]

        # Sample next token
        next_token = sample_next_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p
        )

        # Replace the dummy token at index t-1 with the actual sampled token
        generated_tokens[-1] = next_token

        # Otherwise, append a dummy token for the next step
        generated_tokens.append(0)

    # If we exited without hitting the end token, remove the trailing dummy token
    if generated_tokens[-1] == 0:
        generated_tokens = generated_tokens[:-1]

    logger.info(f"Generated {len(generated_tokens)} discrete tokens.")
    return generated_tokens


@torch.no_grad()
def get_audio_hidden_states(
    model: torch.nn.Module, prompt_ids: List[int], discrete_tokens: List[int]
) -> torch.Tensor:
    """Runs a forward-like backbone pass to extract the hidden states corresponding to audio tokens."""
    device = model.device
    model_dtype = model.dtype
    bb_dtype = next(model.backbone.parameters()).dtype

    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    discrete_tensor = torch.tensor([discrete_tokens], dtype=torch.long, device=device)

    L_prompt = prompt_tensor.shape[1]
    L_audio = discrete_tensor.shape[1]

    # continuous_tokens: zeros of shape [1, L_audio, continuous_dim]
    continuous_tokens = torch.zeros(
        (1, L_audio, model.config.continuous_dim), dtype=model_dtype, device=device
    )
    padding_mask = torch.zeros((1, L_audio), dtype=torch.bool, device=device)

    # Emulate the model's forward path to retrieve backbone hidden states
    p_emb = model.prompt_emb(prompt_tensor + model.config.prompt_offset)
    d_emb = model.discrete_emb(discrete_tensor)
    norm_c = model.continuous_norm(continuous_tokens, padding_mask=padding_mask)
    c_emb = model.continuous_adapter(norm_c)

    d_emb = model.norm_discrete(d_emb)
    c_emb = model.norm_continuous(c_emb)
    audio_emb = d_emb + c_emb

    start_ids = torch.full(
        (1, 1), model.config.start_audio_id, device=device, dtype=torch.long
    )
    start_emb = model.prompt_emb(start_ids)

    inputs_embeds = torch.cat([p_emb, start_emb, audio_emb], dim=1)

    prompt_attn = torch.ones((1, L_prompt), dtype=torch.long, device=device)
    start_attn = torch.ones((1, 1), dtype=torch.long, device=device)
    audio_attn = (~padding_mask).long()
    attention_mask = torch.cat([prompt_attn, start_attn, audio_attn], dim=1)

    outputs_bb = model.backbone(
        inputs_embeds=inputs_embeds.to(bb_dtype), attention_mask=attention_mask
    )
    full_hidden_states = outputs_bb.last_hidden_state

    audio_hidden_states = full_hidden_states[:, L_prompt : L_prompt + L_audio, :]
    return audio_hidden_states


@torch.no_grad()
def get_last_hidden_joint(
    model: torch.nn.Module,
    prompt_tensor: torch.Tensor,
    committed_tokens: List[int],
    committed_continuous: List[torch.Tensor],
) -> torch.Tensor:
    """
    Returns the hidden state of the last token in the sequence:
      [Prompt | START]                          <- step 0 (START output)
      [Prompt | START | (a_0,c_0) | ...]        <- step t (last audio token output)

    Exactly mirrors the training sequence with teacher-forcing, but using
    committed (generated) tokens instead of ground-truth tokens.
    The hidden state at position -1 is always used to predict the NEXT (a_t, c_t).
    """
    device = model.device
    bb_dtype = next(model.backbone.parameters()).dtype
    L_prompt = prompt_tensor.shape[1]

    p_emb = model.prompt_emb(prompt_tensor + model.config.prompt_offset)
    start_ids = torch.full((1, 1), model.config.start_audio_id, device=device, dtype=torch.long)
    start_emb = model.prompt_emb(start_ids)

    prompt_attn = torch.ones((1, L_prompt), dtype=torch.long, device=device)
    start_attn  = torch.ones((1, 1),        dtype=torch.long, device=device)
    attention_mask = torch.cat([prompt_attn, start_attn], dim=1)

    if len(committed_tokens) == 0:
        # Step 0: sequence is just [Prompt | START]
        inputs_embeds = torch.cat([p_emb, start_emb], dim=1)
    else:
        # Step t: sequence is [Prompt | START | (a_0,c_0) | ... | (a_{t-1},c_{t-1})]
        discrete_tensor    = torch.tensor([committed_tokens], dtype=torch.long, device=device)
        # committed_continuous is a list of [dim] tensors, already in normalized space
        continuous_tensor  = torch.stack(committed_continuous, dim=0).unsqueeze(0)  # [1, t, dim]

        d_emb = model.discrete_emb(discrete_tensor)
        # continuous from diffusion head is already normalized — apply adapter directly
        c_emb = model.continuous_adapter(continuous_tensor)

        d_emb = model.norm_discrete(d_emb)
        c_emb = model.norm_continuous(c_emb)
        audio_emb = d_emb + c_emb

        inputs_embeds  = torch.cat([p_emb, start_emb, audio_emb], dim=1)
        audio_attn     = torch.ones((1, len(committed_tokens)), dtype=torch.long, device=device)
        attention_mask = torch.cat([attention_mask, audio_attn], dim=1)

    outputs_bb       = model.backbone(inputs_embeds=inputs_embeds.to(bb_dtype), attention_mask=attention_mask)
    # The last position in the sequence is always the token whose output we need
    last_hidden      = outputs_bb.last_hidden_state[:, -1:, :]   # [1, 1, hidden_dim]
    return last_hidden


@torch.no_grad()
def generate_joint_tokens(
    model: torch.nn.Module,
    prompt_ids: List[int],
    max_len: int = 500,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    num_steps: int = 16,
    diffusion_temperature: float = 1.0,
    guidance_scale: float = 1.0,
) -> tuple[List[int], torch.Tensor]:
    """Generates discrete tokens and continuous latents jointly step-by-step.

    At each step t:
      1. Backbone sees [Prompt | START | committed_audio_0..t-1].
      2. The hidden state of the LAST token is used as context.
         - At t=0 this is START's output (identical to training).
         - At t>0 this is the last committed audio token's output.
      3. Diffusion head generates c_t from that context.
      4. Token head samples a_t from that context.
      5. (a_t, c_t) are appended to the committed lists.
    No dummy tokens are ever fed into the backbone.
    """
    model.eval()
    device      = model.device
    model_dtype = model.dtype

    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    committed_tokens:     List[int]           = []   # discrete tokens committed so far
    committed_continuous: List[torch.Tensor]  = []   # [dim] tensors, normalized, committed so far

    logger.info("Starting joint (autoregressive + diffusion) generation of audio...")

    step_padding_mask = torch.zeros((1, 1), dtype=torch.bool, device=device)

    for step in range(max_len):
        # 1. Get the hidden state of the last token in the current sequence
        last_hidden = get_last_hidden_joint(
            model, prompt_tensor, committed_tokens, committed_continuous
        )

        # 2. Generate continuous latent c_t (in normalized space, matching training targets)
        latents_output = model.diffusion_head.generate(
            num_steps=num_steps,
            context_vector=last_hidden,
            temperature=diffusion_temperature,
            guidance_scale=guidance_scale,
            padding_mask=step_padding_mask,
        )
        c_t = latents_output.audio_features[0, 0, :]   # [dim]

        # 3. Sample discrete token a_t
        logits     = model.token_head(last_hidden)[0, 0, :]   # [vocab_size]
        next_token = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)

        # 4. Commit both
        committed_tokens.append(next_token)
        committed_continuous.append(c_t.to(model_dtype))

    logger.info(f"Jointly generated {len(committed_tokens)} tokens.")

    # Stack continuous latents → [1, L, dim]
    z = torch.stack(committed_continuous, dim=0).unsqueeze(0)
    return committed_tokens, z


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
        help="Path to the HybridTTS checkpoint directory or weights file (model.safetensors / model.pt)",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        required=True,
        help="Path to the MelCausalVAE checkpoint directory",
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
        "--setting",
        type=str,
        default="2",
        help="Experiment setting config name under configs/settings/exps/ (default: setting-1)",
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

    # Load configuration
    logger.info(f"Loading HybridTTS configuration for '{args.setting}'...")
    cfg_dict = load_hybrid_config(setting_name=args.setting)

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

    # Load models
    logger.info("Loading models...")
    cfg_dict["vae_checkpoint"] = args.vae_checkpoint
    hybrid_model = load_hybrid_model(cfg_dict, args.hybrid_checkpoint, device, dtype)
    vae = load_vae(args.vae_checkpoint, device, dtype)
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

    # Calculate target length based on prompt length if default max_len is used
    if args.max_len == 500:
        target_len = int(len(prompt_ids) * args.ratio)
        logger.info(f"Target generation length calculated from prompt length: {target_len} frames (ratio={args.ratio})")
    else:
        target_len = args.max_len
        logger.info(f"Using explicitly specified max_len: {target_len} frames")

    with torch.no_grad():
        if not args.token_first:
            # Main joint inference mechanism: discrete tokens and continuous latents generated step-by-step
            discrete_tokens, z = generate_joint_tokens(
                hybrid_model,
                prompt_ids,
                max_len=target_len,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_steps=args.num_steps,
                diffusion_temperature=args.diffusion_temperature,
                guidance_scale=args.guidance_scale,
            )

            # Step 4: Run VAE decoder to reconstruct Mel Spectrogram
            logger.info("Decoding continuous features using VAE...")
            # For a single sequence, padding mask is all False
            padding_mask = torch.zeros(z.shape[:2], dtype=torch.bool, device=device)
            # Denormalize continuous latents only before VAE decoder
            z_denorm = hybrid_model.continuous_norm.denormalize(z)

            # Look up VQ codebook embeddings for discrete tokens and concatenate to form full 64-dimensional latent
            audio_tokens = list(discrete_tokens)
            
            tokens_tensor = torch.tensor(audio_tokens, dtype=torch.long, device=device)
            vq_emb = vae.encoder.vq.codebook(tokens_tensor)  # [L, 32]
            vq_emb = vq_emb.unsqueeze(0)  # [1, L, 32]
            
            if args.decode_only_token:
                logger.info("Using only generated quantized tokens (continuous features zeroed out).")
                z_denorm = torch.zeros_like(z_denorm)

            z_vae = torch.cat([vq_emb, z_denorm], dim=-1)

            reconstructed_mel, reconstructed_padding_mask = vae.sample(
                num_steps=16,
                temperature=0.2,
                guidance_scale=1.4,
                z=z_vae,
                padding_mask=padding_mask,
            )
        else:
            # Ablation mode: generate all discrete tokens first, and then run diffusion once on the sequence
            logger.info(
                "Running in ablation mode (--token_first): generating all discrete tokens first..."
            )
            # Step 1: Generate discrete tokens autoregressively
            discrete_tokens = generate_discrete_tokens(
                hybrid_model,
                prompt_ids,
                max_len=target_len,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

            if not discrete_tokens:
                logger.error("Failed to generate discrete tokens.")
                sys.exit(1)

            if not args.decode_only_token:
                # Step 2: Get hidden states for the complete audio sequence from backbone
                logger.info("Extracting audio hidden states from backbone...")
                audio_hidden_states = get_audio_hidden_states(
                    hybrid_model, prompt_ids, discrete_tokens
                )

                # Step 3: Run the diffusion head to generate continuous features (latents)
                logger.info(
                    f"Running diffusion head (steps={args.num_steps}, guidance={args.guidance_scale})..."
                )
                padding_mask = torch.zeros(
                    (1, audio_hidden_states.shape[1]), dtype=torch.bool, device=device
                )
                latents_output = hybrid_model.diffusion_head.generate(
                    num_steps=args.num_steps,
                    context_vector=audio_hidden_states,
                    temperature=args.diffusion_temperature,
                    guidance_scale=args.guidance_scale,
                    padding_mask=padding_mask,
                )
                z = latents_output.audio_features

                # Step 4: Run VAE decoder to reconstruct Mel Spectrogram
                logger.info("Decoding continuous features using VAE...")
                # Denormalize continuous latents only before VAE decoder
                z_denorm = hybrid_model.continuous_norm.denormalize(z)
                vae_padding_mask = latents_output.padding_mask
            else:
                logger.info("Skipping backbone extraction and diffusion head (decode_only_token is active).")
                # Create a zero tensor for z_denorm with shape [1, L_audio, continuous_dim]
                z_denorm = torch.zeros(
                    (1, len(discrete_tokens), hybrid_model.config.continuous_dim),
                    dtype=dtype,
                    device=device,
                )
                vae_padding_mask = torch.zeros(
                    (1, len(discrete_tokens)), dtype=torch.bool, device=device
                )

            # Look up VQ codebook embeddings for discrete tokens and concatenate to form full 64-dimensional latent
            audio_tokens = list(discrete_tokens)
            
            tokens_tensor = torch.tensor(audio_tokens, dtype=torch.long, device=device)
            vq_emb = vae.encoder.vq.codebook(tokens_tensor)  # [L, 32]
            vq_emb = vq_emb.unsqueeze(0)  # [1, L, 32]
            
            z_vae = torch.cat([vq_emb, z_denorm], dim=-1)

            reconstructed_mel, reconstructed_padding_mask = vae.sample(
                num_steps=args.num_steps,
                temperature=args.diffusion_temperature,
                guidance_scale=args.guidance_scale,
                z=z_vae,
                padding_mask=vae_padding_mask,
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
