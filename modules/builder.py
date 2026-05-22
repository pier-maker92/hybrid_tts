import os
import json
from typing import Dict, Any
from .hybrid_model import HybridTTS
from .configs import (
    HybridTTSConfig,
    BackboneConfig,
    AdapterConfig,
    TokenHeadConfig,
    DiTConfig,
)


def load_codebook_config(vae_checkpoint: str):
    config_path = os.path.join(vae_checkpoint, "config.json")
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    continuous_dim = cfg_dict.get("latent_dim")
    vq_config = cfg_dict["encoder_config"].get("vq_config", None)
    vq_vocab_size = 0
    if vq_config is not None:
        dim_to_quantize = vq_config.get("dim_to_quantize")
        vq_vocab_size = vq_config.get("num_embeddings")
        if not vq_config.get("add_vq_residual_to_stoch"):
            continuous_dim = continuous_dim - dim_to_quantize
    return (
        continuous_dim,
        vq_vocab_size,
    )


def build_model(cfg_dict: Dict[str, Any]) -> HybridTTS:
    """Builds a HybridTTS model from a configuration dictionary."""

    backbone_cfg = cfg_dict.get("backbone_config")
    if backbone_cfg is None:
        backbone_cfg = cfg_dict.get("backbone")
    if backbone_cfg is not None:
        backbone_cfg = backbone_cfg.copy()

    diffusion_head_cfg = cfg_dict.get("diffusion_head_config")
    if diffusion_head_cfg is None:
        diffusion_head_cfg = cfg_dict.get("diffusion_head")
    if diffusion_head_cfg is not None:
        diffusion_head_cfg = diffusion_head_cfg.copy()

    adapter_cfg = cfg_dict.get("continuous_adapter_config")
    if adapter_cfg is None:
        adapter_cfg = cfg_dict.get("continuous_adapter")
    if adapter_cfg is not None:
        adapter_cfg = adapter_cfg.copy()

    token_head_cfg = cfg_dict.get("token_head_config")
    if token_head_cfg is None:
        token_head_cfg = cfg_dict.get("token_head")
    if token_head_cfg is not None:
        token_head_cfg = token_head_cfg.copy()

    # Dynamic resolution of continuous_dim from VAE config.json
    vae_checkpoint = cfg_dict.get("vae_checkpoint")
    if not vae_checkpoint:
        raise ValueError("VAE checkpoint is required to build model.")
    continuous_dim, discrete_token_vocab_size = load_codebook_config(vae_checkpoint)

    backbone_config = (
        BackboneConfig(**backbone_cfg) if backbone_cfg is not None else BackboneConfig()
    )
    diffusion_head_config = (
        DiTConfig(**diffusion_head_cfg)
        if diffusion_head_cfg is not None
        else DiTConfig()
    )

    continuous_adapter_config = None
    if adapter_cfg is not None:
        continuous_adapter_config = AdapterConfig(**adapter_cfg)

    token_head_config = (
        TokenHeadConfig(**token_head_cfg)
        if token_head_cfg is not None
        else TokenHeadConfig()
    )

    training_cfg = cfg_dict.get("training", {})
    
    hybrid_config = HybridTTSConfig(
        backbone_config=backbone_config,
        diffusion_head_config=diffusion_head_config,
        continuous_adapter_config=continuous_adapter_config,
        token_head_config=token_head_config,
        prompt_vocab_size=cfg_dict.get("prompt_vocab_size"),
        discrete_token_vocab_size=discrete_token_vocab_size,
        continuous_dim=continuous_dim,
        pad_token_id=cfg_dict.get("pad_token_id"),
        prompt_offset=cfg_dict.get("prompt_offset"),
        start_audio_id=cfg_dict.get("start_audio_id"),
        end_audio_id=cfg_dict.get("end_audio_id"),
        debug=cfg_dict.get("debug", False),
        uncond_prob=training_cfg.get("uncond_prob", 0.0),
        no_augment_ratio=training_cfg.get("no_augment_ratio", 0.0),
        discrete_only=training_cfg.get("discrete_only", False),
    )

    return HybridTTS(config=hybrid_config)
