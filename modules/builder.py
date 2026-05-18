from typing import Dict, Any
from .hybrid_model import HybridTTS
from .configs import (
    HybridTTSConfig,
    BackboneConfig,
    AdapterConfig,
    TokenHeadConfig,
    DiTConfig,
)


def build_model(cfg_dict: Dict[str, Any]) -> HybridTTS:
    """Builds a HybridTTS model from a configuration dictionary."""
    import os
    import json

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
    continuous_dim = cfg_dict.get("continuous_dim")

    if vae_checkpoint:
        config_dir = (
            vae_checkpoint
            if os.path.isdir(vae_checkpoint)
            else os.path.dirname(vae_checkpoint)
        )
        config_path = os.path.join(config_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                vae_config = json.load(f)

            encoder_config = vae_config.get("encoder_config")
            latent_dim = encoder_config.get("latent_dim")
            vq_config = encoder_config.get("vq_config")

            if vq_config is not None:
                dim_to_quantize = vq_config.get("dim_to_quantize")
                continuous_dim = latent_dim - dim_to_quantize
            else:
                continuous_dim = latent_dim
            print(
                f"Dynamically resolved continuous_dim from VAE config: {continuous_dim} (latent_dim={latent_dim})"
            )

    backbone_config = BackboneConfig(**backbone_cfg) if backbone_cfg is not None else BackboneConfig()
    diffusion_head_config = DiTConfig(**diffusion_head_cfg) if diffusion_head_cfg is not None else DiTConfig()
    
    continuous_adapter_config = None
    if adapter_cfg is not None:
        continuous_adapter_config = AdapterConfig(**adapter_cfg)
        
    token_head_config = TokenHeadConfig(**token_head_cfg) if token_head_cfg is not None else TokenHeadConfig()

    hybrid_config = HybridTTSConfig(
        backbone_config=backbone_config,
        diffusion_head_config=diffusion_head_config,
        continuous_adapter_config=continuous_adapter_config,
        token_head_config=token_head_config,
        prompt_vocab_size=cfg_dict.get("prompt_vocab_size"),
        discrete_token_vocab_size=cfg_dict.get("discrete_token_vocab_size"),
        continuous_dim=continuous_dim,
    )

    return HybridTTS(config=hybrid_config)
