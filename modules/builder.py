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
    backbone_cfg = cfg_dict.get("backbone_config", cfg_dict.get("backbone", {})).copy()
    diffusion_head_cfg = cfg_dict.get("diffusion_head_config", cfg_dict.get("diffusion_head", {})).copy()
    adapter_cfg = cfg_dict.get("continuous_adapter_config", cfg_dict.get("continuous_adapter", {})).copy()
    token_head_cfg = cfg_dict.get("token_head_config", cfg_dict.get("token_head", {})).copy()

    backbone_config = BackboneConfig(**backbone_cfg)
    diffusion_head_config = DiTConfig(**diffusion_head_cfg)
    
    continuous_adapter_config = None
    if adapter_cfg:
        continuous_adapter_config = AdapterConfig(**adapter_cfg)
        
    token_head_config = TokenHeadConfig(**token_head_cfg)

    hybrid_config = HybridTTSConfig(
        backbone_config=backbone_config,
        diffusion_head_config=diffusion_head_config,
        continuous_adapter_config=continuous_adapter_config,
        token_head_config=token_head_config,
        prompt_vocab_size=cfg_dict.get("prompt_vocab_size", 256),
        discrete_token_vocab_size=cfg_dict.get("discrete_token_vocab_size", 1024),
        continuous_dim=cfg_dict.get("continuous_dim", 64),
    )

    return HybridTTS(config=hybrid_config)
