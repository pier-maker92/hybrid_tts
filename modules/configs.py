from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class DiTConfig:
    sigma: float = 0.0
    audio_latent_dim: int = 128
    net_dim: int = 512
    net_heads: int = 8
    net_depth: int = 6
    backbone_dim: int = 512
    uncond_prob: float = 0.1
    is_causal: bool = False
    use_conv_layer: bool = False
    use_window_attention: bool = False
    window_attention_seconds: float = 0.0
    use_group_bidirectional: bool = False
    use_mlp_sampler: bool = False
    use_ecapa_film: bool = False
    ecapa_dim: int = 192


@dataclass(kw_only=True)
class BackboneConfig:
    model_type: str
    force_weight_tying: bool
    ffn_dim: Optional[int] = None
    dropout: Optional[float] = None
    num_heads: Optional[int] = None
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None
    from_pretrained: Optional[bool] = None
    max_position_embeddings: Optional[int] = None
    n_kv_heads: Optional[int] = None
    rope_theta: Optional[float] = None


@dataclass
class AdapterConfig:
    in_dim: int = 64
    hidden_dim: int = 256
    out_dim: int = 512  # Should match backbone hidden size
    num_layers: int = 2


@dataclass(kw_only=True)
class HybridTTSConfig:
    backbone_config: BackboneConfig = field(default_factory=BackboneConfig)
    diffusion_head_config: Optional[DiTConfig] = None
    continuous_adapter_config: Optional[AdapterConfig] = field(
        default_factory=AdapterConfig
    )

    # Dimensions for Adaptive Norm & Embedding
    pad_token_id: int
    end_audio_id: int
    start_audio_id: int
    continuous_dim: int
    prompt_vocab_size: int
    shift_audio_offset: int
    discrete_token_vocab_size: int
    debug: bool = False
    uncond_prob: float = 0.0
    no_augment_ratio: float = 0.0
    discrete_only: bool = False
    continuous_only: bool = False
    backbone_hidden_size: Optional[int] = None
    continuous_scaling_mode: Optional[str] = None  # 'fixed' or 'learnable' or None

    def apply_backbone_dims(self, hidden_size: int) -> None:
        """Propagate HF backbone hidden size into sub-configs after model load."""
        self.backbone_hidden_size = hidden_size

        if self.continuous_adapter_config is not None:
            self.continuous_adapter_config.out_dim = hidden_size

        if self.diffusion_head_config is not None:
            self.diffusion_head_config.backbone_dim = hidden_size

    @property
    def hidden_size(self) -> int:
        """Return hidden dimension for DeepSpeed compatibility."""
        if self.backbone_hidden_size is None:
            raise RuntimeError(
                "backbone_hidden_size is not set. Instantiate HybridTTS first."
            )
        return self.backbone_hidden_size

    def to_dict(self):
        """Convert config to dict for W&B logging compatibility"""
        d = asdict(self)
        d["model_type"] = "HybridTTS"
        return d
