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


@dataclass
class BackboneConfig:
    model_name_or_path: str = "Qwen/Qwen2-0.5B"
    pretrained: bool = False
    vocab_size: int = 256  # Phonemes/Characters
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class AdapterConfig:
    in_dim: int = 64
    hidden_dim: int = 256
    out_dim: int = 512  # Should match backbone hidden size
    num_layers: int = 2


@dataclass
class TokenHeadConfig:
    in_dim: int = 512
    vocab_size: int = 1024  # VAE discrete token vocab size


@dataclass(kw_only=True)
class HybridTTSConfig:
    backbone_config: BackboneConfig = field(default_factory=BackboneConfig)
    diffusion_head_config: DiTConfig = field(default_factory=DiTConfig)
    continuous_adapter_config: Optional[AdapterConfig] = field(
        default_factory=AdapterConfig
    )
    token_head_config: TokenHeadConfig = field(default_factory=TokenHeadConfig)

    # Dimensions for Adaptive Norm & Embedding
    prompt_vocab_size: int
    discrete_token_vocab_size: int
    continuous_dim: int
    start_audio_id: int
    end_audio_id: int
    pad_token_id: int
    prompt_offset: int
    debug: bool = False
    uncond_prob: float = 0.0
    no_augment_ratio: float = 0.0
    backbone_hidden_size: Optional[int] = None

    def __post_init__(self):
        # Sync VAE/tokenizer-derived dims (known before backbone load).
        self.backbone_config.vocab_size = self.prompt_vocab_size
        self.token_head_config.vocab_size = self.discrete_token_vocab_size

        if self.continuous_adapter_config is not None:
            self.continuous_adapter_config.in_dim = self.continuous_dim

        self.diffusion_head_config.audio_latent_dim = self.continuous_dim

    def apply_backbone_dims(self, hidden_size: int) -> None:
        """Propagate HF backbone hidden size into sub-configs after model load."""
        self.backbone_hidden_size = hidden_size
        self.token_head_config.in_dim = hidden_size

        if self.continuous_adapter_config is not None:
            self.continuous_adapter_config.out_dim = hidden_size

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
