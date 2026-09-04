from dataclasses import dataclass
import torch
from typing import Optional


@dataclass
class DecoderOutput:
    loss: Optional[torch.Tensor] = None
    audio_features: Optional[torch.Tensor] = None
    padding_mask: Optional[torch.Tensor] = None


@dataclass
class HybridTTSOutput:
    token_logits: Optional[torch.Tensor]
    diffusion_loss: Optional[torch.Tensor]
    diffusion_output: Optional[DecoderOutput] = None
    continuous_ratio: Optional[torch.Tensor] = None
    norm_ratio: Optional[torch.Tensor] = None
