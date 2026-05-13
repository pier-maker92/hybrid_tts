import math
import torch
from torch import nn
from torch.nn import Module
from functools import wraps
from einops import rearrange
from beartype import beartype
from typing import Union, Optional
from torch.cuda.amp import autocast
from torch.nn import functional as F


# --------- helpers ---------
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


# --------- attention ---------
class Attend(nn.Module):
    """
    Attention on [B, H, N, D].
    Sfrutta nativamente F.scaled_dot_product_attention per dispatchare su
    FlashAttention, Memory-Efficient o Math fallback.
    """

    def __init__(
        self,
        dropout: float = 0.0,
        scale: Optional[float] = None,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.scale = scale
        self.window_size = window_size

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        group_size: Optional[int] = None,
    ):
        """
        q, k, v: [B, H, N, D]
        mask: None or [B,S] bool tensor (True = valid token)
        """
        B, H, Nq, D = q.shape
        _, _, Nk, _ = k.shape

        p = self.dropout if self.training else 0.0

        attn_mask = None
        is_causal = causal

        needs_custom_mask = (
            (self.window_size is not None)
            or (group_size is not None)
            or (mask is not None and causal)
        )

        if needs_custom_mask:
            is_causal = False

            idx_q = torch.arange(Nq, device=q.device)
            idx_k = torch.arange(Nk, device=k.device)

            custom_mask = torch.ones((Nq, Nk), device=q.device, dtype=torch.bool)

            if causal:
                custom_mask &= idx_k[None, :] <= idx_q[:, None]

            if group_size is not None:
                custom_mask &= (idx_k[None, :] // group_size) <= (
                    idx_q[:, None] // group_size
                )

            if self.window_size is not None:
                if causal:
                    custom_mask &= idx_k[None, :] >= (idx_q[:, None] - self.window_size)
                else:
                    custom_mask &= (
                        idx_k[None, :] - idx_q[:, None]
                    ).abs() <= self.window_size

            attn_mask = custom_mask.view(1, 1, Nq, Nk).expand(B, 1, Nq, Nk).clone()

            if mask is not None:
                if mask.ndim != 2:
                    mask = mask.view(B, Nk)
                pad_mask = mask.view(B, 1, 1, Nk)
                attn_mask = attn_mask & pad_mask

        elif mask is not None:
            if mask.ndim != 2:
                mask = mask.view(B, Nk)
            attn_mask = mask.view(B, 1, 1, Nk)

        return F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=p,
            is_causal=is_causal,
            scale=self.scale,
        )


class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        dropout=0,
        qk_norm=False,
        qk_norm_scale=10,
        window_size=None,
    ):
        super().__init__()
        self.heads = heads
        dim_inner = dim_head * heads
        scale = qk_norm_scale if qk_norm else None

        self.attend = Attend(dropout, scale=scale, window_size=window_size)
        self.qk_norm = qk_norm

        if qk_norm:
            raise NotImplementedError("qk_norm path not enabled")

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(
        self,
        x,
        mask=None,
        rotary_emb=None,
        causal: bool = False,
        group_size: Optional[int] = None,
    ):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        out = self.attend(q, k, v, mask=mask, causal=causal, group_size=group_size)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# --------- embeddings ---------


class LearnedSinusoidalPosEmb(Module):
    """used by @crowsonkb"""

    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return fouriered


class RotaryEmbedding(Module):
    def __init__(self, dim, theta=50000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @autocast(enabled=False)
    @beartype
    def forward(self, t: Union[int, torch.Tensor]):
        if not torch.is_tensor(t):
            t = torch.arange(t, device=self.device)
        t = t.type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@autocast(enabled=False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()


# --------- convolutional positional generating module ---------


class ConvPositionEmbed(Module):
    def __init__(self, dim, *, kernel_size, groups=None):
        super().__init__()
        assert is_odd(kernel_size)
        groups = default(groups, dim)  # full depthwise conv by default
        self.dw_conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.GELU(),
        )

    def forward(self, x, mask=None):
        if exists(mask):
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        x = rearrange(x, "b n c -> b c n")
        x = self.dw_conv1d(x)
        out = rearrange(x, "b c n -> b n c")
        if exists(mask):
            out = out.masked_fill(~mask, 0.0)
        return out


# --------- norms ---------


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class AdaptiveRMSNorm(Module):
    def __init__(self, dim, cond_dim=None):
        super().__init__()
        cond_dim = default(cond_dim, dim)
        self.scale = dim**0.5
        self.to_gamma = nn.Linear(cond_dim, dim)
        self.to_beta = nn.Linear(cond_dim, dim)
        # init to identity
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, cond):
        normed = F.normalize(x, dim=-1) * self.scale
        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        gamma, beta = map(lambda t: rearrange(t, "b d -> b 1 d"), (gamma, beta))
        return normed * gamma + beta


# --------- feedforward ---------


class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, dropout=0.0):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim),
    )
