from functools import partial
from typing import Optional

import torch
from einops import pack, rearrange, reduce, repeat, unpack

from .utils import *


class Transformer(Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        out_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        num_register_tokens: int = 0,
        attn_flash: bool = True,
        adaptive_rmsnorm: bool = True,
        time_hidden_dim: Optional[int] = None,
        use_unet_skip_connection: bool = False,
        skip_connect_scale: Optional[float] = None,
        attn_qk_norm: bool = False,
        use_conv_layer: bool = False,
        is_causal: bool = True,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        assert divisible_by(depth, 2)
        self.layers = nn.ModuleList([])

        self.use_conv_layer = use_conv_layer
        self.window_size = window_size
        self.rotary_emb = RotaryEmbedding(dim=dim_head)

        self.num_register_tokens = num_register_tokens
        self.has_register_tokens = num_register_tokens > 0

        if self.has_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        # time embedding
        time_hidden_dim = default(time_hidden_dim, dim * 4)
        if adaptive_rmsnorm:
            rmsnorm_klass = partial(
                AdaptiveRMSNorm,
                cond_dim=time_hidden_dim,
            )
        else:
            rmsnorm_klass = RMSNorm

        self.sinu_pos_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim),
            nn.Linear(dim, time_hidden_dim),
            nn.SiLU(),
        )

        if use_conv_layer:
            self.conv_embed = ConvPositionEmbed(
                dim=dim,
                kernel_size=31,  # 31 from voicebox
                groups=None,
            )

        self.skip_connect_scale = default(skip_connect_scale, 2**-0.5)

        for ind in range(depth):
            layer = ind + 1
            has_skip = use_unet_skip_connection and layer > (depth // 2)

            self.layers.append(
                nn.ModuleList(
                    [
                        nn.Linear(dim * 2, dim) if has_skip else None,
                        rmsnorm_klass(dim=dim),
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            qk_norm=attn_qk_norm,
                            window_size=window_size,
                        ),
                        rmsnorm_klass(dim=dim),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.final_norm = RMSNorm(dim)
        self.out_linear = nn.Linear(dim, out_dim, bias=False)
        self.is_causal = is_causal  # honor ctor arg

    def _get_causal_mask(self, attention_mask: torch.BoolTensor):
        seq_len = attention_mask.shape[1]
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=attention_mask.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        ).logical_not()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        return attention_mask.unsqueeze(1).unsqueeze(2) & causal_mask

    def _get_custom_causal_mask(
        self, attention_mask: torch.BoolTensor, special_mask: torch.BoolTensor
    ):
        batch, seq_len = attention_mask.shape
        device = attention_mask.device
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        ).logical_not()
        identity = torch.eye(seq_len, device=device, dtype=torch.bool)
        special_mask_row = special_mask.unsqueeze(1).unsqueeze(-1)
        special_mask_col = special_mask.unsqueeze(1).unsqueeze(1)
        interest_mask = (
            identity.unsqueeze(0) & special_mask_row & special_mask_col
        ) | (~special_mask_col)
        final_mask = (
            causal_mask.unsqueeze(0).unsqueeze(0)
            & attention_mask.unsqueeze(1).unsqueeze(2)
            & interest_mask
        )
        return final_mask

    def forward(
        self,
        x: torch.FloatTensor,
        times: torch.FloatTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
        group_size: Optional[int] = None,
    ):
        batch, seq_len, *_ = x.shape
        t = times

        # conv layer
        if hasattr(self, "conv_embed"):
            x = self.conv_embed(x, mask=attention_mask) + x

        # Ensure a 2D boolean mask for FlashAttention varlen when needed
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch, seq_len), device=x.device, dtype=torch.bool
            )

        # time embedding
        time_emb = self.sinu_pos_emb(t)

        # add register tokens to the left
        if self.has_register_tokens:
            register_tokens = repeat(self.register_tokens, "n d -> b n d", b=batch)
            x, ps = pack([register_tokens, x], "b * d")
            if exists(attention_mask):
                attention_mask = F.pad(
                    attention_mask, (self.num_register_tokens, 0), value=True
                )

        # keep track of skip connections
        skip_connects = []
        positions = seq_len

        if self.has_register_tokens:
            main_positions = torch.arange(seq_len, device=self.device, dtype=torch.long)
            register_positions = torch.full(
                (self.num_register_tokens,),
                -10000,
                device=self.device,
                dtype=torch.long,
            )
            positions = torch.cat((register_positions, main_positions))

        rotary_emb = self.rotary_emb(positions)

        # adaptive rmsnorm
        rmsnorm_kwargs = dict(cond=time_emb)

        # layers
        for (
            skip_combiner,
            attn_prenorm,
            attn,
            ff_prenorm,
            ff,
        ) in self.layers:
            if not exists(skip_combiner):
                skip_connects.append(x)
            else:
                skip_connect = skip_connects.pop() * self.skip_connect_scale
                x = torch.cat((x, skip_connect), dim=-1)
                x = skip_combiner(x)

            attn_input = attn_prenorm(x, **rmsnorm_kwargs)
            x = (
                attn(
                    attn_input,
                    mask=attention_mask,
                    rotary_emb=rotary_emb,
                    causal=self.is_causal,
                    group_size=group_size,
                )
                + x
            )

            ff_input = ff_prenorm(x, **rmsnorm_kwargs)
            x = ff(ff_input) + x

        # remove the register tokens
        if self.has_register_tokens:
            _, x = unpack(x, ps, "b * d")

        x = self.final_norm(x)
        return self.out_linear(x)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
