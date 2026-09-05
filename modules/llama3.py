# Adapted from:
# https://github.com/lucadellalib/audiocodecs
# https://github.com/meta-llama/llama3

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["LlamaBackbone", "LlamaKVCache"]

class LlamaKVCache:
    def __init__(self, curr_pos: int, kv_caches: List[Tensor]):
        self.curr_pos = curr_pos
        self.kv_caches = kv_caches

    def batch_select_indices(self, indices: Tensor):
        new_caches = []
        for cache in self.kv_caches:
            if cache is not None:
                new_caches.append(cache.index_select(0, indices))
            else:
                new_caches.append(None)
        self.kv_caches = new_caches


class RMSNorm(nn.Module):
    def __init__(self, dim: int = 512, norm_eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, input: Tensor) -> Tensor:
        input_dtype = input.dtype
        output = input.float()
        output = output * ((output**2).mean(-1, keepdim=True) + self.norm_eps).rsqrt()
        output = output.to(dtype=input_dtype)
        return self.weight * output


class FeedForward(nn.Module):
    def __init__(self, dim: int = 512, ffn_dim: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: Tensor):
        gate = self.act(self.w1(input))
        return self.dropout(self.w2(gate * self.w3(input)))


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int = 512, n_heads: int = 4, n_kv_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dropout = dropout
        self.head_dim = dim // n_heads
        self.n_kv_head_reps = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, input: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None, curr_pos: int = 0, kv_cache: Optional[Tensor] = None):
        B, T, _ = input.shape

        if kv_cache is None:
            assert curr_pos == 0
            kv_cache = torch.zeros(
                B, 2 * T, self.n_kv_heads, self.head_dim, 2,
                device=input.device, dtype=input.dtype
            )
        elif curr_pos + T > kv_cache.shape[1]:
            new_size = 2 * (curr_pos + T)
            kv_cache = F.pad(kv_cache, [0, 0, 0, 0, 0, 0, 0, new_size - kv_cache.shape[1]])

        qs = self.wq(input).view(B, T, self.n_heads, -1)
        ks = self.wk(input).view(B, T, self.n_kv_heads, -1)
        vs = self.wv(input).view(B, T, self.n_kv_heads, -1)

        qs, ks = self._apply_rotary_emb(qs, ks, freqs_cis)

        kv_cache[:, curr_pos : curr_pos + T, :, :, 0] = ks
        kv_cache[:, curr_pos : curr_pos + T, :, :, 1] = vs

        ks = kv_cache[:, : curr_pos + T, :, :, 0]
        vs = kv_cache[:, : curr_pos + T, :, :, 1]

        ks = torch.repeat_interleave(ks, dim=-2, repeats=self.n_kv_head_reps)
        vs = torch.repeat_interleave(vs, dim=-2, repeats=self.n_kv_head_reps)

        qs = qs.transpose(-3, -2)
        ks = ks.transpose(-3, -2)
        vs = vs.transpose(-3, -2)

        # F.scaled_dot_product_attention takes mask of shape (B, 1, T, S) if broadcasting, or (B, H, T, S)
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1) # (B, 1, T, S)

        output = F.scaled_dot_product_attention(
            qs, ks, vs, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )

        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.wo(output)

        return output, curr_pos + T, kv_cache

    def _apply_rotary_emb(self, xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(xq.shape[:-1] + (-1, 2)))
        xk_ = torch.view_as_complex(xk.float().reshape(xk.shape[:-1] + (-1, 2)))

        shape = [1] * len(xq_.shape)
        shape[1] = xq_.shape[1]
        shape[-1] = xq_.shape[-1]
        freqs_cis = freqs_cis.view(shape)

        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(start_dim=3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(start_dim=3)
        return xq_out.type_as(xq), xk_out.type_as(xk)


class LlamaLayer(nn.Module):
    def __init__(self, dim: int = 512, ffn_dim: int = 2048, n_heads: int = 4, n_kv_heads: int = 1, dropout: float = 0.0, norm_eps: float = 1e-6):
        super().__init__()
        self.attention = GroupedQueryAttention(dim, n_heads, n_kv_heads, dropout)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.feed_forward = FeedForward(dim, ffn_dim, dropout)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(self, input: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None, curr_pos: int = 0, kv_cache: Optional[Tensor] = None):
        hidden, curr_pos, kv_cache = self.attention(self.attention_norm(input), freqs_cis, mask, curr_pos, kv_cache=kv_cache)
        hidden = hidden + input
        output = self.feed_forward(self.ffn_norm(hidden))
        output = output + hidden
        return output, curr_pos, kv_cache


class LlamaBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        num_layers: int = 12,
        dim: int = 512,
        ffn_dim: int = 2048,
        n_heads: int = 4,
        n_kv_heads: int = 1,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        max_seq_len: int = 4096,
        tie_word_embeddings: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings

        self.embed_tokens = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)

        self.layers = nn.ModuleList([
            LlamaLayer(dim, ffn_dim, n_heads, n_kv_heads, dropout, norm_eps)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim, norm_eps)

        self.register_buffer(
            "freqs_cis",
            self._precompute_freqs_cis(dim // n_heads, rope_theta, max_seq_len * 2),
            persistent=False
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _precompute_freqs_cis(self, dim: int, rope_theta: float, max_seq_len: int) -> Tensor:
        freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens if self.tie_word_embeddings else nn.Identity()

    def _get_causal_mask(self, seq_len: int, curr_pos: int, device: torch.device, dtype: torch.dtype, attention_mask: Optional[Tensor] = None):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        if curr_pos > 0:
            mask = torch.hstack([torch.zeros((seq_len, curr_pos), device=device), mask])
        mask = mask.to(dtype)

        if attention_mask is not None:
            # mask shape is (T, curr_pos + T)
            # attention_mask shape is (B, curr_pos + T)
            mask = mask.unsqueeze(0).expand(attention_mask.shape[0], -1, -1).clone()
            bool_mask = ~attention_mask.unsqueeze(1) # (B, 1, curr_pos + T)
            mask = mask.masked_fill(bool_mask, float("-inf"))
            return mask
        else:
            if seq_len == 1 and curr_pos > 0:
                return None
            return mask

    def forward(self, inputs_embeds: Tensor, attention_mask: Optional[Tensor] = None, **kwargs):
        T = inputs_embeds.shape[1]
        device = inputs_embeds.device

        curr_pos = 0
        kv_caches = [None] * self.num_layers
        mask = self._get_causal_mask(T, curr_pos, device, inputs_embeds.dtype, attention_mask)

        output = inputs_embeds
        freqs_cis = self.freqs_cis[curr_pos : curr_pos + T].to(device)

        for i, layer in enumerate(self.layers):
            output, _, kv_caches[i] = layer(output, freqs_cis, mask, curr_pos, kv_caches[i])

        output = self.norm(output)
        return output

    def inference_forward(self, inputs_embeds: Tensor, attention_mask: Optional[Tensor] = None, **kwargs):
        past_key_values = kwargs.get("past_key_values", None)

        T = inputs_embeds.shape[1]
        device = inputs_embeds.device

        if past_key_values is not None and isinstance(past_key_values, LlamaKVCache):
            curr_pos = past_key_values.curr_pos
            kv_caches = past_key_values.kv_caches
        else:
            curr_pos = 0
            kv_caches = [None] * self.num_layers

        mask = self._get_causal_mask(T, curr_pos, device, inputs_embeds.dtype, attention_mask)

        output = inputs_embeds
        freqs_cis = self.freqs_cis[curr_pos : curr_pos + T].to(device)

        next_pos = curr_pos
        for i, layer in enumerate(self.layers):
            output, next_pos, kv_caches[i] = layer(output, freqs_cis, mask, curr_pos, kv_caches[i])

        output = self.norm(output)

        hidden_states = output[:, -1:, :]
        new_past_key_values = LlamaKVCache(next_pos, kv_caches)

        return hidden_states, new_past_key_values
