import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .diffusion_head.cfm import DiT
from .configs import HybridTTSConfig
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoConfig
from .output_dataclasses import HybridTTSOutput, DecoderOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HybridTokenizer — decodes what goes into the transformer
# ---------------------------------------------------------------------------


class HybridTokenizer:
    """
    Translates raw prompt IDs (as stored in the batch) to human-readable
    symbols, so that the debug flag can show exactly what enters the backbone.

    Token layout in the prompt_emb LUT (from-scratch training):
        LUT index 0             → <PAD>  (pad_id, also the collator padding value)
        LUT index 1 .. N        → phonemes  (raw_id + prompt_offset → LUT index)
        LUT index N+1           → <start_audio>
        LUT index N+2           → <end_audio>

    For pretrained backbones the same structure applies but IDs are those of
    the backbone's tokenizer vocabulary.
    """

    def __init__(
        self,
        phoneme_vocab: dict,  # symbol → raw phoneme id (0-indexed from dataset)
        start_audio_id: int,  # direct LUT index for <start_audio>
        end_audio_id: int,  # direct LUT index for <end_audio>
        pad_id: int,
        prompt_offset: int = 0,  # raw phoneme ids shifted by this to get LUT index
    ):
        self.prompt_offset = prompt_offset
        self.start_audio_id = start_audio_id
        self.end_audio_id = end_audio_id
        self.pad_id = pad_id

        # Build LUT-index → symbol map
        self.id2sym: Dict[int, str] = {}
        self.id2sym[pad_id] = "<PAD>"
        self.id2sym[start_audio_id] = "<start_audio>"
        self.id2sym[end_audio_id] = "<end_audio>"
        for sym, raw_id in phoneme_vocab.items():
            lut_id = raw_id + prompt_offset
            self.id2sym[lut_id] = sym

    # raw_id here means the value sitting in the batch tensor (before offset)
    def _raw_to_sym(self, raw_id: int) -> str:
        lut_id = raw_id + self.prompt_offset
        return self.id2sym.get(lut_id, f"<unk:{lut_id}>")

    def decode_prompt(self, raw_ids) -> str:
        """Decode a 1-D sequence of raw prompt IDs."""
        return " ".join(
            "<PAD>" if rid == self.pad_id else self._raw_to_sym(rid)
            for rid in (raw_ids.tolist() if hasattr(raw_ids, "tolist") else raw_ids)
        )

    def decode_sample(
        self,
        prompt_ids: torch.Tensor,  # (L_prompt,)  raw ids
        valid_prompt_len: int,
        valid_audio_len: int,
        L_audio_max: int,
    ) -> str:
        prompt_str = self.decode_prompt(prompt_ids[:valid_prompt_len])
        pad_prompt = valid_prompt_len < len(prompt_ids)
        return (
            f"PROMPT_WITH_TAGS({valid_prompt_len}{'*' if pad_prompt else ''}): {prompt_str} "
            f"| AUDIO_INSERTED({valid_audio_len}/{L_audio_max} frames)"
        )


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------


class DynamicNormalizer(nn.Module):
    def __init__(self, dim, momentum=0.001, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        # Register buffers with shape (1, 1, dim) for easy broadcasting
        self.register_buffer("running_mean", torch.zeros(1, 1, dim))
        self.register_buffer("running_var", torch.ones(1, 1, dim))

    def forward(self, x, padding_mask=None):
        # x: (B, L, C)
        if self.training:
            if padding_mask is not None:
                mask = (~padding_mask).to(x.dtype).unsqueeze(-1)  # (B, L, 1)
                count = mask.sum()
                if count > 0:
                    # Sum over batch (0) and length (1) dimensions to get channel-wise values
                    batch_mean = (x * mask).sum(dim=(0, 1)) / count  # (C,)
                    batch_var = (((x - batch_mean.view(1, 1, -1)) ** 2) * mask).sum(
                        dim=(0, 1)
                    ) / count  # (C,)

                    batch_mean = batch_mean.view(1, 1, -1)  # (1, 1, C)
                    batch_var = batch_var.view(1, 1, -1)  # (1, 1, C)
                else:
                    batch_mean = x.mean(dim=(0, 1), keepdim=True)
                    batch_var = x.var(dim=(0, 1), keepdim=True, unbiased=False)
            else:
                batch_mean = x.mean(dim=(0, 1), keepdim=True)
                batch_var = x.var(dim=(0, 1), keepdim=True, unbiased=False)

            with torch.no_grad():
                self.running_mean.copy_(
                    (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                )
                self.running_var.copy_(
                    (1 - self.momentum) * self.running_var + self.momentum * batch_var
                )
            return (x - batch_mean) / (batch_var + self.eps).sqrt()
        else:
            mean = self.running_mean.to(x.dtype)
            var = self.running_var.to(x.dtype)
            return (x - mean) / (var + self.eps).sqrt()

    def denormalize(self, x):
        mean = self.running_mean.to(x.dtype)
        var = self.running_var.to(x.dtype)
        return x * (var + self.eps).sqrt() + mean


class MLPAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        curr_dim = config.in_dim
        for _ in range(config.num_layers - 1):
            layers.append(nn.Linear(curr_dim, config.hidden_dim))
            layers.append(nn.SiLU())
            curr_dim = config.hidden_dim
        layers.append(nn.Linear(curr_dim, config.out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# HybridTTS
# ---------------------------------------------------------------------------


class HybridTTS(nn.Module):
    """
    Sequence layout fed to the backbone (per sample i in the batch):

        [phoneme_0 ... phoneme_{P_i-1} | <start_audio> | audio_0 ... audio_{A_i-1} | <end_audio> | <PAD> ...]

    The dataloader prepares prompt_ids with <start_audio> and <end_audio>.
    _build_full_sequence finds the <start_audio> and <end_audio> positions, and inserts audio frames between them.
    """

    def __init__(self, config: HybridTTSConfig):
        super().__init__()
        self.config = config

        # ---- Backbone -------------------------------------------------------
        bb_cfg = config.backbone_config
        hf_config = AutoConfig.from_pretrained(bb_cfg.model_name_or_path)

        # Unified vocabulary: prompt tokens + discrete audio tokens + 1 for audio EOS
        self.unified_vocab_size = (
            config.prompt_vocab_size + config.discrete_token_vocab_size + 1
        )
        hf_config.vocab_size = self.unified_vocab_size
        hf_config.pad_token_id = config.pad_token_id

        self.backbone = AutoModelForCausalLM.from_config(hf_config)

        hidden_size = self.backbone.config.hidden_size
        self.config.apply_backbone_dims(hidden_size)

        # ---- Continuous adapter (VAE latents → hidden_size) ------------------
        if config.continuous_adapter_config is not None:
            self.continuous_adapter = MLPAdapter(config.continuous_adapter_config)
        else:
            self.continuous_adapter = nn.Linear(config.continuous_dim, hidden_size)

        # ---- Normalisations --------------------------------------------------
        self.dynamic_normalizer = DynamicNormalizer(config.continuous_dim)
        self.norm_continuous = nn.LayerNorm(hidden_size)

        # ---- Output heads ----------------------------------------------------
        self.diffusion_head = DiT(config.diffusion_head_config)

        # ---- Optional tokenizer for debug decoding --------------------------
        # Attach via model.tokenizer = HybridTokenizer(...) from train.py
        self.tokenizer: Optional[HybridTokenizer] = None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _extract_audio_tokens_span(self, discrete_sequence):
        """
        Extract audio start and end position from discrete sequence.
        """
        start_indices = (
            (discrete_sequence == self.config.start_audio_id).long().argmax(dim=1)
        )
        # Sequence ends with AUDIO_EOS
        audio_eos_id = (
            self.config.prompt_vocab_size + self.config.discrete_token_vocab_size
        )
        end_indices = (discrete_sequence == audio_eos_id).long().argmax(dim=1)
        return start_indices, end_indices

    def _add_continuous_token(
        self,
        continuous_sequence: torch.FloatTensor,
        audio_padding_mask: torch.BoolTensor,
        discrete_emb: torch.LongTensor,
        start_positions: torch.Tensor,
        attention_mask: torch.BoolTensor,
    ):
        """
        Insert audio embeddings between <start_audio> and <end_audio> in the prompt.
        """
        B, L_audio, hidden_dim = continuous_sequence.shape
        index_3d = (start_positions + 1).unsqueeze(-1)
        offset = torch.arange(L_audio, device=continuous_sequence.device).unsqueeze(0)
        audio_positions = index_3d + offset
        continuous_sequence = continuous_sequence * (~audio_padding_mask).unsqueeze(-1)

        audio_positions = (
            (torch.where((~audio_padding_mask), audio_positions, 0))
            .unsqueeze(-1)
            .repeat(1, 1, hidden_dim)
        )
        discrete_emb.scatter_add_(
            dim=1, index=audio_positions, src=continuous_sequence.to(discrete_emb.dtype)
        )

        return discrete_emb[:, : attention_mask.shape[1]]

    def _extract_audio_hidden_states(
        self,
        full_hidden_states: torch.Tensor,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
    ):
        """
        Select hidden states that predict audio frames 0 .. L_audio_max-1.
        These are the hidden states from <start_audio> to the LAST AUDIO FRAME BEFORE <end_audio>.
        """
        audio_hidden_states = []
        audio_masks = []
        pad_embed = self.backbone.get_input_embeddings()(
            torch.tensor(self.config.pad_token_id).long().to(full_hidden_states.device)
        )
        for h, s, e in zip(full_hidden_states, start_indices, end_indices):
            audio_hidden_states.append(h[s:e])
            audio_masks.append(torch.ones(e - s, dtype=torch.bool, device=h.device))

        audio_hidden_states = pad_sequence(
            audio_hidden_states,
            batch_first=True,
            padding_value=0,
        )
        audio_masks = pad_sequence(
            audio_masks,
            batch_first=True,
            padding_value=False,
        )
        audio_hidden_states[~audio_masks] = pad_embed
        return audio_hidden_states

    def noise_augment_continuous_token(
        self,
        continuous_tokens: torch.FloatTensor,
        padding_mask: torch.BoolTensor,
        target_std: float = 1.0,
    ):
        """Add Gaussian noise to continuous tokens."""
        std = (
            torch.rand(
                continuous_tokens.shape[0],
                continuous_tokens.shape[1],
                1,
                dtype=continuous_tokens.dtype,
                device=continuous_tokens.device,
            )
            * target_std
        )

        if getattr(self.config, "no_augment_ratio", 0.0) > 0.0:
            # Randomly select a portion of the batch to NOT be augmented (std=0)
            B = continuous_tokens.shape[0]
            keep_mask = (
                torch.rand(B, 1, 1, device=continuous_tokens.device)
                >= self.config.no_augment_ratio
            )
            std = std * keep_mask.to(std.dtype)

        noise = torch.randn_like(continuous_tokens) * std
        corrupted_continuous_tokens = continuous_tokens + noise
        corrupted_continuous_tokens = corrupted_continuous_tokens.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        return corrupted_continuous_tokens

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        discrete_sequence: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        continuous_sequence: torch.FloatTensor,
        audio_padding_mask: torch.BoolTensor,
        **kwargs,
    ) -> HybridTTSOutput:
        """
        Forward pass for training.

        Args:
            discrete_sequence: (B, L_discrete) prompt tokens
            attention_mask: (B, L_discrete) True = valid, False = pad
            continuous_sequence: (B, L_audio, C) continuous tokens
            audio_padding_mask: (B, L_audio) False = valid, True = pad
        """
        start_idx, end_idx = self._extract_audio_tokens_span(discrete_sequence)
        embed_layer = self.backbone.get_input_embeddings()
        input_embs = embed_layer(discrete_sequence)

        if continuous_sequence is not None:
            continuous_sequence = self.dynamic_normalizer(continuous_sequence)
            adapted_c_emb = self.norm_continuous(
                self.continuous_adapter(continuous_sequence)
            )
            adapted_c_emb = self.noise_augment_continuous_token(
                adapted_c_emb, audio_padding_mask
            )

            input_embs = self._add_continuous_token(
                continuous_sequence=adapted_c_emb,
                audio_padding_mask=audio_padding_mask,
                discrete_emb=input_embs,
                start_positions=start_idx,
                attention_mask=attention_mask,
            )

        last_hidden_state = self.backbone(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[
            -1
        ]  # last layer hidden states

        audio_hidden_states = self._extract_audio_hidden_states(
            last_hidden_state,
            start_idx,
            end_idx,
        )
        # token logits
        tied_weights = self.backbone.get_input_embeddings().weight[
            self.config.prompt_vocab_size : self.config.prompt_vocab_size
            + self.config.discrete_token_vocab_size
            + 1
        ]
        token_logits = F.linear(audio_hidden_states, tied_weights)
        # diffuion loss
        if continuous_sequence is not None:
            diffusion_loss = self.diffusion_head(
                target=continuous_sequence,
                target_padding_mask=audio_padding_mask,
                context_vector=audio_hidden_states[
                    :, :-1
                ],  # we stop at last audio frame
            ).loss
        else:
            diffusion_loss = None

        return HybridTTSOutput(
            token_logits=token_logits,
            diffusion_loss=diffusion_loss,
            continuous_ratio=None,
            target_tokens=None,
        )

    # -------------------------------------------------------------------------
    # Evaluation / inference helper
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        batch: Dict[str, torch.Tensor],
        vae: nn.Module,
        num_steps: int = 16,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        **kwargs,
    ):
        """
        Reconstruct audio features from ground-truth tokens via the diffusion head.
        Used for evaluation.
        """
        raise NotImplementedError("not implemented")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
