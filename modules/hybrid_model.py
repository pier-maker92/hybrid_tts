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
        self.continuous_norm = DynamicNormalizer(config.continuous_dim)
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

    def _build_full_sequence(
        self,
        p_emb: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.BoolTensor,
        audio_emb: torch.Tensor,
        padding_mask: torch.BoolTensor,
    ):
        """
        Insert audio embeddings between <start_audio> and <end_audio> in the prompt.
        """
        pass

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

    def _assert_audio_masks_aligned(
        self,
        padding_mask: torch.BoolTensor,
        audio_hidden_mask: torch.BoolTensor,
    ) -> None:
        """padding_mask True=pad; audio_hidden_mask True=valid predictor frame."""
        context_pad = ~audio_hidden_mask
        assert torch.equal(padding_mask, context_pad), (
            "padding_mask and context padding derived from audio hidden states "
            "must match"
        )

    def _debug_log(
        self,
        prompt_ids: torch.Tensor,  # (B, L_prompt)
        prompt_mask: torch.BoolTensor,  # (B, L_prompt)
        valid_lens: torch.Tensor,  # (B,)
        L_audio_max: int,
    ):
        """Print a human-readable view of what enters the backbone."""
        B = prompt_ids.shape[0]
        lines = ["[DEBUG] Backbone input sequences:"]
        for i in range(B):
            valid_p = int(prompt_mask[i].sum().item())
            valid_a = int(valid_lens[i].item())
            if self.tokenizer is not None:
                desc = self.tokenizer.decode_sample(
                    prompt_ids[i],
                    valid_prompt_len=valid_p,
                    valid_audio_len=valid_a,
                    L_audio_max=L_audio_max,
                )
            else:
                # Fallback: raw IDs
                raw_p = prompt_ids[i, :valid_p].tolist()
                desc = (
                    f"PROMPT_WITH_TAGS({valid_p}): {raw_p} "
                    f"| AUDIO_INSERTED({valid_a}/{L_audio_max} frames) "
                )
            lines.append(f"  sample[{i}]: {desc}")
        logger.debug("\n".join(lines))
        print("\n".join(lines))  # also print so it shows up without debug log level

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
        audio_attention_mask: torch.BoolTensor,
        **kwargs,
    ) -> HybridTTSOutput:
        """
        Forward pass for training.

        Args:
            discrete_sequence: (B, L_discrete) prompt tokens
            attention_mask: (B, L_discrete) True = valid, False = pad
            continuous_sequence: (B, L_audio, C) continuous tokens
            audio_attention_mask: (B, L_audio) True = valid, False = pad
        """
        start_idx, end_idx = self._extract_audio_tokens_span(discrete_sequence)
        embed_layer = self.backbone.get_input_embeddings()
        discrete_emb = embed_layer(discrete_sequence)

        last_hidden_state = self.backbone(
            inputs_embeds=discrete_emb,
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

        tied_weights = self.backbone.get_input_embeddings().weight[
            self.config.prompt_vocab_size : self.config.prompt_vocab_size
            + self.config.discrete_token_vocab_size
            + 1
        ]

        token_logits = F.linear(audio_hidden_states, tied_weights)

        return HybridTTSOutput(
            token_logits=token_logits,
            diffusion_loss=None,
            diffusion_output=None,
            continuous_ratio=None,
            target_tokens=None,
        )

    # -------------------------------------------------------------------------
    # Evaluation / inference helper
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def encode_decode(
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
        self.eval()

        prompt_ids = batch["prompt_ids"]
        discrete_tokens = batch["discrete_tokens"]
        continuous_tokens = batch["continuous_tokens"]
        padding_mask = batch["padding_mask"]
        prompt_mask = batch.get("prompt_mask")  # may be absent in old batches

        unified_embed_tokens = self.backbone.get_input_embeddings()

        # Re-run the backbone to get hidden states
        p_emb = unified_embed_tokens(prompt_ids + self.config.prompt_offset)
        B, L_prompt, H = p_emb.shape

        d_emb = unified_embed_tokens(discrete_tokens + self.config.prompt_vocab_size)

        continuous_tokens = continuous_tokens.to(d_emb.dtype)
        if continuous_tokens.shape[-1] == 64:
            continuous_tokens = continuous_tokens[..., 32:]

        norm_c = self.continuous_norm(continuous_tokens, padding_mask=padding_mask)
        c_emb = self.continuous_adapter(norm_c)
        c_emb = self.norm_continuous(c_emb)

        valid_audio = (~padding_mask).unsqueeze(-1)
        d_emb = d_emb * valid_audio
        c_emb = c_emb * valid_audio

        audio_emb = d_emb + c_emb
        L_audio_max = audio_emb.shape[1]

        if prompt_mask is None:
            prompt_mask = torch.ones(
                (B, L_prompt), dtype=torch.bool, device=p_emb.device
            )

        inputs_embeds, attention_mask, start_indices, valid_lens = (
            self._build_full_sequence(
                p_emb, prompt_ids, prompt_mask, audio_emb, padding_mask
            )
        )

        bb_dtype = next(self.backbone.parameters()).dtype
        outputs_bb = self.backbone(
            inputs_embeds=inputs_embeds.to(bb_dtype),
            attention_mask=attention_mask,
        )
        full_hidden_states = outputs_bb.last_hidden_state

        audio_hidden_states, audio_hidden_mask = self._extract_audio_hidden_states(
            full_hidden_states, start_indices, L_audio_max, valid_lens
        )

        self._assert_audio_masks_aligned(padding_mask, audio_hidden_mask)

        # Generate with diffusion head (custom num_steps / guidance_scale)
        latents_output = self.diffusion_head.generate(
            num_steps=num_steps,
            context_vector=audio_hidden_states.to(
                next(self.diffusion_head.parameters()).dtype
            ),
            temperature=temperature,
            guidance_scale=guidance_scale,
            padding_mask=padding_mask,
        )
        z = latents_output.audio_features  # (B, L, C) — normalised space
        z = self.continuous_norm.denormalize(z)  # back to original scale

        # Reconstruct full 64-dim latent by prepending VQ codebook embeddings
        tokens_tensor = batch["discrete_tokens"].long()
        vq_emb = vae.encoder.vq.codebook(tokens_tensor)  # (B, L, 32)
        z_vae = torch.cat([vq_emb, z], dim=-1)  # (B, L, 64)

        reconstructed_mel, reconstructed_padding_mask = vae.sample(
            num_steps=num_steps,
            temperature=temperature,
            guidance_scale=guidance_scale,
            z=z_vae,
            padding_mask=latents_output.padding_mask,
        )

        tied_weights = self.backbone.get_input_embeddings().weight[
            self.config.prompt_vocab_size : self.config.prompt_vocab_size
            + self.config.discrete_token_vocab_size
            + 1
        ]
        return {
            "decoder_output": DecoderOutput(
                audio_features=reconstructed_mel,
                padding_mask=reconstructed_padding_mask,
            ),
            "token_logits": F.linear(audio_hidden_states, tied_weights),
        }

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
