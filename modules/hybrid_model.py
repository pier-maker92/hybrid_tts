import torch
import torch.nn as nn
import logging
from typing import Dict, Optional

from .diffusion_head.cfm import DiT
from .configs import HybridTTSConfig
from transformers import AutoModel, AutoConfig
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
            f"PROMPT({valid_prompt_len}{'*' if pad_prompt else ''}): {prompt_str} "
            f"| <start_audio> "
            f"| AUDIO({valid_audio_len}/{L_audio_max} frames) "
            f"| <end_audio>@pos={1 + valid_audio_len}"
        )


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------


class DynamicNormalizer(nn.Module):
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))

    def forward(self, x, padding_mask=None):
        # x: (B, L, C)
        if self.training:
            if padding_mask is not None:
                mask = (~padding_mask).to(x.dtype).unsqueeze(-1)  # (B, L, 1)
                count = mask.sum()
                if count > 0:
                    batch_mean = (x * mask).sum() / (count * x.shape[-1])
                    batch_var = (((x - batch_mean) ** 2) * mask).sum() / (
                        count * x.shape[-1]
                    )
                else:
                    batch_mean = x.mean()
                    batch_var = x.var(unbiased=False)
            else:
                batch_mean = x.mean()
                batch_var = x.var(unbiased=False)

            with torch.no_grad():
                self.running_mean.copy_(
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * batch_mean.reshape(1)
                )
                self.running_var.copy_(
                    (1 - self.momentum) * self.running_var
                    + self.momentum * batch_var.reshape(1)
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

        [phoneme_0 ... phoneme_{P_i-1} | <PAD> ... | <start_audio> |
         audio_0 ... audio_{A_i-1} | <end_audio> | <PAD> ...]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              L_prompt_max positions     L_audio_max + 2 positions
                                         (start + audio_max + end)

    The <end_audio> token is placed at the per-sample valid position
    (1 + valid_audio_len[i]) inside the audio_extended block via scatter,
    so every sample in the batch gets its own end-of-audio marker at the
    correct frame, regardless of padding.
    """

    def __init__(self, config: HybridTTSConfig):
        super().__init__()
        self.config = config

        # ---- Backbone -------------------------------------------------------
        bb_cfg = config.backbone_config
        if bb_cfg.pretrained:
            self.backbone = AutoModel.from_pretrained(bb_cfg.model_name_or_path)
        else:
            hf_config = AutoConfig.from_pretrained(bb_cfg.model_name_or_path)
            # Training from scratch: use external LUTs only.
            hf_config.vocab_size = 0
            self.backbone = AutoModel.from_config(hf_config)

        hidden_size = self.backbone.config.hidden_size
        self.config.apply_backbone_dims(hidden_size)

        # ---- External embedding LUTs ----------------------------------------
        # prompt_emb covers: PAD(0) + phonemes(1..N) + <start_audio>(N+1) + <end_audio>(N+2)
        # prompt_vocab_size must already encode all of these (set in train.py).
        self.prompt_emb = nn.Embedding(
            config.prompt_vocab_size,
            hidden_size,
            padding_idx=config.pad_token_id,  # gradient zeroed for PAD
        )
        # Audio VQ token embeddings (separate space, discrete_token_vocab_size entries)
        self.discrete_emb = nn.Embedding(config.discrete_token_vocab_size, hidden_size)

        # ---- Continuous adapter (VAE latents → hidden_size) ------------------
        if config.continuous_adapter_config is not None:
            self.continuous_adapter = MLPAdapter(config.continuous_adapter_config)
        else:
            self.continuous_adapter = nn.Linear(config.continuous_dim, hidden_size)

        # ---- Normalisations --------------------------------------------------
        self.continuous_norm = DynamicNormalizer(config.continuous_dim)
        self.norm_discrete = nn.LayerNorm(hidden_size)
        self.norm_continuous = nn.LayerNorm(hidden_size)

        # ---- Output heads ----------------------------------------------------
        self.token_head = nn.Linear(hidden_size, config.discrete_token_vocab_size)
        self.diffusion_head = DiT(config.diffusion_head_config)

        # ---- Optional tokenizer for debug decoding --------------------------
        # Attach via model.tokenizer = HybridTokenizer(...) from train.py
        self.tokenizer: Optional[HybridTokenizer] = None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _build_audio_extended(
        self,
        audio_emb: torch.Tensor,  # (B, L_audio_max, H)
        padding_mask: torch.BoolTensor,  # (B, L_audio_max) True=pad
    ):
        """
        Build the extended audio embedding block and its attention mask.

        Returns:
            audio_ext_emb  : (B, L_audio_max + 2, H)
                             [<start> | audio_0 ... audio_{L-1} | <end> | PAD ...]
            audio_ext_attn : (B, L_audio_max + 2) long   1=attend, 0=ignore
            valid_lens     : (B,)  number of valid audio frames per sample
        """
        B, L_audio_max, H = audio_emb.shape
        device = audio_emb.device

        # Valid audio frame counts per sample
        valid_lens = (~padding_mask).sum(dim=1)  # (B,)

        # Initialise the extended block with PAD embeddings
        pad_emb = self.prompt_emb(
            torch.full(
                (1, 1), self.config.pad_token_id, dtype=torch.long, device=device
            )
        )  # (1, 1, H)
        audio_ext = pad_emb.expand(B, L_audio_max + 2, H).clone()

        # Position 0 → <start_audio>
        start_ids = torch.full(
            (B, 1), self.config.start_audio_id, dtype=torch.long, device=device
        )
        audio_ext[:, 0, :] = self.prompt_emb(start_ids).squeeze(1)

        # Positions 1 .. L_audio_max → audio frame embeddings
        audio_ext[:, 1 : L_audio_max + 1, :] = audio_emb

        # Position (valid_len + 1) → <end_audio>  (per sample, via scatter)
        end_ids = torch.full(
            (B, 1), self.config.end_audio_id, dtype=torch.long, device=device
        )
        end_emb = self.prompt_emb(end_ids)  # (B, 1, H)
        # end positions: valid_len + 1, clamped so it never exceeds L_audio_max + 1
        end_pos = (valid_lens + 1).clamp(max=L_audio_max + 1)  # (B,)
        end_pos_idx = end_pos.view(B, 1, 1).expand(B, 1, H)
        audio_ext.scatter_(1, end_pos_idx, end_emb)

        # Attention mask: attend to positions 0 .. valid_len+1 (inclusive)
        positions = torch.arange(L_audio_max + 2, device=device).unsqueeze(
            0
        )  # (1, L+2)
        audio_ext_attn = (positions <= (valid_lens + 1).unsqueeze(1)).long()  # (B, L+2)

        return audio_ext, audio_ext_attn, valid_lens

    def _extract_audio_hidden_states(
        self,
        full_hidden_states: torch.Tensor,
        L_prompt: int,
        L_audio_max: int,
        valid_lens: torch.Tensor,
    ):
        """
        Select hidden states that predict audio frames 0 .. L_audio_max-1.

        In the audio_ext block, frame t is predicted from position t:
          t=0 → <start_audio>, t>=1 → audio_emb[t-1].

        Returns:
            audio_hidden_states : (B, L_audio_max, H)
            audio_hidden_mask   : (B, L_audio_max) True = valid (non-pad) frame
        """
        audio_hidden_states = full_hidden_states[:, L_prompt : L_prompt + L_audio_max, :]
        device = full_hidden_states.device
        positions = torch.arange(L_audio_max, device=device).unsqueeze(0)
        audio_hidden_mask = positions < valid_lens.unsqueeze(1)
        audio_hidden_states = audio_hidden_states.masked_fill(
            ~audio_hidden_mask.unsqueeze(-1), 0.0
        )
        return audio_hidden_states, audio_hidden_mask

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
                    f"PROMPT({valid_p}): {raw_p} "
                    f"| <start_audio> "
                    f"| AUDIO({valid_a}/{L_audio_max} frames) "
                    f"| <end_audio>@pos={1 + valid_a}"
                )
            lines.append(f"  sample[{i}]: {desc}")
        logger.debug("\n".join(lines))
        print("\n".join(lines))  # also print so it shows up without debug log level

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        prompt_ids: torch.Tensor,
        discrete_tokens: torch.Tensor,
        continuous_tokens: torch.Tensor,
        prompt_mask: Optional[torch.BoolTensor] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        target_continuous: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> HybridTTSOutput:
        """
        Forward pass for training.

        Args:
            prompt_ids        : (B, L_prompt)  raw phoneme IDs (0 = PAD)
            discrete_tokens   : (B, L_audio)   VQ token indices
            continuous_tokens : (B, L_audio, C) VAE continuous latents
            prompt_mask       : (B, L_prompt)  True=valid, False=pad.
                                If None, assumed all valid (no prompt padding).
            padding_mask      : (B, L_audio)   True=pad, False=valid audio frame.
                                If None, assumed all valid.
            target_continuous : optional separate continuous target for diffusion head.
        """
        # ------------------------------------------------------------------
        # 1. Prompt embeddings
        #    prompt_ids are raw (0-indexed phonemes; 0=PAD in the batch).
        #    The LUT is indexed as: raw_id + prompt_offset.
        # ------------------------------------------------------------------
        p_emb = self.prompt_emb(prompt_ids + self.config.prompt_offset)
        B, L_prompt, H = p_emb.shape

        # ------------------------------------------------------------------
        # 2. Audio embeddings (discrete + continuous combined)
        # ------------------------------------------------------------------
        d_emb = self.discrete_emb(discrete_tokens)

        continuous_tokens = continuous_tokens.to(d_emb.dtype)
        if continuous_tokens.shape[-1] == 64:
            continuous_tokens = continuous_tokens[..., 32:]

        if padding_mask is None:
            L_audio = discrete_tokens.shape[1]
            padding_mask = torch.zeros(
                (B, L_audio), dtype=torch.bool, device=discrete_tokens.device
            )

        norm_ct = self.continuous_norm(continuous_tokens, padding_mask=padding_mask)
        c_emb = self.continuous_adapter(norm_ct)

        d_emb = self.norm_discrete(d_emb)
        c_emb = self.norm_continuous(c_emb)
        audio_emb = d_emb + c_emb  # (B, L_audio_max, H)
        L_audio_max = audio_emb.shape[1]

        # ------------------------------------------------------------------
        # 3. Build audio_extended: [<start> | audio | <end@valid_pos> | PAD]
        # ------------------------------------------------------------------
        audio_ext_emb, audio_ext_attn, valid_lens = self._build_audio_extended(
            audio_emb, padding_mask
        )
        # audio_ext_emb shape: (B, L_audio_max + 2, H)

        # ------------------------------------------------------------------
        # 4. Prompt attention mask
        # ------------------------------------------------------------------
        if prompt_mask is None:
            prompt_mask = torch.ones(
                (B, L_prompt), dtype=torch.bool, device=p_emb.device
            )
        prompt_attn = prompt_mask.long()

        # ------------------------------------------------------------------
        # 5. Concatenate full sequence
        #    [prompt (L_prompt) | audio_ext (L_audio_max + 2)]
        # ------------------------------------------------------------------
        inputs_embeds = torch.cat([p_emb, audio_ext_emb], dim=1)
        attention_mask = torch.cat([prompt_attn, audio_ext_attn], dim=1)

        # ------------------------------------------------------------------
        # 6. Optional debug: decode and print the sequence
        # ------------------------------------------------------------------
        if self.config.debug:
            self._debug_log(prompt_ids, prompt_mask, valid_lens, L_audio_max)

        # ------------------------------------------------------------------
        # 7. Backbone pass
        # ------------------------------------------------------------------
        bb_dtype = next(self.backbone.parameters()).dtype
        outputs = self.backbone(
            inputs_embeds=inputs_embeds.to(bb_dtype),
            attention_mask=attention_mask,
        )
        full_hidden_states = outputs.last_hidden_state

        # ------------------------------------------------------------------
        # 8. Audio hidden states for output heads (per-sample valid lengths)
        # ------------------------------------------------------------------
        audio_hidden_states, audio_hidden_mask = self._extract_audio_hidden_states(
            full_hidden_states, L_prompt, L_audio_max, valid_lens
        )

        # ------------------------------------------------------------------
        # 9. Output heads
        # ------------------------------------------------------------------
        token_logits = self.token_head(
            audio_hidden_states.to(next(self.token_head.parameters()).dtype)
        )

        if target_continuous is None:
            target_continuous = continuous_tokens

        target_continuous = target_continuous.to(d_emb.dtype)
        if target_continuous.shape[-1] == 64:
            target_continuous = target_continuous[..., 32:]

        target_continuous = self.continuous_norm(
            target_continuous, padding_mask=padding_mask
        )

        self._assert_audio_masks_aligned(padding_mask, audio_hidden_mask)

        audio_hidden_states = audio_hidden_states.to(
            next(self.diffusion_head.parameters()).dtype
        )
        assert audio_hidden_states.shape[:2] == target_continuous.shape[:2], (
            f"context/target length mismatch: {audio_hidden_states.shape[:2]} "
            f"vs {target_continuous.shape[:2]}"
        )

        diffusion_output = self.diffusion_head(
            target=target_continuous,
            target_padding_mask=padding_mask,
            context_vector=audio_hidden_states,
        )

        return HybridTTSOutput(
            token_logits=token_logits,
            diffusion_loss=diffusion_output.loss,
            diffusion_output=diffusion_output,
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

        # Re-run the backbone to get hidden states
        p_emb = self.prompt_emb(prompt_ids + self.config.prompt_offset)
        B, L_prompt, H = p_emb.shape

        d_emb = self.discrete_emb(discrete_tokens)

        continuous_tokens = continuous_tokens.to(d_emb.dtype)
        if continuous_tokens.shape[-1] == 64:
            continuous_tokens = continuous_tokens[..., 32:]

        norm_c = self.continuous_norm(continuous_tokens, padding_mask=padding_mask)
        c_emb = self.continuous_adapter(norm_c)
        d_emb = self.norm_discrete(d_emb)
        c_emb = self.norm_continuous(c_emb)
        audio_emb = d_emb + c_emb
        L_audio_max = audio_emb.shape[1]

        audio_ext_emb, audio_ext_attn, valid_lens = self._build_audio_extended(
            audio_emb, padding_mask
        )

        if prompt_mask is None:
            prompt_mask = torch.ones(
                (B, L_prompt), dtype=torch.bool, device=p_emb.device
            )
        prompt_attn = prompt_mask.long()

        inputs_embeds = torch.cat([p_emb, audio_ext_emb], dim=1)
        attention_mask = torch.cat([prompt_attn, audio_ext_attn], dim=1)

        bb_dtype = next(self.backbone.parameters()).dtype
        outputs_bb = self.backbone(
            inputs_embeds=inputs_embeds.to(bb_dtype),
            attention_mask=attention_mask,
        )
        full_hidden_states = outputs_bb.last_hidden_state

        audio_hidden_states, audio_hidden_mask = self._extract_audio_hidden_states(
            full_hidden_states, L_prompt, L_audio_max, valid_lens
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

        return {
            "decoder_output": DecoderOutput(
                audio_features=reconstructed_mel,
                padding_mask=reconstructed_padding_mask,
            ),
            "token_logits": self.token_head(audio_hidden_states),
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
