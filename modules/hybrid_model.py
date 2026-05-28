import torch
import logging
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from typing import Dict, Optional
from .diffusion_head.cfm import DiT
from .configs import HybridTTSConfig
from torch.nn.utils.rnn import pad_sequence
from .output_dataclasses import HybridTTSOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HybridTokenizer
# ---------------------------------------------------------------------------


class HybridTokenizer:
    """
    This is a helper for handling the shifting between prompt tokens and discrete audio tokens
    """

    def __init__(
        self,
        prompt_vocab_size: int,
        start_audio_id: int,
        end_audio_id: int,
        pad_id: int,
        discrete_token_vocab_size: int = 1024,
    ):
        self.prompt_vocab_size = prompt_vocab_size
        self.start_audio_id = start_audio_id
        self.end_audio_id = end_audio_id
        self.pad_id = pad_id
        self.discrete_token_vocab_size = discrete_token_vocab_size

    @property
    def unified_vocab_size(self) -> int:
        return self.prompt_vocab_size + self.discrete_token_vocab_size

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Dummy save_pretrained to avoid AttributeError when HF Trainer tries to save it.
        The tokenizer parameters are currently instantiated via config in build_tokenizer.
        """
        pass


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
# Backbone
# ---------------------------------------------------------------------------


class CausalLMWrapper(nn.Module):
    def __init__(self, model_name: str, unified_vocab_size: int, pad_token_id: int):
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM

        if model_name == "qwen-0.5b":
            hf_config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        elif model_name == "llama-1b":
            hf_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        else:
            raise ValueError(f"Unknown model_type: {model_name}")

        hf_config.vocab_size = unified_vocab_size
        hf_config.pad_token_id = pad_token_id
        self.model = AutoModelForCausalLM.from_config(hf_config)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        **kwargs,
    ):
        input_dict = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }

        # deal with the extra args that are passed by the sample function
        for arg_name in ["past_key_values", "use_cache", "position_ids"]:
            if arg_name in kwargs:
                input_dict[arg_name] = kwargs[arg_name]

        # forward pass
        outputs = self.model(**input_dict)
        return outputs.hidden_states[-1]

    def inference_forward(self, inputs_embeds, attention_mask=None, **kwargs):
        input_dict = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }

        # deal with the extra args that are passed by the sample function
        for arg_name in ["past_key_values", "use_cache", "position_ids"]:
            if arg_name in kwargs:
                input_dict[arg_name] = kwargs[arg_name]

        # forward pass
        outputs = self.model(**input_dict)
        hidden_states = outputs.hidden_states[-1][:, -1:, :]
        past_key_values = outputs.past_key_values

        return hidden_states, past_key_values


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id
        )
        self.pos_emb = nn.Embedding(config.max_position_embeddings, config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        self.norm = nn.LayerNorm(config.hidden_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask=None, **kwargs):
        B, L, D = inputs_embeds.size()
        positions = torch.arange(L, device=inputs_embeds.device).unsqueeze(0)
        x = inputs_embeds + self.pos_emb(positions)

        src_key_padding_mask = ~attention_mask if attention_mask is not None else None

        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)

        out = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,
        )
        return self.norm(out)

    def inference_forward(self, inputs_embeds, attention_mask=None, **kwargs):
        out = self(inputs_embeds, attention_mask, **kwargs)
        past_key_value = None
        return out[:, -1, :], past_key_value


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

    def __init__(self, config: HybridTTSConfig, tokenizer: "HybridTokenizer"):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_id
        self.uncond_prob = config.uncond_prob
        self.discrete_only = config.discrete_only
        self.end_audio_id = tokenizer.end_audio_id
        self.start_audio_id = tokenizer.start_audio_id
        self.shift_audio_offset = config.shift_audio_offset
        self.unified_vocab_size = tokenizer.unified_vocab_size
        self.discrete_token_vocab_size = tokenizer.discrete_token_vocab_size
        if config.continuous_adapter_config is not None:
            config.continuous_adapter_config.in_dim = config.continuous_dim
        if config.diffusion_head_config is not None:
            config.diffusion_head_config.audio_latent_dim = config.continuous_dim

        # check on the shift_audio_offset
        if self.shift_audio_offset > 1:
            raise NotImplementedError("shift_audio_offset > 1 is not implemented yet")

        # ---- Backbone -------------------------------------------------------
        bb_cfg = config.backbone_config

        if bb_cfg.model_type == "native":
            bb_cfg.vocab_size = self.unified_vocab_size
            bb_cfg.pad_token_id = self.pad_token_id

            self.backbone = Transformer(bb_cfg)
            hidden_size = bb_cfg.hidden_dim
        else:
            if bb_cfg.from_pretrained:
                raise NotImplementedError(
                    "Fine-tuning from_pretrained is not yet implemented. Focus is on training from scratch."
                )
            self.backbone = CausalLMWrapper(
                bb_cfg.model_type, self.unified_vocab_size, self.pad_token_id
            )
            hidden_size = self.backbone.model.config.hidden_size

        self.config.apply_backbone_dims(hidden_size)

        # ---- Continuous adapter (VAE latents → hidden_size) ------------------
        if config.continuous_adapter_config is not None:
            self.continuous_adapter = MLPAdapter(config.continuous_adapter_config)
        elif not self.discrete_only:
            self.continuous_adapter = nn.Linear(config.continuous_dim, hidden_size)

        # ---- Normalisations --------------------------------------------------
        self.dynamic_normalizer = DynamicNormalizer(config.continuous_dim)
        self.norm_continuous = nn.LayerNorm(hidden_size)

        # ---- Output heads ----------------------------------------------------
        if config.diffusion_head_config is not None:
            self.diffusion_head = DiT(config.diffusion_head_config)
        else:
            self.diffusion_head = None
        # NOTE we use tie embeddings instead of a separate head
        # self.token_head = nn.Linear(
        #     hidden_size,
        #     self.unified_vocab_size,
        #     bias=False,
        # )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _extract_audio_tokens_span(self, discrete_sequence):
        """
        Extract audio start and end position from discrete sequence.
        """
        start_indices = (discrete_sequence == self.start_audio_id).long().argmax(dim=1)
        end_indices = (discrete_sequence == self.end_audio_id).long().argmax(dim=1)
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
        index_3d = (start_positions + 1 + self.shift_audio_offset).unsqueeze(-1)
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
            torch.tensor(self.pad_token_id).long().to(full_hidden_states.device)
        )
        for h, s, e in zip(full_hidden_states, start_indices, end_indices):
            audio_hidden_states.append(h[s : e + self.shift_audio_offset])
            audio_masks.append(
                torch.ones(
                    e - s + self.shift_audio_offset, dtype=torch.bool, device=h.device
                )
            )

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
        target_std: float = 0.1,
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

        if getattr(self.config, "no_augment_ratio") > 0.0:
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

    def uncondition(
        self, discrete_sequence: torch.LongTensor, attention_mask: torch.BoolTensor
    ):
        start_idx, end_idx = self._extract_audio_tokens_span(discrete_sequence)
        # --- Unconditional training: drop prompts randomly ---
        if self.training and getattr(self, "uncond_prob") > 0.0:
            B, L = discrete_sequence.shape
            drop_mask = (
                torch.rand(B, device=discrete_sequence.device) < self.uncond_prob
            )

            if drop_mask.any():
                new_discrete = discrete_sequence.clone()
                new_attention = attention_mask.clone()

                for b in range(B):
                    if drop_mask[b]:
                        s_idx = start_idx[b].item()
                        if s_idx > 0:
                            keep_len = L - s_idx
                            # Shift tokens and attention mask to the left
                            new_discrete[b, :keep_len] = discrete_sequence[b, s_idx:]
                            new_discrete[b, keep_len:] = self.pad_token_id

                            new_attention[b, :keep_len] = attention_mask[b, s_idx:]
                            new_attention[b, keep_len:] = False

                discrete_sequence = new_discrete
                attention_mask = new_attention

            return discrete_sequence, attention_mask

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
        discrete_sequence, attention_mask = self.uncondition(
            discrete_sequence, attention_mask
        )
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
        )

        audio_hidden_states = self._extract_audio_hidden_states(
            last_hidden_state,
            start_idx,
            end_idx,
        )

        # token logits
        target_embed_weight = self.backbone.get_output_embeddings().weight[
            self.end_audio_id :
        ]

        if self.shift_audio_offset:
            token_logits = F.linear(
                audio_hidden_states[:, : -self.shift_audio_offset], target_embed_weight
            )
        else:
            token_logits = F.linear(audio_hidden_states, target_embed_weight)
        # diffuion loss
        if continuous_sequence is not None and self.diffusion_head is not None:
            diffusion_loss = self.diffusion_head(
                target=continuous_sequence,
                target_padding_mask=audio_padding_mask,
                context_vector=audio_hidden_states[
                    :, self.shift_audio_offset : -1
                ],  # we stop at last audio frame
            ).loss
        else:
            diffusion_loss = None

        return HybridTTSOutput(token_logits=token_logits, diffusion_loss=diffusion_loss)

    # -------------------------------------------------------------------------
    # Evaluation / inference helper
    # -------------------------------------------------------------------------
    def _sample_token_ids(
        self,
        token_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.LongTensor:
        if temperature < 0:
            raise ValueError("temperature must be > 0")
        elif temperature == 0:
            return torch.argmax(token_logits, dim=-1)
        # scale logits
        scaled_logits = token_logits / temperature
        # categorical sampling
        dist = torch.distributions.Categorical(logits=scaled_logits)
        # sample ids
        return dist.sample()

    @torch.no_grad()
    def sample(
        self,
        batch: Dict[str, torch.Tensor],
        **kwargs,
    ):
        """
        Reconstruct audio features from ground-truth tokens via the diffusion head.
        Used for evaluation.
        """
        discrete_sequence = batch["discrete_sequence"]
        attention_mask = batch["attention_mask"]

        embed_layer = self.backbone.get_input_embeddings()

        all_continuous_tokens = []
        all_discrete_tokens = []

        max_steps = kwargs.get("max_steps", 250)

        do_cfg = kwargs.get("guidance_scale", 1.0) > 1.0
        B_orig, L = discrete_sequence.shape

        if do_cfg:
            start_idx, end_idx = self._extract_audio_tokens_span(discrete_sequence)
            uncond_discrete = discrete_sequence.clone()
            uncond_mask = attention_mask.clone()

            for b in range(B_orig):
                s_idx = start_idx[b].item()
                if s_idx > 0:
                    keep_len = L - s_idx
                    uncond_discrete[b, :keep_len] = discrete_sequence[b, s_idx:]
                    uncond_discrete[b, keep_len:] = self.pad_token_id

                    uncond_mask[b, :keep_len] = attention_mask[b, s_idx:]
                    uncond_mask[b, keep_len:] = False

            uncond_embs = embed_layer(uncond_discrete)
            input_embs = embed_layer(discrete_sequence)
            input_embs = torch.cat([input_embs, uncond_embs], dim=0)
            attention_mask = torch.cat([attention_mask, uncond_mask], dim=0)
        else:
            input_embs = embed_layer(discrete_sequence)

        discrete_only = getattr(self.config, "discrete_only", False)

        past_key_values = None

        for step in tqdm(range(max_steps)):

            last_hidden_state, past_key_values = self.backbone.inference_forward(
                inputs_embeds=input_embs,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            target_embed_weight = self.backbone.get_output_embeddings().weight[
                self.end_audio_id :
            ]

            if do_cfg:
                cond_hidden = last_hidden_state[:B_orig]
                uncond_hidden = last_hidden_state[B_orig:]

                token_logits_cond = F.linear(
                    cond_hidden.squeeze(1), target_embed_weight
                )
                token_logits_uncond = F.linear(
                    uncond_hidden.squeeze(1), target_embed_weight
                )

                cfg_scale = kwargs.get("guidance_scale")
                token_logits = token_logits_uncond + cfg_scale * (
                    token_logits_cond - token_logits_uncond
                )

                diffusion_context = (cond_hidden, uncond_hidden)
            else:
                token_logits = F.linear(
                    last_hidden_state.squeeze(1), target_embed_weight
                )
                diffusion_context = last_hidden_state

            sampled_id = self._sample_token_ids(token_logits, kwargs.get("temperature"))

            if (sampled_id == 0).all():
                break

            token_id = sampled_id - 1 + self.tokenizer.prompt_vocab_size

            all_discrete_tokens.append((sampled_id - 1).unsqueeze(-1))

            if discrete_only or self.diffusion_head is None:
                next_token = embed_layer(token_id).unsqueeze(1)  # (B, H) → (B, 1, H)
            else:
                generated_continuous_tokens = self.diffusion_head.generate(
                    num_steps=kwargs.get("num_steps"),
                    context_vector=diffusion_context,
                    temperature=kwargs.get("diffusion_temperature"),
                    guidance_scale=kwargs.get("guidance_scale"),
                ).audio_features

                all_continuous_tokens.append(generated_continuous_tokens)
                generated_continuous_tokens = self.norm_continuous(
                    self.continuous_adapter(generated_continuous_tokens)
                ).detach()

                # generated_continuous_tokens = (
                #     generated_continuous_tokens
                #     + torch.randn_like(generated_continuous_tokens) * 0.0
                # )

                next_token = (
                    embed_layer(token_id).unsqueeze(1) + generated_continuous_tokens
                )

            if do_cfg:
                next_token = next_token.repeat(2, 1, 1)
                sampled_mask = torch.cat(
                    [
                        torch.ones_like(sampled_id, dtype=torch.bool).unsqueeze(-1),
                        torch.ones_like(sampled_id, dtype=torch.bool).unsqueeze(-1),
                    ],
                    dim=0,
                )
            else:
                sampled_mask = torch.ones_like(sampled_id, dtype=torch.bool).unsqueeze(
                    -1
                )

            if self.config.backbone_config.model_type == "native":
                input_embs = torch.cat([input_embs, next_token], dim=1)
            else:
                input_embs = next_token

            attention_mask = torch.cat(
                [
                    attention_mask,
                    sampled_mask,
                ],
                dim=1,
            )

        if len(all_continuous_tokens) > 0:
            final_continuous = torch.cat(all_continuous_tokens, dim=1)
            final_continuous = self.dynamic_normalizer.denormalize(final_continuous)
        else:
            final_continuous = None

        final_discrete = torch.cat(all_discrete_tokens, dim=1)

        return {
            "discrete_tokens": final_discrete,
            "continuous_tokens": final_continuous,
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
