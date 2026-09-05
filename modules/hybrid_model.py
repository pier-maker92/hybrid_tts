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
        audio_bpe=None,
        char_tokenizer=None,
        audio_placeholder_id: Optional[int] = None,
    ):
        self.prompt_vocab_size = prompt_vocab_size
        self.start_audio_id = start_audio_id
        self.end_audio_id = end_audio_id
        self.pad_id = pad_id
        self.discrete_token_vocab_size = discrete_token_vocab_size
        self.audio_bpe = audio_bpe
        self.char_tokenizer = char_tokenizer
        self.audio_placeholder_id = audio_placeholder_id

    def encode_text(self, text: str):
        """Encode raw text to token IDs using the character tokenizer.

        Returns a list of integer token IDs. Raises ValueError if no char tokenizer is loaded.
        """
        if self.char_tokenizer is None:
            raise ValueError("No character tokenizer loaded.")
        return self.char_tokenizer.encode(text)

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
    def __init__(
        self,
        model_name: str,
        unified_vocab_size: int,
        pad_token_id: int,
        tie_word_embeddings: bool,
    ):
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
        hf_config.tie_word_embeddings = tie_word_embeddings
        self.model = AutoModelForCausalLM.from_config(hf_config)

        if not tie_word_embeddings:
            self.model.lm_head = nn.Identity()

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
            "output_hidden_states": False,
            "return_dict": True,
        }

        # deal with the extra args that are passed by the sample function
        for arg_name in ["past_key_values", "use_cache", "position_ids"]:
            if arg_name in kwargs:
                input_dict[arg_name] = kwargs[arg_name]

        # forward pass
        outputs = self.model.base_model(**input_dict)
        return outputs.last_hidden_state

    def inference_forward(self, inputs_embeds, attention_mask=None, **kwargs):
        input_dict = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "output_hidden_states": False,
            "return_dict": True,
        }

        # deal with the extra args that are passed by the sample function
        for arg_name in ["past_key_values", "use_cache", "position_ids"]:
            if arg_name in kwargs:
                input_dict[arg_name] = kwargs[arg_name]

        # forward pass
        outputs = self.model.base_model(**input_dict)
        hidden_states = outputs.last_hidden_state[:, -1:, :]
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
        self.discrete = config.discrete
        self.continuous = config.continuous
        self.backbone_voice_condition = config.backbone_config.voice_condition
        self.diffusion_voice_condition = (
            config.diffusion_head_config is not None
            and config.diffusion_head_config.voice_condition
        )
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
        elif bb_cfg.model_type == "llama3":
            from .llama3 import LlamaBackbone
            self.backbone = LlamaBackbone(
                vocab_size=self.unified_vocab_size,
                pad_token_id=self.pad_token_id,
                num_layers=bb_cfg.num_layers or 12,
                dim=bb_cfg.hidden_dim or 512,
                ffn_dim=bb_cfg.ffn_dim or 2048,
                n_heads=bb_cfg.num_heads or 4,
                n_kv_heads=bb_cfg.n_kv_heads or 1,
                dropout=bb_cfg.dropout or 0.0,
                rope_theta=bb_cfg.rope_theta or 10000.0,
                max_seq_len=bb_cfg.max_position_embeddings or 4096,
                tie_word_embeddings=(self.discrete and not self.continuous)
                or bb_cfg.force_weight_tying,
            )
            hidden_size = bb_cfg.hidden_dim or 512
        else:
            if bb_cfg.from_pretrained:
                raise NotImplementedError(
                    "Fine-tuning from_pretrained is not yet implemented. Focus is on training from scratch."
                )
            self.backbone = CausalLMWrapper(
                bb_cfg.model_type,
                self.unified_vocab_size,
                self.pad_token_id,
                tie_word_embeddings=(self.discrete and not self.continuous)
                or bb_cfg.force_weight_tying,
            )
            hidden_size = self.backbone.model.config.hidden_size

        self.config.apply_backbone_dims(hidden_size)

        # ---- Continuous adapter (VAE latents → hidden_size) ------------------
        if config.continuous_adapter_config is not None:
            self.continuous_adapter = MLPAdapter(config.continuous_adapter_config)
        elif self.continuous:
            self.continuous_adapter = nn.Linear(config.continuous_dim, hidden_size)

        if config.speaker_adapter_config is not None:
            self.speaker_adapter = MLPAdapter(config.speaker_adapter_config)
        else:
            self.speaker_adapter = None

        # ---- Normalisations --------------------------------------------------
        self.dynamic_normalizer = DynamicNormalizer(config.continuous_dim)
        self.norm_continuous = nn.LayerNorm(hidden_size)

        # ---- Scaling ---------------------------------------------------------
        scaling_mode = getattr(config, "continuous_scaling_mode", None)
        if scaling_mode == "learnable":
            self.continuous_scale = nn.Parameter(torch.tensor(0.02))
        elif scaling_mode == "fixed":
            self.continuous_scale = 0.02
        else:
            self.continuous_scale = None

        # ---- Output heads ----------------------------------------------------
        if config.diffusion_head_config is not None:
            self.diffusion_head = DiT(config.diffusion_head_config)
        else:
            self.diffusion_head = None

        # NOTE: weight tying is enabled only in discrete-only mode, since tying
        # constrains the hidden states to align with the discrete token embedding
        # space, potentially reducing the richness needed for continuous/acoustic
        # conditioning when predicting both discrete and continuous features.
        if (
            self.discrete
            and self.continuous
            and not bb_cfg.force_weight_tying
        ):
            print("initializing token head")
            self.token_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(
                    hidden_size, self.discrete_token_vocab_size + 1, bias=False
                ),  # Vq tokens + audio EOS
            )

    @torch.no_grad()
    def initialize_discrete_embeddings(
        self,
        codebook_embeddings: torch.Tensor,
    ) -> None:
        if not self.discrete:
            logger.info("Skipping discrete embedding init because discrete=false.")
            return

        if codebook_embeddings.ndim != 2:
            raise ValueError(
                "Quantizer embeddings must have shape [codebook_size, dim], got "
                f"{tuple(codebook_embeddings.shape)}."
            )
        if codebook_embeddings.shape[0] != self.discrete_token_vocab_size:
            raise ValueError(
                "Quantizer codebook size does not match tokenizer discrete vocab: "
                f"{codebook_embeddings.shape[0]} vs {self.discrete_token_vocab_size}."
            )

        embed_weight = self.backbone.get_input_embeddings().weight
        vocab_start = self.tokenizer.prompt_vocab_size
        vocab_end = vocab_start + self.discrete_token_vocab_size
        hidden_dim = embed_weight.shape[1]
        codebook_dim = codebook_embeddings.shape[1]

        resized = embed_weight.new_zeros(self.discrete_token_vocab_size, hidden_dim)
        copy_dim = min(hidden_dim, codebook_dim)
        resized[:, :copy_dim] = codebook_embeddings[:, :copy_dim].to(
            device=embed_weight.device,
            dtype=embed_weight.dtype,
        )
        embed_weight[vocab_start:vocab_end].copy_(resized)
        logger.info(
            "Initialized %d discrete token embeddings from quantizer codebook "
            "(codebook_dim=%d, hidden_dim=%d).",
            self.discrete_token_vocab_size,
            codebook_dim,
            hidden_dim,
        )

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

    def _left_pad_valid_tokens(
        self, discrete_sequence: torch.LongTensor, attention_mask: torch.BoolTensor
    ):
        """
        Pack valid tokens to the right before cached autoregressive decoding.
        Right padding makes the first generated step read from pad positions in batch.
        """
        B, L = discrete_sequence.shape
        packed_discrete = discrete_sequence.new_full((B, L), self.pad_token_id)
        packed_attention = torch.zeros_like(attention_mask, dtype=torch.bool)

        for b in range(B):
            valid_tokens = discrete_sequence[b, attention_mask[b]]
            keep_len = valid_tokens.numel()
            if keep_len > 0:
                packed_discrete[b, L - keep_len :] = valid_tokens
                packed_attention[b, L - keep_len :] = True

        return packed_discrete, packed_attention

    def _make_position_ids(self, attention_mask: torch.BoolTensor):
        return (attention_mask.long().cumsum(dim=1) - 1).clamp_min(0)

    def _extract_voice_condition(
        self,
        voice_conditioner,
        reference_audios_srs,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not (self.backbone_voice_condition or self.diffusion_voice_condition):
            return None
        if voice_conditioner is None:
            raise RuntimeError("voice_condition=true requires a VAE voice_conditioner.")
        if reference_audios_srs is None:
            raise RuntimeError("voice_condition=true requires reference_audios_srs.")
        if not hasattr(voice_conditioner, "extract_speaker_embedding"):
            raise RuntimeError(
                "voice_conditioner must expose extract_speaker_embedding(audios_srs)."
            )

        speaker_embedding = voice_conditioner.extract_speaker_embedding(
            reference_audios_srs
        )
        if speaker_embedding is None:
            raise RuntimeError(
                "voice_condition=true requires a VAE checkpoint with speaker encoder."
            )
        return speaker_embedding.to(device=device, dtype=dtype)

    def _speaker_hidden(self, speaker_embedding: Optional[torch.Tensor]):
        if speaker_embedding is None or self.speaker_adapter is None:
            return None
        return self.norm_continuous(self.speaker_adapter(speaker_embedding))

    def _add_backbone_voice_condition(
        self,
        input_embs: torch.Tensor,
        attention_mask: torch.BoolTensor,
        speaker_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        speaker_hidden = self._speaker_hidden(speaker_embedding)
        if speaker_hidden is None:
            return input_embs
        speaker_hidden = speaker_hidden.unsqueeze(1).to(dtype=input_embs.dtype)
        valid_mask = attention_mask.unsqueeze(-1).to(dtype=input_embs.dtype)
        return input_embs + speaker_hidden * valid_mask

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

    def noise_augment_ode_simulation(
        self,
        continuous_tokens: torch.FloatTensor,
        padding_mask: torch.BoolTensor,
        target_std: float = 0.2,
        min_ode_steps: int = 1,
        max_ode_steps: int = 32,
    ):
        """Simula l'errore del solver ODE in inferenza in modo efficiente."""
        B, T, D = continuous_tokens.shape
        device = continuous_tokens.device
        dtype = continuous_tokens.dtype

        # 1. Campionamento del numero di step (K) per ogni elemento della batch.
        # Questo simula il variare del NFE (Number of Function Evaluations).
        K = torch.randint(
            min_ode_steps, max_ode_steps + 1, (B, 1, 1), device=device, dtype=dtype
        )

        # 2. ODE Truncation Error Scaling
        # L'errore globale di un solver (es. Eulero) scala con delta_t = 1/K.
        # Normalizziamo in modo che a K=1 si abbia il target_std massimo,
        # e a K=32 si abbia un rumore quasi nullo.
        delta_t = 1.0 / K
        ode_std = target_std * delta_t

        # 3. Autoregressive Exposure Bias Scaling
        # L'errore cresce man mano che il trasformatore genera nuovi token.
        time_scale = torch.linspace(0.0, 1.0, steps=T, dtype=dtype, device=device)
        time_scale = time_scale.view(1, T, 1)

        # Varianza finale effettiva
        effective_std = ode_std * time_scale

        # 4. Purely Random Brownian Drift (Vettorizzato, NO FOR LOOPS)
        # Generiamo step di rumore indipendente e li sommiamo cumulativamente.
        # Questo crea il "drift" spezzettato senza dover calcolare nulla di complesso.
        noise_steps = torch.randn((B, T, D), dtype=dtype, device=device)

        # cumsum simula la natura cumulativa dell'errore lungo i token
        correlated_noise = torch.cumsum(noise_steps, dim=1) / (T**0.5)

        # Applichiamo la magnitudo dell'errore ODE
        noise = correlated_noise * effective_std

        # 5. Masking configurabile
        if getattr(self.config, "no_augment_ratio", 0.0) > 0.0:
            keep_mask = (
                torch.rand(B, 1, 1, device=device) >= self.config.no_augment_ratio
            )
            noise = noise * keep_mask.to(dtype)

        corrupted_continuous_tokens = continuous_tokens + noise
        corrupted_continuous_tokens = corrupted_continuous_tokens.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )

        return corrupted_continuous_tokens

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

        if getattr(self.config, "no_augment_ratio") > 0.0:
            # Randomly select a portion of the batch to NOT be augmented (std=0)
            B = continuous_tokens.shape[0]
            keep_mask = (
                torch.rand(B, 1, 1, device=continuous_tokens.device)
                >= self.config.no_augment_ratio
            )
            std = std * keep_mask.to(std.dtype)

        noise = torch.randn_like(continuous_tokens)  # * std
        corrupted_continuous_tokens = continuous_tokens * std + noise * (1 - std)
        corrupted_continuous_tokens = corrupted_continuous_tokens.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        return corrupted_continuous_tokens

    def uncondition(
        self, discrete_sequence: torch.LongTensor, attention_mask: torch.BoolTensor
    ):
        start_idx, end_idx = self._extract_audio_tokens_span(discrete_sequence)
        # --- Unconditional training: drop prompts randomly ---
        B, L = discrete_sequence.shape
        drop_mask = torch.rand(B, device=discrete_sequence.device) < self.uncond_prob

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

    def get_token_logits(self, tokens_hidden_states: torch.FloatTensor):
        """
        Get token logits from the backbone.
        """
        # token head
        if hasattr(self, "token_head"):
            head_dtype = next(self.token_head.parameters()).dtype
            tokens_hidden_states = tokens_hidden_states.to(dtype=head_dtype)
            return self.token_head(tokens_hidden_states)
        else:
            target_embed_weight = self.backbone.get_output_embeddings().weight[
                self.end_audio_id :
            ]
            tokens_hidden_states = tokens_hidden_states.to(
                dtype=target_embed_weight.dtype
            )
            return F.linear(tokens_hidden_states, target_embed_weight)

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
        if self.training and getattr(self, "uncond_prob") > 0.0:
            discrete_sequence, attention_mask = self.uncondition(
                discrete_sequence, attention_mask
            )
        start_idx, end_idx = self._extract_audio_tokens_span(discrete_sequence)
        embed_layer = self.backbone.get_input_embeddings()
        input_embs = embed_layer(discrete_sequence)
        speaker_embedding = self._extract_voice_condition(
            voice_conditioner=kwargs.get("voice_conditioner"),
            reference_audios_srs=kwargs.get("reference_audios_srs"),
            device=input_embs.device,
            dtype=input_embs.dtype,
        )

        norm_ratio = None

        if continuous_sequence is not None:
            continuous_sequence = self.dynamic_normalizer(continuous_sequence)
            corrupted_c_seq = self.noise_augment_continuous_token(
                continuous_sequence.clone(), audio_padding_mask
            )
            adapted_c_emb = self.norm_continuous(
                self.continuous_adapter(corrupted_c_seq)
            )

            if getattr(self, "continuous_scale", None) is not None:
                adapted_c_emb = adapted_c_emb * self.continuous_scale

            discrete_norm = input_embs.norm(dim=-1).mean()
            continuous_norm = adapted_c_emb.norm(dim=-1).mean()
            norm_ratio = discrete_norm / (continuous_norm + 1e-8)

            input_embs = self._add_continuous_token(
                continuous_sequence=adapted_c_emb,
                audio_padding_mask=audio_padding_mask,
                discrete_emb=input_embs,
                start_positions=start_idx,
                attention_mask=attention_mask,
            )

        if self.backbone_voice_condition:
            input_embs = self._add_backbone_voice_condition(
                input_embs=input_embs,
                attention_mask=attention_mask,
                speaker_embedding=speaker_embedding,
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

        # tokens
        token_logits = None
        if self.discrete:
            if self.shift_audio_offset:
                tokens_hidden_states = audio_hidden_states[
                    :, : -self.shift_audio_offset, :
                ]
            else:
                tokens_hidden_states = audio_hidden_states
            token_logits = self.get_token_logits(tokens_hidden_states)

        # continuous features
        diffusion_loss = None
        if continuous_sequence is not None and self.diffusion_head is not None:
            diffusion_dtype = next(self.diffusion_head.parameters()).dtype
            diffusion_loss = self.diffusion_head(
                target=continuous_sequence.to(dtype=diffusion_dtype),
                target_padding_mask=audio_padding_mask,
                context_vector=audio_hidden_states[:, self.shift_audio_offset : -1].to(
                    dtype=diffusion_dtype
                ),  # we stop at last audio frame
                speaker_embedding=(
                    speaker_embedding.to(dtype=diffusion_dtype)
                    if self.diffusion_voice_condition
                    else None
                ),
            ).loss

        return HybridTTSOutput(
            token_logits=token_logits,
            diffusion_loss=diffusion_loss,
            norm_ratio=norm_ratio,
        )

    # -------------------------------------------------------------------------
    # Evaluation / inference helper
    # -------------------------------------------------------------------------
    def _sample_token_ids(
        self,
        token_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.LongTensor:
        if temperature is None:
            temperature = 1.0
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        elif temperature == 0:
            return torch.argmax(token_logits, dim=-1)

        scaled_logits = (token_logits / temperature).float()
        finite_mask = torch.isfinite(scaled_logits)
        invalid_rows = (~finite_mask.any(dim=-1)) | torch.isposinf(scaled_logits).any(
            dim=-1
        )
        if not finite_mask.all():
            logger.warning(
                "Non-finite token logits encountered during sampling; forcing EOS "
                "for affected rows."
            )
            scaled_logits = scaled_logits.masked_fill(~finite_mask, -torch.inf)
            scaled_logits[invalid_rows] = 0.0

        probs = torch.softmax(scaled_logits, dim=-1)
        prob_sums = probs.sum(dim=-1)
        invalid_probs = (
            invalid_rows
            | (~torch.isfinite(probs).all(dim=-1))
            | (probs < 0).any(dim=-1)
            | (~torch.isfinite(prob_sums))
            | (prob_sums <= 0)
        )
        if invalid_probs.any():
            logger.warning(
                "Invalid token probabilities encountered during sampling; using EOS "
                "for affected rows."
            )
            probs = probs.clone()
            probs[invalid_probs] = 0.0
            probs[invalid_probs, 0] = 1.0

        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

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
        speaker_embedding = self._extract_voice_condition(
            voice_conditioner=kwargs.get("voice_conditioner"),
            reference_audios_srs=kwargs.get(
                "reference_audios_srs",
                batch.get("reference_audios_srs"),
            ),
            device=discrete_sequence.device,
            dtype=embed_layer.weight.dtype,
        )

        max_steps = kwargs.get("max_steps", 250)

        guidance_scale = kwargs.get("guidance_scale", None)
        if guidance_scale is None:
            guidance_scale = 1.0
        do_cfg = guidance_scale != 1.0 and not (
            (self.discrete and not self.continuous) or self.diffusion_head is None
        )
        B_orig, L = discrete_sequence.shape

        # # PyTorch SDPA on MPS produces incorrect hidden states for left-padded
        # # batches under no_grad. Keep the working batch-1 path unchanged.
        # NOTE: AI refuse. I don't think this is useful
        # if discrete_sequence.device.type == "mps" and (B_orig > 1 or do_cfg):
        #     backbone_model = getattr(self.backbone, "model", None)
        #     backbone_config = getattr(backbone_model, "config", None)
        #     if backbone_config is not None:
        #         backbone_config._attn_implementation = "eager"

        discrete_sequence, attention_mask = self._left_pad_valid_tokens(
            discrete_sequence, attention_mask
        )

        if do_cfg:
            start_idx, _ = self._extract_audio_tokens_span(discrete_sequence)
            uncond_discrete = discrete_sequence.new_full(
                discrete_sequence.shape, self.pad_token_id
            )
            uncond_mask = torch.zeros_like(attention_mask, dtype=torch.bool)

            for b in range(B_orig):
                s_idx = start_idx[b].item()
                keep_len = L - s_idx
                uncond_discrete[b, L - keep_len :] = discrete_sequence[b, s_idx:]
                uncond_mask[b, L - keep_len :] = attention_mask[b, s_idx:]

            input_embs = torch.cat(
                [embed_layer(discrete_sequence), embed_layer(uncond_discrete)], dim=0
            )
            attention_mask = torch.cat([attention_mask, uncond_mask], dim=0)
            if self.backbone_voice_condition:
                speaker_for_backbone = (
                    torch.cat([speaker_embedding, speaker_embedding], dim=0)
                    if speaker_embedding is not None
                    else None
                )
                input_embs = self._add_backbone_voice_condition(
                    input_embs=input_embs,
                    attention_mask=attention_mask,
                    speaker_embedding=speaker_for_backbone,
                )
        else:
            input_embs = embed_layer(discrete_sequence)
            if self.backbone_voice_condition:
                input_embs = self._add_backbone_voice_condition(
                    input_embs=input_embs,
                    attention_mask=attention_mask,
                    speaker_embedding=speaker_embedding,
                )
        position_ids = self._make_position_ids(attention_mask)

        past_key_values = None
        active_indices = torch.arange(
            B_orig, dtype=torch.long, device=discrete_sequence.device
        )
        active_speaker_embedding = speaker_embedding
        discrete_outputs = [[] for _ in range(B_orig)]
        continuous_outputs = [[] for _ in range(B_orig)]

        for step in tqdm(range(max_steps)):
            B_active = active_indices.numel()
            last_hidden_state, past_key_values = self.backbone.inference_forward(
                inputs_embeds=input_embs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            if do_cfg:
                cond_hidden = last_hidden_state[:B_active]
                uncond_hidden = last_hidden_state[B_active:]
                token_logits = self.get_token_logits(cond_hidden.squeeze(1))
                # token_logits = self.get_token_logits(last_hidden_state.squeeze(1))
                diffusion_context = (cond_hidden, uncond_hidden)
            else:
                token_logits = self.get_token_logits(last_hidden_state.squeeze(1))
                diffusion_context = last_hidden_state

            sampled_id = self._sample_token_ids(token_logits, kwargs.get("temperature"))
            eos_mask = sampled_id == 0  # EOS token is assumed to be 0

            if self.continuous and self.diffusion_head is not None:
                if step >= self.shift_audio_offset and (
                    not eos_mask.all() or self.shift_audio_offset > 0
                ):
                    generation_kwargs = dict(
                        num_steps=kwargs.get("num_steps"),
                        context_vector=cond_hidden if do_cfg else diffusion_context,
                        temperature=kwargs.get("diffusion_temperature"),
                        guidance_scale=guidance_scale,
                        generator=kwargs.get("generator", None),
                        speaker_embedding=(
                            active_speaker_embedding
                            if self.diffusion_voice_condition
                            else None
                        ),
                    )
                    generated_continuous_tokens = self.diffusion_head.generate(
                        **generation_kwargs
                    ).audio_features

                    for local_idx, original_idx in enumerate(active_indices.tolist()):
                        continuous_outputs[original_idx].append(
                            generated_continuous_tokens[local_idx : local_idx + 1]
                        )

                    generated_continuous_tokens = self.norm_continuous(
                        self.continuous_adapter(generated_continuous_tokens)
                    ).detach()
                    if getattr(self, "continuous_scale", None) is not None:
                        generated_continuous_tokens = (
                            generated_continuous_tokens * self.continuous_scale
                        )
                else:
                    generated_continuous_tokens = 0.0

            for local_idx, original_idx in enumerate(active_indices.tolist()):
                if not eos_mask[local_idx]:
                    discrete_outputs[original_idx].append(sampled_id[local_idx] - 1)

            if eos_mask.all():
                break

            survivor_indices = (~eos_mask).nonzero(as_tuple=False).squeeze(-1)
            token_id = (
                sampled_id.index_select(0, survivor_indices)
                - 1
                + self.tokenizer.prompt_vocab_size
            )

            if not self.continuous or self.diffusion_head is None:
                next_token = embed_layer(token_id).unsqueeze(1)  # (B, H) → (B, 1, H)
            else:
                survivor_feedback = generated_continuous_tokens
                if isinstance(generated_continuous_tokens, torch.Tensor):
                    survivor_feedback = generated_continuous_tokens.index_select(
                        0, survivor_indices
                    )
                next_token = embed_layer(token_id).unsqueeze(1) + survivor_feedback

            if self.backbone_voice_condition and active_speaker_embedding is not None:
                next_speaker = active_speaker_embedding.index_select(
                    0, survivor_indices
                )
                next_token = next_token + self._speaker_hidden(next_speaker).unsqueeze(
                    1
                ).to(dtype=next_token.dtype)

            if do_cfg:
                next_token = next_token.repeat(2, 1, 1)
                cache_indices = torch.cat(
                    [survivor_indices, survivor_indices + B_active], dim=0
                )
            else:
                cache_indices = survivor_indices

            if past_key_values is not None:
                if hasattr(past_key_values, "batch_select_indices"):
                    past_key_values.batch_select_indices(cache_indices)
                else:
                    past_key_values = tuple(
                        tuple(state.index_select(0, cache_indices) for state in layer)
                        for layer in past_key_values
                    )

            attention_mask = attention_mask.index_select(0, cache_indices)
            sampled_mask = torch.ones(
                (cache_indices.numel(), 1),
                dtype=torch.bool,
                device=sampled_id.device,
            )
            attention_mask = torch.cat([attention_mask, sampled_mask], dim=1)
            position_ids = (attention_mask.long().sum(dim=1) - 1).unsqueeze(-1)

            if self.config.backbone_config.model_type == "native":
                input_embs = torch.cat(
                    [input_embs.index_select(0, cache_indices), next_token], dim=1
                )
            else:
                input_embs = next_token

            active_indices = active_indices.index_select(0, survivor_indices)
            if active_speaker_embedding is not None:
                active_speaker_embedding = active_speaker_embedding.index_select(
                    0, survivor_indices
                )

        generated_lengths = torch.tensor(
            [len(tokens) for tokens in discrete_outputs],
            dtype=torch.long,
            device=discrete_sequence.device,
        )
        max_discrete_len = int(generated_lengths.max().item())
        final_discrete = torch.full(
            (B_orig, max_discrete_len, 1),
            -1,
            dtype=torch.long,
            device=discrete_sequence.device,
        )
        for b, tokens in enumerate(discrete_outputs):
            if tokens:
                final_discrete[b, : len(tokens), 0] = torch.stack(tokens)

        max_continuous_len = max(len(tokens) for tokens in continuous_outputs)
        if max_continuous_len > 0:
            first_continuous = next(
                tokens[0] for tokens in continuous_outputs if tokens
            )
            final_continuous = torch.zeros(
                (B_orig, max_continuous_len, first_continuous.shape[-1]),
                dtype=first_continuous.dtype,
                device=discrete_sequence.device,
            )
            for b, tokens in enumerate(continuous_outputs):
                if tokens:
                    sample_continuous = torch.cat(tokens, dim=1).squeeze(0)
                    final_continuous[b, : sample_continuous.shape[0]] = (
                        sample_continuous
                    )
            final_continuous = self.dynamic_normalizer.denormalize(final_continuous)
        else:
            final_continuous = None

        return {
            "discrete_tokens": final_discrete,
            "continuous_tokens": final_continuous,
            "discrete_lengths": generated_lengths,
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
