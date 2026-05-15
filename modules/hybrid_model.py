import torch
import torch.nn as nn
from typing import Dict
from .diffusion_head.cfm import DiT
from .configs import HybridTTSConfig
from transformers import AutoModel, AutoConfig
from .output_dataclasses import HybridTTSOutput, DecoderOutput


class DynamicNormalizer(nn.Module):
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x, padding_mask=None):
        # x: (B, L, C)
        if self.training:
            # Compute batch mean and var, considering padding if provided
            if padding_mask is not None:
                # mask is True for padding, so ~mask is True for data
                mask = (~padding_mask).to(x.dtype)  # (B, L)
                mask = mask.unsqueeze(-1)  # (B, L, 1)
                count = mask.sum()
                if count > 0:
                    batch_mean = (x * mask).sum(dim=(0, 1)) / count
                    batch_var = ((x - batch_mean) ** 2 * mask).sum(dim=(0, 1)) / count
                else:
                    batch_mean = x.mean(dim=(0, 1))
                    batch_var = x.var(dim=(0, 1), unbiased=False)
            else:
                batch_mean = x.mean(dim=(0, 1))
                batch_var = x.var(dim=(0, 1), unbiased=False)

            # Update running stats
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
        for i in range(config.num_layers - 1):
            layers.append(nn.Linear(curr_dim, config.hidden_dim))
            layers.append(nn.SiLU())
            curr_dim = config.hidden_dim
        layers.append(nn.Linear(curr_dim, config.out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HybridTTS(nn.Module):
    def __init__(self, config: HybridTTSConfig):
        super().__init__()
        self.config = config

        # Backbone
        bb_cfg = config.backbone_config
        if bb_cfg.pretrained:
            self.backbone = AutoModel.from_pretrained(bb_cfg.model_name_or_path)
        else:
            hf_config = AutoConfig.from_pretrained(bb_cfg.model_name_or_path)
            # When training from scratch, we use external LUTs.
            # Minimize backbone's internal embedding parameters to save memory.
            hf_config.vocab_size = 1
            self.backbone = AutoModel.from_config(hf_config)

        hidden_size = self.backbone.config.hidden_size

        # Override config sizes based on actual backbone hidden size
        self.config.backbone_hidden_size = hidden_size
        self.config.token_head_config.in_dim = hidden_size
        if self.config.continuous_adapter_config is not None:
            self.config.continuous_adapter_config.out_dim = hidden_size
        self.config.diffusion_head_config.backbone_dim = hidden_size

        # Separate LUTs for Prompt and Discrete Audio Tokens
        self.prompt_emb = nn.Embedding(config.prompt_vocab_size, hidden_size)
        self.discrete_emb = nn.Embedding(config.discrete_token_vocab_size, hidden_size)

        # Optional Continuous Adapter
        if config.continuous_adapter_config is not None:
            self.continuous_adapter = MLPAdapter(config.continuous_adapter_config)
        else:
            self.continuous_adapter = nn.Linear(config.continuous_dim, hidden_size)

        # Normalizations
        self.continuous_norm = DynamicNormalizer(config.continuous_dim)
        self.norm_discrete = nn.LayerNorm(hidden_size)
        self.norm_continuous = nn.LayerNorm(hidden_size)

        # Output Heads
        self.token_head = nn.Linear(hidden_size, config.discrete_token_vocab_size)
        self.diffusion_head = DiT(config.diffusion_head_config)

    def forward(
        self,
        prompt_ids: torch.Tensor,
        discrete_tokens: torch.Tensor,
        continuous_tokens: torch.Tensor,
        prompt_mask: torch.BoolTensor = None,
        padding_mask: torch.BoolTensor = None,
        target_continuous: torch.Tensor = None,
        **kwargs,
    ):
        """
        Forward pass for training.
        """

        # 1. Prompt Embeddings
        p_emb = self.prompt_emb(prompt_ids + self.config.prompt_offset)
        B, L_prompt, _ = p_emb.shape

        # 2. Input Representations
        d_emb = self.discrete_emb(discrete_tokens)

        # Normalize continuous tokens before adapter
        norm_continuous_tokens = self.continuous_norm(
            continuous_tokens, padding_mask=padding_mask
        )
        c_emb = self.continuous_adapter(norm_continuous_tokens)

        # 3. Normalization
        d_emb = self.norm_discrete(d_emb)
        c_emb = self.norm_continuous(c_emb)

        # Combine inputs for the audio part
        audio_emb = d_emb + c_emb
        L_audio = audio_emb.shape[1]

        # 4. Special Tokens (using backbone IDs)
        start_ids = torch.full(
            (B, 1),
            self.config.start_audio_id,
            device=prompt_ids.device,
            dtype=torch.long,
        )
        end_ids = torch.full(
            (B, 1), self.config.end_audio_id, device=prompt_ids.device, dtype=torch.long
        )
        start_emb = self.prompt_emb(start_ids)
        end_emb = self.prompt_emb(end_ids)

        # 5. Concatenate Sequence: [Prompt, START, Audio, END]
        inputs_embeds = torch.cat([p_emb, start_emb, audio_emb, end_emb], dim=1)

        # 6. Attention Mask
        if prompt_mask is None:
            prompt_mask = torch.ones(
                (B, L_prompt), dtype=torch.bool, device=p_emb.device
            )
        if padding_mask is None:
            padding_mask = torch.zeros(
                (B, L_audio), dtype=torch.bool, device=audio_emb.device
            )

        # In HF, attention_mask: 1 for attend, 0 for ignore
        prompt_attn = prompt_mask.long()
        start_attn = torch.ones((B, 1), dtype=torch.long, device=p_emb.device)
        audio_attn = (~padding_mask).long()
        end_attn = torch.ones((B, 1), dtype=torch.long, device=p_emb.device)

        attention_mask = torch.cat(
            [prompt_attn, start_attn, audio_attn, end_attn], dim=1
        )

        # 7. Backbone Pass
        # Cast to backbone's actual parameter dtype to handle both AMP and full-bf16.
        bb_dtype = next(self.backbone.parameters()).dtype
        outputs = self.backbone(
            inputs_embeds=inputs_embeds.to(bb_dtype), attention_mask=attention_mask
        )
        full_hidden_states = outputs.last_hidden_state

        # 8. Slicing Audio Hidden States for the Heads
        # Autoregressive setup: Hidden state at position t predicts token at t+1.
        # Position L_prompt is the START token, which predicts the first audio token.
        # Position L_prompt + L_audio - 1 is the penultimate audio token, which predicts the last audio token.
        audio_hidden_states = full_hidden_states[:, L_prompt : L_prompt + L_audio, :]

        # 9. Output Heads
        # Discrete Token Head
        token_logits = self.token_head(audio_hidden_states)

        # Diffusion Head (Continuous)
        # target_continuous is the actual continuous representation from VAE
        if target_continuous is None:
            target_continuous = continuous_tokens  # Fallback if not provided separately

        # Normalize targets for the diffusion head
        target_continuous = self.continuous_norm(
            target_continuous, padding_mask=padding_mask
        )

        diffusion_output = self.diffusion_head(
            target=target_continuous,
            target_padding_mask=(
                padding_mask
                if padding_mask is not None
                else torch.zeros(
                    target_continuous.shape[:2],
                    dtype=torch.bool,
                    device=target_continuous.device,
                )
            ),
            context_vector=audio_hidden_states,
        )

        return HybridTTSOutput(
            token_logits=token_logits,
            diffusion_loss=diffusion_output.loss,
            diffusion_output=diffusion_output,
        )

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
        Reconstruct audio features from ground truth tokens using the model's diffusion head.
        This is used for evaluation purposes.
        """
        self.eval()

        prompt_ids = batch["prompt_ids"]
        discrete_tokens = batch["discrete_tokens"]
        continuous_tokens = batch["continuous_tokens"]
        padding_mask = batch["padding_mask"]

        # 1. Forward pass to get context hidden states
        # continuous_tokens will be cast to model dtype inside forward.
        outputs = self.forward(
            prompt_ids=prompt_ids,
            discrete_tokens=discrete_tokens,
            continuous_tokens=continuous_tokens,
            padding_mask=padding_mask,
        )

        # 2. The diffusion head was already run in forward, but we might want to
        # run it with custom parameters (num_steps, guidance_scale) for evaluation
        # The hidden states for the heads are internal to forward,
        # so we might need to extract them or re-run parts.

        # To avoid code duplication, we'll re-run the backbone pass or extract hidden states.
        # Let's re-run the generation part of the diffusion head with the hidden states from forward.
        # However, forward doesn't return them.
        # A cleaner way is to have a method that returns hidden states.

        # For simplicity in this task, let's just use the diffusion_head.generate directly
        # with the context_vector that we expect.

        # Re-calculate embeddings (same as forward).
        # Cast continuous_tokens once here to match model dtype.
        continuous_tokens = continuous_tokens.to(self.dtype)
        p_emb = self.prompt_emb(prompt_ids + self.config.prompt_offset)
        d_emb = self.discrete_emb(discrete_tokens)
        norm_c = self.continuous_norm(continuous_tokens, padding_mask=padding_mask)
        c_emb = self.continuous_adapter(norm_c)
        d_emb = self.norm_discrete(d_emb)
        c_emb = self.norm_continuous(c_emb)
        audio_emb = d_emb + c_emb

        B = prompt_ids.shape[0]
        L_prompt = p_emb.shape[1]
        L_audio = audio_emb.shape[1]

        start_ids = torch.full(
            (B, 1), self.config.start_audio_id, device=self.device, dtype=torch.long
        )
        start_emb = self.prompt_emb(start_ids)
        inputs_embeds = torch.cat(
            [p_emb, start_emb, audio_emb], dim=1
        )  # Omit end_emb for simplicity

        prompt_attn = torch.ones((B, L_prompt), dtype=torch.long, device=self.device)
        start_attn = torch.ones((B, 1), dtype=torch.long, device=self.device)
        audio_attn = (~padding_mask).long()
        attention_mask = torch.cat([prompt_attn, start_attn, audio_attn], dim=1)

        bb_dtype = next(self.backbone.parameters()).dtype
        outputs_bb = self.backbone(
            inputs_embeds=inputs_embeds.to(bb_dtype), attention_mask=attention_mask
        )
        full_hidden_states = outputs_bb.last_hidden_state

        # Autoregressive context for audio heads
        audio_hidden_states = full_hidden_states[:, L_prompt : L_prompt + L_audio, :]

        # 1. Generate reconstructed continuous latents using our diffusion head
        latents_output = self.diffusion_head.generate(
            num_steps=num_steps,
            context_vector=audio_hidden_states,
            temperature=temperature,
            guidance_scale=guidance_scale,
            padding_mask=padding_mask,
        )
        z = latents_output.audio_features  # [B, L, C]

        # 2. Use these generated latents as context for the VAE to generate Mel
        reconstructed_mel, reconstructed_padding_mask = vae.sample(
            num_steps=num_steps,
            temperature=temperature,
            guidance_scale=guidance_scale,
            z=z,
            padding_mask=latents_output.padding_mask,
        )

        return {
            "decoder_output": DecoderOutput(
                audio_features=reconstructed_mel,
                padding_mask=reconstructed_padding_mask,
            ),
            "token_logits": self.token_head(audio_hidden_states),
        }

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
