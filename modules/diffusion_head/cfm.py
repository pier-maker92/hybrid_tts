import torch
import random
import torch.nn as nn
from einops import rearrange
from .dit import Transformer
from ..configs import DiTConfig
from torchdiffeq import odeint
from typing import Optional, Tuple
from ..output_dataclasses import DecoderOutput
from .mlp import MLP, LearnedSinusoidalPosEmb


class DiT(torch.nn.Module):
    def __init__(
        self,
        config: DiTConfig,
    ):
        super().__init__()

        # inherit from config
        self.sigma = config.sigma
        self.audio_latent_dim = config.audio_latent_dim
        self.net_dim = config.net_dim
        self.net_heads = config.net_heads
        self.net_depth = config.net_depth
        self.backbone_dim = config.backbone_dim
        self.expansion_factor = 1  # FIXME - expansion_factor is not used in this module
        self.uncond_prob = config.uncond_prob

        # transformer scenario
        self.is_causal = config.is_causal
        self.use_conv_layer = config.use_conv_layer
        self.use_window_attention = config.use_window_attention
        self.use_group_bidirectional = config.use_group_bidirectional
        self.use_mlp_sampler = config.use_mlp_sampler
        self.use_ecapa_film = getattr(config, "use_ecapa_film", False)
        self.ecapa_dim = getattr(config, "ecapa_dim", 192)

        latent_fps = 12.5  # 24000 / 256 #FIXME hardcoded to 24kHz dataset
        self.window_size = (
            int(config.window_attention_seconds * latent_fps)
            if config.use_window_attention
            else None
        )
        print(f"VAE is_causal: {self.is_causal}")
        if self.window_size is not None:
            print(
                f"VAE window_attention: {config.window_attention_seconds}s = {self.window_size} mel frames"
            )
        if self.use_group_bidirectional:
            print(
                f"VAE group_bidirectional: enabled (group_size will be set to expansion_factor)"
            )

        # context
        self.context_vector_proj = nn.Sequential(
            nn.Linear(self.backbone_dim, self.net_dim),
            nn.SiLU(),
            nn.Linear(self.net_dim, self.net_dim),
            nn.LayerNorm(self.net_dim),
        )

        # noise projection
        self.noise_proj = nn.Sequential(
            nn.Linear(self.net_dim + self.audio_latent_dim, self.net_dim),
            nn.LayerNorm(self.net_dim),
        )

        # net
        # TODO add a dedicated config
        if self.use_mlp_sampler:
            self.net = MLP(
                n_channels=self.net_dim,
                out_channels=self.audio_latent_dim,
                n_residual_blocks=self.net_depth,
            )
            self.sinu_pos_emb = LearnedSinusoidalPosEmb(self.net_dim)
        else:
            self.net = Transformer(
                dim=self.net_dim,
                depth=self.net_depth,
                out_dim=self.audio_latent_dim,
                heads=self.net_heads,
                use_conv_layer=self.use_conv_layer,
                is_causal=self.is_causal,
                attn_flash=True,
                window_size=self.window_size,
                use_ecapa_film=self.use_ecapa_film,
                ecapa_dim=self.ecapa_dim,
            )

    def handle_context_vector(
        self,
        context_vector: torch.FloatTensor,
        temperature: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
    ):

        # Original behavior: uniform repeat_interleave
        context_vector = context_vector.repeat_interleave(self.expansion_factor, dim=1)
        if padding_mask is not None:
            padding_mask = padding_mask.repeat_interleave(self.expansion_factor, dim=1)

        temperature = temperature or 1.0
        x0 = (
            torch.randn(
                context_vector.shape[0],
                context_vector.shape[1],
                self.audio_latent_dim,
                dtype=context_vector.dtype,
                device=context_vector.device,
                generator=generator,
            )
            * temperature
        )
        return self.context_vector_proj(context_vector), x0, padding_mask

    def prepare_times_mlp(
        self,
        context_vector_semantic: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        B, T = context_vector_semantic.shape[:2]

        times = torch.rand(
            (B, T),
            dtype=context_vector_semantic.dtype,
            device=context_vector_semantic.device,
        )
        mask = torch.rand_like(times) < 0.1
        times = times * (~mask).float().to(context_vector_semantic.dtype)

        t = rearrange(times, "b t -> b t 1")
        times = self.sinu_pos_emb(rearrange(times, "b t -> (b t)"))
        times = rearrange(times, "(b t) h -> b t h", b=B)
        return times, t

    def prepare_flow(
        self,
        target: torch.FloatTensor,
        context_vector: torch.FloatTensor,
        x0: torch.FloatTensor,
    ):
        if random.random() < self.uncond_prob:
            context_vector = torch.zeros_like(context_vector)
        # We need times
        if self.use_mlp_sampler:
            times, t = self.prepare_times_mlp(context_vector)
        else:
            times = torch.rand(
                (target.shape[0],),
                dtype=context_vector.dtype,
                device=context_vector.device,
            )
            t = rearrange(times, "b -> b 1 1")
        # Now we need noise, x0 sampled from a normal distribution
        x0 = torch.randn_like(target).to(context_vector.dtype)
        # w is the noise signal that is transformed by the flow
        w = ((1 - (1 - self.sigma) * t) * x0 + t * target).to(context_vector.dtype)
        # target is the original signal minus the noise
        target = (target - (1 - self.sigma) * x0).to(context_vector.dtype)
        state = self.noise_proj(torch.cat([context_vector, w], dim=-1))
        return state, times, target

    @property
    def _group_size(self) -> Optional[int]:
        if (
            self.use_group_bidirectional
            and self.expansion_factor
            and self.expansion_factor > 1
        ):
            return self.expansion_factor
        return None

    def let_it_flow(
        self,
        times: torch.FloatTensor,
        state: torch.FloatTensor,
        target: torch.FloatTensor,
        flow_mask: torch.BoolTensor,
        ecapa: Optional[torch.FloatTensor] = None,
    ):
        mask_to_loss = ~flow_mask
        v = self.net(
            x=state,
            times=times,
            attention_mask=mask_to_loss if not self.use_mlp_sampler else None,
            group_size=self._group_size if not self.use_mlp_sampler else None,
            ecapa=ecapa,
        )

        v_to_loss = v[mask_to_loss].view(-1, self.audio_latent_dim)
        target_to_loss = target[mask_to_loss].view(-1, self.audio_latent_dim)
        loss = ((v_to_loss - target_to_loss) ** 2).mean()

        return loss

    def forward(
        self,
        target: torch.FloatTensor,
        target_padding_mask: torch.BoolTensor,
        context_vector: torch.FloatTensor,
        ecapa: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        context_vector, prior, _ = self.handle_context_vector(context_vector)

        # ---- flow ----
        state, times, v_target = self.prepare_flow(
            target=target,
            context_vector=context_vector,
            x0=prior,
        )

        # ---- get the flow ----
        loss = self.let_it_flow(
            times=times,
            state=state,
            target=v_target,
            flow_mask=target_padding_mask,
            ecapa=ecapa,
        )

        return DecoderOutput(
            loss=loss,
        )

    def generate(
        self,
        num_steps: int,
        context_vector,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        ecapa: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        cfg_scale = guidance_scale

        if isinstance(context_vector, tuple):
            cond_context, uncond_context = context_vector
        else:
            cond_context = context_vector
            uncond_context = None

        # ---- context vector z ----
        cond_context, y0, upsampled_padding_mask = self.handle_context_vector(
            cond_context,
            temperature=temperature,
            generator=generator,
            padding_mask=padding_mask,
        )

        if uncond_context is not None:
            uncond_context, _, _ = self.handle_context_vector(
                uncond_context,
                temperature=temperature,
                generator=generator,
                padding_mask=padding_mask,
            )
            context_vector = (cond_context, uncond_context)
        else:
            context_vector = cond_context

        B, T = cond_context.shape[:2]

        # ---- time span ----
        t_span = torch.linspace(
            0, 1, num_steps, device=cond_context.device, dtype=cond_context.dtype
        )

        # ---- ODE ----
        def fn(t, state):
            features = self.cfg_forward(
                times=t,
                state=state,
                cfg_scale=cfg_scale,
                context_vector=context_vector,
                attention_mask=(
                    ~upsampled_padding_mask
                    if upsampled_padding_mask is not None
                    else None
                ),
                ecapa=ecapa,
            )
            return features

        odeint_kwargs = dict(atol=1e-5, rtol=1e-5, method="midpoint")
        trajectory = odeint(fn, y0, t_span, **odeint_kwargs)

        generated_latents = trajectory[-1]

        return DecoderOutput(
            audio_features=generated_latents.view(B, -1, self.audio_latent_dim),
            padding_mask=upsampled_padding_mask,
        )

    def cfg_forward(
        self,
        times: torch.FloatTensor,
        state: torch.FloatTensor,
        cfg_scale: float,
        context_vector,
        attention_mask: Optional[torch.BoolTensor] = None,
        ecapa: Optional[torch.FloatTensor] = None,
    ):
        times = times.repeat(state.shape[0])
        if self.use_mlp_sampler:
            times = self.sinu_pos_emb(times)
            times = rearrange(times, "(b t) h -> b t h", b=state.shape[0])

        if isinstance(context_vector, tuple):
            cond_context, uncond_context = context_vector
        else:
            cond_context = context_vector
            uncond_context = None

        cond_state = self.noise_proj(torch.cat([cond_context, state], dim=-1))
        gs = self._group_size
        cond_out = self.net(
            x=cond_state,
            times=times,
            attention_mask=attention_mask if not self.use_mlp_sampler else None,
            group_size=gs if not self.use_mlp_sampler else None,
            ecapa=ecapa,
        )
        if cfg_scale == 1.0:
            return cond_out

        if uncond_context is not None:
            uncond_state = self.noise_proj(torch.cat([uncond_context, state], dim=-1))
        else:
            uncond_state = self.noise_proj(
                torch.cat([torch.zeros_like(cond_context), state], dim=-1)
            )

        uncond_out = self.net(
            x=uncond_state,
            times=times,
            attention_mask=attention_mask if not self.use_mlp_sampler else None,
            group_size=gs if not self.use_mlp_sampler else None,
            ecapa=torch.zeros_like(ecapa) if ecapa is not None else None,
        )

        final = (cfg_scale * cond_out + (1 - cfg_scale) * uncond_out).to(
            cond_context.dtype
        )

        return final

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
