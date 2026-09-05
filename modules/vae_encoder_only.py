import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.submodules.MelCausalVAE.dicodec.modules.configs import (
    DicodecConfig,
    DiTConfig,
    DropoutConfig,
    EncoderConfig,
    ExternalSemanticQuantizerConfig,
    KLChunkRegularizer,
    LowPassFilterConfig,
    MelSpectrogramConfig,
    NoiseConfig,
    SpeakerEncoderConfig,
    WavLMConfig,
    WavLMModuleConfig,
)
from modules.submodules.MelCausalVAE.dicodec.modules.encoder.encoder import Encoder
from modules.submodules.MelCausalVAE.dicodec.modules.feature_extractor import (
    FeatureExtractor,
    WavLMFeatureExtractor,
)
from modules.submodules.MelCausalVAE.dicodec.modules.lp_filter import LowPassFilter
from modules.submodules.MelCausalVAE.dicodec.modules.semantic_quantizer_ae import (
    load_semantic_quantizer_ae,
    read_semantic_quantizer_config,
)
from modules.submodules.MelCausalVAE.dicodec.modules.speaker_encoder import (
    WavLMSpeakerEncoder,
)


def _filter_dataclass_kwargs(config_cls, values: Optional[dict]) -> dict:
    if not values:
        return {}
    allowed = {field.name for field in fields(config_cls)}
    return {key: value for key, value in values.items() if key in allowed}


class DicodecEncoderOnly(nn.Module):
    def __init__(self, config: DicodecConfig):
        super().__init__()
        self.config = config
        self.feature_extractor = FeatureExtractor(config.mel_spectrogram_config)

        self.wavlm = None
        self.wavlm_extractor = None
        self.speaker_encoder = None
        if config.wavlm_module_config is not None:
            from transformers import WavLMModel

            self.wavlm = WavLMModel.from_pretrained(
                config.wavlm_module_config.pretrained_model_name,
                use_safetensors=False,
            )
            self.wavlm.eval()
            for parameter in self.wavlm.parameters():
                parameter.requires_grad_(False)

            feature_extractor_config = (
                config.wavlm_module_config.feature_extractor_config
            )
            if feature_extractor_config is not None:
                self.wavlm_extractor = WavLMFeatureExtractor(
                    feature_extractor_config,
                    wavlm=self.wavlm,
                )
            speaker_encoder_config = config.wavlm_module_config.speaker_encoder_config
            if speaker_encoder_config is not None:
                self.speaker_encoder = WavLMSpeakerEncoder(
                    speaker_encoder_config,
                    wavlm=self.wavlm,
                )

        self.encoder = Encoder(config.encoder_config)
        self.lowpass_filter = LowPassFilter(
            cutoff_hz=config.lowpass_filter_config.cutoff_hz,
            sample_rate=config.lowpass_filter_config.sample_rate,
            order=config.lowpass_filter_config.order,
        )
        self.external_semantic_quantizer = None
        self.external_semantic_quantizer_target = "z_sem"

    def train(self, mode: bool = True):
        super().train(mode)
        if self.wavlm is not None:
            self.wavlm.eval()
        if self.speaker_encoder is not None:
            self.speaker_encoder.eval()
        if self.external_semantic_quantizer is not None:
            self.external_semantic_quantizer.eval()
        return self

    @torch.no_grad()
    def extract_speaker_embedding(self, audios_srs):
        if self.speaker_encoder is None:
            return None
        self.speaker_encoder = self.speaker_encoder.to(device=self.device)
        return self.speaker_encoder(audios_srs).to(
            device=self.device,
            dtype=self.dtype,
        )

    @torch.no_grad()
    def extract_wavlm_features(
        self,
        target_length: int,
        audios_srs,
        audio_16khz=None,
        extractor: Optional[WavLMFeatureExtractor] = None,
    ):
        extractor = self.wavlm_extractor if extractor is None else extractor
        if extractor is None:
            return None, None

        wavlm_output = extractor(audios_srs, audio_16khz=audio_16khz)
        wavlm_features = wavlm_output.audio_features.to(self.dtype)
        wavlm_features = wavlm_features.repeat_interleave(2, dim=1)
        wavlm_features = (
            F.interpolate(
                wavlm_features.float().transpose(1, 2),
                size=target_length,
                mode="linear",
                align_corners=False,
            )
            .transpose(1, 2)
            .to(wavlm_features.dtype)
        )
        wavlm_padding_mask = (
            F.interpolate(
                wavlm_output.padding_mask.float().unsqueeze(1),
                size=target_length,
                mode="nearest",
            )
            .squeeze(1)
            .bool()
        )
        return wavlm_features, wavlm_padding_mask

    @torch.no_grad()
    def extract_features(self, audios_srs, target_audios_srs=None, **kwargs):
        target_audios_srs = audios_srs if target_audios_srs is None else target_audios_srs
        target_output = self.feature_extractor(target_audios_srs)
        target_features = target_output.audio_features.to(self.dtype)
        target_length = target_features.shape[1]

        wavlm_features, wavlm_padding_mask = self.extract_wavlm_features(
            target_length=target_length,
            audios_srs=audios_srs,
            audio_16khz=kwargs.get("audio_16khz"),
        )
        if wavlm_features is not None:
            return wavlm_features, wavlm_padding_mask, target_features, target_output.padding_mask

        encoder_output = self.feature_extractor(audios_srs)
        return (
            encoder_output.audio_features.to(self.dtype),
            encoder_output.padding_mask,
            target_features,
            target_output.padding_mask,
        )

    @torch.no_grad()
    def encode(self, features, padding_mask, **kwargs):
        encoder_output = self.encoder(
            x=features,
            padding_mask=padding_mask,
            step=kwargs.get("training_step", None),
        )
        if kwargs.get("run_quantizer", True) and self.external_semantic_quantizer is not None:
            encoder_output.quantizer_output = self.quantize(
                encoder_output.z,
                padding_mask=encoder_output.padding_mask,
            )
        return encoder_output

    @torch.no_grad()
    def encode_attributes(self, z, padding_mask=None):
        if padding_mask is not None:
            if padding_mask.shape != z.shape[:2]:
                raise ValueError(
                    "padding_mask must have shape [batch, time], got "
                    f"{tuple(padding_mask.shape)} for z shape {tuple(z.shape)}."
                )
            valid_mask = (~padding_mask).to(device=z.device, dtype=z.dtype).unsqueeze(-1)
            valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            z_mean = (z * valid_mask).sum(dim=1, keepdim=True) / valid_count
        else:
            valid_mask = None
            z_mean = z.mean(dim=1, keepdim=True)

        z_centered = z - z_mean
        if valid_mask is not None:
            z_centered = z_centered * valid_mask

        z_lp = self.lowpass_filter(z_centered, valid_mask=valid_mask)
        z_hp = z_centered - z_lp
        beta = torch.sum(z_centered * z_lp, dim=1, keepdim=True) / (
            torch.sum(z_lp.square(), dim=1, keepdim=True) + 1e-8
        )
        z_pros = beta * z_lp
        z_sem = z_centered - z_pros

        if valid_mask is not None:
            z_hp = z_hp * valid_mask
            z_pros = z_pros * valid_mask
            z_sem = z_sem * valid_mask

        from modules.submodules.MelCausalVAE.dicodec.modules.output_dataclasses import (
            AttributesOutput,
        )

        return AttributesOutput(
            z_sem=z_sem,
            z_pros=z_pros,
            z_mean=z_mean,
            z_lp=z_lp,
            z_hp=z_hp,
        )

    def set_external_semantic_quantizer(
        self,
        quantizer: torch.nn.Module,
        target_source: Optional[str] = None,
    ):
        target_source = target_source or "z_sem"
        if target_source not in {"z", "z_sem"}:
            raise ValueError("target_source must be either 'z' or 'z_sem'.")
        quantizer.to(device=self.device, dtype=self.dtype)
        quantizer.eval()
        for parameter in quantizer.parameters():
            parameter.requires_grad_(False)
        self.external_semantic_quantizer = quantizer
        self.external_semantic_quantizer_target = target_source
        self.config.external_semantic_quantizer_config.enabled = True
        self.config.external_semantic_quantizer_config.target_source = target_source
        return quantizer

    @torch.no_grad()
    def quantize(self, z, padding_mask=None, return_attributes=False):
        if self.external_semantic_quantizer is None:
            raise RuntimeError("No external semantic quantizer is loaded.")

        from modules.submodules.MelCausalVAE.dicodec.modules.output_dataclasses import (
            QuantizeOutput,
        )

        attrs = self.encode_attributes(z, padding_mask=padding_mask)
        valid_mask = ~padding_mask if padding_mask is not None else None
        target_source = self.external_semantic_quantizer_target
        quantizer_input = attrs.z_sem if target_source == "z_sem" else z
        ae_out = self.external_semantic_quantizer(
            quantizer_input,
            valid_mask=valid_mask,
        )
        quantized = ae_out.z_rec

        if target_source == "z":
            residual = z - quantized
            z_pros = None
        elif target_source == "z_sem":
            residual = attrs.z_sem - quantized
            z_pros = attrs.z_pros + attrs.z_mean
        else:
            raise ValueError("external semantic quantizer target_source must be 'z' or 'z_sem'.")

        return QuantizeOutput(
            quantized=quantized,
            indices=ae_out.indices,
            residual=residual,
            z_pros=z_pros,
            attributes=attrs if return_attributes else None,
        )

    @property
    def dtype(self):
        return next(self.encoder.parameters()).dtype

    @property
    def device(self):
        return next(self.encoder.parameters()).device


def _build_config(cfg_dict: dict) -> DicodecConfig:
    encoder_cfg = cfg_dict.get("encoder_config", cfg_dict.get("encoder", {})).copy()
    decoder_cfg = cfg_dict.get("decoder_config", cfg_dict.get("decoder", {})).copy()
    mel_spec_cfg = cfg_dict.get("mel_spectrogram_config", {}).copy()

    encoder_cfg.setdefault("use_reparameterization_trick", False)
    encoder_cfg.setdefault("use_std_sweep", False)
    decoder_cfg.setdefault("mel_dim", cfg_dict.get("mel_dim"))
    decoder_cfg.setdefault("audio_latent_dim", cfg_dict.get("latent_dim"))
    decoder_cfg.setdefault("expansion_factor", cfg_dict.get("compress_factor"))
    decoder_cfg.setdefault("upsample", cfg_dict.get("upsample"))
    decoder_cfg.setdefault("local_speaker_conditioning", True)
    decoder_cfg.setdefault("normalize_context_vector", False)

    dropout_dict = encoder_cfg.pop("dropout_regularizer_config", None)
    kl_dict = encoder_cfg.pop("kl_chunk_regularizer_config", None)
    noise_dict = encoder_cfg.pop("noise_regularizer_config", None)
    encoder_config = EncoderConfig(
        dropout_regularizer_config=DropoutConfig(
            **_filter_dataclass_kwargs(DropoutConfig, dropout_dict)
        )
        if dropout_dict
        else None,
        kl_chunk_regularizer_config=KLChunkRegularizer(
            **_filter_dataclass_kwargs(KLChunkRegularizer, kl_dict)
        )
        if kl_dict
        else None,
        noise_regularizer_config=NoiseConfig(
            **_filter_dataclass_kwargs(NoiseConfig, noise_dict)
        )
        if noise_dict
        else None,
        **_filter_dataclass_kwargs(EncoderConfig, encoder_cfg),
    )
    decoder_config = DiTConfig(**_filter_dataclass_kwargs(DiTConfig, decoder_cfg))
    mel_spec_config = MelSpectrogramConfig(
        **_filter_dataclass_kwargs(MelSpectrogramConfig, mel_spec_cfg)
    )

    wavlm_module_dict = cfg_dict.get("wavlm_module_config", None)
    if wavlm_module_dict:
        feature_extractor_dict = wavlm_module_dict.get("feature_extractor_config", None)
        speaker_encoder_dict = wavlm_module_dict.get("speaker_encoder_config", None)
        wavlm_module_config = WavLMModuleConfig(
            pretrained_model_name=wavlm_module_dict.get("pretrained_model_name"),
            feature_extractor_config=WavLMConfig(
                **_filter_dataclass_kwargs(WavLMConfig, feature_extractor_dict)
            )
            if feature_extractor_dict
            else None,
            speaker_encoder_config=SpeakerEncoderConfig(
                **_filter_dataclass_kwargs(SpeakerEncoderConfig, speaker_encoder_dict)
            )
            if speaker_encoder_dict
            else None,
        )
    else:
        wavlm_module_config = None

    lowpass_filter_dict = cfg_dict.get(
        "lowpass_filter_config",
        cfg_dict.get("lowpass_filter", None),
    )
    if lowpass_filter_dict and "kernel_size" in lowpass_filter_dict:
        lowpass_filter_dict = lowpass_filter_dict.copy()
        lowpass_filter_dict.setdefault("order", lowpass_filter_dict["kernel_size"] - 1)
        lowpass_filter_dict.pop("kernel_size", None)
    lowpass_filter_config = (
        LowPassFilterConfig(
            **_filter_dataclass_kwargs(LowPassFilterConfig, lowpass_filter_dict)
        )
        if lowpass_filter_dict
        else LowPassFilterConfig()
    )

    external_quantizer_dict = cfg_dict.get(
        "external_semantic_quantizer_config",
        cfg_dict.get("external_semantic_quantizer", None),
    )
    external_quantizer_config = (
        ExternalSemanticQuantizerConfig(
            **_filter_dataclass_kwargs(
                ExternalSemanticQuantizerConfig,
                external_quantizer_dict,
            )
        )
        if external_quantizer_dict
        else ExternalSemanticQuantizerConfig()
    )

    return DicodecConfig(
        mel_dim=cfg_dict.get("mel_dim"),
        latent_dim=cfg_dict.get("latent_dim"),
        sample_rate=cfg_dict.get("sample_rate"),
        compress_factor=cfg_dict.get("compress_factor"),
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        mel_spectrogram_config=mel_spec_config,
        wavlm_module_config=wavlm_module_config,
        lowpass_filter_config=lowpass_filter_config,
        external_semantic_quantizer_config=external_quantizer_config,
    )


def load_encoder_only(
    checkpoint_dir: str,
    device: torch.device,
    semantic_quantizer_checkpoint: Optional[str] = None,
    semantic_quantizer_type: str = "std_vq",
    semantic_codebook_size: Optional[int] = None,
    semantic_quantizer_source: Optional[str] = None,
) -> DicodecEncoderOnly:
    with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
        cfg_dict = json.load(f)
    model = DicodecEncoderOnly(_build_config(cfg_dict))

    prefixes = (
        "feature_extractor.",
        "wavlm.",
        "wavlm_extractor.",
        "speaker_encoder.",
        "encoder.",
        "lowpass_filter.",
    )
    model_state = model.state_dict()

    def should_load_key(key: str, value: torch.Tensor) -> bool:
        return (
            key.startswith(prefixes)
            and key in model_state
            and model_state[key].shape == value.shape
        )

    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    pt_path = os.path.join(checkpoint_dir, "model.pt")
    if os.path.exists(safetensors_path):
        from safetensors import safe_open

        state_dict = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.startswith(prefixes):
                    continue
                value = f.get_tensor(key)
                if should_load_key(key, value):
                    state_dict[key] = value
    elif os.path.exists(pt_path):
        raw_state = torch.load(pt_path, map_location="cpu")
        if "state_dict" in raw_state:
            raw_state = raw_state["state_dict"]
        state_dict = {
            key: value
            for key, value in raw_state.items()
            if should_load_key(key, value)
        }
    else:
        raise FileNotFoundError(f"No model.safetensors or model.pt in {checkpoint_dir}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    required_missing = [
        key
        for key in missing
        if key.startswith(("encoder.", "feature_extractor.", "wavlm_extractor."))
        or (model.speaker_encoder is not None and key.startswith("speaker_encoder."))
    ]
    if required_missing:
        raise RuntimeError(
            "Missing encoder-only checkpoint weights: "
            + ", ".join(required_missing[:8])
        )
    model.to(device)
    if semantic_quantizer_checkpoint:
        quantizer_config = read_semantic_quantizer_config(semantic_quantizer_checkpoint)
        target_source = (
            semantic_quantizer_source
            or quantizer_config.get("target_source")
            or quantizer_config.get("input_source")
            or "z_sem"
        )
        quantizer_type = quantizer_config.get(
            "quantizer_type",
            semantic_quantizer_type,
        )
        codebook_size = quantizer_config.get(
            "codebook_size",
            quantizer_config.get(
                "num_embeddings",
                quantizer_config.get("num_codebooks", semantic_codebook_size),
            ),
        )
        quantizer = load_semantic_quantizer_ae(
            checkpoint_path=semantic_quantizer_checkpoint,
            latent_dim=model.encoder.config.latent_dim,
            quantizer_type=quantizer_type,
            codebook_size=codebook_size,
            device=model.device,
        )
        model.set_external_semantic_quantizer(quantizer, target_source=target_source)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
