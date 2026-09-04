import json
import os
from typing import Any, Dict, Optional

import torch

from .hybrid_model import HybridTTS, HybridTokenizer
from .configs import (
    HybridTTSConfig,
    BackboneConfig,
    AdapterConfig,
    DiTConfig,
)


def _resolve_quantizer_checkpoint(cfg_dict: Dict[str, Any]) -> str | None:
    training_cfg = cfg_dict.get("training", {}) or {}
    external_cfg = (
        cfg_dict.get("external_semantic_quantizer_config")
        or cfg_dict.get("external_semantic_quantizer")
        or {}
    )
    checkpoint = (
        training_cfg.get("semantic_quantizer_checkpoint")
        or cfg_dict.get("semantic_quantizer_checkpoint")
        or external_cfg.get("checkpoint_path")
    )
    if checkpoint:
        checkpoint = checkpoint.replace(
            "$SCRATCH",
            os.environ.get("SCRATCH", "/Users/software/Research"),
        )
    return checkpoint


def _load_quantizer_vocab_size(cfg_dict: Dict[str, Any]) -> int:
    training_cfg = cfg_dict.get("training", {}) or {}
    external_cfg = (
        cfg_dict.get("external_semantic_quantizer_config")
        or cfg_dict.get("external_semantic_quantizer")
        or {}
    )
    configured_size = (
        training_cfg.get("semantic_codebook_size")
        or cfg_dict.get("semantic_codebook_size")
        or external_cfg.get("codebook_size")
    )
    checkpoint = _resolve_quantizer_checkpoint(cfg_dict)
    if checkpoint and os.path.isdir(checkpoint):
        config_path = os.path.join(checkpoint, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                quantizer_cfg = json.load(f)
            configured_size = (
                quantizer_cfg.get("codebook_size")
                or quantizer_cfg.get("num_embeddings")
                or quantizer_cfg.get("num_codebooks")
                or configured_size
            )
    return int(configured_size or 0)


def _load_quantizer_state_dict(checkpoint: str) -> dict[str, torch.Tensor]:
    model_path = (
        os.path.join(checkpoint, "model.pt")
        if os.path.isdir(checkpoint)
        else checkpoint
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing quantizer checkpoint: {model_path}")

    state_dict = torch.load(model_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def load_quantizer_embeddings(cfg_dict: Dict[str, Any]) -> Optional[torch.Tensor]:
    checkpoint = _resolve_quantizer_checkpoint(cfg_dict)
    if checkpoint is None:
        return None

    state_dict = _load_quantizer_state_dict(checkpoint)
    candidate_keys = (
        "quantizer.quantizer_module.embedding.weight",
        "quantizer.quantizer_module.embedding",
        "quantizer.quantizer_module.codebook",
    )
    for key in candidate_keys:
        value = state_dict.get(key)
        if value is not None:
            if value.ndim != 2:
                raise ValueError(
                    f"Quantizer embedding key {key} has invalid shape "
                    f"{tuple(value.shape)}."
                )
            return value.float()
    raise ValueError(
        "Could not find quantizer embeddings in checkpoint. Tried: "
        + ", ".join(candidate_keys)
    )


def load_codebook_config(vae_checkpoint: str, cfg_dict: Dict[str, Any] | None = None):
    config_path = os.path.join(vae_checkpoint, "config.json")
    with open(config_path, "r") as f:
        vae_cfg = json.load(f)

    continuous_dim = vae_cfg.get("latent_dim")
    vq_vocab_size = 0
    vq_config = vae_cfg.get("encoder_config", {}).get("vq_config", None)
    if vq_config is not None:
        dim_to_quantize = int(vq_config.get("dim_to_quantize", 0))
        vq_vocab_size = vq_config.get("num_embeddings")
        if not vq_config.get("add_vq_residual_to_stoch", False):
            continuous_dim = continuous_dim - dim_to_quantize
    elif cfg_dict is not None:
        vq_vocab_size = _load_quantizer_vocab_size(cfg_dict)

    return (
        continuous_dim,
        int(vq_vocab_size or 0),
    )


def load_codebook_config_from_cfg(cfg_dict: Dict[str, Any]):
    vae_checkpoint = cfg_dict.get("vae_checkpoint")
    if not vae_checkpoint:
        raise ValueError("VAE checkpoint is required to build model.")

    continuous_dim, discrete_token_vocab_size = load_codebook_config(
        vae_checkpoint,
        cfg_dict,
    )

    kmeans_path = cfg_dict.get("kmeans_path")
    if kmeans_path:
        summary_path = (
            os.path.join(kmeans_path, "summary.json")
            if os.path.isdir(kmeans_path)
            else os.path.join(os.path.dirname(kmeans_path), "summary.json")
        )
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"kmeans_path requires summary.json: {summary_path}")
        with open(summary_path, "r") as f:
            summary = json.load(f)
        latent_selection = summary.get("latent_selection", {})
        if latent_selection.get("indices") is not None:
            raise ValueError("Non-contiguous k-means latent indices are not supported.")
        kmeans_end = int(
            latent_selection.get("end", summary.get("feature_dims"))
        )
        continuous_start = int(cfg_dict.get("continuous_start", kmeans_end))
        discrete_token_vocab_size = int(summary["num_clusters"])
        continuous_dim = int(continuous_dim) - continuous_start

    return continuous_dim, discrete_token_vocab_size


def build_model(cfg_dict: Dict[str, Any], tokenizer: HybridTokenizer) -> HybridTTS:
    """Builds a HybridTTS model from a configuration dictionary."""

    backbone_cfg = cfg_dict.get("backbone_config")
    if backbone_cfg is None:
        backbone_cfg = cfg_dict.get("backbone")
    if backbone_cfg is not None:
        backbone_cfg = backbone_cfg.copy()

    diffusion_head_cfg = cfg_dict.get("diffusion_head_config")
    if diffusion_head_cfg is None:
        diffusion_head_cfg = cfg_dict.get("diffusion_head")
    if diffusion_head_cfg is not None:
        diffusion_head_cfg = diffusion_head_cfg.copy()

    adapter_cfg = cfg_dict.get("continuous_adapter_config")
    if adapter_cfg is None:
        adapter_cfg = cfg_dict.get("continuous_adapter")
    if adapter_cfg is not None:
        adapter_cfg = adapter_cfg.copy()

    token_head_cfg = cfg_dict.get("token_head_config")
    if token_head_cfg is None:
        token_head_cfg = cfg_dict.get("token_head")
    if token_head_cfg is not None:
        token_head_cfg = token_head_cfg.copy()

    # Dynamic resolution of continuous_dim/discrete vocab from VAE config.json,
    # optionally overridden by an external k-means codebook.
    continuous_dim, discrete_token_vocab_size = load_codebook_config_from_cfg(cfg_dict)

    shift_audio_offset = backbone_cfg.pop("shift_audio_offset")
    backbone_config = BackboneConfig(**backbone_cfg)

    training_cfg = cfg_dict.get("training")
    discrete_only = training_cfg.get("discrete_only", False)
    continuous_only = training_cfg.get("continuous_only", False)
    if discrete_only and continuous_only:
        raise ValueError("discrete_only and continuous_only are mutually exclusive.")

    diffusion_head_config = None
    if diffusion_head_cfg is not None and not discrete_only:
        diffusion_head_config = DiTConfig(**diffusion_head_cfg)

    continuous_adapter_config = None
    if adapter_cfg is not None and not discrete_only:
        continuous_adapter_config = AdapterConfig(**adapter_cfg)

    hybrid_config = HybridTTSConfig(
        backbone_config=backbone_config,
        diffusion_head_config=diffusion_head_config,
        continuous_adapter_config=continuous_adapter_config,
        prompt_vocab_size=tokenizer.prompt_vocab_size,
        discrete_token_vocab_size=tokenizer.discrete_token_vocab_size,
        continuous_dim=continuous_dim,
        pad_token_id=tokenizer.pad_id,
        start_audio_id=tokenizer.start_audio_id,
        end_audio_id=tokenizer.end_audio_id,
        debug=cfg_dict.get("debug", False),
        uncond_prob=training_cfg.get("uncond_prob"),
        no_augment_ratio=training_cfg.get("no_augment_ratio"),
        discrete_only=discrete_only,
        continuous_only=continuous_only,
        shift_audio_offset=shift_audio_offset,
        continuous_scaling_mode=cfg_dict.get("continuous_scaling_mode"),
    )

    model = HybridTTS(config=hybrid_config, tokenizer=tokenizer)
    if not continuous_only and training_cfg.get(
        "init_discrete_embeddings_from_quantizer",
        True,
    ):
        quantizer_embeddings = load_quantizer_embeddings(cfg_dict)
        if quantizer_embeddings is not None:
            model.initialize_discrete_embeddings(quantizer_embeddings)

    return model
