"""Frozen SM token extraction from exported DiCodec latents, with full z targets."""
from functools import lru_cache
from pathlib import Path
import os

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from data.audio_dataset import DataCollator
from modules.submodules.MelCausalVAE.dicodec.modules.sm_quantizer.configs import ModelConfig, from_dict
from modules.submodules.MelCausalVAE.dicodec.modules.sm_quantizer.encoder import build_encoder
from modules.submodules.MelCausalVAE.dicodec.modules.sm_quantizer.quantizer import OnlineQuantizer


def checkpoint_path(path):
    path = Path(os.path.expandvars(path.replace("$SCRATCH", os.environ.get("SCRATCH", "/Users/software/Research"))))
    return path / "last.pt" if path.is_dir() else path


@lru_cache(maxsize=8)
def checkpoint_config(path):
    payload = torch.load(checkpoint_path(path), map_location="cpu", weights_only=True, mmap=True)
    if payload["config"]["data"].get("input", "z") != "z_sem":
        raise ValueError("Hybrid SM token extraction requires a checkpoint trained on z_sem.")
    return from_dict(ModelConfig, payload["config"]["model"])


class FrozenSMTokenizer(nn.Module):
    def __init__(self, path):
        super().__init__()
        config = checkpoint_config(str(path))
        self.latent_dim = config.latent_dim
        self.encoder = build_encoder(config.latent_dim, config.quantizer.resolved_dim,
                                     config.projection_hidden_dim, config.encoder)
        self.quantizer = OnlineQuantizer(config.quantizer)
        payload = torch.load(checkpoint_path(path), map_location="cpu", weights_only=True, mmap=True)
        state = {key: value for key, value in payload["model"].items()
                 if key.startswith(("encoder.", "quantizer."))}
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, z_sem, valid):
        self.eval()
        with torch.autocast(device_type=z_sem.device.type, enabled=False):
            encoded = self.encoder(z_sem.float().masked_fill(~valid.unsqueeze(-1), 0))
            return self.quantizer(encoded, valid).indices


def build_latent_dataset(training):
    from datasets import load_dataset
    root = Path(training["latent_dataset_path"])
    files = []
    for partition in training["dataset_partitions"]:
        if not partition or Path(partition).name != partition or partition in {".", ".."}:
            raise ValueError("Dataset partitions must be directory names.")
        shards = sorted(p for p in (root / partition).glob("*.parquet") if not p.name.startswith("."))
        if not shards:
            raise FileNotFoundError(f"No Parquet shards in {root / partition}")
        files.extend(map(str, shards))
    if not files:
        raise ValueError("Select at least one dataset partition.")
    columns = ["z", "transcript"] + (["attributes"] if training["discrete"] else [])
    dataset = load_dataset("parquet", data_files=files, split="train", columns=columns,
                           cache_dir=training.get("latent_cache_dir"), keep_in_memory=False)
    return dataset, dataset.select([])


class SMLatentCollator:
    def __init__(self, tokenizer, checkpoint, device, discrete=True):
        self.base = DataCollator(tokenizer)
        self.device = device
        self.quantizer = FrozenSMTokenizer(checkpoint).to(device) if discrete else None

    @torch.no_grad()
    def __call__(self, instances):
        zs, sems, texts = [], [], []
        for item in instances:
            z = torch.as_tensor(item["z"], dtype=torch.float32)
            if z.ndim != 2 or not len(z) or not torch.isfinite(z).all():
                raise ValueError("Expected finite nonempty z [T,D].")
            text = item["transcript"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Every latent utterance requires transcript.")
            if self.quantizer is not None:
                sem = torch.as_tensor(item["attributes"]["z_sem"], dtype=torch.float32)
                if sem.shape != z.shape or sem.shape[-1] != self.quantizer.latent_dim or not torch.isfinite(sem).all():
                    raise ValueError("z_sem must be finite and aligned with z [T,D].")
                sems.append(sem)
            zs.append(z)
            texts.append(text)
        indices = None
        if self.quantizer is not None:
            padded = pad_sequence(sems, batch_first=True).to(self.device)
            lengths = torch.tensor([len(z) for z in zs], device=self.device)
            valid = torch.arange(padded.shape[1], device=self.device)[None] < lengths[:, None]
            indices = self.quantizer(padded, valid).cpu()
        return self.base([{"discrete_tokens": None if indices is None else indices[i, :len(z)].tolist(),
                           "continuous_tokens": z.tolist(), "transcription": texts[i],
                           "ids": item.get("id", str(i))}
                          for i, (z, item) in enumerate(zip(zs, instances))])
