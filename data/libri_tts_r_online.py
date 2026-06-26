import os
import json
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets
from g2p_en import G2p
from data.audio_dataset import SimpleAudioDataset

SLURM_TMPDIR = os.getenv("SLURM_TMPDIR")
if SLURM_TMPDIR is None:
    raise ValueError("SLURM_TMPDIR is not set")
parquet_dir = f"{SLURM_TMPDIR}/datasets/libritts-r"


def _build_phoneme_ids(batch, phoneme_vocab):
    g2p = G2p()
    ids_batch = []
    for text in batch.get("text_normalized", []):
        ids = []
        if text:
            for p in g2p(text):
                ids.append(phoneme_vocab.get(p, 0))
        ids_batch.append(ids)
    return {"phoneme_ids": ids_batch}


class LibriTTSROnline(SimpleAudioDataset):
    """LibriTTS-R dataset that returns raw audio for online VAE encoding."""

    def __init__(self):
        super().__init__()
        vocab_path = os.path.join(os.path.dirname(__file__), "phoneme_vocab.json")
        if not os.path.exists(vocab_path):
            raise ValueError(f"Phoneme vocab not found at {vocab_path}")
        with open(vocab_path) as f:
            self.phoneme_vocab = json.load(f)

        dataset = load_dataset("parquet", data_dir=parquet_dir)

        partitions_per_destination = defaultdict(list)
        for partition in dataset:
            dest = self._partition_to_destination(partition)
            partitions_per_destination[dest].append(dataset[partition])

        for dest, parts in partitions_per_destination.items():
            ds = concatenate_datasets(parts)
            ds = ds.map(
                _build_phoneme_ids,
                fn_kwargs={"phoneme_vocab": self.phoneme_vocab},
                batched=True,
                num_proc=8,
            )
            setattr(self, f"{dest}_dataset", ds)

    def _partition_to_destination(self, partition_name):
        return "train" if partition_name == "train" else "test"
