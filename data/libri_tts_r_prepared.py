import os
import json
from datasets import load_dataset, concatenate_datasets
from g2p_en import G2p
from data.audio_dataset import SimpleAudioDataset

SLURM_TMPDIR = os.getenv("SLURM_TMPDIR")
if SLURM_TMPDIR is None:
    raise ValueError("SLURM_TMPDIR is not set")
parquet_dir = f"{SLURM_TMPDIR}/datasets/libritts-r-prepared"


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


class LibriTTSRPrepared(SimpleAudioDataset):
    """LibriTTS-R pre-prepared dataset (discrete + continuous already extracted)."""

    def __init__(self):
        super().__init__()
        vocab_path = os.path.join(os.path.dirname(__file__), "phoneme_vocab.json")
        if not os.path.exists(vocab_path):
            raise ValueError(f"Phoneme vocab not found at {vocab_path}")
        with open(vocab_path) as f:
            self.phoneme_vocab = json.load(f)

        dataset = load_dataset("parquet", data_dir=parquet_dir)

        train_parts, test_parts = [], []
        for partition in dataset:
            dest = "train" if partition == "train" else "test"
            ds = dataset[partition].map(
                _build_phoneme_ids,
                fn_kwargs={"phoneme_vocab": self.phoneme_vocab},
                batched=True,
                num_proc=8,
            )
            if dest == "train":
                train_parts.append(ds)
            else:
                test_parts.append(ds)

        self.train_dataset = concatenate_datasets(train_parts) if train_parts else None
        self.test_dataset = concatenate_datasets(test_parts) if test_parts else None
