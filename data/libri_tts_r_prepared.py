import os
import json
from tqdm import tqdm
from collections import defaultdict
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

    def __init__(self, force_vocab_build: bool = False):
        super().__init__()
        vocab_path = os.path.join(os.path.dirname(__file__), "phoneme_vocab.json")

        # Load only train and test explicitly to avoid loading backups/sharded twice
        data_files = {
            "train": [f"{parquet_dir}/train/**/*.parquet", f"{parquet_dir}/train/*.parquet"],
            "test": [f"{parquet_dir}/test/**/*.parquet", f"{parquet_dir}/test/*.parquet"]
        }
        
        # Datasets will ignore globs that find nothing, but to be safe, we let HF find the parquets
        hf_cache_dir = f"{SLURM_TMPDIR}/hf_cache"
        os.makedirs(hf_cache_dir, exist_ok=True)
        dataset = load_dataset("parquet", data_files=data_files, cache_dir=hf_cache_dir)

        # Organize partitions by destination (train/test)
        partitions_per_destination = defaultdict(list)
        for partition in dataset.keys():
            dest = "train" if "train" in partition else "test"
            partitions_per_destination[dest].append(dataset[partition])

        # --- Phoneme vocabulary: load or build ---
        self.phoneme_vocab = {}

        if not force_vocab_build and os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                self.phoneme_vocab = json.load(f)
            if self.phoneme_vocab:
                print(
                    f"Loaded existing phoneme vocabulary from {vocab_path} "
                    f"(size: {len(self.phoneme_vocab)})"
                )

        if not self.phoneme_vocab:
            print("Building phoneme vocabulary...")
            g2p = G2p()
            for dest, parts in partitions_per_destination.items():
                ds = concatenate_datasets(parts)
                for example in tqdm(ds, desc=f"Building phoneme vocab for {dest}"):
                    text = example.get("text_normalized")
                    if text:
                        for p in g2p(text):
                            if p not in self.phoneme_vocab:
                                self.phoneme_vocab[p] = len(self.phoneme_vocab)

            with open(vocab_path, "w") as f:
                json.dump(self.phoneme_vocab, f, indent=4)
            print(
                f"Phoneme vocabulary saved to {vocab_path} "
                f"(size: {len(self.phoneme_vocab)})"
            )

        # --- Map phoneme IDs onto each partition ---
        for dest, parts in partitions_per_destination.items():
            ds = concatenate_datasets(parts)
            setattr(self, f"{dest}_dataset", ds)
