import os
import sys
import json
import argparse
from tqdm import tqdm
from typing import Optional, List
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from g2p_en import G2p
from data.audio_dataset import (
    SimpleAudioDataset,
    DataCollator,
    TrainDatasetWrapper,
    TestDatasetWrapper,
)
from modules.hybrid_model import HybridTokenizer
from modules.submodules.MelCausalVAE.modules.feature_extractor import (
    FeatureExtractor,
    MelSpectrogramConfig,
)

# Specify custom cache directory
SLURM_TMPDIR = os.getenv("SLURM_TMPDIR")
if SLURM_TMPDIR is None:
    raise ValueError("SLURM_TMPDIR environment variable not set.")
parquet_dir = f"{SLURM_TMPDIR}/datasets/lj_speech_10_512"
# import mel spec encoder
mel_spec_encoder = FeatureExtractor(config=MelSpectrogramConfig())


def simple_collate_fn(batch):
    return batch


def build_vocab_and_map(batch, phoneme_vocab):
    g2p = G2p()
    phoneme_ids_batch = []
    transcriptions = batch.get("transcript") or batch.get("transcription") or []
    for transcription in transcriptions:
        phoneme_ids = []
        if transcription:
            phonemes = g2p(transcription)
            for p in phonemes:
                phoneme_ids.append(phoneme_vocab.get(p, 0))
        phoneme_ids_batch.append(phoneme_ids)
    return {"phoneme_ids": phoneme_ids_batch, "prompt_ids": phoneme_ids_batch}


class LJSpeechDataset(SimpleAudioDataset):
    def __init__(
        self, languages: Optional[List[str]] = None, force_vocab_build: bool = False
    ):
        super().__init__()
        # Load the prepared dataset
        dataset = load_dataset(
            "parquet",
            data_dir=f"{parquet_dir}",
        )
        if "audio" in dataset["train"].column_names:
            dataset = dataset.remove_columns("audio")

        partitions_per_destination = defaultdict(list)
        for partition in dataset:
            print(
                f"partition: {partition}, destination: {self._partition_to_destination(partition)}"
            )
            partitions_per_destination[
                self._partition_to_destination(partition)
            ].append(dataset[partition])

        self.phoneme_vocab = {}

        g2p = G2p()

        vocab_path = os.path.join(os.path.dirname(__file__), "phoneme_vocab.json")

        if not force_vocab_build and os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                self.phoneme_vocab = json.load(f)
            if self.phoneme_vocab:
                print(
                    f"Loaded existing phoneme vocabulary from {vocab_path} (size: {len(self.phoneme_vocab)})"
                )

        if not self.phoneme_vocab:
            print("Building phoneme vocabulary...")
            # First pass: build vocabulary
            for destination in partitions_per_destination:
                ds = concatenate_datasets(partitions_per_destination[destination])
                for example in tqdm(
                    ds, desc=f"Building phoneme vocab for {destination}"
                ):
                    transcription = example.get("transcript") or example.get(
                        "transcription"
                    )
                    if transcription:
                        phonemes = g2p(transcription)
                        for p in phonemes:
                            if p not in self.phoneme_vocab:
                                self.phoneme_vocab[p] = len(self.phoneme_vocab)

            with open(vocab_path, "w") as f:
                json.dump(self.phoneme_vocab, f, indent=4)
            print(
                f"Phoneme vocabulary saved to {vocab_path} (size: {len(self.phoneme_vocab)})"
            )

        for destination in partitions_per_destination:
            ds = concatenate_datasets(partitions_per_destination[destination])
            ds = ds.map(
                build_vocab_and_map,
                fn_kwargs={"phoneme_vocab": self.phoneme_vocab},
                batched=True,
                num_proc=1,
            )
            setattr(self, f"{destination}_dataset", ds)

    def _partition_to_destination(self, partition_name):
        if "train" in partition_name:
            return "train"
        else:
            return "test"

    # def __len__(self):
    #     return len(self.dataset)

    # def __getitem__(self, idx):
    #     data_dict = {}
    #     data = self.train_dataset[idx]
    #     self._process_audio_output(data_dict, data["audio"])
    #     return data_dict


# parser = argparse.ArgumentParser()
# parser.add_argument("-b", "--batch_size", type=int, default=1)
# parser.add_argument("-s", "--stats", action="store_true", default=False)
# parser.add_argument("-n", "--num_samples", type=int, default=100000)
# args = parser.parse_args()
# if __name__ == "__main__":

#     vocab_path = os.path.join(os.path.dirname(__file__), "phoneme_vocab.json")
#     with open(vocab_path, "r") as f:
#         phoneme_vocab = json.load(f)
#     dataset = LJSpeechDataset()
#     tok = HybridTokenizer(
#         phoneme_vocab=phoneme_vocab,
#         start_audio_id=len(phoneme_vocab),
#         end_audio_id=len(phoneme_vocab) + 1,
#         pad_id=len(phoneme_vocab) + 2,
#     )
#     dataset = TestDatasetWrapper(dataset, "train")
#     # data collator
#     data_collator = DataCollator(pad_id=tok.pad_id)

#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         collate_fn=data_collator,
#         num_workers=1,  # min(os.cpu_count(), 16),
#         shuffle=False,
#     )
#     means = []
#     stds = []
#     counter = 0

#     pbar = tqdm(total=min(args.num_samples, len(dataloader)))
#     for batch in dataloader:
#         breakpoint()
