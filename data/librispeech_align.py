import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from tqdm import tqdm
from typing import Optional, List
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from data.audio_dataset import SimpleAudioDataset, DataCollator, TrainDatasetWrapper
from modules.submodules.MelCausalVAE.modules.feature_extractor import (
    FeatureExtractor,
    MelSpectrogramConfig,
)

# Specify custom cache directory
parquet_dir = f"~/Research/datasets/librispeech-aligned_prepared"
# import mel spec encoder
mel_spec_encoder = FeatureExtractor(config=MelSpectrogramConfig())


def simple_collate_fn(batch):
    return batch


class LibriSpeechAlignDataset(SimpleAudioDataset):
    def __init__(
        self, languages: Optional[List[str]] = None, force_vocab_build: bool = False
    ):
        super().__init__()
        # Load the two datasets
        dataset = load_dataset(
            "parquet",
            data_dir=f"{parquet_dir}",
        ).remove_columns("audio")
        # dataset = load_dataset(
        #     "parquet",
        #     data_files={
        #         "train": f"{parquet_dir}/train_clean_100/*.parquet",
        #         "dev": f"{parquet_dir}/dev_clean/*.parquet",
        #     },
        # )
        partitions_per_destination = defaultdict(list)
        for partition in dataset:
            print(
                f"partition: {partition}, destination: {self._partition_to_destination(partition)}"
            )
            partitions_per_destination[
                self._partition_to_destination(partition)
            ].append(dataset[partition])

        self.phoneme_vocab = {}
        import json

        vocab_path = os.path.join(os.path.dirname(__file__), "phoneme_vocab.json")

        if not force_vocab_build and os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r") as f:
                    self.phoneme_vocab = json.load(f)
                if self.phoneme_vocab:
                    print(
                        f"Loaded existing phoneme vocabulary from {vocab_path} (size: {len(self.phoneme_vocab)})"
                    )
            except Exception as e:
                print(f"Error loading vocabulary from {vocab_path}: {e}")

        if not self.phoneme_vocab:
            from tqdm import tqdm

            print("Building phoneme vocabulary...")
            # First pass: build vocabulary
            for destination in partitions_per_destination:
                ds = concatenate_datasets(partitions_per_destination[destination])
                for example in tqdm(
                    ds, desc=f"Building phoneme vocab for {destination}"
                ):
                    alignments = example.get("phonemes")
                    if alignments is not None:
                        for phoneme_dict in alignments:
                            p = phoneme_dict["phoneme"]
                            if p not in self.phoneme_vocab:
                                self.phoneme_vocab[p] = len(self.phoneme_vocab)

            with open(vocab_path, "w") as f:
                json.dump(self.phoneme_vocab, f, indent=4)
            print(
                f"Phoneme vocabulary saved to {vocab_path} (size: {len(self.phoneme_vocab)})"
            )

        def build_vocab_and_map(batch):
            phoneme_ids_batch = []
            for alignments in batch["phonemes"]:
                phoneme_ids = []
                if alignments is not None:
                    for phoneme_dict in alignments:
                        p = phoneme_dict["phoneme"]
                        phoneme_ids.append(self.phoneme_vocab.get(p, 0))
                phoneme_ids_batch.append(phoneme_ids)
            return {"phoneme_ids": phoneme_ids_batch, "prompt_ids": phoneme_ids_batch}

        for destination in partitions_per_destination:
            ds = concatenate_datasets(partitions_per_destination[destination])
            ds = ds.map(build_vocab_and_map, batched=True, num_proc=1)
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
#     # data collator
#     data_collator = DataCollator()
#     dataset = TrainDatasetWrapper(MLSDataset(), "train")
#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         collate_fn=data_collator,
#         num_workers=min(os.cpu_count(), 16),
#         shuffle=True,
#     )
#     means = []
#     stds = []
#     counter = 0
#     if args.stats:
#         pbar = tqdm(total=min(args.num_samples, len(dataloader)))
#         for batch in dataloader:
#             audio_srs = batch["output_audios_srs"]
#             mel_spec = mel_spec_encoder(audio_srs)
#             featues, padding_mask = mel_spec.audio_features, mel_spec.padding_mask
#             for feature, mask in zip(featues, padding_mask):
#                 means.append(feature[~mask].mean())
#                 stds.append(feature[~mask].std())
#             counter += args.batch_size
#             if counter >= args.num_samples or counter >= len(dataloader):
#                 break
#             pbar.update(args.batch_size)
#         pbar.close()
#         print(f"Mean: {np.mean(means)}")
#         print(f"Std: {np.mean(stds)}")
#     else:
#         print(dataset[0])
#         breakpoint()
