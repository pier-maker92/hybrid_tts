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
from modules.submodules.MelCausalVAE.modules.feature_extractor import FeatureExtractor, MelSpectrogramConfig

# Specify custom cache directory
SLURM_TMPDIR = os.getenv("SLURM_TMPDIR")
parquet_dir = f"{SLURM_TMPDIR}/datasets/librispeech-aligned_prepared"
# import mel spec encoder
mel_spec_encoder = FeatureExtractor(config=MelSpectrogramConfig())


def simple_collate_fn(batch):
    return batch


class LibriSpeechAlignDataset(SimpleAudioDataset):
    def __init__(self, languages: Optional[List[str]] = None):
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

        for destination in partitions_per_destination:
            setattr(
                self,
                f"{destination}_dataset",
                concatenate_datasets(partitions_per_destination[destination]),
            )

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
