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
from modules.feature_extractor import FeatureExtractor, MelSpectrogramConfig

# Specify custom cache directory
cache_dir = "/home/ec2-user/dataset_cache"
parquet_dir = "/home/ec2-user/data"
# import mel spec encoder
mel_spec_encoder = FeatureExtractor(config=MelSpectrogramConfig())


def simple_collate_fn(batch):
    return batch


def load_mls_english_dataset():
    ds = load_dataset(
        f"{parquet_dir}/mls/en",  # parler-tts/mls_eng_10k
        cache_dir=cache_dir,
        num_proc=min(os.cpu_count(), 16),
    )
    return ds  # ["train"]


class MLSDataset(SimpleAudioDataset):
    def __init__(self, languages: Optional[List[str]] = None):
        super().__init__()
        # Load the two datasets
        ds_list = []
        self.language_id_map = {
            "french": "fr",
            "german": "de",
            "spanish": "es",
            "english": "en",
        }

        # Initialize set to collect all unique phonemes
        if languages is None:
            languages = ["french", "german", "spanish", "english"]
        for lang in languages:
            if lang == "english":
                ds = load_mls_english_dataset()
            else:
                ds = load_dataset(
                    f"{parquet_dir}/mls/{self.language_id_map[lang]}",  # "facebook/multilingual_librispeech", lang
                    cache_dir=cache_dir,
                    num_proc=min(
                        os.cpu_count(),
                        16,
                    ),
                )
            ds_list.append(ds)

        partitions_per_destination = defaultdict(list)
        for dataset in ds_list:
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

    # def amount_per_language(self, language: str):
    #     if language == "english":
    #         return "1%"
    #     elif language == "french":
    #         return "10%"
    #     elif language == "german":
    #         return "10%"
    #     elif language == "spanish":
    #         return "10%"
    #     elif language == "italian":
    #         return "10%"

    def _partition_to_destination(self, partition_name):
        if partition_name.split(".")[0] in ["train"]:
            return "train"
        elif partition_name.split(".")[0] in ["dev", "test", "9_hours", "1_hours"]:
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
