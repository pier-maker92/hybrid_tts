import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import Optional, List
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from data.audio_dataset import SimpleAudioDataset, DataCollator, TrainDatasetWrapper
from modules.feature_extractor import MelSpectrogramEncoder, MelSpectrogramConfig

# Specify custom cache directory
parquet_dir = "/home/ec2-user/data"
# Path to the JSON file containing the test set IDs
test_json_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "e2tts_librispeech_pc_test_clean.json"
)
# import mel spec encoder
mel_spec_encoder = MelSpectrogramEncoder(config=MelSpectrogramConfig())


def simple_collate_fn(batch):
    return batch


def _load_test_keys(json_path: str) -> set:
    """Load the set of keys from the e2tts test JSON file."""
    with open(json_path, "r") as f:
        test_data = json.load(f)
    return {item["key"] for item in test_data}


class LibriSpeechAlignTestDataset(SimpleAudioDataset):
    """
    A variant of LibriSpeechAlignDataset that filters the dataset
    to only include samples whose ID matches the keys in the
    e2tts_librispeech_pc_test_clean.json file.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        json_path: str = test_json_path,
        do_filter: bool = True,
    ):
        super().__init__()

        # Load the dataset from parquet
        dataset = load_dataset(
            "parquet",
            data_files={
                "test": f"{parquet_dir}/librispeech-aligned/test_clean/*.parquet"
            },
        )

        partitions_per_destination = defaultdict(list)
        for partition in dataset:
            print(
                f"partition: {partition}, destination: {self._partition_to_destination(partition)}"
            )
            partitions_per_destination[
                self._partition_to_destination(partition)
            ].append(dataset[partition])

        for destination in partitions_per_destination:
            combined = concatenate_datasets(partitions_per_destination[destination])

            if destination == "test":
                # Always ensure we only use test-clean
                original_len = len(combined)
                combined = combined.filter(
                    lambda subset: subset == "test-clean",
                    input_columns=["subset"],
                    num_proc=min(os.cpu_count(), 8),
                    desc="Filtering test-clean subset",
                )
                print(
                    f"Kept only test-clean: {original_len} -> {len(combined)} samples"
                )

                if do_filter:
                    # Load the target test keys from JSON
                    test_keys = _load_test_keys(json_path)
                    print(f"Loaded {len(test_keys)} test keys from JSON")

                    # Filter the test split to only keep samples matching the JSON keys
                    original_len_json = len(combined)
                    combined = combined.filter(
                        lambda id_val: id_val in test_keys,
                        input_columns=["id"],
                        num_proc=min(os.cpu_count(), 8),
                        desc="Filtering test set by JSON keys",
                    )
                    print(
                        f"Filtered test dataset by JSON: {original_len_json} -> {len(combined)} samples "
                        f"({len(test_keys)} keys requested)"
                    )

            setattr(
                self,
                f"{destination}_dataset",
                combined,
            )

    def _partition_to_destination(self, partition_name):
        if partition_name in ["train", "validation"]:
            return "train"
        elif partition_name in ["test"]:
            return "test"
