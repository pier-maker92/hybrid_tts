import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, Audio
from modules.feature_extractor import FeatureExtractor, MelSpectrogramConfig
from data.audio_dataset import SimpleAudioDataset, DataCollator, TrainDatasetWrapper

# Determine dataset root from SLURM_TMPDIR or fallback
slurm_tmpdir = os.environ.get("SLURM_TMPDIR", "/home/ec2-user/data")
dataset_root = os.path.join(slurm_tmpdir, "datasets/libritts")
# import mel spec encoder
mel_spec_encoder = FeatureExtractor(config=MelSpectrogramConfig())


def simple_collate_fn(batch):
    return batch


class LibriTTS(SimpleAudioDataset):
    def __init__(self):
        super().__init__()
        
        # Standard LibriTTS extraction usually creates a LibriTTS/ subdirectory
        global dataset_root
        current_dataset_root = dataset_root
        if os.path.exists(os.path.join(current_dataset_root, "LibriTTS")):
            current_dataset_root = os.path.join(current_dataset_root, "LibriTTS")
            
        print(f"Loading LibriTTS from: {current_dataset_root}")
        
        if not os.path.exists(current_dataset_root):
            raise FileNotFoundError(f"Dataset directory not found: {current_dataset_root}")

        # Find available subsets (folders like train-clean-100, dev-clean, test-clean, etc.)
        subsets = [d for d in os.listdir(current_dataset_root) 
                  if os.path.isdir(os.path.join(current_dataset_root, d)) 
                  and not d.startswith(".")]
        
        dataset_dict = {}
        if not subsets:
            print("No subdirectories found, attempting to load as a single audiofolder dataset.")
            dataset_dict["train"] = load_dataset("audiofolder", data_dir=current_dataset_root, split="train")
        else:
            for subset in subsets:
                if subset == "test-other":
                    continue             
                destination = self._partition_to_destination(subset)
                if destination:
                    print(f"Loading subset: {subset} -> destination: {destination}")
                    ds = load_dataset(
                        "audiofolder",
                        data_dir=os.path.join(current_dataset_root, subset),
                        split="train"
                    )
                    # Remove 'label' column (speaker IDs) to avoid concatenation errors
                    if "label" in ds.column_names:
                        ds = ds.remove_columns("label")

                    # Add an 'id' field if it's missing, using the filename
                    if "id" not in ds.column_names:
                        # Disable decoding temporarily: this avoids 'torchcodec' errors and saves RAM
                        ds = ds.cast_column("audio", Audio(decode=False))
                        ds = ds.map(
                            lambda x: {"id": os.path.basename(x["audio"]["path"])}, 
                            num_proc=min(os.cpu_count(), 8)
                        )
                        # Re-enable decoding for training
                        ds = ds.cast_column("audio", Audio(decode=True))
                    dataset_dict[subset] = ds

        partitions_per_destination = defaultdict(list)
        for partition in dataset_dict:
            destination = self._partition_to_destination(partition)
            if destination:
                print(f"partition: {partition}, destination: {destination}")
                partitions_per_destination[destination].append(dataset_dict[partition])

        for destination in partitions_per_destination:
            setattr(
                self,
                f"{destination}_dataset",
                concatenate_datasets(partitions_per_destination[destination]),
            )

    def _partition_to_destination(self, partition_name):
        # Map original LibriTTS subset names to our train/test splits
        if "train" in partition_name or "dev" in partition_name or partition_name == "validation":
            return "train"
        elif "test" in partition_name:
            return "test"
        return None


# parser = argparse.ArgumentParser()
# parser.add_argument("-b", "--batch_size", type=int, default=1)
# parser.add_argument("-s", "--stats", action="store_true", default=False)
# parser.add_argument("-n", "--num_samples", type=int, default=10000)
# args = parser.parse_args()
# if __name__ == "__main__":
#     # data collator
#     data_collator = DataCollator()
#     dataset = TrainDatasetWrapper(LibriTTS(), "train")
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
