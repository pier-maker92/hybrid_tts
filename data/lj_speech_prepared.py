import os
import sys
import json
import argparse
from tqdm import tqdm
from typing import Optional, List
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
import glob
import pyarrow.parquet as pq
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


class CustomListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        if len(data_list) > 0:
            self.column_names = list(data_list[0].keys())
        else:
            self.column_names = []
            
    def remove_columns(self, column_names):
        if isinstance(column_names, str):
            column_names = [column_names]
        new_data_list = []
        for item in self.data_list:
            new_item = {k: v for k, v in item.items() if k not in column_names}
            new_data_list.append(new_item)
        return CustomListDataset(new_data_list)
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        return self.data_list[idx]


class LJSpeechDataset(SimpleAudioDataset):
    def __init__(
        self,
        languages: Optional[List[str]] = None,
        force_vocab_build: bool = False,
        dataset_dir_name: str = "ljspeech-prepared",
    ):
        super().__init__()
        parquet_dir = f"{SLURM_TMPDIR}/datasets/{dataset_dir_name}"
        print(f"Loading parquet files manually from {parquet_dir}...")
        
        train_data = []
        test_data = []
        
        all_parquets = glob.glob(os.path.join(parquet_dir, "**/*.parquet"), recursive=True)
        if not all_parquets:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
            
        for f in all_parquets:
            table = pq.read_table(f)
            if "audio" in table.column_names:
                table = table.drop(["audio"])
            pylist = table.to_pylist()
            if "test" in f or "dev" in f or "eval" in f:
                test_data.extend(pylist)
            else:
                train_data.extend(pylist)
                
        print(f"Loaded {len(train_data)} train samples and {len(test_data)} test samples.")

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
            for example in tqdm(train_data + test_data, desc="Building phoneme vocab"):
                transcription = example.get("transcript") or example.get("transcription")
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

        for example in tqdm(train_data, desc="Mapping train dataset"):
            transcription = example.get("transcript") or example.get("transcription")
            phoneme_ids = []
            if transcription:
                for p in g2p(transcription):
                    phoneme_ids.append(self.phoneme_vocab.get(p, 0))
            example["phoneme_ids"] = phoneme_ids
            example["prompt_ids"] = phoneme_ids
            
        for example in tqdm(test_data, desc="Mapping test dataset"):
            transcription = example.get("transcript") or example.get("transcription")
            phoneme_ids = []
            if transcription:
                for p in g2p(transcription):
                    phoneme_ids.append(self.phoneme_vocab.get(p, 0))
            example["phoneme_ids"] = phoneme_ids
            example["prompt_ids"] = phoneme_ids

        self.train_dataset = CustomListDataset(train_data)
        self.test_dataset = CustomListDataset(test_data)

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
