import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from modules.melspecEncoder import MelSpectrogramEncoder, MelSpectrogramConfig
from data.audio_dataset import SimpleAudioDataset, DataCollator, TrainDatasetWrapper
from phonemizer.separator import Separator
from phonemizer import phonemize

# import mel spec encoder
mel_spec_encoder = MelSpectrogramEncoder(config=MelSpectrogramConfig())
sep = Separator(phone=" ", word=" <sil> ", syllable=None)


def simple_collate_fn(batch):
    return batch


class Vocab:
    def __init__(self, output_path, mode="phoneme"):
        self.output_path = output_path
        self.mode = mode
        self.tokens = set(["<pad>", "<sil>", "<unk>"])

    def update(self, phoneme_sequences):
        """Update vocab with a list of phoneme strings."""
        for seq in phoneme_sequences:
            if not seq:
                continue
            
            # Add implicit silences just like in the Aligner/Training
            seq = f"<sil> {seq} <sil>"
            
            if self.mode == "phoneme":
                self.tokens.update(seq.split())
            elif self.mode == "char":
                for p in seq.split():
                    if p == "<sil>":
                        self.tokens.add(p)
                    else:
                        self.tokens.update(list(p))
            else:
                 raise ValueError(f"Unknown mode: {self.mode}")

    def save(self):
        """Save vocabulary to json."""
        vocab_dict = {token: i for i, token in enumerate(sorted(list(self.tokens)))}
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        import json
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to {self.output_path} with {len(vocab_dict)} tokens.")


class LibriSpeech100h(SimpleAudioDataset):
    def __init__(self, phoneme_parsing_mode="phoneme", vocab_path="data/vocab.json"):
        super().__init__()
        # Load the two datasets
        datasets = []
        for subset in ["train"]:
            ds = load_dataset(
                "cmu-mlsp/hubert_layer9-librispeech-asr100h",
                num_proc=min(
                    os.cpu_count(),
                    16,
                ),
            )
            datasets.append(ds)

        partitions_per_destination = defaultdict(list)
        for dataset in datasets:
            for partition in dataset:
                partitions_per_destination[self._partition_to_destination(partition)].append(dataset[partition])

        for destination in partitions_per_destination:
            setattr(
                self,
                f"{destination}_dataset",
                concatenate_datasets(partitions_per_destination[destination]),
            )
        # select only the "audio_codes" column
        # select only the "audio_codes" column
        self.train_dataset = self.train_dataset.map(self.get_phonemes, batched=True, num_proc=16, batch_size=1000)
        self.test_dataset = self.test_dataset.map(self.get_phonemes, batched=True, num_proc=16, batch_size=1000)  # type: ignore

        # Build vocabulary
        self.phoneme_parsing_mode = phoneme_parsing_mode
        self.vocab_path = vocab_path
        vocab = Vocab(self.vocab_path, self.phoneme_parsing_mode)
        print(f"Building vocabulary with mode: {self.phoneme_parsing_mode}...")
        
        # Simple iteration over batches using indices to avoid loading everything at once
        phoneme_dataset = self.train_dataset.select_columns("phonemes")
        batch_size = 512
        for i in tqdm(range(0, len(phoneme_dataset), batch_size), desc="Building Vocab"):
            batch = phoneme_dataset[i : i + batch_size]
            vocab.update(batch["phonemes"])
        
        phoneme_dataset = self.test_dataset.select_columns("phonemes")
        batch_size = 512
        for i in tqdm(range(0, len(phoneme_dataset), batch_size), desc="Building Vocab"):
            batch = phoneme_dataset[i : i + batch_size]
            vocab.update(batch["phonemes"])
            
        vocab.save()

    def _partition_to_destination(self, partition_name):
        if "train" in partition_name:
            return "train"
        elif "dev" in partition_name or "test" in partition_name:
            return "test"

    def get_phonemes(self, example):
        text = example["text"]
        phoneme_str = phonemize(
            text,
            language="en-us",  # FIXME hardcoded language
            backend="espeak",
            separator=sep,
            strip=True,
            preserve_punctuation=False,
            njobs=1,
        )
        example["phonemes"] = phoneme_str
        return example


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-b", "--batch_size", type=int, default=1)
#     parser.add_argument("-s", "--stats", action="store_true", default=False)
#     parser.add_argument("-n", "--num_samples", type=int, default=10000)
#     args = parser.parse_args()
#     # data collator
#     data_collator = DataCollator()
#     dataset = TrainDatasetWrapper(LibriSpeech100h(), "train")
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
