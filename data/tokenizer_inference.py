import os
import sys
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Assicuriamo che i moduli possano essere importati
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tokenizer import BPETokenizer
from data.audio_dataset import DataCollator, HubertDatasetWrapper
from data.librispeechHubert import LibriSpeech100h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer", type=str, default="tokenizer.json")
    parser.add_argument("-o", "--output", type=str, default="encoded_sequences.json")
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("-n", "--num_samples", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer file not found: {args.tokenizer}")

    # 1. Load Tokenizer
    tokenizer = BPETokenizer(n_initial_units=1024, target_vocab_size=16384, deduplicate=True, verbose=True)
    tokenizer.load(args.tokenizer)

    # 2. Setup DataLoader (Strictly as requested)
    data_collator = DataCollator()
    dataset = HubertDatasetWrapper(LibriSpeech100h(), split=args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # 3. Inference Loop
    encoded_results = []
    durations_results = []
    compression_ratios = []
    num_batches = len(dataloader)
    if args.num_samples is not None:
        num_batches = args.num_samples // args.batch_size
    counter = 0
    for batch in tqdm(dataloader):
        # Assuming batch is a dict with 'audio_codes'
        if "audio_codes" not in batch:
            raise KeyError("Batch does not contain 'audio_codes' key")

        audio_codes = batch["audio_codes"]

        # Encode
        batch_tokens, batch_durations = tokenizer.encode_batch(audio_codes)
        encoded_results.extend(batch_tokens)
        durations_results.extend(batch_durations)
        for i in range(len(batch_tokens)):
            compression_ratios.append(len(audio_codes[i])/len(batch_tokens[i]))
        
        counter += 1
        if counter >= num_batches:
            break
    
    # create an histogram with the durations
    # Flatten the nested list of durations into a single array
    flat_durations = np.array([d for duration_list in durations_results for d in duration_list])
    durations_histogram = np.histogram(flat_durations, bins=100)
    plt.bar(durations_histogram[1][:-1], durations_histogram[0])
    # limit x axis to 20
    plt.xlim(0, 20)
    plt.savefig("durations_histogram.png")
    plt.close()

    # create also the boxplot of the durations
    plt.boxplot(flat_durations)
    plt.savefig("durations_boxplot.png")
    plt.close()
    
    # 4. print a report on the compression ratios
    print(f"Average compression ratio: {sum(compression_ratios) / len(compression_ratios)}")
    print(f"Min compression ratio: {min(compression_ratios)}")
    print(f"Max compression ratio: {max(compression_ratios)}")
    # print the average duration
    print(f"\nAverage duration: {np.mean(flat_durations)}")
    print(f"max duration: {np.max(flat_durations)}")
    print(f"min duration: {np.min(flat_durations)}")


if __name__ == "__main__":
    main()
