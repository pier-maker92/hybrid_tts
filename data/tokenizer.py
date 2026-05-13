import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict
from collections import Counter, defaultdict


class BPETokenizer:
    """Simple, correct BPE tokenizer implementation."""

    def __init__(self, n_initial_units: int, target_vocab_size: int, deduplicate: bool = True, verbose: bool = True):
        self.n_initial_units = n_initial_units
        self.target_vocab_size = target_vocab_size
        self.deduplicate = deduplicate
        self.verbose = verbose

        # Merge rules as list (for training order)
        self.merges: defaultdict[int, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))

        self._is_trained = False

    def load(self, path: str):
        with open(path, "r") as f:
            merges_dict = json.load(f)
        # convert the keys and values to int
        self.merges = defaultdict(lambda: defaultdict(int))
        for k1, inner_dict in merges_dict.items():
            self.merges[int(k1)] = {int(k2): int(v) for k2, v in inner_dict.items()}
        self._is_trained = True

    def _collapse_repetitions(
        self, corpus: List[np.ndarray], return_durations: bool = False, verbose: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        corpus_collapsed = []
        corpus_durations = []
        for sequence in tqdm(corpus, desc="Collapsing repetitions", disable=not verbose):
            collapsed = [sequence[0]]
            durations = [1]
            for i in range(1, len(sequence)):
                if sequence[i] != sequence[i - 1]:
                    collapsed.append(sequence[i])
                    durations.append(1)
                else:
                    durations[-1] += 1
            corpus_collapsed.append(np.array(collapsed).astype(int))
            corpus_durations.append(np.array(durations).astype(int))
        if return_durations:
            return corpus_collapsed, corpus_durations
        else:
            return corpus_collapsed

    def _count_pairs(self, sequences: List[np.ndarray], verbose: bool = False) -> Counter:
        """Count all adjacent pairs in sequences."""
        pair_counts = Counter()
        for sequence in tqdm(sequences, desc="Counting pairs", disable=not verbose):
            for i in range(len(sequence) - 1):
                pair_counts[(int(sequence[i]), int(sequence[i + 1]))] += 1
        return pair_counts

    def merge_pair_in_sequence_np(
        self,
        sequence: np.ndarray,
        pair: Tuple[int, int],
        new_token: int,
        pair_counts: Counter,
    ) -> Tuple[np.ndarray, Counter]:
        """
        Merge all occurrences of pair in sequence using vectorized numpy operations.
        Updates pair_counts by removing destroyed pairs and adding new ones.
        """
        a, b = pair
        seq = sequence
        n = len(seq)

        if n < 2:
            return seq.copy(), pair_counts

        # 1. Find candidate merge positions
        match = (seq[:-1] == a) & (seq[1:] == b)

        # 2. Enforce non-overlapping merges
        merge_pos = np.where(match)[0]
        if merge_pos.size > 1:
            keep = np.ones_like(merge_pos, dtype=bool)
            keep[1:] = merge_pos[1:] != merge_pos[:-1] + 1
            merge_pos = merge_pos[keep]

        if merge_pos.size == 0:
            return seq.copy(), pair_counts

        # 3. REMOVE old pairs that will be destroyed

        # Remove (left, a) pairs - where left is the token before merge position
        left_exists = merge_pos > 0
        if np.any(left_exists):
            left_pos = merge_pos[left_exists] - 1
            left_tokens = seq[left_pos]
            # Remove (left, a) for each merge
            left_pairs = list(zip(left_tokens, [a] * len(left_tokens)))
            for lp in left_pairs:
                pair_counts[tuple(lp)] -= 1
                if pair_counts[tuple(lp)] <= 0:
                    del pair_counts[tuple(lp)]

        # Remove (b, right) pairs - where right is the token after b
        right_exists = merge_pos + 2 < n
        if np.any(right_exists):
            right_pos = merge_pos[right_exists] + 2
            right_tokens = seq[right_pos]
            # Remove (b, right) for each merge
            right_pairs = list(zip([b] * len(right_tokens), right_tokens))
            for rp in right_pairs:
                pair_counts[tuple(rp)] -= 1
                if pair_counts[tuple(rp)] <= 0:
                    del pair_counts[tuple(rp)]

        # Note: (a, b) is removed by the caller (in train method)

        # 4. Build mask of elements to keep
        mask = np.ones(n, dtype=bool)
        mask[merge_pos + 1] = False  # remove second element of each merged pair

        # 5. Construct new sequence
        new_seq = seq[mask].copy()
        new_seq[np.searchsorted(np.flatnonzero(mask), merge_pos)] = new_token

        # 6. ADD new pairs created by merges

        # Add (left, new_token) pairs
        if np.any(left_exists):
            # Positions in new_seq where new_token was inserted (after a left neighbor)
            new_token_pos_with_left = np.searchsorted(np.flatnonzero(mask), merge_pos[left_exists])
            left_in_new = new_seq[new_token_pos_with_left - 1]
            left_new_pairs = list(zip(left_in_new, [new_token] * len(left_in_new)))
            for lnp in left_new_pairs:
                pair_counts[tuple(lnp)] += 1

        # Add (new_token, right) pairs
        if np.any(right_exists):
            # Positions in new_seq where new_token was inserted (before a right neighbor)
            new_token_pos_with_right = np.searchsorted(np.flatnonzero(mask), merge_pos[right_exists])
            # Right token is at position new_token_pos + 1
            if np.all(new_token_pos_with_right + 1 < len(new_seq)):
                right_in_new = new_seq[new_token_pos_with_right + 1]
                right_new_pairs = list(zip([new_token] * len(right_in_new), right_in_new))
                for rnp in right_new_pairs:
                    pair_counts[tuple(rnp)] += 1

        return new_seq.astype(int), pair_counts

    def _merge_pair_in_sequence(
        self, sequence: np.ndarray, pair: Tuple[int, int], new_token: int, pair_counts: Counter
    ) -> np.ndarray:
        """Merge all occurrences of pair in a single sequence."""
        a, b = pair
        new_seq = []
        i = 0
        assert isinstance(sequence, np.ndarray)

        while i < len(sequence):
            if i < len(sequence) - 1 and sequence[i] == a and sequence[i + 1] == b:
                new_seq.append(new_token)
                i += 2
            else:
                new_seq.append(sequence[i])
                i += 1

        # update pair counts
        for i in range(len(new_seq) - 1):
            if new_seq[i] == new_token:
                pair_counts[(new_seq[i], new_seq[i + 1])] += 1

        return np.array(new_seq).astype(int), pair_counts

    def train(self, corpus: List[np.ndarray], verbose: bool = True, output_path: str = None):

        if self._is_trained:
            raise RuntimeError("Tokenizer already trained")

        num_merges = self.target_vocab_size - self.n_initial_units

        iterator = range(num_merges)
        if verbose:
            iterator = tqdm(iterator, desc="BPE Training")

        # first count all pairs
        pair_counts = self._count_pairs(corpus, verbose=False)  # verbose=False to avoid double progress bar
        for merge_idx in iterator:
            # Get most frequent pair
            pair, count = pair_counts.most_common(1)[0]
            pair_counts.pop(pair)

            # Create new token
            new_token = self.n_initial_units + merge_idx

            # Record merge
            self.merges[pair[0]][pair[1]] = new_token

            # Merge in all sequences
            for i in range(len(corpus)):
                corpus[i], pair_counts = self.merge_pair_in_sequence_np(corpus[i], pair, new_token, pair_counts)

            if verbose:
                iterator.set_postfix({"pair": f"{pair}", "count": count, "vocab": new_token + 1})

        self._is_trained = True

        if output_path is not None:
            self.save(output_path)

        if verbose:
            print(f"Training complete. Final vocab size: {len(self.merges) + self.n_initial_units}")

    def save(self, path: str):
        # Convert numpy int64 keys/values to native Python ints for JSON serialization
        merges_dict = {}
        for k1, inner_dict in self.merges.items():
            merges_dict[int(k1)] = {int(k2): int(v) for k2, v in inner_dict.items()}

        with open(path, "w") as f:
            json.dump(merges_dict, f)

    def encode(self, sequence: List[int]) -> List[int]:
        if self.deduplicate:
            sequence, durations = self._collapse_repetitions(
                [sequence], return_durations=True, verbose=False
            )  # return a list of sequences and durations
            sequence = sequence[0]
            durations = durations[0]
        else:
            raise NotImplementedError("without deduplication is not implemented")
        # start iterating over the sequence and merging pairs
        changed = True
        while changed:
            changed = False
            new_sequence = []
            new_durations = []
            i = 0
            while i < len(sequence) - 1:
                current_token = sequence[i]
                next_token = sequence[i + 1]
                if current_token in self.merges and next_token in self.merges[current_token]:
                    new_token = self.merges[current_token][next_token]
                    new_sequence.append(new_token)
                    new_durations.append(durations[i] + durations[i + 1])
                    changed = True
                    i += 2
                else:
                    new_sequence.append(current_token)
                    new_durations.append(durations[i])
                    i += 1
            sequence = new_sequence
            durations = new_durations
        return np.array(sequence), np.array(durations)

    def encode_batch(self, sequences: List[List[int]]) -> List[List[int]]:
        encoded_sequences = []
        durations_sequences = []
        for sequence in sequences:
            encoded_sequence, durations = self.encode(sequence)
            encoded_sequences.append(encoded_sequence)
            durations_sequences.append(durations)
        return encoded_sequences, durations_sequences

    def decode(self, sequence: np.ndarray, durations: np.ndarray) -> np.ndarray:
        decoded_sequence = []
        for i in range(len(sequence)):
            decoded_sequence.append(sequence[i])
            for j in range(durations[i]):
                decoded_sequence.append(sequence[i])
        return decoded_sequence

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(n_initial_units={self.n_initial_units}, "
            f"target_vocab_size={self.target_vocab_size}, "
            f"deduplicate={self.deduplicate}, "
            f"trained={self._is_trained}, "
        )


# # Training script
# if __name__ == "__main__":
#     import argparse
#     from torch.utils.data import DataLoader
#     from data.audio_dataset import DataCollator, HubertDatasetWrapper
#     from data.librispeechHubert import LibriSpeech100h

#     parser = argparse.ArgumentParser()
#     parser.add_argument("-b", "--batch_size", type=int, default=32)
#     parser.add_argument("-n", "--n_initial_units", type=int, default=1024)
#     parser.add_argument("-k", "--target_vocab_size", type=int, default=16384)
#     parser.add_argument("-d", "--deduplicate", action="store_true")
#     parser.add_argument("-o", "--output", type=str, default="bpe_simple.json")
#     args = parser.parse_args()

#     # Load data
#     print("Loading data...")
#     data_collator = DataCollator()
#     dataset = HubertDatasetWrapper(LibriSpeech100h(), split="train")
#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         collate_fn=data_collator,
#         num_workers=0,
#         shuffle=False,
#     )

#     # Collect sequences
#     sequences = []
#     for batch in tqdm(dataloader, desc="Loading sequences"):
#         for seq in batch["audio_codes"]:
#             sequences.append(np.array([int(x) for x in seq]))
#     print(f"Loaded {len(sequences)} sequences")

#     # Train
#     tokenizer = BPETokenizer(
#         n_initial_units=args.n_initial_units,
#         target_vocab_size=args.target_vocab_size,
#         deduplicate=args.deduplicate,
#         verbose=True,
#     )
#     if args.deduplicate:
#         sequences = tokenizer._collapse_repetitions(sequences)

#     tokenizer.train(sequences, verbose=True, output_path=args.output)

#     print(f"\nSaved to {args.output}")
#     print(f"Final tokenizer: {tokenizer}")
