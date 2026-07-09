import json

class AudioTokenizer:
    """
    A custom BPE tokenizer that operates directly on lists of integers (tokens).
    """

    def __init__(self, base_vocab_size=512):
        self.base_vocab_size = base_vocab_size
        self.merges = {} # (pair) -> new_id
        self.vocab_size = base_vocab_size
        self.vocab = {idx: [idx] for idx in range(self.base_vocab_size)}
        
    def _build_vocab(self):
        """Builds the mapping from token_id to a list of base token ids."""
        self.vocab = {idx: [idx] for idx in range(self.base_vocab_size)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, ids):
        """
        Encodes a list of integer IDs by applying the learned merges.
        """
        # If sequence is too short, return as is
        if len(ids) < 2:
            return ids
        
        while len(ids) >= 2:
            # Find the most frequent pair in the current sequence that we have a merge for
            stats = self._get_stats(ids)
            # Find the pair with the lowest merge index (meaning it was merged earlier/is more frequent)
            # using a very large number as default to ignore pairs not in self.merges
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            
            # If the pair is not in our merges, we are done
            if pair not in self.merges:
                break
                
            # Otherwise, perform the merge
            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)
            
        return ids

    def decode(self, ids):
        """
        Decodes a list of token IDs back into the base vocabulary.
        """
        result = []
        for idx in ids:
            result.extend(self.vocab[idx])
        return result

    def _get_stats(self, ids):
        """Computes frequencies of adjacent pairs of integers."""
        stats = {}
        for pair in zip(ids, ids[1:]):
            stats[pair] = stats.get(pair, 0) + 1
        return stats

    def _merge(self, ids, pair, idx):
        """
        Replaces all consecutive occurrences of `pair` with the new token `idx`.
        """
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, data, vocab_size):
        """
        Trains the BPE tokenizer on a list of integer lists.
        `data` is a list of lists of integers (e.g. dataset samples).
        """
        assert vocab_size >= self.base_vocab_size
        num_merges = vocab_size - self.base_vocab_size
        
        # We start with the base vocabulary size
        self.vocab_size = self.base_vocab_size
        
        # To avoid recomputing stats over all data all the time if it's huge,
        # we'll flatten the data but insert a special token (or just keep it as a list of sequences)
        # Actually, it's safer to keep it as a list of sequences to avoid merging across sequence boundaries.
        sequences = [list(seq) for seq in data]
        
        for i in range(num_merges):
            # Compute stats across all sequences
            stats = {}
            for seq in sequences:
                seq_stats = self._get_stats(seq)
                for k, v in seq_stats.items():
                    stats[k] = stats.get(k, 0) + v
                    
            if not stats:
                print("No more pairs to merge!")
                break
                
            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            
            # Create a new token id
            new_id = self.vocab_size
            self.merges[best_pair] = new_id
            
            # Apply merge to all sequences
            sequences = [self._merge(seq, best_pair, new_id) for seq in sequences]
            
            self.vocab_size += 1
            if (i+1) % 100 == 0:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {new_id}")
                
        self._build_vocab()

    def save(self, filepath):
        """Saves the merges to a JSON file."""
        data = {
            "base_vocab_size": self.base_vocab_size,
            "vocab_size": self.vocab_size,
            "merges": {f"{p0},{p1}": idx for (p0, p1), idx in self.merges.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Loads merges from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        tokenizer = cls(base_vocab_size=data["base_vocab_size"])
        tokenizer.vocab_size = data.get("vocab_size", data["base_vocab_size"] + len(data["merges"]))
        
        # Parse merges
        for k, v in data["merges"].items():
            p0, p1 = map(int, k.split(','))
            tokenizer.merges[(p0, p1)] = v
            
        tokenizer._build_vocab()
        return tokenizer
