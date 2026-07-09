import os
import sys
import argparse
import json
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets

# Aggiungiamo minbpe al path per importare le sue funzioni core (indipendenti dal byte-level)
sys.path.append("/scratch/piermel/minbpe")
from minbpe.base import get_stats, merge

class AudioTokenizer:
    """
    Tokenizer BPE adattato per token audio (interi).
    Bypassa completamente la gestione delle stringhe UTF-8 di minbpe.
    """
    def __init__(self, base_vocab_size):
        self.base_vocab_size = base_vocab_size
        self.merges = {}
        
    def train(self, ids_list, target_vocab_size, verbose=False, return_compressed=True):
        num_merges = target_vocab_size - self.base_vocab_size
        if num_merges <= 0:
            return ids_list if return_compressed else None
            
        for i in range(num_merges):
            stats = {}
            # Calcoliamo le frequenze parallelamente su tutte le sequenze per non fare merge tra file audio diversi
            for ids in ids_list:
                seq_stats = get_stats(ids)
                for pair, count in seq_stats.items():
                    stats[pair] = stats.get(pair, 0) + count
                    
            if not stats:
                print("Nessuna coppia rimanente da unire.")
                break
                
            pair = max(stats, key=stats.get)
            idx = self.base_vocab_size + i
            
            # Applichiamo il merge a tutte le sequenze
            ids_list = [merge(ids, pair, idx) for ids in ids_list]
            self.merges[pair] = idx
            
            if verbose and (i % 100 == 0 or i == num_merges - 1):
                print(f"Merge {i+1:4d}/{num_merges}: {pair} -> nuovo token {idx} (occorrenze: {stats[pair]})")
                
        if return_compressed:
            return ids_list
        return None
        
    def save(self, filepath):
        with open(filepath, 'w') as f:
            # JSON richiede chiavi stringa, convertiamo la tupla
            merges_str = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
            json.dump({
                "base_vocab_size": self.base_vocab_size, 
                "merges": merges_str
            }, f, indent=2)
            
    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.base_vocab_size = data["base_vocab_size"]
        self.merges = {tuple(map(int, k.split(','))): v for k, v in data["merges"].items()}
        
    def encode(self, ids):
        while len(ids) >= 2:
            stats = get_stats(ids)
            # cerchiamo la coppia con l'indice di merge più basso (la prima unione creata)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    training_cfg = cfg_dict.get("training", {})
    tokenizer_cfg = cfg_dict.get("tokenizer", {})
    
    dataset_name = training_cfg.get("dataset_name", "libritts-r-prepared")
    vae_checkpoint = cfg_dict.get("vae_checkpoint", "/scratch/piermel/MelCausalVAE/checkpoints/set15-64")
    scratch_dir = os.environ.get("SCRATCH", "/scratch/piermel")
    vae_config_path = os.path.join(vae_checkpoint.replace("$SCRATCH", scratch_dir), "config.json")
    if not os.path.exists(vae_config_path) and "config.json" in vae_checkpoint:
        vae_config_path = vae_checkpoint.replace("$SCRATCH", scratch_dir)
        
    target_vocab_size = tokenizer_cfg.get("target_vocab_size", 4096)
    num_samples = tokenizer_cfg.get("num_samples", -1)
    save_path = tokenizer_cfg.get("save_path")
    return_compressed = tokenizer_cfg.get("return_compressed", True)
    
    if save_path is None:
        save_path = f"audio_bpe_{target_vocab_size}.json"
        

    # 1. Recupero Base Vocab Size dal VAE
    with open(vae_config_path, 'r') as f:
        vae_config = json.load(f)
    base_vocab_size = vae_config.get("encoder_config", {}).get("vq_config", {}).get("num_embeddings")
    if base_vocab_size is None:
        raise ValueError("Non riesco a trovare 'encoder_config' -> 'vq_config' -> 'num_embeddings' nel config del VAE")
    print(f"Base vocab size letto dal VAE: {base_vocab_size}")
    
    # 2. Caricamento Dataset
    SLURM_TMPDIR = os.getenv("SLURM_TMPDIR", "/scratch/piermel")
    parquet_dir = f"{SLURM_TMPDIR}/datasets/{dataset_name}"
    print(f"Caricamento dataset da {parquet_dir}...")
    dataset = load_dataset("parquet", data_dir=parquet_dir)
    
    partitions_per_destination = defaultdict(list)
    for partition in dataset:
        dest = "train" if "train" in partition else "test"
        partitions_per_destination[dest].append(dataset[partition])
        
    train_ds = concatenate_datasets(partitions_per_destination["train"])
    
    if num_samples > 0:
        print(f"Campionamento: seleziono un subset random di {num_samples} sample...")
        train_ds = train_ds.shuffle(seed=42).select(range(min(num_samples, len(train_ds))))
        
    print(f"Totale sequenze audio per il training: {len(train_ds)}")
    
    # Estraiamo le sequenze di token discreti
    discrete_sequences = []
    print("Estrazione dei token discreti dal dataset in memoria...")
    for seq in train_ds["discrete"]:
        discrete_sequences.append(list(seq))
        
    original_token_count = sum(len(seq) for seq in discrete_sequences)
    print(f"Conteggio totale token prima della compressione: {original_token_count}")
    
    # 3. Training BPE
    print(f"\nInizio addestramento BPE...")
    print(f"Target: {target_vocab_size} token. (Richiede {target_vocab_size - base_vocab_size} merges iterativi)")
    tokenizer = AudioTokenizer(base_vocab_size=base_vocab_size)
    
    compressed_sequences = tokenizer.train(
        ids_list=discrete_sequences, 
        target_vocab_size=target_vocab_size,
        verbose=True,
        return_compressed=return_compressed
    )
    
    # 4. Statistiche (opzionale)
    if return_compressed and compressed_sequences is not None:
        compressed_token_count = sum(len(seq) for seq in compressed_sequences)
        compression_ratio = original_token_count / compressed_token_count
        
        print("\n" + "="*40)
        print("📊 STATISTICHE DI COMPRESSIONE BPE")
        print("="*40)
        print(f"Token totali originali:   {original_token_count:,}")
        print(f"Token totali compressi:   {compressed_token_count:,}")
        print(f"Compressione Media (X):   {compression_ratio:.2f}X")
        print(f"Lunghezza media origin.:  {original_token_count/len(train_ds):.1f} token/audio")
        print(f"Lunghezza media compr.:   {compressed_token_count/len(train_ds):.1f} token/audio")
        print("="*40 + "\n")
    else:
        print("\nAllenamento BPE completato (sequenze compresse non ritornate).")
    
    # 5. Salvataggio
    tokenizer.save(save_path)
    print(f"Modello BPE salvato con successo in '{save_path}'.")

if __name__ == "__main__":
    main()
