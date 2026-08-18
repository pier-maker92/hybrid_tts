import os
import io
import torch
import torchaudio
import argparse
import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from speechbrain.inference.speaker import EncoderClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_dir", type=str, required=True, help="Path to prepared parquet dir (e.g., libritts-r-prepared)")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to raw libritts-r dir with audio")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for new dataset with ECAPA")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ECAPA model on {device}...")
    ecapa_classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    
    # Load prepared dataset
    print(f"Loading prepared dataset from {args.prepared_dir}...")
    prep_subdirs = [d for d in Path(args.prepared_dir).iterdir() if d.is_dir()]
    if not prep_subdirs:
        prep_data_files = {"data": str(Path(args.prepared_dir) / "*.parquet")}
    else:
        prep_data_files = {d.name: str(d / "*.parquet") for d in prep_subdirs}
    
    prepared_ds = load_dataset("parquet", data_files=prep_data_files)
    
    # Load raw dataset
    print(f"Loading raw dataset from {args.raw_dir}...")
    raw_subdirs = [d for d in Path(args.raw_dir).iterdir() if d.is_dir()]
    if not raw_subdirs:
        raw_data_files = {"data": str(Path(args.raw_dir) / "*.parquet")}
    else:
        raw_data_files = {d.name: str(d / "*.parquet") for d in raw_subdirs}
        
    raw_ds = load_dataset("parquet", data_files=raw_data_files)
    
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Process split by split
    for split in prepared_ds.keys():
        print(f"Processing split: {split}")
        prep_split = prepared_ds[split]
        raw_split = raw_ds[split] if split in raw_ds else None
        
        if raw_split is None:
            print(f"Warning: split {split} not found in raw dataset. Looking globally...")
        
        # Build an ID to index mapping for raw dataset if it's there
        id_col = None
        for c in ['id', 'file', 'file_id', 'audio_id', 'item_id']:
            if raw_split is not None and c in raw_split.column_names:
                id_col = c
                break
        
        if id_col is None and raw_split is not None:
            print(f"Warning: Could not find ID column in raw_split columns: {raw_split.column_names}.")
            
        raw_id_to_idx = {}
        if raw_split is not None and id_col is not None:
            print("Building ID index for raw dataset...")
            for i, row in enumerate(tqdm(raw_split.select_columns([id_col]))):
                raw_id_to_idx[row[id_col]] = i
        
        has_audio = 'audio' in prep_split.column_names
        
        # We need to temporarily disable decode for audio if present so it gives dicts
        if has_audio:
            prep_split = prep_split.cast_column("audio", datasets.Audio(decode=False))
        if raw_split is not None and 'audio' in raw_split.column_names:
            raw_split = raw_split.cast_column("audio", datasets.Audio(decode=False))

        # Use datasets map to add column.
        def extract_ecapa_with_idx(batch, indices):
            ecapa_embs = []
            for i, ds_idx in enumerate(indices):
                audio_item = None
                if has_audio:
                    audio_item = batch['audio'][i]
                elif raw_split is not None:
                    # Match by ID
                    prep_id_col = id_col if id_col in batch else ('id' if 'id' in batch else None)
                    if prep_id_col and batch[prep_id_col][i] in raw_id_to_idx:
                        raw_idx = raw_id_to_idx[batch[prep_id_col][i]]
                        audio_item = raw_split[raw_idx]['audio']
                    else:
                        # Fallback to aligned index
                        audio_item = raw_split[ds_idx]['audio']
                
                if audio_item is None:
                    raise ValueError(f"Could not find audio for item index {ds_idx}")
                
                if isinstance(audio_item, dict):
                    if audio_item.get("path") and os.path.exists(audio_item["path"]):
                        wav, sr = torchaudio.load(audio_item["path"])
                    elif audio_item.get("bytes"):
                        wav, sr = torchaudio.load(io.BytesIO(audio_item["bytes"]))
                    else:
                        wav, sr = torchaudio.load(audio_item["path"])
                else: # fallback if it's just a string path
                    wav, sr = torchaudio.load(audio_item)
                
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav.squeeze(0).float(), sr, 16000)
                else:
                    wav = wav.squeeze(0).float()
                
                with torch.no_grad():
                    emb = ecapa_classifier.encode_batch(wav.to(device).unsqueeze(0))
                    ecapa_embs.append(emb.squeeze().cpu().numpy().tolist())
                    
            return {"ecapa": ecapa_embs}
            
        print(f"Extracting ECAPA for {split}...")
            
        new_split = prep_split.map(
            extract_ecapa_with_idx,
            with_indices=True,
            batched=True,
            batch_size=args.batch_size,
            desc=f"ECAPA {split}"
        )
        
        # Save to parquet
        split_out_dir = out_path / split
        split_out_dir.mkdir(parents=True, exist_ok=True)
        out_parquet_path = split_out_dir / "data-00000.parquet"
        print(f"Saving to {out_parquet_path}")
        new_split.to_parquet(out_parquet_path)

if __name__ == "__main__":
    main()
