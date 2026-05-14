import os
import sys
import torch
import argparse
import datasets
import torchaudio
import io
from datasets import load_dataset, Dataset
from typing import Dict, Any
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler

# Ensure we can import from the hybrid_tts project
from modules.submodules.MelCausalVAE.modules.VAE import VAE, VAEConfig
from modules.submodules.MelCausalVAE.modules.builder import build_model
import json
import yaml
import pyarrow as pa
import pyarrow.parquet as pq


def collate_fn(batch):
    # batch is a list of dicts. We return a dict of lists.
    return {key: [item[key] for item in batch] for key in batch[0].keys()}


def main():
    parser = argparse.ArgumentParser(
        description="Extract continuous and discrete features using MelCausalVAE"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset folder (e.g. librispeech-aligned)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to VAE checkpoint directory containing config.json and model.safetensors",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Input parquet dir. Defaults to SLURM_TMPDIR/datasets/{dataset_name}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output parquet dir. Defaults to SLURM_TMPDIR/datasets/{dataset_name}_prepared",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for inference"
    )
    parser.add_argument(
        "--shard_size_mb",
        type=int,
        default=512,
        help="Max size in MB for each parquet shard",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16 for inference"
    )
    args = parser.parse_args()

    # Setup distributed if torchrun is used
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        local_rank = 0
        global_rank = 0

    if local_rank == 0:
        print(f"Loading MelCausalVAE model from {args.checkpoint_dir}...")

    config_path = os.path.join(args.checkpoint_dir, "config.json")
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    model = build_model(cfg_dict)
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.safetensors")
    model.from_pretrained(checkpoint_path)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model.to(dtype)
    model.to(device)
    model.eval()

    slurm_tmp = os.getenv("SLURM_TMPDIR", "")
    input_dir = Path(
        args.input_dir
        if args.input_dir
        else Path(slurm_tmp) / "datasets" / args.dataset_name
    )
    output_dir = Path(
        args.output_dir
        if args.output_dir
        else Path(slurm_tmp) / "datasets" / f"{args.dataset_name}_prepared"
    )

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if local_rank == 0:
        print(f"Loading dataset from {input_dir}...")
        print(f"Saving features to {output_dir}...")
        if args.bf16:
            print("Using bfloat16 precision for inference.")

    # Identify subdirectories to treat as splits (e.g. train_clean_100, dev_clean, etc.)
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if not subdirs:
        # Fallback to the directory itself if no subdirs
        data_files = {"data": str(input_dir / "*.parquet")}
    else:
        data_files = {d.name: str(d / "*.parquet") for d in subdirs}

    dataset = load_dataset("parquet", data_files=data_files)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for split in dataset.keys():
        split_ds = dataset[split]
        sampler = (
            DistributedSampler(split_ds, shuffle=False) if is_distributed else None
        )
        # Cast audio column to NOT decode automatically to avoid torchcodec issues
        split_ds = split_ds.cast_column("audio", datasets.Audio(decode=False))

        dataloader = DataLoader(
            split_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        split_dir = out_path / split
        split_dir.mkdir(parents=True, exist_ok=True)

        shard_idx = 0

        def get_parquet_file(idx):
            return split_dir / f"data_rank{global_rank}_{idx:05d}.parquet"

        parquet_file = get_parquet_file(shard_idx)
        writer = None

        target_sr = model.config.sample_rate
        pbar = tqdm(dataloader, disable=(local_rank != 0), desc=f"Extracting {split}")
        for batch in pbar:
            audios_srs = []
            for audio_item in batch["audio"]:
                # Manually decode using torchaudio
                try:
                    wav, sr = torchaudio.load(io.BytesIO(audio_item["bytes"]))
                    tensor = wav.squeeze(0).float()

                    # Resample if necessary
                    if sr != target_sr:
                        tensor = torchaudio.functional.resample(tensor, sr, target_sr)

                    # Apply dtype
                    tensor = tensor.to(dtype)

                    audios_srs.append((tensor.to(device), target_sr))
                except Exception as e:
                    print(f"Error decoding audio: {e}")
                    # Fallback or skip
                    continue

            if not audios_srs:
                continue

            discrete_feats_batch = []
            continuous_feats_batch = []

            with torch.no_grad():
                features, padding_mask = model.extract_features(audios_srs)
                encoder_output = model.encode(features, padding_mask)
                padding_mask = encoder_output.padding_mask

                B = features.size(0)
                for i in range(B):
                    mask = ~padding_mask[i]

                    discrete_feat = (
                        encoder_output.indices[i][mask].cpu().numpy().tolist()
                    )

                    # Continuous features are always based on tail.
                    # Add residual ONLY if add_vq_residual_to_stoch is True in config.
                    continuous_tensor = encoder_output.tail[i][mask]

                    add_residual = False
                    if (
                        hasattr(model.config.encoder_config, "vq_config")
                        and model.config.encoder_config.vq_config is not None
                    ):
                        add_residual = getattr(
                            model.config.encoder_config.vq_config,
                            "add_vq_residual_to_stoch",
                            False,
                        )

                    if add_residual and encoder_output.residual is not None:
                        continuous_tensor = (
                            continuous_tensor + encoder_output.residual[i][mask]
                        )

                    continuous_feat = continuous_tensor.float().cpu().numpy().tolist()

                    discrete_feats_batch.append(discrete_feat)
                    continuous_feats_batch.append(continuous_feat)

            # Stream batch to parquet to avoid OOM
            batch_data = {col: batch[col] for col in split_ds.column_names}
            batch_data["discrete"] = discrete_feats_batch
            batch_data["continuous"] = continuous_feats_batch

            table = pa.Table.from_pydict(batch_data)
            if writer is None:
                writer = pq.ParquetWriter(parquet_file, table.schema)
            writer.write_table(table)

            # Check shard size
            if (
                parquet_file.exists()
                and parquet_file.stat().st_size >= args.shard_size_mb * 1024 * 1024
            ):
                writer.close()
                writer = None
                shard_idx += 1
                parquet_file = get_parquet_file(shard_idx)

        if writer is not None:
            writer.close()

        if local_rank == 0:
            print(f"Saved {split} (rank {global_rank}) to {parquet_file}")

    if is_distributed:
        torch.distributed.barrier()

    if local_rank == 0:
        print(f"\nAll done! Dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
