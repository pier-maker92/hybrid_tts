import os
import io
import json
import torch
import argparse
import datasets
import torchaudio
import pyarrow as pa
from tqdm import tqdm
from pathlib import Path
import pyarrow.parquet as pq
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from modules.submodules.MelCausalVAE.dicodec.modules.builder import build_model


def collate_fn(batch):
    # batch is a list of dicts. We return a dict of lists.
    return {key: [item[key] for item in batch] for key in batch[0].keys()}


def load_kmeans_codebook(path: str):
    if path is None:
        return None
    if os.path.isdir(path):
        path = os.path.join(path, "encoder_kmeans.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"K-means checkpoint not found: {path}")
    codebook = torch.load(path, map_location="cpu")
    if "centroids" not in codebook:
        raise ValueError(f"K-means checkpoint has no centroids: {path}")
    return codebook


def get_kmeans_latent_slice(codebook, latent_dim: int):
    selection = codebook.get("latent_selection")
    if selection is None:
        start = 0
        end = int(codebook["feature_dims"])
    elif selection.get("indices") is not None:
        raise ValueError("Non-contiguous k-means latent indices are not supported here.")
    else:
        start = int(selection.get("start", 0))
        end = int(selection["end"])
    if start != 0:
        raise ValueError(
            f"Expected k-means slice to start at 0, got [{start}:{end}]."
        )
    if end <= start or end > latent_dim:
        raise ValueError(
            f"K-means latent slice [{start}:{end}] is incompatible with dim {latent_dim}."
        )
    return start, end


def assign_kmeans_tokens(latents, centroids, chunk_size: int):
    assignments = []
    latents = latents.to(dtype=torch.float32)
    centroids = centroids.to(device=latents.device, dtype=torch.float32)
    for chunk in latents.split(chunk_size):
        distances = (chunk[:, None, :] - centroids[None, :, :]).square().sum(dim=-1)
        assignments.append(distances.argmin(dim=1))
    return torch.cat(assignments, dim=0)


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
    parser.add_argument(
        "--kmeans_path",
        type=str,
        default=None,
        help="Path to encoder_kmeans.pt or a directory containing it. If set, discrete tokens are nearest k-means centroids.",
    )
    parser.add_argument(
        "--continuous_start",
        type=int,
        default=None,
        help="First latent dimension to keep as continuous tokens. Defaults to the end of the k-means latent slice.",
    )
    parser.add_argument(
        "--kmeans_chunk_size",
        type=int,
        default=16384,
        help="Chunk size for nearest-centroid assignment.",
    )
    args, _ = parser.parse_known_args()

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

    # Keep feature extraction in fp32. The dicodec checkpoints include WavLM,
    # whose input pipeline produces fp32 tensors; casting the full model to bf16
    # causes WavLM conv input/weight dtype mismatches.
    if args.bf16 and local_rank == 0:
        print("Ignoring --bf16 for feature extraction; using float32.")
    dtype = torch.float32
    model.to(dtype)
    model.to(device)
    model.eval()

    kmeans_codebook = load_kmeans_codebook(args.kmeans_path)
    kmeans_centroids = None
    kmeans_start = None
    kmeans_end = None
    if kmeans_codebook is not None:
        kmeans_start, kmeans_end = get_kmeans_latent_slice(
            kmeans_codebook, int(model.config.latent_dim)
        )
        if args.continuous_start is None:
            args.continuous_start = kmeans_end
        kmeans_centroids = kmeans_codebook["centroids"].to(device=device)
        if local_rank == 0:
            print(
                f"Using k-means tokens from dims [{kmeans_start}:{kmeans_end}], "
                f"continuous dims [{args.continuous_start}:]."
            )
    elif args.continuous_start is None:
        args.continuous_start = 0

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
            print("Requested bfloat16, but feature extraction is forced to float32.")

    # Identify dataset type based on name
    if "LJSpeech-1.1" in args.dataset_name:
        dataset = load_dataset(
            "csv",
            data_files={"train": str(input_dir / "metadata.csv")},
            sep="|",
            column_names=["id", "transcription", "normalized_transcription"],
            quoting=3,  # QUOTE_NONE
        )
        for split in dataset.keys():
            dataset[split] = dataset[split].filter(
                lambda x: x["transcription"] is not None and len(str(x["transcription"]).strip()) > 0
            )
            dataset[split] = dataset[split].map(
                lambda x: {"audio": str(input_dir / "wavs" / f"{x['id']}.wav")}
            )
    else:
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
                    if audio_item.get("path") and os.path.exists(audio_item["path"]):
                        wav, sr = torchaudio.load(audio_item["path"])
                    else:
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
                features, padding_mask, _, _, _ = model.extract_features(audios_srs)
                encoder_output = model.encode(features, padding_mask)
                padding_mask = encoder_output.padding_mask

                B = features.size(0)
                for i in range(B):
                    mask = ~padding_mask[i]

                    # Continuous features are always based on tail/z. If an external
                    # k-means codebook is provided, nearest centroids on the configured
                    # leading latent dims become the discrete tokens, and the remaining
                    # dims become the continuous tokens.
                    latent_source = (
                        encoder_output.tail
                        if encoder_output.tail is not None
                        else encoder_output.z
                    )
                    continuous_tensor = latent_source[i][mask]
                    if kmeans_centroids is not None:
                        selected_for_kmeans = continuous_tensor[:, kmeans_start:kmeans_end]
                        discrete_feat = assign_kmeans_tokens(
                            selected_for_kmeans,
                            kmeans_centroids,
                            args.kmeans_chunk_size,
                        ).cpu().numpy().tolist()
                        continuous_tensor = continuous_tensor[:, args.continuous_start:]
                    else:
                        discrete_feat = (
                            encoder_output.indices[i][mask].cpu().numpy().tolist()
                        )

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
