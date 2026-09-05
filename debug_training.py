#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
from pathlib import Path


CRUCIAL_ROOT = Path("/Volumes/Crucial X6/Research")
SOURCE_DATASET_ROOT = CRUCIAL_ROOT / "Datasets" / "librispeech-aligned"
PARTITIONS = ("train_clean_100", "dev_clean")
DEFAULT_ENV_PYTHON = Path("/opt/miniconda3/envs/hybrid_tts/bin/python")


def has_parquet_data(path: Path) -> bool:
    return any(
        child.is_file()
        and child.suffix == ".parquet"
        and not child.name.startswith("._")
        for child in path.glob("*.parquet")
    )


def ensure_partition_link(source_root: Path, target_root: Path, partition: str) -> None:
    source = source_root / partition
    target = target_root / partition

    if not source.is_dir():
        raise FileNotFoundError(f"Missing LibriSpeech partition on Crucial: {source}")

    target_root.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        if target.is_dir() and has_parquet_data(target):
            print(f"Using existing dataset partition: {target}")
            return
        print(f"Using existing dataset path: {target}")
        return

    target.symlink_to(source, target_is_directory=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare local debug environment and launch HybridTTS training."
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides to append after settings=debug.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prepared environment and command without running training.",
    )
    args = parser.parse_args()

    if not CRUCIAL_ROOT.is_dir():
        raise FileNotFoundError(f"Crucial volume is not mounted: {CRUCIAL_ROOT}")

    slurm_tmpdir = Path.home() / "Research"
    hf_home = CRUCIAL_ROOT / ".cache" / "huggingface"
    dataset_target_root = slurm_tmpdir / "datasets" / "librispeech-aligned"

    for partition in PARTITIONS:
        ensure_partition_link(SOURCE_DATASET_ROOT, dataset_target_root, partition)

    if not args.dry_run:
        hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["SLURM_TMPDIR"] = str(slurm_tmpdir)
    python_executable = os.environ.get("HYBRID_TTS_PYTHON")
    if python_executable is None:
        python_executable = (
            str(DEFAULT_ENV_PYTHON) if DEFAULT_ENV_PYTHON.exists() else sys.executable
        )

    command = [
        python_executable,
        "train.py",
        "settings=debug",
        "training.online_encode=true",
        "training.dataset_partitions=[train_clean_100,dev_clean]",
    ]
    command.extend(args.overrides)

    print(f"HF_HOME={os.environ['HF_HOME']}")
    print(f"SLURM_TMPDIR={os.environ['SLURM_TMPDIR']}")
    print(f"Dataset root={dataset_target_root}")
    print("Command:", " ".join(command))

    if args.dry_run:
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
