#!/usr/bin/env python
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch
import torchaudio
from datasets import load_dataset

SR = 24000


def audio_to_tensor(audio_obj):
    if isinstance(audio_obj, dict):
        wav = torch.as_tensor(audio_obj["array"]).float()
        sr = int(audio_obj["sampling_rate"])
    elif hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        wav = torch.as_tensor(samples.data).float()
        sr = int(samples.sample_rate)
    else:
        raise TypeError(f"Unsupported audio object: {type(audio_obj)}")
    if wav.ndim == 2:
        wav = wav.mean(dim=0) if wav.shape[0] <= 8 else wav.mean(dim=-1)
    wav = wav.squeeze().contiguous()
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak
    return wav, sr


def load_samples(dataset_root, num_samples, max_ref_seconds):
    files = sorted(
        f
        for f in glob.glob(os.path.join(dataset_root, "test_clean", "*.parquet"))
        if not os.path.basename(f).startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No test_clean parquet files under {dataset_root}")
    ds = load_dataset("parquet", data_files=files, split="train")
    samples = []
    for idx, row in enumerate(ds):
        wav, sr = audio_to_tensor(row["audio"])
        duration = wav.numel() / float(sr)
        text = row.get("transcript") or row.get("text") or row.get("transcription") or ""
        if duration <= max_ref_seconds and text.strip():
            samples.append(
                {
                    "id": str(row.get("id", idx)),
                    "text": text.strip(),
                    "wav": wav,
                    "sr": sr,
                    "duration": duration,
                }
            )
        if len(samples) >= num_samples:
            break
    samples.sort(key=lambda x: x["duration"], reverse=True)
    return samples


def pad_pair(hyp, ref):
    hyp = hyp.squeeze().float()
    ref = ref.squeeze().float()
    hyp_len = hyp.numel()
    ref_len = ref.numel()
    max_len = max(hyp_len, ref_len, 1)
    if hyp_len < max_len:
        hyp = torch.nn.functional.pad(hyp, (0, max_len - hyp_len))
    if ref_len < max_len:
        ref = torch.nn.functional.pad(ref, (0, max_len - ref_len))
    lens = torch.tensor([max(hyp_len, ref_len) / max_len], dtype=torch.float32)
    return hyp.unsqueeze(0), ref.unsqueeze(0), lens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--generated_dir", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--audiocodecs_root", default="/scratch/piermel/audiocodecs")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--max_ref_seconds", type=float, default=20.0)
    p.add_argument("--dwer_model", default="small")
    p.add_argument("--dwer_device", default="cuda")
    args = p.parse_args()

    sys.path.insert(0, os.path.join(args.audiocodecs_root, "downstream"))
    from metrics.dwer import DWER
    from metrics.speaker_similarity import SpkSimWavLM
    from metrics.utmos import UTMOS
    sys.path.pop(0)

    hf_cache = os.path.join(os.environ.get("HF_HOME", "/scratch/piermel/.cache/huggingface"), "hub")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_samples(args.dataset_root, args.num_samples, args.max_ref_seconds)

    dwer_metric = DWER(args.dwer_model, SR, save_path=hf_cache, device=args.dwer_device)
    utmos_metric = UTMOS(SR)
    spksim_metric = SpkSimWavLM("microsoft/wavlm-base-sv", SR, save_path=hf_cache)

    rows = []
    start = time.time()
    for idx, sample in enumerate(samples):
        hyp_path = Path(args.generated_dir) / f"{sample['id'].replace('/', '_')}.wav"
        if not hyp_path.exists():
            raise FileNotFoundError(f"Generated wav missing for {sample['id']}: {hyp_path}")
        hyp, hyp_sr = torchaudio.load(str(hyp_path))
        if hyp_sr != SR:
            hyp = torchaudio.functional.resample(hyp, hyp_sr, SR)

        ref = sample["wav"].unsqueeze(0)
        if sample["sr"] != SR:
            ref = torchaudio.functional.resample(ref, sample["sr"], SR)
        ref = ref / (ref.abs().max() + 1e-8)

        hyp_pad, ref_pad, lens = pad_pair(hyp, ref)
        hyp_pad = hyp_pad.to(device)
        ref_pad = ref_pad.to(device)
        lens = lens.to(device)

        dwer_metric.append([sample["id"]], hyp_pad, ref_pad, lens=lens)
        utmos_metric.append([sample["id"]], hyp.to(device))
        spksim_metric.append([sample["id"]], hyp_pad, ref_pad, lens=lens)

        rows.append(
            {
                "id": sample["id"],
                "ref_text": sample["text"],
                "ref_duration_s": sample["duration"],
                "hyp_duration_s": hyp.shape[-1] / float(SR),
                "hyp_path": str(hyp_path),
            }
        )
        if (idx + 1) % 10 == 0 or idx + 1 == len(samples):
            partial = {"progress": idx + 1, "total": len(samples)}
            with open(args.output_json, "w") as f:
                json.dump(partial, f, indent=2)
            print(json.dumps(partial), flush=True)

    dwer_summary = dwer_metric.summarize()
    summary = {
        "num_samples": len(rows),
        "official_metrics": {
            "dWER": float(dwer_summary["error_rate"]),
            "dCER": float(dwer_summary["error_rate_char"]),
            "UTMOS": float(utmos_metric.summarize("average")),
            "SpkSimWavLM": float(spksim_metric.summarize("average")),
        },
        "config": {
            "dwer_model": args.dwer_model,
            "dwer_device": args.dwer_device,
            "spksim_model": "microsoft/wavlm-base-sv",
            "dataset_root": args.dataset_root,
            "generated_dir": args.generated_dir,
            "max_ref_seconds": args.max_ref_seconds,
        },
        "seconds": time.time() - start,
        "samples": rows,
    }
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["official_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
