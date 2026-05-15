import os
import json
import wandb
import torch
import logging
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any, Optional

from jiwer import wer as compute_wer, cer as compute_cer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

logger = logging.getLogger(__name__)


class UTMOSPredictor:
    """Standalone UTMOS predictor using utmosv2."""

    def __init__(self, device: torch.device):
        logger.info("Initializing UTMOSv2 model")
        try:
            import utmosv2

            self.model = utmosv2.create_model(pretrained=True, device=str(device))
        except ImportError:
            logger.warning("utmosv2 not found. UTMOS prediction will be disabled.")
            self.model = None
        self.device = device

    @torch.no_grad()
    def predict(self, wav_path: str) -> Optional[float]:
        if self.model is None:
            return None
        # Disable utmosv2 multiprocessing to avoid issues
        mos = self.model.predict(
            input_path=str(wav_path),
            device=str(self.device),
            num_workers=0,
        )
        return float(mos)


class WhisperASR:
    """Whisper based ASR for WER/CER computation."""

    def __init__(
        self, device: torch.device, model_name: str = "openai/whisper-large-v3"
    ):
        if WhisperProcessor is None:
            logger.warning("Transformers not found. Whisper ASR will be disabled.")
            self.model = None
            return

        logger.info(f"Loading Whisper ASR model: {model_name}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(
            device
        )
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def transcribe(self, audio: torch.Tensor, sr: int) -> str:
        """Transcribe audio tensor and return lowercase text."""

        # Whisper expects 16kHz mono
        if audio.dim() > 1:
            audio = audio.mean(dim=0)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        # Whisper processor expects numpy/list on CPU
        audio_input = (
            audio.detach().cpu().float().numpy()
            if isinstance(audio, torch.Tensor)
            else audio
        )

        input_features = self.processor(
            audio_input, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return transcription.lower().strip()


def run_evaluation(
    model: torch.nn.Module,
    vae: torch.nn.Module,
    vocoder: torch.nn.Module,
    vocoder_type: str,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    dataset_name: str,
    num_samples: int = 100,
    run_id: str = "default_run",
) -> Dict[str, float]:
    """
    Perform evaluation on samples from the test set.
    Calculates UTMOS, WER, CER for ground truth and reconstructed audio.
    """
    model.eval()
    
    # Ensure VAE and Vocoder are on the correct device and in eval mode.
    # The main model is assumed to be managed by the trainer/accelerator.
    if vae is not None:
        vae.to(device).eval()
    if vocoder is not None:
        vocoder.to(device).eval()

    try:
        eval_dir = Path("evaluation")
        gt_cache_path = eval_dir / f"{dataset_name}_ground_truth.json"

        csv_dir = eval_dir / "validation_training" / run_id
        csv_dir.mkdir(parents=True, exist_ok=True)

        gt_cache = {}
        if gt_cache_path.exists():
            try:
                with open(gt_cache_path, "r") as f:
                    gt_cache = json.load(f)
                logger.info(f"Loaded ground truth cache from {gt_cache_path}")
            except Exception as e:
                logger.warning(f"Failed to load ground truth cache: {e}")
                gt_cache = {}

        utmos_predictor = UTMOSPredictor(device)
        asr_model = WhisperASR(device)

        samples_metrics = []
        processed_count = 0

        temp_wav_dir = eval_dir / "temp_wavs" / run_id
        temp_wav_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting evaluation on up to {num_samples} samples...")

        for batch in tqdm(eval_dataloader):
            if processed_count >= num_samples:
                break

            gt_texts = [t.lower().strip() for t in batch.get("transcription", [])]
            sample_ids = batch.get(
                "ids", [f"sample_{processed_count + i}" for i in range(len(gt_texts))]
            )

            # Move batch to device
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            with torch.no_grad():
                # Use bfloat16 autocast for the model
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    reconstruction_results = model.encode_decode(
                        batch=batch,
                        vae=vae,
                        num_steps=16,
                        temperature=0.2,
                        guidance_scale=1.3,
                    )

                reconstructed_mels = reconstruction_results[
                    "decoder_output"
                ].audio_features
                reconstructed_masks = reconstruction_results[
                    "decoder_output"
                ].padding_mask

            for i in range(len(gt_texts)):
                if processed_count >= num_samples:
                    break

                sid = str(sample_ids[i])
                gt_text = gt_texts[i]

                # 1. Get Ground Truth Metrics (if possible)
                if sid in gt_cache:
                    gt_metrics = gt_cache[sid]
                else:
                    gt_metrics = {"UTMOS": None, "WER": 0.0, "CER": 0.0}
                    gt_cache[sid] = gt_metrics

                # 2. Reconstruct audio from mel
                mel = reconstructed_mels[i]
                mask = reconstructed_masks[i]
                # mel: [L, C] -> [1, C, L]
                mel = mel[~mask].unsqueeze(0).permute(0, 2, 1).float().to(device)

                with torch.no_grad():
                    recon_audio = vocoder.decode(mel).squeeze()

                recon_audio = recon_audio / (recon_audio.abs().max() + 1e-8)
                sr = 24000  # FIXME: assume 24kHz

                recon_wav_path = temp_wav_dir / f"{sid}_recon.wav"
                torchaudio.save(str(recon_wav_path), recon_audio.unsqueeze(0).cpu(), sr)

                recon_utmos = utmos_predictor.predict(str(recon_wav_path))
                with torch.cuda.amp.autocast():
                    recon_transcription = asr_model.transcribe(recon_audio, sr)

                recon_wer = compute_wer(gt_text, recon_transcription)
                recon_cer = compute_cer(gt_text, recon_transcription)

                sample_res = {
                    "id": sid,
                    "run_id": run_id,
                    "step": step,
                    "gt_text": gt_text,
                    "gt_UTMOS": gt_metrics["UTMOS"],
                    "recon_UTMOS": recon_utmos,
                    "recon_WER": recon_wer,
                    "recon_CER": recon_cer,
                }
                samples_metrics.append(sample_res)
                processed_count += 1

                if recon_audio is not None and recon_wav_path.exists():
                    recon_wav_path.unlink()

        # Save GT cache
        try:
            with open(gt_cache_path, "w") as f:
                json.dump(gt_cache, f, indent=4)
        except Exception as e:
            logger.warning(f"Failed to save ground truth cache: {e}")

        df = pd.DataFrame(samples_metrics)
        csv_path = csv_dir / f"val_step_{step}.csv"
        df.to_csv(csv_path, index=False)

        summary_metrics = {
            "eval/avg_UTMOS": df["recon_UTMOS"].mean() if "recon_UTMOS" in df else 0.0,
            "eval/avg_WER": df["recon_WER"].mean() if "recon_WER" in df else 0.0,
            "eval/avg_CER": df["recon_CER"].mean() if "recon_CER" in df else 0.0,
        }

        if wandb.run is not None:
            table = wandb.Table(dataframe=df)
            wandb.log({"eval/metrics_table": table, **summary_metrics})

        return summary_metrics

    finally:
        logger.info("Evaluation loop completed.")
