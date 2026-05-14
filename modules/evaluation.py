import os
import json
import torch
import torchaudio
import wandb
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
try:
    from jiwer import wer as compute_wer, cer as compute_cer
except ImportError:
    compute_wer = compute_cer = None
try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
except ImportError:
    WhisperProcessor = WhisperForConditionalGeneration = None

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
        if self.model is None:
            return ""
            
        # Whisper expects 16kHz mono
        if audio.dim() > 1:
            audio = audio.mean(dim=0)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        input_features = self.processor(
            audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device).float()

        # Generate transcription - force float32 on MPS
        is_mps = self.device.type == "mps"
        if is_mps: self.model.float()
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

    for batch in eval_dataloader:
        if processed_count >= num_samples:
            break

        # Check if batch has original audio or if we should reconstruct from tokens
        # For HybridTTS, we likely reconstruct from Prompt + Discrete + Continuous
        
        gt_texts = [t.lower().strip() for t in batch.get("transcription", [])]
        sample_ids = batch.get("ids", [f"sample_{processed_count + i}" for i in range(len(gt_texts))])

        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            # Ensure models are in float32 for evaluation on MPS if needed
            is_mps = device.type == "mps"
            
            # For encode_decode, we do it manually to handle the nested structures
            orig_model_dtype = next(model.parameters()).dtype
            orig_vae_dtype = next(vae.parameters()).dtype if vae else None
            
            if is_mps:
                model.float()
                if vae: vae.float()
                if vocoder: vocoder.float()

            try:
                reconstruction_results = model.encode_decode(
                    batch=batch,
                    vae=vae,
                    num_steps=16,
                    temperature=0.2,
                    guidance_scale=1.3,
                )

                reconstructed_mels = reconstruction_results["decoder_output"].audio_features
                reconstructed_masks = reconstruction_results["decoder_output"].padding_mask
            finally:
                if is_mps:
                    model.to(orig_model_dtype)
                    if vae: vae.to(orig_vae_dtype)
                    # Vocoder usually stays in float32 anyway
            
            # If batch has original audio, use it for GT metrics
            # But in the aligned dataset, we might not have it.
            # If missing, we skip GT MOS but can still do WER against gt_text.

        for i in range(len(gt_texts)):
            if processed_count >= num_samples:
                break

            sid = str(sample_ids[i])
            gt_text = gt_texts[i]
            
            # 1. Get Ground Truth Metrics (if possible)
            if sid in gt_cache:
                gt_metrics = gt_cache[sid]
            else:
                # If we had original audio, we would compute UTMOS here.
                # For now, let's assume we might not have it or it's cached.
                gt_metrics = {"UTMOS": None, "WER": 0.0, "CER": 0.0} 
                gt_cache[sid] = gt_metrics

            # 2. Reconstruct audio from mel
            mel = reconstructed_mels[i]
            mask = reconstructed_masks[i]
            # mel: [L, C] -> [1, C, L]
            mel = mel[~mask].unsqueeze(0).permute(0, 2, 1).float().to(device)

            with torch.no_grad():
                if vocoder is not None:
                    # Vocos uses .decode(mel), BigVGAN usually uses the forward call vocoder(mel)
                    if hasattr(vocoder, "decode"):
                        recon_audio = vocoder.decode(mel).cpu().squeeze()
                    else:
                        recon_audio = vocoder(mel).cpu().squeeze()
                else:
                    logger.warning("No vocoder provided. Skipping audio generation.")
                    recon_audio = None

            if recon_audio is not None:
                recon_audio = recon_audio / (recon_audio.abs().max() + 1e-8)
                sr = 24000 # FIXME: assume 24kHz

                recon_wav_path = temp_wav_dir / f"{sid}_recon.wav"
                torchaudio.save(str(recon_wav_path), recon_audio.unsqueeze(0).cpu(), sr)

                recon_utmos = utmos_predictor.predict(str(recon_wav_path))
                recon_transcription = asr_model.transcribe(recon_audio, sr)
                
                if compute_wer:
                    recon_wer = compute_wer(gt_text, recon_transcription)
                    recon_cer = compute_cer(gt_text, recon_transcription)
                else:
                    recon_wer = recon_cer = 0.0
            else:
                recon_utmos = None
                recon_wer = recon_cer = 1.0

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
