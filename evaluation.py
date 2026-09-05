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
from contextlib import nullcontext
from typing import Dict, List, Any, Optional

from jiwer import wer as compute_wer, cer as compute_cer
from faster_whisper import WhisperModel
from transformers import WhisperTokenizer

logger = logging.getLogger(__name__)


def _empty_generation_metrics(
    sid: str,
    run_id: str,
    step: int,
    gt_text: str,
    gt_metrics: Dict[str, float],
    ref_asr_text: Optional[str] = None,
) -> Dict[str, Any]:
    recon_dwer = compute_wer(ref_asr_text, "") if ref_asr_text else None
    recon_dcer = compute_cer(ref_asr_text, "") if ref_asr_text else None
    return {
        "id": sid,
        "run_id": run_id,
        "step": step,
        "gt_text": gt_text,
        "ref_asr_text": ref_asr_text,
        "recon_asr_text": "",
        "gt_UTMOS": gt_metrics.get("UTMOS"),
        "recon_UTMOS": None,
        "recon_WER": compute_wer(gt_text, ""),
        "recon_CER": compute_cer(gt_text, ""),
        "recon_dWER": recon_dwer,
        "recon_dCER": recon_dcer,
    }


def _metric_mean(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return 0.0
    values = df[column].dropna()
    if values.empty:
        return 0.0
    return float(values.mean())


def _codebook_lookup(quantizer_module: torch.nn.Module, tokens: torch.Tensor):
    if hasattr(quantizer_module, "embedding"):
        embedding = quantizer_module.embedding
        if isinstance(embedding, torch.nn.Embedding):
            return embedding(tokens)
        return torch.nn.functional.embedding(
            tokens,
            embedding.to(device=tokens.device),
        )

    if hasattr(quantizer_module, "codebook"):
        return torch.nn.functional.embedding(
            tokens,
            quantizer_module.codebook.to(device=tokens.device),
        )

    if hasattr(quantizer_module, "toks_to_codes"):
        return quantizer_module.toks_to_codes(tokens)

    raise RuntimeError(
        "External semantic quantizer does not expose embedding/codebook/toks_to_codes."
    )


def _decode_external_quantizer_tokens(
    vae: torch.nn.Module,
    tokens: torch.Tensor,
) -> Optional[torch.Tensor]:
    semantic_quantizer = getattr(vae, "external_semantic_quantizer", None)
    if semantic_quantizer is None:
        return None

    wrapper = getattr(semantic_quantizer, "quantizer", None)
    quantizer_module = getattr(wrapper, "quantizer_module", wrapper)
    if quantizer_module is None:
        raise RuntimeError("External semantic quantizer has no quantizer module.")

    param = next(semantic_quantizer.parameters(), None)
    dtype = param.dtype if param is not None else getattr(vae, "dtype", torch.float32)
    codes = _codebook_lookup(quantizer_module, tokens).unsqueeze(0).to(dtype=dtype)
    if hasattr(semantic_quantizer, "decoder"):
        return semantic_quantizer.decoder(codes)
    return codes


class UTMOSPredictor:
    """UTMOS predictor using tarepan/SpeechMOS."""

    def __init__(self, device: torch.device):
        logger.info("Initializing UTMOS model")
        try:
            self.model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
            ).to(device)
            self.model.eval()
        except Exception as e:
            logger.warning(f"Failed to load UTMOS model: {e}")
            self.model = None
        self.device = device
        self.sample_rate = 16000

    @torch.no_grad()
    def predict(self, audio: torch.Tensor, sr: int) -> Optional[float]:
        if self.model is None:
            return None
            
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            
        # Expects audio in shape [1, length]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        audio = audio.to(self.device)
        score = self.model(audio, self.sample_rate)
        return float(score.cpu().item())


class WhisperASR:
    """faster-whisper based ASR for WER computation."""

    def __init__(self, device: torch.device, model_name: str = "small"):
        logger.info(f"Loading faster-whisper ASR model: {model_name}")
        try:
            whisper_device = "cuda" if device.type == "cuda" else "cpu"
            compute_type = "float16" if whisper_device == "cuda" else "float32"
            self.model = WhisperModel(
                model_name,
                device=whisper_device,
                compute_type=compute_type,
            )
            self.tokenizer = WhisperTokenizer.from_pretrained(f"openai/whisper-{model_name}")
        except Exception as e:
            logger.warning(f"Failed to load faster-whisper model: {e}")
            self.model = None
            self.tokenizer = None
        self.device = device

    def transcribe(self, audio: torch.Tensor, sr: int) -> str:
        """Transcribe audio tensor and return normalized text."""
        if self.model is None:
            return ""

        # Whisper expects 16kHz mono
        if audio.dim() > 1:
            audio = audio.mean(dim=0)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        # faster-whisper expects numpy array on CPU
        audio_input = audio.detach().cpu().numpy()

        segs, _ = self.model.transcribe(
            audio_input,
            beam_size=1,
            language="en",
            without_timestamps=True,
        )
        
        text = ""
        for seg in segs:
            text += seg.text
            
        if self.tokenizer is not None:
            text = self.tokenizer.normalize(text)
            
        return text.strip()


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
            reference_audios_srs = batch.get("reference_audios_srs")
            decoder_speaker_embeddings = None
            if (
                vae is not None
                and reference_audios_srs is not None
                and hasattr(vae, "extract_speaker_embedding")
            ):
                with torch.no_grad():
                    decoder_speaker_embeddings = vae.extract_speaker_embedding(
                        reference_audios_srs[: len(gt_texts)]
                    )

            with torch.no_grad():
                autocast_context = (
                    torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with autocast_context:
                    reconstruction_results = model.sample(
                        batch=batch,
                        num_steps=16,
                        temperature=0.2,
                        guidance_scale=1.3,
                        max_steps=250,
                        reference_audios_srs=reference_audios_srs,
                        voice_conditioner=vae,
                    )

                reconstructed_mels = reconstruction_results["continuous_tokens"]
                reconstructed_lengths = reconstruction_results["discrete_lengths"]
                reconstructed_discrete = reconstruction_results["discrete_tokens"]

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

                if hasattr(asr_model, "tokenizer") and asr_model.tokenizer is not None:
                    gt_text = asr_model.tokenizer.normalize(gt_text).strip()

                ref_audio = None
                ref_sr = None
                ref_asr_text = None
                if reference_audios_srs is not None and i < len(reference_audios_srs):
                    ref_audio, ref_sr = reference_audios_srs[i]
                    if gt_metrics.get("UTMOS") is None:
                        gt_metrics["UTMOS"] = utmos_predictor.predict(
                            ref_audio,
                            ref_sr,
                        )
                    ref_asr_text = asr_model.transcribe(ref_audio, ref_sr)

                # 2. Reconstruct audio from mel
                if reconstructed_mels is None:
                    logger.warning(
                        "Generated no continuous tokens for sample %s; recording "
                        "empty reconstruction.",
                        sid,
                    )
                    samples_metrics.append(
                        _empty_generation_metrics(
                            sid=sid,
                            run_id=run_id,
                            step=step,
                            gt_text=gt_text,
                            gt_metrics=gt_metrics,
                            ref_asr_text=ref_asr_text,
                        )
                    )
                    processed_count += 1
                    continue
                z_acoustic = reconstructed_mels[i]
                mel_len = int(reconstructed_lengths[i].item())
                mel_len = min(mel_len, z_acoustic.shape[0])
                if mel_len <= 0:
                    logger.warning(
                        "Generated EOS before any audio tokens for sample %s; "
                        "recording empty reconstruction.",
                        sid,
                    )
                    samples_metrics.append(
                        _empty_generation_metrics(
                            sid=sid,
                            run_id=run_id,
                            step=step,
                            gt_text=gt_text,
                            gt_metrics=gt_metrics,
                            ref_asr_text=ref_asr_text,
                        )
                    )
                    processed_count += 1
                    continue
                z_acoustic = z_acoustic[:mel_len].unsqueeze(0).float().to(device) # shape: [1, L, C]

                with torch.no_grad():
                    if vae is not None:
                        padding_mask = torch.zeros((1, mel_len), dtype=torch.bool, device=device)
                        speaker_embedding = None
                        if decoder_speaker_embeddings is not None:
                            speaker_embedding = decoder_speaker_embeddings[i : i + 1]
                        tokens_tensor = None
                        if reconstructed_discrete is not None:
                            tokens_tensor = reconstructed_discrete[i][:mel_len, 0].to(device) # shape: [L]

                        if hasattr(vae, "encoder") and hasattr(vae.encoder, "vq") and tokens_tensor is not None:
                            vq_emb = vae.encoder.vq.codebook(tokens_tensor).unsqueeze(0) # shape: [1, L, C_vq]

                            mel, mel_mask = vae.sample(
                                num_steps=16,
                                temperature=0.2,
                                guidance_scale=1.4,
                                z_semantic=vq_emb,
                                z_acoustic=z_acoustic,
                                padding_mask=padding_mask,
                                speaker_embedding=speaker_embedding,
                            )
                        else:
                            z = z_acoustic
                            if tokens_tensor is not None:
                                z_quantized = _decode_external_quantizer_tokens(
                                    vae,
                                    tokens_tensor,
                                )
                                if z_quantized is not None:
                                    z = z_quantized + z_acoustic

                            mel, mel_mask = vae.sample(
                                num_steps=16,
                                temperature=0.2,
                                guidance_scale=1.4,
                                z=z,
                                padding_mask=padding_mask,
                                speaker_embedding=speaker_embedding,
                            )
                        mel = mel[0][~mel_mask[0]].unsqueeze(0).permute(0, 2, 1).float().to(device)
                    else:
                        # Fallback if no VAE is provided
                        mel = z_acoustic.permute(0, 2, 1)

                    recon_audio = vocoder.decode(mel).squeeze()

                recon_audio = recon_audio / (recon_audio.abs().max() + 1e-8)
                sr = 24000  # FIXME: assume 24kHz

                recon_utmos = utmos_predictor.predict(recon_audio, sr)
                recon_transcription = asr_model.transcribe(recon_audio, sr)

                recon_wer = compute_wer(gt_text, recon_transcription)
                recon_cer = compute_cer(gt_text, recon_transcription)
                recon_dwer = (
                    compute_wer(ref_asr_text, recon_transcription)
                    if ref_asr_text
                    else None
                )
                recon_dcer = (
                    compute_cer(ref_asr_text, recon_transcription)
                    if ref_asr_text
                    else None
                )

                sample_res = {
                    "id": sid,
                    "run_id": run_id,
                    "step": step,
                    "gt_text": gt_text,
                    "ref_asr_text": ref_asr_text,
                    "recon_asr_text": recon_transcription,
                    "gt_UTMOS": gt_metrics.get("UTMOS"),
                    "recon_UTMOS": recon_utmos,
                    "recon_WER": recon_wer,
                    "recon_CER": recon_cer,
                    "recon_dWER": recon_dwer,
                    "recon_dCER": recon_dcer,
                }
                samples_metrics.append(sample_res)
                processed_count += 1


        # Save GT cache
        try:
            with open(gt_cache_path, "w") as f:
                json.dump(gt_cache, f, indent=4)
        except Exception as e:
            logger.warning(f"Failed to save ground truth cache: {e}")

        df = pd.DataFrame(samples_metrics)
        csv_path = csv_dir / f"val_step_{step}.csv"
        df.to_csv(csv_path, index=False)

        avg_utmos_ref = _metric_mean(df, "gt_UTMOS")
        avg_utmos = _metric_mean(df, "recon_UTMOS")
        avg_dwer = _metric_mean(df, "recon_dWER")
        summary_metrics = {
            "eval/avg_UTMOS": avg_utmos,
            "eval/avg_UTMOS_ref": avg_utmos_ref,
            "eval/avg_dUTMOS": avg_utmos - avg_utmos_ref,
            "eval/avg_WER": _metric_mean(df, "recon_WER"),
            "eval/avg_CER": _metric_mean(df, "recon_CER"),
            "eval/avg_dWER": avg_dwer,
            "eval/avg_dCER": _metric_mean(df, "recon_dCER"),
        }

        summary_line = (
            f"Evaluation step {step}: "
            f"UTMOS={summary_metrics['eval/avg_UTMOS']:.4f}, "
            f"dWER={summary_metrics['eval/avg_dWER']:.4f}, "
            f"WER={summary_metrics['eval/avg_WER']:.4f}"
        )
        logger.info(summary_line)
        print(summary_line, flush=True)

        if wandb.run is not None:
            table = wandb.Table(dataframe=df)
            wandb.log({"eval/metrics_table": table, **summary_metrics})

        return summary_metrics

    finally:
        logger.info("Evaluation loop completed.")
