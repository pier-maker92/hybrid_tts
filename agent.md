# HybridTTS — Agent Reference

## Overview

End-to-end TTS system combining a **discrete autoregressive stream** (phoneme → VQ audio tokens via causal LLM) and a **continuous diffusion stream** (VAE latents via Conditional Flow Matching). Both streams are jointly trained; at inference they run in lockstep per autoregressive step.

External dependencies: **MelCausalVAE** (audio compression), **Vocos** (mel→waveform vocoder), **Qwen2-0.5B or Llama-3.2-1B** (LLM backbone).

---

## Repository Layout

```
hybrid_tts/
├── modules/
│   ├── hybrid_model.py           # HybridTTS main model + HybridTokenizer + CausalLMWrapper
│   ├── builder.py                # build_model(cfg_dict, tokenizer) factory
│   ├── configs.py                # HybridTTSConfig, BackboneConfig, DiTConfig, AdapterConfig
│   ├── output_dataclasses.py     # HybridTTSOutput
│   ├── diffusion_head/
│   │   ├── cfm.py                # DiT: CFM head (forward + generate)
│   │   ├── dit.py                # Transformer backbone for CFM head
│   │   ├── mlp.py                # MLP-based alternative diffusion head
│   │   └── utils.py              # Rotary embeddings, etc.
│   └── submodules/
│       └── MelCausalVAE/         # Copy of MelCausalVAE (used at train/inference time)
├── data/
│   ├── audio_dataset.py          # DataCollator, TrainDatasetWrapper, OnlineTrainDatasetWrapper, DataCollatorWithVAE
│   ├── librispeech_align.py      # LibriSpeech-aligned dataset
│   ├── libri_tts.py              # LibriTTS-R dataset
│   └── lj_speech_*.py            # LJSpeech variants
├── configs/
│   ├── main.yaml                 # Root config (imports defaults + settings)
│   ├── defaults/
│   │   ├── train.yaml            # Training hyperparams
│   │   ├── backbone.yaml         # Backbone model config
│   │   └── diffusion_head.yaml   # CFM head config
│   └── settings/                 # Experiment overrides
├── train.py                      # HF Trainer-based training script
├── inference.py                  # CLI inference: text/phonemes → WAV
├── evaluation.py                 # UTMOS + WER/CER evaluation
└── util.py                       # build_dataset(), build_tokenizer(), wandb_init()
```

---

## Architecture

### Token space

```
HybridTokenizer:
  prompt_vocab_size          # phoneme vocab (e.g., 256)
  discrete_token_vocab_size  # VQ codebook size (e.g., 1024)
  start_audio_id             # special token
  end_audio_id               # special token
  pad_id
  unified_vocab_size = prompt_vocab_size + discrete_token_vocab_size + 3
```

The backbone sees a unified vocabulary. Phoneme tokens occupy the lower IDs; audio VQ tokens occupy the upper range.

### Model components

```
HybridTTS
  ├── backbone: CausalLMWrapper (Qwen/Llama) OR native Transformer
  │     Input: unified token embeddings + projected continuous latents
  │     Output: hidden_states [B, T, hidden_dim]
  │
  ├── continuous_adapter: MLP or Linear
  │     Maps VAE latents [B, T_audio, continuous_dim] → [B, T_audio, hidden_dim]
  │     Inserted between audio tokens in the sequence
  │
  ├── token_head: 2-layer MLP
  │     hidden_dim → unified_vocab_size + 1 (EOS)
  │     Cross-entropy loss on predicted discrete audio tokens
  │
  └── diffusion_head: DiT (CFM)
        Conditioned on backbone hidden states
        Predicts continuous VAE latents via flow matching
        FM loss on velocity field
```

### Data flow — Training

```
Raw audio
  → MelCausalVAE Encoder
      ├─ discrete_tokens: VQ indices [B, T_audio]    (codebook size 1024)
      └─ continuous_tokens: latents [B, T_audio, 64]

Phoneme prompt + [<start_audio>] + interleaved discrete/continuous + [<end_audio>]
  → HybridTTS.forward()
      ├─ Embed tokens
      ├─ Insert continuous latents via continuous_adapter
      ├─ Backbone (LLM/Transformer)
      ├─ Token head: predict discrete tokens → CrossEntropyLoss
      └─ Diffusion head: reconstruct continuous latents → FM Loss

total_loss = token_loss + diffusion_loss
```

### Data flow — Inference

```
Phoneme prompt + [<start_audio>]
  → HybridTTS.sample() — autoregressive loop (KV-cache enabled):
      per step:
        backbone.inference_forward() → hidden_state[-1]
        token_head → sample discrete token (temperature)
        diffusion_head.generate() → 1 continuous latent frame (ODE, num_steps)
        until <end_audio> token

  → [VQ embeddings | continuous latents]
  → MelCausalVAE.sample() (ODE, ~16 steps) → mel [T, 100]
  → Vocos.decode(mel) → waveform (24kHz)
```

### Continuous dim

`continuous_dim = latent_dim - dim_to_quantize` (when `add_vq_residual_to_stoch=False`). For default config: `64 - 32 = 32`.

---

## Key Classes

### `HybridTTS` (`modules/hybrid_model.py:270-759`)

| Method | Purpose |
|--------|---------|
| `forward(discrete_sequence, attention_mask, continuous_sequence, audio_padding_mask)` | Training; returns `HybridTTSOutput` with `token_logits`, `diffusion_loss` |
| `sample(batch, max_steps, temperature, num_steps, diffusion_temperature, guidance_scale)` | Autoregressive inference with KV-cache |

### `CausalLMWrapper` (`hybrid_model.py:128-200`)

Wraps a HuggingFace CausalLM:
- `forward(inputs_embeds, attention_mask)` → `last_hidden_state`
- `inference_forward(inputs_embeds, attention_mask, past_key_values)` → `(hidden[:, -1:, :], pkv)` — returns only the last token for autoregressive decoding

Vocabulary is resized to `unified_vocab_size`; `lm_head.weight` is **tied** to the token embedding table (DDP serialization workaround: clone before safetensors save).

### `DiT` / CFM head (`modules/diffusion_head/cfm.py`)

- `forward(target, target_padding_mask, context_vector)` → FM loss
- `generate(num_steps, context_vector, temperature, guidance_scale)` → sampled latents via ODE

Context vector: backbone hidden states projected via `nn.Linear(backbone_dim → net_dim)`.

### `DataCollatorWithVAE` (`data/audio_dataset.py`)

Used for **online encoding** (no precomputed tokens). Calls the VAE inside the collator to produce discrete + continuous tokens on the fly. Required when training on raw audio without a precomputed dataset. Set `dataloader_num_workers=0` when using this to avoid CUDA fork issues.

---

## Config System (Hydra)

Entrypoint: `train.py` with `@hydra.main(config_path="configs", config_name="main")`.

Key keys in `defaults/train.yaml`:
- `dataset_name`, `vae_checkpoint`, `vocoder_checkpoint`
- `learning_rate`, `min_learning_rate` (floor for cosine decay)
- `uncond_prob`: CFG dropout rate on context (diffusion head)
- `no_augment_ratio`: fraction of batches without noise augmentation
- `shift_audio_offset`: int, shifts discrete target by N (causal prediction offset)
- `online_vae`: bool, if true use `DataCollatorWithVAE` for on-the-fly encoding

Key keys in `defaults/backbone.yaml`:
- `model_type`: `"qwen-0.5b"` | `"llama-1b"` | `"native"`
- `hidden_dim`, `num_layers`, `num_heads`: auto-inferred from HF config if null
- `force_weight_tying`: bool

Key keys in `defaults/diffusion_head.yaml`:
- `audio_latent_dim`: must match VAE `continuous_dim`
- `net_dim`, `net_heads`, `net_depth`: DiT dimensions
- `is_causal`, `use_window_attention`, `window_attention_seconds`
- `use_mlp_sampler`: use MLP instead of Transformer

---

## Training (`train.py`)

Trainer: `HybridTTSTrainer(Trainer)`.

- Custom cosine LR schedule with `min_learning_rate` floor.
- `compute_loss()`: sums `token_loss` + `diffusion_loss`; both logged separately.
- `EvaluationCallback`: lazily loads VAE + Vocos at first eval step; calls `run_evaluation()`.
- `_save()`: clones `lm_head.weight` before saving (DDP weight-tying workaround).
- `AddGranularLossesToTrainerState`: accumulates per-batch token/diffusion losses.

Launch: `sh /scratch/piermel/scripts/run_job.sh agent/<config_name>`

---

## Inference (`inference.py`)

```bash
python inference.py \
  --hybrid_checkpoint <dir>   \  # contains config.json + model.safetensors
  --text "hello world"        \
  --output out.wav            \
  --vocoder vocos             \
  --num_steps 4               \  # diffusion ODE steps per frame
  --temperature 1.0           \
  --guidance_scale 1.3
```

Pipeline: G2P (g2p_en) → phoneme IDs → `model.sample()` → VQ embed + continuous → `vae.sample()` → mel → `vocos.decode()` → WAV.

---

## Evaluation (`evaluation.py`)

`run_evaluation(model, vae, vocoder, eval_dataloader, device, step)`:
- Generates audio for each eval sample via `model.sample()`
- Metrics: UTMOS (UTMOSv2), WER/CER (Whisper large-v3 via `jiwer`)
- Outputs: CSV per step (`evaluation/validation_training/<run_id>/val_step_<N>.csv`) + W&B table

---

## Dataset Notes

| Dataset | Class | Notes |
|---------|-------|-------|
| LibriSpeech-aligned | `LibriSpeechAlignDataset` | Has phoneme alignments; parquet format |
| LibriTTS-R | `LibriTTSDataset` | 24kHz; train/test splits only (no named subsets); parquet |
| LJSpeech 128/512 | `LJSpeech128Dataset` etc. | Pre-tokenized at different codebook sizes |

For online encoding on LibriTTS-R: set `dataset_filter: ""` (no subset filtering), `online_vae: true`.

---

## External Dependencies

| Dependency | Role | Notes |
|-----------|------|-------|
| `MelCausalVAE` | Audio encoder/decoder | Submodule at `modules/submodules/MelCausalVAE/`; loaded via `build_model()` from its own `config.json` |
| Vocos (`charactr/vocos-mel-24khz`) | Mel → waveform | Loaded at eval/inference; 24kHz, 80-bin mel |
| Qwen2-0.5B / Llama-3.2-1B | LLM backbone | Via HuggingFace; vocab resized; weights cached in `$HF_HOME` |
| UTMOSv2 | MOS evaluation | `git+https://github.com/sarulab-speech/UTMOSv2.git` |
| Whisper large-v3 | WER/CER | Via `transformers`; cached in `$HF_HOME` |

All weights must be pre-cached before submitting to compute nodes (no internet access on cluster).

---

## Known Issues & Invariants

| Issue | Fix/Location |
|-------|-------------|
| CUDA fork error with `dataloader_num_workers > 0` when using `DataCollatorWithVAE` | Set `dataloader_num_workers: 0` in config |
| DDP weight tying: safetensors refuses to save tied weights | `_save()` in `HybridTTSTrainer` clones `lm_head.weight` before save |
| LibriTTS-R at 24kHz, WavLM expects 16kHz | `WavLMFeatureExtractor` resamples internally; no config change needed |
| `continuous_dim` must match VAE `latent_dim - dim_to_quantize` | Check `config.json` of `vae_checkpoint` when building a new experiment |
| Compute nodes have no internet | Cache all HF models before job launch; set `HF_HUB_OFFLINE=1` in job |

---

## Checkpoints

`model.safetensors` + `config.json` (serialized `HybridTTSConfig`) in the checkpoint directory.

VAE checkpoint is a **separate** directory specified in the training config (`vae_checkpoint`). The HybridTTS checkpoint does NOT include VAE weights.
