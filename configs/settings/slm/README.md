# SM tokens + full DiCodec latents

Five independent presets: `slm/asr`, `slm/lm`, `slm/asr_and_lm`, `slm/recon`,
`slm/continuous_only`. All train continuous targets on **full z**, never a residual.

Dataset: Parquet `librispeech-dicodec-v2-ls-25`, partitions `train_clean_100` and
`train_clean_360`, columns `z`, `attributes.z_sem`, `transcript`. Continuous-only
reads just `z` and `transcript`. Arrow cache lives on node-local disk. No waveform
encoding or DiCodec network is loaded; its config.json supplies the latent dimension.

Set `training.sm_quantizer_checkpoint` in each discrete preset to a `.pt` file or
a directory containing `last.pt`. The shipped paths select the four runs under
`$SCRATCH/MelCausalVAE/checkpoints/exps/10-September-2026/`. Backend, codebook size and
encoder architecture come from the checkpoint. Only the frozen encoder and quantizer
are instantiated and moved to the training device, once per process. Quantization
is batched, FP32, eval/no_grad; EMA statistics do not change. The task heads and
optimizer from SM are not instantiated. Text remains conditioning in all five modes.

The tokenizer vocabulary is inferred from the checkpoint. Discrete embeddings in
HybridTTS train normally from random initialization. The continuous-only baseline
loads no SM checkpoint and has no discrete audio prediction loss.

Voice conditioning and audio evaluation are disabled: this path consumes latent
exports without reference waveform inputs. There is no validation split selected;
training does not repurpose training examples as held-out evaluation. BF16 is
selected by the run_job/Accelerate launcher. Dataset utterances are not truncated.

Local submodule is pinned to v2 commit `65c7592f15e6f61e98219db1cf8d38c5a6bb8a89`.
Push the HybridTTS code/config changes and submodule pointer, then update the remote
checkout and initialize/update its submodule before launching. No repo changes
were made on Narval by this integration.

Narval experiments are root-level YAMLs (compatible with the current launcher):

```bash
sh /scratch/piermel/scripts/run_job.sh tts-sm-asr
sh /scratch/piermel/scripts/run_job.sh tts-sm-lm
sh /scratch/piermel/scripts/run_job.sh tts-sm-asr_and_lm
sh /scratch/piermel/scripts/run_job.sh tts-sm-recon
sh /scratch/piermel/scripts/run_job.sh tts-sm-continuous_only
```

Each requests one full A100, 8 CPU cores, 32G RAM, 24 hours. Copies of run_job YAMLs
are in `configs/experiments`. No jobs are launched automatically.

Tests: `python -m unittest data.test_sm_latents` checks all five forward/backward
paths, full-z targets, preserved text conditioning, frozen EMA/BSQ/FSQ, and Parquet
partition loading. Tiny local CLI training/save smoke checks were also run for
hybrid and continuous-only modes; this does not establish full-scale GPU convergence.

## Autoregressive sampling checks

All five presets use the independent per-frame MLP diffusion head. SM quantizer
collators run on the training device with `dataloader_num_workers: 0`.
The continuous-only baseline replaces quantizer tokens with the learned
`<audio_pad>` input embedding and trains its token head with targets
`<audio_pad>, ..., <audio_pad>, EOS` (classes 1 and 0). Batch padding is ignored
with target -100. Generation ends on learned EOS; `--max_len` is a safety cap.
All five variants train token cross-entropy alongside diffusion loss.
Before a long run, run `python -m unittest data.test_ar_regressions data.test_sm_latents data.test_sm_inference`.
These are structural checks; a short real-data overfit and listening test is still
needed to assess convergence and end-to-end dataset/codec compatibility.
