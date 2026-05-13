import os
import math
import torch
import wandb
import logging
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List

from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    set_seed,
)

import torch.distributed as dist
from accelerate import Accelerator
from data.audio_dataset import DataCollator, TrainDatasetWrapper, TestDatasetWrapper

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class AddGranularLossesToTrainerState(TrainerCallback):
    def __init__(self, granular_losses: List[str]):
        self.granular_losses = granular_losses

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        control.granular_losses = {
            k: torch.tensor(0.0).to(args.device) for k in self.granular_losses
        }
        return control


class HybridTTSTrainer(Trainer):
    def __init__(self, dataset_name: str = "dataset", **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        granular_losses = ["token_loss", "diffusion_loss", "total_loss"]
        self.add_callback(AddGranularLossesToTrainerState(granular_losses))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # We need prompt_ids, discrete_tokens, continuous_tokens
        # Here we mock extraction of these inputs from the dataset batch
        # You need to adapt this according to the actual keys in the DataCollator output
        # Assuming DataCollator outputs: "prompt_ids", "discrete_tokens", "continuous_tokens", "padding_mask"

        prompt_ids = inputs.get("prompt_ids", None)
        discrete_tokens = inputs.get("discrete_tokens", None)
        continuous_tokens = inputs.get("continuous_tokens", None)
        padding_mask = inputs.get("padding_mask", None)

        # If inputs are not provided directly by DataCollator (like in MelCausalVAE it's output_audios_srs),
        # You might need to use MelCausalVAE submodule to extract them first.
        # For simplicity, we assume they are provided in inputs.

        # We'll calculate a dummy loss if inputs are missing to keep the training loop running
        if prompt_ids is None or discrete_tokens is None or continuous_tokens is None:
            # Placeholder for actual feature extraction
            batch_size = (
                inputs["output_audios_srs"][0][0].shape[0]
                if "output_audios_srs" in inputs
                else 2
            )
            device = self.args.device
            prompt_ids = torch.randint(0, 256, (batch_size, 50)).to(device)
            discrete_tokens = torch.randint(0, 1024, (batch_size, 100)).to(device)
            continuous_tokens = torch.randn(
                batch_size, 100, model.config.continuous_dim
            ).to(device)
            padding_mask = torch.zeros(batch_size, 100, dtype=torch.bool).to(device)
            target_tokens = discrete_tokens.clone()
        else:
            target_tokens = inputs.get("target_tokens", discrete_tokens)

        outputs = model(
            prompt_ids=prompt_ids,
            discrete_tokens=discrete_tokens,
            continuous_tokens=continuous_tokens,
            padding_mask=padding_mask,
        )

        token_logits = outputs.token_logits
        diffusion_loss = outputs.diffusion_loss

        # Token Loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        token_loss = loss_fct(
            token_logits.view(-1, model.config.discrete_token_vocab_size),
            target_tokens.view(-1),
        )

        total_loss = token_loss + diffusion_loss

        # Accumulate granular losses
        if hasattr(self.control, "granular_losses") and model.training:
            flat_metrics = {
                "token_loss": token_loss.detach(),
                "diffusion_loss": diffusion_loss.detach(),
                "total_loss": total_loss.detach(),
            }
            for key in self.control.granular_losses:
                if flat_metrics.get(key) is not None:
                    val = flat_metrics[key].float()
                    if self.args.n_gpu > 1 and val.dim() > 0:
                        val = val.mean()
                    self.control.granular_losses[key] += (
                        val.to(self.control.granular_losses[key].dtype)
                        / self.args.gradient_accumulation_steps
                    )

        return (total_loss, outputs) if return_outputs else total_loss

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        tr_loss = args[0]
        grad_norm = args[1]
        model = args[2]
        trial = args[3]
        epoch = args[4]
        ignore_keys_for_eval = args[5]

        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            logs: Dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss
            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )

            if hasattr(self.control, "granular_losses"):
                for k, v in self.control.granular_losses.items():
                    logs[k] = self._nested_gather(v).mean().item()
                    self.control.granular_losses[k] -= self.control.granular_losses[k]
                    logs[k] = round(
                        logs[k]
                        / (self.state.global_step - self._globalstep_last_logged),
                        4,
                    )

            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm if isinstance(grad_norm, float) else grad_norm.item()
                )

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)

        super()._maybe_log_save_evaluate(*args, **kwargs)


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    training_cfg = cfg_dict.get("training", {})

    set_seed(training_cfg.get("seed", 42))

    from accelerate import InitProcessGroupKwargs

    kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    logger.info(f"Using device: {accelerator.device}")

    # Dataset loading logic
    dataset_name = training_cfg.pop("dataset_name")
    if dataset_name == "librispeech_aligned":
        from data.librispeech_align import LibriSpeechAlignDataset

        dataset = LibriSpeechAlignDataset()
    else:
        # Fallback to dummy
        dataset = None

    if dataset:
        train_dataset = TrainDatasetWrapper(dataset, "train")
        test_dataset = TestDatasetWrapper(dataset, "test")
    else:
        train_dataset, test_dataset = None, None

    wandb_project = training_cfg.pop("wandb_project", None)
    wandb_run_name = training_cfg.pop("wandb_run_name", None)
    wandb_id = training_cfg.pop("wandb_id", None)
    if training_cfg.get("report_to", "none") == "wandb" and accelerator.is_main_process:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume="allow" if wandb_id else None,
        )

    from modules.builder import build_model

    logger.info("Creating HybridTTS model...")
    model = build_model(cfg_dict)

    training_cfg["learning_rate"] = float(training_cfg.get("learning_rate"))
    min_learning_rate = float(training_cfg.pop("min_learning_rate", 0.0))

    training_args = TrainingArguments(
        remove_unused_columns=False,
        ddp_timeout=7200,
        **training_cfg,
    )

    data_collator = DataCollator()
    trainer = HybridTTSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        dataset_name=dataset_name,
    )

    logger.info("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()
