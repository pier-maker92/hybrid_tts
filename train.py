import os
import json
import math
import torch
import wandb
import hydra
import logging
import datetime
from typing import Dict, List, Optional
from accelerate import Accelerator
from evaluation import run_evaluation
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from modules.builder import build_model
from omegaconf import DictConfig, OmegaConf
from data.audio_dataset import DataCollator
from accelerate import InitProcessGroupKwargs
from modules.hybrid_model import HybridTokenizer
from util import build_dataset, build_tokenizer, wandb_init
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    set_seed,
)


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




def load_vae(checkpoint_dir: str, device: torch.device):
    try:
        from modules.submodules.MelCausalVAE.modules.builder import (
            build_model as build_vae,
        )

        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
        vae = build_vae(cfg_dict)

        checkpoint_path = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.exists(checkpoint_path):
            vae.from_pretrained(checkpoint_path)
        else:
            checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
            if os.path.exists(checkpoint_path):
                vae.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        vae.eval()
        vae.to(device)
        return vae
    except Exception as e:
        logger.error(f"Failed to load VAE from {checkpoint_dir}: {e}")
        return None


def load_vocoder(vocoder_name_or_path: str, device: torch.device):
    try:
        from vocos import Vocos

        if vocoder_name_or_path == "bigvgan" or vocoder_name_or_path == "vocos":
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        else:
            vocoder = Vocos.from_pretrained(vocoder_name_or_path).to(device)
        return vocoder
    except Exception as e:
        logger.error(f"Failed to load Vocoder {vocoder_name_or_path}: {e}")
        return None


class EvaluationCallback(TrainerCallback):
    def __init__(
        self,
        vae_checkpoint,
        vocoder_checkpoint,
        vocoder_type,
        dataset_name,
        eval_dataset,
        eval_device,
        num_samples=100,
        batch_size=1,
        **kwargs,
    ):
        self.vae_checkpoint = vae_checkpoint
        self.vocoder_checkpoint = vocoder_checkpoint
        self.vocoder_type = vocoder_type
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.eval_device = eval_device
        # Build a dedicated DataLoader limited to num_samples
        if num_samples and num_samples > 0:
            indices = list(range(min(num_samples, len(eval_dataset))))
            subset = torch.utils.data.Subset(eval_dataset, indices)
        else:
            subset = eval_dataset
        self.eval_dataloader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollator(tokenizer=kwargs.get("tokenizer")),
        )
        logger.info(
            f"EvaluationCallback: will evaluate on {len(self.eval_dataloader.dataset)} samples (device={self.eval_device})."
        )

    def on_evaluate(self, args, state, control, model, **kwargs):
        if not state.is_world_process_zero:
            return

        device = self.eval_device
        logger.info(
            f"Running custom evaluation at step {state.global_step} on {device}"
        )

        # Load VAE and Vocoder lazily — only during eval, then release GPU memory.
        vae = load_vae(self.vae_checkpoint, device) if self.vae_checkpoint else None
        vocoder = (
            load_vocoder(self.vocoder_checkpoint, device)
            if self.vocoder_checkpoint
            else None
        )
        try:
            # Unwrap model if it's DDP
            unwrapped_model = getattr(model, "module", model)
            run_evaluation(
                model=unwrapped_model,
                vae=vae,
                vocoder=vocoder,
                vocoder_type=self.vocoder_type,
                eval_dataloader=self.eval_dataloader,
                device=device,
                step=state.global_step,
                dataset_name=self.dataset_name,
                num_samples=self.num_samples,
                run_id=wandb.run.id if wandb.run else "eval_run",
            )
        finally:
            # Free GPU memory immediately after eval.
            del vae, vocoder
            torch.cuda.empty_cache()


class HybridTTSTrainer(Trainer):
    def __init__(self, dataset_name: str = "dataset", **kwargs):
        eval_num_samples = kwargs.pop("eval_num_samples", 100)
        self.eval_num_samples = (
            eval_num_samples if eval_num_samples is not None else float("inf")
        )
        self.min_learning_rate = kwargs.pop("min_learning_rate", 0.0)
        self.cfg_dict = kwargs.pop("cfg_dict", None)
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        granular_losses = [
            "token_loss",
            "diffusion_loss",
        ]
        self.add_callback(AddGranularLossesToTrainerState(granular_losses))

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
            
        logger.info("Custom get_train_dataloader called. Enforcing shuffle=True.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=True,
        )

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer

        lr_type = self.args.lr_scheduler_type
        if lr_type not in ["cosine", "linear"]:
            return super().create_scheduler(num_training_steps, optimizer)

        warmup_steps = self.args.get_warmup_steps(num_training_steps)
        initial_lr = self.args.learning_rate
        min_lr = self.min_learning_rate

        logger.info(
            f"Setting up custom LambdaLR scheduler: lr_scheduler_type={lr_type}, "
            f"warmup_steps={warmup_steps}, initial_lr={initial_lr}, min_learning_rate={min_lr}"
        )

        def get_lr_lambda(current_step):
            # 1. Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            # 2. Decay phase
            progress = float(current_step - warmup_steps) / float(
                max(1, num_training_steps - warmup_steps)
            )
            progress = min(1.0, max(0.0, progress))

            min_lr_ratio = min_lr / initial_lr if initial_lr > 0 else 0.0

            if lr_type == "cosine":
                decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            elif lr_type == "linear":
                decay = 1.0 - progress
            else:
                decay = 1.0

            return min_lr_ratio + (1.0 - min_lr_ratio) * decay

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)
        return self.lr_scheduler

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and limit number of samples.
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if eval_dataset is not None and self.eval_num_samples < float("inf"):
            num_samples = min(int(self.eval_num_samples), len(eval_dataset))
            indices = list(range(num_samples))
            eval_dataset = torch.utils.data.Subset(eval_dataset, indices)
            logger.info(f"Evaluating on a subset of {num_samples} samples.")

        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        outputs = model(
            discrete_sequence=inputs.get("discrete_sequence"),
            attention_mask=inputs.get("attention_mask"),
            continuous_sequence=inputs.get("continuous_sequence"),
            audio_padding_mask=inputs.get("audio_padding_mask"),
        )

        token_logits = outputs.token_logits
        diffusion_loss = (
            outputs.diffusion_loss
            if outputs.diffusion_loss is not None
            else torch.tensor(0.0, device=token_logits.device)
        )
        target_tokens = inputs.get("target_tokens")

        # Token Loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # Handle DDP wrapping
        unwrapped_model = getattr(model, "module", model)
        token_loss = loss_fct(
            token_logits.view(
                -1, unwrapped_model.discrete_token_vocab_size + 1
            ),  # discrete_token_vocab_size + audio_EOS
            target_tokens.view(-1),
        )

        total_loss = token_loss + diffusion_loss

        # Accumulate granular losses
        if hasattr(self.control, "granular_losses") and model.training:
            flat_metrics = {
                "token_loss": token_loss.detach(),
                "diffusion_loss": diffusion_loss.detach(),
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
        last_logged = getattr(self, "_globalstep_last_logged", 0)
        should_log = self.control.should_log and self.state.global_step > last_logged

        if should_log and hasattr(self.control, "granular_losses"):
            steps = self.state.global_step - last_logged
            logs_to_add = {}
            for key, tensor_val in self.control.granular_losses.items():
                logs_to_add[key] = round(tensor_val.item() / steps, 4)
                tensor_val.zero_()

            original_log = self.log

            def patched_log(logs, *log_args, **log_kwargs):
                logs.update(logs_to_add)
                original_log(logs, *log_args, **log_kwargs)

            self.log = patched_log
            try:
                super()._maybe_log_save_evaluate(*args, **kwargs)
            finally:
                del self.log
        else:
            super()._maybe_log_save_evaluate(*args, **kwargs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if state_dict is None:
            state_dict = self.model.state_dict()

        # Safetensors does not support saving shared tensors.
        # Since backbone.model.lm_head.weight is tied to backbone.model.model.embed_tokens.weight,
        # we clone one of them to prevent the RuntimeError during save_file.
        if "backbone.model.lm_head.weight" in state_dict:
            state_dict["backbone.model.lm_head.weight"] = state_dict[
                "backbone.model.lm_head.weight"
            ].clone()
        elif "backbone.lm_head.weight" in state_dict:
            state_dict["backbone.lm_head.weight"] = state_dict[
                "backbone.lm_head.weight"
            ].clone()

        super()._save(output_dir, state_dict)

        # Save config
        if getattr(self, "cfg_dict", None) is not None:
            save_dir = output_dir if output_dir is not None else self.args.output_dir
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                config_path = os.path.join(save_dir, "config.json")
                with open(config_path, "w") as f:
                    json.dump(self.cfg_dict, f, indent=4)


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict_to_save = json.loads(json.dumps(cfg_dict))

    # Resolve SCRATCH environment variable
    scratch_dir = os.environ.get("SCRATCH", "/Users/software/Research")
    if cfg_dict.get("vae_checkpoint"):
        cfg_dict["vae_checkpoint"] = cfg_dict["vae_checkpoint"].replace(
            "$SCRATCH", scratch_dir
        )

    training_cfg = cfg_dict.get("training")

    seed = training_cfg.get("seed") if training_cfg else None
    if seed is not None:
        set_seed(seed)

    # initialize accelerator
    kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    logger.info(f"Using device: {accelerator.device}")

    # initialize wandb
    logger.info("Initializing W&B...")
    wandb_init(training_cfg, accelerator)
    # build dataset
    logger.info("Building dataset...")
    train_dataset, test_dataset, dataset_name = build_dataset(training_cfg)

    backbone_cfg = cfg_dict.get("backbone_config")
    backbone_cfg = cfg_dict.get("backbone")
    is_pretrained = backbone_cfg.get("pretrained")
    if is_pretrained:
        raise NotImplementedError("Pretrained backbone not supported yet.")
    model_name_or_path = backbone_cfg.get("model_name_or_path")

    # build tokenizer
    logger.info("Creating HybridTTS tokenizer...")
    tok = build_tokenizer(cfg_dict, pretrinaed=is_pretrained)

    logger.info("Creating HybridTTS model...")
    model = build_model(cfg_dict, tokenizer=tok)

    # Attach HybridTokenizer so that debug mode can decode input sequences
    model.tokenizer = tok

    training_cfg["learning_rate"] = float(training_cfg.get("learning_rate"))
    min_learning_rate = float(training_cfg.pop("min_learning_rate", 0.0))
    # Pop custom config keys so they don't get passed to HuggingFace TrainingArguments
    _ = training_cfg.pop("uncond_prob", 0.0)
    _ = training_cfg.pop("no_augment_ratio", 0.0)
    _ = training_cfg.pop("discrete_only", False)
    eval_num_samples = training_cfg.pop("eval_num_samples", 100)
    run_id = training_cfg.pop("run_id", None)
    resume_from_checkpoint = training_cfg.pop("resume_from_checkpoint", None)
    training_cfg["group_by_length"] = (
        False  # training_cfg.get("group_by_length", False)
    )

    training_args = TrainingArguments(
        remove_unused_columns=False,
        ddp_timeout=7200,
        **training_cfg,
    )

    data_collator = DataCollator(tokenizer=tok)
    trainer = HybridTTSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        dataset_name=dataset_name,
        eval_num_samples=eval_num_samples,
        min_learning_rate=min_learning_rate,
        cfg_dict=cfg_dict_to_save,
    )

    # Optional: Register EvaluationCallback with checkpoint paths (VAE/Vocoder loaded lazily).
    vae_checkpoint = cfg_dict.get("vae_checkpoint", None)
    vocoder_checkpoint = cfg_dict.get("vocoder_checkpoint")

    if vae_checkpoint or vocoder_checkpoint:
        vocoder_type = cfg_dict.get("vocoder_type")
        trainer.add_callback(
            EvaluationCallback(
                vae_checkpoint=vae_checkpoint,
                vocoder_checkpoint=vocoder_checkpoint,
                vocoder_type=vocoder_type,
                dataset_name=dataset_name,
                eval_dataset=test_dataset,
                eval_device=accelerator.device,
                num_samples=eval_num_samples,
                batch_size=training_args.per_device_eval_batch_size,
                tokenizer=tok,
            )
        )
        logger.info(
            "EvaluationCallback registered (VAE/Vocoder will be loaded lazily at eval time)."
        )



    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    main()
