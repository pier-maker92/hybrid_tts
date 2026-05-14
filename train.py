import os
import json
import torch
import wandb
import logging
import datetime
import hydra
from typing import Dict, List
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    set_seed,
)

from accelerate import Accelerator
from transformers import AutoTokenizer
from modules.builder import build_model
from accelerate import InitProcessGroupKwargs
from data.audio_dataset import DataCollator, TrainDatasetWrapper, TestDatasetWrapper
from modules.evaluation import run_evaluation

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
    def __init__(self, vae, vocoder, vocoder_type, dataset_name, eval_dataset, num_samples=100, batch_size=1):
        self.vae = vae
        self.vocoder = vocoder
        self.vocoder_type = vocoder_type
        self.dataset_name = dataset_name
        self.num_samples = num_samples
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
            collate_fn=DataCollator(),
        )
        logger.info(f"EvaluationCallback: will evaluate on {len(self.eval_dataloader.dataset)} samples.")

    def on_evaluate(self, args, state, control, model, **kwargs):
        if state.is_world_process_zero:
            logger.info(f"Running custom evaluation at step {state.global_step}...")
            run_evaluation(
                model=model,
                vae=self.vae,
                vocoder=self.vocoder,
                vocoder_type=self.vocoder_type,
                eval_dataloader=self.eval_dataloader,
                device=args.device,
                step=state.global_step,
                dataset_name=self.dataset_name,
                num_samples=self.num_samples,
                run_id=wandb.run.id if wandb.run else "eval_run",
            )


class HybridTTSTrainer(Trainer):
    def __init__(self, dataset_name: str = "dataset", **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        granular_losses = ["token_loss", "diffusion_loss", "total_loss"]
        self.add_callback(AddGranularLossesToTrainerState(granular_losses))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # We need prompt_ids, discrete_tokens, continuous_tokens
        # Assuming DataCollator outputs: "ids", "prompt_ids", "discrete_tokens", "continuous_tokens", "padding_mask"

        prompt_ids = inputs.get("prompt_ids", None)
        discrete_tokens = inputs.get("discrete_tokens", None)
        continuous_tokens = inputs.get("continuous_tokens", None)
        padding_mask = inputs.get("padding_mask", None)
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

    kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    logger.info(f"Using device: {accelerator.device}")

    # Dataset loading logic
    dataset_name = training_cfg.pop("dataset_name")
    force_vocab_build = training_cfg.get("force_vocab_build", False)
    if dataset_name == "librispeech_aligned":
        from data.librispeech_align import LibriSpeechAlignDataset

        dataset = LibriSpeechAlignDataset(force_vocab_build=force_vocab_build)
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

    vocab_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "phoneme_vocab.json"
    )
    if os.path.exists(vocab_path):
        with open(vocab_path, "r") as f:
            phoneme_vocab = json.load(f)
        phoneme_list = list(phoneme_vocab.keys())
        vocab_size = len(phoneme_vocab)
    else:
        phoneme_list = []
        vocab_size = 256

    backbone_cfg = cfg_dict.get("backbone_config", cfg_dict.get("backbone", {}))
    is_pretrained = backbone_cfg.get("pretrained", False)
    model_name_or_path = backbone_cfg.get("model_name_or_path", "Qwen/Qwen2-0.5B")

    if not is_pretrained:
        cfg_dict["prompt_vocab_size"] = vocab_size + 2  # Phonemes + Special Tokens
        cfg_dict["prompt_offset"] = 0
        cfg_dict["start_audio_id"] = vocab_size
        cfg_dict["end_audio_id"] = vocab_size + 1
        logger.info(
            f"Training from scratch: set prompt_vocab_size to {cfg_dict['prompt_vocab_size']}"
        )
    else:
        logger.info(
            f"Using pretrained backbone, loading tokenizer from {model_name_or_path}"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Add phonemes
        if phoneme_list:
            num_added = tokenizer.add_tokens(phoneme_list)
            phoneme_ids = tokenizer.convert_tokens_to_ids(phoneme_list)
            cfg_dict["prompt_offset"] = min(phoneme_ids)
            logger.info(
                f"Added {num_added} phonemes to tokenizer starting at ID {cfg_dict['prompt_offset']}"
            )
        else:
            cfg_dict["prompt_offset"] = 0

        # Add special tokens
        special_tokens = ["<start_audio>", "<end_audio>"]
        tokenizer.add_tokens(special_tokens)
        cfg_dict["start_audio_id"] = tokenizer.convert_tokens_to_ids("<start_audio>")
        cfg_dict["end_audio_id"] = tokenizer.convert_tokens_to_ids("<end_audio>")
        logger.info(
            f"Special tokens IDs: <start_audio>={cfg_dict['start_audio_id']}, <end_audio>={cfg_dict['end_audio_id']}"
        )

        cfg_dict["prompt_vocab_size"] = len(tokenizer)

    logger.info("Creating HybridTTS model...")
    model = build_model(cfg_dict)

    if is_pretrained and "tokenizer" in locals():
        model.backbone.resize_token_embeddings(len(tokenizer))
        logger.info("Resized backbone token embeddings to match tokenizer.")

    training_cfg["learning_rate"] = float(training_cfg.get("learning_rate"))
    min_learning_rate = float(training_cfg.pop("min_learning_rate", 0.0))
    eval_num_samples = training_cfg.pop("eval_num_samples", 100)

    training_args = TrainingArguments(
        remove_unused_columns=False,
        ddp_timeout=7200,
        **training_cfg,
    )

    data_collator = DataCollator()
    # Limit eval_dataset for the Trainer's own eval loop too
    if test_dataset is not None and eval_num_samples and eval_num_samples > 0:
        indices = list(range(min(eval_num_samples, len(test_dataset))))
        eval_dataset = torch.utils.data.Subset(test_dataset, indices)
        logger.info(f"Trainer eval_dataset limited to {len(eval_dataset)} samples (eval_num_samples={eval_num_samples}).")
    else:
        eval_dataset = test_dataset

    trainer = HybridTTSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        dataset_name=dataset_name,
    )

    # Optional: Load VAE and Vocoder for evaluation
    vae_checkpoint = cfg_dict.get("vae_checkpoint")
    vocoder_checkpoint = cfg_dict.get("vocoder_checkpoint")

    if vae_checkpoint or vocoder_checkpoint:
        logger.info("Loading VAE/Vocoder for evaluation...")
        vae = load_vae(vae_checkpoint, accelerator.device) if vae_checkpoint else None
        vocoder = (
            load_vocoder(vocoder_checkpoint, accelerator.device)
            if vocoder_checkpoint
            else None
        )
        vocoder_type = cfg_dict.get("vocoder_type", "bigvgan")

        if vae is not None or vocoder is not None:
            trainer.add_callback(
                EvaluationCallback(
                    vae=vae,
                    vocoder=vocoder,
                    vocoder_type=vocoder_type,
                    dataset_name=dataset_name,
                    eval_dataset=test_dataset,
                    num_samples=eval_num_samples,
                    batch_size=training_args.per_device_eval_batch_size,
                )
            )
        else:
            logger.warning(
                "Could not load VAE or Vocoder. Evaluation callback disabled."
            )

    logger.info("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()
