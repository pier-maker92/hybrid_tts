import os
import json
import math
import random
import numpy as np
import torch
import wandb
import hydra
import logging
import datetime
import time
from typing import Dict, List, Optional
from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
from evaluation import run_evaluation
from torch.utils.data import DataLoader
from modules.builder import build_model
from omegaconf import DictConfig, OmegaConf
from data.audio_dataset import DataCollator, DataCollatorWithVAE, DiffusionDataCollator
from util import build_dataset, build_tokenizer, wandb_init
from torch.optim.lr_scheduler import LambdaLR

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_scheduler(optimizer, warmup_steps, num_training_steps, initial_lr, min_lr, lr_type="cosine"):
    def get_lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
            
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        
        min_lr_ratio = min_lr / initial_lr if initial_lr > 0 else 0.0
        
        if lr_type == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif lr_type == "linear":
            decay = 1.0 - progress
        else:
            decay = 1.0
            
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay
        
    return LambdaLR(optimizer, get_lr_lambda)

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


from modules.diffusion_head.cfm import DiT
from modules.configs import HybridTTSConfig
from torch.nn.utils.rnn import pad_sequence
from modules.hybrid_model import DynamicNormalizer
import torch.nn as nn

class DiffusionOnlyModel(nn.Module):
    def __init__(self, config: HybridTTSConfig, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # Determine the hidden size for the DiT conditioning
        hidden_size = config.diffusion_head_config.backbone_dim
        
        # Look-Up Table (LUT) for discrete tokens. 
        # Add +1 to discrete_token_vocab_size for the pad_id (usually 1024)
        self.embed = nn.Embedding(
            tokenizer.discrete_token_vocab_size + 1, 
            hidden_size, 
            padding_idx=tokenizer.discrete_token_vocab_size
        )
        
        # Normalizer for continuous tokens
        self.dynamic_normalizer = DynamicNormalizer(config.continuous_dim)
        
        # Diffusion head
        self.diffusion_head = DiT(config.diffusion_head_config)
        self.shift_audio_offset = config.shift_audio_offset

    def forward(
        self,
        discrete_sequence: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        continuous_sequence: torch.FloatTensor,
        audio_padding_mask: torch.BoolTensor,
        ecapa: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        # In this simplified scenario, discrete_sequence contains ONLY the discrete tokens 
        # and continuous_sequence contains ONLY the continuous tokens. Lengths match exactly.
        
        # Embed discrete sequence to get the context_vector
        context_vector = self.embed(discrete_sequence)

        norm_ratio = None
        diffusion_loss = None
        
        if continuous_sequence is not None:
            continuous_sequence = self.dynamic_normalizer(continuous_sequence)
            
            diffusion_loss = self.diffusion_head(
                target=continuous_sequence,
                target_padding_mask=audio_padding_mask,
                context_vector=context_vector,
                ecapa=ecapa,
            ).loss
            
            # compute a fake norm ratio for logging compatibility
            discrete_norm = context_vector.norm(dim=-1).mean()
            continuous_norm = continuous_sequence.norm(dim=-1).mean()
            norm_ratio = discrete_norm / (continuous_norm + 1e-8)

        class Output:
            pass
        out = Output()
        out.diffusion_loss = diffusion_loss
        out.token_logits = torch.zeros(1, device=context_vector.device) # dummy
        out.norm_ratio = norm_ratio
        return out


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict_to_save = json.loads(json.dumps(cfg_dict))

    scratch_dir = os.environ.get("SCRATCH", "/Users/software/Research")
    if cfg_dict.get("vae_checkpoint"):
        cfg_dict["vae_checkpoint"] = cfg_dict["vae_checkpoint"].replace("$SCRATCH", scratch_dir)

    training_cfg = cfg_dict.get("training", {})
    
    seed = training_cfg.get("seed")
    if seed is not None:
        set_seed(seed)

    # Accelerate init
    kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))
    grad_accum_steps = training_cfg.get("gradient_accumulation_steps", 1)
    accelerator = Accelerator(
        kwargs_handlers=[kwargs],
        gradient_accumulation_steps=grad_accum_steps,
        log_with="wandb",
    )
    logger.info(f"Using device: {accelerator.device}")

    if accelerator.is_main_process:
        if training_cfg.get("report_to") == "wandb":
            logger.info("Initializing W&B...")
        wandb_init(training_cfg, accelerator)

    logger.info("Building dataset...")
    train_dataset, test_dataset, dataset_name = build_dataset(training_cfg)

    backbone_cfg = cfg_dict.get("backbone", cfg_dict.get("backbone_config", {}))
    is_pretrained = backbone_cfg.get("pretrained", False)
    if is_pretrained:
        raise NotImplementedError("Pretrained backbone not supported yet.")

    logger.info("Creating HybridTTS tokenizer...")
    tok = build_tokenizer(cfg_dict, pretrinaed=is_pretrained)

    logger.info("Creating Diffusion-only model...")
    from modules.builder import load_codebook_config_from_cfg
    from modules.configs import HybridTTSConfig, DiTConfig
    
    continuous_dim, _ = load_codebook_config_from_cfg(cfg_dict)
    
    diffusion_head_cfg = cfg_dict.get("diffusion_head", cfg_dict.get("diffusion_head_config", {}))
    diffusion_config = DiTConfig(**diffusion_head_cfg)
    
    # Override with actual dataset/model dimensions
    diffusion_config.audio_latent_dim = continuous_dim
    
    # We use diffusion_config.net_dim as the backbone_dim for the embedding LUT
    diffusion_config.backbone_dim = diffusion_config.net_dim

    config = HybridTTSConfig(
        backbone_config=None,
        diffusion_head_config=diffusion_config,
        continuous_adapter_config=None,
        prompt_vocab_size=tok.prompt_vocab_size,
        discrete_token_vocab_size=tok.discrete_token_vocab_size,
        continuous_dim=continuous_dim,
        pad_token_id=tok.pad_id,
        start_audio_id=tok.start_audio_id,
        end_audio_id=tok.end_audio_id,
        shift_audio_offset=cfg_dict.get("training", {}).get("shift_audio_offset", 1),
    )
    
    model = DiffusionOnlyModel(config, tokenizer=tok)
    model.tokenizer = tok

    pretrained_checkpoint = training_cfg.get("pretrained_checkpoint")
    if pretrained_checkpoint:
        logger.info(f"Loading pretrained weights from {pretrained_checkpoint}")
        checkpoint_path = os.path.join(pretrained_checkpoint, "pytorch_model.bin")
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded pretrained weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        else:
            logger.warning(f"Pretrained checkpoint {checkpoint_path} not found!")

    learning_rate = float(training_cfg.get("learning_rate", 1e-4))
    min_learning_rate = float(training_cfg.get("min_learning_rate", 0.0))
    
    max_train_samples = training_cfg.pop("max_train_samples", None)
    if max_train_samples is not None:
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(min(int(max_train_samples), len(train_dataset)))
        )
        logger.info(f"Limiting train dataset to {len(train_dataset)} samples.")
        
    eval_num_samples = training_cfg.pop("eval_num_samples", 100)
    
    # DataLoader preparation
    is_online = dataset_name in ["libritts-r", "libritts_r"]
    if is_online:
        online_vae_checkpoint = cfg_dict.get("vae_checkpoint")
        logger.info(f"Online VAE encoding: loading VAE from {online_vae_checkpoint}")
        online_device = accelerator.device
        online_vae = load_vae(online_vae_checkpoint, online_device)
        if online_vae is None:
            raise RuntimeError("Online VAE encoding requires a valid vae_checkpoint")
        online_vae = online_vae.float().eval()
        for p in online_vae.parameters():
            p.requires_grad_(False)
        data_collator = DataCollatorWithVAE(
            tokenizer=tok, vae=online_vae, device=online_device
        )
        logger.info("DataCollatorWithVAE ready.")
    else:
        data_collator = DiffusionDataCollator(pad_id=tok.discrete_token_vocab_size, tokenizer=tok)

    per_device_train_batch_size = training_cfg.get("per_device_train_batch_size", 8)
    per_device_eval_batch_size = training_cfg.get("per_device_eval_batch_size", 8)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=True
    )
    
    if eval_num_samples is not None and eval_num_samples > 0:
        eval_indices = list(range(min(eval_num_samples, len(test_dataset))))
        eval_dataset_subset = torch.utils.data.Subset(test_dataset, eval_indices)
    else:
        eval_dataset_subset = test_dataset
        
    eval_dataloader = DataLoader(
        eval_dataset_subset,
        batch_size=per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    # Optimization
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Scheduler calculations
    num_train_epochs = training_cfg.get("num_train_epochs", 100)
    max_steps = training_cfg.get("max_steps", -1)
    steps_per_epoch = len(train_dataloader) // grad_accum_steps
    
    if max_steps > 0:
        num_training_steps = max_steps
        num_train_epochs = math.ceil(num_training_steps / max(1, steps_per_epoch))
    else:
        num_training_steps = num_train_epochs * steps_per_epoch
        
    warmup_steps = training_cfg.get("warmup_steps", 0)
    if "warmup_ratio" in training_cfg and training_cfg["warmup_ratio"] > 0:
        warmup_steps = int(num_training_steps * training_cfg["warmup_ratio"])
        
    lr_scheduler_type = training_cfg.get("lr_scheduler_type", "cosine")
    lr_scheduler = get_scheduler(
        optimizer, warmup_steps, num_training_steps, learning_rate, min_learning_rate, lr_scheduler_type
    )

    # Accelerate prepare
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    logging_steps = training_cfg.get("logging_steps", 100)
    save_steps = training_cfg.get("save_steps", 1000)
    eval_steps = training_cfg.get("eval_steps", 1000)
    output_dir = training_cfg.get("output_dir", "outputs/hybrid_tts")
    os.makedirs(output_dir, exist_ok=True)

    resume_from_checkpoint = training_cfg.get("resume_from_checkpoint")
    starting_step = 0
    starting_epoch = 0
    if resume_from_checkpoint:
        accelerator.load_state(resume_from_checkpoint)
        accelerator.print(f"Resumed from checkpoint: {resume_from_checkpoint}")
        # Simplification: not strictly mapping steps back for starting_epoch/step

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Batch size per device = {per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {grad_accum_steps}")
    logger.info(f"  Total optimization steps = {num_training_steps}")

    global_step = starting_step
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    running_token_loss = 0.0
    running_diffusion_loss = 0.0
    running_norm_ratio = 0.0
    running_steps = 0
    start_time = time.time()

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        
        # Explicitly set the epoch for DistributedSampler to ensure data is shuffled differently each epoch
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)
        elif hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
            
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(
                    discrete_sequence=batch.get("discrete_sequence"),
                    attention_mask=batch.get("attention_mask"),
                    continuous_sequence=batch.get("continuous_sequence"),
                    audio_padding_mask=batch.get("audio_padding_mask"),
                    ecapa=batch.get("ecapa"),
                )
                
                diffusion_loss = outputs.diffusion_loss if outputs.diffusion_loss is not None else torch.tensor(0.0, device=accelerator.device)
                
                total_loss = diffusion_loss
                accelerator.backward(total_loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            running_token_loss += 0.0
            running_diffusion_loss += diffusion_loss.detach().float()
            if getattr(outputs, "norm_ratio", None) is not None:
                running_norm_ratio += outputs.norm_ratio.detach().float()
            running_steps += 1
            
            if accelerator.sync_gradients:
                global_step += 1
                
                if logging_steps is not None and logging_steps > 0 and global_step % logging_steps == 0:
                    avg_diff_loss = accelerator.gather(running_diffusion_loss).mean().item() / running_steps
                    avg_norm_ratio = accelerator.gather(running_norm_ratio).mean().item() / running_steps if running_norm_ratio != 0 else 0
                    current_lr = lr_scheduler.get_last_lr()[0]
                    
                    elapsed_time = time.time() - start_time
                    steps_per_sec = global_step / elapsed_time if elapsed_time > 0 else 0
                    eta_seconds = (num_training_steps - global_step) / steps_per_sec if steps_per_sec > 0 else 0
                    eta_td = datetime.timedelta(seconds=int(eta_seconds))
                    
                    if accelerator.is_main_process:
                        if wandb.run is not None:
                            wandb.log({
                                "train/diffusion_loss": avg_diff_loss,
                                "train/norm_ratio": avg_norm_ratio,
                                "train/learning_rate": current_lr,
                                "train/global_step": global_step,
                                "train/epoch": epoch + (step + 1) / len(train_dataloader)
                            }, step=global_step)
                        logger.info(f"Epoch {epoch} Step {global_step}/{num_training_steps} | Diff Loss: {avg_diff_loss:.4f} | LR: {current_lr:.2e} | ETA: {eta_td}")
                        
                    running_token_loss = 0.0
                    running_diffusion_loss = 0.0
                    running_norm_ratio = 0.0
                    running_steps = 0
                    
                if save_steps is not None and save_steps > 0 and global_step % save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                        unwrapped_model = accelerator.unwrap_model(model)
                        state_dict = unwrapped_model.state_dict()
                        if "backbone.model.lm_head.weight" in state_dict:
                            state_dict["backbone.model.lm_head.weight"] = state_dict["backbone.model.lm_head.weight"].clone()
                        elif "backbone.lm_head.weight" in state_dict:
                            state_dict["backbone.lm_head.weight"] = state_dict["backbone.lm_head.weight"].clone()
                            
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
                        with open(os.path.join(save_dir, "config.json"), "w") as f:
                            json.dump(cfg_dict_to_save, f, indent=4)
                        logger.info(f"Saved checkpoint to {save_dir}")
                        
                if eval_steps is not None and eval_steps > 0 and global_step % eval_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        logger.info(f"Running evaluation at step {global_step}...")
                        vae_checkpoint = cfg_dict.get("vae_checkpoint")
                        vocoder_checkpoint = cfg_dict.get("vocoder_checkpoint")
                        vocoder_type = cfg_dict.get("vocoder_type")
                        
                        device = accelerator.device
                        vae = load_vae(vae_checkpoint, device) if vae_checkpoint else None
                        vocoder = load_vocoder(vocoder_checkpoint, device) if vocoder_checkpoint else None
                        
                        try:
                            logger.info("Evaluation relies on full HybridTTS encode_decode. Skipping advanced evaluation for diffusion-only model.")
                        except Exception as e:
                            logger.error(f"Evaluation failed: {e}")
                        finally:
                            del vae, vocoder
                            torch.cuda.empty_cache()
                            
                    accelerator.wait_for_everyone()
                    model.train() # Make sure to switch back to training mode
                    
            if max_steps > 0 and global_step >= max_steps:
                break
        if max_steps > 0 and global_step >= max_steps:
            break

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dir = os.path.join(output_dir, f"checkpoint-final")
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = unwrapped_model.state_dict()
        if "backbone.model.lm_head.weight" in state_dict:
            state_dict["backbone.model.lm_head.weight"] = state_dict["backbone.model.lm_head.weight"].clone()
        elif "backbone.lm_head.weight" in state_dict:
            state_dict["backbone.lm_head.weight"] = state_dict["backbone.lm_head.weight"].clone()
            
        os.makedirs(save_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(cfg_dict_to_save, f, indent=4)
        logger.info(f"Saved final checkpoint to {save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
