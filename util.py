import os
import json
import wandb
from typing import Dict, Any
from modules.hybrid_model import HybridTokenizer
from data.audio_dataset import TrainDatasetWrapper, TestDatasetWrapper


def wandb_init(training_cfg: Dict[str, Any], accelerator):
    wandb_project = training_cfg.pop("wandb_project")
    wandb_run_name = training_cfg.pop("wandb_run_name")
    wandb_id = training_cfg.pop("wandb_id", None)
    if training_cfg.get("report_to") == "wandb" and accelerator.is_main_process:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume="allow" if wandb_id else None,
        )


def build_dataset(training_cfg: Dict[str, Any]):
    # Create AudioDataset
    dataset_name = training_cfg.pop("dataset_name")
    force_vocab_build = training_cfg.get("force_vocab_build")

    if dataset_name == "mls":
        from data.mls import MLSDataset

        dataset = MLSDataset()
    elif dataset_name == "libritts":
        from data.libri_tts import LibriTTS

        dataset = LibriTTS()
    elif dataset_name in [
        "librispeech_aligned",
        "librispeech-aligned",
        "librispeech-aligned_prepared",
    ]:
        from data.librispeech_align import LibriSpeechAlignDataset

        dataset = LibriSpeechAlignDataset(force_vocab_build=force_vocab_build)
    elif dataset_name == "lj_speech":
        from data.lj_speech import LJSpeechDataset
        dataset = LJSpeechDataset(force_vocab_build=force_vocab_build)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    discrete_only = training_cfg.get("discrete_only", False)
    train_dataset = TrainDatasetWrapper(dataset, "train", discrete_only=discrete_only)
    test_dataset = TestDatasetWrapper(dataset, "train", discrete_only=discrete_only) # FIXME workaround for LJSpeech

    return train_dataset, test_dataset, dataset_name


def build_tokenizer(cfg_dict: Dict[str, Any], pretrinaed: bool = False):
    if pretrinaed:
        raise NotImplementedError("Pretrained tokenizer not supported yet.")
    vocab_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "phoneme_vocab.json"
    )
    if os.path.exists(vocab_path):
        with open(vocab_path, "r") as f:
            phoneme_vocab = json.load(f)
        vocab_size = len(phoneme_vocab)
    else:
        raise ValueError("Phoneme vocabulary not found. Please build it first.")
    


    cfg_dict["prompt_vocab_size"] = vocab_size + 3
    cfg_dict["pad_token_id"] = vocab_size + 2
    cfg_dict["start_audio_id"] = vocab_size + 1
    cfg_dict["end_audio_id"] = vocab_size + 0
    cfg_dict["prompt_offset"] = 0

    return HybridTokenizer(
        phoneme_vocab=phoneme_vocab,
        start_audio_id=cfg_dict["start_audio_id"],
        end_audio_id=cfg_dict["end_audio_id"],
        pad_id=cfg_dict["pad_token_id"],
        prompt_offset=cfg_dict["prompt_offset"],
    )
