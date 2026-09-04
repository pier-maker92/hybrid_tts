import os
import json
import wandb
from typing import Dict, Any, Tuple, Optional
from modules.hybrid_model import HybridTokenizer
from data.audio_dataset import (
    TrainDatasetWrapper,
    TestDatasetWrapper,
    OnlineTrainDatasetWrapper,
    OnlineTestDatasetWrapper,
)


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
    train_split = training_cfg.pop("train_split", "train")
    online_encode = training_cfg.get("online_encode", False)

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

        dataset = LibriSpeechAlignDataset(
            force_vocab_build=force_vocab_build,
            keep_audio=online_encode,
            dataset_dir_name=training_cfg.get(
                "dataset_dir_name",
                "librispeech-aligned_prepared",
            ),
        )
        if online_encode:
            train_dataset = OnlineTrainDatasetWrapper(dataset, train_split)
            test_dataset = OnlineTestDatasetWrapper(dataset, "test")
            return train_dataset, test_dataset, dataset_name
    elif dataset_name in ["ljspeech-prepared", "ljspeech-dicodec18-kmeans512-prepared"]:
        from data.lj_speech_prepared import LJSpeechDataset
        dataset = LJSpeechDataset(
            force_vocab_build=force_vocab_build,
            dataset_dir_name=dataset_name,
        )
    elif dataset_name in ["libritts-r", "libritts_r"]:
        if not online_encode:
            raise ValueError(
                f"Dataset {dataset_name} provides raw audio; set "
                "training.online_encode=true to encode it during training."
            )
        from data.libri_tts_r_online import LibriTTSROnline
        dataset = LibriTTSROnline()
        train_dataset = OnlineTrainDatasetWrapper(dataset, train_split)
        test_dataset = OnlineTestDatasetWrapper(dataset, "test")
        return train_dataset, test_dataset, dataset_name
    elif dataset_name in [
        "libritts-r-prepared",
        "libritts_r_prepared",
        "libritts-r-dicodec18-kmeans512-prepared",
        "libritts-r-dicodec18-kmeans512-noecapa-prepared",
    ]:
        from data.libri_tts_r_prepared import LibriTTSRPrepared

        dataset = LibriTTSRPrepared(dataset_dir_name=dataset_name)
        discrete_only = training_cfg.get("discrete_only", False)
        train_dataset = TrainDatasetWrapper(dataset, train_split, discrete_only=discrete_only)
        test_dataset = TrainDatasetWrapper(dataset, "test", discrete_only=discrete_only)
        return train_dataset, test_dataset, dataset_name
    
    elif dataset_name in ["lj_speech_online", "lj-speech-online"]:
        if not online_encode:
            raise ValueError(
                f"Dataset {dataset_name} provides raw audio; set "
                "training.online_encode=true to encode it during training."
            )
        from data.lj_speech_online import (
            LJSpeechOnlineDataset,
            LJSpeechOnlineTrain,
            LJSpeechOnlineTest,
        )
        base = LJSpeechOnlineDataset()
        train_dataset = LJSpeechOnlineTrain(base)
        test_dataset = LJSpeechOnlineTest(base)
        return train_dataset, test_dataset, "libritts-r"  # reuse online collator path
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    discrete_only = training_cfg.get("discrete_only", False)
    train_dataset = TrainDatasetWrapper(dataset, "train", discrete_only=discrete_only)

    test_dataset = TestDatasetWrapper(
        dataset, "train", discrete_only=discrete_only # FIXME handle the test partition in Ljspeech (now missing)
    )

    return train_dataset, test_dataset, dataset_name


class CharTokenizer:
    def __init__(self):
        # 26 letters + space + 7 punctuation marks
        self.allowed_chars = "abcdefghijklmnopqrstuvwxyz ,-.!?\"'"
        self.char2id = {c: i for i, c in enumerate(self.allowed_chars)}
        self.vocab_size = len(self.char2id)

    def encode(self, text: str):
        # Convert to lowercase and filter out any character not in allowed_chars
        text = text.lower()
        return [self.char2id[c] for c in text if c in self.char2id]

def build_tokenizer(cfg_dict: Dict[str, Any], pretrinaed: bool = False):
    if pretrinaed:
        raise NotImplementedError("Pretrained tokenizer not supported yet.")

    char_tokenizer = None

    if cfg_dict.get("text_tokenizer") == "char":
        char_tokenizer = CharTokenizer()
        vocab_size = char_tokenizer.vocab_size
        print(f"Using custom CharTokenizer (vocab_size={vocab_size})")
    else:
        vocab_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "phoneme_vocab.json"
        )
        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                phoneme_vocab = json.load(f)
            vocab_size = len(phoneme_vocab)
        else:
            raise ValueError("Phoneme vocabulary not found. Please build it first.")

    vae_checkpoint = cfg_dict.get("vae_checkpoint")
    if not vae_checkpoint or not os.path.exists(vae_checkpoint):
        raise ValueError(
            "vae_checkpoint is required and must exist to load discrete_token_vocab_size"
        )
    from modules.builder import load_codebook_config_from_cfg
    from modules.audio_tokenizer import AudioTokenizer

    audio_bpe = None
    training_cfg = cfg_dict.get("training", {})
    continuous_only = training_cfg.get("continuous_only", False)
    discrete_only = training_cfg.get("discrete_only", False)
    _, discrete_token_vocab_size = load_codebook_config_from_cfg(cfg_dict)
    if cfg_dict.get("use_tokenize") and training_cfg.get("discrete_only", False):
        bpe_path = cfg_dict.get("bpe_tokenizer_path", "audio_bpe.json")
        if os.path.exists(bpe_path):
            audio_bpe = AudioTokenizer.load(bpe_path)
            discrete_token_vocab_size = audio_bpe.vocab_size
        else:
            raise ValueError(f"bpe_tokenizer_path '{bpe_path}' not found, but use_tokenize is True.")
    if continuous_only and discrete_only:
        raise ValueError("discrete_only and continuous_only are mutually exclusive.")
    if not continuous_only and discrete_token_vocab_size <= 0:
        raise ValueError(
            "Discrete or hybrid training requires a quantizer. Set "
            "training.semantic_quantizer_checkpoint or training.semantic_codebook_size."
        )
    end_audio_id = vocab_size + 2
    pad_token_id = vocab_size + 1
    start_audio_id = vocab_size + 0
    audio_placeholder_id = vocab_size + 3 if continuous_only else None

    prompt_vocab_size = vocab_size + (4 if continuous_only else 3)

    return HybridTokenizer(
        prompt_vocab_size=prompt_vocab_size,
        start_audio_id=start_audio_id,
        end_audio_id=end_audio_id,
        pad_id=pad_token_id,
        discrete_token_vocab_size=discrete_token_vocab_size,
        audio_bpe=audio_bpe,
        char_tokenizer=char_tokenizer,
        audio_placeholder_id=audio_placeholder_id,
    )
