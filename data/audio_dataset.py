import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torchaudio
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def _is_duration_within_max_audio_len(
    duration: Optional[float],
    max_audio_len: float,
) -> bool:
    return duration is None or duration <= max_audio_len


def _alignment_duration_seconds(items) -> Optional[float]:
    if not items:
        return None
    ends = [
        item.get("end")
        for item in items
        if isinstance(item, dict) and item.get("end") is not None
    ]
    if not ends:
        return None
    return float(max(ends))


def _is_alignment_within_max_audio_len(
    *alignment_columns,
    max_audio_len: float,
) -> bool:
    durations = [
        duration
        for duration in (
            _alignment_duration_seconds(items) for items in alignment_columns
        )
        if duration is not None
    ]
    duration = max(durations) if durations else None
    return _is_duration_within_max_audio_len(duration, max_audio_len)


def _is_audio_within_max_audio_len(audio: Optional[Dict], max_audio_len: float) -> bool:
    if audio is None:
        return True
    if "array" not in audio:
        return True
    duration = len(audio["array"]) / audio["sampling_rate"]
    return _is_duration_within_max_audio_len(duration, max_audio_len)


def _load_audio_data(audio_data, search_dirs=None) -> tuple[torch.Tensor, int]:
    if isinstance(audio_data, torch.Tensor):
        return audio_data.to(torch.float32), 24000

    if "array" in audio_data and audio_data["array"] is not None:
        return torch.as_tensor(audio_data["array"], dtype=torch.float32), int(
            audio_data["sampling_rate"]
        )

    if audio_data.get("bytes"):
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data["bytes"]))
    elif audio_data.get("path"):
        audio_path = Path(audio_data["path"])
        tried_paths = [audio_path]
        if not audio_path.is_absolute() and search_dirs is not None:
            for search_dir in search_dirs:
                candidate = Path(search_dir) / audio_path
                tried_paths.append(candidate)
                if candidate.exists():
                    audio_path = candidate
                    break
        if not audio_path.exists():
            tried = ", ".join(str(path) for path in tried_paths)
            raise FileNotFoundError(
                f"Audio bytes are missing and audio path could not be resolved. "
                f"Tried: {tried}"
            )
        waveform, sample_rate = torchaudio.load(str(audio_path))
    else:
        raise ValueError("Audio example must contain array, path, or bytes.")

    return waveform.to(torch.float32), int(sample_rate)


def _filter_by_max_audio_len(dataset, max_audio_len: Optional[float]):
    if max_audio_len is None:
        return dataset

    column_names = set(getattr(dataset, "column_names", []) or [])
    if "duration" in column_names:
        return dataset.filter(
            _is_duration_within_max_audio_len,
            input_columns=["duration"],
            fn_kwargs={"max_audio_len": float(max_audio_len)},
            num_proc=1,
        )
    if "duration_sec" in column_names:
        return dataset.filter(
            _is_duration_within_max_audio_len,
            input_columns=["duration_sec"],
            fn_kwargs={"max_audio_len": float(max_audio_len)},
            num_proc=1,
        )

    alignment_columns = [
        column for column in ("words", "phonemes") if column in column_names
    ]
    if alignment_columns:
        return dataset.filter(
            _is_alignment_within_max_audio_len,
            input_columns=alignment_columns,
            fn_kwargs={"max_audio_len": float(max_audio_len)},
            num_proc=1,
        )

    return dataset.filter(
        _is_audio_within_max_audio_len,
        input_columns=["audio"],
        fn_kwargs={"max_audio_len": float(max_audio_len)},
        num_proc=1,
    )


class SimpleAudioDataset(Dataset):
    def __init__(self):
        pass

    def _process_audio(self, audio: torch.Tensor, sr: int, target_sr: int):
        if target_sr is not None:  # handle resampling
            if sr != target_sr:
                audio = T.Resample(sr, target_sr)(audio)
            sr = target_sr
        # normalize audio
        audio = audio / (audio.abs().max() + 1e-8)
        return audio, sr

    def _process_audio_component(self, audio_data, target_sr, max_duration=None):
        """Helper method to process audio components with optional duration limiting"""
        audio_array = torch.Tensor(audio_data["array"]).to(torch.float32)
        audio, sr = self._process_audio(
            audio_array, audio_data["sampling_rate"], target_sr
        )
        if max_duration and audio.shape[0] > sr * max_duration:
            audio = audio[: sr * max_duration]
        return audio, sr

    def __len__(self):
        return len(self.train_dataset)

    def _process_audio_output(self, data_dict, audio_data):
        audio_output, sr_output = self._process_audio_component(
            audio_data,
            target_sr=24000,  # FIXME: hardcoded
        )
        data_dict.update(
            {"audio_output": [audio_output], "audio_output_sr": [sr_output]}
        )
        corrupted = self.augmenter(audio_output, sr_output)
        data_dict.update(
            {"corrupted_audio": [corrupted], "corrupted_audio_sr": [sr_output]}
        )


class TrainDatasetWrapper(SimpleAudioDataset):
    def __init__(
        self,
        dataset: SimpleAudioDataset,
        split: str,
        keep_continuous: bool = True,
        max_audio_len: Optional[float] = None,
    ):
        super().__init__()
        assert split in ["train", "test"], "split must be either train or test"
        self.dataset = getattr(dataset, f"{split}_dataset")
        self.dataset = _filter_by_max_audio_len(self.dataset, max_audio_len)
        self.keep_continuous = keep_continuous

        if not self.keep_continuous and "continuous" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns("continuous")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = {}
        data = self.dataset[idx]

        data_dict["discrete_tokens"] = data.get("discrete")
        if self.keep_continuous:
            data_dict["continuous_tokens"] = data.get("continuous")
        else:
            data_dict["continuous_tokens"] = None

        data_dict["ids"] = data.get("id")
        data_dict["phoneme_ids"] = data.get("phoneme_ids")

        # Pass raw text for on-the-fly char tokenization.
        data_dict["transcription"] = (
            data.get("text_normalized")
            or data.get("transcript")
            or data.get("transcription")
        )

        return data_dict


class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_id
        self.start_audio_id = tokenizer.start_audio_id
        self.end_audio_id = tokenizer.end_audio_id
        self.audio_placeholder_id = tokenizer.audio_placeholder_id
        self.vq_vocab_size = tokenizer.discrete_token_vocab_size
        self.prompt_vocab_size = tokenizer.prompt_vocab_size
        self.audio_eos_id = self.prompt_vocab_size + self.vq_vocab_size
        self.use_char_tokenizer = getattr(tokenizer, "char_tokenizer", None) is not None
        self.uses_audio_placeholder = self.audio_placeholder_id is not None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}

        # Basic fields
        batch["ids"] = [inst.get("ids") for inst in instances]

        # Text token IDs: either from char tokenizer (on-the-fly) or pre-computed phoneme_ids
        if self.use_char_tokenizer:
            p_ids = []
            for inst in instances:
                text = inst.get("transcription") or ""
                p_ids.append(self.tokenizer.encode_text(text))
        else:
            p_ids = [inst.get("phoneme_ids") for inst in instances]

        d_tokens = [inst.get("discrete_tokens") for inst in instances]
        c_tokens = [inst.get("continuous_tokens") for inst in instances]

        attention_mask = []
        discrete_sequence = []
        audio_padding_mask = []
        continuous_sequence = []
        target_tokens = []
        for p_id, d_token, c_token in zip(p_ids, d_tokens, c_tokens):
            if (
                d_token is not None
                and getattr(self.tokenizer, "audio_bpe", None) is not None
            ):
                d_token = self.tokenizer.audio_bpe.encode(d_token)

            if self.uses_audio_placeholder:
                if c_token is None:
                    raise ValueError(
                        "training.continuous=true with training.discrete=false "
                        "requires continuous_tokens."
                    )
                audio_len = len(c_token)
                prompt_tokens = [self.audio_placeholder_id] * len(p_id)
                shifted_d_token = [self.audio_placeholder_id] * audio_len
                target_tokens.append(torch.full((audio_len + 1,), -100).long())
            else:
                if d_token is None:
                    raise ValueError(
                        "discrete_tokens are required when training.discrete=true."
                    )
                prompt_tokens = p_id
                shifted_d_token = [d + self.prompt_vocab_size for d in d_token]
                target_tokens.append(
                    torch.tensor([d + 1 for d in d_token] + [0]).long()
                )  # Targets are the vq tokens starting at 1. 0 will be the end_audio_id in the decoder.

            # vocab mapping: prompt/special tokens keep their IDs; audio slots are shifted
            # VQ IDs in normal mode, or the generic placeholder in continuous-only mode.
            p_id = prompt_tokens + [self.start_audio_id]
            sequence = p_id + shifted_d_token + [self.end_audio_id]
            discrete_sequence.append(torch.tensor(sequence).long())
            attention_mask.append(torch.tensor([1] * len(sequence)).bool())
            # continuous tail
            if c_token is not None and len(c_token) > 0:
                if not self.uses_audio_placeholder:
                    assert len(c_token) == len(
                        d_token
                    ), "continuous_tokens and discrete_tokens must have the same length"  # we have already added the audio_eos token
                # if (
                #     len(c_token[0]) == 64
                # ):  # FIXME this is a temprary fix to handle the way I have extracted continuous token from prepare_data.py
                #     # in the future, continuous token will be exactly the tail of the VQ
                #     c_token = [c[32:] for c in c_token]
                continuous_sequence.append(torch.tensor(c_token))
                audio_padding_mask.append(torch.tensor([0] * len(c_token)).bool())

        # all discrete
        discrete_sequence = pad_sequence(
            discrete_sequence,
            batch_first=True,
            padding_value=self.pad_id,
        )
        target_tokens = pad_sequence(
            target_tokens,
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        batch["discrete_sequence"] = discrete_sequence
        batch["attention_mask"] = attention_mask
        batch["target_tokens"] = target_tokens

        # all continuous if present
        if len(continuous_sequence) > 0:
            continuous_sequence = pad_sequence(
                continuous_sequence,
                batch_first=True,
                padding_value=0,
            )
            audio_padding_mask = pad_sequence(
                audio_padding_mask,
                batch_first=True,
                padding_value=1,
            )
            batch["continuous_sequence"] = continuous_sequence
            batch["audio_padding_mask"] = audio_padding_mask
        else:
            batch["continuous_sequence"] = None
            batch["audio_padding_mask"] = None

        # Include transcriptions if available
        transcriptions = [inst.get("transcription") for inst in instances]
        if all(x is not None for x in transcriptions):
            batch["transcription"] = transcriptions

        return batch


class DiffusionDataCollator(object):
    """Collate examples for diffusion-only training.
    Passes discrete and continuous tokens exactly as they are without text/phoneme/LLM formatting.
    """

    def __init__(self, pad_id=1024, tokenizer=None):
        self.pad_id = pad_id
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}

        # Basic fields
        batch["ids"] = [inst.get("ids") for inst in instances]

        d_tokens = [inst.get("discrete_tokens") for inst in instances]
        c_tokens = [inst.get("continuous_tokens") for inst in instances]

        discrete_sequence = []
        continuous_sequence = []
        attention_mask = []
        audio_padding_mask = []

        for d_token, c_token in zip(d_tokens, c_tokens):
            if (
                self.tokenizer is not None
                and getattr(self.tokenizer, "audio_bpe", None) is not None
            ):
                d_token = self.tokenizer.audio_bpe.encode(d_token)

            discrete_sequence.append(torch.tensor(d_token).long())
            attention_mask.append(torch.tensor([1] * len(d_token)).bool())

            if c_token:
                continuous_sequence.append(torch.tensor(c_token))
                audio_padding_mask.append(torch.tensor([0] * len(c_token)).bool())

        # Pad discrete sequences
        discrete_sequence = pad_sequence(
            discrete_sequence,
            batch_first=True,
            padding_value=self.pad_id,
        )
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        batch["discrete_sequence"] = discrete_sequence
        batch["attention_mask"] = attention_mask

        # Pad continuous sequences if present
        if len(continuous_sequence) > 0:
            continuous_sequence = pad_sequence(
                continuous_sequence,
                batch_first=True,
                padding_value=0,
            )
            audio_padding_mask = pad_sequence(
                audio_padding_mask,
                batch_first=True,
                padding_value=1,
            )
            batch["continuous_sequence"] = continuous_sequence
            batch["audio_padding_mask"] = audio_padding_mask
        else:
            batch["continuous_sequence"] = None
            batch["audio_padding_mask"] = None

        return batch


class TestDatasetWrapper(SimpleAudioDataset):
    def __init__(
        self,
        dataset: SimpleAudioDataset,
        split: str,
        keep_continuous: bool = True,
        max_audio_len: Optional[float] = None,
    ):
        super().__init__()
        assert split in ["test", "train"], "split must be test or train"
        self.dataset = getattr(dataset, f"{split}_dataset")
        self.dataset = _filter_by_max_audio_len(self.dataset, max_audio_len)
        self.keep_continuous = keep_continuous

        if not self.keep_continuous and "continuous" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns("continuous")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = {}
        data = self.dataset[idx]

        data_dict["discrete_tokens"] = data.get("discrete")
        if self.keep_continuous:
            data_dict["continuous_tokens"] = data.get("continuous")
        else:
            data_dict["continuous_tokens"] = None

        data_dict["ids"] = data.get("id")
        data_dict["phoneme_ids"] = data.get("phoneme_ids")

        # Robust transcription field lookup
        transcription = data.get("text_normalized") or data.get("transcript")
        self._process_transcription(data_dict, transcription)

        data_dict["language"] = data.get("language", "en")

        # LengthGroupedSampler looks for a list of tokens in the model input name (defaults to input_ids)
        # to calculate sequence lengths when precomputed lengths are not passed.
        discrete_len = (
            len(data_dict["discrete_tokens"])
            if data_dict["discrete_tokens"] is not None
            else 0
        )
        phoneme_len = (
            len(data_dict["phoneme_ids"]) if data_dict["phoneme_ids"] is not None else 0
        )
        # Dummy input_ids list with the actual sequence length (phoneme_ids + shifted_discrete + end_audio_id)
        data_dict["input_ids"] = [0] * (phoneme_len + discrete_len + 1)
        return data_dict

    def _process_transcription(self, data_dict, transcription):
        data_dict.update({"transcription": transcription})
        return data_dict


# ---------------------------------------------------------------------------
# Online wrappers (raw audio → DataCollatorWithVAE handles VAE encoding)
# ---------------------------------------------------------------------------


class OnlineTrainDatasetWrapper(Dataset):
    """Wraps a SimpleAudioDataset for online VAE encoding.
    Returns raw audio dicts instead of pre-computed discrete/continuous tokens."""

    def __init__(
        self,
        dataset: SimpleAudioDataset,
        split: str,
        max_audio_len: Optional[float] = None,
    ):
        assert split in ["train", "test"]
        self.dataset = getattr(dataset, f"{split}_dataset")
        self.dataset = _filter_by_max_audio_len(self.dataset, max_audio_len)
        self.audio_roots = getattr(dataset, "audio_roots", {}).get(split, [])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        phoneme_ids = data.get("phoneme_ids") or []
        return {
            "audio": data["audio"],
            "audio_roots": self.audio_roots,
            "phoneme_ids": phoneme_ids,
            "ids": data.get("id", str(idx)),
            "transcription": (
                data.get("text_normalized")
                or data.get("transcript")
                or data.get("transcription", "")
            ),
            # dummy for LengthGroupedSampler compatibility
            "input_ids": [0] * (len(phoneme_ids) + 100),
        }


class OnlineTestDatasetWrapper(OnlineTrainDatasetWrapper):
    pass


class DataCollatorWithVAE:
    """DataCollator that encodes raw audio through a frozen VAE online,
    then delegates to DataCollator for sequence building."""

    TARGET_SR = 24000

    def __init__(
        self,
        tokenizer,
        vae,
        device,
        discrete: bool = True,
        continuous: bool = True,
        quantizer_source: str = "z",
        include_quantizer_residual: bool = False,
    ):
        self._base = DataCollator(tokenizer)
        self.vae = vae
        self.device = device
        self.discrete = discrete
        self.continuous = continuous
        self.quantizer_source = quantizer_source
        self.include_quantizer_residual = include_quantizer_residual
        if not (self.discrete or self.continuous):
            raise ValueError("At least one of discrete or continuous must be enabled.")
        if self.quantizer_source not in {"z", "z_sem"}:
            raise ValueError("quantizer_source must be either 'z' or 'z_sem'.")

    @torch.no_grad()
    def _encode_batch(self, instances):
        self.vae.eval()  # guard: prevent feature_extractor from updating std/mean statistics
        audios_srs = []
        reference_audios_srs = []
        for inst in instances:
            tensor, sr = _load_audio_data(inst["audio"], inst.get("audio_roots"))
            if tensor.dim() > 1:
                tensor = tensor.mean(dim=0)
            tensor = tensor / (tensor.abs().max() + 1e-8)
            if sr != self.TARGET_SR:
                tensor = torchaudio.functional.resample(tensor, sr, self.TARGET_SR)
            reference_audios_srs.append((tensor.detach().cpu(), self.TARGET_SR))
            audios_srs.append((tensor.to(self.device), self.TARGET_SR))

        features, padding_mask = self.vae.extract_features(audios_srs)[:2]
        vae_dtype = (
            self.vae.dtype
            if hasattr(self.vae, "dtype")
            else next(self.vae.parameters()).dtype
        )
        features = features.to(vae_dtype)
        encoder_output = self.vae.encode(
            features,
            padding_mask,
            run_quantizer=self.discrete,
        )

        discrete_list, continuous_list = [], []
        for i in range(len(instances)):
            mask = ~encoder_output.padding_mask[i]
            if not self.discrete:
                discrete_list.append(None)
                continuous_list.append(encoder_output.z[i][mask].float().cpu().tolist())
                continue

            quantizer_output = getattr(encoder_output, "quantizer_output", None)
            indices = (
                getattr(quantizer_output, "indices", None)
                if quantizer_output is not None
                else getattr(encoder_output, "indices", None)
            )
            if indices is None:
                raise RuntimeError(
                    "Discrete token training requires an encoder or semantic quantizer."
                )
            discrete_list.append(indices[i][mask].cpu().tolist())

            if not self.continuous:
                continuous_list.append(None)
                continue

            residual = (
                getattr(quantizer_output, "residual", None)
                if quantizer_output is not None
                else getattr(encoder_output, "residual", None)
            )
            if self.quantizer_source == "z":
                if residual is None:
                    raise RuntimeError("source='z' hybrid training requires residual.")
                continuous_values = residual
            else:
                if quantizer_output is None or quantizer_output.z_pros is None:
                    raise RuntimeError(
                        "source='z_sem' requires an external semantic quantizer "
                        "that provides z_pros."
                    )
                continuous_values = quantizer_output.z_pros
                if self.include_quantizer_residual:
                    if residual is None:
                        raise RuntimeError(
                            "include_quantizer_residual=True requires residual."
                        )
                    continuous_values = continuous_values + residual

            continuous_list.append(continuous_values[i][mask].float().cpu().tolist())
        return discrete_list, continuous_list, reference_audios_srs

    def __call__(self, instances):
        discrete_list, continuous_list, reference_audios_srs = self._encode_batch(
            instances
        )
        patched = [
            {
                "discrete_tokens": discrete_list[i],
                "continuous_tokens": continuous_list[i],
                "phoneme_ids": instances[i].get("phoneme_ids"),
                "ids": instances[i].get("ids"),
                "transcription": instances[i].get("transcription"),
            }
            for i in range(len(instances))
        ]
        batch = self._base(patched)
        batch["reference_audios_srs"] = reference_audios_srs
        return batch
