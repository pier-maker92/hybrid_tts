import torch
from dataclasses import dataclass
from typing import Sequence, Dict
import torchaudio.transforms as T
from torch.utils.data import Dataset

import torch
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence


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
        self, dataset: SimpleAudioDataset, split: str, discrete_only: bool = False
    ):
        super().__init__()
        assert split in ["train", "test"], "split must be either train or test"
        self.dataset = getattr(dataset, f"{split}_dataset")
        self.discrete_only = discrete_only

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = {}
        data = self.dataset[idx]

        data_dict["discrete_tokens"] = data.get("discrete")
        if not self.discrete_only:
            data_dict["continuous_tokens"] = data.get("continuous")
        else:
            data_dict["continuous_tokens"] = None

        data_dict["ids"] = data.get("id")
        data_dict["phoneme_ids"] = data.get("phoneme_ids")

        return data_dict


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(
        self, pad_id, start_audio_id, end_audio_id, vq_vocab_size, prompt_vocab_size
    ):
        self.pad_id = pad_id
        self.start_audio_id = start_audio_id
        self.end_audio_id = end_audio_id
        self.vq_vocab_size = vq_vocab_size
        self.prompt_vocab_size = prompt_vocab_size
        self.audio_eos_id = prompt_vocab_size + vq_vocab_size

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}

        # Basic fields
        batch["ids"] = [inst.get("ids") for inst in instances]

        # Phoneme IDs (to be renamed to prompt_ids in the batch)
        p_ids = [inst.get("phoneme_ids") for inst in instances]
        d_tokens = [inst.get("discrete_tokens") for inst in instances]
        c_tokens = [inst.get("continuous_tokens") for inst in instances]

        attention_mask = []
        discrete_sequence = []
        audio_padding_mask = []
        continuous_sequence = []
        target_tokens = []
        for p_id, d_token, c_token in zip(p_ids, d_tokens, c_tokens):
            # vocab mapping
            # prompt tokens keep their original IDs
            p_id = p_id + [self.start_audio_id]


            # discrete tokens are shifted by prompt_vocab_size
            shifted_d_token = [d + self.prompt_vocab_size for d in d_token]

            # sequence ends with audio_eos_id
            sequence = p_id + shifted_d_token + [self.end_audio_id]
            # append
            target_tokens.append(
                torch.tensor([-100] * len(p_id[1:]) + shifted_d_token + [self.end_audio_id]).long()
            )  # TODO check on this
            discrete_sequence.append(torch.tensor(sequence).long())
            attention_mask.append(torch.tensor([1] * len(sequence)).bool())
            # continuous tail
            if c_token:
                assert len(c_token) == len(
                    d_token
                ), (  # we have already added the audio_eos token
                    "continuous_tokens and discrete_tokens must have the same length"
                )
                if (
                    len(c_token[0]) == 64
                ):  # FIXME this is a temprary fix to handle the way I have extracted continuous token from prepare_data.py
                    # in the future, continuous token will be exactly the tail of the VQ
                    c_token = [c[32:] for c in c_token]
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


class TestDatasetWrapper(SimpleAudioDataset):
    def __init__(
        self, dataset: SimpleAudioDataset, split: str, discrete_only: bool = False
    ):
        super().__init__()
        assert split in ["test", "train"], "split must be test or train"
        self.dataset = getattr(dataset, f"{split}_dataset")
        self.discrete_only = discrete_only

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = {}
        data = self.dataset[idx]

        data_dict["discrete_tokens"] = data.get("discrete")
        if not self.discrete_only:
            data_dict["continuous_tokens"] = data.get("continuous")
        else:
            data_dict["continuous_tokens"] = [None] * len(data.get("continuous"))

        data_dict["ids"] = data.get("id")
        data_dict["phoneme_ids"] = data.get("phoneme_ids")

        # Robust transcription field lookup
        transcription = data.get("text_normalized") or data.get("transcript")
        self._process_transcription(data_dict, transcription)

        data_dict["language"] = data.get("language", "en")

        # LengthGroupedSampler looks for a list of tokens in the model input name (defaults to input_ids)
        # to calculate sequence lengths when precomputed lengths are not passed.
        discrete_len = len(data_dict["discrete_tokens"]) if data_dict["discrete_tokens"] is not None else 0
        phoneme_len = len(data_dict["phoneme_ids"]) if data_dict["phoneme_ids"] is not None else 0
        # Dummy input_ids list with the actual sequence length (phoneme_ids + shifted_discrete + end_audio_id)
        data_dict["input_ids"] = [0] * (phoneme_len + discrete_len + 1)
        return data_dict

    def _process_transcription(self, data_dict, transcription):
        data_dict.update({"transcription": transcription})
        return data_dict
