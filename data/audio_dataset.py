import torch
from dataclasses import dataclass
from typing import Sequence, Dict
import torchaudio.transforms as T
from torch.utils.data import Dataset

import torch
import torchaudio.transforms as T


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
    def __init__(self, dataset: SimpleAudioDataset, split: str):
        super().__init__()
        assert split in ["train", "test"], "split must be either train or test"
        self.dataset = getattr(dataset, f"{split}_dataset")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = {}
        data = self.dataset[idx]

        data_dict["discrete_tokens"] = data.get("discrete")
        data_dict["continuous_tokens"] = data.get("continuous")
        data_dict["ids"] = data.get("id")
        data_dict["phoneme_ids"] = data.get("phoneme_ids")
        return data_dict


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_id, start_audio_id=None, end_audio_id=None):
        self.pad_id = pad_id
        self.start_audio_id = start_audio_id
        self.end_audio_id = end_audio_id

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}

        # Basic fields
        batch["ids"] = [inst.get("ids") for inst in instances]

        # Phoneme IDs (to be renamed to prompt_ids in the batch)
        p_ids = [inst.get("phoneme_ids") for inst in instances]
        d_tokens = [inst.get("discrete_tokens") for inst in instances]
        c_tokens = [inst.get("continuous_tokens") for inst in instances]

        if all(x is not None for x in [p_ids, d_tokens, c_tokens]):
            # Padding phoneme_ids + build prompt_mask (True = valid, False = pad)
            # Append <start_audio> and <end_audio> to each sequence
            if self.start_audio_id is not None and self.end_audio_id is not None:
                p_ids = [list(p) + [self.start_audio_id, self.end_audio_id] for p in p_ids]

            max_p = max(len(x) for x in p_ids)
            padded_p = []
            prompt_mask = []
            for p in p_ids:
                curr_len = len(p)
                pad_len = max_p - curr_len
                p_tensor = torch.tensor(p, dtype=torch.long)
                padded_p.append(
                    torch.nn.functional.pad(p_tensor, (0, pad_len), value=self.pad_id)
                )
                prompt_mask.append(
                    torch.cat(
                        [
                            torch.ones(curr_len, dtype=torch.bool),
                            torch.zeros(pad_len, dtype=torch.bool),
                        ]
                    )
                )
            batch["prompt_ids"] = torch.stack(padded_p)
            batch["prompt_mask"] = torch.stack(prompt_mask)

            # Padding discrete/continuous tokens
            max_d = max(len(x) for x in d_tokens)
            padded_d = []
            padded_c = []
            padding_mask = []

            for d, c in zip(d_tokens, c_tokens):
                d_tensor = torch.tensor(d, dtype=torch.long)
                c_tensor = torch.tensor(c, dtype=torch.float)
                curr_len = len(d)
                pad_len = max_d - curr_len

                padded_d.append(
                    torch.nn.functional.pad(d_tensor, (0, pad_len), value=self.pad_id)
                )
                padded_c.append(
                    torch.nn.functional.pad(
                        c_tensor, (0, 0, 0, pad_len), value=self.pad_id
                    )
                )

                mask = torch.cat(
                    [
                        torch.zeros(curr_len, dtype=torch.bool),
                        torch.ones(pad_len, dtype=torch.bool),
                    ]
                )
                padding_mask.append(mask)

            batch["discrete_tokens"] = torch.stack(padded_d)
            # Create targets for CrossEntropyLoss (padding = -100)
            target_tokens = torch.stack(padded_d).clone()
            target_tokens[torch.stack(padding_mask)] = -100
            batch["target_tokens"] = target_tokens

            batch["continuous_tokens"] = torch.stack(padded_c)
            batch["padding_mask"] = torch.stack(padding_mask)

        # Include transcriptions if available
        transcriptions = [inst.get("transcription") for inst in instances]
        if all(x is not None for x in transcriptions):
            batch["transcription"] = transcriptions

        return batch


class TestDatasetWrapper(SimpleAudioDataset):
    def __init__(self, dataset: SimpleAudioDataset, split: str):
        super().__init__()
        assert split in ["test", "train"], "split must be test or train"
        self.dataset = getattr(dataset, f"{split}_dataset")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = {}
        data = self.dataset[idx]

        data_dict["discrete_tokens"] = data.get("discrete")
        data_dict["continuous_tokens"] = data.get("continuous")
        data_dict["ids"] = data.get("id")
        data_dict["phoneme_ids"] = data.get("phoneme_ids")

        # Robust transcription field lookup
        transcription = data.get("text_normalized") or data.get("transcript")
        self._process_transcription(data_dict, transcription)

        data_dict["language"] = data.get("language", "en")
        return data_dict

    def _process_transcription(self, data_dict, transcription):
        data_dict.update({"transcription": transcription})
        return data_dict
