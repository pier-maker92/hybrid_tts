import os
import json
import torch
import random
from dataclasses import dataclass
import torchaudio.transforms as T
from torch.utils.data import Dataset
from typing import Optional, Sequence, Dict

import torch
import torch.nn as nn
import torchaudio.transforms as T
from torchaudio_filters import LowPass, Pad, BandPass
import random


class AudioAugmenter(nn.Module):
    def __init__(
        self,
        p_gain=0.7,
        p_gaussian=0.7,
        p_colored=0.6,
        p_clipping=0.8,
        p_dropout=1.0,
        p_lowpass=0.4,
        p_bandpass=0.4,
    ):
        super().__init__()

        self.p_gain = p_gain
        self.p_gaussian = p_gaussian
        self.p_colored = p_colored
        self.p_clipping = p_clipping
        self.p_dropout = p_dropout
        self.p_lowpass = p_lowpass
        self.p_bandpass = p_bandpass

    # -------------------------
    # augmentations
    # -------------------------
    def add_gaussian_noise(self, audio, std):
        noise = torch.randn_like(audio) * std
        return audio + noise

    def add_colored_noise(self, audio, std, alpha):
        noise = torch.randn_like(audio)

        freqs = torch.fft.rfftfreq(audio.shape[-1], d=1.0).to(audio.device)
        freqs[0] = 1e-6

        spectrum = torch.fft.rfft(noise)
        spectrum = spectrum / (freqs ** (alpha / 2))

        colored = torch.fft.irfft(spectrum, n=audio.shape[-1])
        return audio + std * colored

    def random_gain(self, audio):
        gain = random.uniform(0.7, 1.3)
        return audio * gain

    def random_clipping(self, audio):
        threshold = random.uniform(0.3, 0.9)
        audio = torch.clamp(audio, -threshold, threshold) / threshold
        return audio

    def time_dropout(self, audio):
        T_len = audio.shape[-1]
        width = int(T_len * random.uniform(0.005, 0.02))

        if width > 0:
            start = random.randint(0, max(0, T_len - width))
            audio[..., start : start + width] = 0

        return audio

    def lowpass(self, audio, sr):
        cutoff = random.uniform(3000, 8000)
        return LowPass(cutoff, sr)(audio)

    def bandpass(self, audio, sr):
        low = random.uniform(100, 400)
        high = random.uniform(7000, 12000)
        bp = BandPass(low, high, sr)
        return bp(audio)

    # -------------------------
    # forward
    # -------------------------
    def forward(self, audio: torch.Tensor, sr: int):
        """
        audio: (B, T) or (1, T)
        """

        if random.random() < self.p_gain:
            audio = self.random_gain(audio)

        if random.random() < self.p_gaussian:
            std = random.uniform(0.001, 0.002)
            audio = self.add_gaussian_noise(audio, std)

        if random.random() < self.p_colored:
            alpha = random.uniform(0.001, 0.002)
            std = random.uniform(0.0001, 0.0005)
            audio = self.add_colored_noise(audio, std=std, alpha=alpha)

        if random.random() < self.p_clipping:
            audio = self.random_clipping(audio)

        if random.random() < 0.5:
            if random.random() < self.p_bandpass:
                audio = self.bandpass(audio, sr)
        else:
            if random.random() < self.p_lowpass:
                audio = self.lowpass(audio, sr)

        return audio


class SimpleAudioDataset(Dataset):
    def __init__(self):
        self.augmenter = AudioAugmenter()
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


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = dict()
        # handling etherogeneous samples in the batch, if a key is not present in the batch, add None in the index corresponding to the sample
        batch_input_audios_srs = [None] * len(instances)
        batch_output_audios_srs = [None] * len(instances)
        batch_condition_audios_srs = [None] * len(instances)
        batch_transcription_ids = [None] * len(instances)
        batch_aligned_transcription_ids = [None] * len(instances)
        batch_transcription = [None] * len(instances)
        batch_language = [None] * len(instances)
        batch_ids = [None] * len(instances)
        batch_corrupted_audios_srs = [None] * len(instances)
        batch_phoneme_alignments = [None] * len(instances)
        batch_discrete_tokens = [None] * len(instances)
        batch_continuous_tokens = [None] * len(instances)
        for i, instance in enumerate(instances):
            if "audio_input" in instance:
                batch_input_audios_srs[i] = (
                    instance["audio_input"][0],
                    instance["audio_input_sr"][0],
                )
            if "audio_output" in instance:
                batch_output_audios_srs[i] = (
                    instance["audio_output"][0],
                    instance["audio_output_sr"][0],
                )
            if "audio_condition" in instance:
                batch_condition_audios_srs[i] = (
                    instance["audio_condition"][0],
                    instance["audio_condition_sr"][0],
                )
            if "corrupted_audio" in instance:
                batch_corrupted_audios_srs[i] = (
                    instance["corrupted_audio"][0],
                    instance["corrupted_audio_sr"][0],
                )
            if "phoneme_alignments" in instance:
                batch_phoneme_alignments[i] = instance["phoneme_alignments"]
            if "transcription_ids" in instance:
                batch_transcription_ids[i] = instance["transcription_ids"]
            if "aligned_transcription_ids" in instance:
                batch_aligned_transcription_ids[i] = instance[
                    "aligned_transcription_ids"
                ]
            if "transcription" in instance:
                batch_transcription[i] = instance["transcription"]
            if "language" in instance:
                batch_language[i] = instance["language"]
            if "ids" in instance:
                batch_ids[i] = instance["ids"]
            if "discrete_tokens" in instance:
                batch_discrete_tokens[i] = instance["discrete_tokens"]
            if "continuous_tokens" in instance:
                batch_continuous_tokens[i] = instance["continuous_tokens"]

        # if not all none add to the batch
        def all_none(batch):
            return all([x is None for x in batch])

        if not all_none(batch_input_audios_srs):
            batch["input_audios_srs"] = batch_input_audios_srs
        if not all_none(batch_output_audios_srs):
            batch["output_audios_srs"] = batch_output_audios_srs
        if not all_none(batch_condition_audios_srs):
            batch["condition_audios_srs"] = batch_condition_audios_srs
        if not all_none(batch_transcription_ids):
            batch["transcription_ids"] = batch_transcription_ids
        if not all_none(batch_aligned_transcription_ids):
            batch["aligned_transcription_ids"] = batch_aligned_transcription_ids
        if not all_none(batch_transcription):
            batch["transcription"] = batch_transcription
        if not all_none(batch_language):
            batch["language"] = batch_language
        if not all_none(batch_ids):
            batch["ids"] = batch_ids
        if not all_none(batch_corrupted_audios_srs):
            batch["corrupted_audios_srs"] = batch_corrupted_audios_srs
        if not all_none(batch_phoneme_alignments):
            batch["phoneme_alignments"] = batch_phoneme_alignments

        if not all_none(batch_discrete_tokens) and not all_none(batch_continuous_tokens):
            max_len = max(len(t) for t in batch_discrete_tokens if t is not None)
            padded_discrete = []
            padded_continuous = []
            padding_mask = []
            for d, c in zip(batch_discrete_tokens, batch_continuous_tokens):
                if d is None or c is None:
                    continue
                d_tensor = torch.tensor(d, dtype=torch.long)
                c_tensor = torch.tensor(c, dtype=torch.float)
                curr_len = len(d)
                pad_len = max_len - curr_len
                
                # pad discrete with -100
                d_pad = torch.nn.functional.pad(d_tensor, (0, pad_len), value=-100)
                # pad continuous with 0.0
                c_pad = torch.nn.functional.pad(c_tensor, (0, 0, 0, pad_len), value=0.0)
                # mask: False for valid, True for padding
                mask = torch.cat([torch.zeros(curr_len, dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)])
                
                padded_discrete.append(d_pad)
                padded_continuous.append(c_pad)
                padding_mask.append(mask)
            
            batch["discrete_tokens"] = torch.stack(padded_discrete)
            batch["continuous_tokens"] = torch.stack(padded_continuous)
            batch["padding_mask"] = torch.stack(padding_mask)

        return batch


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
        data_dict["phoneme_alignments"] = data.get("phonemes", None)
        return data_dict


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

        # Robust transcription field lookup
        transcription = (
            data.get("text_normalized") or data.get("transcript") or "transcript"
        )
        self._process_transcription(data_dict, transcription)

        data_dict["language"] = data.get("language", "en")
        data_dict["ids"] = data.get("id")
        return data_dict

    def _process_transcription(self, data_dict, transcription):
        data_dict.update({"transcription": transcription})
        return data_dict
