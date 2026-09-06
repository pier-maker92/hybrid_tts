import csv
import os

import torchaudio
from torch.utils.data import Dataset

from data.audio_dataset import SimpleAudioDataset


class LJSpeechDataset(SimpleAudioDataset):
    """LJSpeech raw-audio dataset for online VAE encoding and char text prompts."""

    def __init__(self, dataset_dir: str | None = None, train_size: int = 12000):
        super().__init__()
        if dataset_dir is None:
            scratch = os.environ.get("SCRATCH", "/Users/software/Research")
            dataset_dir = os.path.join(scratch, "datasets", "LJSpeech-1.1")
            cluster_dataset_dir = "/scratch/piermel/datasets/LJSpeech-1.1"
            if not os.path.exists(dataset_dir) and os.path.exists(cluster_dataset_dir):
                dataset_dir = cluster_dataset_dir

        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        wav_dir = os.path.join(dataset_dir, "wavs")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"LJSpeech metadata not found: {metadata_path}. "
                "Set training.dataset_dir or SCRATCH to the directory that contains "
                "datasets/LJSpeech-1.1."
            )

        records = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if len(row) < 2:
                    continue
                uid = row[0].strip()
                text = row[2].strip() if len(row) > 2 else row[1].strip()
                wav_path = os.path.join(wav_dir, f"{uid}.wav")
                if os.path.exists(wav_path):
                    records.append(
                        {
                            "id": uid,
                            "transcription": text,
                            "wav_path": wav_path,
                        }
                    )

        if not records:
            raise RuntimeError(f"No LJSpeech wav files found under {wav_dir}")

        self._train = records[:train_size]
        self._test = records[train_size:]

    def _make_item(self, record):
        waveform, sr = torchaudio.load(record["wav_path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        text = record["transcription"]
        return {
            "audio": {"array": waveform.numpy(), "sampling_rate": sr},
            "ids": record["id"],
            "transcription": text,
            "input_ids": [0] * (len(text) + 100),
        }


class LJSpeechTrain(Dataset):
    def __init__(self, base: LJSpeechDataset):
        self.base = base
        self.records = base._train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.base._make_item(self.records[idx])


class LJSpeechTest(Dataset):
    def __init__(self, base: LJSpeechDataset):
        self.base = base
        self.records = base._test

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.base._make_item(self.records[idx])


LJSpeechOnlineDataset = LJSpeechDataset
LJSpeechOnlineTrain = LJSpeechTrain
LJSpeechOnlineTest = LJSpeechTest
