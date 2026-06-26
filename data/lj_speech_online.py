import os
import json
import csv
import torchaudio
import torch
from g2p_en import G2p
from torch.utils.data import Dataset
from data.audio_dataset import SimpleAudioDataset

SCRATCH = os.environ["SCRATCH"]
LJ_DIR = os.path.join(SCRATCH, "datasets", "LJSpeech-1.1")
SAMPLE_RATE = 22050


class LJSpeechOnlineDataset(SimpleAudioDataset):
    """LJSpeech dataset returning raw audio for online VAE encoding.
    Train/test split: first 12000 samples for train, rest for test."""

    def __init__(self):
        super().__init__()
        vocab_path = os.path.join(os.path.dirname(__file__), "phoneme_vocab.json")
        if not os.path.exists(vocab_path):
            raise ValueError(f"Phoneme vocab not found at {vocab_path}")
        with open(vocab_path) as f:
            self.phoneme_vocab = json.load(f)

        g2p = G2p()
        metadata_path = os.path.join(LJ_DIR, "metadata.csv")
        records = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if len(row) < 2:
                    continue
                uid = row[0].strip()
                text = row[2].strip() if len(row) > 2 else row[1].strip()
                phoneme_ids = [self.phoneme_vocab.get(p, 0) for p in g2p(text)]
                wav_path = os.path.join(LJ_DIR, "wavs", f"{uid}.wav")
                if not os.path.exists(wav_path):
                    continue
                records.append({"id": uid, "text": text, "phoneme_ids": phoneme_ids, "wav_path": wav_path})

        self._train = records[:12000]
        self._test = records[12000:]

    def _make_item(self, record):
        waveform, sr = torchaudio.load(record["wav_path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        phoneme_ids = record["phoneme_ids"]
        return {
            "audio": {"array": waveform.numpy(), "sampling_rate": sr},
            "phoneme_ids": phoneme_ids,
            "ids": record["id"],
            "transcription": record["text"],
            "input_ids": [0] * (len(phoneme_ids) + 100),
        }


class LJSpeechOnlineTrain(Dataset):
    def __init__(self, base: LJSpeechOnlineDataset):
        self.base = base
        self.records = base._train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.base._make_item(self.records[idx])


class LJSpeechOnlineTest(Dataset):
    def __init__(self, base: LJSpeechOnlineDataset):
        self.base = base
        self.records = base._test

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.base._make_item(self.records[idx])
