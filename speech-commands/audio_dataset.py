import os
import random

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
        self,
        root_dir,
        txt_file,
        fs=16000,
        downsample=True,
        noise_prob=0,
        noise_max_gain=0.3,
        use_specgram=False,
    ):
        self.root_dir = root_dir
        self.noise_prob = noise_prob
        self.downsample = downsample
        self.noise_max_gain = noise_max_gain
        self.use_specgram = use_specgram
        self.fs = fs // 2 if self.downsample else fs

        if noise_prob != 0:
            self.noise_tensors = self.parse_noise_files()

        if txt_file == "training_list.txt" and not os.path.exists(
            os.path.join(root_dir, "training_list.txt")
        ):
            self.create_training_list()

        self.labels = []
        self.file_paths = []
        self.label_mapping = {"go": 0, "left": 1, "right": 2, "stop": 3}
        with open(os.path.join(root_dir, txt_file), "r") as f:
            for line in f:
                path = line.strip()
                label, _ = line.strip().split("/")
                self.file_paths.append(os.path.join(root_dir, path))
                self.labels.append(self.label_mapping[label])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio, _ = torchaudio.load(self.file_paths[idx])
        resampler = T.Resample(orig_freq=self.fs, new_freq=self.fs)
        audio = resampler(audio)
        label = self.labels[idx]
        label = np.eye(len(self.label_mapping))[label]

        if np.random.uniform(0, 1) < self.noise_prob:
            noise = self.sample_noise(noise_duration_seconds=1)
            audio += noise[: len(audio)]

        return audio.view(-1, 1), label

    def create_training_list(self):
        ds_folder = os.scandir(self.root_dir)

        all_entries = list()
        for entry in ds_folder:
            if os.path.isdir(entry) and entry.name != "_background_noise_":
                wav_files = os.scandir(entry.path)
                for wav_file in wav_files:
                    all_entries.append(f"{entry.name}/{wav_file.name}\n")

        with open(os.path.join(self.root_dir, "testing_list.txt"), "r") as test_file:
            test_list = list(test_file.read().splitlines())

        with open(os.path.join(self.root_dir, "validation_list.txt"), "r") as val_file:
            val_list = list(val_file.read().splitlines())

        with open(os.path.join(self.root_dir, "training_list.txt"), "a") as train_file:
            for line in all_entries:
                if line.strip() not in test_list and line.strip() not in val_list:
                    train_file.write(line)

    def parse_noise_files(self):
        noise_dir = os.scandir(os.path.join(self.root_dir, "_background_noise_"))
        noise_tensors = []
        for entry in noise_dir:
            if entry.name.endswith(".wav"):
                noise, _ = torchaudio.load(entry.path)
                noise_tensors.append(noise)

        return noise_tensors

    def sample_noise(self, noise_duration_seconds):
        noise = random.choice(self.noise_tensors)
        noise_fs = self.fs
        resampler = T.Resample(orig_freq=self.fs, new_freq=noise_fs)
        noise = resampler(noise)
        noise = noise.squeeze(dim=0)
        start_point = np.random.randint(
            len(noise) // 2, len(noise) - noise_fs * noise_duration_seconds
        )
        noise_segment = noise[
            start_point : start_point + noise_fs * noise_duration_seconds
        ]

        return np.random.uniform(0, self.noise_max_gain) * noise_segment

    def collate_fn(self, batch):
        data, labels = zip(*batch)

        labels = torch.tensor(np.array(labels), dtype=torch.float32)

        max_seq_len = max(tensor.size(0) for tensor in data)

        padded_tensors = [
            F.pad(tensor, (0, 0, 0, max_seq_len - tensor.size(0))) for tensor in data
        ]

        if self.use_specgram:
            specgram_list = []
            for tensor in padded_tensors:
                tensor = tensor.permute(1, 0)
                specgram = T.MelSpectrogram(
                    sample_rate=self.fs,
                )(tensor)
                specgram = specgram.permute(0, 2, 1)
                specgram_list.append(specgram)

            collated_data = torch.cat(specgram_list, dim=0)

        else:
            collated_data = torch.cat(
                [tensor.unsqueeze(0) for tensor in padded_tensors], dim=0
            )

        return collated_data, labels
