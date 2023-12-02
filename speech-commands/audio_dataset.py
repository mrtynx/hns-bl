import os

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, root_dir, txt_file):
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
        audio, sample_rate = torchaudio.load(self.file_paths[idx])
        resampler = T.Resample(orig_freq=sample_rate, new_freq=sample_rate // 2)
        audio = resampler(audio)
        label = self.labels[idx]
        label = np.eye(len(self.label_mapping))[label]
        return audio.view(-1, 1), label


def create_training_list(dataset_path):
    ds_folder = os.scandir(dataset_path)

    all_entries = list()
    for entry in ds_folder:
        if os.path.isdir(entry) and entry.name != "_background_noise_":
            wav_files = os.scandir(entry.path)
            for wav_file in wav_files:
                all_entries.append(f"{entry.name}/{wav_file.name}\n")

    with open(os.path.join(dataset_path, "testing_list.txt"), "r") as test_file:
        test_list = list(test_file.read().splitlines())

    with open(os.path.join(dataset_path, "validation_list.txt"), "r") as val_file:
        val_list = list(val_file.read().splitlines())

    with open(os.path.join(dataset_path, "training_list.txt"), "a") as train_file:
        for line in all_entries:
            if line.strip() not in test_list and line.strip() not in val_list:
                train_file.write(line)


def padding_collate_fn(batch):
    # Separate data and labels
    data, labels = zip(*batch)

    # Find the length of the longest sequence in the batch
    max_seq_length = max(len(seq) for seq in data)

    # Pad sequences in data to the length of the longest sequence
    padded_data = [
        torch.cat([seq, torch.zeros(max_seq_length - len(seq), *seq.shape[1:])])
        for seq in data
    ]

    # Stack padded sequences
    padded_data = torch.stack(padded_data, dim=0)

    # Convert to tensor
    # labels = torch.tensor(np.array(labels))
    labels = torch.tensor(np.array(labels), dtype=torch.float32)

    return padded_data, labels
