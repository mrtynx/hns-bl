import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os


class AudioDataset(Dataset):
    def __init__(self, txt_file, root_dir):
        self.labels = []
        self.file_paths = []
        with open(txt_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                self.file_paths.append(os.path.join(root_dir, path))
                self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio, sample_rate = torchaudio.load(self.file_paths[idx])
        # Preprocess the audio here (e.g., resampling, MFCC, etc.)
        label = self.labels[idx]
        return audio, label
