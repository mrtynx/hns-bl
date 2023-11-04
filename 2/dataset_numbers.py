import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


class NumberSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = torch.tensor(self.sequences[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return sequence, label


def prepare_numbers_dataset(path: str, pad="pre"):
    data = load_seq(path)

    x_train, y_train, x_test, y_test = clean_samples(*data)

    seq_len = max_seq_len(x_train, x_test)

    x_train = pad_sequences(x_train, seq_len, pad).transpose(0, 2, 1)
    x_test = pad_sequences(x_test, seq_len, pad).transpose(0, 2, 1)

    x_train = convert_dtype(x_train)
    x_test = convert_dtype(x_test)

    x_train = normalize(x_train)
    x_test = normalize(x_test)

    return x_train, y_train, x_test, y_test


def load_seq(path: str):
    data = loadmat(path)
    x_train = data["XTrain"]
    y_train = data["YTrain"]
    x_test = data["XTest"]
    y_test = data["YTest"]

    return x_train, y_train, x_test, y_test


def clean_samples(x_train, y_train, x_test, y_test):
    train_corrupted_indices = [0, 1, 2, 3, 4, 5, 75]
    test_corrupted_indices = [97]

    x_train = np.squeeze(x_train, axis=1)
    x_test = np.squeeze(x_test, axis=1)

    x_train = np.delete(x_train, train_corrupted_indices, axis=0)
    x_test = np.delete(x_test, test_corrupted_indices, axis=0)
    y_train = np.delete(y_train, train_corrupted_indices)
    y_test = np.delete(y_test, test_corrupted_indices)

    y_train -= 1
    y_test -= 1

    return x_train, y_train, x_test, y_test


def convert_dtype(seq):
    f32_arr = np.zeros_like(seq, dtype=np.float32)
    for i in range(len(seq)):
        f32_arr[i, :, 0] = seq[i, :, 0]
        f32_arr[i, :, 1] = seq[i, :, 1]

    return f32_arr


def normalize(seq):
    max_val = 0
    for i in range(len(seq)):
        if np.any(seq[i, :, 0] > max_val):
            max_val = max(seq[i, :, 0])

        if np.any(seq[i, :, 1] > max_val):
            max_val = max(seq[i, :, 1])

    for i in range(len(seq)):
        seq[i, :, 0] /= max_val
        seq[i, :, 1] /= max_val

    return seq


def max_seq_len(x_train, x_test):
    _, x_train_max_arr = max(enumerate(x_train), key=lambda x: len(x[1][0]))
    _, x_test_max_arr = max(enumerate(x_test), key=lambda x: len(x[1][0]))

    x_train_len = len(x_train_max_arr[1])
    x_test_len = len(x_test_max_arr[1])
    if x_train_len >= x_test_len:
        return x_train_len
    else:
        return x_test_len


def pad_sequences(seq, seq_sz, pad="pre"):
    padded_array = np.zeros((len(seq), 2, seq_sz))
    for i in range(len(seq)):
        arr_len = len(seq[i][0])
        pad_width = seq_sz - arr_len

        if pad == "pre":
            pad_style = (pad_width, 0)
        if pad == "post":
            pad_style = (0, pad_width)

        padded_array[i][0] = np.pad(
            seq[i][0], pad_style, "constant", constant_values=(0,)
        )
        padded_array[i][1] = np.pad(
            seq[i][1], pad_style, "constant", constant_values=(0,)
        )

    return padded_array
