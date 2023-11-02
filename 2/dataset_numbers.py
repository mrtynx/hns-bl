import numpy as np
from scipy.io import loadmat
from torch.utils.data import TensorDataset
import torch


def get_numbers_dataset(path: str):
    data = load_seq(path)
    x_train, y_train, x_test, y_test = clear_samples(*data)
    seq_len = max_seq_len(x_train, x_test)
    x_train_seq = pad_sequences(x_train, seq_len)
    x_test_seq = pad_sequences(x_test, seq_len)

    x_train_T = torch.tensor(x_train_seq, dtype=torch.float32)
    y_train_T = torch.tensor(y_train, dtype=torch.long)
    x_test_T = torch.tensor(x_test_seq, dtype=torch.float32)
    y_test_T = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(x_train_T, y_train_T)
    test_ds = TensorDataset(x_test_T, y_test_T)

    return train_ds, test_ds


def load_seq(path: str):
    data = loadmat(path)
    x_train = data["XTrain"]
    y_train = data["YTrain"]
    x_test = data["XTest"]
    y_test = data["YTest"]

    return x_train, y_train, x_test, y_test


def clear_samples(x_train, y_train, x_test, y_test):
    train_corrupted_indices = [0, 1, 2, 3, 4, 5, 75]
    test_corrupted_indices = [97]

    x_train = np.squeeze(x_train, axis=1)
    x_test = np.squeeze(x_test, axis=1)

    x_train = np.delete(x_train, train_corrupted_indices, axis=0)
    y_train = np.delete(y_train, train_corrupted_indices)
    x_test = np.delete(x_test, test_corrupted_indices, axis=0)
    y_test = np.delete(y_test, test_corrupted_indices)

    return x_train, y_train, x_test, y_test


def max_seq_len(x_train, x_test):
    _, x_train_max_arr = max(enumerate(x_train), key=lambda x: len(x[1][0]))
    _, x_test_max_arr = max(enumerate(x_test), key=lambda x: len(x[1][0]))

    x_train_len = len(x_train_max_arr[1])
    x_test_len = len(x_test_max_arr[1])
    if x_train_len >= x_test_len:
        return x_train_len
    else:
        return x_test_len


def pad_sequences(seq, seq_sz):
    padded_array = np.zeros((len(seq), 2, seq_sz))
    for i in range(len(seq)):
        pad_width = seq_sz - len(seq[i][0])
        padded_array[i][0] = np.pad(
            seq[i][0], (0, pad_width), "constant", constant_values=(0)
        )
        padded_array[i][1] = np.pad(
            seq[i][1], (0, pad_width), "constant", constant_values=(0)
        )

    return padded_array
