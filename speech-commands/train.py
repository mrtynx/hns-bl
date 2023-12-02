import os
import sys

sys.path.append("../")

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from audio_dataset import AudioDataset, padding_collate_fn
from models import CNNLSTMModel
from pretty_confusion_matrix import pp_matrix_from_data
from torch.utils.data import DataLoader

from routine_functions import eval_model, plot_train_perf, train_model

parser = argparse.ArgumentParser()

parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-lr", type=float, default=0.001)
parser.add_argument("-epochs", type=int, default=30)
parser.add_argument("-shuffle", action="store_true")
parser.add_argument("-save_model", action="store_true")


def main():
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    SHUFFLE = args.shuffle
    SAVE_MODEL = args.save_model

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")

    train_ds = AudioDataset(
        root_dir="audio_dataset_files", txt_file="training_list.txt"
    )

    val_ds = AudioDataset(
        root_dir="audio_dataset_files", txt_file="validation_list.txt"
    )

    test_ds = AudioDataset(root_dir="audio_dataset_files", txt_file="testing_list.txt")

    trainloader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE, collate_fn=padding_collate_fn
    )
    valloader = DataLoader(
        val_ds, batch_size=32, shuffle=False, collate_fn=padding_collate_fn
    )
    testloader = DataLoader(
        test_ds, batch_size=32, shuffle=False, collate_fn=padding_collate_fn
    )

    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 4

    model = CNNLSTMModel(input_size, hidden_size, num_layers, output_size).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # %% Train model

    stats, best_model = train_model(
        model,
        trainloader=trainloader,
        valloader=valloader,
        num_epochs=N_EPOCHS,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
    )

    # %% Save best model

    if SAVE_MODEL:
        torch.save(best_model.state_dict(), f"best_model.pt")

    # %% Metrics
    predictions, labels = eval_model(best_model, testloader)

    labels = np.argmax(labels, axis=1)

    plot_train_perf(
        model=best_model,
        path=None,
        train_loss=stats["train_loss"],
        train_acc=stats["train_acc"],
        val_loss=stats["val_loss"],
        val_acc=stats["val_acc"],
    )

    pp_matrix_from_data(
        labels, predictions, columns=[i for i in range(4)], cmap="gnuplot"
    )


if __name__ == "__main__":
    main()

# %%
