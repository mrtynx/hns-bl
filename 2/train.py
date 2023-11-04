import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from dataset_numbers import NumberSequenceDataset, prepare_numbers_dataset
from metrics import eval_model, plot_train_perf
from models import LSTMModel
from pretty_confusion_matrix import pp_matrix_from_data
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-lr", type=float, default=0.001)
parser.add_argument("-epochs", type=int, default=30)
parser.add_argument("-pad", choices=["pre", "post"], default="pre")
parser.add_argument("-shuffle", action="store_true")


def main():
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    PAD = args.pad
    SHUFFLE = args.shuffle

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    x_train, y_train, x_test, y_test = prepare_numbers_dataset("dataset.mat", pad=PAD)

    train_ds = NumberSequenceDataset(x_train, y_train)
    test_ds = NumberSequenceDataset(x_test, y_test)

    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    testloader = DataLoader(test_ds, batch_size=32, shuffle=False)

    input_size = 2
    hidden_size = 128
    num_layers = 2
    output_size = 10

    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loss_hist = []
    train_acc_hist = []
    val_loss_hist = []
    val_acc_hist = []

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_accuracy = 0

        # %% Train
        model.train()
        for step, (inputs, targets) in enumerate(trainloader):
            print(f"\rBatch {step+1}/{len(trainloader)}", end="", flush=True)

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted_classes = torch.max(outputs.data, 1)

            loss = criterion(outputs, targets)
            train_loss += loss.item()

            train_total += targets.size(0)
            train_correct += (predicted_classes == targets).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(trainloader)
        train_accuracy = train_correct / train_total

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_accuracy)

        print(
            f"\rEpoch: {epoch}/{N_EPOCHS} | Train Accuracy {train_accuracy:.2f} | Train Loss {train_loss:.2f}",
            end="",
            flush=True,
        )

        val_loss = 0
        val_accuracy = 0
        val_correct = 0
        val_total = 0

        # %% Valid
        model.eval()
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(testloader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                _, predicted_classes = torch.max(outputs.data, 1)

                loss = criterion(outputs, targets)
                val_loss += loss.item()

                val_total += targets.size(0)
                val_correct += (predicted_classes == targets).sum().item()

            val_loss /= len(testloader)
            val_accuracy = val_correct / val_total

            val_loss_hist.append(val_loss)
            val_acc_hist.append(val_accuracy)

            print(
                f" | Val Accuracy {val_accuracy:.2f} | Val Loss {val_loss:.2f} | LR {LEARNING_RATE:.5f}",
            )

    # %% Metrics
    predictions, labels = eval_model(model, testloader)

    plot_train_perf(
        model=model,
        path=None,
        train_loss=train_loss_hist,
        train_acc=train_acc_hist,
        val_loss=val_loss_hist,
        val_acc=val_acc_hist,
    )

    pp_matrix_from_data(
        labels, predictions, columns=[i for i in range(10)], cmap="gnuplot"
    )


if __name__ == "__main__":
    main()

# %%
