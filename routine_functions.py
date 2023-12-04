import copy
import pickle

import matplotlib.pyplot as plt
import torch
import numpy as np


def train_model(
    model, trainloader, num_epochs, criterion, optimizer, valloader=None, device=None
):
    if valloader is not None:
        val_loss_hist = []
        val_acc_hist = []

    if device is None:
        device = torch.device("cpu")

    train_loss_hist = []
    train_acc_hist = []

    for epoch in range(1, num_epochs + 1):
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

            if (
                criterion.__class__.__name__ == "BCEWithLogitsLoss"
                or criterion.__class__.__name__ == "BCELoss"
            ):
                _, targets = torch.max(targets, 1)

            train_correct += (predicted_classes == targets).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(trainloader)
        train_accuracy = train_correct / train_total

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_accuracy)

        print(
            f"\rEpoch: {epoch}/{num_epochs} | Train Accuracy {train_accuracy:.2f} | Train Loss {train_loss:.2f}",
            end="",
            flush=True,
        )

        if valloader is not None:
            val_loss = 0
            val_accuracy = 0
            val_correct = 0
            val_total = 0
            model.eval()
            with torch.no_grad():
                for _, (inputs, targets) in enumerate(valloader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)
                    _, predicted_classes = torch.max(outputs.data, 1)

                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    val_total += targets.size(0)

                    if (
                        criterion.__class__.__name__ == "BCEWithLogitsLoss"
                        or criterion.__class__.__name__ == "BCELoss"
                    ):
                        _, targets = torch.max(targets, 1)

                    val_correct += (predicted_classes == targets).sum().item()

                val_loss /= len(valloader)
                val_accuracy = val_correct / val_total

                val_loss_hist.append(val_loss)
                val_acc_hist.append(val_accuracy)

                if val_loss <= min(val_loss_hist):
                    best_model = copy.deepcopy(model)

                print(
                    f" | Val Accuracy {val_accuracy:.2f} | Val Loss {val_loss:.2f}",
                    flush=True,
                    end="\n",
                )

    stats = {}
    stats["train_loss"] = train_loss_hist
    stats["train_acc"] = train_acc_hist

    if trainloader is not None:
        stats["val_loss"] = val_loss_hist
        stats["val_acc"] = val_acc_hist

    return stats, best_model


def eval_model(model, loader):
    model.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model.to(device)

    predictions = []
    labels = []

    with torch.no_grad():
        for _, (data, targets) in enumerate(loader):
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            predictions += predicted.tolist()
            labels += targets.tolist()

    return predictions, labels


def parse_pkl_metrics(file):
    with open(file, "rb") as f:
        metrics = pickle.load(f)

    train_loss = metrics["train_loss_history"]
    train_acc = metrics["train_acc_history"]
    val_loss = metrics["val_loss_history"]
    val_acc = metrics["val_acc_history"]

    return train_loss, train_acc, val_loss, val_acc


def plot_train_perf(
    model,
    path=None,
    train_loss=None,
    train_acc=None,
    val_loss=None,
    val_acc=None,
    save_path=None,
):
    if path is not None:
        train_loss, train_acc, val_loss, val_acc = parse_pkl_metrics(path)

    cmap = plt.cm.gnuplot
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label="training", color="cyan")
    plt.plot(val_loss, label="validation", color="magenta")
    plt.title(model.__class__.__name__)
    plt.suptitle("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(f"{save_path}/loss.png")

    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label="training", color="cyan")
    plt.plot(val_acc, label="validation", color="magenta")
    plt.title(model.__class__.__name__)
    plt.suptitle("Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(f"{save_path}/accuracy.png")
