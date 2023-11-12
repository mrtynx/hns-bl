import pickle

import matplotlib.pyplot as plt
import torch


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
    model, path=None, train_loss=None, train_acc=None, val_loss=None, val_acc=None
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

    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label="training", color="cyan")
    plt.plot(val_acc, label="validation", color="magenta")
    plt.title(model.__class__.__name__)
    plt.suptitle("Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True)
