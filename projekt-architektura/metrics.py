import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from torchvision import utils
import torch
from dataset_architecture import get_classes_weights
from sklearn.metrics import confusion_matrix


def test_model(model, testloader):
    model.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    weights = torch.tensor(get_classes_weights()).to(device)

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    y_pred = []
    labels = []

    val_total = 0
    val_correct = 0
    val_loss = 0

    with torch.no_grad():
        for _, (data, target) in enumerate(testloader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()

            val_batch_loss = criterion(output, target)
            val_loss += val_batch_loss.item()

            y_pred += predicted.tolist()
            labels += target.tolist()

    val_accuracy = val_correct / val_total

    val_loss /= len(testloader)

    return y_pred, labels


def generate_confusion_matrix(y_test, y_pred):
    """Generates a confusion_matrix plot based on the given values.
    Args:
        y_test (any): the resulting y_test of the function "train_test_split".
        y_pred (any): the resulting value of the function "predict".
    Returns:
        _.
    """
    # logger = logging.getLogger('ThreatTrekker')
    # logger.debug('Plotting confusion matrix')

    cm = confusion_matrix(
        y_test,
        y_pred,
    )
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # plot the confusion matrix using seaborn
    sn.set_style("darkgrid")
    sn.set(rc={"figure.figsize": (20, 10)})  # Size in inches
    sn.heatmap(cm_norm, annot=True, cmap="viridis", fmt=".2f")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    # plt.savefig(PLOTS_PATH + 'Confusion Matrix')
    plt.show()


def parse_pkl_metrics(file):
    with open(file, "rb") as f:
        metrics = pickle.load(f)

    train_loss = metrics["train_loss_history"]
    train_acc = metrics["train_acc_history"]
    val_loss = metrics["val_loss_history"]
    val_acc = metrics["val_acc_history"]

    return train_loss, train_acc, val_loss, val_acc


def plot_train_perf(path, model_name):
    train_loss, train_acc, val_loss, val_acc = parse_pkl_metrics(path)
    cmap = plt.cm.gnuplot
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label="training", color="cyan")
    plt.plot(val_loss, label="validation", color="magenta")
    plt.title(model_name)
    plt.suptitle("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label="training", color="cyan")
    plt.plot(val_acc, label="validation", color="magenta")
    plt.title(model_name)
    plt.suptitle("Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    # plt.show()
