import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import utils


def test_model(model, testloader):
    model.eval()  # activate evaulation mode, some layers behave differently
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    total = 0
    correct = 0
    final_predicted = []
    final_labels = []
    for inputs, labels in iter(testloader):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            outputs_batch = model(inputs)

        _, predicted = torch.max(outputs_batch.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        final_predicted += predicted.tolist()
        final_labels += labels.tolist()
    print(
        "Accuracy of the network on the test images: %0.2f %%" % (100 * correct / total)
    )

    return final_predicted, final_labels


def plot_train_metrics(acc_history, loss_history):
    plt.plot(np.array(range(1, len(loss_history) + 1)), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(np.array(range(1, len(acc_history) + 1)), acc_history)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
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
