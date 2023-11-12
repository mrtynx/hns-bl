import copy

import torch


def train_model(
    model, trainloader, num_epochs, criterion, optimizer, testloader=None, device=None
):
    if testloader is not None:
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

        if testloader is not None:
            val_loss = 0
            val_accuracy = 0
            val_correct = 0
            val_total = 0
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
