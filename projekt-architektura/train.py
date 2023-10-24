import argparse
import os
import pickle

import torch
from models_architecture import *
from dataset_architecture import get_architectural_dataset
from torchvision import transforms

TRAINED_MODELS_DIR = "trained_models"
DATASET_PTH = "architectural-styles-dataset/"


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    choices=["resnet", "alexnet", "inception", "mobilenet"],
    default="resnet",
    help="Choose a model",
)

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--save_model", choices=["0", "1"], default="1")
parser.add_argument("--test_split", type=float, default=0.2)
# parser.add_argument("--val_split", type=float, default=0)
parser.add_argument(
    "--unfreeze",
    type=int,
    default=-0,
    help="Unfreeze feature layers after N batches. (N must be lower than total batches)",
)


def main():
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    TEST_SPLIT = args.test_split
    UNFREEZE_EPOCH = args.unfreeze
    SAVE_MODEL = int(args.save_model)

    # models selection
    if args.model == "resnet":
        model = resnet18_architecture()
    elif args.model == "alexnet":
        model = alexnet_architecture()
    elif args.model == "inception":
        model = inception_architecture()
    elif args.model == "mobilenet":
        model = mobilenet_architecture()

    MODEL_NAME = model.__class__.__name__

    if UNFREEZE_EPOCH != 0:
        model = freeze_feature_layers(model)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if not os.path.exists(TRAINED_MODELS_DIR):
        os.makedirs(TRAINED_MODELS_DIR)

    transform = transforms.Compose(
        [
            transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(dtype=torch.float32),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    trainloader, testloader = get_architectural_dataset(
        root_path="architectural-styles-dataset/",
        transform=transform,
        batch_sz=BATCH_SIZE,
        test=TEST_SPLIT,
    )

    # training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model = model.to(device)
    model.train()

    acc_history = []
    loss_history = []
    final_labels = []
    final_predicted = []

    for epoch in range(1, N_EPOCHS + 1):
        epoch_loss = 0
        n_batches = len(trainloader)
        correct = 0
        total = 0
        accuracy_train = 0

        if UNFREEZE_EPOCH != 0 and epoch == UNFREEZE_EPOCH + 1:
            model = unfreeze_layers(model)
            print(f"\nUnfreezing {MODEL_NAME} feature layers \n")

        for step, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            # Uspesnost algoritmu
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy_train = correct / total
            epoch_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            print(
                f"\rEpoch: {epoch}/{N_EPOCHS} | Batch {step} / {n_batches} | Accuracy {accuracy_train:.2f}",
                end="",
                flush=True,
            )

            if step % n_batches == 0 and step != 0:
                epoch_loss = epoch_loss / n_batches

                acc_history.append(accuracy_train)
                loss_history.append(epoch_loss)
                epoch_loss = 0

            final_predicted += predicted.tolist()
            final_labels += labels.tolist()
            torch.cuda.empty_cache()

    # save_model
    if SAVE_MODEL:
        if not os.path.exists(f"{TRAINED_MODELS_DIR}/{MODEL_NAME}"):
            os.makedirs(f"{TRAINED_MODELS_DIR}/{MODEL_NAME}")

        torch.save(model.state_dict(), f"{TRAINED_MODELS_DIR}/{MODEL_NAME}/model.pt")

        train_metrics = {"acc_history": acc_history, "loss_history": loss_history}

        with open(f"{TRAINED_MODELS_DIR}/{MODEL_NAME}/train_metrics.pkl", "wb") as f:
            pickle.dump(train_metrics, f)


if __name__ == "__main__":
    main()
