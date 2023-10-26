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
    choices=["resnet", "alexnet", "inception", "mobilenet", "series-parallel"],
    default="resnet",
    help="Choose a model",
)

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--save_model", choices=["0", "1"], default="1")
parser.add_argument("--test_split", type=float, default=0.2)
parser.add_argument("--trial_number", choices=["1", "2", "3"], default="1")
parser.add_argument("--augment", choices=["0", "1"], default="0")
parser.add_argument("--compile", choices=["0", "1"], default="0")
parser.add_argument(
    "--unfreeze",
    type=int,
    default=0,
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
    TRIAL_NUM = args.trial_number
    AUGMENT = int(args.augment)
    COMPILE = int(args.compile)

    # models selection
    if (
        args.model == "resnet18"
        or args.model == "resnet50"
        or args.model == "resnet152"
    ):
        model = resnet_architecture(args.model)
    elif args.model == "alexnet":
        model = alexnet_architecture()
    elif args.model == "inception":
        model = inception_architecture()
    elif args.model == "mobilenet":
        model = mobilenet_architecture()
    # elif args.model == "series-parallel":
    #     model = DeepSeriesParallelCNN(num_classes=25)

    if COMPILE:
        model = torch.compile(model, mode="reduce-overhead")
        torch.set_float32_matmul_precision("high")

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

    if AUGMENT:
        augmentations = transforms.Compose(
            [
                transforms.v2.RandomPosterize(bits=2),
                transforms.v2.RandomAdjustSharpness(sharpness_factor=2),
                transforms.v2.RandomAutocontrast(),
                transforms.v2.RandomEqualize(),
            ]
        )

        transform = transforms.Compose([transform, augmentations])

    trainloader, testloader = get_architectural_dataset(
        root_path="architectural-styles-dataset/",
        # root_path="C:\\Repositories\\hns\\Dataset_Architektura\\architectural-styles-dataset",
        transform=transform,
        batch_sz=BATCH_SIZE,
        test=TEST_SPLIT,
    )

    # training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model = model.to(device)

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

        model.train()

        if UNFREEZE_EPOCH != 0 and epoch == UNFREEZE_EPOCH + 1:
            model = unfreeze_layers(model)
            print(f"\n\rUnfreezing {MODEL_NAME} feature layers \n", end="", flush=True)

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
                f"\rEpoch: {epoch}/{N_EPOCHS} | Batch {step + 1} / {n_batches} | Train Accuracy {accuracy_train:.2f}",
                end="",
                flush=True,
            )

        epoch_loss = epoch_loss / n_batches

        acc_history.append(accuracy_train)
        loss_history.append(epoch_loss)
        epoch_loss = 0

        final_predicted += predicted.tolist()
        final_labels += labels.tolist()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(testloader):
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

                val_accuracy = val_correct / val_total
        print(
            f" | Val Accuracy {val_accuracy:.2f}",
        )

        torch.cuda.empty_cache()

    # save_model
    if SAVE_MODEL:
        SAVEDIR = f"{TRAINED_MODELS_DIR}/{MODEL_NAME}/{TRIAL_NUM}"
        if not os.path.exists(SAVEDIR):
            os.makedirs(SAVEDIR)

        torch.save(model.state_dict(), f"{SAVEDIR}/model.pt")

        train_metrics = {"acc_history": acc_history, "loss_history": loss_history}

        with open(f"{SAVEDIR}/train_metrics.pkl", "wb") as f:
            pickle.dump(train_metrics, f)


if __name__ == "__main__":
    main()
