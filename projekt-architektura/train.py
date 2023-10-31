import argparse
import os
import pickle
import copy

import torch
from dataset_architecture import (
    get_architectural_dataset,
    get_random_split_arch_dataset,
    get_classes_weights,
)
from early_stopping import EarlyStopping
from models_architecture import *
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2

TRAINED_MODELS_DIR = "trained_models"
DATASET_PTH = "architectural-styles-dataset/"


parser = argparse.ArgumentParser()

parser.add_argument(
    "-model",
    choices=[
        "resnet18",
        "resnet50",
        "resnet152",
        "alexnet",
        "inception",
        "mobilenet",
        "series-parallel",
        "serial",
    ],
    default="resnet18",
    help="Choose a model",
)

parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-lr", type=float, default=0.00001)
parser.add_argument("-lr_factor", type=int, default=1)
parser.add_argument("-epochs", type=int, default=10)
parser.add_argument("-no_save", action="store_false")
parser.add_argument("-test_split", type=float, default=0.2)
parser.add_argument("-trial_num", choices=["1", "2", "3"], default="1")
parser.add_argument("-augment", action="store_true")
parser.add_argument("-compile", action="store_true")
parser.add_argument("-patience", type=int, default=0)
parser.add_argument("-uniform", action="store_true")
parser.add_argument(
    "-unfreeze",
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
    SAVE_MODEL = args.no_save
    TRIAL_NUM = args.trial_num
    AUGMENT = args.augment
    COMPILE = args.compile
    PATIENCE = args.patience
    UNIFORM = args.uniform
    LR_FACTOR = args.lr_factor

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
    elif args.model == "series-parallel":
        model = SerialParallelCNN()
    elif args.model == "serial":
        model = DeepSerialCNN(num_classes=25)

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

    if MODEL_NAME == "Inception3":
        RESIZE = (299, 299)
        CROP = (299, 299)
    else:
        RESIZE = (256, 256)
        CROP = (224, 224)

    transform = transforms.Compose(
        [
            transforms.Resize(RESIZE, transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(CROP),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(dtype=torch.float32),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    if AUGMENT:
        augmentations = transforms.Compose(
            [
                v2.RandomPosterize(bits=2),
                v2.RandomAdjustSharpness(sharpness_factor=2),
                v2.RandomAutocontrast(),
                v2.RandomEqualize(),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
            ]
        )

        transform = transforms.Compose([transform, augmentations])

    if UNIFORM:
        trainloader, testloader = get_architectural_dataset(
            root_path="architectural-styles-dataset/",
            # root_path="C:\\Repositories\\hns\\Dataset_Architektura\\architectural-styles-dataset",
            transform=transform,
            batch_sz=BATCH_SIZE,
            test=TEST_SPLIT,
        )
    else:
        train_ds, test_ds = get_random_split_arch_dataset(
            "architectural-styles-dataset/",
            transform=transform,
            split=1 - TEST_SPLIT,
            seed=420,
        )
        trainloader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        testloader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    # training
    weights = torch.tensor(get_classes_weights()).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    early_stopping = EarlyStopping(patience=PATIENCE, delta=0.05)

    model = model.to(device)

    train_acc_history = []
    val_acc_history = []
    loss_history = []
    final_labels = []
    final_predicted = []
    val_loss_history = []

    for epoch in range(1, N_EPOCHS + 1):
        if epoch // 5 == 0:
            LEARNING_RATE /= LR_FACTOR
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
            print(f"\rBatch {step+1}/{n_batches}", end="", flush=True)

            images = images.to(device)
            labels = labels.to(device)

            # Uspesnost algoritmu
            if MODEL_NAME == "Inception3":
                outputs, _ = model(images)
                predicted = torch.argmax(outputs, dim=1)
            else:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            epoch_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        accuracy_train = correct / total
        epoch_loss = epoch_loss / n_batches

        print(
            f"\rEpoch: {epoch}/{N_EPOCHS} | Train Accuracy {accuracy_train:.2f} | Train Loss {epoch_loss:.2f}",
            end="",
            flush=True,
        )

        train_acc_history.append(accuracy_train)
        loss_history.append(epoch_loss)

        final_predicted += predicted.tolist()
        final_labels += labels.tolist()

        val_correct = 0
        val_total = 0
        val_loss = 0
        n_batches = len(testloader)

        model.eval()
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

        val_accuracy = val_correct / val_total
        val_acc_history.append(val_accuracy)

        val_loss /= len(testloader)
        val_loss_history.append(val_loss)

        if val_loss <= min(val_loss_history):
            best_loss_model = copy.deepcopy(model)

        print(
            f" | Val Accuracy {val_accuracy:.2f} | Val Loss {val_loss:.2f} | LR {LEARNING_RATE:.10f}",
        )

        if PATIENCE > 0:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping training")
                break

        torch.cuda.empty_cache()

    # save_model
    if SAVE_MODEL:
        SAVEDIR = f"{TRAINED_MODELS_DIR}/{MODEL_NAME}/{TRIAL_NUM}"
        if not os.path.exists(SAVEDIR):
            os.makedirs(SAVEDIR)

        torch.save(best_loss_model.state_dict(), f"{SAVEDIR}/best_loss_model.pt")

        train_metrics = {
            "train_acc_history": train_acc_history,
            "train_loss_history": loss_history,
            "val_acc_history": val_acc_history,
            "val_loss_history": val_loss_history,
        }

        with open(f"{SAVEDIR}/train_metrics.pkl", "wb") as f:
            pickle.dump(train_metrics, f)


if __name__ == "__main__":
    main()
