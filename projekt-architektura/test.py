import argparse

import torch
from dataset_architecture import get_classes_weights, get_random_split_arch_dataset
from models_architecture import *
from torch.utils.data import DataLoader
from torchvision import transforms

TRAINED_MODELS_DIR = "trained_models"
DATASET_PTH = "architectural-styles-dataset/"
RESIZE = (512, 512)
CROP = (448, 448)


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
    # args = args.parse_args()

    transform = transforms.Compose(
        [
            transforms.Resize(RESIZE, transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(CROP),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(dtype=torch.float32),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    _, test_ds = get_random_split_arch_dataset(
        "architectural-styles-dataset/",
        transform=transform,
        split=1 - 0.2,
        seed=420,
    )
    testloader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = inception_architecture()
    model.load_state_dict(torch.load("VAST/Inception3/4/best_loss_model.pt"))
    model.to(device)

    weights = torch.tensor(get_classes_weights()).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    model.eval()
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

    val_accuracy = val_correct / val_total

    val_loss /= len(testloader)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
