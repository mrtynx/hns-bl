import argparse
import datetime
import pickle

import numpy as np
import torch
from dataset import notMNIST
from models import *
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm, trange

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_path", type=str, default="notMNIST_small.mat", help="Path to dataset"
)
parser.add_argument(
    "--model",
    choices=["cnn1", "cnn2", "cnn3", "mlp"],
    default="mlp",
    help="Choose a model",
)
parser.add_argument(
    "--split",
    type=float,
    default=0.8,
    help="Fraction of train data [default 0.8]",
)
parser.add_argument("--optim", choices=["sgd, adam"], default="adam")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--save_model", type=bool, default=True)
parser.add_argument("--use_cuda", type=bool, default=True)


def main():
    args = parser.parse_args()

    DATA_PATH = args.data_path
    SPLIT = args.split
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    LOG_FOLDER = "runs"

    time = datetime.datetime.now()
    time_str = time.strftime("%d-%m-%H-%M")

    if args.model == "cnn1":
        model = CustomCNN()
        savedir = f"{LOG_FOLDER}/cnn1/{time_str}"
    elif args.model == "cnn2":
        model = CustomCNN2()
        savedir = f"{LOG_FOLDER}/cnn2/{time_str}"
    elif args.model == "cnn3":
        model = CustomCNN3()
        savedir = f"{LOG_FOLDER}/cnn3/{time_str}"
    else:
        model = MLP()
        savedir = f"{LOG_FOLDER}/mlp/{time_str}"

    writer = SummaryWriter(savedir + "/writer")

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    dataset = notMNIST(DATA_PATH)
    split_idx = int(len(dataset) * SPLIT)

    train_ds = Subset(dataset, np.arange(0, split_idx))
    test_ds = Subset(dataset, np.arange(split_idx, len(dataset)))

    print(len(train_ds))
    trainloader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()

    acc_history = []
    loss_history = []
    final_labels = []
    final_predicted = []

    for epoch in trange(1, N_EPOCHS + 1, desc="1st loop"):
        epoch_loss = 0
        n_batches = len(train_ds) // BATCH_SIZE
        correct = 0
        total = 0
        accuracy_train = 0

        for step, (images, labels) in enumerate(
            tqdm(trainloader, desc="Epoch {}/{}".format(epoch, N_EPOCHS))
        ):
            images = images.to(device)
            labels = labels.to(device)

            # Dopredne sirenie,
            # ziskame pravdepodobnosti tried tym, ze posleme do modelu vstupy
            outputs = model(images)

            # Vypocitame chybu algoritmu
            loss = criterion(outputs, labels)

            # Uspesnost algoritmu
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy_train = correct / total
            epoch_loss += loss.item()

            # Je vhodne zavolat zero_grad() pred zavolanim spatneho sirenia
            # pre vynulovanie gradientov z predosleho volania loss.backward()
            optimizer.zero_grad()

            # Spatne sirenie chyby, vypocitaju sa gradienty
            loss.backward()

            # Aktualizacia vah pomocou optimalizatora
            optimizer.step()

            if step % n_batches == 0 and step != 0:
                epoch_loss = epoch_loss / n_batches

                # writer.add_scalar(
                #     'training loss',
                #     epoch_loss,
                #     epoch
                # )

                acc_history.append(accuracy_train)
                loss_history.append(epoch_loss)
                print(
                    "Epoch {}, Loss {:.6f}, Accuracy {:.2f}% ".format(
                        epoch, epoch_loss, accuracy_train * 100
                    )
                )
                epoch_loss = 0

                # print(model.layer1[0].conv1.weight[0][0])
                # print(model.layer2[0].conv1.weight[0][0])
                # print(model.layer3[0].conv1.weight[0][0])

            final_predicted += predicted.tolist()
            final_labels += labels.tolist()
            torch.cuda.empty_cache()

            writer.add_hparams(
                {
                    "optimizer": optimizer.__class__.__name__,
                    "lr": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                },
                {
                    "hparam/train/accuracy": accuracy_train,
                },
            )
            writer.close()

    torch.save(model.state_dict(), f"{savedir}/model.pt")


if __name__ == "__main__":
    main()
