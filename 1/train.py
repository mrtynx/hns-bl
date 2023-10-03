import argparse
import datetime
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm, trange

# from dataset import notMNIST
from models import *

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_path", type=str, default="notMNIST_small.mat", help="Path to dataset"
)
parser.add_argument(
    "--model",
    choices=["cnn1", "cnn2", "cnn3", "mlp", "cnn1_no_drop"],
    default="mlp",
    help="Choose a model",
)
parser.add_argument(
    "--split",
    type=float,
    default=0.8,
    help="Fraction of train data [default 0.8]",
)
parser.add_argument("--optim", choices=["sgd", "adam"], default="adam")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--save_model", type=bool, default=True)
parser.add_argument("--use_cuda", type=bool, default=True)
parser.add_argument("--use_dropout", choices=["0", "1"], default="1")


def main():
    args = parser.parse_args()

    DATA_PATH = args.data_path
    SPLIT = args.split
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    LOG_FOLDER = "runs"
    USE_DROPOUT = int(args.use_dropout)

    time = datetime.datetime.now()
    # time_str = time.strftime("%d-%m-%H-%M")
    time_str = ""

    if args.model == "cnn1":
        model = CustomCNN()
        savedir = "cnn1"
    elif args.model == "cnn2":
        model = CustomCNN2()
        savedir = "cnn2"
    elif args.model == "cnn3":
        model = CustomCNN3()
        savedir = "cnn3"
    elif args.model == "cnn1_no_drop":
        model = CustomCNN_NO_DROPOUT()
        savedir = "cnn1"
    else:
        model = MLP()
        savedir = "mlp"

    if args.optim == "adam":
        LOG_FOLDER += f"/{savedir}"
    else:
        LOG_FOLDER += f"/{savedir}_sgd"

    if not USE_DROPOUT:
        LOG_FOLDER += f"_no_dropout"

    LOG_FOLDER += f"_{str(SPLIT)}"

    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)
    # writer = SummaryWriter(savedir + "/writer")

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # %%
    from scipy.io import loadmat
    from sklearn.model_selection import train_test_split

    from dataset import notMNIST

    data = loadmat("notMNIST_small.mat")
    print(data.keys())

    images = data["images"]
    labels = data["labels"]

    print(images.shape)
    images = [images[:, :, i] for i in range(0, images.shape[2])]
    images = np.asarray(images)
    print(images.shape)
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, train_size=SPLIT, shuffle=True
    )

    train_dataset = notMNIST(x_train, y_train)
    test_dataset = notMNIST(x_test, y_test)

    trainloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # %%

    criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()

    acc_history = []
    loss_history = []
    final_labels = []
    final_predicted = []

    for epoch in trange(1, N_EPOCHS + 1, desc="1st loop"):
        epoch_loss = 0
        n_batches = len(train_dataset) // BATCH_SIZE
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

                # writer.add_scalar("training loss", epoch_loss, epoch)

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

            # writer.add_hparams(
            #     {
            #         "optimizer": optimizer.__class__.__name__,
            #         "lr": LEARNING_RATE,
            #         "batch_size": BATCH_SIZE,
            #     },
            #     {
            #         "hparam/train/accuracy": accuracy_train,
            #     },
            # )
            # writer.close()

    torch.save(model.state_dict(), f"{LOG_FOLDER}/model.pt")

    train_metrics = {"acc_history": acc_history, "loss_history": loss_history}
    with open(f"{LOG_FOLDER}/train_metrics.pkl", "wb") as f:
        pickle.dump(train_metrics, f)


if __name__ == "__main__":
    main()
