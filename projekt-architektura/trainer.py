import argparse
import datetime
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from torchvision import models

import dataset_architecture as da 

from models import resnet


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    choices=["resnet"],
    default="resnet",
    help="Choose a model",
)

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)

parser.add_argument("--save_model", type=bool, default=True)


def main():
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    LEARNING_RATE = args.lr

 

    # models selection
    if args.model == "resnet":
        model = models.resnet18()
       # model.load_state_dict(torch.load("models/resnet.pt"))
        savename = "resnet"

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    output_folder_str = "trained_models"

    if not os.path.exists(output_folder_str):
        os.makedirs(output_folder_str)

    # getting dataset
    
    trainloader, _ = da.get_architectural_dataset("C:\\Repositories\\hns\\Dataset_Architektura\\architectural-styles-dataset",
                                                    resnet.get_resnet_transformation(),
                                                    BATCH_SIZE,
                                                    0.4,0)
   # trainloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # training itself
    criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()

    acc_history = []
    loss_history = []
    final_labels = []
    final_predicted = []

    for epoch in range(1, N_EPOCHS + 1):
        epoch_loss = 0
        #n_batches = len(train_dataset) // BATCH_SIZE
        n_batches = trainloader.n_batches      
        correct = 0
        total = 0
        accuracy_train = 0

        for step, (images, labels) in enumerate(
            trainloader, desc="Epoch {}/{}".format(epoch, N_EPOCHS)
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

                acc_history.append(accuracy_train)
                loss_history.append(epoch_loss)
                print(
                    "Epoch {}, Loss {:.6f}, Accuracy {:.2f}% ".format(
                        epoch, epoch_loss, accuracy_train * 100
                    )
                )
                epoch_loss = 0

            final_predicted += predicted.tolist()
            final_labels += labels.tolist()
            torch.cuda.empty_cache()

    # save_model
    torch.save(model.state_dict(), f"{output_folder_str}/{savename}/model.pt")

    train_metrics = {"acc_history": acc_history, "loss_history": loss_history}

    with open(f"{output_folder_str}/{savename}/train_metrics.pkl", "wb") as f:
        pickle.dump(train_metrics, f)


if __name__ == "__main__":
    main()
