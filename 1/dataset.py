import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.io import loadmat
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset


class notMNIST(Dataset):
    def __init__(self, path):
        self.data = loadmat(path)
        self.images = self.data["images"]
        self.labels = self.data["labels"]

        self.transformation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype=torch.float32),
            ]
        )

        self.__format_images()

    def __format_images(self):
        self.images = [self.images[:, :, i] for i in range(0, self.images.shape[2])]
        self.images = np.asarray(self.images)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        img_tensor = self.transformation(img)
        y_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor.view(1, 28, 28), y_tensor
