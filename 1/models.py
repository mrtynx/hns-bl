import torch.nn as nn
import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(),
        )

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.fc_layers(out)
        return out


class CustomCNN(torch.nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(320, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 10),
            nn.Softmax(),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(x.shape[0], -1)
        out = self.fc_layers(out)
        return out


class CustomCNN2(torch.nn.Module):
    def __init__(self):
        super(CustomCNN2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=3),  # 20 kernels of 3x3
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(3380, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(50, 10),
            # nn.Softmax()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = torch.flatten(out, 1)
        out = self.fc_layers(out)
        out = F.log_softmax(out, dim=1)
        return out


class CustomCNN3(torch.nn.Module):
    def __init__(self):
        super(CustomCNN3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(980, 340),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(340, 100),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(x.shape[0], -1)
        out = self.fc_layers(out)
        out = F.log_softmax(out, dim=1)
        return out
