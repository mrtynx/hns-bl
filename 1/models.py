#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
from sklearn import metrics
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, datasets
import torchvision.transforms as transforms
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt


# # MLP Siet

# In[12]:


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        out = x.view(x.shape[0], -1)  # equivalent to flatten
        out = self.fc_layers(out)
        out = F.log_softmax(out, dim=1)
        return out


# In[13]:


# 2conv, 3conv, a jedna so zmenou poctu/velkosti filtrov


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
            nn.Dropout(0.2),
            nn.Linear(320, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(60, 10),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(x.shape[0], -1)
        out = self.fc_layers(out)
        out = F.log_softmax(out, dim=1)
        return out


# # CNN Siet 2

# In[14]:


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


# # Custom CNN 3

# In[15]:


class CustomCNN3(torch.nn.Module):
    def __init__(self):
        super(CustomCNN3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(10),
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
            nn.Dropout(0.5),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.view(x.shape[0], -1)
        out = self.fc_layers(out)
        out = F.log_softmax(out, dim=1)
        return out


## CNN1 no dropout


class CustomCNN_NO_DROPOUT(torch.nn.Module):
    def __init__(self):
        super(CustomCNN_NO_DROPOUT, self).__init__()
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
            nn.Linear(320, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 10),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(x.shape[0], -1)
        out = self.fc_layers(out)
        out = F.log_softmax(out, dim=1)
        return out
