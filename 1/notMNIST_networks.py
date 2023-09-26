#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,10),
            nn.Softmax()
        )

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.fc_layers(out)
        return out


# # CNN Siet 1

# In[3]:


#2conv, 3conv, a jedna so zmenou poctu/velkosti filtrov
#resnet nepouzivat
#povedal kajan
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
            nn.Softmax()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(x.shape[0], -1)
        out = self.fc_layers(out)
        return out

