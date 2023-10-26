import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False

    return model


def _resnet_freeze_features(model):
    model = _freeze_all_layers(model)

    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

    return model


def _alexnet_freeze_features(model):
    for param in model.features.parameters():
        param.requires_grad = False

    return model


def _inception_freeze_features(model):
    model = _freeze_all_layers(model)
    model.fc.weight.requires_grad = True

    return model


def _mobilenet_freeze_features(model):
    model = _freeze_all_layers(model)
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def unfreeze_layers(model):
    if model.__class__.__name__ == "ResNet":
        return unfreeze_resnet(model)
    else:
        for param in model.parameters():
            param.requires_grad = True

    return model


def freeze_feature_layers(model):
    model_name = model.__class__.__name__

    if model_name == "AlexNet":
        model = _alexnet_freeze_features(model)

    if model_name == "ResNet":
        model = _resnet_freeze_features(model)

    if model_name == "Inception3":
        model = _inception_freeze_features(model)

    if model_name == "MobileNetV3":
        model = _mobilenet_freeze_features(model)

    return model


def alexnet_architecture():
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    model.classifier[6].out_features = 25

    return model


def inception_architecture():
    model = models.inception_v3(weights="DEFAULT")
    # model.Conv2d_1a_3x3.conv.padding = (1, 1)
    model.fc.out_features = 25

    return model


def mobilenet_architecture():
    model = models.mobilenet_v3_large(weights="DEFAULT")
    model.classifier[3].out_features = 25

    return model


def resnet50_architecture():
    model = models.resnet50(weights="DEFAULT")
    model.fc.out_features = 25

    return model


def resnet18_architecture():
    model = models.resnet18(weights="DEFAULT")
    model.fc.out_features = 25

    return model


def resnet152_architecture():
    model = models.resnet152(weights="DEFAULT")
    model.fc.out_features = 25

    return model


def resnet_architecture(variant: str):
    if variant == "resnet18":
        return resnet18_architecture()
    if variant == "resnet50":
        return resnet50_architecture()
    if variant == "resnet152":
        return resnet152_architecture()


def unfreeze_resnet(model):
    for param in model.layer4.parameters():
        param.requires_grad = True

    return model


class DeepSeriesParallelCNN(torch.nn.Module):
    def __init__(self,num_classes=25):
        super(DeepSeriesParallelCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
 
        self.conv2a = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        self.conv2b = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        self.conv2c = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=7),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 6, kernel_size=7),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 6, kernel_size=7),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(11604, 600),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(300, 60),
            nn.ReLU(),
            nn.Linear(60, num_classes),
        )

    def forward(self, x):
        outM = self.conv1(x)
        outA = self.conv2a(outM)
        outB = self.conv2b(outM)
        outC = self.conv2c(outM)
        outA = torch.flatten(outA, 1)
        outB = torch.flatten(outB, 1)
        outC = torch.flatten(outC, 1)
        out = torch.cat([outA, outB, outC], dim=1)
        out = self.fc_layers(out)
        out = F.log_softmax(out, dim=1)
        return out   
    

class  DeepSerialCNN(torch.nn.Module):
    def __init__(self,num_classes=25):
        super(DeepSerialCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.AvgPool2d(kernel_size=2, stride=1),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(17496, 600),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = torch.flatten(out, 1)
        out = self.fc_layers(out)
        out = F.log_softmax(out, dim=1)
        return out    

