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
    # model_name = model.__class__.__name__
    # if model_name == "ResNet":
    #     model = unfreeze_resnet(model)
    # else:
    #     for param in model.parameters():
    #         param.requires_grad = True

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
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 25)

    return model


def inception_architecture():
    model = models.inception_v3(weights="DEFAULT")
    model.fc = torch.nn.Linear(model.fc.in_features, 25)

    return model


def mobilenet_architecture():
    model = models.mobilenet_v3_large(weights="DEFAULT")
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 25)

    return model


def resnet50_architecture():
    model = models.resnet50(weights="DEFAULT")
    model.fc = torch.nn.Linear(model.fc.in_features, 25)

    return model


def resnet18_architecture():
    model = models.resnet18(weights="DEFAULT")
    model.fc = torch.nn.Linear(model.fc.in_features, 25)

    return model


def resnet152_architecture():
    model = models.resnet152(weights="DEFAULT")
    model.fc = torch.nn.Linear(model.fc.in_features, 25)

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
    def __init__(self, num_classes=25):
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
            nn.MaxPool2d(kernel_size=4, stride=4),
        )

        self.conv2b = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
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
            nn.MaxPool2d(kernel_size=4, stride=4),
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


class DeepSerialCNN(torch.nn.Module):
    def __init__(self, num_classes=25):
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
            nn.AvgPool2d(kernel_size=2, stride=1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.AvgPool2d(kernel_size=2, stride=1),
            nn.MaxPool2d(kernel_size=4, stride=4),
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


class Fire(nn.Module):
    def __init__(
        self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels
    ):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(
            squeeze_channels, expand3x3_channels, kernel_size=3, padding=1
        )
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x)),
            ],
            1,
        )


# Define the Residual Block with Fire Modules
class HybridBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HybridBlock, self).__init__()

        # Main path has a Fire module
        self.fire = Fire(
            in_channels, in_channels // 2, out_channels // 2, out_channels // 2
        )

        # Residual path
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.fire(x)

        # Adding the shortcut to the output
        out += self.shortcut(x)

        return nn.ReLU()(out)


# Example Hybrid Network
class HybridNet(nn.Module):
    def __init__(self, num_classes=25):
        super(HybridNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.hybrid1 = HybridBlock(32, 64)
        self.hybrid2 = HybridBlock(64, 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.hybrid1(x)
        x = self.hybrid2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
