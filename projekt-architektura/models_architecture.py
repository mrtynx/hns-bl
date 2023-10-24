import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False

    return model


def _resnet18_freeze_features(model):
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
    model = _freeze_all_layers(models)
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def unfreeze_layers(model):
    for param in model.parameters():
        param.requires_grad = True

    return model


def freeze_feature_layers(model):
    model_name = model.__class__.__name__

    if model_name == "AlexNet":
        model = _alexnet_freeze_features(model)

    if model_name == "ResNet":
        model = _resnet18_freeze_features(model)

    if model_name == "Inception3":
        model = _inception_freeze_features(model)

    if model_name == "MobileNetV3":
        model = _mobilenet_freeze_features(model)

    return model


def resnet18_architecture():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc.out_features = 25

    return model


def alexnet_architecture():
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    model.classifier[6].out_features = 25

    return model


def inception_architecture():
    model = models.inception_v3(weights="DEFAULT")
    model.fc.out_features = 25

    return model


def mobilenet_architecture():
    model = models.mobilenet_v3_large(weights="DEFAULT")
    model.classifier[3].out_features = 25

    return model
