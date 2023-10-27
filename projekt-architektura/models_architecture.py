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
