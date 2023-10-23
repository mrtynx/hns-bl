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


def unfreeze_layers(model):
    for param in model.parameters():
        param.requires_grad = True

    return model


def freeze_feature_layers(model):
    if model.__class__.__name__ == "AlexNet":
        model = _alexnet_freeze_features(model)

    if model.__class__.__name__ == "ResNet":
        model = _resnet18_freeze_features(model)

    return model


def resnet18_architecture():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc.out_features = 25

    return model


def alexnet_architecture():
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    model.classifier[6].out_features = 25

    return model
