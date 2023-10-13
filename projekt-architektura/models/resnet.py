from torchvision import models


def resnet(num_layers=18, weights="DEFAULT", freeze_n_layers=4, out_features=25):
    model = models.resnet18(weights=weights)
    model.fc.out_features = out_features
    count = 0
    for param in model.children():
        if count < freeze_n_layers and len(list(param.parameters())) > 0:
            param.requires_grad_(False)
            count += 1


print("kkk")
