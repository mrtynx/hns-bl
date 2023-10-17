from torchvision import models
import torchvision.transforms as transforms


def resnet(num_layers=18, weights="DEFAULT", freeze_n_layers=4, out_features=25):
    model = models.resnet18(weights=weights)
    model.fc.out_features = out_features
    count = 0
    for param in model.children():
        if count < freeze_n_layers and len(list(param.parameters())) > 0:
            param.requires_grad_(False)
            count += 1
    return model

def get_resnet_transformation():
    preprocess = transforms.Compose([
    transforms.Resize(256),
    #transforms.CenterCrop(224), #unsure whether this is required
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess
