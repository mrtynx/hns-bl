from torchvision import models


model = models.resnet18(weights="DEFAULT")
model.fc.out_features = 25

freeze_first_n_layers = 5
count = 0
for param in model.children():
    if (
        count < freeze_first_n_layers and len(list(param.parameters())) > 0
    ):  # freezing first 3 layers
        print(param)
        param.requires_grad_(False)
        count += 1


for name, param in model.named_parameters():
    print(f"Layer: {name}, Requires Grad: {param.requires_grad}")


def resnet(num_layers=18, weights="DEFAULT", freeze_n_layers=4, out_features=25):
    model = models.resnet18


print(model.layer4)
