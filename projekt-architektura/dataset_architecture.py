from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize images to a consistent size
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ),  # Normalize pixel values
    ]
)


dataset = ImageFolder(root="architectural-styles-dataset/", transform=transform)

batch_size = 3  # Adjust as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
