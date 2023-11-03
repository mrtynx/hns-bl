import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_numbers import get_numbers_dataset
from models import LSTMModel
from pretty_confusion_matrix import pp_matrix_from_data

from metrics import test_model, generate_confusion_matrix

train_ds, test_ds = get_numbers_dataset("dataset.mat")
trainloader = DataLoader(train_ds, batch_size=32, shuffle=True)
testloader = DataLoader(test_ds, batch_size=32, shuffle=False)

input_size = 2
hidden_size = 128
num_layers = 1
output_size = 10

model = LSTMModel(input_size, hidden_size, num_layers, output_size)


# Step 3: Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Training loop
num_epochs = 100

for epoch in range(num_epochs):
    for batch_inputs, batch_labels in trainloader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Step 5: Evaluation (on a validation or test dataset)
# Replace this with your actual evaluation code
# Example: Evaluate on a validation dataset
model.eval()
# validation_dataloader = DataLoader(...)  # Load your validation dataset
# Perform evaluation and calculate accuracy
# Example: calculate accuracy
# correct = 0
# total = 0
# with torch.no_grad():
#     for batch_inputs, batch_labels in testloader:
#         outputs = model(batch_inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += batch_labels.size(0)
#         correct += (predicted == batch_labels).sum().item()

# accuracy = 100 * correct / total
# print(f"Validation Accuracy: {accuracy:.2f}%")

y_pred, labels = test_model(model, testloader)


pp_matrix_from_data(labels, y_pred, columns=[i for i in range(10)], cmap="gnuplot")

# generate_confusion_matrix(labels, y_pred)
