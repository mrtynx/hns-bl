import torch
import torch.nn.functional as F
from torch import nn


class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        use_specgram=False,
        n_channels=64,
    ):
        super(CNNLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_specgram = use_specgram

        # Define the convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=n_channels,
            kernel_size=3 if self.use_specgram else n_channels,
            stride=1 if self.use_specgram else 16,
        )
        self.bn1 = nn.BatchNorm1d(num_features=n_channels)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(4)

        if not self.use_specgram:
            self.conv2 = nn.Conv1d(n_channels, n_channels, kernel_size=3)
            self.bn2 = nn.BatchNorm1d(n_channels)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool1d(4)

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            n_channels,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.25,
        )

        # Define the fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        if not self.use_specgram:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.pool2(x)

        x = x.permute(0, 2, 1)

        # Initialize LSTM hidden state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through LSTM
        out, (h0, c0) = self.lstm(x, (h0, c0))

        # Get the last time step's output
        out = self.fc(out[:, -1, :])

        out = self.sigmoid(out)

        return out


class M5(nn.Module):
    def __init__(self, n_output=4, stride=16, n_channel=32, use_specgram=False):
        super().__init__()
        self.stride = stride
        self.use_specgram = use_specgram

        self.conv1 = nn.Conv1d(
            in_channels=128 if self.use_specgram else 1,
            out_channels=n_channel,
            kernel_size=3 if self.use_specgram else 80,
            stride=1 if self.use_specgram else self.stride,
        )
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)

        if not self.use_specgram:
            self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
            self.bn3 = nn.BatchNorm1d(2 * n_channel)
            self.pool3 = nn.MaxPool1d(4)
            self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
            self.bn4 = nn.BatchNorm1d(2 * n_channel)
            self.pool4 = nn.MaxPool1d(4)

        self.fc1 = nn.Linear((4 if self.use_specgram else 6) * n_channel, n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        if not self.use_specgram:
            x = self.conv3(x)
            x = F.relu(self.bn3(x))
            x = self.pool3(x)
            x = self.conv4(x)
            x = F.relu(self.bn4(x))
            x = self.pool4(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        out = self.sigmoid(x)

        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
