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
