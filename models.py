from torch import nn
import torch


# Define the first MLP model
class MLPModel_small(nn.Module):
    def __init__(self, in_features):
        super(MLPModel_small, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Define the first MLP model
class MLPModel_thin(nn.Module):
    def __init__(self, in_features):
        super(MLPModel_thin, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class MLPModel_simple(nn.Module):
    def __init__(self, in_features):
        super(MLPModel_simple, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 120),
            nn.ReLU(),
            nn.Linear(120, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Define the second MLP model
class MLPModel2_large(nn.Module):
    def __init__(self, in_features):
        super(MLPModel2_large, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class Conv1DModel(nn.Module):
    def __init__(self, seq_len):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * (seq_len // 8), 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layers
        x = self.fc(x)
        return x


class PureCNN(nn.Module):
    def __init__(self, input_features=4116):
        super(PureCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Pooling layer (shared across layers)
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8)  # Reduces dimensionality

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

        # Calculate the flattened size after convolution and pooling
        reduced_size = input_features // (8 * 8 * 8)  # Three pooling layers reduce by 4x each time

        # Fully connected layers
        self.fc1 = nn.Linear(64,16)  # 64 channels * reduced length
        self.fc2 = nn.Linear(16, 1)  # Final output layer

        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch_size, 1, input_features)
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = self.pool(torch.relu(self.conv3(x)))  # Conv3 -> ReLU -> Pool
        x = self.pool(torch.relu(self.conv4(x)))  # Conv3 -> ReLU -> Pool

        # Flatten for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))  # FC1 -> ReLU
        x = self.dropout(x)  # Dropout layer
        x = self.sigmoid(self.fc2(x))  # FC2 -> Sigmoid for binary classification

        return x

class smallCNN(nn.Module):
    def __init__(self, seq_len):
        super(smallCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding="same"),
            nn.BatchNorm1d(8),
            nn.ReLU(),

        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=7, padding="same"),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=7, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),

        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=7, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv4(x)
        x = x + self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layers
        x = self.fc(x)
        return x