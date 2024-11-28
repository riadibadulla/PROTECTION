from torch import nn

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