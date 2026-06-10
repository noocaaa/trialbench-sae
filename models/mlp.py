import torch.nn as nn
from src import config


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for tabular SAE prediction.
    No structural assumptions — learns any non-linear combination of features.
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=None):
        super().__init__()
        dropout = dropout if dropout is not None else config.DROPOUT
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.BatchNorm1d(hidden_dim // 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


