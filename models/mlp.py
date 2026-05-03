import torch.nn as nn
from src.data_loader import load_phase
from src.train import train_model


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for tabular SAE prediction.
    No structural assumptions — learns any non-linear combination of features.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def run(phase, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase)
    train_model(MLP(X_train.shape[1]),
                X_train, X_test, y_train, y_test,
                pos_weight, model_name="MLP", phase=phase, **kwargs)