import torch.nn as nn
from src import config
from src.data_loader import load_phase
from src.train import train_model, DEVICE


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


def run(phase, use_text=False, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase, use_text=use_text)
    model_name = "MLP+Text" if use_text else "MLP"
    train_model(MLP(X_train.shape[1], **kwargs).to(DEVICE),
                X_train, X_test, y_train, y_test,
                pos_weight, model_name=model_name, phase=phase, **kwargs)
