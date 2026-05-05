import torch.nn as nn
from src.data_loader import load_phase
from src.train import train_model


class CNN(nn.Module):
    """
    1D Convolutional Network for tabular SAE prediction.
    Treats the feature vector as a 1D sequence and applies Conv1d
    to detect local patterns between adjacent features.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x).squeeze(1)


def run(phase, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase)
    train_model(CNN(X_train.shape[1]),
                X_train, X_test, y_train, y_test,
                pos_weight, model_name="CNN", phase=phase, **kwargs)