import torch.nn as nn
from src import config
from src.data_loader import load_phase
from src.train import train_model, DEVICE


class CNN(nn.Module):
    """
    1D Convolutional Network for tabular SAE prediction.
    Treats the feature vector as a 1D sequence and applies Conv1d
    to detect local patterns between adjacent features.
    """
    def __init__(self, input_dim, conv_channels=32, dropout=None):
        super().__init__()
        dropout = dropout if dropout is not None else config.DROPOUT
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x).squeeze(1)


def run(phase, use_text=False, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase, use_text=use_text)
    model_name = "CNN+Text" if use_text else "CNN"
    train_model(CNN(X_train.shape[1], **kwargs).to(DEVICE),
                X_train, X_test, y_train, y_test,
                pos_weight, model_name=model_name, phase=phase, **kwargs)
