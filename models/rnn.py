import torch.nn as nn
from src import config
from src.data_loader import load_phase
from src.train import train_model, DEVICE   


class RNN(nn.Module):
    """
    LSTM Recurrent Network for tabular SAE prediction.
    Treats each feature as a time step in a sequence.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim,
                            num_layers=2, batch_first=True, dropout=config.DROPOUT)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(1)


def run(phase, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase)
    train_model(RNN(X_train.shape[1]).to(DEVICE),
                X_train, X_test, y_train, y_test,
                pos_weight, model_name="RNN", phase=phase, **kwargs)
