import torch.nn as nn
from src import config
from src.data_loader import load_phase
from src.train import train_model, DEVICE   


class RNN(nn.Module):
    """
    LSTM Recurrent Network for tabular SAE prediction.
    Treats each feature as a time step in a sequence.
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=None):
        super().__init__()
        dropout = dropout if dropout is not None else config.DROPOUT
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim,
                            num_layers=2, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(1)


def run(phase, use_text=False, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase, use_text=use_text)
    model_name = "RNN+Text" if use_text else "RNN"
    train_model(RNN(X_train.shape[1], **kwargs).to(DEVICE),
                X_train, X_test, y_train, y_test,
                pos_weight, model_name=model_name, phase=phase, **kwargs)
