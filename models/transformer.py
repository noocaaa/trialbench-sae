import torch.nn as nn
from src.data_loader import load_phase
from src.train import train_model


class Transformer(nn.Module):
    """
    Transformer Encoder for tabular SAE prediction.
    Projects the full feature vector as a single token with Pre-LN architecture.
    """
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.encoder(x).squeeze(1)
        return self.fc(x).squeeze(1)


def run(phase, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase)
    train_model(Transformer(X_train.shape[1]),
                X_train, X_test, y_train, y_test,
                pos_weight, model_name="Transformer", phase=phase, **kwargs)