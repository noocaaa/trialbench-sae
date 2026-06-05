import torch
import torch.nn as nn
from src import config
from src.data_loader import load_phase
from src.train import train_model, DEVICE


class Transformer(nn.Module):
    """
    Transformer Encoder for tabular SAE prediction.

    Each feature becomes its own token (like a word in a sentence).
    A learnable [CLS] token aggregates information via self-attention.
    """
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        # Project each scalar feature independently into embed_dim
        self.input_proj = nn.Linear(1, embed_dim)

        # Learnable [CLS] token — the model learns to aggregate here
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=config.DROPOUT, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: [batch, input_dim]
        x = x.unsqueeze(-1)                     # [batch, input_dim, 1]
        x = self.input_proj(x)                  # [batch, input_dim, embed_dim]

        # Prepend CLS token to the sequence
        cls = self.cls_token.expand(x.size(0), -1, -1)  # [batch, 1, embed_dim]
        x = torch.cat([cls, x], dim=1)          # [batch, input_dim + 1, embed_dim]

        x = self.encoder(x)                     # [batch, input_dim + 1, embed_dim]

        # Classify using the CLS token output
        cls_out = x[:, 0]                       # [batch, embed_dim]
        return self.fc(cls_out).squeeze(1)


def run(phase, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase)
    train_model(Transformer(X_train.shape[1]).to(DEVICE),
                X_train, X_test, y_train, y_test,
                pos_weight, model_name="Transformer", phase=phase, **kwargs)
