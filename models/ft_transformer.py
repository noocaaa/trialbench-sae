import torch
import torch.nn as nn
from src import config
from src.data_loader import load_phase
from src.train import train_model, DEVICE


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for tabular data.

    Unlike the basic Transformer (which treats each feature as a sequence element
    with a shared Linear projection), FT-Transformer uses a separate learned
    embedding per feature. Each feature gets its own embedding vector, then
    a [CLS] token aggregates via self-attention.

    Reference: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data",
    NeurIPS 2021.
    """
    def __init__(self, input_dim, embed_dim=32, num_heads=4, num_layers=2, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # One learned embedding per feature: each feature is projected to embed_dim
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(input_dim)
        ])

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=config.DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: [batch, input_dim]
        # Embed each feature independently
        embeddings = []
        for i in range(self.input_dim):
            feat = x[:, i:i+1]                    # [batch, 1]
            emb = self.feature_embeddings[i](feat) # [batch, embed_dim]
            embeddings.append(emb.unsqueeze(1))    # [batch, 1, embed_dim]

        x = torch.cat(embeddings, dim=1)          # [batch, input_dim, embed_dim]

        # Prepend CLS token
        cls = self.cls_token.expand(x.size(0), -1, -1)  # [batch, 1, embed_dim]
        x = torch.cat([cls, x], dim=1)            # [batch, input_dim + 1, embed_dim]

        x = self.encoder(x)                       # [batch, input_dim + 1, embed_dim]

        # Classify using CLS token
        cls_out = x[:, 0]                         # [batch, embed_dim]
        return self.fc(cls_out).squeeze(1)


def run(phase, **kwargs):
    X_train, X_test, y_train, y_test, pos_weight = load_phase(phase)
    train_model(FTTransformer(X_train.shape[1]).to(DEVICE),
                X_train, X_test, y_train, y_test,
                pos_weight, model_name="FT-Transformer", phase=phase, **kwargs)
