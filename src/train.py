# src/train.py — Shared training loop for all deep learning models

import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.evaluate import evaluate


def train_model(
    model,
    X_train, X_test,
    y_train, y_test,
    pos_weight,
    model_name,
    phase,
    epochs=None,
    batch_size=None,
    lr=None,
):
    """
    Shared training loop for all deep learning models.

    Parameters
    ----------
    model      : nn.Module  — any PyTorch model with a forward() method
    X_train    : np.ndarray — preprocessed training features
    X_test     : np.ndarray — preprocessed test features
    y_train    : np.ndarray — training labels (0/1)
    y_test     : np.ndarray — test labels (0/1)
    pos_weight : float      — neg/pos ratio for BCEWithLogitsLoss
    model_name : str        — e.g. "CNN", "MLP"
    phase      : str        — "1", "2", "3", "4"
    epochs     : int        — overrides config.EPOCHS if provided
    batch_size : int        — overrides config.BATCH_SIZE if provided
    lr         : float      — overrides config.LR if provided
    """
    epochs     = epochs     or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    lr         = lr         or config.LR

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 10
    )

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            if config.GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.GRAD_CLIP
                )
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(train_loader)
        history.append(round(avg, 6))
        scheduler.step()
        print(f"  Epoch {epoch+1}/{epochs} — loss: {avg:.4f}"
              f"  lr: {scheduler.get_last_lr()[0]:.2e}")

    # Save loss curve
    os.makedirs("results", exist_ok=True)
    with open(f"results/loss_{model_name}_{phase}.json", "w") as f:
        json.dump(history, f)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test))
        y_prob = torch.sigmoid(logits).numpy()
        y_pred = (y_prob >= 0.5).astype(int)

    evaluate(y_test, y_pred, y_prob, model_name=model_name, phase=phase)