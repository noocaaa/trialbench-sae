# src/train.py — Shared training loop for all deep learning models

import copy
import json
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.evaluate import evaluate
from src.utils import find_best_threshold
from src.mlflow_tracker import tracker

# GPU or CPU automatic detection and selection
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE}")

# NOTE: We use standard BCEWithLogitsLoss for consistency with the paper.
# The pos_weight parameter handles class imbalance, which is sufficient
# for this medical prediction task.


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
    patience=None,
    val_split=None,
    cal_split=None,
    X_val=None,
    y_val=None,
    X_cal=None,
    y_cal=None,
    save_artifacts=True,
    skip_eval=False,
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
    patience   : int        — overrides config.PATIENCE if provided
    val_split  : float      — overrides config.VAL_SPLIT if provided
    cal_split  : float      — overrides config.CAL_SPLIT if provided
    X_val, y_val : optional pre-split validation set (bypasses internal val_split)
    X_cal, y_cal : optional pre-split calibration set (bypasses internal cal_split)
    save_artifacts : bool
        If False, skip saving results, checkpoints, and loss curves.
        Useful for inner CV loops where only the threshold matters.
    skip_eval : bool
        If True, skip the final evaluate() call. Useful when the caller
        only needs the trained model or threshold (e.g. inner CV).

    Returns
    -------
    dict — metrics from evaluate() (empty dict if skip_eval=True)
    """
    epochs     = epochs     if epochs     is not None else config.EPOCHS
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    lr         = lr         if lr         is not None else config.LR
    patience   = patience   if patience   is not None else config.PATIENCE
    val_split  = val_split  if val_split  is not None else config.VAL_SPLIT
    cal_split  = cal_split  if cal_split  is not None else config.CAL_SPLIT

    # ── Step 1: Separate calibration set for threshold tuning ─────
    # If pre-split calibration set is provided, use it directly.
    # Otherwise, split from training data.
    if X_cal is not None and y_cal is not None:
        X_temp, y_temp = X_train, y_train
    elif cal_split > 0:
        X_temp, X_cal, y_temp, y_cal = train_test_split(
            X_train, y_train, test_size=cal_split,
            random_state=42, stratify=y_train,
        )
    else:
        X_temp, y_temp = X_train, y_train
        X_cal, y_cal = None, None

    # ── Step 2: From remaining data, separate validation for early stopping ──
    # If pre-split validation set is provided, use it directly.
    # Otherwise, split from remaining training data.
    if X_val is not None and y_val is not None:
        X_tr, y_tr = X_temp, y_temp
    elif val_split > 0:
        # Adjust proportion since X_temp is (1 - cal_split) of original
        adjusted_val_split = val_split / (1 - cal_split) if cal_split > 0 and X_cal is None else val_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_temp, y_temp, test_size=adjusted_val_split,
            random_state=43, stratify=y_temp,
        )
    else:
        X_tr, y_tr = X_temp, y_temp
        X_val, y_val = None, None  # no validation split; no early stopping

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr).to(DEVICE),
            torch.tensor(y_tr, dtype=torch.float32).to(DEVICE),
        ),
        batch_size=batch_size, shuffle=True,
    )

    # ── Optimizer ─────────────────────────────────────────────────
    opt_name = config.OPTIMIZER.lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer in config: {config.OPTIMIZER}")

    # ── Loss function ───────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    )

    # ── Scheduler ─────────────────────────────────────────────────
    sched_name = config.SCHEDULER.lower()
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr / 10
        )
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
    elif sched_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler in config: {config.SCHEDULER}")

    # ── Cache tensors that never change during training ───────────
    X_val_tensor = torch.tensor(X_val).to(DEVICE) if X_val is not None else None
    X_cal_tensor = torch.tensor(X_cal).to(DEVICE) if X_cal is not None else None
    X_test_tensor = torch.tensor(X_test).to(DEVICE)

    # ── Early stopping state ──────────────────────────────────────
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    stopped_early = False

    history = []
    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            if config.GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.GRAD_CLIP
                )
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────
        if val_split > 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_loss = criterion(
                    val_logits,
                    torch.tensor(y_val, dtype=torch.float32).to(DEVICE),
                ).item()
        else:
            val_loss = train_loss

        history.append({
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
        })

        # ── Scheduler step ────────────────────────────────────────
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = lr

        print(f"  Epoch {epoch+1:2d}/{epochs} — "
              f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
              f"lr: {current_lr:.2e}")

        # ── Early stopping check ──────────────────────────────────
        if val_split > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"  Early stopping triggered (no improvement for {patience} epochs). "
                      f"Restoring best model from epoch {epoch + 1 - patience}.")
                stopped_early = True
                break
        else:
            # No validation split: save the latest state each epoch
            best_state = copy.deepcopy(model.state_dict())

    # ── Restore best model ────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    elif stopped_early:
        print("  WARNING: early stopping triggered but no best state found.")

    # Save loss curve and checkpoint (only if save_artifacts=True)
    if save_artifacts:
        os.makedirs("results", exist_ok=True)
        loss_path = f"results/loss_{model_name}_{phase}.json"
        with open(loss_path, "w") as f:
            json.dump(history, f)

        if best_state is not None:
            os.makedirs("models/checkpoints", exist_ok=True)
            checkpoint_path = f"models/checkpoints/{model_name}_{phase}.pt"
            torch.save(best_state, checkpoint_path)
            print(f"  Checkpoint saved -> {checkpoint_path}")

        # ── Log to MLflow ──
        if tracker.enabled and tracker.get_run_id() is not None:
            tracker.log_training_curve(history)
            tracker.log_artifact(loss_path)
            if best_state is not None and os.path.exists(checkpoint_path):
                tracker.log_artifact(checkpoint_path)

    # ── Threshold tuning on calibration set ──
    threshold = 0.5
    if config.TUNE_THRESHOLD and X_cal is not None:
        model.eval()
        with torch.no_grad():
            cal_logits = model(X_cal_tensor)
            y_prob_cal = torch.sigmoid(cal_logits).cpu().numpy()
        threshold = find_best_threshold(y_cal, y_prob_cal, criterion=config.THRESHOLD_CRITERION)
        if save_artifacts:
            print(f"  Optimal threshold ({config.THRESHOLD_CRITERION}): {threshold:.4f}")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        y_prob = torch.sigmoid(logits).cpu().numpy()
        y_pred = (y_prob >= threshold).astype(int)

    if skip_eval:
        return {}

    return evaluate(
        y_test, y_pred, y_prob,
        model_name=model_name, phase=phase, threshold=threshold,
        save=save_artifacts,
    )
