"""
src/nested_cv.py — Nested cross-validation with threshold calibration.

Design: 5 outer folds (unbiased performance estimate) × 3 inner folds
(threshold calibration). All preprocessing is fit per-outer-fold on the
training split only to prevent data leakage.

Key features:
- Inner loop: train model → collect probabilities → find threshold
- Outer loop: train final model → evaluate with calibrated threshold
- Threshold: computed by concatenating all inner-fold probabilities,
  NOT by averaging thresholds (which is statistically invalid)
- Text features: supported via use_text flag, fit on outer train only
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from src import config
from src.cv_preprocessing import preprocess_cv
from src.data_loader import _DEFAULT_DATA
from src.evaluate import evaluate
from src.train import train_model, DEVICE
from src.train_sklearn import train_sklearn_model
from src.utils import find_best_threshold
from src.mlflow_tracker import tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_raw_phase(phase, data_dir=None):
    """Load raw (unprocessed) train+test DataFrames and merge them."""
    if data_dir is None:
        data_dir = _DEFAULT_DATA

    base = f"{data_dir}/Phase{phase}"
    X_train = pd.read_csv(f"{base}/train_x.csv")
    y_train = pd.read_csv(f"{base}/train_y.csv")
    X_test = pd.read_csv(f"{base}/test_x.csv")
    y_test = pd.read_csv(f"{base}/test_y.csv")

    y_train = y_train["Y/N"].values.astype(int)
    y_test = y_test["Y/N"].values.astype(int)

    X = pd.concat([X_train, X_test], ignore_index=True)
    y = np.concatenate([y_train, y_test])
    return X, y


def _train_dl_for_threshold(model_fn, model_kwargs, X_train, y_train, X_val, y_val, pos_weight):
    """
    Train a fresh DL model on (X_train, y_train) with early stopping on X_val.
    Returns the trained model (for probability extraction).
    No evaluation is run and no files are saved.
    
    Uses a simplified training loop to avoid BatchNorm issues with small batches.
    """
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    model = model_fn(**model_kwargs).to(DEVICE)
    
    # Simple training loop (no checkpointing, no file I/O)
    epochs = min(30, config.EPOCHS)  # Shorter for inner loop
    batch_size = min(config.BATCH_SIZE, len(X_train))
    
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train).to(DEVICE),
            torch.tensor(y_train, dtype=torch.float32).to(DEVICE),
        ),
        batch_size=batch_size, shuffle=True,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    )
    
    X_val_tensor = torch.tensor(X_val).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    
    best_val_loss = float("inf")
    best_state = None
    patience = 5
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            break
    
    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def _eval_dl(model_fn, model_kwargs, X_train, y_train, X_test, y_test, pos_weight, threshold):
    """Train final DL model and evaluate with a fixed threshold."""
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    model = model_fn(**model_kwargs).to(DEVICE)
    
    # Split train into train/val for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=config.VAL_SPLIT,
        random_state=42, stratify=y_train,
    )
    
    epochs = config.EPOCHS
    batch_size = min(config.BATCH_SIZE, len(X_tr))
    if batch_size < 2:
        batch_size = 2  # Ensure batch size > 1 for BatchNorm
    
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr).to(DEVICE),
            torch.tensor(y_tr, dtype=torch.float32).to(DEVICE),
        ),
        batch_size=batch_size, shuffle=True,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    )
    
    X_val_tensor = torch.tensor(X_val).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    
    best_val_loss = float("inf")
    best_state = None
    patience = config.PATIENCE
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            # Skip batches of size 1 (BatchNorm issue)
            if X_batch.size(0) < 2:
                continue
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            if config.GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            break
    
    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Evaluate on test
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
        y_prob = torch.sigmoid(logits).cpu().numpy()
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred


def _eval_sklearn(model_fn, model_kwargs, X_train, y_train, X_test, y_test, pos_weight, threshold,
                    model_name="model", phase="phase"):
    """Train final sklearn model and evaluate with a fixed threshold."""
    model = model_fn(**model_kwargs)
    # Train without evaluation — we evaluate once below to avoid triple evaluation
    train_sklearn_model(
        model,
        X_train, X_test,
        y_train, y_test,
        model_name=model_name,
        phase=phase,
        pos_weight=pos_weight,
        tune_threshold=False,
        threshold=threshold,
        save_artifacts=True,
        skip_eval=True,
    )
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def nested_cv_single_model(
    model_fn,
    phase,
    model_name,
    is_tree=False,
    is_dl=False,
    outer_folds=None,
    inner_folds=None,
    verbose=False,
    model_kwargs=None,
    use_text=False,
    tune_hyperparams=False,
):
    """
    Run nested CV for a single model on a single phase.

    Outer loop (5 folds): unbiased performance estimate.
    Inner loop (3 folds): calibrate decision threshold without data leakage.

    Parameters
    ----------
    model_fn    : callable — returns a fresh model instance
    phase       : str — "1", "2", "3", or "4"
    model_name  : str — display name, e.g. "MLP"
    is_tree     : bool — use tree-friendly preprocessing (ordinal encoding, no scaling)
    is_dl       : bool — PyTorch model (uses train_model) vs sklearn (train_sklearn_model)
    outer_folds : int — number of outer folds (default: config.OUTER_FOLDS)
    inner_folds : int — number of inner folds (default: config.INNER_FOLDS)
    verbose     : bool — print fold progress
    model_kwargs: dict — extra kwargs passed to model_fn (e.g. input_dim)
    use_text    : bool — if True, append TF-IDF text features (fit on outer train only)
    tune_hyperparams : bool — if True, run grid search on sklearn models (inner CV)

    Returns
    -------
    list of dicts — one metrics dict per outer fold
    """
    model_kwargs = model_kwargs or {}
    outer_folds = outer_folds if outer_folds is not None else config.OUTER_FOLDS
    inner_folds = inner_folds if inner_folds is not None else config.INNER_FOLDS

    # Load full dataset (train + test merged)
    X_raw, y = _load_raw_phase(phase)

    outer_skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    fold_results = []

    for outer_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_skf.split(X_raw, y)):
        if verbose:
            print(f"\n  --- Outer fold {outer_idx + 1}/{outer_folds} ---")

        # ------------------------------------------------------------------
        # 1. Split raw data for this outer fold
        # ------------------------------------------------------------------
        X_outer_train_raw = X_raw.iloc[outer_train_idx].copy()
        X_outer_test_raw = X_raw.iloc[outer_test_idx].copy()
        y_outer_train = y[outer_train_idx]
        y_outer_test = y[outer_test_idx]

        # ------------------------------------------------------------------
        # 2. Preprocess with NO leakage (fit on outer train only)
        # ------------------------------------------------------------------
        X_outer_train, X_outer_test = preprocess_cv(
            X_outer_train_raw,
            X_outer_test_raw,
            phase=phase,
            for_tree=is_tree,
            verbose=(verbose and outer_idx == 0),
            use_text=use_text,
        )

        # ------------------------------------------------------------------
        # 3. Compute class weight from outer training data
        # ------------------------------------------------------------------
        neg = (y_outer_train == 0).sum()
        pos = (y_outer_train == 1).sum()
        pos_weight = float(neg / pos) if pos > 0 else 1.0

        # Update input_dim for DL models based on actual preprocessed shape
        if is_dl:
            model_kwargs = dict(model_kwargs)
            model_kwargs["input_dim"] = X_outer_train.shape[1]

        # ------------------------------------------------------------------
        # 4. INNER CV: hyperparameter tuning + threshold calibration
        #
        # Step 1 (optional): Tune hyperparameters using the inner training data.
        #   - Sklearn: Optuna Bayesian optimization (fast, 20 trials)
        #   - Small DL: Limited grid search (3-6 configs, 20 epochs each)
        #   - Large DL: Fixed hyperparameters (too slow to tune)
        #
        # Step 2: Use best params to train models on inner folds and collect
        # predicted probabilities for threshold calibration.
        #
        # FIX #1: Threshold from concatenated probabilities (not averaged).
        # FIX #2: DL uses separate train/val/cal (no double-dipping).
        # FIX #3: Independent random seeds.
        # ------------------------------------------------------------------

        inner_seed = 42 + outer_idx * 1000

        # ── Step 1: Hyperparameter tuning (if enabled) ──
        best_params = None
        if tune_hyperparams:
            from src.hyperparam_tuning import tune_model

            if is_dl:
                # For DL: use a subset of outer_train for quick tuning
                X_tune, X_tune_val, y_tune, y_tune_val = train_test_split(
                    X_outer_train, y_outer_train,
                    test_size=0.2, random_state=inner_seed,
                    stratify=y_outer_train,
                )
                best_params = tune_model(
                    model_name, model_fn,
                    X_tune, y_tune,
                    is_dl=True,
                    X_val=X_tune_val, y_val=y_tune_val,
                    pos_weight=pos_weight,
                    verbose=verbose,
                )
            else:
                # For sklearn: Optuna on full outer_train
                best_params = tune_model(
                    model_name, model_fn,
                    X_outer_train, y_outer_train,
                    is_dl=False,
                    n_trials=20,
                    verbose=verbose,
                )

            if verbose and best_params:
                print(f"    Best params: {best_params}")

        # Merge tuned params with defaults (tuned params take precedence)
        effective_kwargs = dict(model_kwargs)
        if best_params:
            effective_kwargs.update(best_params)
        
        # DL models need input_dim passed to their factory
        if is_dl:
            effective_kwargs["input_dim"] = X_outer_train.shape[1]

        # ── Step 2: Inner CV for threshold calibration ──
        inner_skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=inner_seed)
        all_y_prob = []
        all_y_true = []

        for inner_idx, (inner_train_idx, inner_cal_idx) in enumerate(
            inner_skf.split(X_outer_train, y_outer_train)
        ):
            X_inner_train = X_outer_train[inner_train_idx]
            X_cal = X_outer_train[inner_cal_idx]
            y_inner_train = y_outer_train[inner_train_idx]
            y_cal = y_outer_train[inner_cal_idx]

            if len(np.unique(y_cal)) < 2:
                if verbose:
                    print(f"    Inner fold {inner_idx + 1}: skipped (single class)")
                continue

            if is_dl:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_inner_train, y_inner_train,
                    test_size=0.15, random_state=inner_seed + inner_idx,
                    stratify=y_inner_train,
                )
                model = _train_dl_for_threshold(
                    model_fn, effective_kwargs,
                    X_tr, y_tr, X_val, y_val, pos_weight
                )
                model.eval()
                with torch.no_grad():
                    cal_tensor = torch.tensor(X_cal, dtype=torch.float32).to(DEVICE)
                    cal_logits = model(cal_tensor)
                    y_prob_cal = torch.sigmoid(cal_logits).cpu().numpy()
            else:
                # Use train_sklearn_model to properly handle pos_weight and sample_weight
                model = model_fn(**effective_kwargs)
                model = train_sklearn_model(
                    model,
                    X_inner_train, X_cal,   # X_cal as placeholder for X_test (ignored when skip_eval=True)
                    y_inner_train, y_cal,   # y_cal as placeholder for y_test (ignored when skip_eval=True)
                    model_name="_inner_tmp",
                    phase="_inner",
                    pos_weight=pos_weight,
                    tune_threshold=False,   # we compute threshold ourselves from all inner folds
                    save_artifacts=False,   # no file I/O in inner loop
                    skip_eval=True,         # no evaluation — we just need the trained model
                )
                if hasattr(model, "predict_proba"):
                    y_prob_cal = model.predict_proba(X_cal)[:, 1]
                else:
                    y_prob_cal = model.predict(X_cal).astype(float)

            all_y_prob.append(y_prob_cal)
            all_y_true.append(y_cal)

        # FIX #1: Threshold from all inner-fold probabilities
        if len(all_y_prob) > 0:
            all_y_prob = np.concatenate(all_y_prob)
            all_y_true = np.concatenate(all_y_true)
            threshold = find_best_threshold(
                all_y_true, all_y_prob, criterion=config.THRESHOLD_CRITERION
            )
            if verbose:
                print(f"    Threshold from {len(all_y_true)} inner samples: {threshold:.4f}")
        else:
            print(f"  WARNING: No valid inner folds. Using threshold=0.5")
            threshold = 0.5

        # ------------------------------------------------------------------
        # 5. OUTER FOLD: final evaluation with calibrated threshold
        #
        # Use effective_kwargs (defaults + tuned params) for final training.
        # ------------------------------------------------------------------
        if is_dl:
            y_prob, y_pred = _eval_dl(
                model_fn, effective_kwargs,
                X_outer_train, y_outer_train,
                X_outer_test, y_outer_test,
                pos_weight, threshold,
            )
        else:
            y_prob, y_pred = _eval_sklearn(
                model_fn, effective_kwargs,
                X_outer_train, y_outer_train,
                X_outer_test, y_outer_test,
                pos_weight, threshold,
                model_name=model_name,
                phase=phase,
            )

        metrics = evaluate(
            y_outer_test, y_pred, y_prob,
            model_name=model_name,
            phase=phase,
            threshold=threshold,
            save_path=f"results/{model_name.replace(' ', '_')}_{phase}_fold{outer_idx}.json",
        )

        # Enrich with CV metadata
        metrics["threshold"] = round(threshold, 4)
        metrics["n_inner_samples"] = int(len(all_y_true)) if all_y_prob is not None else 0
        metrics["fold"] = outer_idx
        metrics["n_outer_folds"] = outer_folds
        metrics["n_inner_folds"] = inner_folds

        # Re-save enriched metrics
        fold_path = f"results/{model_name.replace(' ', '_')}_{phase}_fold{outer_idx}.json"
        with open(fold_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # ── Log to MLflow ──
        if tracker.enabled:
            with tracker.start_fold_run(model_name, phase, outer_idx, outer_folds):
                tracker.log_param("threshold", threshold)
                tracker.log_param("n_inner_samples", metrics["n_inner_samples"])
                tracker.log_param("pos_weight", pos_weight)
                tracker.log_param("input_dim", X_outer_train.shape[1])
                if best_params:
                    tracker.log_tuning_results(best_params, tuning_method="optuna" if not is_dl else "grid_search")
                # Log metrics (filter out non-scalar)
                scalar_metrics = {k: v for k, v in metrics.items()
                                  if k in ("accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc")}
                tracker.log_metrics(scalar_metrics)
                tracker.log_artifact(fold_path)

        fold_results.append(metrics)

    return fold_results
