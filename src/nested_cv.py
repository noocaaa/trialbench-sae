"""
src/nested_cv.py — Nested cross-validation with threshold calibration.

Design: 5 outer folds (unbiased performance estimate) × 3 inner folds
(threshold calibration). All preprocessing is fit per-outer-fold on the
training split only to prevent data leakage.

Key simplifications vs. previous version:
- Inner loop trains a model ONLY to get a threshold — no result files saved.
- Outer loop trains ONE final model, evaluates once with the averaged threshold.
- DL and sklearn paths share common logic via helper functions.
- No hardcoded epochs — inner loop uses config.EPOCHS with early stopping.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from src import config
from src.cv_preprocessing import preprocess_cv
from src.data_loader import _DEFAULT_DATA
from src.evaluate import evaluate
from src.train import train_model, DEVICE
from src.train_sklearn import train_sklearn_model
from src.utils import find_best_threshold


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


def _train_dl_for_threshold(model_fn, model_kwargs, X_train, y_train, X_cal, y_cal, pos_weight):
    """
    Train a fresh DL model on (X_train, y_train) with early stopping on X_cal.
    Returns predicted probabilities on X_cal.
    No evaluation is run and no files are saved.
    """
    model = model_fn(**model_kwargs).to(DEVICE)
    train_model(
        model,
        X_train, X_cal,          # X_test placeholder (ignored when skip_eval=True)
        y_train, y_cal,          # y_test placeholder (ignored when skip_eval=True)
        pos_weight,
        model_name="_inner_tmp", # dummy name, never saved
        phase="_inner",
        val_split=0,             # no internal split; we provide X_val
        cal_split=0,             # no internal split; we provide X_cal
        X_val=X_cal,
        y_val=y_cal,
        X_cal=X_cal,
        y_cal=y_cal,
        save_artifacts=False,    # no file I/O in inner loop
        skip_eval=True,          # no evaluation on X_test placeholder
    )
    model.eval()
    with torch.no_grad():
        cal_tensor = torch.tensor(X_cal, dtype=torch.float32).to(DEVICE)
        cal_logits = model(cal_tensor)
        y_prob_cal = torch.sigmoid(cal_logits).cpu().numpy()
    return y_prob_cal


def _train_sklearn_for_threshold(model_fn, model_kwargs, X_train, y_train, X_cal, y_cal, pos_weight):
    """
    Train a fresh sklearn model on (X_train, y_train), get probs on X_cal.
    Returns predicted probabilities on X_cal.
    No evaluation is run and no files are saved.
    """
    model = model_fn(**model_kwargs)
    model = train_sklearn_model(
        model,
        X_train, X_cal,          # X_test placeholder (ignored when skip_eval=True)
        y_train, y_cal,          # y_test placeholder (ignored when skip_eval=True)
        model_name="_inner_tmp",
        phase="_inner",
        pos_weight=pos_weight,
        tune_threshold=False,    # we will compute threshold ourselves
        save_artifacts=False,    # no file I/O in inner loop
        skip_eval=True,          # no evaluation on X_test placeholder
    )
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_cal)[:, 1]
    return model.predict(X_cal).astype(float)


def _eval_dl(model_fn, model_kwargs, X_train, y_train, X_test, y_test, pos_weight, threshold):
    """Train final DL model and evaluate with a fixed threshold."""
    model = model_fn(**model_kwargs).to(DEVICE)
    train_model(
        model,
        X_train, X_test,
        y_train, y_test,
        pos_weight,
        model_name="_final_tmp",
        phase="_final",
        val_split=config.VAL_SPLIT,
        cal_split=config.CAL_SPLIT,
        save_artifacts=False,
        skip_eval=True,
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
        y_prob = torch.sigmoid(logits).cpu().numpy()
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred


def _eval_sklearn(model_fn, model_kwargs, X_train, y_train, X_test, y_test, pos_weight, threshold):
    """Train final sklearn model and evaluate with a fixed threshold."""
    model = model_fn(**model_kwargs)
    train_sklearn_model(
        model,
        X_train, X_test,
        y_train, y_test,
        model_name="_final_tmp",
        phase="_final",
        pos_weight=pos_weight,
        tune_threshold=False,
        threshold=threshold,
        save_artifacts=False,
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
        # 4. INNER CV: threshold calibration
        # ------------------------------------------------------------------
        inner_skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42 + outer_idx)
        thresholds = []

        for inner_idx, (inner_train_idx, inner_cal_idx) in enumerate(
            inner_skf.split(X_outer_train, y_outer_train)
        ):
            X_inner_train = X_outer_train[inner_train_idx]
            X_cal = X_outer_train[inner_cal_idx]
            y_inner_train = y_outer_train[inner_train_idx]
            y_cal = y_outer_train[inner_cal_idx]

            # Skip if calibration set has only one class (can't compute threshold)
            if len(np.unique(y_cal)) < 2:
                continue

            if is_dl:
                y_prob_cal = _train_dl_for_threshold(
                    model_fn, model_kwargs,
                    X_inner_train, y_inner_train, X_cal, y_cal, pos_weight
                )
            else:
                y_prob_cal = _train_sklearn_for_threshold(
                    model_fn, model_kwargs,
                    X_inner_train, y_inner_train, X_cal, y_cal, pos_weight
                )

            threshold = find_best_threshold(y_cal, y_prob_cal, criterion=config.THRESHOLD_CRITERION)
            thresholds.append(threshold)

        # Average thresholds across inner folds (fallback to 0.5 if none valid)
        threshold_mean = float(np.mean(thresholds)) if thresholds else 0.5

        if verbose:
            print(
                f"    Thresholds from inner folds: "
                f"{[round(t, 4) for t in thresholds]} -> mean={threshold_mean:.4f}"
            )

        # ------------------------------------------------------------------
        # 5. OUTER FOLD: final evaluation with averaged threshold
        # ------------------------------------------------------------------
        if is_dl:
            y_prob, y_pred = _eval_dl(
                model_fn, model_kwargs,
                X_outer_train, y_outer_train,
                X_outer_test, y_outer_test,
                pos_weight, threshold_mean,
            )
        else:
            y_prob, y_pred = _eval_sklearn(
                model_fn, model_kwargs,
                X_outer_train, y_outer_train,
                X_outer_test, y_outer_test,
                pos_weight, threshold_mean,
            )

        metrics = evaluate(
            y_outer_test, y_pred, y_prob,
            model_name=model_name,
            phase=phase,
            threshold=threshold_mean,
            save_path=f"results/{model_name.replace(' ', '_')}_{phase}_fold{outer_idx}.json",
        )

        # Enrich with CV metadata
        metrics["thresholds_inner"] = [round(t, 4) for t in thresholds]
        metrics["threshold_mean"] = round(threshold_mean, 4)
        metrics["y_prob"] = np.array(y_prob).tolist()
        metrics["fold"] = outer_idx
        metrics["n_outer_folds"] = outer_folds
        metrics["n_inner_folds"] = inner_folds

        # Re-save enriched metrics
        fold_path = f"results/{model_name.replace(' ', '_')}_{phase}_fold{outer_idx}.json"
        with open(fold_path, "w") as f:
            json.dump(metrics, f, indent=2)

        fold_results.append(metrics)

    return fold_results
