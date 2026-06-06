import inspect
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from src import config
from src.evaluate import evaluate
from src.utils import find_best_threshold

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

def train_sklearn_model(model, X_train, X_test, y_train, y_test, model_name, phase,
                        pos_weight=None, tune_threshold=True, threshold_criterion="f1",
                        cal_split=None, X_cal=None, y_cal=None, threshold=None,
                        save_artifacts=True, skip_eval=False, **kwargs):
    """
    Training function for sklearn models.

    Parameters
    ----------
    model              : sklearn model — any model with fit() and predict_proba() methods
    X_train            : np.ndarray — preprocessed training features
    X_test             : np.ndarray — preprocessed test features
    y_train            : np.ndarray — training labels (0/1)
    y_test             : np.ndarray — test labels (0/1)
    model_name         : str        — e.g. "LogisticRegression", "RandomForest"
    phase              : str        — "1", "2", "3", "4"
    pos_weight         : float      — neg/pos ratio for exact class weighting (like PyTorch)
    tune_threshold     : bool       — if True, tune threshold on a calibration split
    threshold_criterion: str        — "f1" or "youden"
    cal_split          : float      — fraction of training data for threshold calibration
    X_cal, y_cal       : optional pre-split calibration set (bypasses internal cal_split)
    threshold          : float      — explicit threshold when tune_threshold=False
    kwargs             : dict       — additional arguments (passed through)
    save_artifacts     : bool       — if False, skip saving results and info files
    skip_eval          : bool       — if True, skip the final evaluate() call

    Returns
    -------
    sklearn model (fitted)
    """
    cal_split = cal_split if cal_split is not None else config.CAL_SPLIT
    # ── Helper: build sample_weight dict for fit() ────────────────
    def _fit_kwargs(y):
        fk = {}
        if pos_weight is not None and pos_weight > 0:
            if 'sample_weight' in inspect.signature(model.fit).parameters:
                fk['sample_weight'] = np.where(y == 1, pos_weight, 1.0)
        return fk

    threshold = threshold if threshold is not None else 0.5

    if tune_threshold and (X_cal is not None and y_cal is not None):
        # Use pre-split calibration set (no internal splitting)
        # Fit on training portion
        model.fit(X_train, y_train, **_fit_kwargs(y_train))

        # Get calibration probabilities
        if hasattr(model, "predict_proba"):
            y_prob_cal = model.predict_proba(X_cal)[:, 1]
        else:
            y_prob_cal = model.predict(X_cal).astype(float)

        # Find optimal threshold on calibration set
        threshold = find_best_threshold(y_cal, y_prob_cal, criterion=threshold_criterion)

        # Re-fit on full training data for final evaluation
        model.fit(X_train, y_train, **_fit_kwargs(y_train))
    elif tune_threshold and cal_split > 0:
        # Split training data to get a calibration set for threshold tuning
        # (sklearn models do not need a validation set for early stopping)
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X_train, y_train, test_size=cal_split,
            random_state=42, stratify=y_train,
        )

        # Fit on training portion
        model.fit(X_tr, y_tr, **_fit_kwargs(y_tr))

        # Get calibration probabilities
        if hasattr(model, "predict_proba"):
            y_prob_cal = model.predict_proba(X_cal)[:, 1]
        else:
            y_prob_cal = model.predict(X_cal).astype(float)

        # Find optimal threshold on calibration set (no leakage with val set)
        threshold = find_best_threshold(y_cal, y_prob_cal, criterion=threshold_criterion)

        # Re-fit on full training data for final evaluation
        model.fit(X_train, y_train, **_fit_kwargs(y_train))
    else:
        # Fit on full data without threshold tuning
        model.fit(X_train, y_train, **_fit_kwargs(y_train))

    # Predict on test set
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    if not skip_eval:
        evaluate(
            y_test, y_pred, y_prob,
            model_name=model_name, phase=phase, threshold=threshold,
            save=save_artifacts,
        )

    if save_artifacts:
        save_model_info(model, model_name, phase, pos_weight=pos_weight, threshold=threshold)
        _save_sklearn_model(model, model_name, phase)

    return model


def _save_sklearn_model(model, model_name, phase, fold_idx=None):
    """Save fitted sklearn model to disk for later SHAP analysis."""
    if not _JOBLIB_AVAILABLE:
        return
    save_dir = "models/saved"
    os.makedirs(save_dir, exist_ok=True)
    suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    base_name = model_name.replace(' ', '_')
    path = f"{save_dir}/{base_name}_{phase}{suffix}.joblib"
    joblib.dump(model, path)
    # Also save feature count for SHAP reference
    if hasattr(model, 'n_features_in_'):
        meta = {'n_features_in': model.n_features_in_}
        meta_path = f"{save_dir}/{base_name}_{phase}{suffix}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f)


def _make_json_serializable(obj):
    """Convert objects to JSON-serializable types."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    else:
        # Convert non-serializable objects (e.g., estimators) to their string representation
        return str(obj)


def save_model_info(model, model_name, phase, pos_weight=None, threshold=None):
    """Save model hyperparameters and info."""
    os.makedirs("results", exist_ok=True)

    info = {
        'model_type': model_name,
        'phase': phase,
        'model_params': _make_json_serializable(model.get_params())
    }

    # Add model-specific info
    if hasattr(model, 'n_features_in_'):
        info['n_features_in'] = model.n_features_in_

    if pos_weight is not None:
        info['pos_weight'] = pos_weight
        info['sample_weight_used'] = 'sample_weight' in inspect.signature(model.fit).parameters

    if threshold is not None:
        info['threshold'] = round(float(threshold), 4)

    with open(f"results/{model_name}_{phase}_info.json", "w") as f:
        json.dump(info, f, indent=2)
