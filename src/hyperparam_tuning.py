"""
src/hyperparam_tuning.py — Hyperparameter tuning for ALL models.

Design:
- Sklearn models: Optuna Bayesian optimization (fast, efficient)
- Small DL models (MLP, CNN, RNN): Limited grid search (2 values x 2 params)
- Large DL models (Transformer, FT-Transformer): Fixed hyperparameters

Usage:
    from src.hyperparam_tuning import tune_model
    best_params = tune_model(
        model_name="xgboost",
        model_fn=factory_fn,
        X_train=X_train, y_train=y_train,
        is_dl=False,
        n_trials=20,  # for Optuna
    )
"""

import numpy as np
import warnings

# Suppress Optuna's verbose logging
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.mlflow_tracker import tracker


# ── Optuna search spaces for sklearn models ───────────────────────

def _suggest_logreg(trial):
    return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}

def _suggest_rf(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }

def _suggest_xgboost(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

def _suggest_lightgbm(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }

def _suggest_svm(trial):
    return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}

def _suggest_knn(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 21),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
    }


OPTUNA_SPACES = {
    "logistic_regression": _suggest_logreg,
    "random_forest": _suggest_rf,
    "xgboost": _suggest_xgboost,
    "lightgbm": _suggest_lightgbm,
    "svm": _suggest_svm,
    "knn": _suggest_knn,
}


# ── Small grid for DL models (fast to evaluate) ───────────────────
# Each config: 20 epochs max with early stopping (patience=3)
# Total cost: 4-6 configs × ~10 epochs ≈ 40-60 epochs per model

DL_GRIDS = {
    "mlp": {
        "hidden_dim": [128, 256],
        "dropout": [0.3],
    },
    "cnn": {
        "conv_channels": [32, 64],
        "dropout": [0.3],
    },
    "rnn": {
        "hidden_dim": [64, 128],
        "dropout": [0.3],
    },
}

# Large DL models: too slow to tune, use defaults
DL_FIXED = {
    "transformer": {"embed_dim": 64, "num_heads": 4, "num_layers": 2},
    "ft_transformer": {"embed_dim": 32, "num_heads": 4, "num_layers": 2},
}


# ── Core tuning functions ─────────────────────────────────────────

def _cv_score(model_fn, params, X, y, n_folds=3, random_state=42):
    """Cross-validate a model with given params, return mean ROC-AUC."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scores = []
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = model_fn(**params)
        model.fit(X_tr, y_tr)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
        else:
            y_prob = model.predict(X_val).astype(float)
        scores.append(roc_auc_score(y_val, y_prob))
    return np.mean(scores)


def tune_sklearn_optuna(model_name, model_fn, X_train, y_train, n_trials=20, n_folds=3, verbose=False):
    """
    Bayesian hyperparameter tuning for sklearn models using Optuna.

    Parameters
    ----------
    model_name : str — e.g. "xgboost", "random_forest"
    model_fn   : callable — factory that returns a fresh model
    X_train, y_train : training data
    n_trials   : int — number of Optuna trials (default: 20)
    n_folds    : int — CV folds for evaluation
    verbose    : bool

    Returns
    -------
    best_params : dict
    """
    key = model_name.lower().replace("+", "_").replace("-", "_").replace(" ", "_")
    aliases = {
        "logisticregression": "logistic_regression",
        "randomforest": "random_forest",
        "logistic_regression_text": "logistic_regression",
        "random_forest_text": "random_forest",
        "xgboost_text": "xgboost",
        "lightgbm_text": "lightgbm",
        "svm_text": "svm",
        "knn_text": "knn",
    }
    key = aliases.get(key, key)

    suggest_fn = OPTUNA_SPACES.get(key)
    if suggest_fn is None:
        if verbose:
            print(f"    No Optuna space for {model_name}, using defaults")
        return {}

    def objective(trial):
        params = suggest_fn(trial)
        return _cv_score(model_fn, params, X_train, y_train, n_folds=n_folds, random_state=42)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best_params = study.best_params

    if verbose:
        print(f"    Optuna: {n_trials} trials, best ROC-AUC={study.best_value:.4f}")
        print(f"    Best params: {best_params}")

    # ── Log to MLflow ──
    if tracker.enabled and tracker.get_run_id() is not None:
        tracker.log_param("optuna_n_trials", n_trials)
        tracker.log_param("optuna_best_value", study.best_value)
        tracker.log_param("optuna_n_completed", len(study.trials))
        # Log all trials as a JSON artifact
        trials_data = []
        for t in study.trials:
            trials_data.append({
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
            })
        tracker.log_json_artifact(trials_data, "optuna_trials.json")

    return best_params


def tune_dl_grid(model_name, model_fn, X_train, y_train, X_val, y_val, pos_weight, verbose=False):
    """
    Limited grid search for small DL models.
    Each config is trained with early stopping (max 20 epochs).
    Uses ROC-AUC for selection (better for imbalanced data).

    Returns best hyperparameters (not the model itself).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    key = model_name.lower().replace("+", "_").replace("-", "_").replace(" ", "_")
    aliases = {
        "mlp_text": "mlp",
        "cnn_text": "cnn",
        "rnn_text": "rnn",
    }
    key = aliases.get(key, key)

    grid = DL_GRIDS.get(key)
    if grid is None:
        fixed = DL_FIXED.get(key, {})
        if verbose:
            print(f"    Using fixed hparams for {model_name}: {fixed}")
        return fixed

    from itertools import product
    keys = list(grid.keys())
    values = list(grid.values())
    all_configs = list(product(*values))

    if verbose:
        print(f"    DL grid: {len(all_configs)} configs × ~10 epochs for {model_name}")

    best_auc = 0.0
    best_params = None

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    for combo in all_configs:
        params = dict(zip(keys, combo))
        try:
            model = model_fn(input_dim=X_train.shape[1], **params).to("cpu")
        except TypeError:
            if verbose:
                print(f"    Model {model_name} doesn't accept tuning params, skipping")
            return {}

        batch_size = min(64, len(X_train))
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
        )

        best_loss = float("inf")
        patience = 3
        no_improve = 0

        for epoch in range(10):
            model.train()
            for X_b, y_b in train_loader:
                if X_b.size(0) < 2:
                    continue
                optimizer.zero_grad()
                loss = criterion(model(X_b).squeeze(), y_b)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t).squeeze()
                val_loss = criterion(val_logits, y_val_t).item()

            if val_loss < best_loss:
                best_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Evaluate with ROC-AUC
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(X_val_t)).cpu().numpy()
        auc = roc_auc_score(y_val, val_prob)

        if auc > best_auc:
            best_auc = auc
            best_params = params

    if verbose:
        print(f"    Best DL params: {best_params} (val_auc={best_auc:.4f})")

    # ── Log to MLflow ──
    if tracker.enabled and tracker.get_run_id() is not None:
        tracker.log_param("dl_grid_n_configs", len(all_configs))
        tracker.log_param("dl_grid_best_auc", best_auc)
        if best_params:
            tracker.log_tuning_results(best_params, tuning_method="grid_search")

    return best_params


def tune_model(model_name, model_fn, X_train, y_train, is_dl=False, X_val=None, y_val=None, pos_weight=None, n_trials=20, verbose=False):
    """
    Universal tuning function: picks the right method based on model type.

    Parameters
    ----------
    model_name : str — display name, e.g. "XGBoost", "MLP"
    model_fn   : callable — factory function
    X_train, y_train : training data
    is_dl      : bool — True for PyTorch models
    X_val, y_val : validation data (required for DL tuning)
    pos_weight : float — class weight (required for DL tuning)
    n_trials   : int — Optuna trials (sklearn only)
    verbose    : bool

    Returns
    -------
    best_params : dict — hyperparameters to pass to model_fn
    """
    if is_dl:
        if X_val is None or y_val is None:
            if verbose:
                print(f"    No val set for DL tuning, using defaults")
            return {}
        return tune_dl_grid(model_name, model_fn, X_train, y_train, X_val, y_val, pos_weight, verbose)
    else:
        return tune_sklearn_optuna(model_name, model_fn, X_train, y_train, n_trials=n_trials, verbose=verbose)
