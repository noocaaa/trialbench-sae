import os
import json
import glob
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def set_seed(seed=42):
    """Set random seeds for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def find_best_threshold(y_true, y_prob, criterion="f1"):
    """
    Find the optimal classification threshold by grid search.

    Parameters
    ----------
    y_true    : array of int (0/1)  — ground truth labels
    y_prob    : array of float      — predicted probability for positive class
    criterion : str                 — "f1" or "youden" (default: "f1")

    Returns
    -------
    float — best threshold in [0.01, 0.99]
    """
    thresholds = np.arange(0.005, 1.0, 0.005)
    best_score = -1
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        if criterion == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif criterion == "youden":
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = tpr + tnr - 1
        else:
            raise ValueError(f"Unknown criterion: {criterion}. Use 'f1' or 'youden'.")

        if score > best_score:
            best_score = score
            best_threshold = t

    return float(best_threshold)


def clear_results(results_dir="results"):
    """
    Delete all JSON and HTML result files in the results folder.

    Parameters
    ----------
    results_dir : str — path to the results folder (default: "results")

    Example
    -------
    clear_results()
    """
    files = glob.glob(os.path.join(results_dir, "*.json")) + glob.glob(os.path.join(results_dir, "*.html"))
    if not files:
        print("No results to clear.")
        return
    for f in files:
        os.remove(f)
    print(f"Cleared {len(files)} result(s) from '{results_dir}/'.")


def load_results(results_dir="results"):
    """
    Load all saved JSON result files into a pandas DataFrame.

    Parameters
    ----------
    results_dir : str — path to the results folder (default: "results")

    Returns
    -------
    pd.DataFrame — one row per model/phase, sorted by F1 descending

    Example
    -------
    df = load_results()
    print(df)
    """
    # Skip loss curve and info files — only load metric result files
    files = [f for f in glob.glob(os.path.join(results_dir, "*.json"))
             if not os.path.basename(f).startswith("loss_") and not f.endswith("_info.json")]
    if not files:
        print("No results found.")
        return pd.DataFrame()
    records = []
    for f in files:
        with open(f) as fp:
            records.append(json.load(fp))
    df = pd.DataFrame(records)
    if "f1" in df.columns:
        df = df.sort_values("f1", ascending=False).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def print_results_table(results_dir="results"):
    """
    Print a formatted comparison table of all saved results.

    Parameters
    ----------
    results_dir : str — path to the results folder (default: "results")

    Example
    -------
    print_results_table()
    """
    df = load_results(results_dir)
    if df.empty:
        return
    float_cols = ["accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    print("\n" + df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
        columns=["model", "phase"] + float_cols
    ))


def get_best_model(metric="f1", results_dir="results"):
    """
    Return the name of the best performing model for a given metric.

    Parameters
    ----------
    metric      : str — metric to rank by, e.g. "f1", "roc_auc", "pr_auc" (default: "f1")
    results_dir : str — path to the results folder (default: "results")

    Returns
    -------
    str — model name of the best result

    Example
    -------
    best = get_best_model(metric="roc_auc")
    print(best)
    """
    df = load_results(results_dir)
    if df.empty or df[metric].isna().all():
        print(f"No valid results found for metric '{metric}'.")
        return None
    best = df.loc[df[metric].idxmax()]
    print(f"Best model by {metric}: {best['model']} (phase {best['phase']}) — {metric}: {best[metric]:.4f}")
    return best["model"]
