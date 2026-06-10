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
    Vectorized implementation for speed.

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
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    # Vectorized: compare all thresholds at once
    # shape: (n_thresholds, n_samples)
    y_pred = (y_prob >= thresholds[:, None]).astype(int)
    y_true_expanded = y_true[None, :]  # shape: (1, n_samples)
    
    if criterion == "f1":
        # Vectorized TP, FP, FN for all thresholds
        tp = ((y_pred == 1) & (y_true_expanded == 1)).sum(axis=1)
        fp = ((y_pred == 1) & (y_true_expanded == 0)).sum(axis=1)
        fn = ((y_pred == 0) & (y_true_expanded == 1)).sum(axis=1)
        
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
        
        f1_scores = np.divide(2 * precision * recall, precision + recall, 
                              out=np.zeros_like(precision), where=(precision + recall) > 0)
        
        best_idx = np.argmax(f1_scores)
        return float(thresholds[best_idx])
        
    elif criterion == "youden":
        tp = ((y_pred == 1) & (y_true_expanded == 1)).sum(axis=1)
        tn = ((y_pred == 0) & (y_true_expanded == 0)).sum(axis=1)
        fp = ((y_pred == 1) & (y_true_expanded == 0)).sum(axis=1)
        fn = ((y_pred == 0) & (y_true_expanded == 1)).sum(axis=1)
        
        tpr = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
        tnr = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) > 0)
        
        youden_scores = tpr + tnr - 1
        best_idx = np.argmax(youden_scores)
        return float(thresholds[best_idx])
        
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Use 'f1' or 'youden'.")


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
