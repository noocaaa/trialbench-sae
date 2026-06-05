from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score
)
import json, os
import numpy as np

from src.mlflow_tracker import tracker


def evaluate(y_true, y_pred, y_prob, model_name, phase, save=True, threshold=None, save_path=None, save_predictions=True):
    """
    Evaluate a binary classifier and optionally save results to disk.

    Parameters
    ----------
    y_true     : array of int (0/1)  — ground truth labels from the test set
    y_pred     : array of int (0/1)  — hard binary predictions
    y_prob     : array of float      — predicted probability for the positive class (label=1)
    model_name : str                 — model label, e.g. "CNN", "Random Forest"
    phase      : str                 — trial phase, e.g. "1", "2", "3", "4"
    save       : bool                — if True, saves results to results/<model_name>_<phase>.json
    threshold  : float or None       — classification threshold used (saved for reference)
    save_predictions : bool          — if False, omit y_pred/y_test from JSON (smaller files)

    Returns
    -------
    dict with keys: model, phase, accuracy, f1, precision, recall, roc_auc, pr_auc,
                    threshold, y_pred, y_test (if save_predictions=True)
    """
    # Guard against single-class test sets
    unique_labels = np.unique(y_true)
    if len(unique_labels) > 1:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc  = average_precision_score(y_true, y_prob)
    else:
        roc_auc = np.nan
        pr_auc  = np.nan

    metrics = {
        "model":     model_name,
        "phase":     phase,
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc,
        "pr_auc":    pr_auc,
    }

    if save_predictions:
        # ── saved for real confusion matrices in sanity_check ─────
        metrics["y_pred"] = np.array(y_pred).tolist()
        metrics["y_test"] = np.array(y_true).tolist()

    if threshold is not None:
        metrics["threshold"] = round(float(threshold), 4)

    print(f"\n  === {model_name} | Phase {phase} ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:12s}: {v:.4f}")

    # Determine file path
    if save_path is not None:
        fname = save_path
    else:
        fname = f"results/{model_name.replace(' ', '_')}_{phase}.json"

    if save:
        os.makedirs("results", exist_ok=True)
        with open(fname, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved -> {fname}")

    # ── Log to MLflow (simple mode) ──
    if tracker.enabled and tracker.get_run_id() is not None:
        scalar_metrics = {k: v for k, v in metrics.items()
                          if k in ("accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc")}
        tracker.log_metrics(scalar_metrics)
        if threshold is not None:
            tracker.log_param("threshold", threshold)
        if save and os.path.exists(fname):
            tracker.log_artifact(fname)

    return metrics
