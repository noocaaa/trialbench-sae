import os
import json
import glob
import pandas as pd


def clear_results(results_dir="results"):
    """
    Delete all JSON files in the results folder.

    Parameters
    ----------
    results_dir : str — path to the results folder (default: "results")

    Example
    -------
    clear_results()
    """
    # Skip loss curve files — only load metric result files
    files = [f for f in glob.glob(os.path.join(results_dir, "*.json")) and glob.glob(os.path.join(results_dir, "*.html"))]
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
    # Skip loss curve files — only load metric result files
    files = [f for f in glob.glob(os.path.join(results_dir, "*.json"))
             if not os.path.basename(f).startswith("loss_")]
    if not files:
        print("No results found.")
        return pd.DataFrame()
    records = []
    for f in files:
        with open(f) as fp:
            records.append(json.load(fp))
    df = pd.DataFrame(records).sort_values("f1", ascending=False).reset_index(drop=True)
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