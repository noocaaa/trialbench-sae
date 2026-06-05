"""
Aggregate nested CV results: compute mean ± std for each model-phase combination.
"""
import os
import json
import glob
import pandas as pd

SPECIAL_CASES = {
    "logistic_regression": "LogisticRegression",
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "svm": "SVM",
    "knn": "KNN",
    "mlp": "MLP",
    "cnn": "CNN",
    "rnn": "RNN",
    "transformer": "Transformer",
}

def load_fold_results(results_dir="results"):
    """Load all fold result files (e.g., rf_1_fold0.json)."""
    files = glob.glob(os.path.join(results_dir, "*_fold*.json"))
    records = []
    for f in files:
        basename = os.path.basename(f).replace(".json", "")
        # Parse: model_phase_foldN
        parts = basename.rsplit("_fold", 1)
        if len(parts) != 2:
            continue
        prefix, fold_str = parts
        fold = int(fold_str)

        # Parse model and phase from prefix
        # e.g., "random_forest_1" -> model="random_forest", phase="1"
        # e.g., "logistic_regression_1" -> model="logistic_regression", phase="1"
        prefix_parts = prefix.rsplit("_", 1)
        if len(prefix_parts) != 2:
            continue
        model_raw, phase = prefix_parts

        with open(f) as fp:
            data = json.load(fp)

        data["model"] = SPECIAL_CASES.get(model_raw, model_raw.replace("_", " ").title())
        data["phase"] = phase
        data["fold"] = fold
        records.append(data)

    return pd.DataFrame(records)


def aggregate(results_dir="results"):
    """Compute mean ± std per model-phase."""
    df = load_fold_results(results_dir)
    if df.empty:
        print("No fold results found.")
        return pd.DataFrame()

    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"]

    # Group by model and phase
    grouped = df.groupby(["model", "phase"])

    rows = []
    for (model, phase), group in grouped:
        row = {"model": model, "phase": phase, "n_folds": len(group)}
        for metric in metrics:
            if metric in group.columns:
                mean_val = group[metric].mean()
                std_val = group[metric].std()
                row[f"{metric}_mean"] = round(mean_val, 4)
                row[f"{metric}_std"] = round(std_val, 4)
                row[metric] = f"{mean_val:.4f} +/- {std_val:.4f}"
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_and_print(results_dir="results"):
    """Load, aggregate, and print results in a nice table."""
    df = aggregate(results_dir)
    if df.empty:
        return

    print("\n" + "="*80)
    print("  NESTED CV RESULTS: Mean ± Std over folds")
    print("="*80)

    # Pretty print
    display_cols = ["model", "phase", "accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    available = [c for c in display_cols if c in df.columns]

    for phase in sorted(df["phase"].unique(), key=lambda x: int(x)):
        print(f"\n--- Phase {phase} ---")
        phase_df = df[df["phase"] == phase].sort_values("roc_auc_mean", ascending=False)
        print(phase_df[available].to_string(index=False))

    # Save aggregated results
    df.to_json("results/aggregated_results.json", orient="records", indent=2)
    print("\n  Saved aggregated results -> results/aggregated_results.json")

    # Best model per phase by ROC-AUC
    print("\n" + "="*80)
    print("  BEST MODEL PER PHASE (by ROC-AUC)")
    print("="*80)
    for phase in sorted(df["phase"].unique(), key=lambda x: int(x)):
        phase_df = df[df["phase"] == phase]
        best = phase_df.loc[phase_df["roc_auc_mean"].idxmax()]
        print(f"  Phase {phase}: {best['model']} - ROC-AUC: {best['roc_auc']}")


if __name__ == "__main__":
    aggregate_and_print()
