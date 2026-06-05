"""
src/shap_analysis.py — Post-hoc SHAP explainability for sklearn models.

Run AFTER training:
    python src/shap_analysis.py --model XGBoost --phase 1
    python src/shap_analysis.py --model all --phase all

Outputs:
    results/shap/<model>_<phase>/
        ├── summary.png          (beeswarm: global feature importance)
        ├── bar.png              (mean |SHAP| per feature)
        ├── waterfall_pos.png    (example: trial predicted SAE)
        ├── waterfall_neg.png    (example: trial predicted no-SAE)
        └── top_features.json    (top 20 features ranked)

Requirements:
    pip install shap matplotlib joblib
"""

import argparse
import json
import os
import sys

import numpy as np

# ── Matplotlib backend (no GUI needed) ────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── SHAP ──────────────────────────────────────────────────────────
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    print("[ERROR] shap not installed. Run: pip install shap")
    sys.exit(1)

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    print("[ERROR] joblib not installed. Run: pip install joblib")
    sys.exit(1)

from src.data_loader import load_phase, TEXT_COLS, ZERO_VARIANCE_COLS, NULL_THRESHOLD, _DEFAULT_DATA
import pandas as pd


PHASES = ["1", "2", "3", "4"]

# Models that support TreeSHAP (fast) vs KernelSHAP (slow)
TREE_MODELS = {"XGBoost", "RandomForest", "LightGBM", "XGBoost+Text", "RandomForest+Text", "LightGBM+Text"}


def _get_real_feature_names(phase, for_tree=False, use_text=False, text_max_features=100):
    """
    Replicate preprocessing from data_loader._preprocess to get actual feature names.
    Returns list of feature names after all preprocessing steps.
    """
    base = f"{_DEFAULT_DATA}/Phase{phase}"
    X_train_raw = pd.read_csv(f"{base}/train_x.csv")

    # Step 1: drop ID and text columns
    drop_cols = ["Unnamed: 0"] + [c for c in TEXT_COLS if c in X_train_raw.columns]
    X = X_train_raw.drop(columns=drop_cols, errors="ignore")

    # Step 2: drop zero-variance columns
    zero_var = [c for c in ZERO_VARIANCE_COLS if c in X.columns]
    X = X.drop(columns=zero_var, errors="ignore")

    # Step 3: drop high-null columns
    null_rates = X.isnull().mean()
    high_null = null_rates[null_rates > NULL_THRESHOLD].index.tolist()
    X = X.drop(columns=high_null, errors="ignore")

    # Step 4: encode categorical columns
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        if for_tree:
            # Ordinal encoding keeps same column names
            pass
        else:
            # One-hot encoding expands columns
            X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)

    # Capture feature names before imputer/scaler (they don't change names)
    feature_names = list(X.columns)

    # Add text feature names if applicable
    if use_text:
        from src.text_features import extract_text_features
        X_test_raw = pd.read_csv(f"{base}/test_x.csv")
        _, X_text = extract_text_features(
            X_train_raw, X_test_raw, phase=phase, verbose=False, max_features=text_max_features
        )
        if X_text is not None and X_text.shape[1] > 0:
            feature_names += [f"tfidf_{i}" for i in range(X_text.shape[1])]

    return feature_names


def run_shap(model_name, phase, use_text=False, max_display=20, n_background=100):
    """
    Load a saved sklearn model and compute SHAP explanations.

    Parameters
    ----------
    model_name   : str  — e.g. "XGBoost", "RandomForest"
    phase        : str  — "1", "2", "3", "4"
    use_text     : bool — must match training config
    max_display  : int  — top N features to plot
    n_background : int  — background samples for KernelSHAP (non-tree models)
    """
    clean_name = model_name.replace(" ", "_").replace("+", "_")
    model_path = f"models/saved/{clean_name}_{phase}.joblib"

    if not os.path.exists(model_path):
        print(f"  [SKIP] Model not found: {model_path}")
        print("         Train first: python run_all.py --models ... --phases ...")
        return False

    print(f"\n  SHAP | {model_name} | Phase {phase}")
    print(f"  Loading model: {model_path}")

    # ── Load model ────────────────────────────────────────────────
    model = joblib.load(model_path)

    # ── Load data (same preprocessing as training) ────────────────
    is_tree = model_name.replace("+Text", "") in {"XGBoost", "RandomForest", "LightGBM"}
    X_train, X_test, y_train, y_test, _ = load_phase(
        phase, for_tree=is_tree, use_text=use_text, verbose=False
    )

    feature_names = _get_real_feature_names(phase, for_tree=is_tree, use_text=use_text)

    # ── Create SHAP explainer ─────────────────────────────────────
    # Tree models → TreeExplainer (fast, exact)
    # Linear models → LinearExplainer or KernelExplainer
    # Others → KernelExplainer (slower, approximate)
    base_name = model_name.replace("+Text", "")

    if base_name in TREE_MODELS:
        print(f"  Using TreeSHAP (exact, fast)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # TreeExplainer returns list for binary: [neg_class, pos_class]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
            expected_value = explainer.expected_value[1]
        else:
            expected_value = explainer.expected_value
    else:
        print(f"  Using KernelSHAP (approximate, slower)")
        # Subsample background for speed
        bg_idx = np.random.choice(len(X_train), min(n_background, len(X_train)), replace=False)
        background = X_train[bg_idx]
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test, nsamples=100, l1_reg="num_features(20)")
        # KernelExplainer returns list for binary
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            expected_value = explainer.expected_value[1]
        else:
            expected_value = explainer.expected_value

    # ── Guard: feature name count must match SHAP values ──────────
    if len(feature_names) != shap_values.shape[1]:
        print(f"  [WARNING] Feature name count ({len(feature_names)}) "
              f"!= SHAP values shape ({shap_values.shape[1]}). "
              f"Falling back to generic names.")
        feature_names = [f"feat_{i}" for i in range(shap_values.shape[1])]

    # ── Build Explanation object for plotting ─────────────────────
    explanation = shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=X_test,
        feature_names=feature_names,
    )

    # ── Output directory ──────────────────────────────────────────
    out_dir = f"results/shap/{clean_name}_{phase}"
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Summary plot (beeswarm) ────────────────────────────────
    plt.figure(figsize=(10, max(6, max_display * 0.4)))
    shap.summary_plot(explanation, max_display=max_display, show=False)
    plt.title(f"SHAP Summary — {model_name} | Phase {phase}")
    plt.tight_layout()
    summary_path = f"{out_dir}/summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {summary_path}")

    # ── 2. Bar plot (mean |SHAP|) ─────────────────────────────────
    plt.figure(figsize=(10, max(6, max_display * 0.4)))
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.title(f"SHAP Feature Importance — {model_name} | Phase {phase}")
    plt.tight_layout()
    bar_path = f"{out_dir}/bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {bar_path}")

    # ── 3. Waterfall: example positive prediction ─────────────────
    pos_idx = np.where(y_test == 1)[0]
    if len(pos_idx) > 0:
        idx = pos_idx[0]
        plt.figure(figsize=(12, 7))
        shap.plots.waterfall(explanation[idx], max_display=max_display, show=False)
        plt.title(f"SHAP Explanation — Trial {idx} (SAE = Yes)")
        plt.tight_layout()
        wf_pos_path = f"{out_dir}/waterfall_pos.png"
        plt.savefig(wf_pos_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved -> {wf_pos_path}")

    # ── 4. Waterfall: example negative prediction ─────────────────
    neg_idx = np.where(y_test == 0)[0]
    if len(neg_idx) > 0:
        idx = neg_idx[0]
        plt.figure(figsize=(12, 7))
        shap.plots.waterfall(explanation[idx], max_display=max_display, show=False)
        plt.title(f"SHAP Explanation — Trial {idx} (SAE = No)")
        plt.tight_layout()
        wf_neg_path = f"{out_dir}/waterfall_neg.png"
        plt.savefig(wf_neg_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved -> {wf_neg_path}")

    # ── 5. Top features JSON ──────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
    top_features = {
        feature_names[i]: round(float(mean_abs_shap[i]), 6)
        for i in top_indices
    }
    json_path = f"{out_dir}/top_features.json"
    with open(json_path, "w") as f:
        json.dump(top_features, f, indent=2)
    print(f"  Saved -> {json_path}")

    return True


def main(models, phases, use_text=False, max_display=20):
    """Run SHAP for multiple model-phase combinations."""
    print("\n" + "=" * 60)
    print("  SHAP ANALYSIS — Post-hoc Explainability")
    print("=" * 60)

    total = 0
    success = 0
    for phase in phases:
        for model_name in models:
            total += 1
            if run_shap(model_name, phase, use_text=use_text, max_display=max_display):
                success += 1

    print(f"\n{'=' * 60}")
    print(f"  Done: {success}/{total} model-phase combinations explained")
    print(f"  Outputs in: results/shap/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP analysis for trained sklearn models")
    parser.add_argument(
        "--model", nargs="+", required=True,
        help='Model name(s), e.g. "XGBoost" "LightGBM" or "all"'
    )
    parser.add_argument(
        "--phase", nargs="+", required=True,
        help='Phase(s): 1, 2, 3, 4 or "all"'
    )
    parser.add_argument(
        "--use-text", action="store_true",
        help="Must match the --use-text flag used during training"
    )
    parser.add_argument(
        "--max-display", type=int, default=20,
        help="Max features to show in plots (default: 20)"
    )
    args = parser.parse_args()

    # Handle "all"
    models = args.model
    if "all" in models:
        models = ["LogisticRegression", "RandomForest", "XGBoost",
                  "LightGBM", "SVM", "KNN"]

    phases = args.phase
    if "all" in phases:
        phases = PHASES

    main(models, phases, use_text=args.use_text, max_display=args.max_display)
