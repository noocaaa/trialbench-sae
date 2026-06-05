"""
src/shap_analysis_dl.py — Post-hoc SHAP for Deep Learning models (MLP, CNN, RNN, FT-Transformer).

Run AFTER training:
    python src/shap_analysis_dl.py --model MLP --phase 1
    python src/shap_analysis_dl.py --model all --phase all

Outputs:
    results/shap/<model>_<phase>/
        ├── summary.png          (beeswarm)
        ├── bar.png              (mean |SHAP|)
        ├── waterfall_pos.png    (example: predicted SAE)
        ├── waterfall_neg.png    (example: predicted no-SAE)
        └── top_features.json

Requirements:
    pip install shap matplotlib torch
"""

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    print("[ERROR] shap not installed. Run: pip install shap")
    sys.exit(1)

import torch
import torch.nn as nn
from src.data_loader import load_phase
from src.utils import set_seed

set_seed(42)

PHASES = ["1", "2", "3", "4"]
DL_MODELS = ["MLP", "CNN", "RNN", "FT_Transformer"]


def _load_dl_model(model_name, phase, use_text=False):
    """
    Reconstruct model architecture, load checkpoint, and return model + data.
    """
    from models.cnn import CNN
    from models.mlp import MLP
    from models.rnn import RNN
    from models.ft_transformer import FTTransformer

    # Load data to get input_dim
    X_train, X_test, y_train, y_test, pos_weight = load_phase(
        phase, for_tree=False, use_text=use_text, verbose=False
    )
    input_dim = X_train.shape[1]

    # Reconstruct model with correct class names and signatures
    if model_name == "MLP":
        model = MLP(input_dim=input_dim)
    elif model_name == "CNN":
        model = CNN(input_dim=input_dim)
    elif model_name == "RNN":
        model = RNN(input_dim=input_dim)
    elif model_name == "FT_Transformer":
        model = FTTransformer(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown DL model: {model_name}")

    # Load checkpoint
    clean_name = model_name.replace(" ", "_")
    ckpt_path = f"models/checkpoints/{clean_name}_{phase}_best.pt"
    if not os.path.exists(ckpt_path):
        return None, None, None, None, None

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    return model, X_train, X_test, y_train, y_test


def run_shap_dl(model_name, phase, use_text=False, max_display=20, n_background=100):
    """SHAP analysis for a single DL model-phase combination."""
    print(f"\n  SHAP-DL | {model_name} | Phase {phase}")

    result = _load_dl_model(model_name, phase, use_text=use_text)
    if result[0] is None:
        print(f"  [SKIP] Checkpoint not found: models/checkpoints/{model_name}_{phase}_best.pt")
        return False

    model, X_train, X_test, y_train, y_test = result
    n_features = X_test.shape[1]
    feature_names = [f"feat_{i}" for i in range(n_features)]

    # Convert to torch tensors
    X_test_t = torch.FloatTensor(X_test)

    # Background for DeepExplainer
    bg_idx = np.random.choice(len(X_train), min(n_background, len(X_train)), replace=False)
    background = torch.FloatTensor(X_train[bg_idx])

    # ── DeepExplainer ─────────────────────────────────────────────
    print(f"  Using DeepExplainer (background n={len(background)})")
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_test_t)

    # DeepExplainer returns list for binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        expected_value = explainer.expected_value

    # ── Build Explanation ─────────────────────────────────────────
    explanation = shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=X_test,
        feature_names=feature_names,
    )

    # ── Output directory ──────────────────────────────────────────
    clean_name = model_name.replace(" ", "_")
    out_dir = f"results/shap/{clean_name}_{phase}"
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Summary plot ───────────────────────────────────────────
    plt.figure(figsize=(10, max(6, max_display * 0.4)))
    shap.summary_plot(explanation, max_display=max_display, show=False)
    plt.title(f"SHAP Summary — {model_name} | Phase {phase}")
    plt.tight_layout()
    summary_path = f"{out_dir}/summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {summary_path}")

    # ── 2. Bar plot ───────────────────────────────────────────────
    plt.figure(figsize=(10, max(6, max_display * 0.4)))
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.title(f"SHAP Feature Importance — {model_name} | Phase {phase}")
    plt.tight_layout()
    bar_path = f"{out_dir}/bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {bar_path}")

    # ── 3. Waterfall: positive example ────────────────────────────
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

    # ── 4. Waterfall: negative example ────────────────────────────
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
    """Run SHAP-DL for multiple model-phase combinations."""
    print("\n" + "=" * 60)
    print("  SHAP ANALYSIS — Deep Learning Models")
    print("=" * 60)

    total = 0
    success = 0
    for phase in phases:
        for model_name in models:
            total += 1
            if run_shap_dl(model_name, phase, use_text=use_text, max_display=max_display):
                success += 1

    print(f"\n{'=' * 60}")
    print(f"  Done: {success}/{total} model-phase combinations explained")
    print(f"  Outputs in: results/shap/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP analysis for trained DL models")
    parser.add_argument("--model", nargs="+", required=True,
                        help='Model name(s): MLP, CNN, RNN, FT_Transformer, or "all"')
    parser.add_argument("--phase", nargs="+", required=True,
                        help='Phase(s): 1, 2, 3, 4 or "all"')
    parser.add_argument("--use-text", action="store_true",
                        help="Must match the --use-text flag used during training")
    parser.add_argument("--max-display", type=int, default=20,
                        help="Max features to show in plots (default: 20)")
    args = parser.parse_args()

    models = args.model
    if "all" in models:
        models = DL_MODELS

    phases = args.phase
    if "all" in phases:
        phases = PHASES

    main(models, phases, use_text=args.use_text, max_display=args.max_display)
