"""
sanity_check.py - Automated sanity checks for SAE Prediction models
Run this after training all models to verify results are trustworthy.

Usage:
    python sanity_check.py
"""

import json, glob, os, sys

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from src.data_loader import load_phase
from src.evaluate import evaluate
from src.utils import load_results as _load_results_utils

PHASES  = ["1", "2", "3", "4"]
RESULTS = "results"


# ── helpers ───────────────────────────────────────────────────────
def load_results(exclude_loss=True, exclude_dummy=False, exclude_info=True):
    """Load result files, delegating filtering to the caller."""
    files = glob.glob(os.path.join(RESULTS, "*.json"))
    if exclude_loss:
        files = [f for f in files if not os.path.basename(f).startswith("loss_")]
    if exclude_dummy:
        files = [f for f in files if "Dummy" not in os.path.basename(f)]
    if exclude_info:
        files = [f for f in files if not f.endswith("_info.json")]
    return sorted(files)


# ── 1. Dataset statistics ──────────────────────────────────────────
def check_dataset():
    print("\n" + "="*55)
    print("  CHECK 1 - Dataset Statistics & Class Balance")
    print("="*55)

    stats = []
    for phase in PHASES:
        try:
            X_train, X_test, y_train, y_test, pos_weight = load_phase(phase)
            n         = len(y_test)
            pos       = int(y_test.sum())
            neg       = n - pos
            ratio     = pos / n
            dummy_acc = max(ratio, 1 - ratio)
            stats.append({
                "phase": phase, "test_n": n,
                "positives": pos, "negatives": neg,
                "pos_ratio": ratio, "dummy_acc": dummy_acc,
                "pos_weight": pos_weight,
            })
            print(f"\n  Phase {phase}:")
            print(f"    Test samples    : {n}")
            print(f"    Positives (SAE) : {pos}  ({ratio*100:.1f}%)")
            print(f"    Negatives       : {neg}  ({(1-ratio)*100:.1f}%)")
            print(f"    pos_weight      : {pos_weight:.2f}")
            print(f"    Dummy accuracy  : {dummy_acc*100:.1f}%  <- models must beat this")
        except Exception as e:
            print(f"  Phase {phase}: ERROR - {e}")

    return pd.DataFrame(stats)


# ── 2. Metric consistency check ────────────────────────────────────
def check_metric_consistency():
    print("\n" + "="*55)
    print("  CHECK 2 - Metric Consistency")
    print("="*55)

    files = load_results(exclude_dummy=True)
    if not files:
        print("  No results found - run the models first.")
        return

    all_ok = True
    for f in files:
        with open(f) as fp:
            r = json.load(fp)
        name = f"{r['model']} Phase {r['phase']}"
        issues = []

        # F1 should be harmonic mean of precision and recall
        p, rec, f1 = r.get("precision", 0), r.get("recall", 0), r.get("f1", 0)
        if p + rec > 0:
            expected_f1 = 2 * p * rec / (p + rec)
            if abs(expected_f1 - f1) > 0.01:
                issues.append(f"F1 mismatch: got {f1:.4f}, expected {expected_f1:.4f}")

        # ROC-AUC must be > 0.5
        roc = r.get("roc_auc", None)
        if roc is not None and not np.isnan(roc) and roc < 0.5:
            issues.append(f"ROC-AUC = {roc:.4f} < 0.5 (worse than random)")

        # All metrics must be in [0, 1]
        for met in ["accuracy", "f1", "precision", "recall", "pr_auc"]:
            v = r.get(met, None)
            if v is not None and not np.isnan(v) and not (0 <= v <= 1):
                issues.append(f"{met} = {v:.4f} out of [0,1] range")

        if issues:
            all_ok = False
            print(f"\n  ERROR {name}:")
            for issue in issues:
                print(f"     -> {issue}")
        else:
            print(f"  OK {name}: all metrics consistent")

    if all_ok:
        print("\n  All metrics passed consistency checks!")


# ── 3. Dummy classifier baseline ──────────────────────────────────
def check_dummy_baseline():
    print("\n" + "="*55)
    print("  CHECK 3 - Dummy Classifier Baseline")
    print("="*55)
    print("  Models must beat these scores to be meaningful.\n")

    dummy_results = []
    for phase in PHASES:
        try:
            X_train, X_test, y_train, y_test, _ = load_phase(phase)
            d = DummyClassifier(strategy="most_frequent")
            d.fit(X_train, y_train)
            y_pred = d.predict(X_test)
            y_prob = d.predict_proba(X_test)[:, 1]
            m = evaluate(y_test, y_pred, y_prob,
                         model_name="Dummy", phase=phase, save=False)
            dummy_results.append(m)
        except Exception as e:
            print(f"  Phase {phase}: ERROR - {e}")

    return pd.DataFrame(dummy_results)


# ── 4. Model vs Dummy comparison ───────────────────────────────────
def check_vs_dummy(dummy_df):
    if dummy_df.empty:
        return
    print("\n" + "="*55)
    print("  CHECK 4 - Models vs Dummy Baseline")
    print("="*55)

    files = load_results(exclude_dummy=True)
    real_results = []
    for f in files:
        with open(f) as fp:
            real_results.append(json.load(fp))

    for r in sorted(real_results, key=lambda x: (x["model"], x["phase"])):
        phase     = str(r["phase"])
        dummy_row = dummy_df[dummy_df["phase"] == phase]
        if dummy_row.empty:
            continue

        beats_f1  = r["f1"] > float(dummy_row["f1"].values[0])
        beats_roc = r.get("roc_auc", 0) > float(
            dummy_row.get("roc_auc", pd.Series([0])).values[0])

        icon = "OK" if (beats_f1 and beats_roc) else "ERROR"
        print(f"\n  {icon} {r['model']} Phase {phase}")
        print(f"     F1      : {r['f1']:.4f}  vs dummy {float(dummy_row['f1'].values[0]):.4f}"
              f"  {'^ better' if beats_f1 else 'v WORSE'}")
        roc_val   = r.get("roc_auc")
        dummy_roc = float(dummy_row.get("roc_auc", pd.Series([0])).values[0])
        if roc_val  is not None and not np.isnan(roc_val):
            print(f"     ROC-AUC : {roc_val:.4f}  vs dummy {dummy_roc:.4f}"
                  f"  {'^ better' if beats_roc else 'v WORSE'}")


# ── 5. Confusion matrices - using REAL model predictions ──────────
def check_confusion_matrices():
    print("\n" + "="*55)
    print("  CHECK 5 - Confusion Matrices (real model predictions)")
    print("="*55)

    files = load_results(exclude_dummy=True)
    if not files:
        print("  No results found - run the models first.")
        return

    # Check if y_pred is saved in results
    with open(files[0]) as fp:
        sample = json.load(fp)
    if "y_pred" not in sample:
        print("  CAREFUL!  y_pred not found in results.")
        print("  -> Re-run models with the updated evaluate.py to save predictions.")
        return

    os.makedirs(RESULTS, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Group by model
        models = {}
        for f in files:
            with open(f) as fp:
                r = json.load(fp)
            models.setdefault(r["model"], {})[str(r["phase"])] = r

        n_models = len(models)
        fig = make_subplots(
            rows=n_models, cols=len(PHASES),
            subplot_titles=[f"{m} - Phase {p}"
                            for m in models for p in PHASES],
        )

        labels = ["No SAE", "SAE"]
        for row, (model_name, phases) in enumerate(models.items(), 1):
            for col, phase in enumerate(PHASES, 1):
                r = phases.get(phase)
                if not r or "y_pred" not in r:
                    continue

                y_pred = np.array(r["y_pred"])
                y_test = np.array(r["y_test"])
                cm     = confusion_matrix(y_test, y_pred, labels=[0, 1])
                pct    = cm / cm.sum() * 100

                text = [[f"{cm[ri][ci]}<br>({pct[ri][ci]:.1f}%)"
                         for ci in range(2)] for ri in range(2)]

                fig.add_trace(go.Heatmap(
                    z=cm, x=labels, y=labels,
                    text=text, texttemplate="%{text}",
                    colorscale="Blues", showscale=False,
                    textfont=dict(size=11),
                ), row=row, col=col)

                tn, fp, fn, tp = cm.ravel()
                print(f"  {model_name} Phase {phase}: "
                      f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
                print(f"    -> False Negatives (missed SAEs): {fn}"
                      f"  <- dangerous in medical context")

        fig.update_layout(
            title="Confusion Matrices - Real Model Predictions",
            paper_bgcolor="#0f0f17", plot_bgcolor="#1a1a2e",
            font=dict(color="#ccccee"),
            height=300 * n_models,
        )
        out = os.path.join(RESULTS, "confusion_matrices.html")
        fig.write_html(out)
        print(f"\n  Saved -> {out}")

    except Exception as e:
        print(f"  Could not generate plot: {e}")


# ── 6. Loss curve check ────────────────────────────────────────────
def check_loss_curves():
    print("\n" + "="*55)
    print("  CHECK 6 - Loss Curves")
    print("="*55)

    files = glob.glob(os.path.join(RESULTS, "loss_*.json"))
    if not files:
        print("  No loss curves saved yet - run your models first.")
        return

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        models_found = {}
        for f in sorted(files):
            name  = os.path.basename(f).replace("loss_", "").replace(".json", "")
            parts = name.rsplit("_", 1)
            model, phase = (parts[0], parts[1]) if len(parts) == 2 else (name, "?")
            with open(f) as fp:
                models_found.setdefault(model, {})[phase] = json.load(fp)

        PHASE_COLORS = {"1": "#7c6af7", "2": "#f7916a", "3": "#6af7c8", "4": "#f7e06a"}
        n_models = len(models_found)
        fig = make_subplots(rows=1, cols=n_models,
                            subplot_titles=list(models_found.keys()))

        for col, (model, phases) in enumerate(models_found.items(), 1):
            for phase, losses in sorted(phases.items()):
                epochs = list(range(1, len(losses) + 1))
                # Handle both old format (list of floats) and new format (list of dicts)
                if losses and isinstance(losses[0], dict):
                    train_losses = [entry["train_loss"] for entry in losses]
                    val_losses = [entry["val_loss"] for entry in losses]
                else:
                    train_losses = losses
                    val_losses = None
                fig.add_trace(go.Scatter(
                    x=epochs, y=train_losses,
                    mode="lines+markers",
                    name=f"Phase {phase}",
                    legendgroup=f"Phase {phase}",
                    showlegend=(col == 1),
                    line=dict(color=PHASE_COLORS.get(phase, "#ffffff"), width=2),
                    marker=dict(size=4),
                ), row=1, col=col)

            for phase, losses in sorted(phases.items()):
                if losses and isinstance(losses[0], dict):
                    tl = [entry["train_loss"] for entry in losses]
                    vl = [entry["val_loss"] for entry in losses]
                    drop = tl[0] - tl[-1]
                    print(f"  {model} Phase {phase}: "
                          f"train {tl[0]:.4f} -> {tl[-1]:.4f}  "
                          f"val {vl[0]:.4f} -> {vl[-1]:.4f}  (drop {drop:.4f})")
                else:
                    drop = losses[0] - losses[-1]
                    print(f"  {model} Phase {phase}: "
                          f"{losses[0]:.4f} -> {losses[-1]:.4f}  (drop {drop:.4f})")

        fig.update_layout(
            title="Training Loss Curves - all models and phases",
            paper_bgcolor="#0f0f17", plot_bgcolor="#1a1a2e",
            font=dict(color="#ccccee"),
            legend=dict(bgcolor="#1a1a2e", bordercolor="#333355"),
        )
        for i in range(1, n_models + 1):
            fig.update_xaxes(title_text="Epoch", gridcolor="#333355", row=1, col=i)
            fig.update_yaxes(title_text="Loss",  gridcolor="#333355", row=1, col=i)

        out = os.path.join(RESULTS, "loss_curves.html")
        fig.write_html(out)
        print(f"\n  Saved -> {out}")

        for model, phases in models_found.items():
            for phase, losses in phases.items():
                if losses and isinstance(losses[0], dict):
                    train_losses = [entry["train_loss"] for entry in losses]
                else:
                    train_losses = losses
                if train_losses[-1] >= train_losses[0] * 0.95:
                    print(f"  CAREFUL - {model} Phase {phase}: loss barely decreased - model may not be learning")
                else:
                    print(f"    OK    - {model} Phase {phase}: loss converged normally")

    except Exception as e:
        print(f"  Could not generate plot: {e}")


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "*"*55)
    print("  SAE PREDICTION - SANITY CHECK REPORT")
    print("*"*55)

    stats_df = check_dataset()
    check_metric_consistency()
    dummy_df = check_dummy_baseline()
    check_vs_dummy(dummy_df)
    check_confusion_matrices()
    check_loss_curves()

    print("\n" + "="*55)
    print("  DONE - check results/ folder for HTML plots")
    print("="*55 + "\n")
