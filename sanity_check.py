"""
sanity_check.py — Automated sanity checks for SAE Prediction models
Run this after training all models to verify results are trustworthy.

Usage:
    python sanity_check.py
"""
import json, glob, os
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from src.data_loader import load_phase
from src.evaluate import evaluate

PHASES  = ["1", "2", "3", "4"]
RESULTS = "results"


# ── 1. Dataset statistics ──────────────────────────────────────────
def check_dataset():
    print("\n" + "="*55)
    print("  CHECK 1 — Dataset Statistics & Class Balance")
    print("="*55)

    stats = []
    for phase in PHASES:
        try:
            X_train, X_test, y_train, y_test, pos_weight = load_phase(phase)
            n       = len(y_test)
            pos     = int(y_test.sum())
            neg     = n - pos
            ratio   = pos / n
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
            print(f"    ⚠  Dummy accuracy baseline : {dummy_acc*100:.1f}%  ← your accuracy must beat this")
        except Exception as e:
            print(f"  Phase {phase}: ERROR — {e}")

    return pd.DataFrame(stats)


# ── 2. Metric consistency check ────────────────────────────────────
def check_metric_consistency():
    print("\n" + "="*55)
    print("  CHECK 2 — Metric Consistency")
    print("="*55)

    files = glob.glob(os.path.join(RESULTS, "*.json"))
    if not files:
        print("  No results found — run your models first.")
        return

    all_ok = True
    for f in sorted(files):
        r = json.load(open(f))
        name = f"{r['model']} Phase {r['phase']}"

        issues = []

        # F1 should be harmonic mean of precision and recall
        p, rec, f1 = r.get("precision",0), r.get("recall",0), r.get("f1",0)
        if p + rec > 0:
            expected_f1 = 2 * p * rec / (p + rec)
            if abs(expected_f1 - f1) > 0.01:
                issues.append(f"F1 mismatch: got {f1:.4f}, expected {expected_f1:.4f}")

        # ROC-AUC must be > 0.5
        roc = r.get("roc_auc", None)
        if roc is not None and not np.isnan(roc) and roc < 0.5:
            issues.append(f"ROC-AUC = {roc:.4f} < 0.5 (worse than random — labels may be flipped)")

        # All metrics must be in [0, 1]
        for met in ["accuracy", "f1", "precision", "recall", "pr_auc"]:
            v = r.get(met, None)
            if v is not None and not np.isnan(v) and not (0 <= v <= 1):
                issues.append(f"{met} = {v:.4f} out of [0,1] range")

        if issues:
            all_ok = False
            print(f"\n  ❌ {name}:")
            for issue in issues: print(f"     → {issue}")
        else:
            print(f"  ✅ {name}: all metrics consistent")

    if all_ok:
        print("\n  All metrics passed consistency checks!")


# ── 3. Dummy classifier baseline ──────────────────────────────────
def check_dummy_baseline():
    print("\n" + "="*55)
    print("  CHECK 3 — Dummy Classifier Baseline")
    print("="*55)
    print("  Your models must beat these scores to be meaningful.\n")

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
            print(f"  Phase {phase}: ERROR — {e}")

    return pd.DataFrame(dummy_results)


# ── 4. Model vs Dummy comparison ───────────────────────────────────
def check_vs_dummy(dummy_df):
    if dummy_df.empty: return
    print("\n" + "="*55)
    print("  CHECK 4 — Your Models vs Dummy")
    print("="*55)

    files = glob.glob(os.path.join(RESULTS, "*.json"))
    real_results = [json.load(open(f)) for f in files
                    if "Dummy" not in f]

    for r in sorted(real_results, key=lambda x: (x["model"], x["phase"])):
        phase = str(r["phase"])
        dummy_row = dummy_df[dummy_df["phase"] == phase]
        if dummy_row.empty: continue

        beats_f1  = r["f1"]      > float(dummy_row["f1"].values[0])
        beats_roc = r.get("roc_auc", 0) > float(dummy_row.get("roc_auc", pd.Series([0])).values[0])

        icon = "✅" if (beats_f1 and beats_roc) else "❌"
        print(f"\n  {icon} {r['model']} Phase {phase}")
        print(f"     F1      : {r['f1']:.4f}  vs dummy {float(dummy_row['f1'].values[0]):.4f}  {'↑ better' if beats_f1 else '↓ WORSE'}")
        roc_val = r.get('roc_auc')
        dummy_roc = float(dummy_row.get('roc_auc', pd.Series([0])).values[0])
        if roc_val and not np.isnan(roc_val):
            print(f"     ROC-AUC : {roc_val:.4f}  vs dummy {dummy_roc:.4f}  {'↑ better' if beats_roc else '↓ WORSE'}")


# ── 5. Confusion matrices ──────────────────────────────────────────
def check_confusion_matrices():
    print("\n" + "="*55)
    print("  CHECK 5 — Confusion Matrices (Logistic Regression proxy)")
    print("="*55)
    print("  Using LR as a fast proxy — run per model for exact CMs.\n")

    os.makedirs(RESULTS, exist_ok=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=len(PHASES),
                            subplot_titles=[f"Phase {p}" for p in PHASES])

        for i, phase in enumerate(PHASES, 1):
            X_train, X_test, y_train, y_test, _ = load_phase(phase)
            clf = LogisticRegression(max_iter=500, class_weight="balanced")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            labels = ["No SAE", "SAE"]
            pct    = cm / cm.sum() * 100

            text = [[f"{cm[r][c]}<br>({pct[r][c]:.1f}%)"
                     for c in range(2)] for r in range(2)]

            fig.add_trace(go.Heatmap(
                z=cm, x=labels, y=labels,
                text=text, texttemplate="%{text}",
                colorscale="Blues", showscale=False,
                textfont=dict(size=12),
            ), row=1, col=i)

            tn, fp, fn, tp = cm.ravel()
            print(f"  Phase {phase}: TP={tp}  TN={tn}  FP={fp}  FN={fn}")
            print(f"    → False Negatives (missed SAEs): {fn}  ← dangerous in medical context")

        fig.update_layout(
            title="Confusion Matrices — Logistic Regression (proxy)",
            paper_bgcolor="#0f0f17", plot_bgcolor="#1a1a2e",
            font=dict(color="#ccccee"),
        )
        out = os.path.join(RESULTS, "confusion_matrices.html")
        fig.write_html(out)
        print(f"\n  Saved → {out}  (open in browser)")

    except Exception as e:
        print(f"  Could not generate plot: {e}")


# ── 6. Loss curve check ────────────────────────────────────────────
def check_loss_curves():
    print("\n" + "="*55)
    print("  CHECK 6 — Loss Curve Files")
    print("="*55)

    files = glob.glob(os.path.join(RESULTS, "loss_*.json"))
    if not files:
        print("  No loss curves saved yet.")
        print("  → Add loss history saving to your model run() functions.")
        print("    Example: save list of epoch losses as results/loss_CNN_1.json")
        return

    try:
        import plotly.graph_objects as go
        COLORS = ["#7c6af7", "#f7916a", "#6af7c8", "#f7e06a"]
        fig = go.Figure()
        for i, f in enumerate(sorted(files)):
            name = os.path.basename(f).replace("loss_","").replace(".json","")
            losses = json.load(open(f))
            fig.add_trace(go.Scatter(
                x=list(range(1, len(losses)+1)), y=losses,
                mode="lines+markers", name=name,
                line=dict(color=COLORS[i % len(COLORS)], width=2),
            ))
        fig.update_layout(
            title="Training Loss Curves",
            xaxis_title="Epoch", yaxis_title="Loss",
            paper_bgcolor="#0f0f17", plot_bgcolor="#1a1a2e",
            font=dict(color="#ccccee"),
            xaxis=dict(gridcolor="#333355"),
            yaxis=dict(gridcolor="#333355"),
        )
        out = os.path.join(RESULTS, "loss_curves.html")
        fig.write_html(out)
        print(f"  Saved → {out}  (open in browser)")
    except Exception as e:
        print(f"  Could not generate plot: {e}")


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "★"*55)
    print("  SAE PREDICTION — SANITY CHECK REPORT")
    print("★"*55)

    stats_df  = check_dataset()
    check_metric_consistency()
    dummy_df  = check_dummy_baseline()
    check_vs_dummy(dummy_df)
    check_confusion_matrices()
    check_loss_curves()

    print("\n" + "="*55)
    print("  DONE — check results/ folder for HTML plots")
    print("="*55 + "\n")