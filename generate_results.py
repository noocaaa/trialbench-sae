"""
generate_report_figures.py
==========================
Run from your project root:
    python generate_report_figures.py

Reads results/ and exports all figures needed for the final report into img/.
Requires: pip install plotly kaleido pandas numpy scikit-learn

Figures produced
----------------
REPLACE (old ones were from pre-text/pre-tune run):
  img/ROC-AUC.png               — DL model ROC-AUC grouped bar by phase
  img/F1vsROC.png               — F1 vs ROC-AUC scatter, all models
  img/roc-auc-phase-Classical.png — classical models ROC-AUC line by phase
  img/mlp_vs_rf.png             — MLP vs RandomForest ROC-AUC line by phase
  img/mlp_vs_rf_scatter.png     — MLP vs RF F1 vs ROC-AUC scatter

NEW (add to report):
  img/roc_heatmap.png           — ROC-AUC heatmap all models × all phases
  img/dummy_vs_models_roc.png   — all models vs dummy baseline (ROC-AUC)
  img/dummy_vs_models_f1.png    — all models vs dummy baseline (F1) — shows imbalance trap
  img/threshold_calibration.png — calibrated thresholds per model per phase
  img/pr_auc_bar.png            — PR-AUC grouped bar (complements ROC)
  img/rnn_collapse.png          — RNN recall/precision/ROC across phases
  img/std_analysis.png          — std deviation bar (model stability)
  img/feature_dim_bar.png       — input feature dimensions per model (tree vs non-tree)
"""

import json
import glob
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ── Config ────────────────────────────────────────────────────────
RESULTS_DIR = "results"
IMG_DIR     = "img"
os.makedirs(IMG_DIR, exist_ok=True)

W, H     = 1100, 520   # default export size (px)
H_TALL   = 680
H_WIDE   = 420
SCALE    = 2           # retina

# ── Theme (matches existing dashboard) ───────────────────────────
BG      = "#080b14"
SURFACE = "#0d1117"
CARD    = "#111827"
BORDER  = "#1f2937"
GRID    = "#1a2234"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"
WHITE   = "#ffffff"

MODEL_COLORS = {
    "MLP":                "#8b5cf6",
    "CNN":                "#3b82f6",
    "RNN":                "#ef4444",   # red — matches red in report tables
    "FT-Transformer":     "#14b8a6",
    "LogisticRegression": "#f59e0b",
    "RandomForest":       "#f97316",
    "XGBoost":            "#ec4899",
    "LightGBM":           "#84cc16",
    "SVM":                "#fb923c",
    "KNN":                "#a78bfa",
    "Dummy":              "#64748b",
}

PHASE_COLORS = {
    "1": "#8b5cf6",
    "2": "#3b82f6",
    "3": "#06b6d4",
    "4": "#10b981",
}

PHASE_LABELS = {
    "1": "Phase I",
    "2": "Phase II",
    "3": "Phase III",
    "4": "Phase IV",
}

# Dummy baselines (majority-class classifier)
DUMMY_ROC = {"1": 0.500, "2": 0.500, "3": 0.500, "4": 0.500}
DUMMY_F1  = {"1": 0.606, "2": 0.854, "3": 0.917, "4": 0.555}

# Model groups
TREE_MODELS  = {"XGBoost", "LightGBM", "RandomForest"}
DL_MODELS    = {"MLP", "CNN", "RNN", "FT-Transformer"}
ALL_MODELS_ORDER = [
    "XGBoost", "LightGBM", "RandomForest",
    "SVM", "LogisticRegression", "KNN",
    "MLP", "FT-Transformer", "CNN", "RNN",
]

def mc(model):
    return MODEL_COLORS.get(model, "#94a3b8")

def base_layout(**kwargs):
    d = dict(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, size=13, family="Arial, sans-serif"),
        margin=dict(l=60, r=30, t=55, b=60),
        legend=dict(
            bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
            font=dict(color=TEXT, size=12),
        ),
    )
    d.update(kwargs)
    return d

def save(fig, name, width=W, height=H):
    path = os.path.join(IMG_DIR, name)
    fig.write_image(path, width=width, height=height, scale=SCALE)
    print(f"  [OK]  {path}")

# ── Load aggregated results ───────────────────────────────────────
def load_aggregated():
    agg_path = os.path.join(RESULTS_DIR, "aggregated_results.json")
    if os.path.exists(agg_path):
        with open(agg_path) as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        df["phase"] = df["phase"].astype(str)
        return df

    # Fall back: compute from fold JSONs
    files = [
        f for f in glob.glob(os.path.join(RESULTS_DIR, "*.json"))
        if not os.path.basename(f).endswith("_info.json")
        and "fold" in os.path.basename(f)
        and not os.path.basename(f).startswith("loss_")
    ]
    if not files:
        sys.exit("No result files found in results/. Run models first.")

    records = []
    for f in files:
        with open(f) as fp:
            records.append(json.load(fp))

    df = pd.DataFrame(records)
    df["phase"] = df["phase"].astype(str)

    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    agg = (
        df.groupby(["model", "phase"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["model", "phase"] + [
        f"{m}_{s}" for m in metrics for s in ["mean", "std"]
    ]
    return agg

def load_fold_data():
    """Load individual fold JSONs for threshold analysis."""
    files = [
        f for f in glob.glob(os.path.join(RESULTS_DIR, "*.json"))
        if not os.path.basename(f).endswith("_info.json")
        and "fold" in os.path.basename(f)
        and not os.path.basename(f).startswith("loss_")
    ]
    records = []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
            records.append({
                "model": d.get("model", ""),
                "phase": str(d.get("phase", "")),
                "fold":  d.get("fold", 0),
                "threshold": d.get("threshold", 0.5),
                "roc_auc":   d.get("roc_auc", np.nan),
                "f1":        d.get("f1", np.nan),
                "accuracy":  d.get("accuracy", np.nan),
                "recall":    d.get("recall", np.nan),
                "precision": d.get("precision", np.nan),
                "pr_auc":    d.get("pr_auc", np.nan),
            })
    return pd.DataFrame(records)


print("Loading results...")
df   = load_aggregated()
fdf  = load_fold_data()

# Normalise column names — handle both aggregated_results.json format and computed format
def get(df, model, phase, col):
    """Get mean value, handling both _mean suffix and direct column."""
    r = df[(df["model"] == model) & (df["phase"] == str(phase))]
    if r.empty:
        return np.nan
    col_mean = col + "_mean" if col + "_mean" in r.columns else col
    v = r[col_mean].values
    return float(v[0]) if len(v) > 0 else np.nan

def get_std(df, model, phase, col):
    r = df[(df["model"] == model) & (df["phase"] == str(phase))]
    if r.empty:
        return 0.0
    col_std = col + "_std" if col + "_std" in r.columns else None
    if col_std and col_std in r.columns:
        v = r[col_std].values
        return float(v[0]) if len(v) > 0 else 0.0
    return 0.0

models = [m for m in ALL_MODELS_ORDER if m in df["model"].unique()]
phases = ["1", "2", "3", "4"]

print(f"  Models found: {models}")
print(f"  Phases found: {sorted(df['phase'].unique())}")
print()

# ══════════════════════════════════════════════════════════════════
# 1. ROC-AUC HEATMAP  (img/roc_heatmap.png)  — NEW
# ══════════════════════════════════════════════════════════════════
print("1. ROC-AUC heatmap...")
mat   = [[get(df, m, p, "roc_auc") for p in phases] for m in models]
mat_r = [[round(v, 3) if not np.isnan(v) else 0 for v in row] for row in mat]

fig = go.Figure(go.Heatmap(
    z=mat_r,
    x=[PHASE_LABELS[p] for p in phases],
    y=models,
    colorscale=[[0, "#1a1a2e"], [0.45, "#7c3aed"], [0.75, "#06b6d4"], [1, "#10b981"]],
    zmin=0.45, zmax=0.95,
    text=[[f"{v:.3f}" for v in row] for row in mat_r],
    texttemplate="%{text}",
    textfont=dict(size=14, color="white", family="Arial"),
    hoverongaps=False,
))
fig.update_layout(
    **base_layout(margin=dict(l=160, r=30, t=55, b=60)),
    title=dict(text="ROC-AUC — All Models × All Phases", font=dict(size=15, color=TEXT)),
    height=480, width=820,
    xaxis=dict(side="top", tickfont=dict(size=13)),
    yaxis=dict(tickfont=dict(size=13), autorange="reversed"),
)
# Highlight tree models with a box annotation
save(fig, "roc_heatmap.png", width=820, height=480)

# ══════════════════════════════════════════════════════════════════
# 2. ROC-AUC LINE — CLASSICAL MODELS  (img/roc-auc-phase-Classical.png) — REPLACE
# ══════════════════════════════════════════════════════════════════
print("2. ROC-AUC classical models line chart...")
classical = ["XGBoost", "LightGBM", "RandomForest", "SVM", "LogisticRegression", "KNN"]
classical = [m for m in classical if m in models]

fig = go.Figure()
for m in classical:
    ys = [get(df, m, p, "roc_auc") for p in phases]
    es = [get_std(df, m, p, "roc_auc") for p in phases]
    xs = [PHASE_LABELS[p] for p in phases]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, name=m,
        mode="lines+markers",
        line=dict(color=mc(m), width=2.5),
        marker=dict(size=10, color=mc(m), line=dict(color=WHITE, width=1.5)),
        error_y=dict(type="data", array=es, visible=True, color=mc(m),
                     thickness=1.5, width=4),
    ))
# Dummy line
fig.add_trace(go.Scatter(
    x=[PHASE_LABELS[p] for p in phases],
    y=[DUMMY_ROC[p] for p in phases],
    name="Dummy (majority class)",
    mode="lines",
    line=dict(color=MUTED, dash="dot", width=1.5),
))
fig.update_layout(
    **base_layout(),
    title=dict(text="Classical Models — ROC-AUC by Phase", font=dict(size=15, color=TEXT)),
    yaxis=dict(title="ROC-AUC", range=[0.45, 0.95], gridcolor=GRID),
    xaxis=dict(title="Clinical Trial Phase"),
    height=H, width=W,
)
save(fig, "roc-auc-phase-Classical.png")

# ══════════════════════════════════════════════════════════════════
# 3. ROC-AUC GROUPED BAR — DL MODELS  (img/ROC-AUC.png) — REPLACE
# ══════════════════════════════════════════════════════════════════
print("3. DL models ROC-AUC grouped bar...")
dl_models = ["MLP", "FT-Transformer", "CNN", "RNN"]
dl_models = [m for m in dl_models if m in models]

fig = go.Figure()
for p in phases:
    ys = [get(df, m, p, "roc_auc") for m in dl_models]
    es = [get_std(df, m, p, "roc_auc") for m in dl_models]
    fig.add_trace(go.Bar(
        name=PHASE_LABELS[p],
        x=dl_models, y=ys,
        marker_color=PHASE_COLORS[p],
        opacity=0.85,
        error_y=dict(type="data", array=es, visible=True, color=WHITE,
                     thickness=1.5, width=4),
        text=[f"{v:.3f}" if not np.isnan(v) else "" for v in ys],
        textposition="outside",
        textfont=dict(size=11, color=TEXT),
    ))
fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED,
              annotation_text="Dummy = 0.500",
              annotation_font_color=MUTED, annotation_font_size=11)
fig.update_layout(
    **base_layout(),
    title=dict(text="Deep Learning Models — ROC-AUC by Phase", font=dict(size=15, color=TEXT)),
    barmode="group",
    yaxis=dict(title="ROC-AUC", range=[0.0, 1.05], gridcolor=GRID),
    xaxis=dict(title="Model"),
    height=H, width=W,
)
save(fig, "ROC-AUC.png")

# ══════════════════════════════════════════════════════════════════
# 4. F1 vs ROC-AUC SCATTER — ALL MODELS  (img/F1vsROC.png) — REPLACE
# ══════════════════════════════════════════════════════════════════
print("4. F1 vs ROC-AUC scatter...")
fig = go.Figure()
for m in models:
    xs, ys, texts = [], [], []
    for p in phases:
        roc = get(df, m, p, "roc_auc")
        f1  = get(df, m, p, "f1")
        if not np.isnan(roc) and not np.isnan(f1):
            xs.append(roc)
            ys.append(f1)
            texts.append(f"Ph.{p}")
    if xs:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            name=m, text=texts,
            textposition="top right",
            textfont=dict(size=9, color=MUTED),
            marker=dict(color=mc(m), size=13,
                        line=dict(color=WHITE, width=1.5),
                        symbol="circle" if m not in {"RNN"} else "x"),
        ))

# Dummy reference points
fig.add_trace(go.Scatter(
    x=[DUMMY_ROC[p] for p in phases],
    y=[DUMMY_F1[p]  for p in phases],
    mode="markers+text",
    name="Dummy",
    text=[f"Ph.{p}" for p in phases],
    textposition="top right",
    textfont=dict(size=9, color=MUTED),
    marker=dict(color=MUTED, size=10, symbol="x"),
))
fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED, line_width=1)
fig.add_vline(x=0.5, line_dash="dot", line_color=MUTED, line_width=1)
fig.update_layout(
    **base_layout(),
    title=dict(text="F1 vs ROC-AUC — All Models & Phases", font=dict(size=15, color=TEXT)),
    xaxis=dict(title="ROC-AUC", range=[0.3, 1.02], gridcolor=GRID),
    yaxis=dict(title="F1",      range=[0.3, 1.02], gridcolor=GRID),
    height=H, width=W,
)
save(fig, "F1vsROC.png")

# ══════════════════════════════════════════════════════════════════
# 5. MLP vs RANDOM FOREST — ROC-AUC LINE  (img/mlp_vs_rf.png) — REPLACE
# ══════════════════════════════════════════════════════════════════
print("5. MLP vs RF ROC-AUC...")
fig = go.Figure()
for m, dash in [("XGBoost", "solid"), ("RandomForest", "dash"), ("MLP", "dot")]:
    if m not in models:
        continue
    ys = [get(df, m, p, "roc_auc") for p in phases]
    es = [get_std(df, m, p, "roc_auc") for p in phases]
    fig.add_trace(go.Scatter(
        x=[PHASE_LABELS[p] for p in phases], y=ys, name=m,
        mode="lines+markers",
        line=dict(color=mc(m), width=2.5, dash=dash),
        marker=dict(size=10, color=mc(m), line=dict(color=WHITE, width=1.5)),
        error_y=dict(type="data", array=es, visible=True, color=mc(m),
                     thickness=1.5, width=4),
    ))
fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED,
              annotation_text="Dummy", annotation_font_color=MUTED)
fig.update_layout(
    **base_layout(),
    title=dict(text="Best Tree (XGBoost) vs Random Forest vs Best DL (MLP)", font=dict(size=14, color=TEXT)),
    yaxis=dict(title="ROC-AUC", range=[0.45, 0.95], gridcolor=GRID),
    xaxis=dict(title="Clinical Trial Phase"),
    height=H, width=W,
)
save(fig, "mlp_vs_rf.png")

# ══════════════════════════════════════════════════════════════════
# 6. MLP vs RF F1 vs ROC SCATTER  (img/mlp_vs_rf_scatter.png) — REPLACE
# ══════════════════════════════════════════════════════════════════
print("6. MLP vs RF F1 vs ROC scatter...")
fig = go.Figure()
for m in ["XGBoost", "RandomForest", "MLP"]:
    if m not in models:
        continue
    xs, ys, texts = [], [], []
    for p in phases:
        roc = get(df, m, p, "roc_auc")
        f1  = get(df, m, p, "f1")
        if not np.isnan(roc) and not np.isnan(f1):
            xs.append(roc)
            ys.append(f1)
            texts.append(f"Ph.{p}")
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", name=m,
        text=texts, textposition="top right",
        textfont=dict(size=10, color=MUTED),
        marker=dict(color=mc(m), size=14, line=dict(color=WHITE, width=1.5)),
    ))
fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED, line_width=1)
fig.add_vline(x=0.5, line_dash="dot", line_color=MUTED, line_width=1)
fig.update_layout(
    **base_layout(),
    title=dict(text="F1 vs ROC-AUC: Tree Ensembles vs MLP", font=dict(size=14, color=TEXT)),
    xaxis=dict(title="ROC-AUC", range=[0.55, 0.95], gridcolor=GRID),
    yaxis=dict(title="F1",      range=[0.55, 0.98], gridcolor=GRID),
    height=H, width=W,
)
save(fig, "mlp_vs_rf_scatter.png")

# ══════════════════════════════════════════════════════════════════
# 7. DUMMY vs MODELS — ROC-AUC  (img/dummy_vs_models_roc.png) — NEW
# ══════════════════════════════════════════════════════════════════
print("7. Dummy vs models ROC-AUC...")
fig = go.Figure()
for m in models:
    ys = [get(df, m, p, "roc_auc") for p in phases]
    fig.add_trace(go.Scatter(
        x=[PHASE_LABELS[p] for p in phases], y=ys, name=m,
        mode="lines+markers",
        line=dict(color=mc(m), width=2 if m not in {"RNN","CNN"} else 1.5),
        marker=dict(size=8, color=mc(m)),
        opacity=1.0 if m not in {"RNN","CNN"} else 0.6,
    ))
fig.add_trace(go.Scatter(
    x=[PHASE_LABELS[p] for p in phases],
    y=[DUMMY_ROC[p] for p in phases],
    name="Dummy (majority class)",
    mode="lines+markers",
    line=dict(color=MUTED, dash="dot", width=2),
    marker=dict(size=8, symbol="x"),
))
fig.add_hrect(y0=0.45, y1=0.55, fillcolor=MUTED, opacity=0.08,
              annotation_text="near-random zone",
              annotation_font_color=MUTED, annotation_font_size=10)
fig.update_layout(
    **base_layout(),
    title=dict(text="ROC-AUC vs Dummy Baseline — All Models", font=dict(size=15, color=TEXT)),
    yaxis=dict(title="ROC-AUC", range=[0.40, 1.0], gridcolor=GRID),
    xaxis=dict(title="Clinical Trial Phase"),
    height=H, width=W,
)
save(fig, "dummy_vs_models_roc.png")

# ══════════════════════════════════════════════════════════════════
# 8. DUMMY vs MODELS — F1  (img/dummy_vs_models_f1.png) — NEW
#    Shows the imbalance trap: some models can't beat the dummy in F1
# ══════════════════════════════════════════════════════════════════
print("8. Dummy vs models F1 (imbalance trap)...")
fig = go.Figure()
for m in models:
    ys = [get(df, m, p, "f1") for p in phases]
    fig.add_trace(go.Scatter(
        x=[PHASE_LABELS[p] for p in phases], y=ys, name=m,
        mode="lines+markers",
        line=dict(color=mc(m), width=2),
        marker=dict(size=8, color=mc(m)),
    ))
fig.add_trace(go.Scatter(
    x=[PHASE_LABELS[p] for p in phases],
    y=[DUMMY_F1[p] for p in phases],
    name="Dummy F1 (always predicts SAE)",
    mode="lines+markers",
    line=dict(color="#ef4444", dash="dash", width=2.5),
    marker=dict(size=9, symbol="x", color="#ef4444"),
))
fig.update_layout(
    **base_layout(),
    title=dict(
        text="F1 vs Dummy Baseline — Why F1 is Unreliable in Phases II & III",
        font=dict(size=14, color=TEXT)
    ),
    yaxis=dict(title="F1 Score", range=[0.3, 1.05], gridcolor=GRID),
    xaxis=dict(title="Clinical Trial Phase"),
    height=H, width=W,
)
save(fig, "dummy_vs_models_f1.png")

# ══════════════════════════════════════════════════════════════════
# 9. PR-AUC GROUPED BAR  (img/pr_auc_bar.png) — NEW
# ══════════════════════════════════════════════════════════════════
print("9. PR-AUC grouped bar...")
fig = go.Figure()
for p in phases:
    ys = [get(df, m, p, "pr_auc") for m in models]
    es = [get_std(df, m, p, "pr_auc") for m in models]
    fig.add_trace(go.Bar(
        name=PHASE_LABELS[p], x=models, y=ys,
        marker_color=PHASE_COLORS[p], opacity=0.85,
        error_y=dict(type="data", array=es, visible=True, color=WHITE,
                     thickness=1.5, width=4),
    ))
fig.update_layout(
    **base_layout(margin=dict(l=60, r=30, t=55, b=80)),
    title=dict(text="PR-AUC — All Models by Phase", font=dict(size=15, color=TEXT)),
    barmode="group",
    yaxis=dict(title="PR-AUC", range=[0, 1.05], gridcolor=GRID),
    xaxis=dict(title="Model", tickangle=-20),
    height=H, width=W,
)
save(fig, "pr_auc_bar.png")

# ══════════════════════════════════════════════════════════════════
# 10. RNN COLLAPSE PANEL  (img/rnn_collapse.png) — NEW
#     Shows ROC-AUC, Recall, Precision for RNN vs MLP across phases
# ══════════════════════════════════════════════════════════════════
print("10. RNN collapse vs MLP panel...")
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=["ROC-AUC", "Recall", "Precision"],
    shared_yaxes=False,
)
for m, lw in [("RNN", 2.5), ("MLP", 2)]:
    if m not in models:
        continue
    for col_idx, metric in enumerate(["roc_auc", "recall", "precision"], 1):
        ys = [get(df, m, p, metric) for p in phases]
        es = [get_std(df, m, p, metric) for p in phases]
        fig.add_trace(go.Scatter(
            x=[PHASE_LABELS[p] for p in phases], y=ys,
            name=m if col_idx == 1 else None,
            showlegend=(col_idx == 1),
            mode="lines+markers",
            line=dict(color=mc(m), width=lw),
            marker=dict(size=9, color=mc(m), line=dict(color=WHITE, width=1)),
            error_y=dict(type="data", array=es, visible=True, color=mc(m),
                         thickness=1.5, width=4),
        ), row=1, col=col_idx)

# Add dummy ROC reference
fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED,
              row=1, col=1, annotation_text="random",
              annotation_font_color=MUTED, annotation_font_size=9)
fig.update_layout(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(color=TEXT, size=12, family="Arial, sans-serif"),
    margin=dict(l=50, r=30, t=65, b=60),
    legend=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=TEXT, size=12)),
    title=dict(text="RNN Failure: Sequential Inductive Bias on Tabular Data (vs MLP)",
               font=dict(size=13, color=TEXT)),
    height=H_WIDE, width=W,
)
for i in range(1, 4):
    fig.update_xaxes(gridcolor=GRID, linecolor=BORDER, row=1, col=i)
    fig.update_yaxes(gridcolor=GRID, linecolor=BORDER, range=[0.3, 1.05], row=1, col=i)
save(fig, "rnn_collapse.png", width=W, height=H_WIDE)

# ══════════════════════════════════════════════════════════════════
# 11. THRESHOLD CALIBRATION  (img/threshold_calibration.png) — NEW
#     Shows that calibrated thresholds deviate substantially from 0.5
# ══════════════════════════════════════════════════════════════════
print("11. Threshold calibration...")
if not fdf.empty and "threshold" in fdf.columns:
    th_agg = fdf.groupby(["model", "phase"])["threshold"].agg(["mean","std"]).reset_index()
    th_agg.columns = ["model", "phase", "threshold_mean", "threshold_std"]

    fig = go.Figure()
    for m in models:
        sub = th_agg[th_agg["model"] == m].sort_values("phase")
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=[PHASE_LABELS[p] for p in sub["phase"]],
            y=sub["threshold_mean"],
            name=m, mode="lines+markers",
            line=dict(color=mc(m), width=2),
            marker=dict(size=8, color=mc(m)),
            error_y=dict(type="data", array=sub["threshold_std"].tolist(),
                         visible=True, color=mc(m), thickness=1.2, width=3),
        ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#e2e8f0",
                  annotation_text="Fixed 0.5 (uncalibrated)",
                  annotation_font_color=TEXT, annotation_font_size=11)
    fig.update_layout(
        **base_layout(),
        title=dict(text="Calibrated Decision Thresholds per Model & Phase",
                   font=dict(size=15, color=TEXT)),
        yaxis=dict(title="Threshold", range=[0, 1.0], gridcolor=GRID),
        xaxis=dict(title="Clinical Trial Phase"),
        height=H, width=W,
    )
    save(fig, "threshold_calibration.png")
else:
    print("  (skipped — no threshold data in fold JSONs)")

# ══════════════════════════════════════════════════════════════════
# 12. MODEL STABILITY — STD of ROC-AUC  (img/std_analysis.png) — NEW
#     Higher std = less stable across folds
# ══════════════════════════════════════════════════════════════════
print("12. Model stability (std of ROC-AUC)...")
fig = go.Figure()
for p in phases:
    stds = [get_std(df, m, p, "roc_auc") for m in models]
    fig.add_trace(go.Bar(
        name=PHASE_LABELS[p], x=models, y=stds,
        marker_color=PHASE_COLORS[p], opacity=0.85,
    ))
fig.update_layout(
    **base_layout(margin=dict(l=60, r=30, t=55, b=80)),
    title=dict(text="ROC-AUC Standard Deviation — Model Stability Across Folds",
               font=dict(size=14, color=TEXT)),
    barmode="group",
    yaxis=dict(title="Std Dev (ROC-AUC)", gridcolor=GRID),
    xaxis=dict(title="Model", tickangle=-20),
    height=H, width=W,
)
save(fig, "std_analysis.png")

# ══════════════════════════════════════════════════════════════════
# 13. FEATURE DIMENSIONALITY  (img/feature_dim_bar.png) — NEW
#     Shows real per-phase input dimensions extracted from metadata
# ══════════════════════════════════════════════════════════════════
print("13. Feature dimensionality...")

# Load real dimensions from shap_ready metadata (averaged per model/phase)
def load_feature_dims():
    dims = {}
    for path in glob.glob("models/shap_ready/*_metadata.json"):
        with open(path) as f:
            data = json.load(f)
        model = data.get("model_name", "")
        phase = str(data.get("phase", ""))
        nfeat = data.get("n_features", data.get("input_dim", None))
        if model and phase and nfeat is not None:
            dims.setdefault((model, phase), []).append(nfeat)
    # average per (model, phase)
    avg = {}
    for (m, p), vals in dims.items():
        avg[(m, p)] = int(round(np.mean(vals)))
    return avg

fdims = load_feature_dims()
fd_models = [m for m in ALL_MODELS_ORDER if m in df["model"].unique()]

# Build a clean table: rows = phases, cols = models
# This lets us label a bar only when its value differs from the previous phase
fig = go.Figure()
for p in phases:
    ys = [fdims.get((m, p), np.nan) for m in fd_models]
    texts = []
    for m, v in zip(fd_models, ys):
        if np.isnan(v):
            texts.append("")
            continue
        # Show number if it changes from previous phase, or if it's the first phase
        prev_p = str(int(p) - 1)
        prev_v = fdims.get((m, prev_p), None)
        if prev_v is None or int(v) != int(prev_v):
            texts.append(f"{int(v)}")
        else:
            texts.append("")
    fig.add_trace(go.Bar(
        name=PHASE_LABELS[p],
        x=fd_models, y=ys,
        marker_color=PHASE_COLORS[p], opacity=0.85,
        text=texts,
        textposition="outside",
        textfont=dict(size=10, color=TEXT),
    ))

# Horizontal separator line between the two encoding groups
fig.add_hline(y=160, line_dash="solid", line_color="#f97316",
              line_width=1.5, opacity=0.5)

# Annotation bands (subtle background)
fig.add_hrect(y0=0, y1=160, fillcolor="#f97316", opacity=0.05)
fig.add_hrect(y0=160, y1=450, fillcolor="#8b5cf6", opacity=0.05)

# Encoding type labels on the left where there is empty space
fig.add_annotation(
    x=0.02, y=0.88,
    text="<b>One-hot encoding</b><br>(non-tree models)",
    showarrow=False,
    font=dict(color="#8b5cf6", size=10),
    align="left",
    xref="paper", yref="paper",
)
fig.add_annotation(
    x=0.02, y=0.37,
    text="<b>Ordinal encoding</b><br>(tree models)",
    showarrow=False,
    font=dict(color="#f97316", size=10),
    align="left",
    xref="paper", yref="paper",
)
fig.update_layout(
    **base_layout(margin=dict(l=60, r=30, t=55, b=80)),
    title=dict(
        text="Input Feature Dimensionality by Model & Phase",
        font=dict(size=14, color=TEXT)
    ),
    barmode="group",
    yaxis=dict(title="Number of Features", gridcolor=GRID, range=[0, 480]),
    xaxis=dict(title="Model", tickangle=-20),
    height=H, width=W,
)
save(fig, "feature_dim_bar.png")

# ══════════════════════════════════════════════════════════════════
# 14. FULL COMPARISON — ALL METRICS OVERVIEW  (img/all_metrics_phase1.png) — NEW
#     Radar chart per phase showing all metrics for all models
# ══════════════════════════════════════════════════════════════════
print("14. All-metrics radar Phase I...")
metrics_radar = ["accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"]
metric_labels = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC", "PR-AUC"]

fig = go.Figure()
for m in models:
    vals = [get(df, m, "1", met) for met in metrics_radar]
    if any(np.isnan(v) for v in vals):
        continue
    vals_closed = vals + [vals[0]]
    fig.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=metric_labels + [metric_labels[0]],
        fill="toself", name=m,
        line=dict(color=mc(m), width=2),
        opacity=0.55,
    ))
fig.update_layout(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(color=TEXT, size=12, family="Arial, sans-serif"),
    margin=dict(l=60, r=60, t=70, b=60),
    legend=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=TEXT, size=11)),
    title=dict(text="All Metrics Radar — Phase I (balanced, all metrics informative)",
               font=dict(size=13, color=TEXT)),
    polar=dict(
        bgcolor=CARD,
        radialaxis=dict(visible=True, range=[0.4, 1.0], gridcolor=GRID,
                        tickfont=dict(color=MUTED, size=10)),
        angularaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT, size=12)),
    ),
    height=560, width=700,
)
save(fig, "all_metrics_phase1.png", width=700, height=560)

print()
print("=" * 55)
print(f"  All figures saved to  {IMG_DIR}/")
print("=" * 55)
print()
print("  REPLACE in report (results changed with --tune --use-text):")
print("    img/ROC-AUC.png")
print("    img/F1vsROC.png")
print("    img/roc-auc-phase-Classical.png")
print("    img/mlp_vs_rf.png")
print("    img/mlp_vs_rf_scatter.png")
print()
print("  NEW figures to add to report:")
print("    img/roc_heatmap.png            -> Section 4.5 cross-phase summary")
print("    img/dummy_vs_models_roc.png    -> Section 5.4 metric selection")
print("    img/dummy_vs_models_f1.png     -> Section 5.4 metric selection")
print("    img/rnn_collapse.png           -> Section 5.3 RNN failure")
print("    img/threshold_calibration.png  -> Section 3.5 eval protocol")
print("    img/pr_auc_bar.png             -> Section 4 results")
print("    img/std_analysis.png           -> Section 5 stability discussion")
print("    img/feature_dim_bar.png        -> Section 2.3 preprocessing")
print("    img/all_metrics_phase1.png     -> Section 4.1 Phase I results")
print()
print("  DO NOT regenerate (dataset properties, unchanged):")
print("    img/nju_logo.png")
print("    img/Feature_Set.png")
print("    img/Class Balance.png")
print("    img/Missing Values.png")
print("    img/Feature Importance.png")