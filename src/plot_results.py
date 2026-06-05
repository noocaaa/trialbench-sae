"""
plot_results.py — SAE Prediction Results Dashboard
Run: python plot_results.py
"""
import json, glob, os, sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

METRICS       = ["accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"]
METRIC_LABELS = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC", "PR-AUC"]
PHASES        = ["1", "2", "3", "4"]

BG      = "#080b14"
SURFACE = "#0d1117"
CARD    = "#111827"
BORDER  = "#1f2937"
GRID    = "#1a2234"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"

C = {
    "purple": "#8b5cf6", "blue":   "#3b82f6",
    "cyan":   "#06b6d4", "green":  "#10b981",
    "yellow": "#f59e0b", "pink":   "#ec4899",
    "red":    "#ef4444", "orange": "#f97316",
}

MODEL_COLORS = {
    "MLP": "#8b5cf6", "CNN": "#3b82f6", "RNN": "#06b6d4",
    "Transformer": "#10b981", "FT-Transformer": "#14b8a6",
    "LogisticRegression": "#f59e0b",
    "RandomForest": "#f97316", "XGBoost": "#ec4899",
    "LightGBM": "#84cc16", "SVM": "#ef4444", "KNN": "#a78bfa",
}
PHASE_COLORS = {"1": "#8b5cf6", "2": "#3b82f6", "3": "#06b6d4", "4": "#10b981"}

TAB_S = {"backgroundColor": SURFACE, "color": MUTED,
          "border": f"1px solid {BORDER}", "borderRadius": "6px 6px 0 0",
          "padding": "10px 20px", "fontFamily": "monospace", "fontSize": "12px"}
TAB_A = {**TAB_S, "backgroundColor": CARD,
          "color": C["purple"], "borderBottom": f"2px solid {C['purple']}"}

base = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(color=TEXT, size=11, family="monospace"),
    margin=dict(l=50, r=20, t=45, b=50),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(color=TEXT, size=11)),
)


def get_model_color(model):
    for k, v in MODEL_COLORS.items():
        if k.lower() in model.lower():
            return v
    import hashlib
    idx = int(hashlib.md5(model.encode()).hexdigest(), 16) % len(C)
    return list(C.values())[idx]


# Always resolve paths relative to the project root (one level above this file)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR  = os.path.join(_PROJECT_ROOT, "results")


def load_results(results_dir=None):
    if results_dir is None:
        results_dir = _RESULTS_DIR
    files = [f for f in glob.glob(os.path.join(results_dir, "*.json"))
             if not os.path.basename(f).startswith("loss_") and not f.endswith("_info.json")]
    if not files:
        return pd.DataFrame()
    records = []
    for f in files:
        with open(f) as fp:
            records.append(json.load(fp))
    df = pd.DataFrame(records)
    df["phase"] = df["phase"].astype(str)
    return df


def load_loss_curves(results_dir=None):
    if results_dir is None:
        results_dir = _RESULTS_DIR
    curves = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "loss_*.json"))):
        name  = os.path.basename(f).replace("loss_", "").replace(".json", "")
        parts = name.rsplit("_", 1)
        model, phase = (parts[0], parts[1]) if len(parts) == 2 else (name, "?")
        with open(f) as fp:
            curves.setdefault(model, {})[phase] = json.load(fp)
    return curves


def card(children, style=None):
    s = {"backgroundColor": CARD, "borderRadius": "10px",
         "padding": "16px", "border": f"1px solid {BORDER}"}
    if style:
        s.update(style)
    return html.Div(children, style=s)


def stitle(text, color=None):
    return html.H3(text, style={"color": color or C["purple"], "fontSize": "11px",
                                 "letterSpacing": "2px", "textTransform": "uppercase",
                                 "marginBottom": "12px", "marginTop": "0"})


def mbadge(label, value, color):
    return html.Div([
        html.Div(label, style={"color": MUTED, "fontSize": "10px",
                                "letterSpacing": "1px", "textTransform": "uppercase",
                                "marginBottom": "4px"}),
        html.Div(str(value), style={"color": color, "fontSize": "22px",
                                     "fontWeight": "700", "fontFamily": "monospace"}),
    ], style={"backgroundColor": SURFACE, "borderRadius": "8px",
              "padding": "12px 16px", "border": f"1px solid {BORDER}",
              "minWidth": "110px", "textAlign": "center"})


def run(results_dir=None):
    df     = load_results(results_dir)
    df     = df.dropna(subset=["model"])
    curves = load_loss_curves(results_dir)

    if df.empty:
        print("No results found. Run models first.")
        return

    models = sorted(df["model"].unique())
    phases = sorted(df["phase"].unique())
    mc     = {m: get_model_color(m) for m in models}

    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    app.layout = html.Div(
        style={"backgroundColor": BG, "minHeight": "100vh",
               "fontFamily": "monospace", "padding": "20px"},
        children=[
            # Header
            html.Div([
                html.Div([
                    html.H1("SAE Prediction — Dashboard",
                            style={"color": TEXT, "margin": "0", "fontSize": "18px",
                                   "fontWeight": "700", "letterSpacing": "2px"}),
                    html.Span("TrialBench · AI4Science",
                              style={"color": MUTED, "fontSize": "11px",
                                     "display": "block", "marginTop": "2px"}),
                ]),
                html.Div([
                    html.Span(f"{len(models)} models",
                              style={"color": C["cyan"], "fontSize": "11px",
                                     "backgroundColor": CARD, "padding": "4px 10px",
                                     "borderRadius": "20px",
                                     "border": f"1px solid {BORDER}",
                                     "marginRight": "8px"}),
                    html.Span(f"{len(phases)} phases",
                              style={"color": C["green"], "fontSize": "11px",
                                     "backgroundColor": CARD, "padding": "4px 10px",
                                     "borderRadius": "20px",
                                     "border": f"1px solid {BORDER}"}),
                ]),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "center", "marginBottom": "16px",
                      "paddingBottom": "16px",
                      "borderBottom": f"1px solid {BORDER}"}),

            # Controls
            card([
                html.Div([
                    # Models
                    html.Div([
                        html.Div("Models", style={"color": C["purple"],
                                                   "fontSize": "11px",
                                                   "letterSpacing": "1px",
                                                   "textTransform": "uppercase",
                                                   "marginBottom": "8px"}),
                        dcc.Checklist(
                            id="sel-models",
                            options=[{"label": f"  {m}", "value": m} for m in models],
                            value=list(models),
                            labelStyle={"display": "block", "color": TEXT,
                                        "marginBottom": "6px", "fontSize": "12px"},
                            inputStyle={"marginRight": "6px",
                                        "accentColor": C["purple"]},
                        ),
                    ]),
                    # Phases
                    html.Div([
                        html.Div("Phases", style={"color": C["blue"],
                                                   "fontSize": "11px",
                                                   "letterSpacing": "1px",
                                                   "textTransform": "uppercase",
                                                   "marginBottom": "8px"}),
                        dcc.Checklist(
                            id="sel-phases",
                            options=[{"label": f"  Phase {p}", "value": p}
                                     for p in phases],
                            value=list(phases),
                            labelStyle={"display": "block", "color": TEXT,
                                        "marginBottom": "6px", "fontSize": "12px"},
                            inputStyle={"marginRight": "6px",
                                        "accentColor": C["blue"]},
                        ),
                    ]),
                    # Metric
                    html.Div([
                        html.Div("Metric", style={"color": C["cyan"],
                                                   "fontSize": "11px",
                                                   "letterSpacing": "1px",
                                                   "textTransform": "uppercase",
                                                   "marginBottom": "8px"}),
                        dcc.RadioItems(
                            id="sel-metric",
                            options=[{"label": f"  {l}", "value": v}
                                     for l, v in zip(METRIC_LABELS, METRICS)],
                            value="roc_auc",
                            labelStyle={"display": "block", "color": TEXT,
                                        "marginBottom": "6px", "fontSize": "12px"},
                            inputStyle={"marginRight": "6px",
                                        "accentColor": C["cyan"]},
                        ),
                    ]),
                ], style={"display": "flex", "gap": "50px", "flexWrap": "wrap"}),
            ], style={"marginBottom": "14px"}),

            # Tabs
            dcc.Tabs(id="tabs", value="overview", style={"marginBottom": "0"},
                     children=[
                         dcc.Tab(label="Overview",           value="overview",
                                 style=TAB_S, selected_style=TAB_A),
                         dcc.Tab(label="Compare Models",     value="compare",
                                 style=TAB_S, selected_style=TAB_A),
                         dcc.Tab(label="Training Curves",    value="training",
                                 style=TAB_S, selected_style=TAB_A),
                         dcc.Tab(label="Confusion Matrices", value="confusion",
                                 style=TAB_S, selected_style=TAB_A),
                         dcc.Tab(label="Sanity Check",       value="sanity",
                                 style=TAB_S, selected_style=TAB_A),
                     ]),

            html.Div(id="tab-content",
                     style={"backgroundColor": CARD,
                            "borderRadius": "0 10px 10px 10px",
                            "border": f"1px solid {BORDER}",
                            "padding": "20px", "minHeight": "500px"}),
        ]
    )

    @callback(Output("tab-content", "children"),
              Input("tabs",       "value"),
              Input("sel-models", "value"),
              Input("sel-phases", "value"),
              Input("sel-metric", "value"))
    def render(tab, sel_models, sel_phases, metric):
        sel_models = sel_models or []
        sel_phases = sel_phases or []
        metric     = metric or "roc_auc"
        mlabel     = METRIC_LABELS[METRICS.index(metric)]
        data = df[df["model"].isin(sel_models) & df["phase"].isin(sel_phases)]
        am   = [m for m in models if m in sel_models]
        ap   = [p for p in phases  if p in sel_phases]

        def val(m, ph, met=None):
            met = met or metric
            r = data.loc[(data.model == m) & (data.phase == ph), met]
            return float(r.values[0]) if not r.empty else None

        # ── OVERVIEW ─────────────────────────────────────────────
        if tab == "overview":
            best_roc = data.loc[data["roc_auc"].idxmax()] if not data.empty and data["roc_auc"].notna().any() else None
            avg_roc  = data["roc_auc"].mean() if not data.empty else 0

            stats = html.Div([
                mbadge("Results",   len(data),                      C["purple"]),
                mbadge("Best ROC",  f"{best_roc['roc_auc']:.3f}" if best_roc is not None else "—", C["cyan"]),
                mbadge("Best Model",f"{best_roc['model']}"        if best_roc is not None else "—", C["green"]),
                mbadge("Best Phase",f"Ph.{best_roc['phase']}"     if best_roc is not None else "—", C["blue"]),
                mbadge("Avg ROC",   f"{avg_roc:.3f}",               C["yellow"]),
            ], style={"display": "flex", "gap": "10px",
                      "flexWrap": "wrap", "marginBottom": "20px"})

            # Heatmap
            mat = [[val(m, p, "roc_auc") or 0 for p in ap] for m in am]
            hm  = go.Figure(go.Heatmap(
                z=mat, x=[f"Phase {p}" for p in ap], y=am,
                colorscale=[[0, "#1a1a2e"], [0.5, C["purple"]], [1, C["cyan"]]],
                zmin=0.4, zmax=1.0,
                text=[[f"{v:.3f}" for v in row] for row in mat],
                texttemplate="%{text}", textfont=dict(size=13, color="white"),
            ))
            hm.update_layout(**{k: v for k, v in base.items() if k != "margin"},
                              margin=dict(l=120, r=20, t=40, b=50),
                              title="ROC-AUC Heatmap",
                              height=max(200, 60 * len(am) + 100))

            # Line
            line = go.Figure()
            for m in am:
                xs = [f"Phase {p}" for p in ap if val(m, p) is not None]
                ys = [val(m, p) for p in ap if val(m, p) is not None]
                if xs:
                    line.add_trace(go.Scatter(
                        x=xs, y=ys, name=m, mode="lines+markers",
                        line=dict(color=mc[m], width=2),
                        marker=dict(size=9, color=mc[m],
                                    line=dict(color="white", width=1)),
                    ))
            line.add_hline(y=0.5, line_dash="dot", line_color=MUTED,
                            annotation_text="random", annotation_font_color=MUTED)
            line.update_layout(**base, title=f"{mlabel} by Phase",
                                yaxis=dict(range=[0, 1.05], gridcolor=GRID),
                                xaxis=dict(gridcolor=GRID))

            # Radar
            radar = go.Figure()
            for m in am:
                mdf  = data[data["model"] == m]
                vals = [float(mdf[me].mean()) if not mdf.empty else 0 for me in METRICS]
                radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=METRIC_LABELS + [METRIC_LABELS[0]],
                    fill="toself", name=m,
                    line=dict(color=mc[m], width=2), opacity=0.6,
                ))
            radar.update_layout(
                **{k: v for k, v in base.items() if k not in ["xaxis", "yaxis"]},
                title="Avg Metric Profile",
                polar=dict(bgcolor=CARD,
                           radialaxis=dict(visible=True, range=[0, 1],
                                           gridcolor=GRID,
                                           tickfont=dict(color=MUTED, size=9)),
                           angularaxis=dict(gridcolor=GRID)),
            )

            return html.Div([
                stats,
                html.Div([
                    html.Div(dcc.Graph(figure=hm),    style={"flex": "2"}),
                    html.Div(dcc.Graph(figure=radar), style={"flex": "1"}),
                ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),
                dcc.Graph(figure=line),
            ])

        # ── COMPARE ──────────────────────────────────────────────
        elif tab == "compare":
            bar = go.Figure()
            for p in ap:
                bar.add_trace(go.Bar(
                    name=f"Phase {p}", x=am,
                    y=[val(m, p) or 0 for m in am],
                    marker_color=PHASE_COLORS.get(p, C["purple"]), opacity=0.85,
                    text=[f"{val(m,p):.3f}" if val(m,p) else "—" for m in am],
                    textposition="outside",
                ))
            bar.update_layout(**base, title=f"{mlabel} by Model", barmode="group",
                               yaxis=dict(range=[0, 1.15], gridcolor=GRID),
                               xaxis=dict(gridcolor=GRID))

            scatter = go.Figure()
            for m in am:
                mdf = data[data["model"] == m]
                scatter.add_trace(go.Scatter(
                    x=mdf["roc_auc"].tolist(), y=mdf["f1"].tolist(),
                    mode="markers+text", name=m,
                    text=[f"Ph.{p}" for p in mdf["phase"]],
                    textposition="top right",
                    textfont=dict(size=9, color=MUTED),
                    marker=dict(color=mc[m], size=12,
                                line=dict(color="white", width=1)),
                ))
            scatter.add_hline(y=0.5, line_dash="dot", line_color=MUTED)
            scatter.add_vline(x=0.5, line_dash="dot", line_color=MUTED)
            scatter.update_layout(**base, title="F1 vs ROC-AUC",
                                   xaxis=dict(title="ROC-AUC", range=[0.3, 1.05], gridcolor=GRID),
                                   yaxis=dict(title="F1",      range=[0.3, 1.05], gridcolor=GRID))

            box = go.Figure()
            for m in am:
                vals = data[data["model"] == m][metric].dropna().tolist()
                box.add_trace(go.Box(y=vals, name=m, marker_color=mc[m],
                                     line_color=mc[m], opacity=0.7, boxmean=True))
            box.update_layout(**base, title=f"{mlabel} Distribution across Phases",
                               yaxis=dict(range=[0, 1.05], gridcolor=GRID),
                               xaxis=dict(gridcolor=GRID))

            # Table
            th_s = {"padding": "8px 12px", "color": C["purple"],
                    "textAlign": "left", "backgroundColor": GRID, "fontSize": "11px"}
            td_s = {"padding": "7px 12px", "color": TEXT,
                    "borderBottom": f"1px solid {GRID}", "fontSize": "11px"}
            rows = []
            for _, r in data.sort_values(["model", "phase"]).iterrows():
                rows.append(html.Tr([
                    html.Td(html.Span(r["model"],
                                     style={"color": get_model_color(r["model"]),
                                            "fontWeight": "700"}), style=td_s),
                    html.Td(f"Phase {r['phase']}", style=td_s),
                    *[html.Td(f"{r[m]:.4f}" if not pd.isna(r.get(m, np.nan))
                              else "—", style=td_s) for m in METRICS],
                ]))
            table = html.Table(
                [html.Tr([html.Th(c, style=th_s) for c in
                          ["Model", "Phase"] + METRIC_LABELS])] + rows,
                style={"width": "100%", "borderCollapse": "collapse"},
            )

            return html.Div([
                html.Div([
                    html.Div(dcc.Graph(figure=bar),     style={"flex": "1"}),
                    html.Div(dcc.Graph(figure=scatter), style={"flex": "1"}),
                ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),
                dcc.Graph(figure=box, style={"marginBottom": "16px"}),
                card([stitle("Full Results Table"),
                      html.Div(table, style={"overflowX": "auto"})]),
            ])

        # ── TRAINING CURVES ───────────────────────────────────────
        elif tab == "training":
            if not curves:
                return html.P("No loss curves found. Run models first.",
                              style={"color": C["yellow"]})

            figs, conv_rows = [], []
            for model_name in sorted(curves.keys()):
                if model_name not in am:
                    continue
                fig = go.Figure()
                for phase, losses in sorted(curves[model_name].items()):
                    if phase not in ap:
                        continue
                    # Handle both old format (list of floats) and new format (list of dicts)
                    if losses and isinstance(losses[0], dict):
                        train_losses = [entry["train_loss"] for entry in losses]
                        val_losses = [entry["val_loss"] for entry in losses]
                    else:
                        train_losses = losses
                        val_losses = None
                    epochs = list(range(1, len(train_losses) + 1))
                    fig.add_trace(go.Scatter(
                        x=epochs, y=train_losses,
                        mode="lines+markers", name=f"Phase {phase} (train)",
                        line=dict(color=PHASE_COLORS.get(phase, C["purple"]), width=2),
                        marker=dict(size=4),
                        hovertemplate=f"Phase {phase} train<br>Epoch %{{x}}<br>Loss %{{y:.4f}}<extra></extra>",
                    ))
                    if val_losses is not None:
                        fig.add_trace(go.Scatter(
                            x=epochs, y=val_losses,
                            mode="lines+markers", name=f"Phase {phase} (val)",
                            line=dict(color=PHASE_COLORS.get(phase, C["purple"]), width=2, dash="dash"),
                            marker=dict(size=3),
                            hovertemplate=f"Phase {phase} val<br>Epoch %{{x}}<br>Loss %{{y:.4f}}<extra></extra>",
                        ))
                    drop = train_losses[0] - train_losses[-1]
                    pct  = drop / train_losses[0] * 100
                    conv_rows.append(html.Tr([
                        html.Td(html.Span(model_name, style={"color": get_model_color(model_name), "fontWeight": "700"}),
                                style={"padding": "6px 10px", "fontSize": "11px"}),
                        html.Td(f"Phase {phase}", style={"padding": "6px 10px",
                                                          "color": PHASE_COLORS.get(phase, TEXT),
                                                          "fontSize": "11px"}),
                        html.Td(f"{train_losses[0]:.4f}", style={"padding": "6px 10px", "color": TEXT, "fontSize": "11px"}),
                        html.Td(f"{train_losses[-1]:.4f}", style={"padding": "6px 10px", "color": TEXT, "fontSize": "11px"}),
                        html.Td(f"{pct:.1f}%",
                                style={"padding": "6px 10px", "fontSize": "11px",
                                       "color": C["green"] if pct > 5 else C["yellow"],
                                       "fontWeight": "700"}),
                        html.Td("✅" if pct > 5 else "⚠️",
                                style={"padding": "6px 10px", "fontSize": "14px"}),
                    ], style={"borderBottom": f"1px solid {GRID}"}))

                fig.update_layout(**base, title=f"{model_name} — Loss",
                                   xaxis=dict(title="Epoch", gridcolor=GRID),
                                   yaxis=dict(title="Loss",  gridcolor=GRID),
                                   height=280)
                figs.append(html.Div(dcc.Graph(figure=fig),
                                     style={"flex": "1", "minWidth": "280px"}))

            th_s = {"padding": "8px 10px", "color": C["purple"],
                    "textAlign": "left", "backgroundColor": GRID, "fontSize": "11px"}
            conv_table = html.Table([
                html.Tr([html.Th(h, style=th_s) for h in
                         ["Model", "Phase", "Start", "End", "Drop %", "OK?"]])
            ] + conv_rows, style={"width": "100%", "borderCollapse": "collapse"})

            return html.Div([
                html.Div(figs, style={"display": "flex", "gap": "12px",
                                      "flexWrap": "wrap", "marginBottom": "16px"}),
                card([stitle("Convergence Summary"), conv_table]),
            ])

        # ── CONFUSION MATRICES ────────────────────────────────────
        elif tab == "confusion":
            has_preds = ("y_pred" in df.columns and df["y_pred"].notna().any())

            if not has_preds:
                return card([
                    html.P("⚠️ Predictions not saved yet.",
                           style={"color": C["yellow"], "fontSize": "13px"}),
                    html.P("Re-run models with the updated evaluate.py.",
                           style={"color": MUTED, "fontSize": "12px"}),
                ])

            fn_fig = go.Figure()
            for m in am:
                fns, xs = [], []
                for p in ap:
                    r = df[(df["model"] == m) & (df["phase"] == p)]
                    if r.empty or not isinstance(r.iloc[0].get("y_pred"), list):
                        continue
                    cm_v = confusion_matrix(np.array(r.iloc[0]["y_test"]),
                                            np.array(r.iloc[0]["y_pred"]),
                                            labels=[0, 1])
                    fns.append(cm_v.ravel()[2])  # FN
                    xs.append(f"Phase {p}")
                if fns:
                    fn_fig.add_trace(go.Bar(name=m, x=xs, y=fns,
                                            marker_color=mc[m], opacity=0.85,
                                            text=fns, textposition="outside"))
            fn_fig.update_layout(**base,
                                  title="False Negatives — Missed SAEs (lower = better)",
                                  barmode="group",
                                  yaxis=dict(gridcolor=GRID, title="# Missed SAEs"),
                                  xaxis=dict(gridcolor=GRID))

            n_m, n_p = len(am), len(ap)
            if n_m > 0 and n_p > 0:
                cm_fig = make_subplots(rows=n_m, cols=n_p,
                                        subplot_titles=[f"{m} · Ph.{p}"
                                                        for m in am for p in ap])
                labels = ["No SAE", "SAE"]
                for ri, m in enumerate(am, 1):
                    for ci, p in enumerate(ap, 1):
                        r = df[(df["model"] == m) & (df["phase"] == p)]
                        if r.empty or not isinstance(r.iloc[0].get("y_pred"), list):
                            continue
                        cm_v = confusion_matrix(np.array(r.iloc[0]["y_test"]),
                                                np.array(r.iloc[0]["y_pred"]),
                                                labels=[0, 1])
                        pct  = cm_v / cm_v.sum() * 100
                        text = [[f"{cm_v[a][b]}<br>{pct[a][b]:.1f}%"
                                 for b in range(2)] for a in range(2)]
                        cm_fig.add_trace(go.Heatmap(
                            z=cm_v, x=labels, y=labels,
                            text=text, texttemplate="%{text}",
                            colorscale=[[0, CARD], [1, mc[m]]],
                            showscale=False, textfont=dict(size=10, color="white"),
                        ), row=ri, col=ci)
                cm_fig.update_layout(
                    **{k: v for k, v in base.items() if k not in ["xaxis", "yaxis"]},
                    title="Confusion Matrices — Real Predictions",
                    height=max(300, 220 * n_m),
                )
            else:
                cm_fig = go.Figure()

            return html.Div([
                dcc.Graph(figure=fn_fig, style={"marginBottom": "12px"}),
                dcc.Graph(figure=cm_fig),
            ])

        # ── SANITY ───────────────────────────────────────────────
        elif tab == "sanity":
            dummy_rows = []
            try:
                from src.data_loader import load_phase as lp
                for phase in ap:
                    X_tr, X_te, y_tr, y_te, _ = lp(phase)
                    d = DummyClassifier(strategy="most_frequent")
                    d.fit(X_tr, y_tr)
                    yp  = d.predict(X_te)
                    ypr = d.predict_proba(X_te)[:, 1]
                    dummy_rows.append({"phase": phase,
                                       "f1":      f1_score(y_te, yp, zero_division=0),
                                       "roc_auc": roc_auc_score(y_te, ypr)})
            except Exception:
                pass
            dummy_df = pd.DataFrame(dummy_rows)

            roc_fig = go.Figure()
            for m in am:
                mdf = data[data["model"] == m].sort_values("phase")
                roc_fig.add_trace(go.Scatter(
                    x=[f"Phase {p}" for p in mdf["phase"]], y=mdf["roc_auc"],
                    mode="lines+markers", name=m,
                    line=dict(color=mc[m], width=2), marker=dict(size=8)))
            if not dummy_df.empty:
                roc_fig.add_trace(go.Scatter(
                    x=[f"Phase {p}" for p in dummy_df["phase"]],
                    y=dummy_df["roc_auc"], mode="lines+markers",
                    name="Dummy", line=dict(color=MUTED, dash="dash"),
                    marker=dict(size=8, symbol="x")))
            roc_fig.add_hline(y=0.5, line_dash="dot", line_color=C["red"],
                               annotation_text="random (0.5)",
                               annotation_font_color=C["red"])
            roc_fig.update_layout(**base, title="ROC-AUC vs Dummy",
                                   yaxis=dict(range=[0, 1.05], gridcolor=GRID),
                                   xaxis=dict(gridcolor=GRID))

            f1_fig = go.Figure()
            for m in am:
                mdf = data[data["model"] == m].sort_values("phase")
                f1_fig.add_trace(go.Scatter(
                    x=[f"Phase {p}" for p in mdf["phase"]], y=mdf["f1"],
                    mode="lines+markers", name=m,
                    line=dict(color=mc[m], width=2), marker=dict(size=8)))
            if not dummy_df.empty:
                f1_fig.add_trace(go.Scatter(
                    x=[f"Phase {p}" for p in dummy_df["phase"]],
                    y=dummy_df["f1"], mode="lines+markers",
                    name="Dummy", line=dict(color=MUTED, dash="dash"),
                    marker=dict(size=8, symbol="x")))
            f1_fig.update_layout(**base, title="F1 vs Dummy (misleading in Ph.2/3)",
                                  yaxis=dict(range=[0, 1.05], gridcolor=GRID),
                                  xaxis=dict(gridcolor=GRID))

            # Consistency table
            th_s = {"padding": "8px 10px", "color": C["purple"],
                    "backgroundColor": GRID, "textAlign": "left", "fontSize": "11px"}
            td_s = {"padding": "6px 10px", "fontSize": "11px",
                    "borderBottom": f"1px solid {GRID}"}
            check_rows = []
            for _, r in data.sort_values(["model", "phase"]).iterrows():
                p   = r.get("precision", 0)
                rec = r.get("recall", 0)
                f1  = r.get("f1", 0)
                exp = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
                roc = r.get("roc_auc", np.nan)
                f1_ok  = abs(exp - f1) < 0.01
                roc_ok = not np.isnan(roc) and roc >= 0.5
                check_rows.append(html.Tr([
                    html.Td(html.Span(r["model"],
                                     style={"color": get_model_color(r["model"]),
                                            "fontWeight": "700"}), style=td_s),
                    html.Td(f"Phase {r['phase']}", style=td_s),
                    html.Td(f"{f1:.4f}",  style={**td_s, "color": TEXT}),
                    html.Td(f"{exp:.4f}", style={**td_s, "color": MUTED}),
                    html.Td("✅" if f1_ok  else "❌", style={**td_s, "fontSize": "14px"}),
                    html.Td(f"{roc:.4f}" if not np.isnan(roc) else "NaN",
                            style={**td_s, "color": TEXT}),
                    html.Td("✅" if roc_ok else "❌", style={**td_s, "fontSize": "14px"}),
                ]))

            check_table = html.Table([
                html.Tr([html.Th(h, style=th_s) for h in
                         ["Model", "Phase", "F1", "Expected F1",
                          "F1 ✓", "ROC-AUC", "ROC ✓"]])
            ] + check_rows, style={"width": "100%", "borderCollapse": "collapse"})

            return html.Div([
                html.Div([
                    html.Div(dcc.Graph(figure=roc_fig), style={"flex": "1"}),
                    html.Div(dcc.Graph(figure=f1_fig),  style={"flex": "1"}),
                ], style={"display": "flex", "gap": "12px", "marginBottom": "16px"}),
                card([stitle("Metric Consistency"),
                      html.Div(check_table, style={"overflowX": "auto"})]),
                card([
                    stitle("Notes on Class Imbalance", C["yellow"]),
                    html.Ul([
                        html.Li("Phase 2 (74.6% SAE) and Phase 3 (84.7% SAE): "
                                "a dummy predicting always SAE achieves F1 ≈ 0.85 / 0.92. "
                                "Models scoring lower in F1 is EXPECTED.",
                                style={"color": TEXT, "fontSize": "12px", "marginBottom": "6px"}),
                        html.Li("Use ROC-AUC as primary metric — it is unaffected by class imbalance. "
                                "Dummy always scores 0.5.",
                                style={"color": TEXT, "fontSize": "12px"}),
                    ]),
                ], style={"marginTop": "12px"}),
            ])

        return html.P("Select a tab.", style={"color": MUTED})

    print("\n  Dashboard -> http://127.0.0.1:8050\n")
    app.run(debug=False, use_reloader=False)


if __name__ == "__main__":
    run()
