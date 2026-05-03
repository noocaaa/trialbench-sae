import json, glob, os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, callback

METRICS       = ["accuracy", "f1", "precision", "recall", "roc_auc", "pr_auc"]
METRIC_LABELS = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC", "PR-AUC"]
COLORS        = ["#7c6af7", "#f7916a", "#6af7c8", "#f7e06a", "#f76aab", "#6ab4f7"]
BG, SURFACE   = "#0f0f17", "#1a1a2e"


def load_results(results_dir="results"):
    files = glob.glob(os.path.join(results_dir, "*.json"))
    if not files:
        raise FileNotFoundError("No results found in results/")
    df = pd.DataFrame([json.load(open(f)) for f in files])
    df["phase"] = df["phase"].astype(str)
    return df


def run(results_dir="results"):
    df     = load_results(results_dir)
    models = sorted(df["model"].unique())
    phases = sorted(df["phase"].unique())
    mc     = {m: COLORS[i % len(COLORS)] for i, m in enumerate(models)}

    base_layout = dict(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color="#ccccee", size=10),
        margin=dict(l=45, r=15, t=40, b=45),
        legend=dict(bgcolor=SURFACE, bordercolor="#333355",
                    font=dict(color="#ccccee", size=10)),
    )

    app = dash.Dash(__name__)

    app.layout = html.Div(
        style={"backgroundColor": BG, "minHeight": "100vh",
               "fontFamily": "monospace", "padding": "16px"},
        children=[
            html.H2("SAE Prediction — Results Dashboard",
                    style={"color": "white", "textAlign": "center", "marginBottom": "16px"}),

            # Controls
            html.Div(
                style={"display": "flex", "gap": "40px", "flexWrap": "wrap",
                       "backgroundColor": SURFACE, "padding": "16px",
                       "borderRadius": "10px", "marginBottom": "20px"},
                children=[
                    html.Div([
                        html.B("Metric", style={"color": "#7c6af7", "display": "block", "marginBottom": "6px"}),
                        dcc.RadioItems(
                            id="metric",
                            options=[{"label": l, "value": v} for l, v in zip(METRIC_LABELS, METRICS)],
                            value="f1",
                            labelStyle={"display": "block", "color": "#ccccee", "marginBottom": "5px"},
                            inputStyle={"marginRight": "6px"},
                        ),
                    ]),
                    html.Div([
                        html.B("Models", style={"color": "#6af7c8", "display": "block", "marginBottom": "6px"}),
                        dcc.Checklist(
                            id="sel-models",
                            options=[{"label": m, "value": m} for m in models],
                            value=list(models),
                            labelStyle={"display": "block", "color": "#ccccee", "marginBottom": "5px"},
                            inputStyle={"marginRight": "6px"},
                        ),
                    ]),
                    html.Div([
                        html.B("Phases", style={"color": "#f7c86a", "display": "block", "marginBottom": "6px"}),
                        dcc.Checklist(
                            id="sel-phases",
                            options=[{"label": f"Phase {p}", "value": p} for p in phases],
                            value=list(phases),
                            labelStyle={"display": "block", "color": "#ccccee", "marginBottom": "5px"},
                            inputStyle={"marginRight": "6px"},
                        ),
                    ]),
                ]
            ),

            # Charts grid
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px"},
                children=[
                    dcc.Graph(id="bar-chart"),
                    dcc.Graph(id="radar-chart"),
                    dcc.Graph(id="line-chart"),
                    dcc.Graph(id="box-chart"),
                    dcc.Graph(id="heat-chart"),
                    dcc.Graph(id="scatter-chart"),
                ]
            ),
        ]
    )

    @callback(
        Output("bar-chart",     "figure"),
        Output("radar-chart",   "figure"),
        Output("line-chart",    "figure"),
        Output("box-chart",     "figure"),
        Output("heat-chart",    "figure"),
        Output("scatter-chart", "figure"),
        Input("metric",      "value"),
        Input("sel-models",  "value"),
        Input("sel-phases",  "value"),
    )
    def update(met, sel_models, sel_phases):
        sel_models = sel_models or []
        sel_phases = sel_phases or []
        data   = df[df["model"].isin(sel_models) & df["phase"].isin(sel_phases)]
        am     = [m for m in models if m in sel_models]
        ap     = [p for p in phases if p in sel_phases]
        mlabel = METRIC_LABELS[METRICS.index(met)]

        def val(m, ph):
            r = data.loc[(data.model == m) & (data.phase == ph), met]
            return float(r.values[0]) if not r.empty else 0.0

        # ── Bar ───────────────────────────────────────────────────
        bar = go.Figure()
        for ph in ap:
            bar.add_trace(go.Bar(
                name=f"Ph.{ph}", x=am,
                y=[val(m, ph) for m in am], opacity=0.85,
            ))
        bar.update_layout(**base_layout, title=f"{mlabel} by Model",
                          barmode="group",
                          yaxis=dict(range=[0, 1.05], gridcolor="#333355"),
                          xaxis=dict(gridcolor="#333355"))

        # ── Radar ─────────────────────────────────────────────────
        radar = go.Figure()
        for m in am:
            mdf  = data[data.model == m]
            vals = [float(mdf[me].mean()) if not mdf.empty else 0.0 for me in METRICS]
            radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=METRIC_LABELS + [METRIC_LABELS[0]],
                fill="toself", name=m,
                line=dict(color=mc[m]), opacity=0.75,
            ))
        radar.update_layout(
            **{k: v for k, v in base_layout.items() if k not in ["xaxis", "yaxis"]},
            title="Avg Metric Profile",
            polar=dict(
                bgcolor=SURFACE,
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#333355",
                                tickfont=dict(color="#666688")),
                angularaxis=dict(gridcolor="#333355"),
            ),
        )

        # ── Line ──────────────────────────────────────────────────
        line = go.Figure()
        for m in am:
            xs, ys = [], []
            for ph in ap:
                r = data.loc[(data.model == m) & (data.phase == ph), met]
                if not r.empty:
                    xs.append(f"Ph.{ph}")
                    ys.append(float(r.values[0]))
            if xs:
                line.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines+markers", name=m,
                    line=dict(color=mc[m], width=2),
                    marker=dict(size=8, color=mc[m]),
                ))
        line.update_layout(**base_layout, title=f"{mlabel} by Phase",
                           yaxis=dict(range=[0, 1.05], gridcolor="#333355"),
                           xaxis=dict(gridcolor="#333355"))

        # ── Box ───────────────────────────────────────────────────
        box = go.Figure()
        for m in am:
            vals = data[data.model == m][met].dropna().tolist()
            box.add_trace(go.Box(
                y=vals, name=m, marker_color=mc[m],
                line_color=mc[m], fillcolor=mc[m],
                opacity=0.7, boxmean=True,
            ))
        box.update_layout(**base_layout, title=f"{mlabel} Distribution",
                          yaxis=dict(range=[0, 1.05], gridcolor="#333355"),
                          xaxis=dict(gridcolor="#333355"))

        # ── Heatmap ───────────────────────────────────────────────
        mat  = [[val(m, ph) for ph in ap] for m in am]
        heat = go.Figure(go.Heatmap(
            z=mat,
            x=[f"Ph.{p}" for p in ap],
            y=am,
            colorscale="Plasma", zmin=0, zmax=1,
            text=[[f"{v:.3f}" for v in row] for row in mat],
            texttemplate="%{text}",
            textfont=dict(size=11, color="white"),
        ))
        heat.update_layout(
            **{k: v for k, v in base_layout.items() if k not in ["xaxis", "yaxis"]},
            title=f"Heatmap — {mlabel}",
        )

        # ── Scatter ───────────────────────────────────────────────
        scatter = go.Figure()
        for m in am:
            mdf = data[data.model == m]
            if not mdf.empty:
                scatter.add_trace(go.Scatter(
                    x=mdf["roc_auc"].tolist(),
                    y=mdf["f1"].tolist(),
                    mode="markers+text",
                    name=m,
                    text=[f"Ph.{p}" for p in mdf["phase"]],
                    textposition="top right",
                    textfont=dict(size=9, color="#aaaacc"),
                    marker=dict(color=mc[m], size=10,
                                line=dict(color="white", width=1)),
                ))
        scatter.update_layout(
            **base_layout,
            title="F1 vs ROC-AUC",
            xaxis=dict(title="ROC-AUC", range=[0, 1.05], gridcolor="#333355"),
            yaxis=dict(title="F1",      range=[0, 1.05], gridcolor="#333355"),
        )

        return bar, radar, line, box, heat, scatter

    print("\n  Dashboard at → http://127.0.0.1:8050\n")
    app.run(debug=False, use_reloader=False)


if __name__ == "__main__":
    run()


# ── Sanity check tab (separate Dash app page) ──────────────────────
def run_sanity_dashboard(results_dir="results"):
    """
    Standalone sanity check dashboard.
    Shows: confusion matrices, metric consistency, dummy comparison.
    Run with: python plot_results.py --sanity
    """
    import glob, json
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from data_loader import load_phase

    PHASES = ["1", "2", "3", "4"]
    COLORS = ["#7c6af7", "#f7916a", "#6af7c8", "#f7e06a"]

    # ── Pre-compute all data ───────────────────────────────────────
    stats, dummy_rows, cm_data = [], [], {}

    for phase in PHASES:
        try:
            X_train, X_test, y_train, y_test, pw = load_phase(phase)
            n, pos = len(y_test), int(y_test.sum())
            stats.append({"phase": phase, "n": n, "pos": pos,
                           "neg": n-pos, "ratio": pos/n,
                           "dummy_acc": max(pos/n, 1-pos/n)})

            # Dummy
            d = DummyClassifier(strategy="most_frequent")
            d.fit(X_train, y_train)
            yp = d.predict(X_test)
            ypr = d.predict_proba(X_test)[:, 1]
            from sklearn.metrics import f1_score, roc_auc_score
            dummy_rows.append({
                "phase": phase,
                "f1":      f1_score(y_test, yp, zero_division=0),
                "roc_auc": roc_auc_score(y_test, ypr) if len(np.unique(y_test)) > 1 else float("nan"),
            })

            # Confusion matrix via LR proxy
            clf = LogisticRegression(max_iter=500, class_weight="balanced")
            clf.fit(X_train, y_train)
            cm_data[phase] = confusion_matrix(y_test, clf.predict(X_test))
        except Exception as e:
            print(f"Phase {phase} error: {e}")

    stats_df = pd.DataFrame(stats)
    dummy_df = pd.DataFrame(dummy_rows)

    # Load real results
    real = pd.DataFrame([json.load(open(f))
                         for f in glob.glob(os.path.join(results_dir, "*.json"))
                         if "Dummy" not in f])
    if not real.empty:
        real["phase"] = real["phase"].astype(str)

    app = dash.Dash(__name__)

    # ── Figures ───────────────────────────────────────────────────
    base = dict(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                font=dict(color="#ccccee", size=10),
                margin=dict(l=45, r=15, t=40, b=45))

    # Class balance bars
    bal_fig = go.Figure()
    bal_fig.add_trace(go.Bar(name="Positives (SAE)",
                              x=[f"Ph.{r['phase']}" for _, r in stats_df.iterrows()],
                              y=[r["pos"] for _, r in stats_df.iterrows()],
                              marker_color="#7c6af7"))
    bal_fig.add_trace(go.Bar(name="Negatives",
                              x=[f"Ph.{r['phase']}" for _, r in stats_df.iterrows()],
                              y=[r["neg"] for _, r in stats_df.iterrows()],
                              marker_color="#f7916a"))
    bal_fig.update_layout(**base, title="Class Balance per Phase",
                           barmode="stack",
                           yaxis=dict(gridcolor="#333355"),
                           xaxis=dict(gridcolor="#333355"))

    # Dummy vs models F1
    vs_fig = go.Figure()
    if not real.empty:
        models = sorted(real["model"].unique())
        mc     = {m: COLORS[i % len(COLORS)] for i, m in enumerate(models)}
        for m in models:
            mdf = real[real["model"] == m].sort_values("phase")
            vs_fig.add_trace(go.Scatter(
                x=[f"Ph.{p}" for p in mdf["phase"]], y=mdf["f1"],
                mode="lines+markers", name=m,
                line=dict(color=mc[m], width=2), marker=dict(size=8),
            ))
    if not dummy_df.empty:
        vs_fig.add_trace(go.Scatter(
            x=[f"Ph.{p}" for p in dummy_df["phase"]], y=dummy_df["f1"],
            mode="lines+markers", name="Dummy (baseline)",
            line=dict(color="grey", width=2, dash="dash"),
            marker=dict(size=8, symbol="x"),
        ))
    vs_fig.update_layout(**base, title="F1 — Your Models vs Dummy Baseline",
                          yaxis=dict(range=[0, 1.05], gridcolor="#333355"),
                          xaxis=dict(gridcolor="#333355"))

    # ROC-AUC vs dummy
    roc_fig = go.Figure()
    if not real.empty:
        for m in sorted(real["model"].unique()):
            mdf = real[real["model"] == m].sort_values("phase")
            roc_fig.add_trace(go.Scatter(
                x=[f"Ph.{p}" for p in mdf["phase"]], y=mdf["roc_auc"],
                mode="lines+markers", name=m,
                line=dict(color=mc[m], width=2), marker=dict(size=8),
            ))
    # Random baseline = 0.5
    roc_fig.add_hline(y=0.5, line_dash="dash", line_color="grey",
                       annotation_text="Random (0.5)", annotation_position="top left")
    roc_fig.update_layout(**base, title="ROC-AUC — Must Stay Above 0.5",
                           yaxis=dict(range=[0, 1.05], gridcolor="#333355"),
                           xaxis=dict(gridcolor="#333355"))

    # Confusion matrices
    from plotly.subplots import make_subplots
    cm_fig = make_subplots(rows=1, cols=len(cm_data),
                            subplot_titles=[f"Phase {p}" for p in cm_data])
    for i, (phase, cm) in enumerate(cm_data.items(), 1):
        labels = ["No SAE", "SAE"]
        pct    = cm / cm.sum() * 100
        text   = [[f"{cm[r][c]}<br>({pct[r][c]:.1f}%)" for c in range(2)] for r in range(2)]
        cm_fig.add_trace(go.Heatmap(
            z=cm, x=labels, y=labels,
            text=text, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            textfont=dict(size=11),
        ), row=1, col=i)
    cm_fig.update_layout(**{k:v for k,v in base.items() if k not in ["xaxis","yaxis"]},
                          title="Confusion Matrices — Logistic Regression (proxy)")

    # Metric consistency table
    consistency_rows = []
    if not real.empty:
        for _, r in real.iterrows():
            p, rec, f1 = r.get("precision",0), r.get("recall",0), r.get("f1",0)
            exp_f1 = 2*p*rec/(p+rec) if (p+rec) > 0 else 0
            roc    = r.get("roc_auc", float("nan"))
            ok     = abs(exp_f1 - f1) < 0.01 and (np.isnan(roc) or roc >= 0.5)
            consistency_rows.append({
                "Model": r["model"], "Phase": r["phase"],
                "F1": f"{f1:.4f}", "Expected F1": f"{exp_f1:.4f}",
                "Match": "✅" if abs(exp_f1-f1) < 0.01 else "❌",
                "ROC-AUC": f"{roc:.4f}" if not np.isnan(roc) else "NaN",
                "ROC OK": "✅" if not np.isnan(roc) and roc >= 0.5 else "❌",
            })

    # ── Layout ────────────────────────────────────────────────────
    def card(children, title=None):
        return html.Div([
            html.B(title, style={"color": "#7c6af7", "display":"block", "marginBottom":"8px"}) if title else None,
            *([children] if not isinstance(children, list) else children)
        ], style={"backgroundColor": SURFACE, "padding": "16px",
                  "borderRadius": "10px", "marginBottom": "16px"})

    table_rows = [html.Tr([html.Th(c, style={"color":"#aaaacc","padding":"6px 10px","borderBottom":"1px solid #333355"})
                           for c in ["Model","Phase","F1","Expected F1","Match","ROC-AUC","ROC OK"]])]
    for row in consistency_rows:
        table_rows.append(html.Tr([
            html.Td(row[c], style={"padding":"5px 10px","color":"#ccccee","borderBottom":"1px solid #1a1a2e"})
            for c in ["Model","Phase","F1","Expected F1","Match","ROC-AUC","ROC OK"]
        ]))

    app.layout = html.Div(
        style={"backgroundColor": BG, "minHeight":"100vh", "fontFamily":"monospace", "padding":"16px"},
        children=[
            html.H2("SAE Prediction — Sanity Check Dashboard",
                    style={"color":"white","textAlign":"center","marginBottom":"16px"}),

            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px"}, children=[
                dcc.Graph(figure=bal_fig),
                dcc.Graph(figure=vs_fig),
                dcc.Graph(figure=roc_fig),
                dcc.Graph(figure=cm_fig),
            ]),

            card(html.Table(table_rows,
                            style={"width":"100%","borderCollapse":"collapse"}),
                 title="Metric Consistency Check"),
        ]
    )

    print("\n  Sanity Dashboard at → http://127.0.0.1:8051\n")
    app.run(debug=False, use_reloader=False, port=8051)


if __name__ == "__main__":
    import sys
    if "--sanity" in sys.argv:
        run_sanity_dashboard()
    else:
        run()