"""
notebooks/eda.py — Exploratory Data Analysis for SAE Prediction
Run with: python notebooks/eda.py
Opens an interactive Dash dashboard in your browser.

Tabs:
  1. Overview      — class balance, train/test split, SAE rate per phase
  2. Features      — missing values, feature importance, distribution explorer
  3. Correlations  — heatmap + interpretation
  4. Data Dictionary — full column reference with type, source, nulls, usage
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback

# ── Theme ─────────────────────────────────────────────────────────
BG      = "#0f0f17"
SURFACE = "#1a1a2e"
CARD    = "#16213e"
GRID    = "#2a2a44"
BORDER  = "#3a3a5c"
TEXT    = "#ccccee"
MUTED   = "#7777aa"

C_PURPLE = "#7c6af7"
C_ORANGE = "#f7916a"
C_GREEN  = "#6af7c8"
C_YELLOW = "#f7e06a"
C_PINK   = "#f76aab"

PHASES = ["1", "2", "3", "4"]
PHASE_COLORS = {"1": C_PURPLE, "2": C_ORANGE, "3": C_GREEN, "4": C_YELLOW}

base_layout = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(color=TEXT, size=11),
    margin=dict(l=50, r=20, t=45, b=50),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(color=TEXT)),
)

# ── Data path ─────────────────────────────────────────────────────
_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_ROOT, "data", "serious-adverse-event-forecasting")

# ── Column metadata ───────────────────────────────────────────────
TEXT_COLS = [
    "brief_summary/textblock", "brief_title", "condition",
    "condition_browse/mesh_term", "eligibility/criteria/textblock",
    "intervention/description", "intervention/intervention_name",
    "intervention_browse/mesh_term", "keyword",
    "location/facility/address/city", "responsible_party/responsible_party_type",
    "smiless", "icdcode", "study_design_info/intervention_model_description",
    "study_design_info/masking_description", "patient_data/sharing_ipd",
]

DESCRIPTIONS = {
    "Active Comparator Arm Number":            ("Numeric",     "Derived", "Arms using an active treatment as comparator"),
    "Experimental Arm Number":                 ("Numeric",     "Derived", "Arms testing the experimental treatment"),
    "Placebo Comparator Arm Number":           ("Numeric",     "Derived", "Arms using a placebo"),
    "No Intervention Arm Number":              ("Numeric",     "Derived", "Arms with no intervention at all"),
    "Sham Comparator Arm Number":              ("Numeric",     "Derived", "Arms using a fake/sham procedure"),
    "Other Arm Number":                        ("Numeric",     "Derived", "Arms classified as other"),
    "Drug intervention Number":                ("Numeric",     "Derived", "Number of drug interventions"),
    "Biological intervention Number":          ("Numeric",     "Derived", "Biological interventions (antibodies, gene therapy...)"),
    "Behavioral intervention Number":          ("Numeric",     "Derived", "Behavioral/psychological interventions"),
    "Device intervention Number":              ("Numeric",     "Derived", "Medical device interventions"),
    "Diagnostic Test intervention Number":     ("Numeric",     "Derived", "Diagnostic test interventions"),
    "Dietary Supplement intervention Number":  ("Numeric",     "Derived", "Dietary supplement interventions"),
    "Genetic intervention Number":             ("Numeric",     "Derived", "Genetic interventions (high risk)"),
    "Procedure intervention Number":           ("Numeric",     "Derived", "Surgical/medical procedure interventions"),
    "Radiation intervention Number":           ("Numeric",     "Derived", "Radiation therapy interventions"),
    "Combination Product intervention Number": ("Numeric",     "Derived", "Combination product interventions"),
    "Other intervention Number":               ("Numeric",     "Derived", "Other intervention types"),
    "MaskingType-Participant":                 ("Binary",      "Derived", "1 = participants blinded to treatment"),
    "MaskingType-Investigator":                ("Binary",      "Derived", "1 = investigators blinded"),
    "MaskingType-Care Provider":               ("Binary",      "Derived", "1 = care providers blinded"),
    "MaskingType-Outcomes Assessor":           ("Binary",      "Derived", "1 = outcomes assessors blinded"),
    "study_design_info/masking_num":           ("Numeric",     "Derived", "Total blinded parties (0=open label, 4=quadruple blind)"),
    "number_of_arms":                          ("Numeric",     "Raw",     "Total number of arms/groups in the trial"),
    "enrollment":                              ("Numeric",     "Raw",     "Number of participants enrolled"),
    "phase":                                   ("Categorical", "Raw",     "Trial phase — always same value per dataset (not useful)"),
    "study_type":                              ("Categorical", "Raw",     "Always Interventional — no variation (not useful)"),
    "study_design_info/allocation":            ("Categorical", "Raw",     "Randomized or Non-Randomized assignment"),
    "study_design_info/intervention_model":    ("Categorical", "Raw",     "Model: Parallel, Crossover, Factorial, etc."),
    "study_design_info/intervention_model_description": ("Text","Raw",   "Free text model description (84% missing)"),
    "study_design_info/masking":               ("Categorical", "Raw",     "Full masking text (e.g. Quadruple)"),
    "study_design_info/masking_description":   ("Text",        "Raw",     "Free text masking details (92% missing)"),
    "study_design_info/primary_purpose":       ("Categorical", "Raw",     "Purpose: Treatment, Prevention, Diagnostic, etc."),
    "eligibility/gender":                      ("Categorical", "Raw",     "Eligible gender: All, Male, or Female"),
    "eligibility/healthy_volunteers":          ("Categorical", "Raw",     "Whether healthy volunteers are accepted"),
    "eligibility/minimum_age":                 ("Categorical", "Raw",     "Minimum participant age"),
    "eligibility/maximum_age":                 ("Categorical", "Raw",     "Maximum age (32% missing = no upper limit)"),
    "eligibility/criteria/textblock":          ("Text",        "Raw",     "Full inclusion/exclusion criteria — very informative"),
    "smiless":                                 ("Text",        "Raw",     "SMILES chemical structure of the drug (56% missing)"),
    "icdcode":                                 ("Text",        "Raw",     "ICD-10 disease codes (17% missing)"),
    "condition":                               ("Text",        "Raw",     "Disease or condition being studied"),
    "condition_browse/mesh_term":              ("Text",        "Raw",     "MeSH medical terms for condition (29% missing)"),
    "intervention/intervention_name":          ("Text",        "Raw",     "Name of the drug/intervention"),
    "intervention/description":                ("Text",        "Raw",     "Description of the intervention (6% missing)"),
    "intervention_browse/mesh_term":           ("Text",        "Raw",     "MeSH terms for intervention (40% missing)"),
    "brief_title":                             ("Text",        "Raw",     "Short trial title — very informative for SAE"),
    "brief_summary/textblock":                 ("Text",        "Raw",     "Full trial summary — very informative for SAE"),
    "keyword":                                 ("Text",        "Raw",     "Keywords (44% missing)"),
    "has_expanded_access":                     ("Categorical", "Raw",     "Whether expanded access is available"),
    "oversight_info/has_dmc":                  ("Categorical", "Raw",     "Whether a Data Monitoring Committee exists"),
    "oversight_info/is_fda_regulated_drug":    ("Categorical", "Raw",     "Whether the drug is FDA regulated"),
    "oversight_info/is_fda_regulated_device":  ("Categorical", "Raw",     "Whether the device is FDA regulated"),
    "sponsors/lead_sponsor/agency_class":      ("Categorical", "Raw",     "Sponsor: Industry, NIH, Other Gov, Individual"),
    "responsible_party/responsible_party_type":("Categorical", "Raw",     "Responsible party type"),
    "location/facility/address/city":          ("Categorical", "Raw",     "City where the trial is conducted"),
    "ipd_info_type-Analytic Code":             ("Binary",      "Derived", "Analytic code shared? (87% missing)"),
    "ipd_info_type-Clinical Study Report (CSR)":("Binary",    "Derived", "CSR shared? (87% missing)"),
    "ipd_info_type-Informed Consent Form (ICF)":("Binary",    "Derived", "ICF shared? (87% missing)"),
    "ipd_info_type-Statistical Analysis Plan (SAP)":("Binary","Derived", "SAP shared? (87% missing)"),
    "ipd_info_type-Study Protocol":            ("Binary",      "Derived", "Study protocol shared? (87% missing)"),
    "patient_data/sharing_ipd":                ("Categorical", "Raw",     "Whether individual patient data is shared"),
}

TYPE_COLORS   = {"Numeric": C_PURPLE, "Binary": C_GREEN, "Categorical": C_ORANGE, "Text": C_YELLOW}
SOURCE_COLORS = {"Derived": "#a78bfa", "Raw": "#94a3b8"}

GROUPS = {
    "Arm Counts": ["Active Comparator Arm Number","Experimental Arm Number",
                   "Placebo Comparator Arm Number","No Intervention Arm Number",
                   "Sham Comparator Arm Number","Other Arm Number"],
    "Intervention Types": ["Drug intervention Number","Biological intervention Number",
                           "Behavioral intervention Number","Device intervention Number",
                           "Diagnostic Test intervention Number",
                           "Dietary Supplement intervention Number",
                           "Genetic intervention Number","Procedure intervention Number",
                           "Radiation intervention Number",
                           "Combination Product intervention Number",
                           "Other intervention Number"],
    "Masking": ["MaskingType-Participant","MaskingType-Investigator",
                "MaskingType-Care Provider","MaskingType-Outcomes Assessor",
                "study_design_info/masking_num","study_design_info/masking",
                "study_design_info/masking_description"],
    "Trial Design": ["number_of_arms","enrollment","phase","study_type",
                     "study_design_info/allocation",
                     "study_design_info/intervention_model",
                     "study_design_info/intervention_model_description",
                     "study_design_info/primary_purpose"],
    "Eligibility": ["eligibility/gender","eligibility/healthy_volunteers",
                    "eligibility/minimum_age","eligibility/maximum_age",
                    "eligibility/criteria/textblock"],
    "Drug & Disease": ["smiless","icdcode","condition","condition_browse/mesh_term",
                       "intervention/intervention_name","intervention/description",
                       "intervention_browse/mesh_term"],
    "Text & Summary": ["brief_title","brief_summary/textblock","keyword"],
    "Administrative": ["has_expanded_access","oversight_info/has_dmc",
                        "oversight_info/is_fda_regulated_drug",
                        "oversight_info/is_fda_regulated_device",
                        "sponsors/lead_sponsor/agency_class",
                        "responsible_party/responsible_party_type",
                        "location/facility/address/city"],
    "IPD Sharing": ["ipd_info_type-Analytic Code",
                    "ipd_info_type-Clinical Study Report (CSR)",
                    "ipd_info_type-Informed Consent Form (ICF)",
                    "ipd_info_type-Statistical Analysis Plan (SAP)",
                    "ipd_info_type-Study Protocol",
                    "patient_data/sharing_ipd"],
}

GROUP_COLORS = {
    "Arm Counts": C_PURPLE, "Intervention Types": C_ORANGE,
    "Masking": C_GREEN, "Trial Design": C_YELLOW,
    "Eligibility": C_PINK, "Drug & Disease": "#6af7f7",
    "Text & Summary": "#f7c86a", "Administrative": "#a78bfa",
    "IPD Sharing": "#94a3b8",
}


def load_phase_raw(phase):
    bp = os.path.join(_DATA_DIR, f"Phase{phase}")
    X  = pd.read_csv(os.path.join(bp, "train_x.csv"))
    y  = pd.read_csv(os.path.join(bp, "train_y.csv"))
    Xt = pd.read_csv(os.path.join(bp, "test_x.csv"))
    yt = pd.read_csv(os.path.join(bp, "test_y.csv"))
    X["split"] = "train";  Xt["split"] = "test"
    X["label"] = y["Y/N"].values; Xt["label"] = yt["Y/N"].values
    return pd.concat([X, Xt], ignore_index=True)


def get_num_cols(df):
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and c not in ["label", "Unnamed: 0"]]


def card(children, extra=None):
    s = {"backgroundColor": CARD, "borderRadius": "12px", "padding": "16px",
         "border": f"1px solid {BORDER}"}
    if extra:
        s.update(extra)
    return html.Div(children, style=s)


def sec(text, color=C_PURPLE):
    return html.H3(text, style={"color": color, "marginBottom": "10px", "marginTop": "0",
                                 "fontSize": "13px", "letterSpacing": "1px"})


def badge(text, color):
    return html.Span(text, style={"backgroundColor": color, "color": "white",
                                   "padding": "2px 9px", "borderRadius": "10px",
                                   "fontSize": "10px", "marginRight": "6px"})


def render_all_phases(tab, dfs):
    """Render overview of all 4 phases combined."""
    import pandas as pd

    # Combined dataframe with phase column
    frames = []
    for ph, df in dfs.items():
        d = df.copy()
        d["phase_num"] = ph
        frames.append(d)
    all_df = pd.concat(frames, ignore_index=True)

    total  = len(all_df)
    pos_n  = int(all_df["label"].sum())
    neg_n  = total - pos_n

    # ── Stats per phase ────────────────────────────────────────────
    phase_stats = []
    for ph, df in dfs.items():
        n   = len(df)
        pos = int(df["label"].sum())
        neg = n - pos
        tr  = int((df["split"] == "train").sum())
        te  = int((df["split"] == "test").sum())
        phase_stats.append({
            "phase": f"Phase {ph}", "n": n, "pos": pos, "neg": neg,
            "pos_pct": pos/n*100, "train": tr, "test": te,
            "color": PHASE_COLORS[ph],
        })

    # ── Chart 1: Dataset size per phase ───────────────────────────
    size_fig = go.Figure()
    for s in phase_stats:
        size_fig.add_trace(go.Bar(
            name=s["phase"], x=[s["phase"]],
            y=[s["n"]], marker_color=s["color"], opacity=0.85,
            text=[f"{s['n']:,}"], textposition="outside",
        ))
    size_fig.update_layout(
        **{k: v for k, v in base_layout.items() if k != "margin"},
        margin=dict(l=40, r=20, t=50, b=40),
        title="Dataset Size per Phase", showlegend=False,
        yaxis=dict(gridcolor=GRID, title="# Trials"),
        xaxis=dict(gridcolor=GRID),
    )

    # ── Chart 2: SAE rate per phase (stacked bar) ─────────────────
    rate_fig = go.Figure()
    rate_fig.add_trace(go.Bar(
        name="SAE (positive)",
        x=[s["phase"] for s in phase_stats],
        y=[s["pos_pct"] for s in phase_stats],
        marker_color=C_PURPLE, opacity=0.85,
        text=[f"{s['pos_pct']:.1f}%" for s in phase_stats],
        textposition="inside",
    ))
    rate_fig.add_trace(go.Bar(
        name="No SAE (negative)",
        x=[s["phase"] for s in phase_stats],
        y=[100 - s["pos_pct"] for s in phase_stats],
        marker_color=C_ORANGE, opacity=0.85,
        text=[f"{100-s['pos_pct']:.1f}%" for s in phase_stats],
        textposition="inside",
    ))
    rate_fig.update_layout(
        **{k: v for k, v in base_layout.items() if k != "margin"},
        margin=dict(l=40, r=20, t=50, b=40),
        title="SAE vs No-SAE Rate per Phase (%)",
        barmode="stack",
        yaxis=dict(gridcolor=GRID, title="%"),
        xaxis=dict(gridcolor=GRID),
    )

    # ── Chart 3: Train vs Test per phase ──────────────────────────
    split_fig = go.Figure()
    split_fig.add_trace(go.Bar(
        name="Train",
        x=[s["phase"] for s in phase_stats],
        y=[s["train"] for s in phase_stats],
        marker_color=C_GREEN, opacity=0.85,
        text=[s["train"] for s in phase_stats], textposition="inside",
    ))
    split_fig.add_trace(go.Bar(
        name="Test",
        x=[s["phase"] for s in phase_stats],
        y=[s["test"] for s in phase_stats],
        marker_color=C_YELLOW, opacity=0.85,
        text=[s["test"] for s in phase_stats], textposition="inside",
    ))
    split_fig.update_layout(
        **{k: v for k, v in base_layout.items() if k != "margin"},
        margin=dict(l=40, r=20, t=50, b=40),
        title="Train / Test Split per Phase",
        barmode="stack",
        yaxis=dict(gridcolor=GRID, title="# Trials"),
        xaxis=dict(gridcolor=GRID),
    )

    # ── Chart 4: Overall pie ──────────────────────────────────────
    pie = go.Figure(go.Pie(
        labels=["No SAE", "SAE"],
        values=[neg_n, pos_n],
        marker_colors=[C_ORANGE, C_PURPLE], hole=0.45,
        textinfo="label+percent+value", textfont=dict(size=12),
    ))
    pie.update_layout(
        **{k: v for k, v in base_layout.items() if k != "margin"},
        margin=dict(l=10, r=10, t=40, b=10),
        title="Overall Class Balance (All Phases)",
    )

    # ── Summary stats table ───────────────────────────────────────
    th_s = {"padding": "8px 12px", "color": C_PURPLE, "textAlign": "left",
            "backgroundColor": GRID, "fontSize": "11px"}
    td_s = {"padding": "7px 12px", "fontSize": "11px", "borderBottom": f"1px solid {GRID}"}

    rows = []
    for s in phase_stats:
        bal = "Balanced" if 0.4 < s["pos_pct"]/100 < 0.6 else "Imbalanced"
        bal_c = C_GREEN if bal == "Balanced" else C_ORANGE
        rows.append(html.Tr([
            html.Td(html.Span(s["phase"], style={"color": s["color"],
                                                   "fontWeight": "bold"}), style=td_s),
            html.Td(f"{s['n']:,}",  style={**td_s, "color": TEXT}),
            html.Td(f"{s['train']:,}", style={**td_s, "color": C_GREEN}),
            html.Td(f"{s['test']:,}",  style={**td_s, "color": C_YELLOW}),
            html.Td(f"{s['pos']:,} ({s['pos_pct']:.1f}%)", style={**td_s, "color": C_PURPLE}),
            html.Td(f"{s['neg']:,} ({100-s['pos_pct']:.1f}%)", style={**td_s, "color": C_ORANGE}),
            html.Td(html.Span(bal, style={"color": bal_c, "fontWeight": "bold",
                                           "fontSize": "11px"}), style=td_s),
        ]))

    # Total row
    rows.append(html.Tr([
        html.Td("TOTAL", style={**td_s, "color": "white", "fontWeight": "bold",
                                  "backgroundColor": SURFACE}),
        html.Td(f"{total:,}", style={**td_s, "color": "white", "fontWeight": "bold",
                                      "backgroundColor": SURFACE}),
        html.Td(f"{int(all_df[all_df['split']=='train']['label'].count()):,}",
                style={**td_s, "color": C_GREEN, "fontWeight": "bold",
                       "backgroundColor": SURFACE}),
        html.Td(f"{int(all_df[all_df['split']=='test']['label'].count()):,}",
                style={**td_s, "color": C_YELLOW, "fontWeight": "bold",
                       "backgroundColor": SURFACE}),
        html.Td(f"{pos_n:,} ({pos_n/total*100:.1f}%)",
                style={**td_s, "color": C_PURPLE, "fontWeight": "bold",
                       "backgroundColor": SURFACE}),
        html.Td(f"{neg_n:,} ({neg_n/total*100:.1f}%)",
                style={**td_s, "color": C_ORANGE, "fontWeight": "bold",
                       "backgroundColor": SURFACE}),
        html.Td("—", style={**td_s, "backgroundColor": SURFACE}),
    ]))

    summary_table = html.Table([
        html.Tr([html.Th(h, style=th_s) for h in
                 ["Phase", "Total", "Train", "Test", "SAE (pos)", "No SAE (neg)", "Balance"]])
    ] + rows, style={"width": "100%", "borderCollapse": "collapse"})

    def big_stat(label, value, color):
        return html.Div([
            html.P(label, style={"color": MUTED, "margin": "0", "fontSize": "10px"}),
            html.H2(str(value), style={"color": color, "margin": "4px 0 0 0", "fontSize": "26px"}),
        ], style={"backgroundColor": SURFACE, "borderRadius": "10px",
                  "padding": "12px 18px", "border": f"1px solid {BORDER}",
                  "textAlign": "center"})

    return html.Div([
        # Top stats
        html.Div([
            big_stat("Total Trials (available)", f"{total:,}",        "white"),
            big_stat("Total in TrialBench",       "31,306",            MUTED),
            big_stat("SAE (positive)",             f"{pos_n:,}",        C_PURPLE),
            big_stat("No SAE (negative)",          f"{neg_n:,}",        C_ORANGE),
            big_stat("Overall SAE Rate",           f"{pos_n/total*100:.1f}%", C_PINK),
            big_stat("Phases",                     "4",                 C_GREEN),
        ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap",
                  "marginBottom": "18px"}),

        # Summary table
        card([
            sec("Summary by Phase"),
            summary_table,
        ], extra={"marginBottom": "16px"}),

        # Charts
        html.Div([
            html.Div(dcc.Graph(figure=size_fig),  style={"flex": "1"}),
            html.Div(dcc.Graph(figure=pie),        style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

        html.Div([
            html.Div(dcc.Graph(figure=rate_fig),  style={"flex": "1"}),
            html.Div(dcc.Graph(figure=split_fig), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

        card([
            sec("Key Observations", C_GREEN),
            html.Ul([
                html.Li("Phase 2 is the largest dataset (8,116 trials) — most data to learn from",
                        style={"color": TEXT, "marginBottom": "5px", "fontSize": "12px"}),
                html.Li("Phase 3 has the highest SAE rate (84.7%) — model will be biased toward SAE",
                        style={"color": TEXT, "marginBottom": "5px", "fontSize": "12px"}),
                html.Li("Phase 1 & 4 are most balanced (~45/55%) — most reliable evaluation",
                        style={"color": TEXT, "marginBottom": "5px", "fontSize": "12px"}),
                html.Li(f"Total available: {total:,} / 31,306 from TrialBench (57.2% of full dataset)",
                        style={"color": TEXT, "fontSize": "12px"}),
            ]),
        ], extra={"marginTop": "4px"}),
    ])


def run():
    dfs = {}
    for ph in PHASES:
        try:
            dfs[ph] = load_phase_raw(ph)
            print(f"Phase {ph}: {len(dfs[ph])} rows, {len(dfs[ph].columns)} cols")
        except Exception as e:
            print(f"Phase {ph} not found: {e}")

    if not dfs:
        print("No data found. Check data/ folder.")
        return

    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    tab_s = {"backgroundColor": SURFACE, "color": MUTED, "border": f"1px solid {BORDER}",
             "borderRadius": "8px 8px 0 0", "padding": "10px 18px",
             "fontFamily": "monospace", "fontSize": "13px"}
    tab_a = {**tab_s, "backgroundColor": CARD, "color": C_PURPLE,
             "borderBottom": f"2px solid {C_PURPLE}"}

    app.layout = html.Div(
        style={"backgroundColor": BG, "minHeight": "100vh",
               "fontFamily": "monospace", "padding": "20px"},
        children=[
            # Header
            html.Div([
                html.H1("SAE Prediction — EDA",
                        style={"color": "white", "margin": "0", "fontSize": "20px",
                               "letterSpacing": "2px"}),
                html.P("Serious Adverse Event Forecasting · TrialBench Dataset",
                       style={"color": MUTED, "margin": "4px 0 0 0", "fontSize": "11px"}),
            ], style={"marginBottom": "16px", "borderBottom": f"1px solid {BORDER}",
                      "paddingBottom": "14px"}),

            # Phase selector
            card([
                html.Div([
                    html.B("Select Phase:", style={"color": C_GREEN, "marginRight": "20px",
                                                   "fontSize": "13px"}),
                    dcc.RadioItems(
                        id="phase",
                        options=[{"label": f"  Phase {p}", "value": p} for p in dfs.keys()]
                                + [{"label": "  All Phases", "value": "all"}],
                        value=list(dfs.keys())[0],
                        labelStyle={"display": "inline-block", "color": TEXT,
                                    "marginRight": "24px", "cursor": "pointer"},
                        inputStyle={"marginRight": "6px"},
                    ),
                ], style={"display": "flex", "alignItems": "center"}),
            ], extra={"marginBottom": "14px"}),

            # Tabs
            dcc.Tabs(id="tabs", value="overview", style={"marginBottom": "0"}, children=[
                dcc.Tab(label="Overview",        value="overview",     style=tab_s, selected_style=tab_a),
                dcc.Tab(label="Features",        value="features",     style=tab_s, selected_style=tab_a),
                dcc.Tab(label="Correlations",    value="correlations", style=tab_s, selected_style=tab_a),
                dcc.Tab(label="Data Dictionary", value="dictionary",   style=tab_s, selected_style=tab_a),
            ]),

            html.Div(id="tab-content",
                     style={"backgroundColor": CARD, "borderRadius": "0 12px 12px 12px",
                            "border": f"1px solid {BORDER}", "padding": "20px",
                            "minHeight": "600px"}),
        ]
    )

    @callback(Output("tab-content", "children"),
              Input("tabs", "value"), Input("phase", "value"))
    def render_tab(tab, phase):
        # ── All phases combined ────────────────────────────────────
        if phase == "all":
            if tab == "overview":
                return render_all_phases(tab, dfs)
            elif tab == "dictionary":
                # Dictionary is phase-independent — use phase 1 as reference
                df    = dfs.get("1", pd.DataFrame())
                phase = "All Phases"
            else:
                # Features & Correlations — combine all phases
                frames = []
                for ph, phdf in dfs.items():
                    d = phdf.copy()
                    frames.append(d)
                df    = pd.concat(frames, ignore_index=True)
                phase = "All Phases"
        else:
            df = dfs.get(phase, pd.DataFrame())

        if df.empty:
            return html.P("No data.", style={"color": C_ORANGE})

        NUM_COLS = get_num_cols(df)
        sae    = df[df["label"] == 1]
        no_sae = df[df["label"] == 0]
        total  = len(df)
        pos_n  = int(df["label"].sum())
        neg_n  = total - pos_n

        # ── OVERVIEW ──────────────────────────────────────────────
        if tab == "overview":
            vc  = df["label"].value_counts()
            pie = go.Figure(go.Pie(
                labels=["No SAE", "SAE"],
                values=[vc.get(0, 0), vc.get(1, 0)],
                marker_colors=[C_ORANGE, C_PURPLE], hole=0.45,
                textinfo="label+percent+value", textfont=dict(size=12),
            ))
            pie.update_layout(**{k: v for k, v in base_layout.items() if k != "margin"},
                               margin=dict(l=10, r=10, t=40, b=10),
                               title=f"Class Balance — Phase {phase}")

            sc = df.groupby(["split", "label"]).size().reset_index(name="count")
            sf = go.Figure()
            for lab, color, name in [(0, C_ORANGE, "No SAE"), (1, C_PURPLE, "SAE")]:
                sub = sc[sc["label"] == lab]
                sf.add_trace(go.Bar(x=sub["split"], y=sub["count"], name=name,
                                    marker_color=color, opacity=0.85))
            sf.update_layout(**base_layout, title="Train / Test Split", barmode="group",
                              xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID))

            ps = []
            for ph, phdf in dfs.items():
                n = len(phdf); p = phdf["label"].sum()
                ps.append({"phase": f"Phase {ph}", "r": p/n*100, "n": n})
            ps_df = pd.DataFrame(ps)
            pf = make_subplots(specs=[[{"secondary_y": True}]])
            pf.add_trace(go.Bar(x=ps_df["phase"], y=ps_df["r"], name="SAE Rate (%)",
                                marker_color=[PHASE_COLORS[p] for p in PHASES], opacity=0.85,
                                text=[f"{v:.1f}%" for v in ps_df["r"]], textposition="outside"),
                         secondary_y=False)
            pf.add_trace(go.Scatter(x=ps_df["phase"], y=ps_df["n"], name="# Samples",
                                    mode="lines+markers", line=dict(color=C_GREEN, width=2),
                                    marker=dict(size=8)), secondary_y=True)
            pf.update_layout(**base_layout, title="SAE Rate & Dataset Size across Phases",
                              yaxis=dict(title="SAE Rate (%)", gridcolor=GRID),
                              yaxis2=dict(title="# Samples", gridcolor=GRID))

            def stat(label, value, color):
                return html.Div([
                    html.P(label, style={"color": MUTED, "margin": "0", "fontSize": "10px"}),
                    html.H2(str(value), style={"color": color, "margin": "4px 0 0 0", "fontSize": "26px"}),
                ], style={"backgroundColor": SURFACE, "borderRadius": "10px",
                          "padding": "12px 18px", "border": f"1px solid {BORDER}",
                          "textAlign": "center"})

            train_n = int((df["split"] == "train").sum())
            test_n  = int((df["split"] == "test").sum())
            bal = "Almost balanced" if 0.4 < pos_n/total < 0.6 else "Imbalanced"
            bal_color = C_GREEN if "balanced" in bal else C_ORANGE

            return html.Div([
                html.Div([
                    stat("Total Trials",    total,                    "white"),
                    stat("SAE (positive)",  pos_n,                    C_PURPLE),
                    stat("No SAE",          neg_n,                    C_ORANGE),
                    stat("Train",           train_n,                  C_GREEN),
                    stat("Test",            test_n,                   C_YELLOW),
                    stat("SAE Rate",        f"{pos_n/total*100:.1f}%", C_PINK),
                ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap",
                          "marginBottom": "18px"}),

                html.Div([
                    html.Div(dcc.Graph(figure=pie), style={"flex": "1"}),
                    html.Div(dcc.Graph(figure=sf),  style={"flex": "1"}),
                ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

                dcc.Graph(figure=pf),

                card([
                    sec("Implications for Model Training", C_GREEN),
                    html.Ul([
                        html.Li([html.Span(f"Phase {phase} SAE rate: {pos_n/total*100:.1f}% — "),
                                 html.Span(bal, style={"color": bal_color, "fontWeight": "bold"})],
                                style={"color": TEXT, "marginBottom": "6px", "fontSize": "12px"}),
                        html.Li("Train/test split preserves class ratios (stratified sampling)",
                                style={"color": TEXT, "marginBottom": "6px", "fontSize": "12px"}),
                        html.Li("Phases 2 & 3 have >70% SAE — models may over-predict SAE in those phases",
                                style={"color": TEXT, "fontSize": "12px"}),
                    ]),
                ], extra={"marginTop": "14px"}),
            ])

        # ── FEATURES ─────────────────────────────────────────────
        elif tab == "features":
            TRUE_NUM = [c for c in NUM_COLS if pd.api.types.is_numeric_dtype(df[c])]
            imp = []
            for col in TRUE_NUM:
                m0 = no_sae[col].mean(skipna=True)
                m1 = sae[col].mean(skipna=True)
                diff = abs(m1 - m0)
                bv = max(abs(m0), abs(m1), 0.001)
                imp.append({"feature": col, "pct": diff/bv*100, "m0": m0, "m1": m1})
            imp_df = pd.DataFrame(imp).sort_values("pct", ascending=True)

            colors = [C_PURPLE if v > 20 else C_ORANGE if v > 8 else "#444466"
                      for v in imp_df["pct"]]
            imp_fig = go.Figure(go.Bar(
                x=imp_df["pct"], y=imp_df["feature"], orientation="h",
                marker_color=colors, opacity=0.9,
                text=[f"{v:.1f}%" for v in imp_df["pct"]], textposition="outside",
                customdata=list(zip(imp_df["m0"], imp_df["m1"])),
                hovertemplate="<b>%{y}</b><br>No SAE: %{customdata[0]:.3f}<br>SAE: %{customdata[1]:.3f}<br>Diff: %{x:.1f}%<extra></extra>",
            ))
            imp_fig.update_layout(
                **{k: v for k, v in base_layout.items() if k != "margin"},
                margin=dict(l=300, r=80, t=50, b=45), height=620,
                title=f"Univariate Feature Importance — Phase {phase}",
                xaxis=dict(title="Mean difference SAE vs No-SAE (%)", gridcolor=GRID),
                yaxis=dict(gridcolor=GRID),
            )

            miss = df[NUM_COLS].isnull().mean().sort_values(ascending=False) * 100
            miss = miss[miss > 0]
            if miss.empty:
                mf = go.Figure()
                mf.add_annotation(text="No missing values!", xref="paper", yref="paper",
                                   x=0.5, y=0.5, showarrow=False,
                                   font=dict(color=C_GREEN, size=16))
            else:
                mc = [C_PINK if v > 50 else C_ORANGE if v > 20 else C_GREEN for v in miss.values]
                mf = go.Figure(go.Bar(x=miss.values, y=miss.index, orientation="h",
                                      marker_color=mc, opacity=0.85,
                                      text=[f"{v:.0f}%" for v in miss.values],
                                      textposition="outside"))
            mf.update_layout(**{k: v for k, v in base_layout.items() if k != "margin"},
                              margin=dict(l=280, r=60, t=40, b=45), height=380,
                              title="Missing Values (%)",
                              xaxis=dict(title="%", gridcolor=GRID),
                              yaxis=dict(gridcolor=GRID))

            return html.Div([
                html.Div([
                    html.Div(dcc.Graph(figure=imp_fig), style={"flex": "2"}),
                    html.Div(dcc.Graph(figure=mf),      style={"flex": "1"}),
                ], style={"display": "flex", "gap": "12px", "marginBottom": "16px"}),

                card([
                    sec("Feature Distribution Explorer"),
                    html.Div([
                        html.Span("Feature: ", style={"color": MUTED, "marginRight": "10px",
                                                       "fontSize": "12px"}),
                        dcc.Dropdown(
                            id="feat-drop",
                            options=[{"label": c, "value": c} for c in TRUE_NUM],
                            value=TRUE_NUM[0] if TRUE_NUM else None,
                            style={"width": "380px", "backgroundColor": SURFACE,
                                   "color": "#111", "border": f"1px solid {BORDER}"},
                        ),
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
                    dcc.Graph(id="dist-graph",
                              figure=go.Figure().update_layout(
                                  paper_bgcolor=CARD, plot_bgcolor=CARD,
                                  font=dict(color=TEXT),
                                  margin=dict(l=50, r=20, t=45, b=50),
                              )),
                ]),

                card([
                    sec("Key Observations", C_GREEN),
                    html.Ul([
                        html.Li("Purple bars (>20% diff) = most informative features for model training",
                                style={"color": TEXT, "marginBottom": "5px", "fontSize": "12px"}),
                        html.Li("Grey bars (~0%) = noise — consider dropping them",
                                style={"color": TEXT, "marginBottom": "5px", "fontSize": "12px"}),
                        html.Li("Pink missing bars (>50%) = unreliable — low utility",
                                style={"color": TEXT, "fontSize": "12px"}),
                    ]),
                ], extra={"marginTop": "12px"}),
            ])

        # ── CORRELATIONS ─────────────────────────────────────────
        elif tab == "correlations":
            TRUE_NUM = [c for c in NUM_COLS if pd.api.types.is_numeric_dtype(df[c])]
            corr = df[TRUE_NUM].corr()
            hf = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale="RdBu", zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}", textfont=dict(size=8),
            ))
            hf.update_layout(**{k: v for k, v in base_layout.items() if k != "margin"},
                              margin=dict(l=220, r=20, t=50, b=220),
                              title=f"Feature Correlation Matrix — Phase {phase}", height=680)

            high = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    v = corr.iloc[i, j]
                    if abs(v) > 0.4:
                        high.append({"A": corr.columns[i], "B": corr.columns[j], "r": v})
            high_df = pd.DataFrame(high).sort_values("r", ascending=False, key=abs) if high else pd.DataFrame()

            if not high_df.empty:
                hrows = []
                for _, row in high_df.iterrows():
                    color = C_PURPLE if row["r"] > 0 else C_ORANGE
                    note  = "Redundant — consider dropping one" if abs(row["r"]) > 0.7 else "Moderate correlation"
                    hrows.append(html.Tr([
                        html.Td(row["A"], style={"padding": "6px 10px", "color": TEXT, "fontSize": "11px"}),
                        html.Td(row["B"], style={"padding": "6px 10px", "color": TEXT, "fontSize": "11px"}),
                        html.Td(f"{row['r']:.3f}", style={"padding": "6px 10px", "color": color,
                                                           "fontWeight": "bold", "fontSize": "12px"}),
                        html.Td(note, style={"padding": "6px 10px", "color": MUTED, "fontSize": "11px"}),
                    ], style={"borderBottom": f"1px solid {GRID}"}))

                htable = html.Table([html.Tr([
                    html.Th(h, style={"padding": "8px 10px", "color": C_PURPLE,
                                      "textAlign": "left", "backgroundColor": GRID,
                                      "fontSize": "11px"})
                    for h in ["Feature A", "Feature B", "r", "Note"]
                ])] + hrows, style={"width": "100%", "borderCollapse": "collapse"})
            else:
                htable = html.P("No high correlations found.", style={"color": C_GREEN})

            return html.Div([
                dcc.Graph(figure=hf),
                card([
                    sec("High Correlations (|r| > 0.4)", C_ORANGE),
                    html.P("Correlated features carry redundant info. Dropping one of each pair "
                           "can reduce noise without losing information.",
                           style={"color": MUTED, "fontSize": "12px", "marginBottom": "12px"}),
                    htable,
                ], extra={"marginTop": "14px"}),
            ])

        # ── DATA DICTIONARY ──────────────────────────────────────
        elif tab == "dictionary":
            all_cols = [c for c in df.columns if c not in ["Unnamed: 0", "label", "split"]]

            legend = html.Div([
                html.Div([
                    html.Span("Data Type: ", style={"color": MUTED, "marginRight": "8px", "fontSize": "11px"}),
                    *[badge(t, c) for t, c in TYPE_COLORS.items()],
                    html.Span("   Source: ", style={"color": MUTED, "margin": "0 8px", "fontSize": "11px"}),
                    html.Span("Derived", style={"color": SOURCE_COLORS["Derived"],
                                                 "fontWeight": "bold", "fontSize": "11px",
                                                 "marginRight": "6px"}),
                    html.Span("= engineered by TrialBench   ", style={"color": MUTED, "fontSize": "11px"}),
                    html.Span("Raw", style={"color": SOURCE_COLORS["Raw"],
                                             "fontWeight": "bold", "fontSize": "11px",
                                             "marginRight": "6px"}),
                    html.Span("= from ClinicalTrials.gov", style={"color": MUTED, "fontSize": "11px"}),
                ]),
            ], style={"backgroundColor": SURFACE, "padding": "10px 14px", "borderRadius": "8px",
                      "marginBottom": "16px"})

            sections = []
            for gname, gcols in GROUPS.items():
                gc = GROUP_COLORS.get(gname, TEXT)
                rows = []
                for col in gcols:
                    if col not in all_cols:
                        continue
                    dtype_str, source, desc = DESCRIPTIONS.get(col, ("?", "?", "—"))
                    null_pct = df[col].isnull().mean() * 100
                    nc = C_PINK if null_pct > 50 else C_ORANGE if null_pct > 20 else C_GREEN
                    used = "Yes" if col not in TEXT_COLS else "No"
                    used_color = C_GREEN if used == "Yes" else C_PINK
                    nunique = df[col].nunique()
                    ex = str(df[col].dropna().iloc[0])[:40] if not df[col].dropna().empty else "N/A"

                    rows.append(html.Tr([
                        html.Td(col, style={"padding": "7px 10px", "color": TEXT,
                                            "fontSize": "11px", "fontWeight": "bold",
                                            "whiteSpace": "nowrap"}),
                        html.Td(badge(dtype_str, TYPE_COLORS.get(dtype_str, "#333")),
                                style={"padding": "7px 10px"}),
                        html.Td(source, style={"padding": "7px 10px", "fontSize": "11px",
                                               "color": SOURCE_COLORS.get(source, TEXT)}),
                        html.Td(f"{null_pct:.0f}%", style={"padding": "7px 10px",
                                                             "color": nc, "fontWeight": "bold",
                                                             "fontSize": "11px"}),
                        html.Td(str(nunique), style={"padding": "7px 10px",
                                                      "color": MUTED, "fontSize": "11px"}),
                        html.Td(html.Span(used, style={"color": used_color, "fontWeight": "bold",
                                                        "fontSize": "11px"}),
                                style={"padding": "7px 10px"}),
                        html.Td(desc, style={"padding": "7px 10px", "color": MUTED,
                                             "fontSize": "11px"}),
                        html.Td(html.Code(ex[:35] + "..." if len(ex) > 35 else ex,
                                          style={"fontSize": "10px", "color": C_GREEN,
                                                 "backgroundColor": SURFACE,
                                                 "padding": "2px 6px", "borderRadius": "4px"}),
                                style={"padding": "7px 10px"}),
                    ], style={"borderBottom": f"1px solid {GRID}"}))

                th_style = {"padding": "8px 10px", "color": C_PURPLE, "textAlign": "left",
                            "backgroundColor": GRID, "fontSize": "11px"}
                table = html.Table([
                    html.Tr([html.Th(h, style=th_style)
                             for h in ["Column", "Type", "Source", "Nulls",
                                       "Unique", "Used now?", "Description", "Example"]])
                ] + rows, style={"width": "100%", "borderCollapse": "collapse"})

                sections.append(html.Div([
                    html.H4(gname, style={"color": gc, "margin": "20px 0 8px 0",
                                          "fontSize": "13px", "letterSpacing": "1px",
                                          "borderLeft": f"4px solid {gc}",
                                          "paddingLeft": "10px"}),
                    table,
                ]))

            return html.Div([legend] + sections)

        return html.P("Select a tab.", style={"color": MUTED})

    # ── Distribution callback ──────────────────────────────────────
    @callback(Output("dist-graph", "figure", allow_duplicate=True),
              Input("feat-drop", "value"), Input("phase", "value"),
              prevent_initial_call=True)
    def update_dist(feature, phase):
        # Handle all phases combined
        if phase == "all":
            frames = [dfs[ph].copy() for ph in dfs]
            df = pd.concat(frames, ignore_index=True)
            phase_label = "All Phases"
        else:
            df = dfs.get(phase, pd.DataFrame())
            phase_label = f"Phase {phase}"

        fig = go.Figure()
        if feature and not df.empty and feature in df.columns:
            for lab, color, name in [(0, C_ORANGE, "No SAE"), (1, C_PURPLE, "SAE")]:
                vals = df[df["label"] == lab][feature].dropna()
                fig.add_trace(go.Histogram(x=vals, name=name, marker_color=color,
                                           opacity=0.65, nbinsx=40))
            fig.update_layout(**base_layout, barmode="overlay",
                               title=f"Distribution: {feature} — {phase_label}",
                               xaxis=dict(gridcolor=GRID),
                               yaxis=dict(title="Count", gridcolor=GRID))
        else:
            fig.update_layout(**base_layout, title="Select a feature above")
        return fig

    print("\n  EDA Dashboard → http://127.0.0.1:8052\n")
    app.run(debug=False, use_reloader=False, port=8052)


if __name__ == "__main__":
    run()