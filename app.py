"""
Alzheimer's Digital Twin — project hub (Dash).

Run from repo root:  python app.py
Then open http://127.0.0.1:8050
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from flask import abort, send_from_directory
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "results" / "metrics" / "lstm_metrics.json"
SIMULATION_PATH = ROOT / "results" / "metrics" / "simulation_summary.json"
DEMO_SIM_PATH = ROOT / "results" / "metrics" / "demo_patients_simulation.json"
RESULTS_ROOT = (ROOT / "results").resolve()

# Curated gallery: relative to results/ — only these paths are linked from the UI.
GALLERY_VISUALS: list[tuple[str, str]] = [
    ("visualizations/results_dashboard.png", "Consolidated analysis dashboard"),
    ("visualizations/population_trajectories.png", "Population trajectories"),
    ("visualizations/demo_trajectories.png", "Demo patient trajectories"),
    ("visualizations/hippocampus_atrophy.png", "Hippocampus atrophy"),
    ("visualizations/monte_carlo_ci.png", "Monte Carlo uncertainty"),
    ("visualizations/all_patients_simulation.png", "All-patient simulation"),
    ("visualizations/subgroup_analysis.png", "Subgroup analysis"),
    ("visualizations/what_if_apoe4.png", "What-if: APOE4"),
    ("visualizations/what_if_intervention.png", "What-if: intervention"),
]

GALLERY_METRICS_PNG: list[tuple[str, str]] = [
    ("metrics/confusion_matrix.png", "Confusion matrix (classification)"),
    ("metrics/roc_curves.png", "ROC curves"),
    ("metrics/feature_importance.png", "Feature importance"),
    ("metrics/shap_importance.png", "SHAP summary"),
]


def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_simulation_summary() -> dict | None:
    if not SIMULATION_PATH.exists():
        return None
    try:
        return json.loads(SIMULATION_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_demo_simulation() -> dict | None:
    if not DEMO_SIM_PATH.exists():
        return None
    try:
        return json.loads(DEMO_SIM_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


DEMO_SIM = load_demo_simulation()


def hub_asset_url(rel: str) -> str:
    """URL path served by hub_results (relative to results/)."""
    return f"/hub-results/{rel.replace(chr(92), '/')}"


def metrics_figure(data: dict | None) -> go.Figure:
    if not data or not all(k in data for k in ("mmse_mae", "mmse_r2", "hippo_mae", "hippo_r2")):
        fig = go.Figure()
        fig.add_annotation(
            text="No regression metrics yet.<br><sub>Export MMSE / hippocampus metrics to lstm_metrics.json</sub>",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#7a7a73"),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=40, t=40, b=40),
            height=320,
        )
        return fig

    mmse_mae = float(data["mmse_mae"])
    mmse_r2 = float(data["mmse_r2"])
    hippo_mae = float(data["hippo_mae"])
    hippo_r2 = float(data["hippo_r2"])

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Mean absolute error", "R² (coefficient of determination)"),
        vertical_spacing=0.22,
    )
    fig.add_trace(
        go.Bar(
            x=["MMSE", "Hippocampus"],
            y=[mmse_mae, hippo_mae],
            marker=dict(color="#2f4f3f", line=dict(width=0)),
            text=[f"{mmse_mae:.2f}", f"{hippo_mae:.1f}"],
            textposition="outside",
            textfont=dict(size=11, color="#3c3c38"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=["MMSE", "Hippocampus"],
            y=[mmse_r2, hippo_r2],
            marker=dict(color="#8fa396", line=dict(width=0)),
            text=[f"{mmse_r2:.3f}", f"{hippo_r2:.4f}"],
            textposition="outside",
            textfont=dict(size=11, color="#3c3c38"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    axis_style = dict(
        showgrid=True,
        gridcolor="#e6e6e0",
        zeroline=False,
        linecolor="#e6e6e0",
        tickfont=dict(size=11),
    )
    fig.update_xaxes(axis_style, row=1, col=1)
    fig.update_xaxes(axis_style, row=2, col=1)
    fig.update_yaxes(axis_style, row=1, col=1)
    fig.update_yaxes(axis_style, row=2, col=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fafaf7",
        font=dict(family="DM Sans, sans-serif", color="#3c3c38", size=12),
        margin=dict(l=44, r=28, t=72, b=36),
        height=360,
    )
    return fig


def classification_figure(data: dict | None) -> go.Figure | None:
    if not data:
        return None
    if not all(k in data for k in ("clf3_cv_accuracy", "mci_conv_auc_cv")):
        return None
    acc = float(data["clf3_cv_accuracy"])
    auc = float(data["mci_conv_auc_cv"])
    acc_std = float(data.get("clf3_cv_accuracy_std", 0) or 0)
    auc_std = float(data.get("mci_conv_auc_cv_std", 0) or 0)

    fig = go.Figure(
        go.Bar(
            x=["3-class accuracy (CV)", "MCI → Dementia AUC (CV)"],
            y=[acc, auc],
            error_y=dict(type="data", array=[acc_std, auc_std], color="#7a7a73", thickness=1.5, width=6),
            marker=dict(color=["#2f4f3f", "#8fa396"], line=dict(width=0)),
            text=[f"{acc:.3f} ± {acc_std:.3f}", f"{auc:.3f} ± {auc_std:.3f}"],
            textposition="outside",
            textfont=dict(size=11, color="#3c3c38"),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fafaf7",
        font=dict(family="DM Sans, sans-serif", color="#3c3c38", size=12),
        xaxis=dict(showgrid=False, linecolor="#e6e6e0", tickfont=dict(size=11)),
        yaxis=dict(gridcolor="#e6e6e0", zeroline=False, range=[0, 1.05], title=dict(text="Score", font=dict(size=11, color="#7a7a73"))),
        margin=dict(l=48, r=28, t=28, b=72),
        height=300,
    )
    return fig


def regression_metric_cards(data: dict | None) -> html.Div:
    if not data:
        return html.Div(
            className="metrics-grid",
            children=[
                html.Div(
                    className="metric-card",
                    children=[
                        html.Span("Regression", className="metric-card__label"),
                        html.Div("—", className="metric-card__value"),
                        html.P(
                            "Export lstm_metrics.json after LSTM training.",
                            className="metric-card__hint",
                        ),
                    ],
                )
            ],
        )

    specs = [
        ("mmse_mae", "MMSE — mean absolute error", "{:.4f}", "Lower is better"),
        ("mmse_r2", "MMSE — R²", "{:.4f}", "Variance explained"),
        ("hippo_mae", "Hippocampus — MAE", "{:.2f}", "Lower is better"),
        ("hippo_r2", "Hippocampus — R²", "{:.4f}", "Variance explained"),
    ]
    cards = []
    for key, label, fmt, hint in specs:
        if key not in data:
            continue
        val = float(data[key])
        cards.append(
            html.Div(
                className="metric-card",
                children=[
                    html.Span(label, className="metric-card__label"),
                    html.Div(fmt.format(val), className="metric-card__value"),
                    html.P(hint, className="metric-card__hint"),
                ],
            )
        )
    if not cards:
        return regression_metric_cards(None)
    return html.Div(className="metrics-grid", children=cards)


def classification_metric_cards(data: dict | None) -> html.Div | None:
    if not data or "clf3_cv_accuracy" not in data:
        return None
    acc = float(data["clf3_cv_accuracy"])
    acc_std = float(data.get("clf3_cv_accuracy_std", 0) or 0)
    items: list[tuple[str, str, str]] = [
        ("3-class accuracy (CV)", f"{acc:.4f} ± {acc_std:.4f}", "Stratified CV, XGBoost"),
    ]
    if "mci_conv_auc_cv" in data:
        auc = float(data["mci_conv_auc_cv"])
        auc_std = float(data.get("mci_conv_auc_cv_std", 0) or 0)
        items.append(("MCI → Dementia AUC (CV)", f"{auc:.4f} ± {auc_std:.4f}", "Conversion risk ranking"))
    return html.Div(
        className="metrics-grid",
        children=[
            html.Div(
                className="metric-card",
                children=[
                    html.Span(label, className="metric-card__label"),
                    html.Div(val, className="metric-card__value"),
                    html.P(hint, className="metric-card__hint"),
                ],
            )
            for label, val, hint in items
        ],
    )


def simulation_metric_cards(sim: dict | None) -> html.Div | None:
    if not sim:
        return None
    mapping: list[tuple[str, str, str, str]] = [
        ("base_patient", "Base patient RID", "{}", "Anchor record for twin runs"),
        ("n_future_visits", "Future visits simulated", "{}", "Horizon length"),
        ("mc_samples", "Monte Carlo samples", "{}", "Draws for uncertainty"),
        ("mc_ci90_width_last", "90% CI width (last visit)", "{:.2f}", "MMSE points, approximate"),
        ("apoe4_mmse_spread", "APOE4 MMSE spread", "{:.2f}", "Exploratory what-if delta"),
        ("intervention_40pct_benefit", "40% intervention benefit", "{:.2f}", "MMSE change vs baseline path"),
    ]
    cards = []
    for key, label, fmt, hint in mapping:
        if key not in sim:
            continue
        raw = sim[key]
        if isinstance(raw, float):
            val = fmt.format(raw)
        else:
            val = str(raw)
        cards.append(
            html.Div(
                className="metric-card",
                children=[
                    html.Span(label, className="metric-card__label"),
                    html.Div(val, className="metric-card__value"),
                    html.P(hint, className="metric-card__hint"),
                ],
            )
        )
    if not cards:
        return None
    return html.Div(className="metrics-grid", children=cards)


def figure_gallery_rows(groups: list[tuple[str, list[tuple[str, str]]]]) -> list:
    """Build figure elements; skip missing files."""
    rows: list = []
    for group_title, entries in groups:
        present = [(rel, cap) for rel, cap in entries if (RESULTS_ROOT / rel).is_file()]
        if not present:
            continue
        rows.append(html.H3(group_title, className="metrics-block__title"))
        rows.append(
            html.Div(
                className="gallery-grid",
                children=[
                    html.Figure(
                        className="figure-card",
                        children=[
                            html.Img(src=hub_asset_url(rel), alt=cap),
                            html.Figcaption(cap),
                        ],
                    )
                    for rel, cap in present
                ],
            )
        )
    return rows


def _demo_patient(demo: dict, rid: str) -> dict | None:
    if not demo or "patients" not in demo:
        return None
    return demo["patients"].get(str(rid))


def _plotly_layout_base(title: str, yaxis_title: str, height: int = 400) -> dict:
    return dict(
        title=dict(text=title, font=dict(family="Fraunces, Georgia, serif", size=16, color="#121211"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fafaf7",
        font=dict(family="DM Sans, sans-serif", size=12, color="#3c3c38"),
        margin=dict(l=52, r=28, t=72, b=52),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(
            title=dict(text="Visit number", font=dict(size=11, color="#7a7a73")),
            gridcolor="#e6e6e0",
            zeroline=False,
            linecolor="#e6e6e0",
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=11, color="#7a7a73")),
            gridcolor="#e6e6e0",
            zeroline=False,
            linecolor="#e6e6e0",
        ),
        hovermode="x unified",
    )


def demo_mmse_figure(rid: str, demo: dict | None) -> go.Figure:
    fig = go.Figure()
    if not demo:
        fig.update_layout(**_plotly_layout_base("MMSE trajectory", "MMSE (0–30)", 360))
        fig.add_annotation(text="Add results/metrics/demo_patients_simulation.json", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color="#7a7a73", size=14))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig
    p = _demo_patient(demo, rid)
    if not p:
        fig.update_layout(**_plotly_layout_base("MMSE trajectory", "MMSE (0–30)", 360))
        fig.add_annotation(text="Patient not found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color="#7a7a73", size=14))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    v_obs, m_obs = p["visits_obs"], p["mmse_obs"]
    v_roll = [v_obs[-1]] + p["pred_visits"]
    m_roll = [m_obs[-1]] + p["mmse_pred"]

    fig.add_trace(
        go.Scatter(
            x=v_obs,
            y=m_obs,
            mode="lines+markers",
            name="Observed (ADNI)",
            line=dict(color="#2f4f3f", width=2.8),
            marker=dict(size=9, color="#2f4f3f"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=v_roll,
            y=m_roll,
            mode="lines+markers",
            name="Predicted (twin rollout)",
            line=dict(color="#8fa396", width=2.2, dash="dash"),
            marker=dict(size=8, symbol="square", color="#8fa396"),
        )
    )
    int_m = p.get("mmse_pred_intervention_40pct")
    if int_m:
        m_int = [m_obs[-1]] + list(int_m)
        fig.add_trace(
            go.Scatter(
                x=v_roll,
                y=m_int,
                mode="lines+markers",
                name="40% decline attenuation",
                line=dict(color="#5a8f6f", width=2, dash="dot"),
                marker=dict(size=7, symbol="diamond", color="#5a8f6f"),
            )
        )
    fig.add_hline(y=24, line_dash="dot", line_color="#c4c4bd", opacity=0.9)
    fig.add_annotation(
        xref="paper",
        yref="y",
        x=0.01,
        y=24.6,
        text="MCI threshold (24)",
        showarrow=False,
        font=dict(size=10, color="#7a7a73"),
        xanchor="left",
    )
    fig.update_layout(**_plotly_layout_base(f"MMSE — RID {p['rid']}", "MMSE score"))
    fig.update_yaxes(range=[0, 32])
    return fig


def demo_hippo_figure(rid: str, demo: dict | None) -> go.Figure:
    fig = go.Figure()
    if not demo:
        fig.update_layout(**_plotly_layout_base("Hippocampus trajectory", "Volume (mm³)", 360))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig
    p = _demo_patient(demo, rid)
    if not p:
        fig.update_layout(**_plotly_layout_base("Hippocampus trajectory", "Volume (mm³)", 360))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    v_obs, h_obs = p["visits_obs"], p["hippo_obs"]
    v_roll = [v_obs[-1]] + p["pred_visits"]
    h_roll = [h_obs[-1]] + p["hippo_pred"]

    fig.add_trace(
        go.Scatter(
            x=v_obs,
            y=h_obs,
            mode="lines+markers",
            name="Observed (ADNI)",
            line=dict(color="#2f4f3f", width=2.8),
            marker=dict(size=9, color="#2f4f3f"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=v_roll,
            y=h_roll,
            mode="lines+markers",
            name="Predicted (twin rollout)",
            line=dict(color="#8fa396", width=2.2, dash="dash"),
            marker=dict(size=8, symbol="square", color="#8fa396"),
        )
    )
    fig.update_layout(**_plotly_layout_base(f"Hippocampus — RID {p['rid']}", "Hippocampus volume (mm³)"))
    return fig


def demo_meta_children(rid: str, demo: dict | None) -> list:
    if not demo:
        return [html.P("No demo bundle loaded.", className="demo-meta__empty")]
    p = _demo_patient(demo, rid)
    if not p:
        return [html.P("Patient not found.", className="demo-meta__empty")]
    method = demo.get("prediction_method", "unknown")
    method_label = {
        "lstm_rollout": "LSTM autoregressive rollout (matches notebook 05)",
        "linear_extrapolation": "Linear extrapolation (checkpoint unavailable when JSON was built)",
    }.get(method, method.replace("_", " "))
    items = [
        ("RID", str(p["rid"])),
        ("APOEε4 alleles", str(p["apoe4"])),
        ("Last observed diagnosis", str(p.get("dx_last", "—"))),
        ("Observed visits", str(p.get("n_obs_visits", "—"))),
        ("Future steps in JSON", str(len(p.get("pred_visits", [])))),
        ("Forecast source", method_label),
    ]
    dl_children: list = []
    for k, v in items:
        dl_children.append(html.Dt(k, className="demo-meta__dt"))
        dl_children.append(html.Dd(v, className="demo-meta__dd"))
    return [html.Dl(className="demo-meta__dl", children=dl_children)]


def build_demo_simulation_section() -> html.Section:
    if DEMO_SIM and DEMO_SIM.get("patients"):
        rids = [str(r) for r in DEMO_SIM.get("demo_rids", [])]
        first = rids[0] if rids else None
        patients = DEMO_SIM["patients"]
        options = []
        for r in rids:
            pr = patients.get(r, {})
            dx = str(pr.get("dx_last", "—"))
            ap = pr.get("apoe4", "—")
            options.append({"label": f"RID {r}  ·  {dx}  ·  APOEε4 = {ap}", "value": r})
        return html.Section(
            id="demo-simulation",
            className="section",
            children=[
                html.Div(
                    className="section__inner",
                    children=[
                        html.P("Interactive", className="section__label"),
                        html.H2("Demo patient simulation.", className="section__title"),
                        html.P(
                            "Choose one of five curated ADNI subjects. Observed points come from the cohort table; "
                            "curves ahead of the last visit are twin rollouts exported with the same logic as "
                            "`05_simulation.ipynb` (regenerate via scripts/export_demo_simulation_json.py).",
                            className="section__lead",
                        ),
                        html.Div(
                            className="demo-toolbar",
                            children=[
                                html.Label("Patient", className="demo-toolbar__label", htmlFor="demo-patient-select"),
                                dcc.Dropdown(
                                    id="demo-patient-select",
                                    options=options,
                                    value=first,
                                    clearable=False,
                                    className="demo-dropdown",
                                    style={"maxWidth": "28rem", "fontSize": "0.9375rem"},
                                ),
                            ],
                        ),
                        html.Div(id="demo-patient-meta", className="demo-meta", children=demo_meta_children(first or "", DEMO_SIM)),
                        html.Div(
                            className="demo-charts",
                            children=[
                                html.Div(
                                    className="chart-wrap chart-wrap--demo",
                                    children=[
                                        dcc.Graph(
                                            id="demo-graph-mmse",
                                            figure=demo_mmse_figure(first or "", DEMO_SIM),
                                            config=dict(displayModeBar=False),
                                            style={"height": "420px"},
                                        )
                                    ],
                                ),
                                html.Div(
                                    className="chart-wrap chart-wrap--demo",
                                    children=[
                                        dcc.Graph(
                                            id="demo-graph-hippo",
                                            figure=demo_hippo_figure(first or "", DEMO_SIM),
                                            config=dict(displayModeBar=False),
                                            style={"height": "420px"},
                                        )
                                    ],
                                ),
                            ],
                        ),
                    ],
                )
            ],
        )

    return html.Section(
        id="demo-simulation",
        className="section",
        children=[
            html.Div(
                className="section__inner",
                children=[
                    html.P("Interactive", className="section__label"),
                    html.H2("Demo patient simulation.", className="section__title"),
                    html.P(
                        "Run scripts/export_demo_simulation_json.py (with data/raw/ADNIMERGE.csv and optionally "
                        "models/checkpoints/lstm_best.pt) to write results/metrics/demo_patients_simulation.json, then reload this page.",
                        className="section__lead",
                    ),
                ],
            )
        ],
    )


def build_layout() -> html.Div:
    metrics = load_metrics()
    simulation = load_simulation_summary()
    clf_fig = classification_figure(metrics)

    gallery_children = figure_gallery_rows(
        [
            ("Trajectory & policy figures", GALLERY_VISUALS),
            ("Model diagnostics", GALLERY_METRICS_PNG),
        ]
    )

    metrics_children: list = [
        html.P("Evidence", className="section__label"),
        html.H2("Exported metrics & charts.", className="section__title"),
        html.P(
            "Regression and classification values are read from results/metrics/lstm_metrics.json. "
            "Simulation summaries use simulation_summary.json when present.",
            className="section__lead",
        ),
        html.Div(
            className="metrics-block",
            children=[
                html.H3("Regression (LSTM)", className="metrics-block__title"),
                html.P("MMSE and hippocampal volume heads.", className="metrics-block__lead"),
                regression_metric_cards(metrics),
                html.Div(
                    className="chart-wrap",
                    children=[
                        dcc.Graph(
                            id="metrics-chart",
                            figure=metrics_figure(metrics),
                            config=dict(displayModeBar=False),
                            style={"height": "380px"},
                        )
                    ],
                ),
            ],
        ),
    ]

    clf_cards = classification_metric_cards(metrics)
    if clf_cards:
        clf_block: list = [
            html.H3("Classification (XGBoost)", className="metrics-block__title"),
            html.P(
                "Three-class diagnosis and MCI-to-dementia conversion signals from the patient-level pipeline.",
                className="metrics-block__lead",
            ),
            clf_cards,
        ]
        if clf_fig:
            clf_block.append(
                html.Div(
                    className="chart-wrap",
                    children=[
                        dcc.Graph(
                            id="clf-chart",
                            figure=clf_fig,
                            config=dict(displayModeBar=False),
                            style={"height": "320px"},
                        )
                    ],
                )
            )
        metrics_children.append(html.Div(className="metrics-block", children=clf_block))

    sim_cards = simulation_metric_cards(simulation)
    if sim_cards:
        metrics_children.append(
            html.Div(
                className="metrics-block",
                children=[
                    html.H3("Simulation snapshot", className="metrics-block__title"),
                    html.P(
                        "Monte Carlo-style summaries exported with the simulation notebook.",
                        className="metrics-block__lead",
                    ),
                    sim_cards,
                ],
            )
        )

    return html.Div(
        className="shell",
        children=[
            html.Nav(
                className="site-nav",
                children=[
                    html.Div(
                        className="site-nav__inner",
                        children=[
                            html.A("Digital Twin", href="#top", className="site-nav__brand"),
                            html.Ul(
                                className="site-nav__links",
                                children=[
                                    html.Li(html.A("About", href="#about")),
                                    html.Li(html.A("Pipeline", href="#pipeline")),
                                    html.Li(html.A("Metrics", href="#metrics")),
                                    html.Li(html.A("Demo twin", href="#demo-simulation")),
                                    html.Li(html.A("Gallery", href="#gallery")),
                                    html.Li(html.A("Notebooks", href="#notebooks")),
                                ],
                            ),
                        ],
                    )
                ],
            ),
            html.Main(
                children=[
                    html.Section(
                        id="top",
                        className="hero",
                        children=[
                            html.Div(
                                className="hero__inner",
                                children=[
                                    html.P("Research prototype", className="hero__eyebrow"),
                                    html.H1(
                                        "Alzheimer's digital twin",
                                        className="hero__title",
                                    ),
                                    html.P(
                                        "Longitudinal ADNI signals, sequence models, classification, "
                                        "visual analytics, and simulation—presented with room to breathe.",
                                        className="hero__subtitle",
                                    ),
                                    html.Div(className="hero__rule"),
                                ],
                            )
                        ],
                    ),
                    html.Section(
                        id="about",
                        className="section section--tight-top",
                        children=[
                            html.Div(
                                className="section__inner",
                                children=[
                                    html.P("Context", className="section__label"),
                                    html.H2("Clinical trajectories, modelled with restraint.", className="section__title"),
                                    html.Div(
                                        className="grid-2",
                                        children=[
                                            html.Div(
                                                className="prose",
                                                children=[
                                                    html.P(
                                                        "This repository explores a digital twin framing for Alzheimer's "
                                                        "disease progression: structured visits, cognitive scores, imaging-derived "
                                                        "volume, and genetics—aligned in time where possible."
                                                    ),
                                                    html.P(
                                                        "Recent work adds patient-level classification (three-class diagnosis and "
                                                        "MCI-to-dementia conversion), a visualization dashboard, and Monte Carlo–style "
                                                        "simulation exports alongside the original LSTM regression path."
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="prose",
                                                children=[
                                                    html.P(
                                                        "Data centre on ADNIMERGE-style tables. The LSTM stack uses aligned "
                                                        "feature sequences; XGBoost models consume engineered patient snapshots; "
                                                        "figures and JSON summaries land under results/ for this hub to surface."
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                    html.Section(
                        id="pipeline",
                        className="section",
                        children=[
                            html.Div(
                                className="section__inner",
                                children=[
                                    html.P("Flow", className="section__label"),
                                    html.H2("From cohort to twin scenarios.", className="section__title"),
                                    html.P(
                                        "Five beats that mirror the notebooks—kept visually light.",
                                        className="section__lead",
                                    ),
                                    html.Div(
                                        className="pipeline",
                                        children=[
                                            html.Div(
                                                className="pipeline__step",
                                                children=[
                                                    html.Div("01", className="pipeline__num"),
                                                    html.H3("Curate"),
                                                    html.P(
                                                        "Load and filter longitudinal rows; harmonise diagnosis codes; "
                                                        "document missingness."
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="pipeline__step",
                                                children=[
                                                    html.Div("02", className="pipeline__num"),
                                                    html.H3("Encode"),
                                                    html.P(
                                                        "Sequence windows for the LSTM with scaled features matching the "
                                                        "trained checkpoint."
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="pipeline__step",
                                                children=[
                                                    html.Div("03", className="pipeline__num"),
                                                    html.H3("Classify"),
                                                    html.P(
                                                        "XGBoost pipelines for three-class diagnosis and MCI→Dementia conversion, "
                                                        "with CV metrics exported to JSON."
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="pipeline__step",
                                                children=[
                                                    html.Div("04", className="pipeline__num"),
                                                    html.H3("Visualize"),
                                                    html.P(
                                                        "Trajectory plots, subgroup views, SHAP, ROC, and consolidated dashboards "
                                                        "written to results/visualizations and results/metrics."
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="pipeline__step",
                                                children=[
                                                    html.Div("05", className="pipeline__num"),
                                                    html.H3("Simulate"),
                                                    html.P(
                                                        "What-if and Monte Carlo summaries—patient anchors, intervention deltas, "
                                                        "and uncertainty bands—exported for review."
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                    html.Section(id="metrics", className="section", children=[html.Div(className="section__inner", children=metrics_children)]),
                    build_demo_simulation_section(),
                    html.Section(
                        id="gallery",
                        className="section",
                        children=[
                            html.Div(
                                className="section__inner",
                                children=[
                                    html.P("Figures", className="section__label"),
                                    html.H2("Static outputs from the analysis stack.", className="section__title"),
                                    html.P(
                                        "PNG exports produced by the visualization and classification notebooks. "
                                        "Missing files are omitted automatically.",
                                        className="section__lead",
                                    ),
                                    *(
                                        gallery_children
                                        if gallery_children
                                        else [
                                            html.P(
                                                "No figure files found under results/. Run the visualization notebook to populate this gallery.",
                                                className="section__lead",
                                            )
                                        ]
                                    ),
                                ],
                            )
                        ],
                    ),
                    html.Section(
                        id="notebooks",
                        className="section",
                        children=[
                            html.Div(
                                className="section__inner",
                                children=[
                                    html.P("Workspace", className="section__label"),
                                    html.H2("Notebooks & artefacts.", className="section__title"),
                                    html.P(
                                        "Open these in JupyterLab from the repository root after installing requirements.",
                                        className="section__lead",
                                    ),
                                    html.Div(
                                        className="notebook-list",
                                        children=[
                                            html.Div(
                                                className="notebook-row",
                                                children=[
                                                    html.Span("data_exploration.ipynb", className="notebook-row__name"),
                                                    html.Span("Cohort profile & features", className="notebook-row__desc"),
                                                ],
                                            ),
                                            html.Div(
                                                className="notebook-row",
                                                children=[
                                                    html.Span("02_lstm_model.ipynb", className="notebook-row__name"),
                                                    html.Span("LSTM training & regression metrics", className="notebook-row__desc"),
                                                ],
                                            ),
                                            html.Div(
                                                className="notebook-row",
                                                children=[
                                                    html.Span("03_classification.ipynb", className="notebook-row__name"),
                                                    html.Span("XGBoost diagnosis & conversion", className="notebook-row__desc"),
                                                ],
                                            ),
                                            html.Div(
                                                className="notebook-row",
                                                children=[
                                                    html.Span("04_visualization.ipynb", className="notebook-row__name"),
                                                    html.Span("Dashboards, trajectories, SHAP / ROC", className="notebook-row__desc"),
                                                ],
                                            ),
                                            html.Div(
                                                className="notebook-row",
                                                children=[
                                                    html.Span("05_simulation.ipynb", className="notebook-row__name"),
                                                    html.Span("What-if & Monte Carlo summaries", className="notebook-row__desc"),
                                                ],
                                            ),
                                            html.Div(
                                                className="notebook-row",
                                                children=[
                                                    html.Span("results/metrics/demo_patients_simulation.json", className="notebook-row__name"),
                                                    html.Span("Observed + rollout series for the demo twin", className="notebook-row__desc"),
                                                ],
                                            ),
                                            html.Div(
                                                className="notebook-row",
                                                children=[
                                                    html.Span("scripts/export_demo_simulation_json.py", className="notebook-row__name"),
                                                    html.Span("Regenerate demo JSON (LSTM or linear fallback)", className="notebook-row__desc"),
                                                ],
                                            ),
                                            html.Div(
                                                className="notebook-row",
                                                children=[
                                                    html.Span("data/raw/ADNIMERGE.csv", className="notebook-row__name"),
                                                    html.Span("Source table (local only)", className="notebook-row__desc"),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
            html.Footer(
                className="site-footer",
                children=[
                    html.P(
                        "Alzheimer's digital twin — exploratory research software. Not a medical device."
                    )
                ],
            ),
        ],
    )


app = Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?"
        "family=DM+Sans:ital,wght@0,400;0,500;1,400&"
        "family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&"
        "display=swap",
    ],
    title="Alzheimer's Digital Twin",
)
app.layout = build_layout()
server = app.server


if DEMO_SIM and DEMO_SIM.get("patients"):

    @app.callback(
        Output("demo-graph-mmse", "figure"),
        Output("demo-graph-hippo", "figure"),
        Output("demo-patient-meta", "children"),
        Input("demo-patient-select", "value"),
    )
    def _update_demo_patient(rid: str | None):
        r = str(rid) if rid else str(DEMO_SIM["demo_rids"][0])
        return (
            demo_mmse_figure(r, DEMO_SIM),
            demo_hippo_figure(r, DEMO_SIM),
            demo_meta_children(r, DEMO_SIM),
        )


@server.route("/hub-results/<path:subpath>")
def hub_results(subpath: str):
    """Serve PNGs from results/ for the gallery (path traversal safe)."""
    try:
        rel = Path(subpath)
        if rel.is_absolute() or ".." in rel.parts:
            abort(404)
        candidate = (RESULTS_ROOT / rel).resolve()
        candidate.relative_to(RESULTS_ROOT)
    except (ValueError, OSError):
        abort(404)
    if not candidate.is_file() or candidate.suffix.lower() != ".png":
        abort(404)
    return send_from_directory(str(candidate.parent), candidate.name)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
