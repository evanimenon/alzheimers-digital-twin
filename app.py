"""
Alzheimer's Digital Twin — project hub (Dash).

Run from repo root:  python app.py
Then open http://127.0.0.1:8050
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "results" / "metrics" / "lstm_metrics.json"


def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def metrics_figure(data: dict | None) -> go.Figure:
    if not data:
        fig = go.Figure()
        fig.add_annotation(
            text="No metrics file yet.<br><sub>Train the model and write results/metrics/lstm_metrics.json</sub>",
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

    mmse_mae = float(data.get("mmse_mae", 0))
    mmse_r2 = float(data.get("mmse_r2", 0))
    hippo_mae = float(data.get("hippo_mae", 0))
    hippo_r2 = float(data.get("hippo_r2", 0))

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


def metric_cards(data: dict | None) -> html.Div:
    if not data:
        return html.Div(
            className="metrics-grid",
            children=[
                html.Div(
                    className="metric-card",
                    children=[
                        html.Span("Metrics", className="metric-card__label"),
                        html.Div("—", className="metric-card__value"),
                        html.P(
                            "Run training and export lstm_metrics.json to populate this panel.",
                            className="metric-card__hint",
                        ),
                    ],
                )
            ],
        )

    items = [
        ("MMSE — mean absolute error", f"{float(data['mmse_mae']):.4f}", "Lower is better"),
        ("MMSE — R²", f"{float(data['mmse_r2']):.4f}", "Variance explained"),
        ("Hippocampus — MAE", f"{float(data['hippo_mae']):.2f}", "Lower is better"),
        ("Hippocampus — R²", f"{float(data['hippo_r2']):.4f}", "Variance explained"),
    ]
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


def build_layout() -> html.Div:
    metrics = load_metrics()

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
                                        "A quiet workspace for longitudinal ADNI signals, "
                                        "sequence modelling, and calibrated outcomes you can reason about—not just numbers on a slide.",
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
                                                        "The goal is not spectacle. It is reproducible pipelines, honest "
                                                        "metrics, and enough whitespace in the interface that you can think."
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="prose",
                                                children=[
                                                    html.P(
                                                        "Data centre on ADNIMERGE-style tables (e.g. MMSE, diagnosis, "
                                                        "hippocampal volume, APOE4, education, visit index). The LSTM path "
                                                        "uses five aligned features per sequence, with exported JSON metrics "
                                                        "for MMSE and hippocampus targets."
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
                                    html.H2("From raw cohort to twin signals.", className="section__title"),
                                    html.P(
                                        "Three beats—ingest, model, interpret—kept visually light so the emphasis stays on meaning.",
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
                                                        "Load and filter longitudinal rows; harmonise codes; "
                                                        "carry forward defensible missingness rules."
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="pipeline__step",
                                                children=[
                                                    html.Div("02", className="pipeline__num"),
                                                    html.H3("Encode"),
                                                    html.P(
                                                        "Sequence windows (e.g. up to eight visits) with scaled features "
                                                        "matching the trained checkpoint shape."
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="pipeline__step",
                                                children=[
                                                    html.Div("03", className="pipeline__num"),
                                                    html.H3("Project"),
                                                    html.P(
                                                        "LSTM heads for MMSE and hippocampus; metrics and history written "
                                                        "under results/ for this hub to read."
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
                        id="metrics",
                        className="section",
                        children=[
                            html.Div(
                                className="section__inner",
                                children=[
                                    html.P("Evidence", className="section__label"),
                                    html.H2("Latest model metrics.", className="section__title"),
                                    html.P(
                                        "Values are read from results/metrics/lstm_metrics.json when present.",
                                        className="section__lead",
                                    ),
                                    metric_cards(metrics),
                                    html.Div(
                                        className="chart-wrap",
                                        children=[
                                            dcc.Graph(
                                                id="metrics-chart",
                                                figure=metrics_figure(metrics),
                                                config=dict(displayModeBar=False),
                                                style={"height": "320px"},
                                            )
                                        ],
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
                                                    html.Span("LSTM, plots, metrics export", className="notebook-row__desc"),
                                                ],
                                            ),
                                            html.Div(
                                                className="notebook-row",
                                                children=[
                                                    html.Span("data/raw/ADNIMERGE.csv", className="notebook-row__name"),
                                                    html.Span("Source table (not shipped to UI)", className="notebook-row__desc"),
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


if __name__ == "__main__":
    app.run(debug=True, port=8050)
