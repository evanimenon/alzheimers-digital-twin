"""
Alzheimer's Digital Twin — project hub (Dash).
Updated: team names · interactive gallery lightbox · layman-friendly explanations.

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
DEMO_DATA_PATH = ROOT / "results" / "metrics" / "demo_data.json"
RESULTS_ROOT = (ROOT / "results").resolve()

# ── Team ──────────────────────────────────────────────────────────────────────
TEAM_MEMBERS = [
    "Evani Menon",
    "Manojna Reddy Kamaram",
    "Diksha Kaushik",
    "Ishaan Arora",
    "Palak",
    "Isobel Kuriyan",
]

# ── Gallery manifests ─────────────────────────────────────────────────────────
GALLERY_VISUALS: list[tuple[str, str]] = [
    ("visualizations/brain_atrophy_3d.png",       "3D brain atrophy — baseline vs 24 months"),
    ("visualizations/results_dashboard.png",      "Consolidated analysis dashboard"),
    ("visualizations/population_trajectories.png","Population trajectories"),
    ("visualizations/demo_trajectories.png",      "Demo patient trajectories"),
    ("visualizations/hippocampus_atrophy.png",    "Hippocampus atrophy"),
    ("visualizations/monte_carlo_ci.png",         "Monte Carlo uncertainty"),
    ("visualizations/all_patients_simulation.png","All-patient simulation"),
    ("visualizations/subgroup_analysis.png",      "Subgroup analysis"),
    ("visualizations/what_if_apoe4.png",          "What-if: APOE4"),
    ("visualizations/what_if_intervention.png",   "What-if: intervention"),
]

GALLERY_METRICS_PNG: list[tuple[str, str]] = [
    ("metrics/confusion_matrix.png",   "Confusion matrix (classification)"),
    ("metrics/roc_curves.png",         "ROC curves"),
    ("metrics/feature_importance.png", "Feature importance"),
    ("metrics/shap_importance.png",    "SHAP summary"),
    ("metrics/training_curves.png",    "LSTM training curves"),
    ("metrics/lstm_eval_plots.png",    "LSTM evaluation plots"),
    ("metrics/evaluation.png",         "Hippocampus prediction accuracy"),
]

# ── Lightbox image info (titles + layman explanations, keyed by filename stem) ─
_IMAGE_INFO: dict[str, dict[str, str]] = {
    "brain_atrophy_3d": {
        "title": "3D Brain Atrophy: Baseline vs 24 Months",
        "html": (
            "<p>This is the digital twin's most visual output — a 3D model of a real patient's "
            "brain, coloured by how much tissue is predicted to have shrunk at each location.</p>"
            "<p><strong>Blue/yellow regions</strong> are relatively healthy — the brain is "
            "maintaining its structure there. <strong>Orange and red regions</strong> indicate "
            "atrophy: the brain is losing tissue, and the warmer the colour, the more severe "
            "the loss.</p>"
            "<p><strong>Left brain (Baseline):</strong> The patient's brain at their first clinic "
            "visit. The red hotspot in the medial temporal lobe marks the hippocampus — already "
            "showing early atrophy at 5,256 mm³, below the healthy reference of ~6,000 mm³.</p>"
            "<p><strong>Right brain (24 months later):</strong> The same patient two years on. "
            "The hippocampus has shrunk further to 4,836 mm³ — an 8% volume loss. More "
            "critically, the MMSE cognitive score has collapsed from 27 to just 8 out of 30, "
            "and the atrophy has spread outward to parietal and frontal regions.</p>"
            "<p>The brain shape itself comes from a standard MRI atlas (MNI152). The atrophy "
            "heatmap is driven entirely by this patient's real ADNI measurements — making it "
            "a true personalised digital twin visualisation.</p>"
        ),
    },
    "results_dashboard": {
        "title": "The Big Picture: Results Dashboard",
        "html": (
            "<p>Think of this as the project's master report card — six panels that tell the "
            "complete story at once.</p>"
            "<p><strong>Panel A (top-left)</strong> counts patients: 1,591 cognitively normal (CN), "
            "1,564 with Mild Cognitive Impairment (MCI), and 595 with dementia. MCI is the "
            "crucial middle stage where early action is most possible.</p>"
            "<p><strong>Panel B (top-centre)</strong> shows how memory-test (MMSE) scores are "
            "distributed across each group. Healthy people cluster near 30 (perfect score); "
            "dementia patients spread lower. This three-way separation is exactly what the "
            "model learns to detect.</p>"
            "<p><strong>Panel C (top-right)</strong> is the headline scorecard. White bars are "
            "the targets we set; coloured bars are where we landed. The MCI→Dementia AUC "
            "of 0.9 means the model is excellent at ranking future risk.</p>"
            "<p><strong>Panels D–F</strong> dig deeper: how fast each group declines per visit, "
            "how brain volume correlates with cognition, and how carrying the APOE4 risk gene "
            "dramatically shifts the proportion of dementia patients.</p>"
        ),
    },
    "population_trajectories": {
        "title": "Three Paths: How Each Group Changes Over Time",
        "html": (
            "<p>Each coloured line is one real patient's MMSE memory-test score tracked across "
            "multiple clinic visits. What you are seeing is Alzheimer's progression — or its "
            "absence — in raw form.</p>"
            "<p><strong>Blue lines (Cognitively Normal):</strong> Scores stay high (27–30) and "
            "relatively flat. These patients are not declining.</p>"
            "<p><strong>Orange lines (MCI):</strong> Scores hover in the mid-range (24–28). "
            "There is more variation — some people are stable, others slowly slipping.</p>"
            "<p><strong>Red lines (Dementia):</strong> These start lower and trend downward. "
            "The thick bold lines are each group's average. The clear separation between the "
            "three groups validates that the data contains the signal needed to train the model.</p>"
            "<p>The dotted horizontal line at MMSE = 24 is the clinical threshold for MCI. "
            "Patients who cross below this line are flagged for closer monitoring.</p>"
        ),
    },
    "demo_trajectories": {
        "title": "5 Real Patients: Observed vs. Predicted",
        "html": (
            "<p>This chart focuses on 5 specific patients from the ADNI dataset. "
            "Each panel covers a single person's journey.</p>"
            "<p>The <strong>solid coloured line</strong> shows their actual measured MMSE scores "
            "at each clinic visit — this is ground truth, real data from real people.</p>"
            "<p>The <strong>dashed purple line</strong> is the model's prediction of what would "
            "happen after their last observed visit, using only the history up to that point. "
            "Where the dashed line closely follows subsequent real data, the model is accurately "
            "forecasting the patient's future.</p>"
            "<p>The horizontal dotted line at MMSE = 24 marks the MCI clinical threshold. "
            "All five patients are diagnosed with dementia, which is why most trajectories "
            "trend toward or below this line. Labels above each panel show the patient's ID, "
            "APOE4 status, and total MMSE drop observed during the study.</p>"
        ),
    },
    "hippocampus_atrophy": {
        "title": "Brain Shrinkage: Tracking the Hippocampus",
        "html": (
            "<p>The hippocampus is a small seahorse-shaped region deep in the brain — "
            "and it is the first area to shrink in Alzheimer's disease. This chart tracks "
            "its volume (in cubic millimetres, mm³) at each clinic visit for 5 patients.</p>"
            "<p>Each bar represents one visit. <strong>Shorter bars = less hippocampal "
            "tissue.</strong> The white overlay line traces the trend. A falling line means "
            "the brain is losing tissue — a direct biological marker of disease progression.</p>"
            "<p><strong>Patient RID 667</strong> shows the most dramatic loss — a 28.7% drop "
            "(2,326 mm³), which is clinically significant. Compare that to "
            "<strong>RID 128</strong> with only a 1.3% change — very stable despite a dementia "
            "diagnosis.</p>"
            "<p>This variability demonstrates why personalised digital twins matter: "
            "one-size-fits-all treatment models miss the huge differences between individuals.</p>"
        ),
    },
    "monte_carlo_ci": {
        "title": "How Certain Are We? Monte Carlo Uncertainty Bands",
        "html": (
            "<p>Rather than making a single prediction, the model was run 200 times with "
            "small random variations — a technique called <strong>Monte Carlo simulation</strong>. "
            "This gives a range of possible futures, not just one answer.</p>"
            "<p>The <strong>shaded bands</strong> represent this range. The darker inner band "
            "covers 50% of all simulations; the lighter outer band covers 90%. "
            "Narrow bands = the model is confident. Wide bands = more uncertainty.</p>"
            "<p><strong>Left chart (MMSE):</strong> The 90% confidence interval is only about "
            "1 MMSE point wide at the last visit — very tight. The model is highly consistent "
            "in its cognitive forecasts for this patient.</p>"
            "<p><strong>Right chart (Hippocampus):</strong> The band is somewhat wider, "
            "reflecting that brain volume is harder to pin down precisely. But the central "
            "prediction still shows a clear downward trend in brain volume over time.</p>"
        ),
    },
    "all_patients_simulation": {
        "title": "Natural Course vs. Early Intervention: All 5 Patients",
        "html": (
            "<p>This is the digital twin's most powerful use case: asking <em>'what if?'</em> "
            "Each panel shows one patient's future under two competing scenarios.</p>"
            "<p>The <strong>solid coloured line</strong> is the real historical data. "
            "The <strong>dashed lines</strong> project two futures: the natural trajectory "
            "(no intervention) and a '40% slowdown' scenario (a hypothetical treatment that "
            "reduces the rate of cognitive decline by 40%).</p>"
            "<p>The gap between the two dashed lines is the treatment benefit. Even a 40% "
            "slowdown compounds significantly over 12 visits — potentially keeping a patient "
            "above the critical MCI threshold for years longer than they would otherwise be.</p>"
            "<p>This type of counterfactual simulation is exactly how pharmaceutical companies "
            "evaluate hypothetical treatments before clinical trials, using patient data to "
            "estimate likely outcomes without exposing anyone to risk.</p>"
        ),
    },
    "subgroup_analysis": {
        "title": "By the Numbers: Group-Level Comparisons",
        "html": (
            "<p>A clean statistical comparison of the three diagnosis groups across three "
            "key dimensions.</p>"
            "<p><strong>Panel 1 — Mean MMSE ± standard deviation:</strong> CN patients average "
            "29.0 (near perfect), MCI patients 27.5, and dementia patients 21.6. "
            "The error bars show spread within each group — dementia has the widest spread, "
            "reflecting how heterogeneous the disease is from person to person.</p>"
            "<p><strong>Panel 2 — Mean Hippocampus Volume:</strong> CN patients have the "
            "largest hippocampi (~5,945 mm³), followed by MCI (~5,778), and dementia the "
            "smallest (~5,483). The differences are statistically meaningful — this is brain "
            "tissue, and even small percentage losses matter clinically.</p>"
            "<p><strong>Panel 3 — Visits per patient:</strong> Most patients attended only "
            "2–3 visits, giving a limited longitudinal window. This is a real constraint of "
            "observational studies — people move, withdraw, or simply stop attending.</p>"
        ),
    },
    "what_if_apoe4": {
        "title": "What If Your Genetics Were Different? APOE4 Scenarios",
        "html": (
            "<p>The <strong>APOE4 gene variant</strong> is the most well-known genetic risk "
            "factor for late-onset Alzheimer's. You can inherit 0, 1, or 2 copies — more "
            "copies means higher risk.</p>"
            "<p>This chart takes one real patient (RID 750) and asks: <em>holding everything "
            "else equal, how would their future trajectory change based solely on their "
            "APOE4 count?</em></p>"
            "<p><strong>Blue (APOE4 = 0, low risk):</strong> The model predicts the highest "
            "sustained MMSE scores and the most stable hippocampus volume over 6 future visits.</p>"
            "<p><strong>Orange (APOE4 = 1, moderate risk):</strong> Slightly lower cognitive "
            "trajectory and marginally faster brain shrinkage.</p>"
            "<p><strong>Red (APOE4 = 2, high risk):</strong> The steepest predicted decline. "
            "The spread between low and high risk is roughly 4 MMSE points — clinically "
            "meaningful, as it can be the difference between scoring above or below the MCI "
            "threshold on any given visit.</p>"
        ),
    },
    "what_if_intervention": {
        "title": "What If We Could Slow the Decline? Intervention Scenarios",
        "html": (
            "<p>For patient RID 750, this chart models four futures based on how effective a "
            "hypothetical treatment might be at slowing disease progression.</p>"
            "<p><strong>Red (No intervention):</strong> The baseline 'do nothing' path — "
            "cognitive and brain decline continue at the natural predicted rate.</p>"
            "<p><strong>Orange (20% slowdown):</strong> A modest treatment reduces the "
            "decline rate by one-fifth. The MMSE line is slightly higher; the hippocampus "
            "shrinks a little more slowly.</p>"
            "<p><strong>Green (40% slowdown):</strong> A stronger treatment cuts decline "
            "nearly in half. The benefit becomes clearly visible, especially in the "
            "hippocampus panel on the right, where the lines diverge significantly over time.</p>"
            "<p><strong>Blue (Complete halt):</strong> A theoretical ceiling — if progression "
            "could be completely stopped, scores stay flat. This serves as an upper bound "
            "to compare other scenarios against.</p>"
            "<p>Key insight: even small interventions <em>compound</em> over time. "
            "Starting treatment early matters enormously.</p>"
        ),
    },
    "confusion_matrix": {
        "title": "How Often Does the Classifier Get It Right?",
        "html": (
            "<p>A confusion matrix is a grid comparing what the model <em>predicted</em> "
            "(columns) against what was <em>actually true</em> (rows). "
            "The diagonal numbers are correct predictions; everything off-diagonal is a mistake.</p>"
            "<p><strong>Reading the grid:</strong></p>"
            "<ul>"
            "<li>The model correctly identified <strong>155 out of 207 CN patients</strong> "
            "(75%), with 52 misclassified as MCI.</li>"
            "<li>It correctly identified <strong>327 out of 367 dementia patients</strong> "
            "(89%) — very strong performance on the most critical class.</li>"
            "<li>It correctly identified <strong>208 out of 325 MCI patients</strong> "
            "(64%) — MCI is the hardest class because it sits between normal and dementia.</li>"
            "</ul>"
            "<p>Overall cross-validated accuracy is 77%. The most common mistake is "
            "confusing MCI with dementia — which is also a challenge for human clinicians, "
            "since MCI is inherently a borderline diagnosis.</p>"
        ),
    },
    "roc_curves": {
        "title": "Detecting Future Dementia: The ROC Curve",
        "html": (
            "<p>This chart measures how well the model identifies which MCI patients will "
            "eventually progress to dementia — a crucial clinical prediction.</p>"
            "<p>The <strong>ROC curve</strong> (blue line) shows the trade-off between "
            "catching true cases (sensitivity, y-axis) and avoiding false alarms "
            "(1-specificity, x-axis). A perfect model curves sharply to the top-left corner. "
            "A random guess follows the diagonal dashed line.</p>"
            "<p>The <strong>AUC = 0.892</strong> means that 89.2% of the time, the model "
            "correctly ranks a patient who will develop dementia as higher risk than one who "
            "will not. In clinical screening, an AUC above 0.85 is considered excellent.</p>"
            "<p>The orange crosshairs mark a useful operating point: at roughly 20% false "
            "positive rate, we achieve about 80% true positive rate — catching 4 out of 5 "
            "future dementia cases while only falsely flagging 1 in 5 currently healthy patients.</p>"
        ),
    },
    "feature_importance": {
        "title": "What Clues Matter Most? XGBoost Feature Importance",
        "html": (
            "<p>These bar charts show which data variables the XGBoost classification model "
            "relies on most when deciding a patient's diagnosis or predicting conversion.</p>"
            "<p><strong>Left chart — 3-class (CN / MCI / Dementia):</strong> The most recent "
            "MMSE score dominates by a wide margin. Your current cognitive performance is the "
            "single best predictor of your diagnosis. Number of visits and baseline MMSE "
            "round out the top three.</p>"
            "<p><strong>Right chart — MCI → Dementia conversion:</strong> Current MMSE leads "
            "again, but now 'total MMSE drop' jumps to second place — because the "
            "<em>trajectory</em> of decline matters more than just the current score when "
            "predicting whether MCI will eventually convert to dementia.</p>"
            "<p>Notably, APOE4 (genetics) ranks near the bottom in both charts. This does "
            "not mean genetics are unimportant — it means their effect is likely already "
            "captured through the cognitive and imaging measurements.</p>"
        ),
    },
    "shap_importance": {
        "title": "A Deeper Look at Impact: SHAP Analysis",
        "html": (
            "<p>SHAP (SHapley Additive exPlanations) is a rigorous method for understanding "
            "exactly how much each feature changes the model's prediction for individual "
            "patients — going beyond simple feature importance rankings.</p>"
            "<p>The bar chart shows the average absolute SHAP value for each feature. "
            "A longer bar means that feature has a bigger impact on individual predictions "
            "across the dataset.</p>"
            "<p><strong>MMSE last</strong> (most recent cognitive score) has by far the "
            "largest impact. <strong>n visits</strong> (how long someone has been in the study) "
            "is second — a proxy for disease duration. "
            "<strong>MMSE slope and Hippo slope</strong> follow — capturing the rate of "
            "change, not just the current state.</p>"
            "<p>SHAP makes the model interpretable: a clinician can see exactly <em>why</em> "
            "a patient was flagged as high-risk, rather than treating the AI as a black box. "
            "This kind of transparency is essential for clinical adoption.</p>"
        ),
    },
    "training_curves": {
        "title": "Did the Model Actually Learn? LSTM Training Curves",
        "html": (
            "<p>These two charts show the LSTM neural network learning in real time "
            "across 50 training epochs — one epoch means the model processed every "
            "training example once.</p>"
            "<p><strong>Left chart (Loss curves):</strong> Both training loss (blue) and "
            "validation loss (orange) start high and fall steeply — this is exactly what "
            "healthy learning looks like. Crucially, the validation curve follows the training "
            "curve closely without flattening early. This means the model is generalising "
            "to unseen data rather than memorising the training set (overfitting).</p>"
            "<p><strong>Right chart (Validation MAE vs. Baseline):</strong> The purple line "
            "shows prediction error on new data dropping from 6.5 MMSE points all the way "
            "down to about 3.1. The dashed orange line at 3.5 is the naive baseline (just use "
            "the last known score as your prediction). Our LSTM beats this baseline around "
            "epoch 40 — proving it learned genuine patterns that a simple heuristic cannot.</p>"
        ),
    },
    "lstm_eval_plots": {
        "title": "LSTM Prediction Quality: Three Lenses",
        "html": (
            "<p>Three panels that together paint a complete picture of how well the LSTM "
            "predicts cognitive scores.</p>"
            "<p><strong>Left — Predicted vs. Actual MMSE (R² = 0.689):</strong> Each blue "
            "dot is one patient visit. Dots close to the red diagonal line = accurate "
            "predictions. An R² of 0.689 means the model explains 69% of the variation in "
            "MMSE scores — respectable for a noisy biological outcome.</p>"
            "<p><strong>Centre — Residual Distribution:</strong> 'Residual' means predicted "
            "minus actual. A good model has residuals clustered tightly around zero — no "
            "systematic bias in either direction. The purple distribution here is centred "
            "at zero and bell-shaped, which is exactly what we want.</p>"
            "<p><strong>Right — Baseline vs. LSTM MAE:</strong> A simple baseline (just "
            "repeat the last score) makes an average error of 3.5 MMSE points. Our LSTM "
            "achieves 1.76 — cutting the error in half. In clinical terms, being wrong by "
            "1.76 points instead of 3.5 on a 30-point scale is a meaningful improvement.</p>"
        ),
    },
    "evaluation": {
        "title": "Hippocampus Volume Predictions: Near-Perfect Accuracy",
        "html": (
            "<p>Each green dot represents one patient visit where the model predicted their "
            "hippocampus volume (y-axis) and we compared it to the actual MRI measurement "
            "(x-axis). The dashed diagonal line is perfect prediction.</p>"
            "<p>The dots hug the diagonal almost perfectly — the LSTM achieves R² = 0.99 "
            "and a Mean Absolute Error of just 69 mm³. With average hippocampus volumes "
            "around 5,000 mm³, a 69 mm³ error is less than 1.4% — exceptional accuracy "
            "for a neural sequence model.</p>"
            "<p>Why is this so much better than MMSE prediction? Brain volume measured by "
            "MRI is a more stable biological measurement than a cognitive test score. MRI "
            "volumes do not fluctuate day-to-day due to mood, sleep, or test anxiety — so "
            "there is less 'noise' for the model to fight against.</p>"
            "<p>This result validates that the digital twin can faithfully track biological "
            "disease markers, not just cognitive symptoms.</p>"
        ),
    },
}

# Serialise to a JS const that can be embedded in the page
_JS_IMAGE_INFO_CONST = "const IMAGE_INFO = " + json.dumps(_IMAGE_INFO, ensure_ascii=False) + ";"


# ── Helper: friendly title for gallery captions ───────────────────────────────
def _gallery_title(rel: str, fallback: str) -> str:
    stem = Path(rel).stem
    return _IMAGE_INFO.get(stem, {}).get("title", fallback)


# ── Data loading ──────────────────────────────────────────────────────────────
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


def normalize_team_demo_data(raw: dict) -> dict | None:
    if not raw:
        return None
    patients: dict[str, dict] = {}
    demo_rids: list[int] = []
    for key in sorted(raw.keys(), key=lambda k: int(str(k))):
        blob = raw[key]
        if not isinstance(blob, dict) or "observed_visits" not in blob:
            continue
        rid = int(blob.get("rid", key))
        sk = str(rid)
        ov = [float(x) for x in blob["observed_visits"]]
        om = [float(x) for x in blob["observed_mmse"]]
        pv = [float(x) for x in blob.get("pred_visits", [])]
        pm = [float(x) for x in blob.get("pred_mmse", [])]
        # Read hippocampus directly from the exported arrays written by the notebook
        ho = [float(x) for x in blob.get("observed_hippo", [])]
        hp = [float(x) for x in blob.get("pred_hippo", [])]
        try:
            apoe_f = float(blob.get("apoe4", 0) or 0)
        except (TypeError, ValueError):
            apoe_f = 0.0
        patients[sk] = {
            "rid": rid, "apoe4": apoe_f, "dx_last": str(blob.get("dx_last", "—")),
            "n_obs_visits": len(ov), "visits_obs": ov, "mmse_obs": om,
            "hippo_obs": ho, "pred_visits": pv, "mmse_pred": pm, "hippo_pred": hp,
        }
        demo_rids.append(rid)
    if not patients:
        return None
    return {"version": 1, "prediction_method": "notebook_02_export",
            "demo_rids": demo_rids, "patients": patients}


def load_demo_simulation() -> dict | None:
    if not DEMO_DATA_PATH.exists():
        return None
    try:
        raw = json.loads(DEMO_DATA_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return normalize_team_demo_data(raw)


DEMO_SIM = load_demo_simulation()


def hub_asset_url(rel: str) -> str:
    return f"/hub-results/{rel.replace(chr(92), '/')}"


# ── Plotly figures ────────────────────────────────────────────────────────────
def metrics_figure(data: dict | None) -> go.Figure:
    if not data or not all(k in data for k in ("mmse_mae", "mmse_r2", "hippo_mae", "hippo_r2")):
        fig = go.Figure()
        fig.add_annotation(
            text="No regression metrics yet.<br><sub>Export metrics to lstm_metrics.json</sub>",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#7a7a73"),
        )
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=40, r=40, t=40, b=40), height=320)
        return fig
    mmse_mae, mmse_r2 = float(data["mmse_mae"]), float(data["mmse_r2"])
    hippo_mae, hippo_r2 = float(data["hippo_mae"]), float(data["hippo_r2"])
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Mean absolute error", "R² (coefficient of determination)"),
                        vertical_spacing=0.22)
    fig.add_trace(go.Bar(x=["MMSE", "Hippocampus"], y=[mmse_mae, hippo_mae],
                         marker=dict(color="#2f4f3f", line=dict(width=0)),
                         text=[f"{mmse_mae:.2f}", f"{hippo_mae:.1f}"],
                         textposition="outside", textfont=dict(size=11, color="#3c3c38"),
                         showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=["MMSE", "Hippocampus"], y=[mmse_r2, hippo_r2],
                         marker=dict(color="#8fa396", line=dict(width=0)),
                         text=[f"{mmse_r2:.3f}", f"{hippo_r2:.4f}"],
                         textposition="outside", textfont=dict(size=11, color="#3c3c38"),
                         showlegend=False), row=2, col=1)
    axis_style = dict(showgrid=True, gridcolor="#e6e6e0", zeroline=False,
                      linecolor="#e6e6e0", tickfont=dict(size=11))
    for r in (1, 2):
        fig.update_xaxes(axis_style, row=r, col=1)
        fig.update_yaxes(axis_style, row=r, col=1)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafaf7",
                      font=dict(family="DM Sans, sans-serif", color="#3c3c38", size=12),
                      margin=dict(l=44, r=28, t=72, b=36), height=360)
    return fig


def classification_figure(data: dict | None) -> go.Figure | None:
    if not data or not all(k in data for k in ("clf3_cv_accuracy", "mci_conv_auc_cv")):
        return None
    acc, auc = float(data["clf3_cv_accuracy"]), float(data["mci_conv_auc_cv"])
    acc_std = float(data.get("clf3_cv_accuracy_std", 0) or 0)
    auc_std = float(data.get("mci_conv_auc_cv_std", 0) or 0)
    fig = go.Figure(go.Bar(
        x=["3-class accuracy (CV)", "MCI → Dementia AUC (CV)"], y=[acc, auc],
        error_y=dict(type="data", array=[acc_std, auc_std], color="#7a7a73", thickness=1.5, width=6),
        marker=dict(color=["#2f4f3f", "#8fa396"], line=dict(width=0)),
        text=[f"{acc:.3f} ± {acc_std:.3f}", f"{auc:.3f} ± {auc_std:.3f}"],
        textposition="outside", textfont=dict(size=11, color="#3c3c38"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafaf7",
        font=dict(family="DM Sans, sans-serif", color="#3c3c38", size=12),
        xaxis=dict(showgrid=False, linecolor="#e6e6e0", tickfont=dict(size=11)),
        yaxis=dict(gridcolor="#e6e6e0", zeroline=False, range=[0, 1.05],
                   title=dict(text="Score", font=dict(size=11, color="#7a7a73"))),
        margin=dict(l=48, r=28, t=28, b=72), height=300,
    )
    return fig


def regression_metric_cards(data: dict | None) -> html.Div:
    if not data:
        return html.Div(className="metrics-grid", children=[html.Div(
            className="metric-card", children=[
                html.Span("Regression", className="metric-card__label"),
                html.Div("—", className="metric-card__value"),
                html.P("Export lstm_metrics.json after LSTM training.", className="metric-card__hint"),
            ])])
    specs = [
        ("mmse_mae",   "MMSE — mean absolute error", "{:.4f}", "Lower is better"),
        ("mmse_r2",    "MMSE — R²",                  "{:.4f}", "Variance explained"),
        ("hippo_mae",  "Hippocampus — MAE",           "{:.2f}", "Lower is better"),
        ("hippo_r2",   "Hippocampus — R²",            "{:.4f}", "Variance explained"),
    ]
    cards = [html.Div(className="metric-card", children=[
                html.Span(label, className="metric-card__label"),
                html.Div(fmt.format(float(data[key])), className="metric-card__value"),
                html.P(hint, className="metric-card__hint"),
             ]) for key, label, fmt, hint in specs if key in data]
    return html.Div(className="metrics-grid", children=cards or regression_metric_cards(None).children)


def classification_metric_cards(data: dict | None) -> html.Div | None:
    if not data or "clf3_cv_accuracy" not in data:
        return None
    acc, acc_std = float(data["clf3_cv_accuracy"]), float(data.get("clf3_cv_accuracy_std", 0) or 0)
    items = [("3-class accuracy (CV)", f"{acc:.4f} ± {acc_std:.4f}", "Stratified CV, XGBoost")]
    if "mci_conv_auc_cv" in data:
        auc = float(data["mci_conv_auc_cv"])
        auc_std = float(data.get("mci_conv_auc_cv_std", 0) or 0)
        items.append(("MCI → Dementia AUC (CV)", f"{auc:.4f} ± {auc_std:.4f}", "Conversion risk ranking"))
    return html.Div(className="metrics-grid", children=[
        html.Div(className="metric-card", children=[
            html.Span(label, className="metric-card__label"),
            html.Div(val, className="metric-card__value"),
            html.P(hint, className="metric-card__hint"),
        ]) for label, val, hint in items])


def simulation_metric_cards(sim: dict | None) -> html.Div | None:
    if not sim:
        return None
    mapping = [
        ("base_patient",            "Base patient RID",             "{}",    "Anchor record for twin runs"),
        ("n_future_visits",         "Future visits simulated",      "{}",    "Horizon length"),
        ("mc_samples",              "Monte Carlo samples",          "{}",    "Draws for uncertainty"),
        ("mc_ci90_width_last",      "90% CI width (last visit)",    "{:.2f}","MMSE points, approximate"),
        ("apoe4_mmse_spread",       "APOE4 MMSE spread",            "{:.2f}","Exploratory what-if delta"),
        ("intervention_40pct_benefit","40% intervention benefit",   "{:.2f}","MMSE change vs baseline path"),
    ]
    cards = []
    for key, label, fmt, hint in mapping:
        if key not in sim:
            continue
        raw = sim[key]
        val = fmt.format(float(raw)) if isinstance(raw, float) else str(raw)
        cards.append(html.Div(className="metric-card", children=[
            html.Span(label, className="metric-card__label"),
            html.Div(val, className="metric-card__value"),
            html.P(hint, className="metric-card__hint"),
        ]))
    return html.Div(className="metrics-grid", children=cards) if cards else None


# ── Gallery ───────────────────────────────────────────────────────────────────
def figure_gallery_rows(groups: list[tuple[str, list[tuple[str, str]]]]) -> list:
    """Build clickable figure cards; skip missing files."""
    rows: list = []
    for group_title, entries in groups:
        present = [(rel, cap) for rel, cap in entries if (RESULTS_ROOT / rel).is_file()]
        if not present:
            continue
        rows.append(html.H3(group_title, className="metrics-block__title"))
        rows.append(html.Div(
            className="gallery-grid",
            children=[
                html.Figure(
                    className="figure-card",
                    children=[
                        html.Img(src=hub_asset_url(rel), alt=cap),
                        html.Figcaption(_gallery_title(rel, cap)),
                    ],
                )
                for rel, cap in present
            ],
        ))
    return rows


# ── Demo twin helpers ─────────────────────────────────────────────────────────
def _demo_patient(demo: dict, rid: str) -> dict | None:
    if not demo or "patients" not in demo:
        return None
    return demo["patients"].get(str(rid))


def _plotly_layout_base(title: str, yaxis_title: str, height: int = 400) -> dict:
    return dict(
        title=dict(text=title, font=dict(family="Fraunces, Georgia, serif", size=15,
                                          color="#121211"), x=0, xanchor="left",
                   pad=dict(t=4, b=4)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fafaf7",
        font=dict(family="DM Sans, sans-serif", size=12, color="#3c3c38"),
        margin=dict(l=56, r=28, t=80, b=56), height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right",
                    x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(title=dict(text="Visit number", font=dict(size=11, color="#7a7a73")),
                   gridcolor="#e6e6e0", zeroline=False, linecolor="#e6e6e0"),
        yaxis=dict(title=dict(text=yaxis_title, font=dict(size=11, color="#7a7a73")),
                   gridcolor="#e6e6e0", zeroline=False, linecolor="#e6e6e0"),
        hovermode="x unified",
    )


def demo_mmse_figure(rid: str, demo: dict | None) -> go.Figure:
    fig = go.Figure()
    if not demo:
        fig.update_layout(**_plotly_layout_base("MMSE trajectory", "MMSE (0–30)", 420))
        fig.add_annotation(text="Add results/metrics/demo_data.json", xref="paper",
                           yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(color="#7a7a73", size=14))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        return fig
    p = _demo_patient(demo, rid)
    if not p:
        fig.update_layout(**_plotly_layout_base("MMSE trajectory", "MMSE (0–30)", 420))
        fig.add_annotation(text="Patient not found", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color="#7a7a73", size=14))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        return fig

    v_obs, m_obs = p["visits_obs"], p["mmse_obs"]
    v_pred       = p["pred_visits"]
    m_pred       = p["mmse_pred"]

    # ── shade the prediction zone ──────────────────────────────────────
    if v_pred:
        x_zone = [v_obs[-1]] + v_pred + v_pred[::-1] + [v_obs[-1]]
        y_hi   = [m_obs[-1]] + [min(mp + 2.0, 30) for mp in m_pred]
        y_lo   = [m_obs[-1]] + [max(mp - 2.0, 0)  for mp in m_pred]
        fig.add_trace(go.Scatter(
            x=x_zone, y=y_hi + y_lo[::-1],
            fill="toself", fillcolor="rgba(143,163,150,0.12)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"))

    # ── MCI danger zone shading (0–24) ─────────────────────────────────
    all_v = list(v_obs) + list(v_pred)
    if all_v:
        fig.add_trace(go.Scatter(
            x=[min(all_v) - 0.3, max(all_v) + 0.3, max(all_v) + 0.3, min(all_v) - 0.3],
            y=[0, 0, 24, 24],
            fill="toself", fillcolor="rgba(239,83,80,0.04)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"))

    # ── Observed trace ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=v_obs, y=m_obs, mode="lines+markers", name="Observed (ADNI)",
        line=dict(color="#2f4f3f", width=3),
        marker=dict(size=10, color="#2f4f3f", line=dict(width=2, color="#fafaf7")),
        hovertemplate="Visit %{x}<br>MMSE: <b>%{y:.1f}</b><extra>Observed</extra>"))

    # ── Predicted trace ────────────────────────────────────────────────
    if v_pred:
        v_roll = [v_obs[-1]] + list(v_pred)
        m_roll = [m_obs[-1]] + list(m_pred)
        fig.add_trace(go.Scatter(
            x=v_roll, y=m_roll, mode="lines+markers", name="Predicted (LSTM)",
            line=dict(color="#5b8c6e", width=2.5, dash="dash"),
            marker=dict(size=9, symbol="diamond", color="#5b8c6e",
                        line=dict(width=2, color="#fafaf7")),
            hovertemplate="Visit %{x}<br>MMSE: <b>%{y:.1f}</b><extra>Predicted</extra>"))

    # ── MCI threshold line ─────────────────────────────────────────────
    fig.add_hline(y=24, line_dash="dot", line_color="#d4a843", line_width=1.5, opacity=0.8)
    fig.add_annotation(
        xref="paper", yref="y", x=0.02, y=25.0,
        text="⚠ MCI threshold (24)",
        showarrow=False, font=dict(size=10, color="#c4943a", family="DM Sans"), xanchor="left")

    # ── Score band annotations ─────────────────────────────────────────
    fig.add_annotation(
        xref="paper", yref="y", x=0.99, y=28.5,
        text="Normal (≥27)", showarrow=False,
        font=dict(size=9, color="#8fa396"), xanchor="right")
    fig.add_annotation(
        xref="paper", yref="y", x=0.99, y=21,
        text="MCI zone", showarrow=False,
        font=dict(size=9, color="#ef5350", family="DM Sans"), xanchor="right")

    layout = _plotly_layout_base(f"MMSE Trajectory — RID {p['rid']}", "MMSE score (0–30)", 420)
    layout["yaxis"]["range"] = [0, 32]
    layout["yaxis"]["tickvals"] = [0, 10, 20, 24, 27, 30]
    layout["xaxis"]["dtick"] = 1
    fig.update_layout(**layout)
    return fig


def demo_hippo_figure(rid: str, demo: dict | None) -> go.Figure:
    import math as _math
    fig = go.Figure()
    if not demo:
        fig.update_layout(**_plotly_layout_base("Hippocampus trajectory", "Volume (mm³)", 420))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        return fig
    p = _demo_patient(demo, rid)
    if not p:
        fig.update_layout(**_plotly_layout_base("Hippocampus trajectory", "Volume (mm³)", 420))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        return fig

    v_obs = p["visits_obs"]
    h_obs = p["hippo_obs"]
    hp    = p.get("hippo_pred", [])
    pv    = p.get("pred_visits", [])
    has_pred = bool(hp and pv and not all(
        _math.isnan(h) if isinstance(h, float) else False for h in hp))

    # ── Healthy hippocampus reference band (5500–6500 mm³) ─────────────
    all_v = list(v_obs) + (list(pv) if has_pred else [])
    if all_v:
        xmin, xmax = min(all_v) - 0.3, max(all_v) + 0.3
        fig.add_trace(go.Scatter(
            x=[xmin, xmax, xmax, xmin],
            y=[5500, 5500, 6500, 6500],
            fill="toself", fillcolor="rgba(79,195,247,0.06)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_annotation(
            xref="paper", yref="y", x=0.99, y=6400,
            text="Healthy range (5500–6500)", showarrow=False,
            font=dict(size=9, color="#4fc3f7"), xanchor="right")

    # ── Compute % change from baseline for annotation ──────────────────
    pct_change = None
    if h_obs and hp:
        baseline = h_obs[0]
        final    = hp[-1] if hp else h_obs[-1]
        if baseline > 0:
            pct_change = (final - baseline) / baseline * 100

    # ── Prediction uncertainty band ────────────────────────────────────
    if has_pred:
        v_roll = [v_obs[-1]] + list(pv)
        h_roll = [h_obs[-1]] + list(hp)
        band   = [max(h * 0.015, 30) for h in hp]   # ±1.5% uncertainty
        y_hi   = [h_obs[-1]] + [h + b for h, b in zip(hp, band)]
        y_lo   = [h_obs[-1]] + [h - b for h, b in zip(hp, band)]
        fig.add_trace(go.Scatter(
            x=v_roll + v_roll[::-1], y=y_hi + y_lo[::-1],
            fill="toself", fillcolor="rgba(143,163,150,0.12)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"))

    # ── Observed trace ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=v_obs, y=h_obs, mode="lines+markers", name="Observed (ADNI)",
        line=dict(color="#2f4f3f", width=3),
        marker=dict(size=10, color="#2f4f3f", line=dict(width=2, color="#fafaf7")),
        hovertemplate="Visit %{x}<br>Vol: <b>%{y:,.0f} mm³</b><extra>Observed</extra>",
        connectgaps=False))

    # ── Predicted trace ────────────────────────────────────────────────
    if has_pred:
        fig.add_trace(go.Scatter(
            x=v_roll, y=h_roll, mode="lines+markers", name="Predicted (atrophy model)",
            line=dict(color="#5b8c6e", width=2.5, dash="dash"),
            marker=dict(size=9, symbol="diamond", color="#5b8c6e",
                        line=dict(width=2, color="#fafaf7")),
            hovertemplate="Visit %{x}<br>Vol: <b>%{y:,.0f} mm³</b><extra>Predicted</extra>",
            connectgaps=False))

    # ── % atrophy annotation ───────────────────────────────────────────
    if pct_change is not None:
        sign  = "▼" if pct_change < 0 else "▲"
        color = "#ef5350" if pct_change < -3 else "#8fa396"
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.08,  # slightly higher → better spacing
            text=f"{sign} {abs(pct_change):.1f}% projected atrophy",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",  # 🔥 important
            align="right",     # 🔥 text alignment inside box
            font=dict(size=11, color=color, family="DM Sans"),
            bgcolor="rgba(250,250,247,0.9)",
            bordercolor=color,
            borderwidth=1,
            borderpad=6        # 🔥 adds breathing space
        )

    # ── Layout ─────────────────────────────────────────────────────────
    all_vals = list(h_obs) + (list(hp) if has_pred else [])
    y_min = max(min(all_vals) * 0.94, 0) if all_vals else 3000
    y_max = max(all_vals) * 1.04         if all_vals else 8000

    layout = _plotly_layout_base(
        f"Hippocampus Volume — RID {p['rid']}", "Volume (mm³)", 420)
    layout["yaxis"]["range"]      = [y_min, y_max]
    layout["yaxis"]["tickformat"] = ",.0f"
    layout["xaxis"]["dtick"]      = 1
    fig.update_layout(**layout)
    return fig


def demo_meta_children(rid: str, demo: dict | None) -> list:
    if not demo:
        return [html.P("No demo bundle loaded.", className="demo-meta__empty")]
    p = _demo_patient(demo, rid)
    if not p:
        return [html.P("Patient not found.", className="demo-meta__empty")]
    method = demo.get("prediction_method", "unknown")
    method_label = {
        "notebook_02_export": "LSTM predictions bundled with 02_lstm_model.ipynb → demo_data.json",
        "lstm_rollout": "LSTM autoregressive rollout",
        "linear_extrapolation": "Linear extrapolation",
    }.get(method, method.replace("_", " "))
    items = [
        ("RID", str(p["rid"])), ("APOEε4 alleles", str(p["apoe4"])),
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


# ── Brain atrophy 3D section ──────────────────────────────────────────────────
BRAIN_HTML_PATH  = (RESULTS_ROOT / "visualizations" / "brain_atrophy_3d.html").resolve()
BRAIN_RIDS       = [750, 667, 1282, 50, 128]
BRAIN_HTML_PATHS = {
    rid: (RESULTS_ROOT / "visualizations" / f"brain_atrophy_3d_{rid}.html").resolve()
    for rid in BRAIN_RIDS
}

def build_brain_section() -> html.Section:
    # Build per-patient dropdown options
    patient_labels = {
        750:  "RID 750  ·  APOE4=1  ·  Hippo 5,256→4,836 mm³  (▼8.0%)",
        667:  "RID 667  ·  APOE4=1  ·  Hippo 8,116→8,038 mm³  (▼1.0%)",
        1282: "RID 1282 ·  APOE4=0  ·  Hippo 4,601→4,143 mm³  (▼9.9%)",
        50:   "RID 50   ·  APOE4=0  ·  Hippo 4,831→4,556 mm³  (▼5.7%)",
        128:  "RID 128  ·  APOE4=2  ·  Hippo 5,703→4,985 mm³  (▼12.6%)",
    }

    any_available = any(p.is_file() for p in BRAIN_HTML_PATHS.values())

    # Dropdown to pick patient
    patient_dropdown = html.Div(
        style={"marginBottom": "1.25rem"},
        children=[
            html.Label(
                "SELECT PATIENT",
                style={"fontSize": "0.7rem", "letterSpacing": "0.08em",
                       "color": "#9a9a92", "fontWeight": "500",
                       "display": "block", "marginBottom": "0.4rem"},
            ),
            dcc.Dropdown(
                id="brain-patient-select",
                options=[
                    {"label": label, "value": rid}
                    for rid, label in patient_labels.items()
                    if BRAIN_HTML_PATHS[rid].is_file()
                ] if any_available else [{"label": "No HTML files found — run notebook", "value": 0}],
                value=750 if BRAIN_HTML_PATHS.get(750, None) and BRAIN_HTML_PATHS[750].is_file() else (
                    next((r for r in BRAIN_RIDS if BRAIN_HTML_PATHS[r].is_file()), 0)
                ),
                clearable=False,
                style={"fontFamily": "DM Sans, Inter, sans-serif",
                       "fontSize": "0.875rem",
                       "maxWidth": "620px"},
            ),
        ],
    )

    iframe_container = html.Div(
        id="brain-iframe-container",
        className="brain-iframe-wrap",
        children=[
            html.P(
                "Run 06_brain_visualization.ipynb to generate per-patient HTML files, "
                "then reload this page.",
                className="section__lead",
            )
        ] if not any_available else [
            html.Iframe(
                id="brain-iframe",
                src=f"/hub-results/visualizations/brain_atrophy_3d_750.html",
                style={
                    "width": "100%",
                    "height": "600px",
                    "border": "none",
                    "borderRadius": "8px",
                    "background": "#0d0f1a",
                },
            )
        ]
    )

    return html.Section(id="brain-3d", className="section", children=[
        html.Div(className="section__inner", children=[
            html.P("3D Digital Twin", className="section__label"),
            html.H2(
                "Brain atrophy — all 5 demo patients.",
                className="section__title",
            ),
            html.P(
                "Personalised 3D brain models coloured by atrophy severity at each location. "
                "Select a patient from the dropdown to explore their unique trajectory. "
                "Use the Baseline / Latest visit / Overlay buttons inside the viewer to compare. "
                "Driven by real ADNI hippocampus volumes and MMSE scores.",
                className="section__lead",
            ),
            patient_dropdown,
            # Stats row — updated dynamically by callback
            html.Div(id="brain-stats-row", className="brain-stats-row"),
            iframe_container,
            html.P(
                "💡 Tip: click and drag to rotate · scroll to zoom · double-click to reset view",
                style={"fontSize": "0.8125rem", "color": "#9a9a92", "marginTop": "0.5rem",
                       "textAlign": "center"},
            ),
        ])
    ])


# ── Demo simulation section ───────────────────────────────────────────────────
def build_demo_simulation_section() -> html.Section:
    if DEMO_SIM and DEMO_SIM.get("patients"):
        rids = [str(r) for r in DEMO_SIM.get("demo_rids", [])]
        first = rids[0] if rids else None
        patients = DEMO_SIM["patients"]
        options = [{"label": f"RID {r}  ·  {patients.get(r, {}).get('dx_last','—')}  ·  "
                             f"APOEε4 = {patients.get(r, {}).get('apoe4','—')}", "value": r}
                   for r in rids]
        return html.Section(id="demo-simulation", className="section", children=[
            html.Div(className="section__inner", children=[
                html.P("Interactive", className="section__label"),
                html.H2("Demo patient simulation.", className="section__title"),
                html.P("Five demo subjects from results/metrics/demo_data.json, produced when "
                       "you run 02_lstm_model.ipynb (observed prefix + LSTM one-step forecasts "
                       "on held-out visit indices).", className="section__lead"),
                html.Div(className="demo-toolbar", children=[
                    html.Label("Patient", className="demo-toolbar__label",
                               htmlFor="demo-patient-select"),
                    dcc.Dropdown(id="demo-patient-select", options=options, value=first,
                                 clearable=False, className="demo-dropdown",
                                 style={"maxWidth": "28rem", "fontSize": "0.9375rem"}),
                ]),
                html.Div(id="demo-patient-meta", className="demo-meta",
                         children=demo_meta_children(first or "", DEMO_SIM)),
                html.Div(className="demo-charts", children=[
                    html.Div(className="chart-wrap chart-wrap--demo", children=[
                        dcc.Graph(id="demo-graph-mmse",
                                  figure=demo_mmse_figure(first or "", DEMO_SIM),
                                  config=dict(displayModeBar=False), style={"height": "420px"})]),
                    html.Div(className="chart-wrap chart-wrap--demo", children=[
                        dcc.Graph(id="demo-graph-hippo",
                                  figure=demo_hippo_figure(first or "", DEMO_SIM),
                                  config=dict(displayModeBar=False), style={"height": "420px"})]),
                ]),
            ])])
    return html.Section(id="demo-simulation", className="section", children=[
        html.Div(className="section__inner", children=[
            html.P("Interactive", className="section__label"),
            html.H2("Demo patient simulation.", className="section__title"),
            html.P("Run 02_lstm_model.ipynb to write results/metrics/demo_data.json, "
                   "then reload this page.", className="section__lead"),
        ])])


# ── Lightbox modal (manipulated by JS, not Dash callbacks) ────────────────────
def build_lightbox() -> html.Div:
    return html.Div(
        id="lightbox-overlay",
        className="lb-overlay",
        children=[
            html.Div(className="lb-card", children=[
                # ── Left pane: full image ──────────────────────────────────
                html.Div(className="lb-img-pane", children=[
                    html.Img(id="lb-main-img", src="", alt=""),
                ]),
                # ── Right pane: title + explanation ───────────────────────
                html.Div(className="lb-info-pane", children=[
                    html.Button("×", className="lb-close", title="Close (Esc)"),
                    html.P("Figure Analysis", className="lb-label"),
                    html.H2("", id="lb-title", className="lb-title"),
                    html.Div(className="lb-divider"),
                    html.Div(id="lb-explanation", className="lb-explanation"),
                    html.Div(className="lb-footer",
                             children=html.P("Scroll for more ↓", className="lb-scroll-hint")),
                ]),
            ])
        ],
    )


# ── Main layout ───────────────────────────────────────────────────────────────
def build_layout() -> html.Div:
    metrics = load_metrics()
    simulation = load_simulation_summary()
    clf_fig = classification_figure(metrics)

    gallery_children = figure_gallery_rows([
        ("Trajectory & policy figures", GALLERY_VISUALS),
        ("Model diagnostics",           GALLERY_METRICS_PNG),
    ])

    metrics_children: list = [
        html.P("Evidence", className="section__label"),
        html.H2("Exported metrics & charts.", className="section__title"),
        html.P("Regression and classification values are read from "
               "results/metrics/lstm_metrics.json. "
               "Simulation summaries use simulation_summary.json when present.",
               className="section__lead"),
        html.Div(className="metrics-block", children=[
            html.H3("Regression (LSTM)", className="metrics-block__title"),
            html.P("MMSE and hippocampal volume heads.", className="metrics-block__lead"),
            regression_metric_cards(metrics),
            html.Div(className="chart-wrap", children=[
                dcc.Graph(id="metrics-chart", figure=metrics_figure(metrics),
                          config=dict(displayModeBar=False), style={"height": "380px"})]),
        ]),
    ]

    clf_cards = classification_metric_cards(metrics)
    if clf_cards:
        clf_block: list = [
            html.H3("Classification (XGBoost)", className="metrics-block__title"),
            html.P("Three-class diagnosis and MCI-to-dementia conversion signals from "
                   "the patient-level pipeline.", className="metrics-block__lead"),
            clf_cards,
        ]
        if clf_fig:
            clf_block.append(html.Div(className="chart-wrap", children=[
                dcc.Graph(id="clf-chart", figure=clf_fig,
                          config=dict(displayModeBar=False), style={"height": "320px"})]))
        metrics_children.append(html.Div(className="metrics-block", children=clf_block))

    sim_cards = simulation_metric_cards(simulation)
    if sim_cards:
        metrics_children.append(html.Div(className="metrics-block", children=[
            html.H3("Simulation snapshot", className="metrics-block__title"),
            html.P("Monte Carlo-style summaries exported with the simulation notebook.",
                   className="metrics-block__lead"),
            sim_cards,
        ]))

    # Team byline text
    team_str = "  ·  ".join(TEAM_MEMBERS)

    return html.Div(className="shell", children=[
        # ── Nav ───────────────────────────────────────────────────────────
        html.Nav(className="site-nav", children=[
            html.Div(className="site-nav__inner", children=[
                html.A("Digital Twin", href="#top", className="site-nav__brand"),
                html.Ul(className="site-nav__links", children=[
                    html.Li(html.A("About",      href="#about")),
                    html.Li(html.A("Pipeline",   href="#pipeline")),
                    html.Li(html.A("Metrics",    href="#metrics")),
                    html.Li(html.A("Brain 3D",   href="#brain-3d")),
                    html.Li(html.A("Demo twin",  href="#demo-simulation")),
                    html.Li(html.A("Gallery",    href="#gallery")),
                    html.Li(html.A("Notebooks",  href="#notebooks")),
                ]),
            ])
        ]),

        html.Main(children=[
            # ── Hero ──────────────────────────────────────────────────────
            html.Section(id="top", className="hero", children=[
                html.Div(className="hero__inner", children=[
                    html.P("Research prototype", className="hero__eyebrow"),
                    html.H1("Alzheimer's digital twin", className="hero__title"),
                    html.P(
                        "Longitudinal ADNI signals, sequence models, classification, "
                        "visual analytics, and simulation — presented with room to breathe.",
                        className="hero__subtitle",
                    ),
                    html.Div(className="hero__rule"),
                    # Team byline
                    html.P(className="hero__byline", children=[
                        html.Span("By  ", className="hero__byline-label"),
                        team_str,
                    ]),
                ])
            ]),

            # ── About ─────────────────────────────────────────────────────
            html.Section(id="about", className="section section--tight-top", children=[
                html.Div(className="section__inner", children=[
                    html.P("Context", className="section__label"),
                    html.H2("Clinical trajectories, modelled with restraint.", className="section__title"),
                    html.Div(className="grid-2", children=[
                        html.Div(className="prose", children=[
                            html.P("This repository explores a digital twin framing for Alzheimer's "
                                   "disease progression: structured visits, cognitive scores, imaging-derived "
                                   "volume, and genetics — aligned in time where possible."),
                            html.P("Recent work adds patient-level classification (three-class diagnosis and "
                                   "MCI-to-dementia conversion), a visualization dashboard, and Monte Carlo–style "
                                   "simulation exports alongside the original LSTM regression path."),
                        ]),
                        html.Div(className="prose", children=[
                            html.P("Data centre on ADNIMERGE-style tables. The LSTM stack uses aligned "
                                   "feature sequences; XGBoost models consume engineered patient snapshots; "
                                   "figures and JSON summaries land under results/ for this hub to surface."),
                        ]),
                    ]),
                ])
            ]),

            # ── Pipeline ──────────────────────────────────────────────────
            html.Section(id="pipeline", className="section", children=[
                html.Div(className="section__inner", children=[
                    html.P("Flow", className="section__label"),
                    html.H2("From cohort to twin scenarios.", className="section__title"),
                    html.P("Five beats that mirror the notebooks — kept visually light.",
                           className="section__lead"),
                    html.Div(className="pipeline", children=[
                        html.Div(className="pipeline__step", children=[
                            html.Div("01", className="pipeline__num"), html.H3("Curate"),
                            html.P("Load and filter longitudinal rows; harmonise diagnosis codes; "
                                   "document missingness."),
                        ]),
                        html.Div(className="pipeline__step", children=[
                            html.Div("02", className="pipeline__num"), html.H3("Encode"),
                            html.P("Sequence windows for the LSTM with scaled features matching "
                                   "the trained checkpoint."),
                        ]),
                        html.Div(className="pipeline__step", children=[
                            html.Div("03", className="pipeline__num"), html.H3("Classify"),
                            html.P("XGBoost pipelines for three-class diagnosis and MCI→Dementia "
                                   "conversion, with CV metrics exported to JSON."),
                        ]),
                        html.Div(className="pipeline__step", children=[
                            html.Div("04", className="pipeline__num"), html.H3("Visualize"),
                            html.P("Trajectory plots, subgroup views, SHAP, ROC, and consolidated "
                                   "dashboards written to results/visualizations and results/metrics."),
                        ]),
                        html.Div(className="pipeline__step", children=[
                            html.Div("05", className="pipeline__num"), html.H3("Simulate"),
                            html.P("What-if and Monte Carlo summaries — patient anchors, intervention "
                                   "deltas, and uncertainty bands — exported for review."),
                        ]),
                    ]),
                ])
            ]),

            # ── Metrics ───────────────────────────────────────────────────
            html.Section(id="metrics", className="section", children=[
                html.Div(className="section__inner", children=metrics_children)]),

            # ── 3D Brain Atrophy ──────────────────────────────────────────
            build_brain_section(),

            # ── Demo twin ─────────────────────────────────────────────────
            build_demo_simulation_section(),

            # ── Gallery ───────────────────────────────────────────────────
            html.Section(id="gallery", className="section", children=[
                html.Div(className="section__inner", children=[
                    html.P("Figures", className="section__label"),
                    html.H2("Static outputs from the analysis stack.", className="section__title"),
                    html.P(
                        "Click any image to open a full-screen view with an explanation of "
                        "what it shows and why it matters.",
                        className="section__lead",
                    ),
                    *(gallery_children if gallery_children else [
                        html.P("No figure files found under results/. Run the visualization "
                               "notebook to populate this gallery.", className="section__lead")
                    ]),
                ])
            ]),

            # ── Notebooks ─────────────────────────────────────────────────
            html.Section(id="notebooks", className="section", children=[
                html.Div(className="section__inner", children=[
                    html.P("Workspace", className="section__label"),
                    html.H2("Notebooks & artefacts.", className="section__title"),
                    html.P("Open these in JupyterLab from the repository root after installing requirements.",
                           className="section__lead"),
                    html.Div(className="notebook-list", children=[
                        html.Div(className="notebook-row", children=[
                            html.Span("data_exploration.ipynb",    className="notebook-row__name"),
                            html.Span("Cohort profile & features", className="notebook-row__desc")]),
                        html.Div(className="notebook-row", children=[
                            html.Span("02_lstm_model.ipynb",               className="notebook-row__name"),
                            html.Span("LSTM training & regression metrics",className="notebook-row__desc")]),
                        html.Div(className="notebook-row", children=[
                            html.Span("03_classification.ipynb",         className="notebook-row__name"),
                            html.Span("XGBoost diagnosis & conversion",  className="notebook-row__desc")]),
                        html.Div(className="notebook-row", children=[
                            html.Span("04_visualization.ipynb",             className="notebook-row__name"),
                            html.Span("Dashboards, trajectories, SHAP / ROC",className="notebook-row__desc")]),
                        html.Div(className="notebook-row", children=[
                            html.Span("05_simulation.ipynb",           className="notebook-row__name"),
                            html.Span("What-if & Monte Carlo summaries",className="notebook-row__desc")]),
                        html.Div(className="notebook-row", children=[
                            html.Span("results/metrics/demo_data.json",                  className="notebook-row__name"),
                            html.Span("Demo patients for LSTM + hub (written by 02_lstm_model)",className="notebook-row__desc")]),
                        html.Div(className="notebook-row", children=[
                            html.Span("data/raw/ADNIMERGE.csv",  className="notebook-row__name"),
                            html.Span("Source table (local only)",className="notebook-row__desc")]),
                    ]),
                ])
            ]),
        ]),

        # ── Footer ────────────────────────────────────────────────────────
        html.Footer(className="site-footer", children=[
            html.Div(className="site-footer__inner", children=[
                html.P("Alzheimer's digital twin — exploratory research software. Not a medical device."),
                html.P(f"Project by {', '.join(TEAM_MEMBERS)}.", className="site-footer__team"),
            ])
        ]),

        # ── Lightbox modal (JS-controlled) ────────────────────────────────
        build_lightbox(),
    ])


# ── App init ──────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?"
        "family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&"
        "family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&"
        "display=swap",
    ],
    title="Alzheimer's Digital Twin",
)
app.layout = build_layout()
server = app.server


# ── Custom CSS + JS injected via index_string ─────────────────────────────────
_CUSTOM_CSS = """
/* ── Brain 3D section ────────────────────────────────────────────── */
.brain-stats-row {
  display: flex;
  gap: 1.25rem;
  flex-wrap: wrap;
  margin: 1.25rem 0 1.5rem;
  align-items: stretch;
}
.brain-stat {
  background: #f5f5f8;
  border-radius: 8px;
  padding: 0.75rem 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 155px;
  flex: 1 1 155px;
}
.brain-stat__val {
  font-size: 1.2rem;
  font-weight: 600;
  color: #2f4f3f;
  font-family: 'DM Sans', sans-serif;
  white-space: nowrap;
}
.brain-stat__val--warn { color: #c0392b; }
.brain-stat__label {
  font-size: 0.72rem;
  color: #7a7a73;
  font-family: 'DM Sans', sans-serif;
  line-height: 1.4;
}
.brain-iframe-wrap {
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid #e0e0da;
  background: #0f1114;
}

/* ── Hero byline ─────────────────────────────────────────────────── */
.hero__byline {
  margin-top: 1.25rem;
  font-size: 0.8125rem;
  font-family: 'DM Sans', sans-serif;
  color: #7a7a73;
  letter-spacing: 0.02em;
  line-height: 1.8;
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 0.2rem;
}
.hero__byline-label {
  color: #8fa396;
  font-weight: 500;
  text-transform: uppercase;
  font-size: 0.6875rem;
  letter-spacing: 0.1em;
  margin-right: 0.3rem;
  flex-shrink: 0;
}

/* ── Footer team line ────────────────────────────────────────────── */
.site-footer__inner { display: flex; flex-direction: column; gap: 0.25rem; }
.site-footer__team  { font-size: 0.75rem; color: #9a9a92; }

/* ── Demo section alignment fixes ───────────────────────────────── */
.demo-toolbar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}
.demo-toolbar__label {
  font-size: 0.8125rem;
  font-weight: 500;
  color: #5a5a50;
  font-family: 'DM Sans', sans-serif;
  white-space: nowrap;
  flex-shrink: 0;
}
.demo-dropdown {
  flex: 1 1 220px;
  min-width: 220px;
}
.demo-charts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
  margin-top: 1rem;
  align-items: start;
}
.chart-wrap {
  background: #fff;
  border-radius: 10px;
  border: 1px solid #ebebе5;
  overflow: hidden;
  min-width: 0;
}
.chart-wrap--demo {
  background: #fafaf7;
  border: 1px solid #e8e8e2;
  border-radius: 10px;
  overflow: hidden;
  min-width: 0;
  padding-top: 0;
}
/* Ensure Dash graph containers don't overflow */
.chart-wrap .js-plotly-plot,
.chart-wrap--demo .js-plotly-plot {
  width: 100% !important;
}
.demo-meta {
  margin: 0.75rem 0 0;
  padding: 0.875rem 1rem;
  background: #f8f8f5;
  border-radius: 8px;
  border: 1px solid #ebebе5;
}
.demo-meta__dl {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 0.3rem 1.25rem;
  margin: 0;
}
.demo-meta__dt {
  font-size: 0.75rem;
  font-weight: 500;
  color: #7a7a73;
  font-family: 'DM Sans', sans-serif;
  white-space: nowrap;
}
.demo-meta__dd {
  font-size: 0.8125rem;
  color: #3c3c38;
  font-family: 'DM Sans', sans-serif;
  margin: 0;
  word-break: break-word;
}
.demo-meta__empty {
  font-size: 0.875rem;
  color: #7a7a73;
  font-family: 'DM Sans', sans-serif;
  margin: 0;
}
@media (max-width: 720px) {
  .demo-charts { grid-template-columns: 1fr; }
}

/* ── Gallery card — hover & click affordance ─────────────────────── */
.figure-card {
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: transform 0.28s cubic-bezier(0.34,1.56,0.64,1),
              box-shadow 0.28s ease;
  border-radius: 8px;
}
.figure-card:hover {
  transform: translateY(-6px) scale(1.012);
  box-shadow: 0 18px 44px rgba(47,79,63,0.16), 0 4px 10px rgba(0,0,0,0.07);
  z-index: 2;
}
.figure-card::after {
  content: 'Click to explore →';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg,
    rgba(47,79,63,0.93) 0%,
    rgba(143,163,150,0.88) 100%);
  color: rgba(255,255,255,0.97);
  display: grid;
  place-items: center;
  font-size: 0.75rem;
  font-family: 'DM Sans', sans-serif;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-weight: 500;
  opacity: 0;
  transition: opacity 0.22s ease;
  border-radius: inherit;
  pointer-events: none;
}
.figure-card:hover::after { opacity: 1; }
.figure-card figcaption   { transition: color 0.2s ease; }
.figure-card:hover figcaption { color: #2f4f3f; }

/* ── Lightbox overlay ────────────────────────────────────────────── */
.lb-overlay {
  position: fixed;
  inset: 0;
  z-index: 10000;
  background: rgba(6, 8, 7, 0.88);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.28s ease;
}
.lb-overlay.lb-active {
  opacity: 1;
  pointer-events: auto;
}

/* ── Lightbox card ───────────────────────────────────────────────── */
.lb-card {
  display: flex;
  width: min(1120px, 96vw);
  max-height: 90vh;
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 40px 90px rgba(0,0,0,0.65), 0 0 0 1px rgba(255,255,255,0.04);
  transform: scale(0.90) translateY(20px);
  opacity: 0;
  transition: transform 0.38s cubic-bezier(0.34,1.46,0.64,1),
              opacity 0.28s ease;
}
.lb-overlay.lb-active .lb-card {
  transform: scale(1) translateY(0);
  opacity: 1;
}

/* ── Image pane ──────────────────────────────────────────────────── */
.lb-img-pane {
  flex: 0 0 58%;
  background: #080b09;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.75rem;
  min-height: 400px;
  overflow: hidden;
}
.lb-img-pane img {
  max-width: 100%;
  max-height: 86vh;
  object-fit: contain;
  border-radius: 4px;
  display: block;
}

/* ── Info pane ───────────────────────────────────────────────────── */
.lb-info-pane {
  flex: 0 0 42%;
  background: #fafaf7;
  display: flex;
  flex-direction: column;
  padding: 2.25rem 2rem 1.5rem;
  overflow-y: auto;
  position: relative;
}
.lb-label {
  font-size: 0.6875rem;
  font-family: 'DM Sans', sans-serif;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: #8fa396;
  font-weight: 500;
  margin-bottom: 0.5rem;
}
.lb-title {
  font-size: 1.25rem;
  font-family: 'Fraunces', Georgia, serif;
  color: #121211;
  font-weight: 500;
  line-height: 1.35;
  margin: 0 2.5rem 1.25rem 0;
}
.lb-divider {
  width: 36px;
  height: 2px;
  background: #2f4f3f;
  border-radius: 2px;
  margin-bottom: 1.25rem;
  flex-shrink: 0;
}
.lb-explanation {
  font-size: 0.9375rem;
  font-family: 'DM Sans', sans-serif;
  color: #4a4a44;
  line-height: 1.78;
  flex: 1;
}
.lb-explanation p          { margin: 0 0 0.875rem; }
.lb-explanation p:last-child { margin-bottom: 0; }
.lb-explanation strong     { color: #2f4f3f; font-weight: 600; }
.lb-explanation em         { color: #5a5a50; }
.lb-explanation ul         { margin: 0.5rem 0 0.875rem; padding-left: 1.25rem; }
.lb-explanation li         { margin-bottom: 0.4rem; }
.lb-footer { margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #e6e6e0; }
.lb-scroll-hint {
  font-size: 0.75rem;
  color: #b0b0a8;
  font-family: 'DM Sans', sans-serif;
  margin: 0;
}

/* ── Close button ────────────────────────────────────────────────── */
.lb-close {
  position: absolute;
  top: 1.1rem;
  right: 1.1rem;
  width: 2rem;
  height: 2rem;
  border-radius: 50%;
  border: none;
  background: rgba(0,0,0,0.07);
  color: #3c3c38;
  font-size: 1.25rem;
  line-height: 1;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.15s ease, transform 0.15s ease;
  flex-shrink: 0;
}
.lb-close:hover { background: rgba(0,0,0,0.14); transform: scale(1.12); }

/* ── Responsive: stack panes on narrow screens ───────────────────── */
@media (max-width: 740px) {
  .lb-card { flex-direction: column; width: 96vw; }
  .lb-img-pane { flex: none; max-height: 38vh; padding: 1rem; }
  .lb-info-pane { flex: 1; min-height: 0; padding: 1.5rem 1.25rem; }
  .lb-title { font-size: 1.05rem; margin-right: 2rem; }
}
"""

_LIGHTBOX_JS = f"""
(function () {{
  /* ── Image info dictionary ── */
  {_JS_IMAGE_INFO_CONST}

  /* ── Open lightbox on figure-card click (event delegation) ── */
  document.body.addEventListener('click', function (e) {{
    var card = e.target.closest('.figure-card');
    if (!card) return;
    var img = card.querySelector('img');
    if (!img) return;

    /* Key = filename without extension */
    var src      = img.src || '';
    var filename = src.split('/').pop();
    var key      = filename.replace(/\\.png$/i, '');
    var info     = IMAGE_INFO[key] || {{
      title: (card.querySelector('figcaption') || {{}}).textContent || filename,
      html:  '<p>No additional description is available for this image.</p>'
    }};

    var overlay = document.getElementById('lightbox-overlay');
    if (!overlay) return;

    overlay.querySelector('#lb-main-img').src    = img.src;
    overlay.querySelector('#lb-title').textContent = info.title;
    overlay.querySelector('#lb-explanation').innerHTML = info.html;

    /* Scroll info pane to top */
    var infoPne = overlay.querySelector('.lb-info-pane');
    if (infoPne) infoPne.scrollTop = 0;

    overlay.classList.add('lb-active');
    document.body.style.overflow = 'hidden';
  }});

  /* ── Close on overlay backdrop or × button ── */
  document.addEventListener('click', function (e) {{
    var overlay = document.getElementById('lightbox-overlay');
    if (!overlay) return;
    if (e.target === overlay || e.target.closest('.lb-close')) {{
      overlay.classList.remove('lb-active');
      document.body.style.overflow = '';
      /* clear src after transition to avoid stale image flash */
      setTimeout(function () {{
        if (!overlay.classList.contains('lb-active')) {{
          overlay.querySelector('#lb-main-img').src = '';
        }}
      }}, 320);
    }}
  }});

  /* ── Close on Escape key ── */
  document.addEventListener('keydown', function (e) {{
    if (e.key === 'Escape') {{
      var overlay = document.getElementById('lightbox-overlay');
      if (overlay && overlay.classList.contains('lb-active')) {{
        overlay.classList.remove('lb-active');
        document.body.style.overflow = '';
      }}
    }}
  }});
}})();
"""

app.index_string = (
    """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
"""
    + _CUSTOM_CSS
    + """
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
    <script>
"""
    + _LIGHTBOX_JS
    + """
    </script>
  </body>
</html>"""
)


# ── Callbacks ─────────────────────────────────────────────────────────────────
if DEMO_SIM and DEMO_SIM.get("patients"):
    @app.callback(
        Output("demo-graph-mmse",   "figure"),
        Output("demo-graph-hippo",  "figure"),
        Output("demo-patient-meta", "children"),
        Input("demo-patient-select", "value"),
    )
    def _update_demo_patient(rid: str | None):
        r = str(rid) if rid else str(int(DEMO_SIM["demo_rids"][0]))
        return (
            demo_mmse_figure(r, DEMO_SIM),
            demo_hippo_figure(r, DEMO_SIM),
            demo_meta_children(r, DEMO_SIM),
        )

    # ── Brain 3D patient selector ─────────────────────────────────────────────
    BRAIN_STATS = {
        750:  {"hippo_bl": 5256, "hippo_end": 4836, "mmse_bl": 27, "mmse_end": 8,  "pct": 8.0},
        667:  {"hippo_bl": 8116, "hippo_end": 8038, "mmse_bl": 28, "mmse_end": 20, "pct": 1.0},
        1282: {"hippo_bl": 4601, "hippo_end": 4143, "mmse_bl": 26, "mmse_end": 23, "pct": 9.9},
        50:   {"hippo_bl": 4831, "hippo_end": 4556, "mmse_bl": 25, "mmse_end": 23, "pct": 5.7},
        128:  {"hippo_bl": 5703, "hippo_end": 4985, "mmse_bl": 29, "mmse_end": 26, "pct": 12.6},
    }

    @app.callback(
        Output("brain-iframe",     "src",      allow_duplicate=True),
        Output("brain-stats-row",  "children", allow_duplicate=True),
        Input("brain-patient-select", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def _update_brain_patient(rid):
        rid = int(rid) if rid else 750
        src_url = f"/hub-results/visualizations/brain_atrophy_3d_{rid}.html"

        s = BRAIN_STATS.get(rid, BRAIN_STATS[750])
        warn_cls = "brain-stat__val brain-stat__val--warn"
        stats = html.Div(className="brain-stats-row", children=[
            html.Div(className="brain-stat", children=[
                html.Span(f"{s['hippo_bl']:,} mm³", className="brain-stat__val"),
                html.Span("Hippocampus at baseline", className="brain-stat__label"),
            ]),
            html.Div(className="brain-stat brain-stat--arrow", children=[
                html.Span(f"↓ {s['pct']:.1f}%", className=warn_cls),
                html.Span("hippocampus volume loss", className="brain-stat__label"),
            ]),
            html.Div(className="brain-stat", children=[
                html.Span(f"{s['hippo_end']:,} mm³", className="brain-stat__val"),
                html.Span("Hippocampus latest visit", className="brain-stat__label"),
            ]),
            html.Div(className="brain-stat", children=[
                html.Span(f"{s['mmse_bl']} → {s['mmse_end']}", className=warn_cls),
                html.Span("MMSE score (out of 30)", className="brain-stat__label"),
            ]),
        ])
        return src_url, stats.children


# ── Static file serving ───────────────────────────────────────────────────────
@server.route("/hub-results/<path:subpath>")
def hub_results(subpath: str):
    """Serve PNGs from results/ (path-traversal safe)."""
    try:
        rel = Path(subpath)
        if rel.is_absolute() or ".." in rel.parts:
            abort(404)
        candidate = (RESULTS_ROOT / rel).resolve()
        candidate.relative_to(RESULTS_ROOT)
    except (ValueError, OSError):
        abort(404)
    if not candidate.is_file() or candidate.suffix.lower() not in (".png", ".html"):
        abort(404)
    return send_from_directory(str(candidate.parent), candidate.name)


if __name__ == "__main__":
    app.run(debug=True, port=8050)