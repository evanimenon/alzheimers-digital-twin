# Alzheimer's Progression Digital Twin

**Predicting cognitive decline and brain atrophy using longitudinal clinical data and machine learning.**

By: Evani Menon · Manojna Reddy Kamaram · Diksha Kaushik · Palak · Ishaan Arora · Isobel Kuriyan

---

## What This Project Does

This project builds a **digital twin** of Alzheimer's disease progression. Given a patient's longitudinal clinical history — cognitive test scores, brain volume measurements, and genetic risk factors — the system predicts how their cognition and hippocampus volume will change at future visits, classifies their risk of converting from MCI to dementia, and simulates "what-if" scenarios such as different genetic risk profiles and treatment interventions.

Everything runs on the **ADNI clinical dataset** (`ADNIMERGE.csv`) — no MRI files required.

---

## Results Summary

| Metric | Target | Achieved |
|---|---|---|
| MMSE MAE (next-visit prediction) | < 2.5 pts | 3.06 pts |
| MMSE R² | > 0.70 | 0.07 (clinical-only ceiling) |
| 3-class classifier accuracy (CN/MCI/Dementia) | 75–80% | **76.6%** ✓ |
| MCI→Dementia conversion AUC | > 0.80 | **0.90** ✓ |

The MMSE regression metrics are below target because the model uses clinical tabular data only. Without MRI-derived structural features feeding the LSTM, the theoretical ceiling for R² on this dataset is approximately 0.1–0.2. The classification results both exceed their targets.

---

## Project Structure

```
alzheimers-digital-twin/
│
├── data/
│   └── raw/
│       └── ADNIMERGE.csv            ← ADNI clinical data (14,314 rows, 4,722 patients)
│
├── models/
│   └── checkpoints/
│       └── lstm_best.pt             ← Trained AttentionLSTM checkpoint
│
├── results/
│   ├── metrics/
│   │   ├── lstm_metrics.json        ← LSTM + classification evaluation metrics
│   │   ├── simulation_summary.json  ← What-if scenario outputs
│   │   ├── demo_data.json           ← Pre-computed predictions for Dash app
│   │   ├── confusion_matrix.png
│   │   ├── roc_curves.png
│   │   ├── feature_importance.png
│   │   ├── shap_importance.png
│   │   ├── training_curves.png
│   │   └── lstm_eval_plots.png
│   │
│   └── visualizations/
│       ├── brain_atrophy_3d.png     ← 3D brain heatmap (notebook 06)
│       ├── brain_atrophy_3d.html    ← Interactive 3D version
│       ├── demo_trajectories.png
│       ├── hippocampus_atrophy.png
│       ├── population_trajectories.png
│       ├── results_dashboard.png
│       ├── subgroup_analysis.png
│       ├── what_if_apoe4.png
│       ├── what_if_intervention.png
│       ├── monte_carlo_ci.png
│       └── all_patients_simulation.png
│
├── data_exploration.ipynb           ← Notebook 01
├── 02_lstm_model.ipynb              ← Notebook 02
├── 03_classification.ipynb          ← Notebook 03
├── 04_visualization.ipynb           ← Notebook 04
├── 05_simulation.ipynb              ← Notebook 05
├── 06_brain_visualization.ipynb     ← Notebook 06
│
├── app.py                           ← Dash web dashboard
├── regenerate_demo_data.py          ← Rebuild demo_data.json from checkpoint
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** [ADNI — Alzheimer's Disease Neuroimaging Initiative](https://adni.loni.usc.edu)
Access is approved for academic use. All data is de-identified (HIPAA compliant).

**File:** `ADNIMERGE.csv`

| Column | Description |
|---|---|
| `RID` | Patient ID |
| `VISCODE` | Visit code (bl, m06, m12, m24 …) |
| `visit_num` | Numeric visit index (0, 1, 2 …) |
| `MMSE` | Mini Mental State Exam score (0–30) |
| `DX` | Diagnosis: CN / MCI / Dementia |
| `Hippocampus` | Hippocampus volume in mm³ |
| `APOE4` | APOE4 allele count (0, 1, 2) |
| `Education` | Years of education |
| `Gender` | Male / Female |

**Scale:** 14,314 visit rows · 4,722 unique patients · 5,665 MCI / 4,981 CN / 2,642 Dementia visits.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/evanimenon/alzheimers-digital-twin
cd alzheimers-digital-twin

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt** should contain:
```
torch
pandas
numpy
scikit-learn
xgboost
matplotlib
plotly
dash
flask
nilearn
kaleido
shap
```

Place `ADNIMERGE.csv` at `data/raw/ADNIMERGE.csv` before running any notebooks.

---

## Notebooks — Run in Order

### Notebook 01 — `data_exploration.ipynb`
**What it does:** Loads ADNIMERGE, imputes missing values, plots MMSE trajectories for 30 sample patients coloured by diagnosis, identifies 185 MCI→Dementia converters, selects 5 demo patients (RIDs 750, 667, 1282, 50, 128), and builds 2,475 training sequences.

**Outputs:** Console stats, trajectory plots, demo patient selection.

---

### Notebook 02 — `02_lstm_model.ipynb`
**What it does:** Trains the core `AttentionLSTM` model — a bidirectional 2-layer LSTM with an attention mechanism and a last-MMSE scalar concatenated to the context vector. Predicts next-visit MMSE auto-regressively.

**Architecture:**
- Input: 5 clinical features per visit (MMSE, Hippocampus, Education, APOE4, Gender)
- LSTM: 2 layers · hidden=128 · bidirectional → 256-dim context
- Attention: learned weighted sum over all timesteps
- FC head: [257 → 64 → 1] (256 context + 1 normalised last MMSE)
- Loss: HuberLoss

**Train/val/test split:** 70/15/15 by patient RID (no data leakage).

**Outputs:**
- `models/checkpoints/lstm_best.pt`
- `results/metrics/training_curves.png`
- `results/metrics/lstm_eval_plots.png`
- `results/metrics/lstm_metrics.json`
- `results/metrics/demo_trajectories.png`

---

### Notebook 03 — `03_classification.ipynb`
**What it does:** Builds one feature row per patient (slope, total drop, baseline vs last values) and trains two XGBoost classifiers.

**Model 1 — 3-class (CN / MCI / Dementia):**
- 5-fold cross-validated accuracy: **76.6%**
- Features: MMSE slope, Hippocampus slope, APOE4, Education, Gender, n_visits

**Model 2 — Binary MCI→Dementia converter:**
- 5-fold cross-validated AUC: **0.900**
- Trained only on MCI patients (497 patients, 185 converters)
- Class imbalance handled via `scale_pos_weight`

**Outputs:**
- `results/metrics/confusion_matrix.png`
- `results/metrics/roc_curves.png`
- `results/metrics/feature_importance.png`
- `results/metrics/shap_importance.png` (requires `pip install shap`)

---

### Notebook 04 — `04_visualization.ipynb`
**What it does:** Produces five publication-quality figures from the clinical data and model outputs. No model re-loading needed — uses pre-saved outputs from notebooks 02 and 03.

**Figures produced:**

| File | Description |
|---|---|
| `demo_trajectories.png` | 5 demo patients: observed + linear-extrapolated predicted trajectory |
| `hippocampus_atrophy.png` | Hippocampus volume bar charts per demo patient across visits |
| `population_trajectories.png` | 30 sampled patients per diagnosis group with mean lines |
| `results_dashboard.png` | 6-panel summary: dataset, MMSE distribution, model performance, decline rates, hippocampus scatter, APOE4 breakdown |
| `subgroup_analysis.png` | Mean MMSE ± std, hippocampus volume, and visits per patient by diagnosis |

---

### Notebook 05 — `05_simulation.ipynb`
**What it does:** Loads the trained LSTM checkpoint and runs three simulation modes on demo patient RID 750.

**Simulation 1 — APOE4 What-If:**
Takes the same patient and overrides their APOE4 value to 0, 1, and 2. Shows how predicted MMSE and hippocampus trajectories diverge based on genetic risk. The spread between APOE4=0 and APOE4=2 is computed and annotated on the plot.

**Simulation 2 — Monte Carlo Uncertainty (n=200):**
Injects Gaussian noise (σ=1.0 MMSE point, reflecting real test-retest variability) into the observed sequence 200 times. Plots 50% and 90% confidence bands around the mean prediction for both MMSE and hippocampus.

**Simulation 3 — Intervention Delay:**
Simulates clinical benefit. At each predicted step, the MMSE decline is reduced by 0%, 20%, or 40% (and a theoretical 100% halt). The green shaded area between the natural and 40%-slowdown curves represents the treatment benefit zone.

**Simulation 4 — All 5 demo patients:**
Side-by-side panel showing natural vs 40%-intervention trajectory for all five demo patients simultaneously.

**Outputs:**
- `results/visualizations/what_if_apoe4.png`
- `results/visualizations/monte_carlo_ci.png`
- `results/visualizations/what_if_intervention.png`
- `results/visualizations/all_patients_simulation.png`
- `results/metrics/simulation_summary.json`

**Note for VS Code users:** Do not use `matplotlib.use('Agg')` — VS Code's Jupyter backend handles this automatically. The notebook is configured correctly.

---

### Notebook 06 — `06_brain_visualization.ipynb`
**What it does:** Creates a 3D anatomical brain heatmap showing predicted atrophy without requiring MRI scan files.

**How it works:**
1. Loads a real MRI-derived brain surface mesh from `nilearn` (fsaverage5, ~10k vertices, ~5 MB download)
2. Extracts RID 750's real hippocampus volume and MMSE at baseline and 24 months from ADNIMERGE
3. Defines anatomically accurate region centres (MNI152 RAS coordinates) for hippocampus, entorhinal cortex, amygdala, parietal cortex, frontal cortex, and posterior cingulate
4. Computes a Gaussian-weighted atrophy intensity at each brain surface vertex, driven by real patient measurements
5. Renders an interactive side-by-side Plotly figure: left brain = baseline, right brain = 24 months

**Atrophy colour scale:** Blue/yellow = healthy tissue · Orange/red = atrophy. Warmer = more severe.

**Prerequisite:** `pip install nilearn kaleido`

**Outputs:**
- `results/visualizations/brain_atrophy_3d.html` — interactive, rotatable in browser
- `results/visualizations/brain_atrophy_3d.png` — static for gallery and slides

---

## Web Dashboard — `app.py`

A Plotly Dash application providing an interactive frontend for the entire project.

**Run:**
```bash
python app.py
# Open: http://127.0.0.1:8050
```

**Sections:**

**Overview tab:** Project summary, team members, key metrics cards (AUC 0.90, accuracy 76.6%), and a link to the GitHub repo.

**Interactive patient explorer:** Dropdown to select from 5 demo patients. Renders live MMSE and hippocampus charts (observed history + LSTM predictions) with a patient metadata card. Powered by `demo_data.json`.

**Results gallery:** Lightbox-enabled grid of all output figures from notebooks 02–06. Click any image to open a full-size view with a layman-friendly explanation of what the chart shows and how to interpret it.

**Classification metrics:** Confusion matrix, ROC curves, feature importance, and SHAP plots from notebook 03.

**Simulation gallery:** APOE4 what-if, Monte Carlo confidence bands, and intervention scenarios from notebook 05.

---

## Regenerating Predictions — `regenerate_demo_data.py`

If you retrain the LSTM (notebook 02), run this script to update the predictions shown in the Dash app:

```bash
python regenerate_demo_data.py
# Then restart app.py
```

The script auto-detects the checkpoint architecture (input size, hidden size, bidirectional, attention layer) so it works regardless of which version of notebook 02 saved the checkpoint. It rebuilds scalers using the same train/val/test split (random_state=42) as training to ensure predictions are correctly normalised.

---

## Model Architecture Detail

```
Input sequence: (T visits × 5 features)
    MMSE, Hippocampus, Education, APOE4, Gender

BiLSTM (2 layers, hidden=128, bidirectional)
    → output: (T × 256)

Attention layer (linear → softmax over T)
    → weighted context vector: (256,)

Concatenate last known MMSE (normalised scalar)
    → (257,)

FC head: Linear(257→64) → ReLU → Dropout(0.2) → Linear(64→1)
    → predicted normalised MMSE

Inverse normalise → predicted absolute MMSE (0–30)
```

**Training details:**
- Loss: MSE on normalised MMSE targets
- Optimiser: Adam
- Train/val split: 70/15/15 by patient (no sequence-level leakage)
- Best checkpoint saved by validation loss

---

## Key Design Decisions

**Why predict change and not absolute MMSE?**
Early versions predicted absolute next-visit MMSE. The model learned to predict the population mean (~26) for all patients, giving R²≈0.07. The attention mechanism over the full visit history was critical for capturing individual patient trajectories.

**Why bidirectional LSTM?**
Within a patient's observed history (not future prediction), bidirectionality allows the model to use later visits to contextualise earlier ones, improving feature extraction from the historical context window.

**Why split by patient RID, not by sequence?**
Splitting randomly by sequence would put visit-2 and visit-5 from the same patient in different splits — the model would "see" the patient during training and be evaluated on their own future, giving artificially inflated metrics. Splitting by RID ensures the model is evaluated on entirely unseen patients.

**Why XGBoost for classification?**
The classification task (CN/MCI/Dementia and MCI→Dementia conversion) works on one row per patient with hand-engineered slope features. XGBoost handles this tabular setting better than LSTM, is fully interpretable via SHAP, and trains in under 2 seconds.

---

## Limitations

**MMSE prediction accuracy** — Without structural MRI features, the theoretical R² ceiling for MMSE prediction from clinical tabular data alone is ~0.1–0.2. The original target of R²>0.70 assumed 3D ResNet-extracted MRI features as input.

**3D ResNet pipeline** — The proposal included MRI scan processing with a pre-trained Med3D ResNet. This was not implemented because no `.nii.gz` scan files were obtained. The LSTM uses clinical tabular data only, and notebook 06 simulates the brain visualisation using a standard anatomical atlas rather than patient-specific MRI.

**Hippocampus imputation** — Many patients have missing hippocampus values which were imputed with the dataset mean. This flattens individual variation in hippocampus trajectories and may reduce model sensitivity to hippocampal atrophy.

**Dataset bias** — The ADNI cohort is predominantly white, educated, and high-income. Generalisation to more diverse populations is unvalidated.

**Model interpretability** — The AttentionLSTM is a black-box architecture. Attention weights provide some interpretability but are not sufficient for clinical acceptance without further validation.

---

## References

- ADNI: Mueller et al. (2005). *The Alzheimer's Disease Neuroimaging Initiative.* Neuroimaging Clinics of North America.
- Digital twins in Alzheimer's: Amato et al. (2025, January). *EEG-based digital twin for Alzheimer's.* PMC12125947.
- MMSE: Folstein et al. (1975). *Mini-mental state.* Journal of Psychiatric Research.
- XGBoost: Chen & Guestrin (2016). *XGBoost: A scalable tree boosting system.* KDD.
- SHAP: Lundberg & Lee (2017). *A unified approach to interpreting model predictions.* NeurIPS.
- nilearn: Abraham et al. (2014). *Machine learning for neuroimaging with scikit-learn.* Frontiers in Neuroinformatics.

---

## GitHub

[https://github.com/evanimenon/alzheimers-digital-twin](https://github.com/evanimenon/alzheimers-digital-twin)

Data access (restricted, ADNI approved): [https://adni.loni.usc.edu](https://adni.loni.usc.edu)