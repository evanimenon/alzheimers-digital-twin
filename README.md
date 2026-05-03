# Alzheimer’s digital twin

Research codebase for longitudinal ADNI-style data: exploration notebooks, LSTM and XGBoost pipelines, simulation and visualization exports, and a small **project hub** (web UI) that surfaces committed metrics and figures.

## Project hub (website)

The hub is a [Dash](https://dash.plotly.com/) app in `app.py`. It reads **already generated** artefacts under `results/` (for example `metrics/demo_data.json`, `metrics/lstm_metrics.json`, PNGs in `results/visualizations/` and `results/metrics/`). It does not train models by itself.

### Run it locally

From the repository root:

```bash
cd alzheimers-digital-twin

# Optional: use a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install only what the hub needs (fast)
pip install "dash>=2.14" "plotly>=5.18"

# Or install the full project stack (slower; includes notebooks, PyTorch, etc.)
# pip install -r requirements.txt

python app.py
```

Then open **http://127.0.0.1:8050** in your browser.

To stop the server, press `Ctrl+C` in the terminal.

### Regenerating data the hub shows

- **Demo patient dropdown / charts:** run `02_lstm_model.ipynb`; it writes `results/metrics/demo_data.json` (and related plots).
- **Metrics and other figures:** use the other numbered notebooks (`03_classification`, `04_visualization`, `05_simulation`, etc.) as documented in those notebooks.

If a given JSON or PNG is missing, the corresponding section may show an empty state until that file exists in `results/`.

## Repository layout (high level)

| Path | Role |
|------|------|
| `app.py`, `assets/` | Dash project hub |
| `data_exploration.ipynb`, `02_*.ipynb` … | Analysis and model pipelines |
| `data/raw/` | Source tables (e.g. ADNIMERGE), when present locally |
| `results/metrics/`, `results/visualizations/` | Exported metrics and figures consumed by the hub |

---

*Exploratory research software—not a medical device.*
