#!/usr/bin/env python3
"""
Export results/metrics/demo_patients_simulation.json for the Dash hub.

If models/checkpoints/lstm_best.pt exists, uses the same AttentionLSTM + rollout
logic as 05_simulation.ipynb. Otherwise falls back to linear extrapolation from
observed visits (still matches visit indexing used in the notebooks).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "ADNIMERGE.csv"
OUT_PATH = ROOT / "results" / "metrics" / "demo_patients_simulation.json"
CKPT_PATH = ROOT / "models" / "checkpoints" / "lstm_best.pt"

DEMO_RIDS = [750, 667, 1282, 50, 128]
FEAT_COLS = ["MMSE", "Hippocampus", "APOE4", "Education", "visit_num"]
MAX_LEN = 8
N_FUTURE = 6


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["Hippocampus"] = df["Hippocampus"].fillna(df["Hippocampus"].mean())
    df["APOE4"] = df["APOE4"].fillna(df["APOE4"].median())
    df["Education"] = df["Education"].fillna(df["Education"].median())
    for col in FEAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df.dropna(subset=["MMSE", "visit_num"]).sort_values(["RID", "visit_num"]).reset_index(drop=True)


def linear_rollout(visits: np.ndarray, mmse: np.ndarray, hippo: np.ndarray, n_future: int) -> tuple[list[float], list[float], list[float]]:
    last_v = float(visits[-1])
    pred_visits = [last_v + i + 1 for i in range(n_future)]
    if len(visits) < 2:
        mmse_p = [float(mmse[-1])] * n_future
        hippo_p = [float(hippo[-1])] * n_future
        return pred_visits, mmse_p, hippo_p
    mv = np.polyfit(visits, mmse, 1)
    hv = np.polyfit(visits, hippo, 1)
    mmse_p = [float(np.clip(mv[0] * v + mv[1], 0, 30)) for v in pred_visits]
    hippo_p = [float(np.clip(hv[0] * v + hv[1], 500, 12000)) for v in pred_visits]
    return pred_visits, mmse_p, hippo_p


def lstm_rollout(df: pd.DataFrame) -> tuple[str, dict]:
    import torch
    import torch.nn as nn
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    class AttentionLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=True):
            super().__init__()
            self.bidirectional = bidirectional
            d = 2 if bidirectional else 1
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional
            )
            self.attn = nn.Linear(hidden_size * d, 1)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * d + 1, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, x, mask, last_mmse):
            out, _ = self.lstm(x)
            scores = self.attn(out).squeeze(-1)
            scores = scores.masked_fill(mask == 0, -1e9)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            ctx = (out * weights).sum(dim=1)
            last_n = last_mmse if last_mmse.dim() == 2 else last_mmse.unsqueeze(-1)
            ctx = torch.cat([ctx, last_n], dim=-1)
            return self.fc(ctx)

    device = torch.device("cpu")
    feat_scaler = StandardScaler()
    feat_scaler.fit(df[FEAT_COLS].values.astype("float32"))
    target_mean = float(df["MMSE"].mean())
    target_std = float(df["MMSE"].std())
    sub = df.dropna(subset=["MMSE", "Hippocampus"])
    hippo_lr = LinearRegression().fit(sub[["MMSE"]], sub["Hippocampus"])
    n_feat = len(FEAT_COLS)

    state = torch.load(CKPT_PATH, map_location=device)
    model = AttentionLSTM(input_size=n_feat, hidden_size=128, num_layers=2, bidirectional=True).to(device)
    model.load_state_dict(state)
    model.eval()

    def predict_next(seq_raw: np.ndarray) -> tuple[float, float]:
        seq_s = feat_scaler.transform(seq_raw.astype("float32"))
        t = seq_s.shape[0]
        use = min(t, MAX_LEN)
        pad = MAX_LEN - use
        padded = np.zeros((MAX_LEN, n_feat), dtype="float32")
        mask = np.zeros(MAX_LEN, dtype="float32")
        padded[pad:] = seq_s[-use:]
        mask[pad:] = 1.0
        raw_last_mmse = float(seq_raw[-1, FEAT_COLS.index("MMSE")])
        last_mmse_n = (raw_last_mmse - target_mean) / target_std
        x_t = torch.tensor(padded).unsqueeze(0)
        m_t = torch.tensor(mask).unsqueeze(0)
        lm_t = torch.tensor([[last_mmse_n]], dtype=torch.float32)
        with torch.no_grad():
            out = model(x_t, m_t, lm_t)
        mmse_p = float(np.clip(out.item() * target_std + target_mean, 0, 30))
        hippo_p = float(np.clip(hippo_lr.predict(np.array([[mmse_p]], dtype=np.float64))[0], 3000, 9000))
        return mmse_p, hippo_p

    def build_next_row(prev_raw: np.ndarray, new_mmse: float, new_hippo: float) -> np.ndarray:
        nr = prev_raw.copy()
        nr[FEAT_COLS.index("MMSE")] = new_mmse
        nr[FEAT_COLS.index("Hippocampus")] = new_hippo
        nr[FEAT_COLS.index("visit_num")] = prev_raw[FEAT_COLS.index("visit_num")] + 1
        return nr

    def predict_trajectory(context_raw: np.ndarray, n_future: int, mmse_scale: float = 1.0) -> tuple[list[float], list[float]]:
        seq = context_raw.copy().astype("float32")
        baseline_mmse = float(seq[0, FEAT_COLS.index("MMSE")])
        last_mmse = float(seq[-1, FEAT_COLS.index("MMSE")])
        mmse_preds, hippo_preds = [], []
        for _ in range(n_future):
            raw_mmse, raw_hippo = predict_next(seq)
            if mmse_scale != 1.0:
                decline = last_mmse - raw_mmse
                raw_mmse = float(np.clip(last_mmse - decline * mmse_scale, 0, 30))
                raw_hippo = float(np.clip(hippo_lr.predict(np.array([[raw_mmse]], dtype=np.float64))[0], 3000, 9000))
            mmse_preds.append(raw_mmse)
            hippo_preds.append(raw_hippo)
            last_mmse = raw_mmse
            next_row = build_next_row(seq[-1], raw_mmse, raw_hippo)
            seq = np.vstack([seq, next_row])
        return mmse_preds, hippo_preds

    patients: dict = {}
    for rid in DEMO_RIDS:
        pat = df[df["RID"] == rid].dropna(subset=["MMSE", "visit_num"]).sort_values("visit_num").reset_index(drop=True)
        if len(pat) < 2:
            continue
        ctx = pat[FEAT_COLS].values.astype(np.float32)
        visits = pat["visit_num"].values.astype(float)
        mmse = pat["MMSE"].values.astype(float)
        hippo = pat["Hippocampus"].values.astype(float)
        apoe = int(pat["APOE4"].dropna().iloc[0]) if len(pat["APOE4"].dropna()) > 0 else 0
        dx = pat["DX"].dropna().iloc[-1] if len(pat["DX"].dropna()) > 0 else ""
        pred_m, pred_h = predict_trajectory(ctx, n_future=N_FUTURE)
        last_v = float(visits[-1])
        pred_v = [last_v + i + 1 for i in range(N_FUTURE)]
        int_m, _ = predict_trajectory(ctx.copy(), n_future=N_FUTURE, mmse_scale=0.6)
        key = str(int(rid))
        patients[key] = {
            "rid": int(rid),
            "apoe4": apoe,
            "dx_last": str(dx),
            "n_obs_visits": int(len(visits)),
            "visits_obs": visits.tolist(),
            "mmse_obs": mmse.tolist(),
            "hippo_obs": hippo.tolist(),
            "pred_visits": pred_v,
            "mmse_pred": pred_m,
            "hippo_pred": pred_h,
            "mmse_pred_intervention_40pct": int_m,
        }
    return "lstm_rollout", patients


def linear_export(df: pd.DataFrame) -> tuple[str, dict]:
    patients: dict = {}
    for rid in DEMO_RIDS:
        pat = df[df["RID"] == rid].dropna(subset=["MMSE", "visit_num", "Hippocampus"]).sort_values("visit_num").reset_index(drop=True)
        if len(pat) < 2:
            pat = df[df["RID"] == rid].dropna(subset=["MMSE", "visit_num"]).sort_values("visit_num").reset_index(drop=True)
        if len(pat) < 1:
            continue
        visits = pat["visit_num"].values.astype(float)
        mmse = pat["MMSE"].values.astype(float)
        hippo = pat["Hippocampus"].values.astype(float) if pat["Hippocampus"].notna().any() else np.full_like(mmse, np.nan)
        if np.isnan(hippo).all():
            hippo = np.full_like(mmse, float(df["Hippocampus"].median()))
        apoe = int(pat["APOE4"].dropna().iloc[0]) if len(pat["APOE4"].dropna()) > 0 else 0
        dx = pat["DX"].dropna().iloc[-1] if len(pat["DX"].dropna()) > 0 else ""
        pred_v, pred_m, pred_h = linear_rollout(visits, mmse, hippo, N_FUTURE)
        key = str(int(rid))
        patients[key] = {
            "rid": int(rid),
            "apoe4": apoe,
            "dx_last": str(dx),
            "n_obs_visits": int(len(visits)),
            "visits_obs": visits.tolist(),
            "mmse_obs": mmse.tolist(),
            "hippo_obs": [float(x) for x in hippo],
            "pred_visits": pred_v,
            "mmse_pred": pred_m,
            "hippo_pred": pred_h,
        }
    return "linear_extrapolation", patients


def main() -> int:
    if not CSV_PATH.is_file():
        print(f"Missing {CSV_PATH}", file=sys.stderr)
        return 1
    df = load_frame()
    if CKPT_PATH.is_file():
        print("Using LSTM checkpoint for rollout…")
        method, patients = lstm_rollout(df)
    else:
        print("No checkpoint; using linear extrapolation (re-run after training to upgrade).")
        method, patients = linear_export(df)

    payload = {
        "version": 1,
        "prediction_method": method,
        "n_future_visits": N_FUTURE,
        "demo_rids": [patients[k]["rid"] for k in sorted(patients.keys(), key=int)],
        "patients": patients,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(patients)} patients, method={method})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
