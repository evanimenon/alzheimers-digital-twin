import json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

CSV_PATH  = 'data/raw/ADNIMERGE.csv'
CKPT_PATH = 'models/checkpoints/lstm_best.pt'
OUT_PATH  = 'results/metrics/demo_data.json'
DEMO_RIDS = [750, 667, 1282, 50, 128]
N_FUTURE  = 6
FEAT_COLS = ['MMSE', 'Hippocampus', 'APOE4', 'Education', 'visit_num']
MAX_LEN   = 8

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.3, bidirectional=True):
        super().__init__()
        D = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        self.attn = nn.Linear(hidden_size * D, 1)
        self.fc   = nn.Sequential(
            nn.Linear(hidden_size * D + 1, 64),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )
    def forward(self, x, mask, last_mmse):
        out, _  = self.lstm(x)
        scores  = self.attn(out).squeeze(-1)
        scores  = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        ctx     = (out * weights).sum(dim=1)
        last_n  = last_mmse if last_mmse.dim() == 2 else last_mmse.unsqueeze(-1)
        ctx     = torch.cat([ctx, last_n], dim=-1)
        return self.fc(ctx)

# Load data
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['MMSE', 'visit_num']).reset_index(drop=True)
for col in FEAT_COLS:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

scaler      = StandardScaler()
scaler.fit(df[FEAT_COLS].values.astype('float32'))
target_mean = float(df['MMSE'].mean())
target_std  = float(df['MMSE'].std())

# Load model
model = AttentionLSTM(input_size=len(FEAT_COLS)).cpu()
model.load_state_dict(torch.load(CKPT_PATH, map_location='cpu'))
model.eval()
print('Model loaded from ' + CKPT_PATH)

def predict_next(seq_raw):
    seq_s = scaler.transform(seq_raw.astype('float32'))
    T     = seq_s.shape[0]
    use   = min(T, MAX_LEN)
    pad   = MAX_LEN - use
    padded       = np.zeros((MAX_LEN, len(FEAT_COLS)), dtype='float32')
    mask         = np.zeros(MAX_LEN, dtype='float32')
    padded[pad:] = seq_s[-use:]
    mask[pad:]   = 1.0
    raw_last = float(seq_raw[-1, FEAT_COLS.index('MMSE')])
    last_n   = (raw_last - target_mean) / target_std
    x_t  = torch.tensor(padded).unsqueeze(0)
    m_t  = torch.tensor(mask).unsqueeze(0)
    lm_t = torch.tensor([[last_n]], dtype=torch.float32)
    with torch.no_grad():
        out = model(x_t, m_t, lm_t)
    return float(np.clip(out.item() * target_std + target_mean, 0, 30))

def predict_trajectory(context_raw, n_future=6):
    seq   = context_raw.copy().astype('float32')
    preds = []
    for _ in range(n_future):
        p = predict_next(seq)
        preds.append(round(p, 2))
        next_row = seq[-1].copy()
        next_row[FEAT_COLS.index('MMSE')]      = p
        next_row[FEAT_COLS.index('visit_num')] += 1
        seq = np.vstack([seq, next_row])
    return preds

# Build demo_data
demo_data = {}
for rid in DEMO_RIDS:
    pat = df[df['RID'] == rid].sort_values('visit_num').reset_index(drop=True)
    if len(pat) == 0:
        print('RID ' + str(rid) + ': not found, skipping')
        continue

    obs_mmse   = pat['MMSE'].values.tolist()
    obs_visits = pat['visit_num'].values.tolist()
    features   = pat[FEAT_COLS].values.tolist()
    apoe4      = float(pat['APOE4'].iloc[0]) if not pat['APOE4'].isna().all() else 0.0
    dx_last    = str(pat['DX'].iloc[-1]) if 'DX' in pat.columns else 'Unknown'
    last_visit = int(pat['visit_num'].iloc[-1])

    ctx       = pat[FEAT_COLS].values.astype('float32')
    pred_mmse = predict_trajectory(ctx, n_future=N_FUTURE)
    pred_visits = [last_visit + i + 1 for i in range(N_FUTURE)]

    pred_str = str([round(m, 1) for m in pred_mmse])
    print('RID ' + str(rid) + ': ' + str(len(obs_mmse)) + ' observed | pred=' + pred_str)

    demo_data[str(rid)] = {
        'rid':             rid,
        'apoe4':           apoe4,
        'dx_last':         dx_last,
        'observed_visits': [int(v) for v in obs_visits],
        'observed_mmse':   [round(m, 1) for m in obs_mmse],
        'pred_visits':     pred_visits,
        'pred_mmse':       pred_mmse,
        'features':        [[round(v, 1) for v in row] for row in features],
    }

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w') as f:
    json.dump(demo_data, f, indent=2)
print('Saved ' + OUT_PATH)
print('Restart app.py to see updated predictions.')
