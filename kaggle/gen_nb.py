"""Generator notebook Kaggle - chay file nay de tao kaggle_train_5levels.ipynb chuan JSON."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "kaggle_train_5levels.ipynb")

c0 = """# Du bao nhu cau ton kho da cap do - Train tren Kaggle (5 cap: L1+L2+L3+L9+L12)

**Cach dung tren Kaggle:**
1. Tao Notebook moi, bat GPU T4 x2.
2. Add input: m5-forecasting-accuracy (chua calendar.csv, sales_train_validation.csv, sell_prices.csv).
3. Copy toan bo notebook nay vao Kaggle Code, Run All.
4. Output o /kaggle/working/: hierarchical_5levels.csv, preds_L12.npz, PNG minh hoa.

**Chien luoc <6h:** train Global (TFT+LSTM) tren mau stratified L12 (~3000 chuoi), ARIMA chi fit tren aggregates (1+3+10+70 chuoi), sau do predict toan bo L12 va bottom-up sum len L9/L3/L2/L1.
"""

c1 = """import os, gc, copy, glob, pickle, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))

H, SEQ = 28, 56
START_DAY, END_DAY = 1500, 1913
TRAIN_END_DAY = 1885
TRAIN_SAMPLE_N = 3000
TFT_CFG = dict(d_model=16, num_heads=2, epochs=10, batch_size=256, lr=1e-3)
LSTM_CFG = dict(hidden_dim=32, num_layers=1, epochs=15, batch_size=512, lr=2e-3)
ARIMA_ORDER = (7, 1, 1)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE:', DEVICE)

KAGGLE_INPUTS = glob.glob('/kaggle/input/*/calendar.csv')
if len(KAGGLE_INPUTS) > 0:
    DATA_DIR = os.path.dirname(KAGGLE_INPUTS[0])
else:
    DATA_DIR = 'data/raw'
OUT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else 'kaggle_out'
os.makedirs(OUT_DIR, exist_ok=True)
print('DATA_DIR:', DATA_DIR)
print('OUT_DIR:', OUT_DIR)
print(os.listdir(DATA_DIR)[:10])
"""

c2 = """# UTILS: memory + metrics
def reduce_mem_usage(df, verbose=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        ct = df[col].dtypes
        if ct in numerics:
            cmin, cmax = df[col].min(), df[col].max()
            if str(ct)[:3] == 'int':
                if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    if verbose:
        end = df.memory_usage().sum() / 1024**2
        print(f'mem {start:.1f}->{end:.1f} MB')
    return df

def calc_rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))
def calc_mae(a, b):
    return float(np.mean(np.abs(a - b)))
def calc_wmape(a, b):
    t = np.sum(np.abs(a))
    return 0.0 if t == 0 else float(np.sum(np.abs(a - b)) / t * 100)
def calc_smape(a, b, eps=1e-5):
    d = (np.abs(a) + np.abs(b)) / 2.0 + eps
    return float(np.mean(np.abs(b - a) / d) * 100)
def evaluate_forecast(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.clip(np.asarray(y_pred).flatten(), 0, None)
    return {'RMSE': calc_rmse(y_true, y_pred), 'MAE': calc_mae(y_true, y_pred),
            'WMAPE (%)': calc_wmape(y_true, y_pred), 'sMAPE (%)': calc_smape(y_true, y_pred)}
print('utils OK')
"""

c3 = """# TFT COMPONENTS
class GLU(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.fc_val = nn.Linear(inp, out)
        self.fc_gate = nn.Linear(inp, out)
    def forward(self, x):
        return self.fc_val(x) * torch.sigmoid(self.fc_gate(x))

class GRN(nn.Module):
    def __init__(self, inp, hid, out, dropout=0.1, context_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(inp, hid)
        self.fc_context = nn.Linear(context_dim, hid, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hid, hid)
        self.drop = nn.Dropout(dropout)
        self.glu = GLU(hid, out)
        self.norm = nn.LayerNorm(out)
        self.skip = nn.Linear(inp, out) if inp != out else nn.Identity()
    def forward(self, x, context=None):
        h = self.fc1(x)
        if context is not None and self.fc_context is not None:
            ctx = context.unsqueeze(1) if h.dim() == 3 else context
            h = h + self.fc_context(ctx)
        h = self.drop(self.fc2(F.elu(h)))
        return self.norm(self.skip(x) + self.glu(h))

class VSN(nn.Module):
    def __init__(self, n_vars, inp, hid, dropout=0.1, context_dim=None):
        super().__init__()
        self.var_grns = nn.ModuleList([GRN(inp, hid, hid, dropout) for _ in range(n_vars)])
        self.weight_grn = GRN(n_vars * inp, hid, n_vars, dropout, context_dim)
    def forward(self, flat, indiv, context=None):
        w = F.softmax(self.weight_grn(flat, context), dim=-1)
        stacked = torch.stack([g(v) for g, v in zip(self.var_grns, indiv)], dim=-2)
        return torch.sum(stacked * w.unsqueeze(-1), dim=-2), w

class InterpretableMHA(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.h = heads
        self.dk = d_model // heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        qh = self.q(q).view(B, -1, self.h, self.dk).transpose(1, 2)
        kh = self.k(k).view(B, -1, self.h, self.dk).transpose(1, 2)
        vs = self.v(v)
        sc = torch.matmul(qh, kh.transpose(-2, -1)) / (self.dk ** 0.5)
        if mask is not None:
            sc = sc.masked_fill(mask == 0, -1e9)
        aw = self.drop(F.softmax(sc, dim=-1))
        mean_aw = aw.mean(dim=1)
        return self.o(torch.matmul(mean_aw, vs)), mean_aw

class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.q = quantiles
    def forward(self, pred, true):
        true = true.unsqueeze(-1)
        losses = []
        for i, q in enumerate(self.q):
            e = true - pred[..., i:i+1]
            losses.append(torch.max((q - 1) * e, q * e))
        return torch.mean(torch.cat(losses, dim=-1))
print('TFT components OK')
"""

c4 = """# GLOBAL MODELS: ARIMA + LSTM Global + TFT Global
from statsmodels.tsa.arima.model import ARIMA

class ARIMABaseline:
    def __init__(self, order=(7, 1, 1)):
        self.order = order
        self.f = None
    def fit(self, s):
        self.f = ARIMA(np.asarray(s), order=self.order).fit()
        return self
    def predict(self, horizon=28):
        return np.clip(np.asarray(self.f.forecast(steps=horizon)), 0, None)

class LSTMDataset(Dataset):
    def __init__(self, X, S, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.S = torch.tensor(S, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.S[i], self.Y[i]

class GlobalLSTMNet(nn.Module):
    def __init__(self, inp, hid=32, layers=1, H=28, cards=None, drop=0.2):
        super().__init__()
        self.H = H
        self.embs = nn.ModuleDict({k: nn.Embedding(c + 1, hid, padding_idx=0) for k, c in (cards or {}).items()}) if cards else None
        self.lstm = nn.LSTM(inp, hid, layers, batch_first=True, dropout=drop if layers > 1 else 0.0)
        self.fc = nn.Sequential(nn.Linear(hid * 2 if cards else hid, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, H))
    def forward(self, x, s=None):
        _, (hn, _) = self.lstm(x)
        h = hn[-1]
        if self.embs is not None and s is not None:
            e = sum(self.embs[k](s[k]) for k in self.embs)
            h = torch.cat([h, e], dim=-1)
        return self.fc(h)

class LSTMGlobalForecaster:
    def __init__(self, past_cols, static_cols, target='sales', seq=56, H=28, hid=32, layers=1, lr=2e-3, batch=512, epochs=15, device=None):
        self.past_cols = past_cols; self.static_cols = static_cols; self.target = target
        self.seq = seq; self.H = H; self.hid = hid; self.layers = layers
        self.lr = lr; self.batch = batch; self.epochs = epochs
        self.device = device or DEVICE
        self.scaler = StandardScaler(); self.tscaler = StandardScaler()
        self.maps = {}; self.cards = {}; self.model = None
    def _windows(self, df):
        f = self.scaler.transform(df[self.past_cols].fillna(0).values)
        t = self.tscaler.transform(df[[self.target]].fillna(0).values).flatten()
        L, H = self.seq, self.H
        OX, OY = [], []
        for s in range(len(df) - L - H + 1):
            OX.append(f[s:s + L]); OY.append(t[s + L:s + L + H])
        return np.array(OX), np.array(OY)
    def _static(self, df):
        return np.stack([df[c].map(self.maps[c]).fillna(0).astype(int).values for c in self.static_cols], axis=1)
    def fit(self, train_dfs):
        print(f'[LSTM] train {len(train_dfs)} series on {self.device}')
        all_df = pd.concat(train_dfs, ignore_index=True)
        for c in self.static_cols:
            u = sorted(all_df[c].astype(str).unique().tolist())
            self.maps[c] = {v: i + 1 for i, v in enumerate(u)}
            self.cards[c] = len(u)
        self.scaler.fit(all_df[self.past_cols].fillna(0).values)
        self.tscaler.fit(all_df[[self.target]].fillna(0).values)
        Xs, Ys, Ss = [], [], []
        for df in train_dfs:
            df = df.sort_values('d_num').reset_index(drop=True)
            X, Y = self._windows(df)
            Xs.append(X); Ys.append(Y)
            Ss.append(np.tile(self._static(df)[0:1], (len(X), 1)))
        Xa = np.concatenate(Xs); Ya = np.concatenate(Ys); Sa = np.concatenate(Ss)
        print('windows:', len(Xa))
        loader = DataLoader(LSTMDataset(Xa, Sa, Ya), batch_size=self.batch, shuffle=True,
                            num_workers=2, pin_memory=(self.device == 'cuda'))
        self.model = GlobalLSTMNet(len(self.past_cols), self.hid, self.layers, self.H, self.cards).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
        crit = nn.MSELoss()
        best = float('inf'); best_state = None
        use_amp = (self.device == 'cuda')
        scaler_amp = torch.cuda.amp.GradScaler() if use_amp else None
        for ep in range(1, self.epochs + 1):
            self.model.train(); tot = 0.0
            for x, s, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                sd = {c: s[:, i].to(self.device) for i, c in enumerate(self.static_cols)}
                opt.zero_grad()
                if scaler_amp:
                    with torch.cuda.amp.autocast():
                        pred = self.model(x, sd); loss = crit(pred, y)
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler_amp.step(opt); scaler_amp.update()
                else:
                    pred = self.model(x, sd); loss = crit(pred, y); loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                tot += loss.item() * len(x)
            tot /= len(Xa); sched.step(tot)
            if tot < best:
                best = tot; best_state = copy.deepcopy(self.model.state_dict())
            if ep % 5 == 0 or ep == self.epochs:
                print(f'[LSTM] ep {ep}/{self.epochs} loss={tot:.5f}')
        self.model.load_state_dict(best_state)
        print('[LSTM] done')
        return self
    def predict(self, last_seq, static_row):
        self.model.eval()
        f = self.scaler.transform(last_seq[self.past_cols].fillna(0).values)
        xt = torch.tensor(f, dtype=torch.float32).unsqueeze(0).to(self.device)
        sw = self._static(static_row)
        sd = {c: torch.tensor(sw[:, i], dtype=torch.long).to(self.device) for i, c in enumerate(self.static_cols)}
        with torch.no_grad():
            p = self.model(xt, sd).cpu().numpy().flatten()
        return np.clip(self.tscaler.inverse_transform(p.reshape(-1, 1)).flatten(), 0, None)

class TFTDataset(Dataset):
    def __init__(self, P, F, S, Y):
        self.P = torch.tensor(P, dtype=torch.float32)
        self.F = torch.tensor(F, dtype=torch.float32)
        self.S = torch.tensor(S, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.P)
    def __getitem__(self, i):
        return self.P[i], self.F[i], self.S[i], self.Y[i]

class TFTGlobalNet(nn.Module):
    def __init__(self, npast, nfut, cards, d=16, heads=2, drop=0.1, seq=56, H=28, qs=[0.1, 0.5, 0.9]):
        super().__init__()
        self.seq = seq; self.qs = qs
        self.ctx = nn.ModuleDict({k: nn.Embedding(c + 1, d, padding_idx=0) for k, c in cards.items()})
        self.pe = nn.ModuleList([nn.Linear(1, d) for _ in range(npast)])
        self.fe = nn.ModuleList([nn.Linear(1, d) for _ in range(nfut)])
        self.pvsn = VSN(npast, d, d, drop, d)
        self.fvsn = VSN(nfut, d, d, drop, d)
        self.enc = nn.LSTM(d, d, batch_first=True)
        self.dec = nn.LSTM(d, d, batch_first=True)
        self.glu = GLU(d, d); self.norm = nn.LayerNorm(d)
        self.enrich = GRN(d, d, d, drop, d)
        self.attn = InterpretableMHA(heads, d, drop)
        self.post = GRN(d, d, d, drop)
        self.head = nn.Linear(d, len(qs))
    def forward(self, px, fx, sx):
        se = sum(self.ctx[k](sx[k]) for k in self.ctx)
        pe = [e(px[:, :, i:i + 1]) for i, e in enumerate(self.pe)]
        pf = torch.cat(pe, -1)
        pv, pw = self.pvsn(pf, pe, se)
        fe = [e(fx[:, :, i:i + 1]) for i, e in enumerate(self.fe)]
        ff = torch.cat(fe, -1)
        fv, fw = self.fvsn(ff, fe, se)
        enc, (hn, cn) = self.enc(pv)
        dec, _ = self.dec(fv[:, self.seq:], (hn, cn))
        comb = torch.cat([enc, dec], 1)
        vsn = torch.cat([pv, fv[:, self.seq:]], 1)
        tf = self.norm(vsn + self.glu(comb))
        en = self.enrich(tf, se)
        L = en.size(1)
        mask = torch.tril(torch.ones(L, L, device=en.device))
        ao, _ = self.attn(en, en, en, mask)
        out = self.post(en + ao)
        return self.head(out[:, self.seq:]), {}

class TFTGlobalForecaster:
    def __init__(self, past_cols, fut_cols, static_cols, target='sales', seq=56, H=28, d=16, heads=2, lr=1e-3, batch=256, epochs=10, device=None):
        self.pc = past_cols; self.fc = fut_cols; self.sc = static_cols
        self.target = target; self.seq = seq; self.H = H
        self.d = d; self.heads = heads; self.lr = lr; self.batch = batch; self.epochs = epochs
        self.device = device or DEVICE
        self.ps = StandardScaler(); self.fs = StandardScaler(); self.ts = StandardScaler()
        self.maps = {}; self.cards = {}; self.model = None
    def _windows(self, df):
        p = self.ps.transform(df[self.pc].fillna(0).values)
        f = self.fs.transform(df[self.fc].fillna(0).values)
        t = self.ts.transform(df[[self.target]].fillna(0).values).flatten()
        L, H = self.seq, self.H
        P, Fw, T = [], [], []
        for s in range(len(df) - L - H + 1):
            P.append(p[s:s + L]); Fw.append(f[s:s + L + H]); T.append(t[s + L:s + L + H])
        return np.array(P), np.array(Fw), np.array(T)
    def _static(self, df):
        return np.stack([df[c].map(self.maps[c]).fillna(0).astype(int).values for c in self.sc], axis=1)
    def fit(self, train_dfs):
        print(f'[TFT] train {len(train_dfs)} series on {self.device}')
        all_df = pd.concat(train_dfs, ignore_index=True)
        for c in self.sc:
            u = sorted(all_df[c].astype(str).unique().tolist())
            self.maps[c] = {v: i + 1 for i, v in enumerate(u)}
            self.cards[c] = len(u)
        self.ps.fit(all_df[self.pc].fillna(0).values)
        self.fs.fit(all_df[self.fc].fillna(0).values)
        self.ts.fit(all_df[[self.target]].fillna(0).values)
        Ps, Fs, Ts, Ss = [], [], [], []
        for df in train_dfs:
            df = df.sort_values('d_num').reset_index(drop=True)
            P, Fw, T = self._windows(df)
            Ps.append(P); Fs.append(Fw); Ts.append(T)
            Ss.append(np.tile(self._static(df)[0:1], (len(P), 1)))
        Pa = np.concatenate(Ps); Fa = np.concatenate(Fs); Ta = np.concatenate(Ts); Sa = np.concatenate(Ss)
        print('windows:', len(Pa))
        loader = DataLoader(TFTDataset(Pa, Fa, Sa, Ta), batch_size=self.batch, shuffle=True,
                            num_workers=2, pin_memory=(self.device == 'cuda'))
        self.model = TFTGlobalNet(len(self.pc), len(self.fc), self.cards, self.d, self.heads, seq=self.seq, H=self.H).to(self.device)
        crit = QuantileLoss()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
        best = float('inf'); bs = None
        amp = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None
        for ep in range(1, self.epochs + 1):
            self.model.train(); tot = 0
            for px, fx, sx, y in loader:
                px, fx, y = px.to(self.device), fx.to(self.device), y.to(self.device)
                sd = {c: sx[:, i].to(self.device) for i, c in enumerate(self.sc)}
                opt.zero_grad()
                if amp:
                    with torch.cuda.amp.autocast():
                        qp, _ = self.model(px, fx, sd); loss = crit(qp, y)
                    amp.scale(loss).backward(); amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    amp.step(opt); amp.update()
                else:
                    qp, _ = self.model(px, fx, sd); loss = crit(qp, y); loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                tot += loss.item() * len(px)
            tot /= len(Pa); sched.step(tot)
            if tot < best:
                best = tot; bs = copy.deepcopy(self.model.state_dict())
            if ep % 5 == 0 or ep == self.epochs:
                print(f'[TFT] ep {ep}/{self.epochs} loss={tot:.5f}')
        self.model.load_state_dict(bs)
        print('[TFT] done')
        return self
    def predict_median(self, last_seq, fut_df):
        self.model.eval()
        full = pd.concat([last_seq, fut_df], ignore_index=True).sort_values('d_num').reset_index(drop=True)
        ps = self.ps.transform(full[self.pc].fillna(0).values)
        fs = self.fs.transform(full[self.fc].fillna(0).values)
        pt = torch.tensor(ps[:self.seq], dtype=torch.float32).unsqueeze(0).to(self.device)
        ft = torch.tensor(fs[:self.seq + self.H], dtype=torch.float32).unsqueeze(0).to(self.device)
        sw = self._static(last_seq.head(1))
        sd = {c: torch.tensor(sw[:, i], dtype=torch.long).to(self.device) for i, c in enumerate(self.sc)}
        with torch.no_grad():
            q, _ = self.model(pt, ft, sd)
            q = q.squeeze(0).cpu().numpy()
        med = self.ts.inverse_transform(q[:, 1:2]).flatten()
        return np.clip(med, 0, None)
print('models OK')
"""

c5 = """# PIPELINE 5 CAP: L1+L2+L3+L9+L12
cal = pd.read_csv(os.path.join(DATA_DIR, 'calendar.csv'))
cal['date'] = pd.to_datetime(cal['date'])
cal['is_weekend'] = cal['wday'].isin([1, 2]).astype(np.int8)
cal['sin_wday'] = np.sin(2 * np.pi * cal['wday'] / 7).astype(np.float32)
cal['cos_wday'] = np.cos(2 * np.pi * cal['wday'] / 7).astype(np.float32)
cal['sin_month'] = np.sin(2 * np.pi * cal['month'] / 12).astype(np.float32)
cal['cos_month'] = np.cos(2 * np.pi * cal['month'] / 12).astype(np.float32)
for c in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
    cal[c] = cal[c].fillna('None')
cal['has_event'] = (cal['event_name_1'] != 'None').astype(np.int8)
for t in ['Cultural', 'Religious', 'Sporting', 'National']:
    cal['evt_' + t] = (cal['event_type_1'] == t).astype(np.int8)
cal['event_count'] = ((cal['event_name_1'] != 'None').astype(int) + (cal['event_name_2'] != 'None').astype(int)).astype(np.int8)
is_evt = cal['has_event'].values
n = len(cal)
evt_idx = np.where(is_evt == 1)[0]
d2n = np.full(n, 999, dtype=np.int16)
dsl = np.full(n, 999, dtype=np.int16)
if len(evt_idx) > 0:
    nx = np.searchsorted(evt_idx, np.arange(n))
    v = nx < len(evt_idx)
    d2n[v] = (evt_idx[nx[v]] - np.arange(n)[v]).astype(np.int16)
    pv = np.searchsorted(evt_idx, np.arange(n), side='right') - 1
    vp = pv >= 0
    dsl[vp] = (np.arange(n)[vp] - evt_idx[pv[vp]]).astype(np.int16)
cal['days_to_next_event'] = np.clip(d2n, 0, 60).astype(np.int16)
cal['days_since_last_event'] = np.clip(dsl, 0, 60).astype(np.int16)
cal = reduce_mem_usage(cal)

train_end_wk = int(cal.set_index('d').loc['d_' + str(TRAIN_END_DAY), 'wm_yr_wk'])
prices = pd.read_csv(os.path.join(DATA_DIR, 'sell_prices.csv'))
prices = reduce_mem_usage(prices)
tp = prices[prices['wm_yr_wk'] <= train_end_wk]
stats = tp.groupby(['store_id', 'item_id'])['sell_price'].agg(price_max='max', price_min='min', price_mean='mean', price_std='std').reset_index()
prices = prices.merge(stats, on=['store_id', 'item_id'], how='left')
prices['price_discount'] = (1.0 - prices['sell_price'] / prices['price_max']).fillna(0).clip(0, 1).astype(np.float32)
prices['price_norm_mean'] = (prices['sell_price'] / prices['price_mean']).fillna(1.0).astype(np.float32)
prices['price_momentum'] = prices.groupby(['store_id', 'item_id'])['sell_price'].pct_change().fillna(0).astype(np.float32)
prices = reduce_mem_usage(prices)
print('train_end_wk:', train_end_wk, '| prices:', len(prices))

sales = pd.read_csv(os.path.join(DATA_DIR, 'sales_train_validation.csv'))
print('sales shape (wide):', sales.shape)
day_cols = ['d_' + str(i) for i in range(START_DAY, END_DAY + 1) if ('d_' + str(i)) in sales.columns]
id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
df = pd.melt(sales, id_vars=id_cols, value_vars=day_cols, var_name='d', value_name='sales')
del sales
gc.collect()
df = df.merge(cal, on='d', how='left').merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
df['is_listed'] = df['sell_price'].notna().astype(np.int8)
df['d_num'] = df['d'].str.split('_').str[1].astype(np.int16)
# PHUONG AN A: median fill CHI tren train (d_num <= TRAIN_END_DAY)
train_mask = df['d_num'] <= TRAIN_END_DAY
if train_mask.any():
    dept_med_map = df.loc[train_mask].groupby('dept_id')['sell_price'].median()
    glob_med = float(df.loc[train_mask, 'sell_price'].median())
else:
    dept_med_map = df.groupby('dept_id')['sell_price'].median()
    glob_med = float(df['sell_price'].median())
df['sell_price'] = df['sell_price'].fillna(df['dept_id'].map(dept_med_map)).fillna(glob_med).astype(np.float32)
for c, v in [('price_discount', 0), ('price_norm_mean', 1.0), ('price_momentum', 0)]:
    df[c] = df[c].fillna(v).astype(np.float32)
df['d_num'] = df['d'].str.split('_').str[1].astype(np.int16)
df['mean_price'] = df['sell_price']
df['avg_discount'] = df['price_discount']
snap_cols = [c for c in df.columns if c.startswith('snap_')]
def get_snap(r):
    key = 'snap_' + str(r['state_id'])
    return r[key] if key in df.columns else 0
df['snap_active'] = df.apply(get_snap, axis=1).astype(np.int8)
df = df.sort_values(['id', 'd_num']).reset_index(drop=True)
for lag in [28, 35, 42, 56]:
    df['sales_lag_' + str(lag)] = df.groupby('id')['sales'].shift(lag).astype(np.float32)
    # PHUONG AN A: gia chi dung ban lagged
    df['sell_price_lag_' + str(lag)] = df.groupby('id')['sell_price'].shift(lag).astype(np.float32)
    df['price_discount_lag_' + str(lag)] = df.groupby('id')['price_discount'].shift(lag).astype(np.float32)
    df['price_momentum_lag_' + str(lag)] = df.groupby('id')['price_momentum'].shift(lag).astype(np.float32)
df['sales_rolling_mean_7'] = df.groupby('id')['sales'].transform(lambda x: x.shift(28).rolling(7, min_periods=1).mean()).astype(np.float32)
df['sales_rolling_mean_28'] = df.groupby('id')['sales'].transform(lambda x: x.shift(28).rolling(28, min_periods=1).mean()).astype(np.float32)
df['sell_price_rolling_mean_7'] = df.groupby('id')['sell_price'].transform(lambda x: x.shift(28).rolling(7, min_periods=1).mean()).astype(np.float32)
before = len(df)
df = df.dropna(subset=['sales_lag_28']).reset_index(drop=True)
print(f'L12 rows: {before}->{len(df)} | ids: {df["id"].nunique()}')
sku_df = reduce_mem_usage(df)

def add_agg_lags(g, target):
    g = g.sort_values('d_num').reset_index(drop=True)
    for lag in [28, 35, 42, 56]:
        g['sales_lag_' + str(lag)] = g.groupby('keys')[target].shift(lag).astype(np.float32)
        g['mean_price_lag_' + str(lag)] = g.groupby('keys')['mean_price'].shift(lag).astype(np.float32)
        g['avg_discount_lag_' + str(lag)] = g.groupby('keys')['avg_discount'].shift(lag).astype(np.float32)
    g['sales_rolling_mean_7'] = g.groupby('keys')[target].transform(lambda x: x.shift(28).rolling(7, min_periods=1).mean()).astype(np.float32)
    g['sales_rolling_mean_28'] = g.groupby('keys')[target].transform(lambda x: x.shift(28).rolling(28, min_periods=1).mean()).astype(np.float32)
    g['mean_price_rolling_mean_7'] = g.groupby('keys')['mean_price'].transform(lambda x: x.shift(28).rolling(7, min_periods=1).mean()).astype(np.float32)
    return g.dropna(subset=['sales_lag_28']).reset_index(drop=True)

sku_df['_sd'] = sku_df['store_id'] + '|' + sku_df['dept_id']
base_cols = ['mean_price', 'avg_discount', 'is_weekend', 'sin_wday', 'cos_wday', 'sin_month', 'cos_month', 'snap_active', 'has_event', 'event_count', 'evt_Cultural', 'evt_Religious', 'evt_Sporting', 'evt_National', 'days_to_next_event', 'days_since_last_event']
agg_dict = {'sales': 'sum', 'sell_price': 'median', 'price_discount': 'mean'}
for bc in base_cols:
    if bc not in ('mean_price', 'avg_discount'):
        agg_dict[bc] = 'first'
L9 = sku_df.groupby(['_sd', 'store_id', 'dept_id', 'state_id', 'date', 'd_num']).agg(sales=('sales', 'sum'), mean_price=('sell_price', 'median'), avg_discount=('price_discount', 'mean'), is_weekend=('is_weekend', 'first'), sin_wday=('sin_wday', 'first'), cos_wday=('cos_wday', 'first'), sin_month=('sin_month', 'first'), cos_month=('cos_month', 'first'), snap_active=('snap_active', 'first'), has_event=('has_event', 'first'), event_count=('event_count', 'first'), evt_Cultural=('evt_Cultural', 'first'), evt_Religious=('evt_Religious', 'first'), evt_Sporting=('evt_Sporting', 'first'), evt_National=('evt_National', 'first'), days_to_next_event=('days_to_next_event', 'first'), days_since_last_event=('days_since_last_event', 'first')).reset_index().rename(columns={'_sd': 'keys', 'sales': 'target'})
L9 = add_agg_lags(L9, 'target')
L3 = sku_df.groupby(['store_id', 'state_id', 'date', 'd_num']).agg(sales=('sales', 'sum'), mean_price=('sell_price', 'median'), avg_discount=('price_discount', 'mean'), is_weekend=('is_weekend', 'first'), sin_wday=('sin_wday', 'first'), cos_wday=('cos_wday', 'first'), sin_month=('sin_month', 'first'), cos_month=('cos_month', 'first'), snap_active=('snap_active', 'first'), has_event=('has_event', 'first'), event_count=('event_count', 'first'), evt_Cultural=('evt_Cultural', 'first'), evt_Religious=('evt_Religious', 'first'), evt_Sporting=('evt_Sporting', 'first'), evt_National=('evt_National', 'first'), days_to_next_event=('days_to_next_event', 'first'), days_since_last_event=('days_since_last_event', 'first')).reset_index()
L3['keys'] = L3['store_id']
L3 = L3.rename(columns={'sales': 'target'})
L3 = add_agg_lags(L3, 'target')
L2 = sku_df.groupby(['state_id', 'date', 'd_num']).agg(sales=('sales', 'sum'), mean_price=('sell_price', 'median'), avg_discount=('price_discount', 'mean'), is_weekend=('is_weekend', 'first'), sin_wday=('sin_wday', 'first'), cos_wday=('cos_wday', 'first'), sin_month=('sin_month', 'first'), cos_month=('cos_month', 'first'), snap_active=('snap_active', 'first'), has_event=('has_event', 'first'), event_count=('event_count', 'first'), evt_Cultural=('evt_Cultural', 'first'), evt_Religious=('evt_Religious', 'first'), evt_Sporting=('evt_Sporting', 'first'), evt_National=('evt_National', 'first'), days_to_next_event=('days_to_next_event', 'first'), days_since_last_event=('days_since_last_event', 'first')).reset_index()
L2['keys'] = L2['state_id']
L2 = L2.rename(columns={'sales': 'target'})
L2 = add_agg_lags(L2, 'target')
L1 = sku_df.groupby(['date', 'd_num']).agg(sales=('sales', 'sum'), mean_price=('sell_price', 'median'), avg_discount=('price_discount', 'mean'), is_weekend=('is_weekend', 'first'), sin_wday=('sin_wday', 'first'), cos_wday=('cos_wday', 'first'), sin_month=('sin_month', 'first'), cos_month=('cos_month', 'first'), snap_active=('snap_active', 'first'), has_event=('has_event', 'first'), event_count=('event_count', 'first'), evt_Cultural=('evt_Cultural', 'first'), evt_Religious=('evt_Religious', 'first'), evt_Sporting=('evt_Sporting', 'first'), evt_National=('evt_National', 'first'), days_to_next_event=('days_to_next_event', 'first'), days_since_last_event=('days_since_last_event', 'first')).reset_index()
L1['keys'] = 'ALL'
L1 = L1.rename(columns={'sales': 'target'})
L1 = add_agg_lags(L1, 'target')
print(f"N series: L1={L1['keys'].nunique()} L2={L2['keys'].nunique()} L3={L3['keys'].nunique()} L9={L9['keys'].nunique()} L12={sku_df['id'].nunique()}")
"""

c6 = """# TRAIN
# PHUONG AN A: PAST gom sales lags + gia lagged; FUTURE chi lich/SNAP/su kien (khong gia)
PAST = ['sales', 'sales_lag_28', 'sales_lag_35', 'sales_rolling_mean_7', 'sales_rolling_mean_28', 'sell_price_lag_28', 'price_discount_lag_28', 'price_momentum_lag_28', 'sell_price_rolling_mean_7']
FUT = ['is_weekend', 'sin_wday', 'cos_wday', 'sin_month', 'cos_month', 'snap_active', 'has_event', 'event_count', 'evt_Cultural', 'evt_Religious', 'evt_Sporting', 'evt_National', 'days_to_next_event', 'days_since_last_event']
STATIC = ['store_id', 'state_id', 'cat_id', 'dept_id', 'item_id']

def split_df(g, key='d_num'):
    md = g[key].max()
    return g[g[key] <= md - 56].copy(), g[(g[key] > md - 56) & (g[key] <= md - 28)].copy(), g[g[key] > md - 28].copy()

all_ids = sku_df['id'].unique().tolist()
if TRAIN_SAMPLE_N is not None and len(all_ids) > TRAIN_SAMPLE_N:
    meta = sku_df[['id', 'cat_id', 'store_id']].drop_duplicates()
    sampled = []
    for _, grp in meta.groupby(['cat_id', 'store_id']):
        k = max(1, int(len(grp) * TRAIN_SAMPLE_N / len(meta)))
        sampled += grp.sample(n=min(k, len(grp)), random_state=SEED)['id'].tolist()
    sampled = sampled[:TRAIN_SAMPLE_N]
    print(f'sample train {len(sampled)}/{len(all_ids)}')
else:
    sampled = all_ids
    print(f'train full {len(all_ids)} series')

sampled_set = set(sampled)
train_dfs = []
sku_groups = {}
for sid, g in sku_df.groupby('id'):
    g = g.sort_values('d_num').reset_index(drop=True)
    tr, va, te = split_df(g)
    sku_groups[sid] = (tr, va, te)
    if sid in sampled_set:
        train_dfs.append(tr)

print('--- ARIMA aggregates ---')
agg_preds, agg_trues = {}, {}
for name, ADF, tcol in [('L1', L1, 'target'), ('L2', L2, 'target'), ('L3', L3, 'target'), ('L9', L9, 'target')]:
    for key, g in ADF.groupby('keys'):
        g = g.sort_values('d_num').reset_index(drop=True)
        tr, va, te = split_df(g)
        m = ARIMABaseline(order=ARIMA_ORDER).fit(pd.concat([tr, va])[tcol].values)
        agg_preds[(name, key)] = m.predict(H)
        agg_trues[(name, key)] = te[tcol].values[:H]
print('ARIMA done:', len(agg_preds))

print('--- LSTM Global ---')
lstm = LSTMGlobalForecaster(PAST, STATIC, target='sales', seq=SEQ, H=H, hid=LSTM_CFG['hidden_dim'], layers=LSTM_CFG['num_layers'], lr=LSTM_CFG['lr'], batch=LSTM_CFG['batch_size'], epochs=LSTM_CFG['epochs']).fit(train_dfs)

print('--- TFT Global ---')
tft = TFTGlobalForecaster(PAST, FUT, STATIC, target='sales', seq=SEQ, H=H, d=TFT_CFG['d_model'], heads=TFT_CFG['num_heads'], lr=TFT_CFG['lr'], batch=TFT_CFG['batch_size'], epochs=TFT_CFG['epochs']).fit(train_dfs)

print('--- predict L12 full ---')
lstm_L12, tft_L12, true_L12, meta_L12 = {}, {}, {}, {}
for sid, (tr, va, te) in sku_groups.items():
    y = te['sales'].values[:H]
    last = pd.concat([tr, va]).iloc[-SEQ:]
    true_L12[sid] = y
    meta_L12[sid] = (te['store_id'].iloc[0], te['state_id'].iloc[0], te['dept_id'].iloc[0])
    lstm_L12[sid] = lstm.predict(last, te.head(1))
    tft_L12[sid] = tft.predict_median(last, te)
print('predicted series:', len(true_L12))
"""

c7 = """# EVAL BOTTOM-UP 5 CAP + SAVE + PLOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def aggregate(ids, pred_dict, true_dict, idx_fn):
    groups = {}
    for sid in ids:
        g = idx_fn(sid)
        groups.setdefault(g, []).append(sid)
    yt, yp, labels = [], [], []
    for g, sids in groups.items():
        yt.append(np.sum([true_dict[s] for s in sids], axis=0))
        yp.append(np.sum([pred_dict[s] for s in sids], axis=0))
        labels.append(g)
    return yt, yp, labels

id2store = {k: v[0] for k, v in meta_L12.items()}
id2state = {k: v[1] for k, v in meta_L12.items()}
id2sd = {k: v[0] + '|' + v[2] for k, v in meta_L12.items()}
levels = [('L12', None), ('L9', id2sd), ('L3', id2store), ('L2', id2state), ('L1', None)]
rows = []
for model_name, pred in [('LSTM_GLOBAL', lstm_L12), ('TFT_GLOBAL', tft_L12)]:
    for lv, mp in levels:
        if lv == 'L12':
            ytl = [true_L12[k] for k in true_L12]
            ypl = [pred[k] for k in true_L12]
            n = len(ytl)
        elif lv == 'L1':
            ytl = [np.sum(list(true_L12.values()), axis=0)]
            ypl = [np.sum(list(pred.values()), axis=0)]
            n = 1
        else:
            ytl, ypl, lb = aggregate(list(true_L12.keys()), pred, true_L12, lambda s, m=mp: m[s])
            n = len(lb)
        ms = [evaluate_forecast(a, b) for a, b in zip(ytl, ypl)]
        rows.append({'Level': lv, 'Model': model_name, 'N_Series': n,
                     'RMSE': float(np.mean([m['RMSE'] for m in ms])),
                     'MAE': float(np.mean([m['MAE'] for m in ms])),
                     'WMAPE': float(np.mean([m['WMAPE (%)'] for m in ms])),
                     'MAPE': float(np.mean([m['sMAPE (%)'] for m in ms]))})
for (lv, key), p in agg_preds.items():
    m = evaluate_forecast(agg_trues[(lv, key)], p)
    rows.append({'Level': lv, 'Model': 'ARIMA', 'N_Series': 1, 'RMSE': m['RMSE'], 'MAE': m['MAE'], 'WMAPE': m['WMAPE (%)'], 'MAPE': m['sMAPE (%)']})
res = pd.DataFrame(rows)
ar = res[res.Model == 'ARIMA'].groupby('Level').agg(N_Series=('N_Series', 'sum'), RMSE=('RMSE', 'mean'), MAE=('MAE', 'mean'), WMAPE=('WMAPE', 'mean'), MAPE=('MAPE', 'mean')).reset_index()
ar['Model'] = 'ARIMA'
final = pd.concat([res[res.Model != 'ARIMA'], ar], ignore_index=True).sort_values(['Level', 'Model'])
print(final.to_string(index=False))
final.to_csv(os.path.join(OUT_DIR, 'hierarchical_5levels.csv'), index=False)
print('saved hierarchical_5levels.csv')

ex_sid = list(true_L12.keys())[0]
plt.figure(figsize=(10, 4))
plt.plot(true_L12[ex_sid], label='true', marker='o')
plt.plot(tft_L12[ex_sid], label='TFT P50', marker='x')
plt.plot(lstm_L12[ex_sid], label='LSTM', marker='s')
plt.title('L12 sample ' + str(ex_sid) + ' H=28')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'forecast_L12_sample.png'), dpi=130)
plt.close()
print('saved PNG DONE')
"""

cells = [c0, c1, c2, c3, c4, c5, c6, c7]
nb_cells = []
for i, src in enumerate(cells):
    if i == 0:
        nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
    else:
        nb_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "cells": nb_cells}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(("Wrote OK cells=" + str(len(nb_cells))).encode("ascii", "ignore").decode())
