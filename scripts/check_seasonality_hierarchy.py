"""
Script: scripts/check_seasonality_hierarchy.py
Muc dich: Kiem dinh mua vu + da tang theo 12 cap phan cap tren FULL M5.
  - Doc full M5 tu thu muc temp (m5full/) da tai ve, khong dung data/raw subset.
  - Tong hop 12 levels, chay battery test thong nhat, xuat bang + 9 hinh mau.
Output:
  - reports/seasonality_by_level.csv
  - reports/seasonality/*.png (3 levels x 3 loai plot)
"""
import os
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, f_oneway
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import STL

M5_DIR = r"C:\Users\PC\AppData\Local\Temp\opencode\m5full"
OUT_CSV = os.path.join("reports", "seasonality_by_level.csv")
FIG_DIR = os.path.join("reports", "seasonality")
os.makedirs("reports", exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_DAYS = 1913
DOW_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
rng = np.random.default_rng(42)


def load_calendar():
    cal = pd.read_csv(os.path.join(M5_DIR, "calendar.csv"))
    cal = cal.iloc[:N_DAYS].copy().reset_index(drop=True)
    cal['is_event'] = cal['event_name_1'].notna().astype(int)
    cal['snap_any'] = ((cal['snap_CA'] == 1) | (cal['snap_TX'] == 1) | (cal['snap_WI'] == 1)).astype(int)
    return cal


def build_level_accumulators(meta):
    """Khoi tao dict tich luy cho cac level co cardinality nho."""
    acc = {}
    # L1
    acc['L1_Total'] = {'__all__': np.zeros(N_DAYS, dtype=np.float64)}
    # helper lay danh sach keys
    for lvl, keys in [
        ('L2_State', meta['state_id'].unique().tolist()),
        ('L3_Store', meta['store_id'].unique().tolist()),
        ('L4_Cat', meta['cat_id'].unique().tolist()),
        ('L5_Dept', meta['dept_id'].unique().tolist()),
        ('L6_StateCat', (meta['state_id'] + '|' + meta['cat_id']).unique().tolist()),
        ('L7_StateDept', (meta['state_id'] + '|' + meta['dept_id']).unique().tolist()),
        ('L8_StoreCat', (meta['store_id'] + '|' + meta['cat_id']).unique().tolist()),
        ('L9_StoreDept', (meta['store_id'] + '|' + meta['dept_id']).unique().tolist()),
    ]:
        acc[lvl] = {k: np.zeros(N_DAYS, dtype=np.float64) for k in keys}
    # L10 item, L11 item-state: tich luy full (3049 / ~7709 keys)
    acc['L10_Item'] = {k: np.zeros(N_DAYS, dtype=np.float64) for k in meta['item_id'].unique()}
    is_keys = (meta['item_id'] + '|' + meta['state_id']).unique().tolist()
    acc['L11_ItemState'] = {k: np.zeros(N_DAYS, dtype=np.float64) for k in is_keys}
    return acc


def pick_l12_sample(meta, n=500):
    """Stratified theo cat x store (3x10=30 strata)."""
    sample_ids = []
    per, rem = divmod(n, 30)
    for (c, s), g in meta.groupby(['cat_id', 'store_id']):
        k = per + (1 if rem > 0 else 0)
        rem -= 1
        take = min(k, len(g))
        sample_ids.extend(g.sample(n=take, random_state=42)['id'].tolist())
    return sample_ids


def accumulate():
    meta = pd.read_csv(
        os.path.join(M5_DIR, "sales_train_validation.csv"),
        usecols=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
    print(f"[+] Meta: {len(meta)} series", flush=True)
    acc = build_level_accumulators(meta)
    l12_ids = pick_l12_sample(meta, n=500)
    l12_acc = {i: np.zeros(N_DAYS, dtype=np.float64) for i in l12_ids}
    l12_set = set(l12_ids)
    dcols = [f"d_{i}" for i in range(1, N_DAYS + 1)]
    usecols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'] + dcols
    n_done = 0
    for chunk in pd.read_csv(os.path.join(M5_DIR, "sales_train_validation.csv"),
                             usecols=usecols, chunksize=4000):
        chunk = chunk.reset_index(drop=True)
        M = chunk[dcols].values.astype(np.float64)
        # L1
        acc['L1_Total']['__all__'] += M.sum(axis=0)
        # dung groupby index de cong don
        df_idx = chunk[['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id', 'id']].copy()
        df_idx['__row__'] = np.arange(len(chunk))
        for lvl, key_fn in [
            ('L2_State', lambda d: d['state_id']),
            ('L3_Store', lambda d: d['store_id']),
            ('L4_Cat', lambda d: d['cat_id']),
            ('L5_Dept', lambda d: d['dept_id']),
        ]:
            for k, idx in df_idx.groupby(key_fn(df_idx)).groups.items():
                acc[lvl][k] += M[list(idx)].sum(axis=0)
        # cac level ghep khoa: group truc tiep
        for lvl, col in [('L6_StateCat', ['state_id', 'cat_id']),
                         ('L7_StateDept', ['state_id', 'dept_id']),
                         ('L8_StoreCat', ['store_id', 'cat_id']),
                         ('L9_StoreDept', ['store_id', 'dept_id'])]:
            grp = df_idx.groupby(col).groups
            for kk, idx in grp.items():
                key = '|'.join(kk) if isinstance(kk, tuple) else kk
                acc[lvl][key] += M[list(idx)].sum(axis=0)
        # L10 / L11
        for k, idx in df_idx.groupby('item_id').groups.items():
            acc['L10_Item'][k] += M[list(idx)].sum(axis=0)
        is_key = df_idx['item_id'] + '|' + df_idx['state_id']
        for k, idx in df_idx.groupby(is_key).groups.items():
            acc['L11_ItemState'][k] += M[list(idx)].sum(axis=0)
        # L12 sample
        sub = df_idx[df_idx['id'].isin(l12_set)]
        for _, r in sub.iterrows():
            l12_acc[r['id']] += M[int(r['__row__'])]
        n_done += len(chunk)
        print(f"  ... {n_done}/30490 rows", flush=True)
        del M, chunk, df_idx
        gc.collect()
    return acc, l12_acc, meta


def test_one(y, cal, do_yearly_acf=False):
    """Battery test cho 1 chuoi daily length 1913. Tra ve dict metrics."""
    out = {}
    out['mean'] = float(np.mean(y))
    out['zero_rate'] = float(np.mean(y == 0))
    wday = cal['weekday'].values
    month = cal['month'].values
    # dow spread
    try:
        means = [y[wday == d].mean() for d in DOW_ORDER]
        out['dow_spread_pct'] = float((max(means) - min(means)) / (np.mean(means) + 1e-9) * 100)
    except Exception:
        out['dow_spread_pct'] = np.nan
    # kruskal/anova
    try:
        groups = [y[wday == d] for d in DOW_ORDER]
        out['kw_dow_p'] = float(kruskal(*groups).pvalue)
        out['anova_dow_p'] = float(f_oneway(*groups).pvalue)
    except Exception:
        out['kw_dow_p'] = np.nan
        out['anova_dow_p'] = np.nan
    try:
        mg = [y[month == m] for m in sorted(np.unique(month))]
        out['kw_month_p'] = float(kruskal(*mg).pvalue)
    except Exception:
        out['kw_month_p'] = np.nan
    # ACF weekly
    try:
        a = acf(y, nlags=35, fft=True, missing='drop')
        out['acf7'] = float(a[7])
        out['acf14'] = float(a[14])
        out['acf28'] = float(a[28])
    except Exception:
        out['acf7'] = out['acf14'] = out['acf28'] = np.nan
    # ACF yearly (dat)
    if do_yearly_acf:
        try:
            a365 = acf(y, nlags=366, fft=True, missing='drop')
            out['acf365'] = float(a365[365])
        except Exception:
            out['acf365'] = np.nan
    else:
        out['acf365'] = np.nan
    # Ljung-Box tai lag 7
    try:
        lb = acorr_ljungbox(y, lags=[7], return_df=True)
        out['lb7_p'] = float(lb['lb_pvalue'].iloc[0])
    except Exception:
        out['lb7_p'] = np.nan
    # Periodogram: power tai ~7 so voi max
    try:
        f, Pxx = periodogram(y, detrend='constant')
        periods = 1 / f[1:]
        pw = Pxx[1:]
        j7 = int(np.argmin(np.abs(periods - 7)))
        out['per7_ratio'] = float(pw[j7] / (pw.max() + 1e-9))
        top = np.argsort(pw)[::-1][:3]
        out['is_top7'] = bool(any(abs(periods[i] - 7) < 0.6 for i in top))
    except Exception:
        out['per7_ratio'] = np.nan
        out['is_top7'] = False
    # STL weekly Fs
    try:
        stl = STL(pd.Series(y), period=7, robust=True).fit()
        S, R = stl.seasonal.values, stl.resid.values
        m = ~np.isnan(R)
        out['Fs7'] = float(max(0, 1 - np.var(R[m]) / (np.var(S[m] + R[m]) + 1e-9)))
    except Exception:
        out['Fs7'] = np.nan
    # Fourier R2 weekly K=3
    try:
        t = np.arange(len(y))
        X = np.column_stack([np.sin(2 * np.pi * k * t / 7) for k in range(1, 4)]
                            + [np.cos(2 * np.pi * k * t / 7) for k in range(1, 4)])
        X1 = np.column_stack([np.ones(len(t)), X])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        yh = X1 @ beta
        out['R2_fourier'] = float(1 - ((y - yh) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-9))
    except Exception:
        out['R2_fourier'] = np.nan
    # OLS weekday+event+snap R2
    try:
        X2 = pd.get_dummies(cal['weekday'], drop_first=True).astype(float).values
        X2 = np.column_stack([X2, cal[['is_event', 'snap_any']].values])
        X2 = np.column_stack([np.ones(len(y)), X2])
        b2, *_ = np.linalg.lstsq(X2, y, rcond=None)
        yh2 = X2 @ b2
        out['R2_full'] = float(1 - ((y - yh2) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-9))
    except Exception:
        out['R2_full'] = np.nan
    return out


def summarize(rows):
    df = pd.DataFrame(rows)
    # loc chuoi qua nhieu zero / constant
    df = df[(df['zero_rate'] < 0.5)]
    if len(df) == 0:
        return None, 0
    return {
        'median_Fs7': float(df['Fs7'].median()),
        'share_kw_dow_sig': float((df['kw_dow_p'] < 0.05).mean()),
        'share_acf7_strong': float((df['acf7'] > 0.2).mean()),
        'median_R2_fourier': float(df['R2_fourier'].median()),
        'median_R2_full': float(df['R2_full'].median()),
        'median_dow_spread': float(df['dow_spread_pct'].median()),
        'median_zero_rate': float(df['zero_rate'].median()),
        'median_acf7': float(df['acf7'].median()),
        'share_top7': float(df['is_top7'].mean()),
    }, len(df)


def plot_trio(y, cal, title_prefix, out_prefix):
    dow_mean = pd.DataFrame({'w': cal['weekday'].values, 'y': y}).groupby('w')['y'].mean().reindex(DOW_ORDER)
    # 1. bar
    plt.figure(figsize=(8, 4.5))
    colors = ['#e74c3c' if d in ('Saturday', 'Sunday') else '#3498db' for d in DOW_ORDER]
    plt.bar(DOW_ORDER, dow_mean.values, color=colors)
    plt.axhline(dow_mean.mean(), linestyle='--', color='gray')
    plt.title(f"{title_prefix}: mean theo thu")
    plt.ylabel("Don vi / ngay")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_prefix + "_bar.png", dpi=130)
    plt.close()
    # 2. line 28 ngay
    seg = y[100:128]
    plt.figure(figsize=(10, 4))
    plt.plot(range(28), seg, marker='o')
    for i in range(28):
        wd = cal['weekday'].values[100 + i]
        if wd in ('Saturday', 'Sunday'):
            plt.axvspan(i - 0.4, i + 0.4, color='orange', alpha=0.2)
    plt.title(f"{title_prefix}: 28 ngay lien tiep (cam=Sat/Sun)")
    plt.ylabel("Don vi / ngay")
    plt.tight_layout()
    plt.savefig(out_prefix + "_line.png", dpi=130)
    plt.close()
    # 3. ACF
    a = acf(y, nlags=35, fft=True, missing='drop')
    plt.figure(figsize=(10, 4))
    plt.stem(range(36), a)
    plt.axhline(1.96 / np.sqrt(len(y)), linestyle='--', color='gray')
    plt.axhline(-1.96 / np.sqrt(len(y)), linestyle='--', color='gray')
    plt.title(f"{title_prefix}: ACF (lag7={a[7]:.2f}, lag28={a[28]:.2f})")
    plt.xlabel("Lag (ngay)")
    plt.ylabel("ACF")
    plt.tight_layout()
    plt.savefig(out_prefix + "_acf.png", dpi=130)
    plt.close()


def main():
    cal = load_calendar()
    acc, l12_acc, meta = accumulate()
    # Chuan bi dict level -> list arrays (gioi han sample cho level nang)
    levels = {}
    for lvl in ['L1_Total', 'L2_State', 'L3_Store', 'L4_Cat', 'L5_Dept',
                'L6_StateCat', 'L7_StateDept', 'L8_StoreCat', 'L9_StoreDept']:
        levels[lvl] = list(acc[lvl].items())
    # L10/L11: sample 400
    l10_items = list(acc['L10_Item'].items())
    l10_sel = [l10_items[i] for i in rng.choice(len(l10_items), size=min(400, len(l10_items)), replace=False)]
    levels['L10_Item'] = l10_sel
    l11_items = list(acc['L11_ItemState'].items())
    l11_sel = [l11_items[i] for i in rng.choice(len(l11_items), size=min(400, len(l11_items)), replace=False)]
    levels['L11_ItemState'] = l11_sel
    levels['L12_ItemStore'] = list(l12_acc.items())  # 500
    n_level = {'L1_Total': 1, 'L2_State': 3, 'L3_Store': 10, 'L4_Cat': 3, 'L5_Dept': 7,
               'L6_StateCat': 9, 'L7_StateDept': 21, 'L8_StoreCat': 30, 'L9_StoreDept': 70,
               'L10_Item': 3049, 'L11_ItemState': int((meta['item_id'] + '|' + meta['state_id']).nunique()),
               'L12_ItemStore': 30490}
    rows_out = []
    # giu lai 3 chuoi mau de ve hinh
    y_total = acc['L1_Total']['__all__']
    y_store = acc['L3_Store'].get('CA_1', list(acc['L3_Store'].values())[0])
    # chon 1 SKU mau co Fs cao trong L12 (test nhanh Fs tren 30 chuoi dau)
    best_id, best_fs = None, -1
    cands = list(l12_acc.items())[:30]
    for k, y in cands:
        try:
            stl = STL(pd.Series(y), period=7, robust=True).fit()
            S, R = stl.seasonal.values, stl.resid.values
            m = ~np.isnan(R)
            fs = max(0, 1 - np.var(R[m]) / (np.var(S[m] + R[m]) + 1e-9))
            if fs > best_fs:
                best_fs, best_id = fs, k
        except Exception:
            pass
    y_sku = l12_acc[best_id] if best_id else cands[0][1]
    print(f"[+] SKU mau ve hinh: {best_id} Fs~{best_fs:.3f}", flush=True)
    for lvl, series in levels.items():
        # yearly acf chi cho level gon (card nho) de tiet kiem
        do_yearly = lvl in ('L1_Total', 'L2_State', 'L3_Store', 'L4_Cat', 'L5_Dept')
        tested = []
        for k, y in series:
            # bo chuoi constant
            if np.std(y) < 1e-9:
                continue
            tested.append(test_one(np.asarray(y, dtype=float), cal, do_yearly_acf=do_yearly))
        summ, n_tested = summarize(tested)
        if summ is None:
            continue
        rows_out.append({'level': lvl, 'n_series': n_level[lvl], 'n_tested': n_tested, **summ})
        print(f"[+] {lvl}: tested {n_tested}/{n_level[lvl]} medianFs={summ['median_Fs7']:.3f} "
              f"acf7={summ['median_acf7']:.3f} R2f={summ['median_R2_fourier']:.3f}", flush=True)
    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Da luu {OUT_CSV}", flush=True)
    # 9 hinh mau
    plot_trio(y_total, cal, "L1_Total", os.path.join(FIG_DIR, "L1_Total"))
    plot_trio(y_store, cal, "L3_Store_CA_1", os.path.join(FIG_DIR, "L3_Store_CA_1"))
    safe = str(best_id).replace('|', '_').replace('/', '_')
    plot_trio(y_sku, cal, f"L12_SKU_{safe}", os.path.join(FIG_DIR, f"L12_SKU_{safe}"))
    print(f"[OK] Da luu 9 hinh vao {FIG_DIR}", flush=True)


if __name__ == "__main__":
    main()
