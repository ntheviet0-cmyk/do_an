# %% [markdown]
# # TFT M5 - zero-inflation test (L1/L2/L3/L9/L12)
#
# Notebook test doc lap. Muc tieu:
# - Kiem tra pipeline TFT tren M5 voi du lieu that, tap trung vao 5 cap L1/L2/L3/L9/L12.
# - Xu ly zero inflation bang occurrence head, Tweedie NLL va dac trung lifecycle/zero-aware.
# - Dua context qua khu L1/L2/L3/L9 vao encoder ma khong broadcast xuong toan panel.
# - So sanh bottom-up (BU), calibration theo store, global va series-level.
# - Chon model tren validation, khong chon tren test.
# - Chay smoke test tren Kaggle T4 hoac local CPU voi subset nho.
#
# Luu y: WRMSSE-5 chi gom L1/L2/L3/L9/L12, khong phai WRMSSE 12 cap official.
# Scale va dollar weights van tuan theo nguyen tac M5 va chi dung history truoc forecast.

# %%
import os
import sys
import json
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from wrmsse5 import LEVELS, build_contexts, evaluate_wrmsse5
except ModuleNotFoundError:
    from kaggle.wrmsse5 import LEVELS, build_contexts, evaluate_wrmsse5

try:
    from hurdle_features import (
        HIER_CHANNELS,
        LOCAL_STATE_COLS,
        build_hierarchy_context,
        build_local_state_features,
        fit_feature_scaler,
        hierarchy_feature_names,
    )
except ModuleNotFoundError:
    from kaggle.hurdle_features import (
        HIER_CHANNELS,
        LOCAL_STATE_COLS,
        build_hierarchy_context,
        build_local_state_features,
        fit_feature_scaler,
        hierarchy_feature_names,
    )

import matplotlib
if not Path("/kaggle").exists():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROFILE = os.environ.get("PROFILE", "full_5level").lower()
SMOKE = os.environ.get("SMOKE", "0" if PROFILE == "full_5level" else "1") == "1"
N_SERIES = int(os.environ.get("N_SERIES", 120 if SMOKE else 3000))
EPOCHS = int(os.environ.get("EPOCHS", 2 if SMOKE else 12))
MAX_TRAIN_ORIGINS = int(os.environ.get("MAX_TRAIN_ORIGINS", 120 if SMOKE else 240))
MAX_TRAIN_ORIGINS = None if MAX_TRAIN_ORIGINS <= 0 else MAX_TRAIN_ORIGINS
BATCH = int(os.environ.get("BATCH", 128 if SMOKE else 512))
SEQ = 56
H = 28
D_MODEL = int(os.environ.get("D_MODEL", 32 if SMOKE else 64))
HEADS = int(os.environ.get("HEADS", 2 if SMOKE else 4))
DROPOUT = 0.10
LR = float(os.environ.get("LR", 1e-3))
WEIGHT_DECAY = 1e-4
TWEEDIE_P = 1.3
LOSS = os.environ.get("LOSS", "tweedie")
SERIES_WEIGHT_POWER = float(os.environ.get("SERIES_WEIGHT_POWER", "0.5"))
SERIES_CAL_K = float(os.environ.get("SERIES_CAL_K", "5.0"))
HIERARCHICAL_LOSS_WEIGHT = float(os.environ.get("HIERARCHICAL_LOSS_WEIGHT", "0.2"))
ZERO_LOSS_WEIGHT = float(os.environ.get("ZERO_LOSS_WEIGHT", "0.1"))
CALIBRATION_DAYS = int(os.environ.get("CALIBRATION_DAYS", "14"))
BEST_METRIC = os.environ.get("BEST_METRIC", "wrmsse5").lower()
FORECAST_SPLIT = (
    "evaluation"
    if PROFILE == "full_5level"
    else os.environ.get("FORECAST_SPLIT", "validation").lower()
)
if PROFILE == "5000_l12":
    SMOKE = False
    N_SERIES = int(os.environ.get("N_SERIES", "5000"))
    EPOCHS = int(os.environ.get("EPOCHS", "10"))
    MAX_TRAIN_ORIGINS = int(os.environ.get("MAX_TRAIN_ORIGINS", "160"))
    BATCH = int(os.environ.get("BATCH", "256"))
    D_MODEL = int(os.environ.get("D_MODEL", "64"))
    HEADS = int(os.environ.get("HEADS", "4"))
    SERIES_WEIGHT_POWER = float(os.environ.get("SERIES_WEIGHT_POWER", "0.5"))
    BEST_METRIC = os.environ.get("BEST_METRIC", "wrmsse5").lower()
elif PROFILE == "full_5level":
    SMOKE = False
    N_SERIES = 30490
    EPOCHS = int(os.environ.get("FULL_EPOCHS", "20"))
    MAX_TRAIN_ORIGINS = int(os.environ.get("MAX_TRAIN_ORIGINS", "160"))
    BATCH = int(os.environ.get("BATCH", "512"))
    D_MODEL = int(os.environ.get("D_MODEL", "64"))
    HEADS = int(os.environ.get("HEADS", "4"))
    BEST_METRIC = os.environ.get("BEST_METRIC", "wrmsse5").lower()
assert BEST_METRIC in {"l12", "wrmsse5"}
assert FORECAST_SPLIT in {"validation", "evaluation"}
assert 0.0 <= HIERARCHICAL_LOSS_WEIGHT <= 1.0
assert ZERO_LOSS_WEIGHT >= 0.0
assert 0 < CALIBRATION_DAYS < H
SEED = 42

if FORECAST_SPLIT == "evaluation":
    TRAIN_START, TRAIN_END = 57, 1885
    VAL_START, VAL_END = 1886, 1913
    TEST_START, TEST_END = 1914, 1941
    SALES_FILE = "sales_train_evaluation.csv"
else:
    TRAIN_START, TRAIN_END = 57, 1857
    VAL_START, VAL_END = 1858, 1885
    TEST_START, TEST_END = 1886, 1913
    SALES_FILE = "sales_train_validation.csv"
VAL_ORIGIN = VAL_START - 2
TEST_ORIGIN = TEST_START - 2
TRAIN_LAST_ORIGIN = TRAIN_END - H - 1

assert H == 28
assert TRAIN_LAST_ORIGIN >= SEQ - 1


def safe_str(value):
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
NUM_WORKERS = 0 if os.name == "nt" else (2 if Path("/kaggle").exists() else 0)

if os.environ.get("OUT_DIR"):
    OUT_DIR = Path(os.environ["OUT_DIR"])
elif Path("/kaggle/working").exists():
    OUT_DIR = Path("/kaggle/working")
else:
    OUT_DIR = Path("test_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE_DESC = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

print("device:", DEVICE, "|", DEVICE_DESC)
print("smoke:", SMOKE, "| series:", N_SERIES, "| epochs:", EPOCHS)
print("max_train_origins:", MAX_TRAIN_ORIGINS, "| batch:", BATCH)
print(
    "profile:", PROFILE or "custom",
    "| series_weight_power:", SERIES_WEIGHT_POWER,
    "| series_cal_k:", SERIES_CAL_K,
    "| best_metric:", BEST_METRIC,
    "| hierarchical_loss_weight:", HIERARCHICAL_LOSS_WEIGHT,
    "| zero_loss_weight:", ZERO_LOSS_WEIGHT,
    "| calibration_days:", CALIBRATION_DAYS,
    "| forecast_split:", FORECAST_SPLIT,
)
print("output:", safe_str(OUT_DIR.resolve()))

# %% [markdown]
# Mac dinh notebook chay full 30.490 series, evaluation split, toi da 20 epoch.
# Chay nhanh: dat `PROFILE=smoke SMOKE=1` (120 series, 2 epoch).
# Chay 5.000 series tap trung L12 tren Kaggle T4: dat `PROFILE=5000_l12`
# (them `%env PROFILE=5000_l12` truoc cell config, hoac sua dong `PROFILE = ...`).
# Tuong duong: `SMOKE=0 N_SERIES=5000 EPOCHS=10 BATCH=256 D_MODEL=64 HEADS=4 MAX_TRAIN_ORIGINS=160 SERIES_WEIGHT_POWER=0.5 BEST_METRIC=l12`
# Doi so epoch full bang `FULL_EPOCHS`; profile full luon khoa N_SERIES=30.490.
# Co the doi `LOSS=mse` de so sanh voi Tweedie, hoac `MAX_TRAIN_ORIGINS=0` de dung toan bo origin.
# Can dat `wrmsse5.py` cung thu muc khi chay source tren Kaggle.

# %% [markdown]
# ## 1. Load M5
#
# Notebook tim data theo thu tu: `M5_DATA_ROOT`, Kaggle input, `../data/raw`, `data/raw`.
# Local hien co `sales_train_validation.csv` (d_1-d_1913), nen test dung d_1886-d_1913.
# Luu y: ban local trong repo co the chi la sample 100 series (2 stores, 1 dept),
# vi vay smoke local chi de kiem tra pipeline; Kaggle full moi co 30.490 series.
# Tren Kaggle co the doi sang `sales_train_evaluation.csv` va test d_1914-d_1941.

# %%
def find_data_root():
    candidates = []
    if os.environ.get("M5_DATA_ROOT"):
        configured = Path(os.environ["M5_DATA_ROOT"]).expanduser()
        candidates.append(configured.parent if configured.is_file() else configured)
    candidates += [
        Path("/kaggle/input/m5-forecasting-accuracy"),
        Path("../data/raw"),
        Path("data/raw"),
        Path("../../data/raw"),
    ]
    for path in candidates:
        if (path / SALES_FILE).exists():
            return path.resolve()

    # Kaggle dataset slugs and uploaded dataset layouts can add extra folders.
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = sorted(kaggle_input.rglob(SALES_FILE))
        complete = [
            match.parent
            for match in matches
            if (match.parent / "calendar.csv").exists()
            and (match.parent / "sell_prices.csv").exists()
        ]
        if complete:
            return complete[0].resolve()

        attached = sorted(path.name for path in kaggle_input.iterdir())
        attached_text = ", ".join(attached) if attached else "(khong co dataset nao)"
    else:
        attached_text = "/kaggle/input khong ton tai"
    raise FileNotFoundError(
        f"Khong tim thay {SALES_FILE} kem calendar.csv va sell_prices.csv. "
        "Tren Kaggle, chon Add Input > M5 Forecasting Accuracy, sau do Run All lai. "
        f"Hoac dat M5_DATA_ROOT toi thu muc chua ba file. Input hien co: {attached_text}"
    )


DATA_ROOT = find_data_root()
print("data_root:", safe_str(DATA_ROOT))


def select_series(sales, n_series, seed=42):
    if n_series >= len(sales):
        return sales.reset_index(drop=True)

    stores = sorted(sales["store_id"].unique())
    per_store = max(1, int(math.ceil(n_series / len(stores))))
    parts = []

    for store in stores:
        g = sales[sales["store_id"].eq(store)].copy()
        first = g.sort_values(["dept_id", "item_id"]).groupby("dept_id", sort=True).head(1)
        rest = g.drop(first.index).sample(frac=1.0, random_state=seed)
        picked = pd.concat([first, rest]).drop_duplicates("id").head(per_store)
        parts.append(picked)

    out = pd.concat(parts).drop_duplicates("id")
    if len(out) < n_series:
        rest = sales[~sales["id"].isin(out["id"])]
        if len(rest):
            out = pd.concat([out, rest.sample(n=min(n_series - len(out), len(rest)), random_state=seed)])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


sales_all = pd.read_csv(DATA_ROOT / SALES_FILE, dtype={"id": str})
meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
day_cols = sorted([c for c in sales_all.columns if c.startswith("d_")], key=lambda x: int(x[2:]))
n_days = len(day_cols)

assert n_days >= TEST_END, f"Can it nhat {TEST_END} ngay, dang co {n_days}"
if PROFILE == "full_5level" and len(sales_all) != 30490:
    raise ValueError(
        f"PROFILE=full_5level can dung 30.490 series, nhung {SALES_FILE} chi co "
        f"{len(sales_all):,}. Kiem tra lai M5_DATA_ROOT va dataset Kaggle."
    )
sales_sel = select_series(sales_all, N_SERIES, SEED)
series_ids = sales_sel["id"].tolist()
n_series = len(series_ids)
if PROFILE == "full_5level" and n_series != 30490:
    raise ValueError(
        f"Full profile phai chon 30.490 series, hien chi chon {n_series:,}."
    )

print("series selected:", n_series)
print("days:", n_days)
print(sales_sel.groupby(["store_id", "state_id"]).size().to_string())

# %% [markdown]
# ## 2. Build matrices: sales, causal past price, calendar

# %%
SALES_DF = sales_sel.set_index("id").loc[series_ids, day_cols]
SALES_DF.columns = np.arange(1, n_days + 1)
SALES_DF = SALES_DF.astype(np.float32).fillna(0.0)
SALES = SALES_DF.to_numpy(dtype=np.float32)

cal = pd.read_csv(DATA_ROOT / "calendar.csv")
cal["d_num"] = cal["d"].str[2:].astype(np.int16)
cal["event_name_any"] = cal["event_name_1"].fillna(cal["event_name_2"])
cal["event_type"] = cal["event_type_1"].fillna(cal["event_type_2"]).fillna("None")
cal["has_event"] = cal["event_name_any"].notna().astype(np.float32)
cal["event_type_code"] = cal["event_type"].astype("category").cat.codes.astype(np.float32) + 1.0
cal = cal.sort_values("d_num").set_index("d_num")
cal = cal.reindex(np.arange(1, n_days + 1)).reset_index()
cal["wday"] = cal["wday"].fillna(0).astype(np.float32)
cal["month"] = cal["month"].fillna(0).astype(np.float32)
cal["year"] = cal["year"].fillna(0).astype(np.float32)
cal["wm_yr_wk"] = cal["wm_yr_wk"].fillna(0).astype(np.int64)
cal["snap_CA"] = cal["snap_CA"].fillna(0).astype(np.float32)
cal["snap_TX"] = cal["snap_TX"].fillna(0).astype(np.float32)
cal["snap_WI"] = cal["snap_WI"].fillna(0).astype(np.float32)
cal["has_event"] = cal["has_event"].fillna(0).astype(np.float32)
cal["event_type_code"] = cal["event_type_code"].fillna(0).astype(np.float32)

CAL = {
    "is_weekend": cal["wday"].isin([1, 2]).to_numpy(np.float32),
    "sin_wday": np.sin(2.0 * np.pi * cal["wday"].to_numpy(np.float32) / 7.0).astype(np.float32),
    "cos_wday": np.cos(2.0 * np.pi * cal["wday"].to_numpy(np.float32) / 7.0).astype(np.float32),
    "sin_month": np.sin(2.0 * np.pi * cal["month"].to_numpy(np.float32) / 12.0).astype(np.float32),
    "cos_month": np.cos(2.0 * np.pi * cal["month"].to_numpy(np.float32) / 12.0).astype(np.float32),
    "sin_year": np.sin(2.0 * np.pi * cal["d_num"].to_numpy(np.float32) / 365.25).astype(np.float32),
    "cos_year": np.cos(2.0 * np.pi * cal["d_num"].to_numpy(np.float32) / 365.25).astype(np.float32),
    "has_event": cal["has_event"].to_numpy(np.float32),
    "event_type_code": cal["event_type_code"].to_numpy(np.float32),
}

SNAP = np.zeros((n_series, n_days), dtype=np.float32)
for state, col in [("CA", "snap_CA"), ("TX", "snap_TX"), ("WI", "snap_WI")]:
    mask = sales_sel["state_id"].eq(state).to_numpy()
    if mask.any():
        SNAP[mask] = cal[col].to_numpy(np.float32)[None, :]

prices = pd.read_csv(
    DATA_ROOT / "sell_prices.csv",
    usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
)
pairs = sales_sel[["store_id", "item_id"]].drop_duplicates()
prices = prices.merge(pairs, on=["store_id", "item_id"], how="inner")

# Expand weekly prices directly to days. Avoiding a 59M-row melt/merge keeps the
# full 30,490-series profile within Kaggle RAM.
price_weekly = prices.pivot_table(
    index=["store_id", "item_id"],
    columns="wm_yr_wk",
    values="sell_price",
    aggfunc="last",
)
pair_order = sales_sel.set_index("id").loc[series_ids, ["store_id", "item_id"]]
pair_index = pd.MultiIndex.from_frame(pair_order)
price_weekly = price_weekly.reindex(pair_index)
week_positions = pd.Index(price_weekly.columns).get_indexer(cal["wm_yr_wk"].to_numpy())
price_values = np.full((n_series, n_days), np.nan, dtype=np.float32)
valid_week = week_positions >= 0
price_values[:, valid_week] = price_weekly.to_numpy(np.float32)[:, week_positions[valid_week]]
PRICE_RAW = pd.DataFrame(price_values, index=series_ids, columns=np.arange(1, n_days + 1))
PRICE_OBS = PRICE_RAW.to_numpy(np.float32)
PRICE_AVAILABLE = np.isfinite(PRICE_OBS)

# Price is unknown in the forecast horizon. Past prices are filled causally;
# pre-listing days stay at zero and are identified by is_listed.
PRICE_DF = PRICE_RAW.ffill(axis=1).fillna(0.0)
PRICE = PRICE_DF.to_numpy(np.float32)

price_median_28 = (
    PRICE_DF.shift(1, axis=1).T.rolling(28, min_periods=1).median().T
)
PRICE_REL = (PRICE_DF / price_median_28.replace(0.0, np.nan))
PRICE_REL = PRICE_REL.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 5.0)
PRICE_REL = PRICE_REL.to_numpy(np.float32)
PROMO = (PRICE_REL < 0.95).astype(np.float32)

print("sales matrix:", SALES.shape)
print("price matrix:", PRICE.shape)
print("past price availability rate:", float(PRICE_AVAILABLE.mean()))
print("zero rate:", float((SALES == 0).mean()))
print("promo rate:", float(PROMO.mean()))

del prices, pairs, price_weekly, price_values, pair_order, pair_index, cal, sales_all

# %% [markdown]
# ## 3. Feature engineering
#
# Past features duoc tinh tai forecast origin. Future chi gom calendar/SNAP/event biet truoc.
# Gia va promotion tuong lai khong duoc dua vao decoder.
# Cac lag 1/7/14/28/35/56/364 deu an toan voi direct multi-horizon vi chi lay tu qua khu.

# %%
PAST_COLS = [
    "sales", "lag_1", "lag_7", "lag_14", "lag_28", "lag_35", "lag_56", "lag_364",
    "roll_mean_7", "roll_mean_28", "roll_mean_56", "roll_std_28",
    "sell_price", "price_rel", "promo", "is_listed", "snap", "has_event",
    "sin_wday", "cos_wday", "sin_month", "cos_month",
]
FUT_COLS = [
    "is_weekend", "sin_wday", "cos_wday", "sin_month", "cos_month",
    "sin_year", "cos_year", "snap", "has_event", "event_type_code",
]
FEATURE_COLS = list(dict.fromkeys(PAST_COLS + FUT_COLS))

FEAT = np.zeros((n_series, n_days, len(FEATURE_COLS)), dtype=np.float32)


def store_feature(name, values):
    FEAT[:, :, FEATURE_COLS.index(name)] = values


# Write one feature at a time instead of retaining ~9 GB of intermediate maps.
S = SALES_DF
store_feature("sales", SALES)
for k in [1, 7, 14, 28, 35, 56, 364]:
    store_feature(f"lag_{k}", S.shift(k, axis=1).fillna(0.0).to_numpy(np.float32))

for window in [7, 28, 56]:
    store_feature(
        f"roll_mean_{window}",
        S.T.rolling(window, min_periods=1).mean().T.to_numpy(np.float32),
    )
store_feature(
    "roll_std_28",
    S.T.rolling(28, min_periods=2).std().T.fillna(0.0).to_numpy(np.float32),
)

store_feature("sell_price", PRICE)
store_feature("price_rel", PRICE_REL)
store_feature("promo", PROMO)
store_feature("is_listed", PRICE_AVAILABLE.astype(np.float32))
store_feature("snap", SNAP)
for name in [
    "is_weekend", "has_event", "event_type_code", "sin_wday", "cos_wday",
    "sin_month", "cos_month", "sin_year", "cos_year",
]:
    store_feature(name, CAL[name][None, :])

PAST_IDX = [FEATURE_COLS.index(c) for c in PAST_COLS]
FUT_IDX = [FEATURE_COLS.index(c) for c in FUT_COLS]

STATIC_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
STATIC_DF = sales_sel.set_index("id").loc[series_ids, STATIC_COLS]
STATIC = np.zeros((n_series, len(STATIC_COLS)), dtype=np.int64)
STATIC_CARD = []
for j, col in enumerate(STATIC_COLS):
    codes, uniques = pd.factorize(STATIC_DF[col].astype(str), sort=True)
    STATIC[:, j] = codes + 1
    STATIC_CARD.append(len(uniques) + 1)


def dense_codes(codes):
    uniques = np.unique(codes)
    mapping = {value: i for i, value in enumerate(uniques)}
    return np.array([mapping[v] for v in codes], dtype=np.int64)


store_code = STATIC[:, STATIC_COLS.index("store_id")]
dept_code = STATIC[:, STATIC_COLS.index("dept_id")]
state_code = STATIC[:, STATIC_COLS.index("state_id")]
GROUPS = {
    "L1": np.zeros(n_series, dtype=np.int64),
    "L2": dense_codes(state_code),
    "L3": dense_codes(store_code),
    "L9": dense_codes(store_code * 1000 + dept_code),
    "L12": np.arange(n_series, dtype=np.int64),
}
if PROFILE == "full_5level":
    actual_group_counts = {level: int(GROUPS[level].max()) + 1 for level in LEVELS}
    expected_group_counts = {"L1": 1, "L2": 3, "L3": 10, "L9": 70, "L12": 30490}
    if actual_group_counts != expected_group_counts:
        raise ValueError(
            f"Hierarchy full khong dung: {actual_group_counts}; can {expected_group_counts}"
        )
HIERARCHY_GROUP_TENSORS = {
    level: torch.as_tensor(GROUPS[level], dtype=torch.long, device=DEVICE)
    for level in ("L1", "L2", "L3", "L9")
}

HIER_LEVELS = ("L1", "L2", "L3", "L9")
LOCAL_STATE = build_local_state_features(SALES, PRICE_AVAILABLE)
HIER_CONTEXT = build_hierarchy_context(SALES, PRICE_AVAILABLE, GROUPS, HIER_LEVELS)
HIER_CONTEXT_COLS = hierarchy_feature_names(HIER_LEVELS)
MODEL_PAST_COLS = PAST_COLS + list(LOCAL_STATE_COLS) + HIER_CONTEXT_COLS

VAL_EVAL_CONTEXTS = build_contexts(
    GROUPS,
    SALES[:, :VAL_START - 1],
    PRICE_OBS[:, :VAL_START - 1],
)
TEST_EVAL_CONTEXTS = build_contexts(
    GROUPS,
    SALES[:, :TEST_START - 1],
    PRICE_OBS[:, :TEST_START - 1],
)
CAL_SELECT_EVAL_CONTEXTS = build_contexts(
    GROUPS,
    SALES[:, :VAL_START - 1 + CALIBRATION_DAYS],
    PRICE_OBS[:, :VAL_START - 1 + CALIBRATION_DAYS],
)

del S
del SALES_DF, PRICE_DF, PRICE_RAW, price_median_28

print(
    "features:", FEAT.shape,
    "| past base/local/hierarchy:", len(PAST_COLS), len(LOCAL_STATE_COLS), len(HIER_CONTEXT_COLS),
    "| future:", len(FUT_COLS),
)
print("static cards:", STATIC_CARD)
print({k: int(v.max()) + 1 for k, v in GROUPS.items()})

# %% [markdown]
# ## 4. Split, scaler and window dataset

# %%
train_rev_28 = (
    SALES[:, TRAIN_END - 28:TRAIN_END].astype(np.float64)
    * np.nan_to_num(PRICE_OBS[:, TRAIN_END - 28:TRAIN_END], nan=0.0).astype(np.float64)
).sum(axis=1)
train_rev = train_rev_28
if SERIES_WEIGHT_POWER > 0 and train_rev.sum() > 0:
    series_weight = np.power(np.maximum(train_rev, 0.0), SERIES_WEIGHT_POWER)
    series_weight = series_weight / max(float(series_weight.mean()), 1e-9)
    series_weight = np.clip(series_weight, 0.25, 4.0)
else:
    series_weight = np.ones(n_series, dtype=np.float64)
series_weight = series_weight.astype(np.float32)
del PRICE, PRICE_REL, PROMO, SNAP, PRICE_AVAILABLE, PRICE_OBS
print(
    "series weight:",
    f"min={series_weight.min():.3f}",
    f"median={np.median(series_weight):.3f}",
    f"max={series_weight.max():.3f}",
)

train_feat = FEAT[:, :TRAIN_END, :]
FEAT_MEAN = train_feat.mean(axis=(0, 1)).astype(np.float32)
FEAT_STD = train_feat.std(axis=(0, 1)).astype(np.float32)
FEAT_STD[FEAT_STD < 1e-6] = 1.0
FEAT -= FEAT_MEAN[None, None, :]
FEAT /= FEAT_STD[None, None, :]
del train_feat

LOCAL_STATE_MEAN, LOCAL_STATE_STD = fit_feature_scaler(LOCAL_STATE, TRAIN_END)
HIER_CONTEXT_MEAN = {}
HIER_CONTEXT_STD = {}
for level in HIER_LEVELS:
    mean, std = fit_feature_scaler(HIER_CONTEXT[level], TRAIN_END)
    HIER_CONTEXT_MEAN[level] = mean
    HIER_CONTEXT_STD[level] = std
    HIER_CONTEXT[level] -= mean[None, None, :]
    HIER_CONTEXT[level] /= std[None, None, :]

first_origin = SEQ - 1
train_origin_values = list(range(first_origin, TRAIN_LAST_ORIGIN + 1))
if MAX_TRAIN_ORIGINS is not None and len(train_origin_values) > MAX_TRAIN_ORIGINS:
    pick = np.linspace(0, len(train_origin_values) - 1, MAX_TRAIN_ORIGINS).round().astype(int)
    train_origin_values = [train_origin_values[i] for i in pick]


class SeriesOriginGrid:
    """Indexable Cartesian product without materializing millions of tuples."""
    def __init__(self, n_series, origins):
        self.n_series = n_series
        self.origins = origins

    def __len__(self):
        return self.n_series * len(self.origins)

    def __getitem__(self, index):
        series, origin_pos = divmod(index, len(self.origins))
        return series, self.origins[origin_pos]


train_origins = SeriesOriginGrid(n_series, train_origin_values)

val_origins = [(s, VAL_ORIGIN) for s in range(n_series)]
test_origins = [(s, TEST_ORIGIN) for s in range(n_series)]

print("train windows:", len(train_origins))
print("val windows:", len(val_origins))
print("test windows:", len(test_origins))
print("train target range:", "d_%d-d_%d" % (TRAIN_START, TRAIN_END))
print("val target range:", "d_%d-d_%d" % (VAL_START, VAL_END))
print("test target range:", "d_%d-d_%d" % (TEST_START, TEST_END))


class M5WindowDataset(Dataset):
    def __init__(
        self,
        feat,
        local_state,
        hierarchy_context,
        groups,
        target,
        static,
        origins,
        past_idx,
        fut_idx,
        sample_weight=None,
        seq=56,
        horizon=28,
    ):
        self.feat = feat
        self.local_state = local_state
        self.hierarchy_context = hierarchy_context
        self.groups = groups
        self.target = target
        self.static = static
        self.origins = origins
        self.past_idx = np.asarray(past_idx, dtype=np.int64)
        self.fut_idx = np.asarray(fut_idx, dtype=np.int64)
        self.sample_weight = sample_weight
        self.seq = seq
        self.horizon = horizon

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        s, origin = self.origins[i]
        start = origin - self.seq + 1
        past_base = self.feat[s, start:origin + 1][:, self.past_idx]
        past_local = self.local_state[s, start:origin + 1].astype(np.float32)
        past_local = (past_local - LOCAL_STATE_MEAN) / LOCAL_STATE_STD
        past_hierarchy = np.concatenate([
            self.hierarchy_context[level][self.groups[level][s], start:origin + 1]
            for level in HIER_LEVELS
        ], axis=1)
        past = np.concatenate([past_base, past_local, past_hierarchy], axis=1)
        future = self.feat[s, origin + 1:origin + 1 + self.horizon][:, self.fut_idx]
        target = self.target[s, origin + 1:origin + 1 + self.horizon]
        weight = 1.0 if self.sample_weight is None else float(self.sample_weight[s])
        mask = np.full(self.horizon, weight, dtype=np.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(past)),
            torch.from_numpy(np.ascontiguousarray(future)),
            torch.from_numpy(self.static[s].astype(np.int64)),
            torch.from_numpy(np.ascontiguousarray(target)),
            torch.from_numpy(np.ascontiguousarray(mask)),
            torch.tensor(s, dtype=torch.long),
            torch.tensor(origin, dtype=torch.long),
        )


class OriginBatchSampler:
    """Build batches from one origin so aggregate losses compare the same days."""
    def __init__(self, n_series, n_origins, batch_size, seed=42):
        self.n_series = n_series
        self.n_origins = n_origins
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for origin_pos in rng.permutation(self.n_origins):
            series_order = rng.permutation(self.n_series)
            for start in range(0, self.n_series, self.batch_size):
                series_batch = series_order[start:start + self.batch_size]
                yield [int(s) * self.n_origins + int(origin_pos) for s in series_batch]

    def __len__(self):
        return self.n_origins * int(math.ceil(self.n_series / self.batch_size))


dataset_args = (FEAT, LOCAL_STATE, HIER_CONTEXT, GROUPS, SALES, STATIC)
train_ds = M5WindowDataset(
    *dataset_args,
    train_origins,
    PAST_IDX,
    FUT_IDX,
    sample_weight=series_weight,
    seq=SEQ,
    horizon=H,
)
val_ds = M5WindowDataset(
    *dataset_args, val_origins, PAST_IDX, FUT_IDX, seq=SEQ, horizon=H
)
test_ds = M5WindowDataset(
    *dataset_args, test_origins, PAST_IDX, FUT_IDX, seq=SEQ, horizon=H
)

if HIERARCHICAL_LOSS_WEIGHT > 0:
    train_loader = DataLoader(
        train_ds,
        batch_sampler=OriginBatchSampler(n_series, len(train_origin_values), BATCH, SEED),
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
    )
else:
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        drop_last=False,
    )
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
)

batch = next(iter(train_loader))
print("past:", tuple(batch[0].shape), "| future:", tuple(batch[1].shape))
print("static:", tuple(batch[2].shape), "| target:", tuple(batch[3].shape), "| mask:", tuple(batch[4].shape))

# %% [markdown]
# ## 5. TFT-lite model
#
# Model gom static embeddings, variable selection networks, LSTM encoder/decoder,
# interpretable-style attention va mean head. Head dung softplus de dam bao mu > 0.

# %%
class GRN(nn.Module):
    def __init__(self, d_model, ctx_dim=0, dropout=0.1):
        super().__init__()
        in_dim = d_model + ctx_dim
        self.fc1 = nn.Linear(in_dim, 2 * d_model)
        self.fc2 = nn.Linear(2 * d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, ctx=None):
        h = torch.cat([x, ctx], dim=-1) if ctx is not None else x
        h = F.elu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        gate = torch.sigmoid(self.gate(h))
        return self.norm(x + gate * h)


class VSN(nn.Module):
    def __init__(self, n_features, d_model, dropout=0.1):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ELU(),
            nn.Linear(d_model, n_features),
        )
        self.proj = nn.Linear(n_features, d_model)
        self.grn = GRN(d_model, dropout=dropout)

    def forward(self, x, ctx):
        w = torch.softmax(self.weight_net(ctx), dim=-1)
        return self.grn(self.proj(x * w.unsqueeze(1)))


class StaticEmbedding(nn.Module):
    def __init__(self, cards, d_model):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, d_model) for card in cards])
        self.proj = nn.Linear(d_model * len(cards), d_model)

    def forward(self, static):
        parts = [emb(static[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.proj(torch.cat(parts, dim=-1))


class TFTLite(nn.Module):
    def __init__(self, n_past, n_future, cards, d_model=32, heads=2, dropout=0.1, horizon=28):
        super().__init__()
        self.horizon = horizon
        self.static = StaticEmbedding(cards, d_model)
        self.past_vsn = VSN(n_past, d_model, dropout)
        self.fut_vsn = VSN(n_future, d_model, dropout)
        self.encoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.decoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.static_ctx = nn.Linear(d_model, d_model)
        self.enrich = GRN(d_model, ctx_dim=d_model, dropout=dropout)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff = GRN(d_model, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)
        self.zero_head = nn.Linear(d_model, 1)

    def forward(self, past, future, static, return_zero_logits=False):
        ctx = self.static(static)
        past = self.past_vsn(past, ctx)
        future = self.fut_vsn(future, ctx)
        enc, (h, c) = self.encoder(past)
        dec, _ = self.decoder(future, (h, c))
        x = torch.cat([enc, dec], dim=1)
        x = x + self.static_ctx(ctx).unsqueeze(1)
        x = self.enrich(x, ctx.unsqueeze(1).expand(-1, x.size(1), -1))

        length = x.size(1)
        causal = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn, _ = self.attn(x, x, x, attn_mask=causal, need_weights=False)
        x = self.attn_norm(x + self.drop(attn))
        x = self.ff(x)
        forecast_state = x[:, -self.horizon:, :]
        severity = F.softplus(self.head(forecast_state).squeeze(-1)) + 1e-4
        zero_logits = self.zero_head(forecast_state).squeeze(-1)
        mean = torch.sigmoid(zero_logits) * severity
        if return_zero_logits:
            return mean, zero_logits
        return mean


model = TFTLite(
    n_past=len(MODEL_PAST_COLS),
    n_future=len(FUT_COLS),
    cards=STATIC_CARD,
    d_model=D_MODEL,
    heads=HEADS,
    dropout=DROPOUT,
    horizon=H,
).to(DEVICE)
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
# ## 6. Loss, metrics and trainer

# %%
def tweedie_loss_elementwise(y, mu, p=TWEEDIE_P):
    mu = mu.clamp_min(1e-6)
    return -y * torch.pow(mu, 1.0 - p) / (1.0 - p) + torch.pow(mu, 2.0 - p) / (2.0 - p)


def masked_loss(y, mu, mask):
    if LOSS == "tweedie":
        loss = tweedie_loss_elementwise(y, mu)
    else:
        loss = (mu - y) ** 2
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def zero_occurrence_loss(y, logits, mask):
    occurrence = (y > 0).to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, occurrence, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def hierarchical_aux_loss(y, mu, mask, series_index):
    """Relative aggregate error for L1/L2/L3/L9 within an origin-aligned batch."""
    # Keep index_add inputs in one dtype under CUDA autocast.
    valid = (mask > 0).float()
    pred = mu.float() * valid
    actual = y.float() * valid
    losses = []

    for level in ("L1", "L2", "L3", "L9"):
        codes = HIERARCHY_GROUP_TENSORS[level][series_index]
        _, inverse = torch.unique(codes, sorted=False, return_inverse=True)
        n_groups = int(inverse.max().item()) + 1
        pred_sum = torch.zeros(n_groups, pred.size(1), device=pred.device, dtype=pred.dtype)
        actual_sum = torch.zeros_like(pred_sum)
        count = torch.zeros_like(pred_sum)
        pred_sum.index_add_(0, inverse, pred)
        actual_sum.index_add_(0, inverse, actual)
        count.index_add_(0, inverse, valid)

        group_valid = count > 0
        denominator = actual_sum.detach().abs() + count.clamp_min(1.0)
        relative_error = (pred_sum - actual_sum) / denominator
        losses.append((relative_error[group_valid] ** 2).mean())

    return torch.stack(losses).mean()


def eval_all(y_true, y_pred, contexts, tag):
    rows, overall = evaluate_wrmsse5(y_true, y_pred, contexts)
    df = pd.DataFrame(rows)
    df.insert(0, "model", tag)
    return df, overall


def make_summary_row(frame, tag, overall):
    l1 = frame[frame["level"].eq("L1")].iloc[0]
    l12 = frame[frame["level"].eq("L12")].iloc[0]
    row = {
        "model": tag,
        "WRMSSE_5levels": overall,
        "L1_WMAPE": l1["WMAPE"],
        "L12_WMAPE": l12["WMAPE"],
        "L12_Bias": l12["Bias"],
        "L12_Ratio": l12["Ratio"],
    }
    for level in LEVELS:
        row[f"{level}_WRMSSE"] = frame.loc[frame["level"].eq(level), "WRMSSE"].iloc[0]
    return row


def seasonal_naive(sales, target_start, lag, horizon=28):
    history_end = target_start - 1
    if history_end < lag:
        raise ValueError(f"Need at least {lag} history days")
    pattern = sales[:, history_end - lag:history_end]
    repeats = int(math.ceil(horizon / lag))
    return np.tile(pattern, (1, repeats))[:, :horizon].astype(np.float32)


def weekday_mean_naive(sales, target_start, horizon=28, weeks=4):
    history_end = target_start - 1
    window = 7 * weeks
    if history_end < window:
        raise ValueError(f"Need at least {window} history days")
    recent = sales[:, history_end - window:history_end].reshape(len(sales), weeks, 7)
    weekday_mean = recent.mean(axis=1)
    return np.tile(weekday_mean, (1, int(math.ceil(horizon / 7))))[:, :horizon].astype(np.float32)


def zero_metrics(y_true, y_pred, threshold=0.5):
    actual = y_true.ravel()
    pred = y_pred.ravel()
    actual_zero = actual < 0.5
    pred_zero = pred < threshold
    tp = float((actual_zero & pred_zero).sum())
    fp = float((~actual_zero & pred_zero).sum())
    fn = float((actual_zero & ~pred_zero).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)

    nonzero = ~actual_zero
    return {
        "zero_precision": precision,
        "zero_recall": recall,
        "zero_f1": f1,
        "zero_rate_actual": float(actual_zero.mean()),
        "zero_rate_pred": float(pred_zero.mean()),
        "mae_nonzero": float(np.abs(actual[nonzero] - pred[nonzero]).mean()) if nonzero.any() else np.nan,
        "bias_nonzero": float((pred[nonzero] - actual[nonzero]).mean()) if nonzero.any() else np.nan,
        "bias_zero": float((pred[actual_zero] - actual[actual_zero]).mean()) if actual_zero.any() else np.nan,
    }


def fit_store_factors(pred, true, groups, low=0.8, high=1.2):
    factors = {}
    for g in np.unique(groups):
        mask = groups == g
        denominator = float(pred[mask].sum())
        factor = 1.0 if denominator <= 1e-9 else float(true[mask].sum() / denominator)
        factors[g] = float(np.clip(factor, low, high))
    return factors


def apply_factors(pred, groups, factors):
    out = pred.copy()
    for g, factor in factors.items():
        out[groups == g] *= factor
    return out


def fit_series_factors(pred, true, k=5.0, low=0.5, high=2.0):
    pred_sum = pred.sum(axis=1).astype(np.float64)
    true_sum = true.sum(axis=1).astype(np.float64)
    factors = (true_sum + k) / np.maximum(pred_sum + k, 1e-9)
    factors = np.where(pred_sum + true_sum > 0, factors, 1.0)
    return np.clip(factors, low, high).astype(np.float32)


def apply_series_factors(pred, factors):
    return pred * factors[:, None]


def run_epoch(model, loader, optimizer=None, scaler=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_base_loss = 0.0
    total_hier_loss = 0.0
    total_zero_loss = 0.0
    total_rows = 0

    for past, future, static, target, mask, series_index, _ in loader:
        past = past.to(DEVICE, non_blocking=True)
        future = future.to(DEVICE, non_blocking=True)
        static = static.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)
        series_index = series_index.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                mu, zero_logits = model(past, future, static, return_zero_logits=True)
                base_loss = masked_loss(target, mu, mask)
                zero_loss = zero_occurrence_loss(target, zero_logits, mask)
                if train and HIERARCHICAL_LOSS_WEIGHT > 0:
                    hier_loss = hierarchical_aux_loss(target, mu, mask, series_index)
                else:
                    hier_loss = torch.zeros((), device=target.device, dtype=base_loss.dtype)
                loss = (
                    base_loss
                    + HIERARCHICAL_LOSS_WEIGHT * hier_loss
                    + ZERO_LOSS_WEIGHT * zero_loss
                )

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

        total_loss += float(loss.item()) * target.size(0)
        total_base_loss += float(base_loss.item()) * target.size(0)
        total_hier_loss += float(hier_loss.item()) * target.size(0)
        total_zero_loss += float(zero_loss.item()) * target.size(0)
        total_rows += target.size(0)

    denominator = max(total_rows, 1)
    return {
        "loss": total_loss / denominator,
        "base_loss": total_base_loss / denominator,
        "hier_loss": total_hier_loss / denominator,
        "zero_loss": total_zero_loss / denominator,
    }


@torch.no_grad()
def predict_ds(model, dataset, batch_size=BATCH):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
    )
    model.eval()
    preds, targets, masks = [], [], []
    for past, future, static, target, mask, _, _ in loader:
        past = past.to(DEVICE, non_blocking=True)
        future = future.to(DEVICE, non_blocking=True)
        static = static.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            mu = model(past, future, static)
        preds.append(mu.float().cpu().numpy())
        targets.append(target.numpy())
        masks.append(mask.numpy())
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(masks, axis=0),
    )

# %% [markdown]
# ## 7. Train smoke/full

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = max(1, EPOCHS * len(train_loader))
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    total_steps=total_steps,
    pct_start=0.3,
    anneal_strategy="cos",
)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

best_score = float("inf")
best_state = None
patience = 0
history = []

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    train_stats = run_epoch(
        model,
        train_loader,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
    )
    val_pred, val_true, val_valid = predict_ds(model, val_ds)
    val_pred = val_pred * val_valid
    val_df, val_wrmsse = eval_all(val_true, val_pred, VAL_EVAL_CONTEXTS, "val_bu")
    val_l1_wmape = float(val_df.loc[val_df["level"].eq("L1"), "WMAPE"].iloc[0])
    val_l12_wrmsse = float(val_df.loc[val_df["level"].eq("L12"), "WRMSSE"].iloc[0])
    score = val_l12_wrmsse if BEST_METRIC == "l12" else val_wrmsse

    history.append({
        "epoch": epoch,
        "train_loss": train_stats["loss"],
        "train_base_loss": train_stats["base_loss"],
        "train_hier_loss": train_stats["hier_loss"],
        "train_zero_loss": train_stats["zero_loss"],
        "learning_rate": optimizer.param_groups[0]["lr"],
        "val_wrmsse_5levels": val_wrmsse,
        "val_l12_wrmsse": val_l12_wrmsse,
        "val_l1_wmape": val_l1_wmape,
        "val_score": score,
        "seconds": time.time() - start,
    })

    improved = score < best_score - 1e-4
    if improved:
        best_score = score
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience = 0
    else:
        patience += 1

    print(
        f"epoch {epoch:02d} | loss {train_stats['loss']:.5f} "
        f"(base {train_stats['base_loss']:.5f}, hier {train_stats['hier_loss']:.5f}) | "
        f"zero {train_stats['zero_loss']:.5f} | "
        f"lr {optimizer.param_groups[0]['lr']:.2e} | "
        f"val_wrmsse5 {val_wrmsse:.5f} | val_l12 {val_l12_wrmsse:.5f} | "
        f"val_l1_wmape {val_l1_wmape:.5f} | {history[-1]['seconds']:.1f}s"
    )

    if patience >= 3:
        print("early stopping")
        break

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"loaded best checkpoint | {BEST_METRIC}:", best_score)

history_df = pd.DataFrame(history)
display(history_df) if "display" in globals() else print(history_df.to_string(index=False))

# %% [markdown]
# ## 8. Predict test, calibration and 5-level metrics

# %%
val_pred, val_true, val_valid = predict_ds(model, val_ds)
test_pred, test_true, test_valid = predict_ds(model, test_ds)
val_pred = val_pred * val_valid
test_pred = test_pred * test_valid

baseline_variants = []
for baseline_tag, builder in [
    ("NAIVE_7", lambda start: seasonal_naive(SALES, start, 7, H)),
    ("NAIVE_28", lambda start: seasonal_naive(SALES, start, 28, H)),
    ("WEEKDAY_MEAN_4", lambda start: weekday_mean_naive(SALES, start, H, 4)),
]:
    val_baseline = builder(VAL_START) * val_valid
    test_baseline = builder(TEST_START) * test_valid
    baseline_variants.append((
        baseline_tag,
        val_baseline[:, CALIBRATION_DAYS:],
        test_baseline,
    ))

store_groups = GROUPS["L3"]
cal_pred = val_pred[:, :CALIBRATION_DAYS]
cal_true = val_true[:, :CALIBRATION_DAYS]
select_pred = val_pred[:, CALIBRATION_DAYS:]
select_true = val_true[:, CALIBRATION_DAYS:]

# Select the calibration strategy on days not used to fit its factors.
selection_store_factors = fit_store_factors(cal_pred, cal_true, store_groups, low=0.8, high=1.2)
selection_global_factor = float(np.clip(cal_true.sum() / max(cal_pred.sum(), 1e-9), 0.8, 1.2))
selection_store = apply_factors(select_pred, store_groups, selection_store_factors)
selection_series_factors = fit_series_factors(
    apply_factors(cal_pred, store_groups, selection_store_factors),
    cal_true,
    k=SERIES_CAL_K,
    low=0.5,
    high=2.0,
)
selection_variants = {
    "BU": select_pred,
    "STORE_CAL": selection_store,
    "GLOBAL_CAL": select_pred * selection_global_factor,
    "SERIES_CAL": apply_series_factors(selection_store, selection_series_factors),
}

# Once the strategy is selected, refit factors on all validation days for test.
store_factors = fit_store_factors(val_pred, val_true, store_groups, low=0.8, high=1.2)
global_factor = float(np.clip(val_true.sum() / max(val_pred.sum(), 1e-9), 0.8, 1.2))
val_store_full = apply_factors(val_pred, store_groups, store_factors)
series_factors = fit_series_factors(val_store_full, val_true, k=SERIES_CAL_K, low=0.5, high=2.0)
test_variants = {
    "BU": test_pred,
    "STORE_CAL": apply_factors(test_pred, store_groups, store_factors),
    "GLOBAL_CAL": test_pred * global_factor,
    "SERIES_CAL": apply_series_factors(
        apply_factors(test_pred, store_groups, store_factors),
        series_factors,
    ),
}
test_bu = test_variants["BU"]
test_store = test_variants["STORE_CAL"]
test_global = test_variants["GLOBAL_CAL"]
test_series = test_variants["SERIES_CAL"]

print("refit global calibration factor:", round(global_factor, 4))
print("refit store factors:", {k: round(v, 4) for k, v in store_factors.items()})
print(
    "series factors:",
    f"min={series_factors.min():.3f}",
    f"median={np.median(series_factors):.3f}",
    f"max={series_factors.max():.3f}",
)

variants = [
    (tag, selection_variants[tag], test_variants[tag])
    for tag in ["BU", "STORE_CAL", "GLOBAL_CAL", "SERIES_CAL"]
]

val_metric_frames = []
val_summary_rows = []
test_metric_frames = []
test_summary_rows = []
for tag, vp, tp in variants:
    val_frame, val_overall = eval_all(select_true, vp, CAL_SELECT_EVAL_CONTEXTS, tag)
    val_metric_frames.append(val_frame)
    val_summary_rows.append(make_summary_row(val_frame, tag, val_overall))
    test_frame, test_overall = eval_all(test_true, tp, TEST_EVAL_CONTEXTS, tag)
    test_metric_frames.append(test_frame)
    test_summary_rows.append(make_summary_row(test_frame, tag, test_overall))

val_metrics_df = pd.concat(val_metric_frames, ignore_index=True)
val_summary_df = pd.DataFrame(val_summary_rows)
if BEST_METRIC == "l12":
    val_summary_df = val_summary_df.sort_values(["L12_WRMSSE", "WRMSSE_5levels"]).reset_index(drop=True)
else:
    val_summary_df = val_summary_df.sort_values(["WRMSSE_5levels", "L12_WRMSSE"]).reset_index(drop=True)
best_tag = str(val_summary_df.iloc[0]["model"])
val_summary_df["selected"] = val_summary_df["model"].eq(best_tag)

metrics_df = pd.concat(test_metric_frames, ignore_index=True)
summary_df = pd.DataFrame(test_summary_rows)
summary_df["selected"] = summary_df["model"].eq(best_tag)
summary_df = summary_df.sort_values(
    ["selected", "WRMSSE_5levels"],
    ascending=[False, True],
).reset_index(drop=True)

baseline_val_frames = []
baseline_test_frames = []
baseline_summary_rows = []
for tag, vp, tp in baseline_variants:
    val_frame, val_overall = eval_all(select_true, vp, CAL_SELECT_EVAL_CONTEXTS, tag)
    test_frame, test_overall = eval_all(test_true, tp, TEST_EVAL_CONTEXTS, tag)
    baseline_val_frames.append(val_frame)
    baseline_test_frames.append(test_frame)
    baseline_summary_rows.append(make_summary_row(test_frame, tag, test_overall))
baseline_val_metrics_df = pd.concat(baseline_val_frames, ignore_index=True)
baseline_metrics_df = pd.concat(baseline_test_frames, ignore_index=True)
baseline_summary_df = pd.DataFrame(baseline_summary_rows).sort_values("WRMSSE_5levels")

best_baseline_by_level = baseline_val_metrics_df.groupby("level")["WRMSSE"].min()
selected_val = val_metrics_df[val_metrics_df["model"].eq(best_tag)].set_index("level")
beats_baseline = all(
    selected_val.loc[level, "WRMSSE"] <= best_baseline_by_level.loc[level]
    for level in LEVELS
)
val_summary_df["beats_baseline_all_levels"] = val_summary_df["model"].eq(best_tag) & beats_baseline
summary_df["beats_baseline_all_levels_validation"] = summary_df["model"].eq(best_tag) & beats_baseline

print("validation summary:")
print(val_summary_df.to_string(index=False))
print("test summary:")
print(summary_df.to_string(index=False))
print("seasonal baseline test summary:")
print(baseline_summary_df.to_string(index=False))

pred_map = test_variants
best_pred = pred_map[best_tag]
best_metrics = metrics_df[metrics_df["model"].eq(best_tag)].reset_index(drop=True)
print("chosen by validation:", best_tag)
print(best_metrics.to_string(index=False))

zero_df = pd.DataFrame([
    {"model": tag, **zero_metrics(test_true, tp)}
    for tag, _, tp in variants
])
print(zero_df.to_string(index=False))

zero_threshold_rows = []
for tag, _, tp in variants:
    for threshold in [0.10, 0.25, 0.50, 0.75, 1.00]:
        zero_threshold_rows.append({
            "model": tag,
            "threshold": threshold,
            **zero_metrics(test_true, tp, threshold=threshold),
        })
zero_threshold_df = pd.DataFrame(zero_threshold_rows)
print(zero_threshold_df.to_string(index=False))

history_df.to_csv(OUT_DIR / "history_test.csv", index=False)
metrics_df.to_csv(OUT_DIR / "metrics_test.csv", index=False)
summary_df.to_csv(OUT_DIR / "summary_test.csv", index=False)
val_metrics_df.to_csv(OUT_DIR / "val_metrics_test.csv", index=False)
val_summary_df.to_csv(OUT_DIR / "val_summary_test.csv", index=False)
baseline_val_metrics_df.to_csv(OUT_DIR / "baseline_val_metrics_test.csv", index=False)
baseline_metrics_df.to_csv(OUT_DIR / "baseline_metrics_test.csv", index=False)
baseline_summary_df.to_csv(OUT_DIR / "baseline_summary_test.csv", index=False)
zero_df.to_csv(OUT_DIR / "zero_metrics_test.csv", index=False)
zero_threshold_df.to_csv(OUT_DIR / "zero_threshold_test.csv", index=False)
np.savez_compressed(
    OUT_DIR / "predictions_test.npz",
    ids=np.array(series_ids),
    y_true=test_true,
    y_pred_bu=test_bu,
    y_pred_store_cal=test_store,
    y_pred_global_cal=test_global,
    y_pred_series_cal=test_series,
    series_factors=series_factors,
    valid_mask=test_valid,
)
torch.save(
    {
        "state_dict": model.state_dict(),
        "past_cols": MODEL_PAST_COLS,
        "past_base_cols": PAST_COLS,
        "local_state_cols": list(LOCAL_STATE_COLS),
        "hierarchy_context_cols": HIER_CONTEXT_COLS,
        "future_cols": FUT_COLS,
        "static_card": STATIC_CARD,
        "config": {
            "seq": SEQ,
            "horizon": H,
            "d_model": D_MODEL,
            "heads": HEADS,
            "dropout": DROPOUT,
            "tweedie_p": TWEEDIE_P,
            "loss": LOSS,
            "series_weight_power": SERIES_WEIGHT_POWER,
            "series_cal_k": SERIES_CAL_K,
            "hierarchical_loss_weight": HIERARCHICAL_LOSS_WEIGHT,
            "zero_loss_weight": ZERO_LOSS_WEIGHT,
            "best_metric": BEST_METRIC,
            "forecast_split": FORECAST_SPLIT,
            "future_price_known": False,
            "feature_version": "v2_local_state_hierarchy_context",
            "evaluator": "wrmsse5_v1",
            "seed": SEED,
        },
        "feat_mean": FEAT_MEAN,
        "feat_std": FEAT_STD,
        "local_state_mean": LOCAL_STATE_MEAN,
        "local_state_std": LOCAL_STATE_STD,
        "hierarchy_context_mean": HIER_CONTEXT_MEAN,
        "hierarchy_context_std": HIER_CONTEXT_STD,
    },
    OUT_DIR / "model_test.pt",
)
with open(OUT_DIR / "config_test.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "smoke": SMOKE,
            "profile": PROFILE or "custom",
            "n_series": n_series,
            "epochs": EPOCHS,
            "max_train_origins": MAX_TRAIN_ORIGINS,
            "batch": BATCH,
            "seq": SEQ,
            "horizon": H,
            "d_model": D_MODEL,
            "heads": HEADS,
            "loss": LOSS,
            "tweedie_p": TWEEDIE_P,
            "series_weight_power": SERIES_WEIGHT_POWER,
            "series_cal_k": SERIES_CAL_K,
            "hierarchical_loss_weight": HIERARCHICAL_LOSS_WEIGHT,
            "zero_loss_weight": ZERO_LOSS_WEIGHT,
            "calibration_days": CALIBRATION_DAYS,
            "best_metric": BEST_METRIC,
            "forecast_split": FORECAST_SPLIT,
            "future_price_known": False,
            "feature_version": "v2_local_state_hierarchy_context",
            "sales_file": SALES_FILE,
            "evaluator": "wrmsse5_v1",
            "best_model": best_tag,
            "beats_baseline_all_levels_validation": bool(beats_baseline),
            "best_val_l12_wrmsse": float(
                val_summary_df.loc[val_summary_df["model"].eq(best_tag), "L12_WRMSSE"].iloc[0]
            ),
            "best_test_wrmsse_5levels": float(
                summary_df.loc[summary_df["model"].eq(best_tag), "WRMSSE_5levels"].iloc[0]
            ),
        },
        f,
        indent=2,
    )
print("saved to:", OUT_DIR.resolve())

# %% [markdown]
# ## 9. Zero-rate buckets and error analysis

# %%
zero_rate_train = (SALES[:, :TRAIN_END] == 0).mean(axis=1)
buckets = pd.cut(
    zero_rate_train,
    bins=[-0.01, 0.25, 0.50, 0.75, 0.99, 1.0],
    labels=["<25%", "25-50%", "50-75%", "75-99%", "99-100%"],
)

bucket_rows = []
for label in buckets.categories:
    mask = (buckets == label)
    if not mask.any():
        continue
    yt = test_true[mask]
    yp = best_pred[mask]
    bucket_rows.append({
        "zero_bucket": label,
        "n_series": int(mask.sum()),
        "actual_sum": float(yt.sum()),
        "pred_sum": float(yp.sum()),
        "MAE": float(np.abs(yt - yp).mean()),
        "WMAPE": float(np.abs(yt - yp).sum() / max(np.abs(yt).sum(), 1e-9)),
        "Bias": float((yp - yt).sum() / max(np.abs(yt).sum(), 1e-9)),
        "Ratio": float(yp.sum() / max(yt.sum(), 1e-9)),
    })

bucket_df = pd.DataFrame(bucket_rows)
print(bucket_df.to_string(index=False))
bucket_df.to_csv(OUT_DIR / "zero_bucket_test.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(zero_rate_train, bins=20, color="#4c72b0", alpha=0.85)
axes[0].set_title("Training zero rate by series")
axes[0].set_xlabel("zero rate")
axes[0].set_ylabel("series")

daily_true = test_true.sum(axis=0)
daily_bu = test_bu.sum(axis=0)
daily_best = best_pred.sum(axis=0)
axes[1].plot(daily_true, label="actual", linewidth=2)
axes[1].plot(daily_bu, label="BU", linestyle="--")
axes[1].plot(daily_best, label=best_tag, linestyle=":")
axes[1].set_title("Test total units by day")
axes[1].set_xlabel("day in horizon")
axes[1].set_ylabel("units")
axes[1].legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "test_diagnostics.png", dpi=120)
plt.show()

# %% [markdown]
# ## 10. Feature importance from VSN

# %%
past_weight_sum = torch.zeros(len(MODEL_PAST_COLS), device=DEVICE)
future_weight_sum = torch.zeros(len(FUT_COLS), device=DEVICE)
weight_rows = 0
with torch.no_grad():
    for _, _, static, _, _, _, _ in val_loader:
        static = static.to(DEVICE, non_blocking=True)
        ctx = model.static(static)
        past_weight_sum += torch.softmax(model.past_vsn.weight_net(ctx), dim=-1).sum(dim=0)
        future_weight_sum += torch.softmax(model.fut_vsn.weight_net(ctx), dim=-1).sum(dim=0)
        weight_rows += static.size(0)
past_weights = (past_weight_sum / max(weight_rows, 1)).cpu().numpy()
future_weights = (future_weight_sum / max(weight_rows, 1)).cpu().numpy()

importance_df = pd.concat([
    pd.DataFrame({"feature": MODEL_PAST_COLS, "weight": past_weights, "kind": "past"}),
    pd.DataFrame({"feature": FUT_COLS, "weight": future_weights, "kind": "future"}),
], ignore_index=True).sort_values("weight", ascending=False)

print(importance_df.to_string(index=False))
importance_df.to_csv(OUT_DIR / "feature_importance_test.csv", index=False)

# %% [markdown]
# ## 11. Doc ket qua
#
# - `metrics_test.csv`: metric theo tung cap cho BU, store calibration, global calibration, series calibration.
# - `summary_test.csv`: so sanh 4 bien the; cot `selected` duoc chon tren validation, khong phai test.
# - `val_metrics_test.csv`, `val_summary_test.csv`: metric validation de chon model.
# - `history_test.csv`: base/hierarchical loss, learning rate va val metric theo epoch.
# - `zero_metrics_test.csv`, `zero_threshold_test.csv`, `zero_bucket_test.csv`: chat luong du doan zero.
# - `predictions_test.npz`: du doan test de audit, gom `y_pred_series_cal`.
# - `model_test.pt`, `config_test.json`: checkpoint va cau hinh.
#
# Buoc tiep theo:
# 1. Chay full 5 cap: `PROFILE=full_5level FORECAST_SPLIT=evaluation`.
# 2. Chon checkpoint theo WRMSSE-5, khong toi uu rieng L12.
# 3. Ablate local lifecycle va hierarchy context de do dong gop theo tung cap.
