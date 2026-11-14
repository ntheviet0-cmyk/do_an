# %% [markdown]
# # TFT M5 - Gemini Update: Zero-Inflation Hurdle + Dynamic VSN + Hierarchical Calibration
#
# Bản nâng cấp toàn diện xử lý Zero-Inflation và cải thiện kiến trúc TFT trên M5 (L1/L2/L3/L9/L12):
# 1. **Pre-launch Causal Filtering**: Loại bỏ các cửa sổ huấn luyện trước ngày mở bán đầu tiên (`origin < first_active_day[s]`), chấm dứt hiện tượng zero giả tạo kéo tụt dự báo.
# 2. **Croston/TSB & Horizon-Aligned Known Lags**: Bổ sung `croston_idi_ratio`, `log_tsb_expected_28` vào Encoder và đưa trực tiếp các lag nhân quả (`lag_28`, `lag_35`, `lag_56`, `lag_364` đều `<= origin`) vào Decoder.
# 3. **Kiến trúc Hurdle-TFT chuẩn (Lim et al., 2021)**:
#    - **Dynamic Time-Varying VSN**: Trọng số chọn biến tính động theo từng bước thời gian $t$ kết hợp ngữ cảnh tĩnh `ctx`.
#    - **GLU-gated GRN** & **Static LSTM State Initialization** (`h0, c0` khởi tạo từ `static`).
#    - **Positional / Horizon Embedding** + **Interpretable Multi-Head Attention** (dùng chung ma trận Value $W_V$).
# 4. **Decoupled Two-Stage Hurdle Head (Chống Double Shrinkage)**:
#    - Tách xác suất bán $p = \sigma(\text{zero\_logits})$ và quy mô đơn hàng dương $\mu_+ = 1.0 + \text{softplus}(\text{pos\_head})$.
#    - Nhánh $\mu_+$ được huấn luyện bằng hàm `positive_magnitude_loss` **chỉ trên các ngày có bán dương ($y > 0$)**, không bị các ngày $y=0$ kéo chìm về 0.
# 5. **Shuffled Store-Grouped Origin Sampler + Model EMA**:
#    - Gom chuỗi theo cụm Cửa hàng/Ngành hàng (`L3/L9`) tại mỗi origin và **xáo trộn thứ tự các batch qua mọi origin**, xóa bỏ hiện tượng dao động Validation loss giữa các epoch.
#    - Duy trì trọng số trung bình trượt mũ (**EMA**) để ổn định dự báo đa cấp.
# 6. **Hierarchical Multi-Level Shrinkage Calibration (`L3 -> L9 -> L12`)**:
#    - Hiệu chỉnh phân cấp từ Cửa hàng (`L3`) -> Ngành hàng (`L9`) -> Mặt hàng (`L12`), chọn siêu tham số nghiêm ngặt trên nửa sau của Validation.

# %%
import os
import sys
import copy
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
        compute_first_active_day,
        fit_feature_scaler,
        hierarchy_feature_names,
    )
except ModuleNotFoundError:
    from kaggle.hurdle_features import (
        HIER_CHANNELS,
        LOCAL_STATE_COLS,
        build_hierarchy_context,
        build_local_state_features,
        compute_first_active_day,
        fit_feature_scaler,
        hierarchy_feature_names,
    )

import matplotlib
if not Path("/kaggle").exists():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print

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
LR = float(os.environ.get("LR", 5e-4))
WEIGHT_DECAY = 1e-4
TWEEDIE_P = float(os.environ.get("TWEEDIE_P", "1.25"))
LOSS = os.environ.get("LOSS", "tweedie")
SERIES_WEIGHT_POWER = float(os.environ.get("SERIES_WEIGHT_POWER", "0.5"))
SERIES_CAL_K = float(os.environ.get("SERIES_CAL_K", "10.0"))
L9_CAL_K = float(os.environ.get("L9_CAL_K", "20.0"))
HIERARCHICAL_LOSS_WEIGHT = float(os.environ.get("HIERARCHICAL_LOSS_WEIGHT", "0.25"))
ZERO_LOSS_WEIGHT = float(os.environ.get("ZERO_LOSS_WEIGHT", "0.15"))
POS_LOSS_WEIGHT = float(os.environ.get("POS_LOSS_WEIGHT", "0.20"))
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.995"))
FILTER_PRELAUNCH = os.environ.get("FILTER_PRELAUNCH", "1") == "1"
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
    EPOCHS = int(os.environ.get("FULL_EPOCHS", "15"))
    MAX_TRAIN_ORIGINS = int(os.environ.get("MAX_TRAIN_ORIGINS", "160"))
    BATCH = int(os.environ.get("BATCH", "512"))
    D_MODEL = int(os.environ.get("D_MODEL", "64"))
    HEADS = int(os.environ.get("HEADS", "4"))
    BEST_METRIC = os.environ.get("BEST_METRIC", "wrmsse5").lower()
assert BEST_METRIC in {"l12", "wrmsse5"}
assert FORECAST_SPLIT in {"validation", "evaluation"}
assert 0.0 <= HIERARCHICAL_LOSS_WEIGHT <= 1.0
assert ZERO_LOSS_WEIGHT >= 0.0
assert POS_LOSS_WEIGHT >= 0.0
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
    "| lr:", LR,
    "| tweedie_p:", TWEEDIE_P,
    "| hier_w:", HIERARCHICAL_LOSS_WEIGHT,
    "| zero_w:", ZERO_LOSS_WEIGHT,
    "| pos_w:", POS_LOSS_WEIGHT,
    "| filter_prelaunch:", FILTER_PRELAUNCH,
    "| forecast_split:", FORECAST_SPLIT,
)
print("output:", safe_str(OUT_DIR.resolve()))

# %% [markdown]
# ## 1. Load M5
#
# Tìm dữ liệu theo thứ tự: `M5_DATA_ROOT`, Kaggle input, `../data/raw`, `data/raw`.
# Nếu local chỉ có `sales_train_validation.csv` mà `FORECAST_SPLIT=evaluation`, tự động fallback về `sales_train_validation.csv` khi chạy smoke test local.

# %%
def find_data_root():
    global SALES_FILE, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END, VAL_ORIGIN, TEST_ORIGIN, TRAIN_LAST_ORIGIN
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
        if SMOKE and SALES_FILE == "sales_train_evaluation.csv" and (path / "sales_train_validation.csv").exists():
            SALES_FILE = "sales_train_validation.csv"
            TRAIN_END = 1857
            VAL_START, VAL_END = 1858, 1885
            TEST_START, TEST_END = 1886, 1913
            VAL_ORIGIN = VAL_START - 2
            TEST_ORIGIN = TEST_START - 2
            TRAIN_LAST_ORIGIN = TRAIN_END - H - 1
            return path.resolve()

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
        f"Input hien co: {attached_text}"
    )


DATA_ROOT = find_data_root()
print("data_root:", safe_str(DATA_ROOT), "| file:", SALES_FILE)


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
# ## 2. Build matrices: sales, causal past price, calendar, first active day

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
FIRST_ACTIVE_DAY = compute_first_active_day(SALES, PRICE_AVAILABLE)

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
print("median first active day:", float(np.median(FIRST_ACTIVE_DAY)))

del prices, pairs, price_weekly, price_values, pair_order, pair_index, cal, sales_all

# %% [markdown]
# ## 3. Feature engineering (with Horizon-Aligned Known Lags & Croston/TSB State)
#
# - `PAST_COLS`: thông tin lịch sử đến `origin`.
# - `FUT_COLS`: gồm lịch/SNAP/event biết trước VÀ 4 biến neo nhân quả (`lag_28`, `lag_35`, `lag_56`, `lag_364`).
#   Tại ngày mục tiêu `t = origin + h` (`1 <= h <= 28`), `lag_28[t]` là doanh số tại `origin + h - 28 <= origin` (hoàn toàn thuộc quá khứ trước `origin`, không rò rỉ tương lai, không chứa giá tương lai).

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
    "lag_28", "lag_35", "lag_56", "lag_364",
]
FEATURE_COLS = list(dict.fromkeys(PAST_COLS + FUT_COLS))

FEAT = np.zeros((n_series, n_days, len(FEATURE_COLS)), dtype=np.float32)


def store_feature(name, values):
    FEAT[:, :, FEATURE_COLS.index(name)] = values


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
    "| total past:", len(MODEL_PAST_COLS),
    "| future:", len(FUT_COLS),
)
print("static cards:", STATIC_CARD)
print({k: int(v.max()) + 1 for k, v in GROUPS.items()})

# %% [markdown]
# ## 4. Split, scaler, Pre-launch Window Filtering & Shuffled Store-Grouped Sampler

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


class ActiveSeriesOriginGrid:
    """Compact indexable (series, origin) collection filtering pre-launch windows causally."""
    def __init__(self, n_series, origins, first_active_day=None, filter_prelaunch=True):
        self.n_series = n_series
        self.origins = np.asarray(origins, dtype=np.int32)
        if filter_prelaunch and first_active_day is not None:
            pairs = []
            origin_Buckets = {}
            for o_idx, origin in enumerate(self.origins):
                active_s = np.flatnonzero(first_active_day <= int(origin)).astype(np.int32)
                if len(active_s) == 0:
                    active_s = np.arange(n_series, dtype=np.int32)
                origin_col = np.full(len(active_s), int(origin), dtype=np.int32)
                start_idx = sum(len(p) for p in pairs)
                pairs.append(np.stack([active_s, origin_col], axis=1))
                origin_Buckets[int(origin)] = (
                    np.arange(start_idx, start_idx + len(active_s), dtype=np.int32),
                    active_s,
                )
            self.pairs = np.concatenate(pairs, axis=0)
            self.origin_buckets = origin_Buckets
        else:
            self.pairs = None
            self.origin_buckets = None

    def __len__(self):
        if self.pairs is not None:
            return len(self.pairs)
        return self.n_series * len(self.origins)

    def __getitem__(self, index):
        if self.pairs is not None:
            row = self.pairs[index]
            return int(row[0]), int(row[1])
        series, origin_pos = divmod(int(index), len(self.origins))
        return int(series), int(self.origins[origin_pos])


train_origins = ActiveSeriesOriginGrid(
    n_series,
    train_origin_values,
    first_active_day=FIRST_ACTIVE_DAY,
    filter_prelaunch=FILTER_PRELAUNCH,
)

val_origins = [(s, VAL_ORIGIN) for s in range(n_series)]
test_origins = [(s, TEST_ORIGIN) for s in range(n_series)]

raw_total_windows = n_series * len(train_origin_values)
print(
    "train windows:", len(train_origins),
    f"({100.0 * len(train_origins) / max(raw_total_windows, 1):.1f}% active after causal pre-launch filter)",
)
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


class ShuffledOriginGroupBatchSampler:
    """Build store/department-clustered batches within each origin, then shuffle batch order globally.

    1. Within each origin, series are ordered by L9 (Store-Department) with random tie-breaking,
       ensuring each batch of 512 series contains complete L9/L3 groups for exact hierarchical loss.
    2. Across origins, the list of batches is globally permuted so consecutive gradient updates
       sample across the entire multi-year timeline, preventing epoch-end origin drift.
    """
    def __init__(self, origin_grid, l9_groups, batch_size, seed=42):
        self.origin_grid = origin_grid
        self.l9_groups = np.asarray(l9_groups, dtype=np.int64)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self._cached_len = self._compute_len()

    def _compute_len(self):
        if self.origin_grid.origin_buckets is not None:
            return sum(
                int(math.ceil(len(indices) / self.batch_size))
                for indices, _ in self.origin_grid.origin_buckets.values()
            )
        return len(self.origin_grid.origins) * int(
            math.ceil(self.origin_grid.n_series / self.batch_size)
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        all_batches = []
        n_origins = len(self.origin_grid.origins)

        if self.origin_grid.origin_buckets is not None:
            for _, (dataset_indices, active_series) in self.origin_grid.origin_buckets.items():
                perm = rng.permutation(len(dataset_indices))
                shuffled_idx = dataset_indices[perm]
                shuffled_l9 = self.l9_groups[active_series[perm]]
                order = np.argsort(shuffled_l9, kind="mergesort")
                ordered_idx = shuffled_idx[order]
                for start in range(0, len(ordered_idx), self.batch_size):
                    all_batches.append(ordered_idx[start:start + self.batch_size].tolist())
        else:
            for origin_pos in range(n_origins):
                perm = rng.permutation(self.origin_grid.n_series)
                order = perm[np.argsort(self.l9_groups[perm], kind="mergesort")]
                for start in range(0, self.origin_grid.n_series, self.batch_size):
                    series_batch = order[start:start + self.batch_size]
                    all_batches.append([int(s) * n_origins + int(origin_pos) for s in series_batch])

        for b_idx in rng.permutation(len(all_batches)):
            yield all_batches[int(b_idx)]

    def __len__(self):
        return self._cached_len


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
        batch_sampler=ShuffledOriginGroupBatchSampler(train_origins, GROUPS["L9"], BATCH, SEED),
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
# ## 5. Upgraded Hurdle-TFT Model (Dynamic VSN, GLU-GRN, Static State Init, Positional Embedding, Interpretable Attention, Decoupled Two-Stage Hurdle Head)

# %%
class GLU(nn.Module):
    """Gated Linear Unit (Lim et al., 2021)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc_val = nn.Linear(in_dim, out_dim)
        self.fc_gate = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc_val(x) * torch.sigmoid(self.fc_gate(x))


class GRN(nn.Module):
    """Gated Residual Network with GLU gating and optional static context."""
    def __init__(self, d_model, ctx_dim=0, dropout=0.1):
        super().__init__()
        in_dim = d_model + ctx_dim
        self.fc1 = nn.Linear(in_dim, 2 * d_model)
        self.fc2 = nn.Linear(2 * d_model, d_model)
        self.glu = GLU(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, ctx=None):
        h = torch.cat([x, ctx], dim=-1) if ctx is not None else x
        h = F.elu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return self.norm(x + self.glu(h))


class DynamicVSN(nn.Module):
    """Time-Varying Variable Selection Network conditioned on both step inputs x_t and static context."""
    def __init__(self, n_features, d_model, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.input_proj = nn.Linear(n_features, d_model)
        self.ctx_proj = nn.Linear(d_model, d_model, bias=False)
        self.weight_net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_features),
        )
        self.var_value_proj = nn.Linear(n_features, d_model)
        self.grn = GRN(d_model, dropout=dropout)

    def forward(self, x, ctx, return_weights=False):
        # x: [B, T, F], ctx: [B, D]
        h = F.elu(self.input_proj(x) + self.ctx_proj(ctx).unsqueeze(1))
        w = torch.softmax(self.weight_net(h), dim=-1)  # [B, T, F]
        out = self.grn(self.var_value_proj(x * w))
        if return_weights:
            return out, w
        return out


class StaticEmbedding(nn.Module):
    """Static Covariate Encoder producing 4 specialized context vectors."""
    def __init__(self, cards, d_model):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, d_model) for card in cards])
        self.proj = nn.Linear(d_model * len(cards), d_model)
        self.grn_vsn = GRN(d_model)
        self.grn_enrich = GRN(d_model)
        self.grn_h = GRN(d_model)
        self.grn_c = GRN(d_model)

    def forward(self, static):
        parts = [emb(static[:, i]) for i, emb in enumerate(self.embeddings)]
        base = self.proj(torch.cat(parts, dim=-1))
        return base

    def contexts(self, static):
        base = self.forward(static)
        return (
            self.grn_vsn(base),
            self.grn_enrich(base),
            torch.tanh(self.grn_h(base)),
            torch.tanh(self.grn_c(base)),
        )


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-Head Attention with shared value matrix across heads (Lim et al., 2021)."""
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, need_weights=False):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x)  # Shared value [B, T, D]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), -1e4)
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        mean_attn = attn.mean(dim=1)  # [B, T, T]
        out = self.out_proj(torch.matmul(mean_attn, v))
        if need_weights:
            return out, mean_attn
        return out, None


class TFTLite(nn.Module):
    """Upgraded Hurdle-TFT with Dynamic VSN, Static State Init, Positional Encoding & Decoupled Hurdle Head."""
    def __init__(self, n_past, n_future, cards, d_model=64, heads=4, dropout=0.1, seq_len=56, horizon=28):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.static = StaticEmbedding(cards, d_model)
        self.past_vsn = DynamicVSN(n_past, d_model, dropout)
        self.fut_vsn = DynamicVSN(n_future, d_model, dropout)
        self.encoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.decoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.lstm_glu = GLU(d_model, d_model)
        self.lstm_norm = nn.LayerNorm(d_model)
        self.pos_emb = nn.Embedding(seq_len + horizon, d_model)
        self.enrich = GRN(d_model, ctx_dim=d_model, dropout=dropout)
        self.attn = InterpretableMultiHeadAttention(d_model, heads, dropout=dropout)
        self.attn_glu = GLU(d_model, d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff = GRN(d_model, dropout=dropout)
        self.drop = nn.Dropout(dropout)

        # Decoupled Two-Stage Hurdle Heads:
        # 1. zero_head predicts logits of P(Y > 0)
        # 2. head predicts conditional positive excess above 1.0: E[Y | Y > 0] = 1.0 + softplus(head)
        self.zero_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, past, future, static, return_zero_logits=False, return_aux=False):
        ctx_vsn, ctx_enrich, h0, c0 = self.static.contexts(static)
        if return_aux:
            past_vsn, past_w = self.past_vsn(past, ctx_vsn, return_weights=True)
            fut_vsn, fut_w = self.fut_vsn(future, ctx_vsn, return_weights=True)
        else:
            past_vsn = self.past_vsn(past, ctx_vsn)
            fut_vsn = self.fut_vsn(future, ctx_vsn)

        enc, (h, c) = self.encoder(past_vsn, (h0.unsqueeze(0), c0.unsqueeze(0)))
        dec, _ = self.decoder(fut_vsn, (h, c))

        vsn_all = torch.cat([past_vsn, fut_vsn], dim=1)
        lstm_all = torch.cat([enc, dec], dim=1)
        x = self.lstm_norm(vsn_all + self.drop(self.lstm_glu(lstm_all)))

        length = x.size(1)
        positions = torch.arange(length, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(positions)
        x = self.enrich(x, ctx_enrich.unsqueeze(1).expand(-1, length, -1))

        causal = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_out, attn_weights = self.attn(x, attn_mask=causal, need_weights=return_aux)
        x = self.attn_norm(x + self.drop(self.attn_glu(attn_out)))
        x = self.ff(x)

        forecast_state = x[:, -self.horizon:, :]
        # Conditional positive magnitude >= 1.0 (retail counts are integers >= 1 when positive)
        mu_pos = 1.0 + F.softplus(self.head(forecast_state).squeeze(-1))
        zero_logits = self.zero_head(forecast_state).squeeze(-1)
        p_pos = torch.sigmoid(zero_logits)
        mean = p_pos * mu_pos

        if return_aux:
            return mean, zero_logits, mu_pos, {
                "past_weights": past_w,
                "future_weights": fut_w,
                "attn_weights": attn_weights,
            }
        if return_zero_logits:
            return mean, zero_logits, mu_pos
        return mean


class ModelEMA:
    """Exponential Moving Average of model parameters for stable validation/test inference."""
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.copy_(ema_p * d + p.detach() * (1.0 - d))
        for ema_b, b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(b.detach())


model = TFTLite(
    n_past=len(MODEL_PAST_COLS),
    n_future=len(FUT_COLS),
    cards=STATIC_CARD,
    d_model=D_MODEL,
    heads=HEADS,
    dropout=DROPOUT,
    seq_len=SEQ,
    horizon=H,
).to(DEVICE)
ema_helper = ModelEMA(model, decay=EMA_DECAY)
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
# ## 6. Loss (Tweedie + Focal Occurrence + Conditional Positive Magnitude + Store-Grouped Hierarchical), Metrics & Trainer

# %%
def tweedie_loss_elementwise(y, mu, p=TWEEDIE_P):
    mu = mu.clamp_min(1e-5)
    return -y * torch.pow(mu, 1.0 - p) / (1.0 - p) + torch.pow(mu, 2.0 - p) / (2.0 - p)


def masked_loss(y, mu, mask):
    if LOSS == "tweedie":
        loss = tweedie_loss_elementwise(y, mu)
    else:
        loss = (mu - y) ** 2
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def zero_occurrence_loss(y, logits, mask, gamma=1.5):
    """Focal-modulated Binary Cross-Entropy for occurrence P(Y > 0)."""
    occurrence = (y > 0).to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, occurrence, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(occurrence > 0.5, probs, 1.0 - probs)
    focal_weight = torch.pow((1.0 - pt).clamp_min(1e-6), gamma)
    loss = focal_weight * bce
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def positive_magnitude_loss(y, mu_pos, mask):
    """Train conditional positive magnitude E[Y | Y > 0] ONLY on days where actual sales y > 0.

    Prevents zero-sale days from dragging down the positive severity head (solves Double Shrinkage).
    """
    pos_mask = mask * (y > 0).to(mask.dtype)
    pos_count = pos_mask.sum()
    if pos_count <= 0:
        return torch.zeros((), device=y.device, dtype=mu_pos.dtype)
    log_err = F.smooth_l1_loss(
        torch.log1p(mu_pos.float()),
        torch.log1p(y.float().clamp_min(0.0)),
        reduction="none",
        beta=0.25,
    )
    rel_err = torch.abs(mu_pos.float() - y.float()) / (y.float().detach() + 1.0)
    loss = 0.7 * log_err + 0.3 * rel_err
    return (loss * pos_mask.float()).sum() / pos_count.float().clamp_min(1.0)


def hierarchical_aux_loss(y, mu, mask, series_index):
    """Relative aggregate error for L1/L2/L3/L9 within a store-grouped origin batch."""
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


def fit_group_factors(pred, true, groups, k=0.0, low=0.75, high=1.30):
    """Empirical Bayes shrinkage group calibration factor: (true_sum + k) / (pred_sum + k)."""
    factors = {}
    for g in np.unique(groups):
        mask = groups == g
        p_sum = float(pred[mask].sum())
        t_sum = float(true[mask].sum())
        factor = 1.0 if (p_sum + t_sum) <= 1e-9 else float((t_sum + k) / max(p_sum + k, 1e-9))
        factors[g] = float(np.clip(factor, low, high))
    return factors


def fit_store_factors(pred, true, groups, low=0.8, high=1.2):
    return fit_group_factors(pred, true, groups, k=0.0, low=low, high=high)


def apply_factors(pred, groups, factors):
    out = pred.copy()
    for g, factor in factors.items():
        out[groups == g] *= factor
    return out


def fit_series_factors(pred, true, k=10.0, low=0.5, high=2.0):
    pred_sum = pred.sum(axis=1).astype(np.float64)
    true_sum = true.sum(axis=1).astype(np.float64)
    factors = (true_sum + k) / np.maximum(pred_sum + k, 1e-9)
    factors = np.where(pred_sum + true_sum > 0, factors, 1.0)
    return np.clip(factors, low, high).astype(np.float32)


def apply_series_factors(pred, factors):
    return pred * factors[:, None]


def run_epoch(model, loader, optimizer=None, scaler=None, scheduler=None, ema=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_base_loss = 0.0
    total_hier_loss = 0.0
    total_zero_loss = 0.0
    total_pos_loss = 0.0
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
                mu, zero_logits, mu_pos = model(past, future, static, return_zero_logits=True)
                base_loss = masked_loss(target, mu, mask)
                zero_loss = zero_occurrence_loss(target, zero_logits, mask)
                pos_loss = positive_magnitude_loss(target, mu_pos, mask)
                if train and HIERARCHICAL_LOSS_WEIGHT > 0:
                    hier_loss = hierarchical_aux_loss(target, mu, mask, series_index)
                else:
                    hier_loss = torch.zeros((), device=target.device, dtype=base_loss.dtype)
                loss = (
                    base_loss
                    + HIERARCHICAL_LOSS_WEIGHT * hier_loss
                    + ZERO_LOSS_WEIGHT * zero_loss
                    + POS_LOSS_WEIGHT * pos_loss
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
                if ema is not None:
                    ema.update(model)

        bs = target.size(0)
        total_loss += float(loss.item()) * bs
        total_base_loss += float(base_loss.item()) * bs
        total_hier_loss += float(hier_loss.item()) * bs
        total_zero_loss += float(zero_loss.item()) * bs
        total_pos_loss += float(pos_loss.item()) * bs
        total_rows += bs

    denominator = max(total_rows, 1)
    return {
        "loss": total_loss / denominator,
        "base_loss": total_base_loss / denominator,
        "hier_loss": total_hier_loss / denominator,
        "zero_loss": total_zero_loss / denominator,
        "pos_loss": total_pos_loss / denominator,
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
# ## 7. Train with Cosine Warmup + Model EMA Evaluation

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = max(1, EPOCHS * len(train_loader))
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    total_steps=total_steps,
    pct_start=0.15,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
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
        ema=ema_helper,
    )
    # Evaluate both raw model and EMA model on validation, pick the better state
    val_pred_raw, val_true, val_valid = predict_ds(model, val_ds)
    val_pred_ema, _, _ = predict_ds(ema_helper.ema, val_ds)
    val_pred_raw = val_pred_raw * val_valid
    val_pred_ema = val_pred_ema * val_valid

    val_df_raw, val_wrmsse_raw = eval_all(val_true, val_pred_raw, VAL_EVAL_CONTEXTS, "val_raw")
    val_df_ema, val_wrmsse_ema = eval_all(val_true, val_pred_ema, VAL_EVAL_CONTEXTS, "val_ema")

    raw_l12 = float(val_df_raw.loc[val_df_raw["level"].eq("L12"), "WRMSSE"].iloc[0])
    ema_l12 = float(val_df_ema.loc[val_df_ema["level"].eq("L12"), "WRMSSE"].iloc[0])
    score_raw = raw_l12 if BEST_METRIC == "l12" else val_wrmsse_raw
    score_ema = ema_l12 if BEST_METRIC == "l12" else val_wrmsse_ema

    use_ema = score_ema <= score_ema_or_raw if (score_ema_or_raw := min(score_raw, score_ema)) else True
    val_df = val_df_ema if score_ema <= score_raw else val_df_raw
    val_wrmsse = val_wrmsse_ema if score_ema <= score_raw else val_wrmsse_raw
    val_l12_wrmsse = ema_l12 if score_ema <= score_raw else raw_l12
    val_l1_wmape = float(val_df.loc[val_df["level"].eq("L1"), "WMAPE"].iloc[0])
    score = min(score_raw, score_ema)
    active_net = ema_helper.ema if score_ema <= score_raw else model

    history.append({
        "epoch": epoch,
        "train_loss": train_stats["loss"],
        "train_base_loss": train_stats["base_loss"],
        "train_hier_loss": train_stats["hier_loss"],
        "train_zero_loss": train_stats["zero_loss"],
        "train_pos_loss": train_stats["pos_loss"],
        "learning_rate": optimizer.param_groups[0]["lr"],
        "val_wrmsse_5levels": val_wrmsse,
        "val_wrmsse_raw": val_wrmsse_raw,
        "val_wrmsse_ema": val_wrmsse_ema,
        "val_l12_wrmsse": val_l12_wrmsse,
        "val_l1_wmape": val_l1_wmape,
        "val_score": score,
        "used_ema": bool(score_ema <= score_raw),
        "seconds": time.time() - start,
    })

    improved = score < best_score - 1e-4
    if improved:
        best_score = score
        best_state = {k: v.detach().cpu().clone() for k, v in active_net.state_dict().items()}
        patience = 0
    else:
        patience += 1

    print(
        f"epoch {epoch:02d} | loss {train_stats['loss']:.4f} "
        f"(base {train_stats['base_loss']:.4f}, pos {train_stats['pos_loss']:.4f}, "
        f"zero {train_stats['zero_loss']:.4f}, hier {train_stats['hier_loss']:.4f}) | "
        f"lr {optimizer.param_groups[0]['lr']:.2e} | "
        f"val_wrmsse5 {val_wrmsse:.5f} (raw {val_wrmsse_raw:.4f}/ema {val_wrmsse_ema:.4f}) | "
        f"val_l12 {val_l12_wrmsse:.5f} | {history[-1]['seconds']:.1f}s"
    )

    if patience >= 4:
        print("early stopping")
        break

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"loaded best checkpoint | {BEST_METRIC}:", best_score)

history_df = pd.DataFrame(history)
display(history_df) if "display" in globals() else print(history_df.to_string(index=False))

# %% [markdown]
# ## 8. Predict test, Hierarchical Calibration (`BU`, `STORE_CAL`, `HIER_L3_L9_CAL`, `GLOBAL_CAL`, `SERIES_CAL`) & 5-level metrics

# %%
val_pred, val_true, val_valid = predict_ds(model, val_ds)
test_pred, test_true, test_valid = predict_ds(model, test_ds)

# Causal inactive series rule: if a series has never been listed or sold up to origin, keep small floor
val_inactive = (FIRST_ACTIVE_DAY > VAL_ORIGIN)[:, None]
test_inactive = (FIRST_ACTIVE_DAY > TEST_ORIGIN)[:, None]
val_pred = np.where(val_inactive, 0.0, val_pred) * val_valid
test_pred = np.where(test_inactive, 0.0, test_pred) * test_valid

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
l9_groups = GROUPS["L9"]
cal_pred = val_pred[:, :CALIBRATION_DAYS]
cal_true = val_true[:, :CALIBRATION_DAYS]
select_pred = val_pred[:, CALIBRATION_DAYS:]
select_true = val_true[:, CALIBRATION_DAYS:]

# Select the calibration strategy on days not used to fit its factors.
selection_store_factors = fit_store_factors(cal_pred, cal_true, store_groups, low=0.8, high=1.2)
selection_global_factor = float(np.clip(cal_true.sum() / max(cal_pred.sum(), 1e-9), 0.8, 1.2))
selection_store = apply_factors(select_pred, store_groups, selection_store_factors)

cal_store_aligned = apply_factors(cal_pred, store_groups, selection_store_factors)
selection_l9_factors = fit_group_factors(
    cal_store_aligned, cal_true, l9_groups, k=L9_CAL_K, low=0.75, high=1.30
)
selection_l3_l9 = apply_factors(selection_store, l9_groups, selection_l9_factors)

selection_series_factors = fit_series_factors(
    apply_factors(cal_store_aligned, l9_groups, selection_l9_factors),
    cal_true,
    k=SERIES_CAL_K,
    low=0.5,
    high=2.0,
)
selection_variants = {
    "BU": select_pred,
    "STORE_CAL": selection_store,
    "HIER_L3_L9_CAL": selection_l3_l9,
    "GLOBAL_CAL": select_pred * selection_global_factor,
    "SERIES_CAL": apply_series_factors(selection_l3_l9, selection_series_factors),
}

# Once the strategy is selected, refit factors on all validation days for test.
store_factors = fit_store_factors(val_pred, val_true, store_groups, low=0.8, high=1.2)
global_factor = float(np.clip(val_true.sum() / max(val_pred.sum(), 1e-9), 0.8, 1.2))
val_store_full = apply_factors(val_pred, store_groups, store_factors)
l9_factors = fit_group_factors(val_store_full, val_true, l9_groups, k=L9_CAL_K, low=0.75, high=1.30)
val_l3_l9_full = apply_factors(val_store_full, l9_groups, l9_factors)
series_factors = fit_series_factors(val_l3_l9_full, val_true, k=SERIES_CAL_K, low=0.5, high=2.0)

test_store_aligned = apply_factors(test_pred, store_groups, store_factors)
test_l3_l9_aligned = apply_factors(test_store_aligned, l9_groups, l9_factors)

test_variants = {
    "BU": test_pred,
    "STORE_CAL": test_store_aligned,
    "HIER_L3_L9_CAL": test_l3_l9_aligned,
    "GLOBAL_CAL": test_pred * global_factor,
    "SERIES_CAL": apply_series_factors(test_l3_l9_aligned, series_factors),
}
test_bu = test_variants["BU"]
test_store = test_variants["STORE_CAL"]
test_hier_l3_l9 = test_variants["HIER_L3_L9_CAL"]
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
    for tag in ["BU", "STORE_CAL", "HIER_L3_L9_CAL", "GLOBAL_CAL", "SERIES_CAL"]
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
    y_pred_hier_l3_l9_cal=test_hier_l3_l9,
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
            "l9_cal_k": L9_CAL_K,
            "hierarchical_loss_weight": HIERARCHICAL_LOSS_WEIGHT,
            "zero_loss_weight": ZERO_LOSS_WEIGHT,
            "pos_loss_weight": POS_LOSS_WEIGHT,
            "filter_prelaunch": FILTER_PRELAUNCH,
            "best_metric": BEST_METRIC,
            "forecast_split": FORECAST_SPLIT,
            "future_price_known": False,
            "feature_version": "v3_gemini_update_dynamic_vsn_decoupled_hurdle",
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
            "l9_cal_k": L9_CAL_K,
            "hierarchical_loss_weight": HIERARCHICAL_LOSS_WEIGHT,
            "zero_loss_weight": ZERO_LOSS_WEIGHT,
            "pos_loss_weight": POS_LOSS_WEIGHT,
            "filter_prelaunch": FILTER_PRELAUNCH,
            "calibration_days": CALIBRATION_DAYS,
            "best_metric": BEST_METRIC,
            "forecast_split": FORECAST_SPLIT,
            "future_price_known": False,
            "feature_version": "v3_gemini_update_dynamic_vsn_decoupled_hurdle",
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
plt.close(fig)

# %% [markdown]
# ## 10. Dynamic Feature Importance & Temporal Attention Extraction from Upgraded TFT

# %%
past_weight_sum = torch.zeros(len(MODEL_PAST_COLS), device=DEVICE)
future_weight_sum = torch.zeros(len(FUT_COLS), device=DEVICE)
attn_horizon_sum = torch.zeros((H, SEQ + H), device=DEVICE)
weight_rows = 0
with torch.no_grad():
    for past, future, static, _, _, _, _ in val_loader:
        past = past.to(DEVICE, non_blocking=True)
        future = future.to(DEVICE, non_blocking=True)
        static = static.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            _, _, _, aux = model(past, future, static, return_aux=True)
        past_weight_sum += aux["past_weights"].float().mean(dim=1).sum(dim=0)
        future_weight_sum += aux["future_weights"].float().mean(dim=1).sum(dim=0)
        attn_horizon_sum += aux["attn_weights"][:, -H:, :].float().sum(dim=0)
        weight_rows += static.size(0)

past_weights = (past_weight_sum / max(weight_rows, 1)).cpu().numpy()
future_weights = (future_weight_sum / max(weight_rows, 1)).cpu().numpy()
mean_attn_map = (attn_horizon_sum / max(weight_rows, 1)).cpu().numpy()

importance_df = pd.concat([
    pd.DataFrame({"feature": MODEL_PAST_COLS, "weight": past_weights, "kind": "past"}),
    pd.DataFrame({"feature": FUT_COLS, "weight": future_weights, "kind": "future"}),
], ignore_index=True).sort_values("weight", ascending=False)

print(importance_df.head(25).to_string(index=False))
importance_df.to_csv(OUT_DIR / "feature_importance_test.csv", index=False)
np.save(OUT_DIR / "attention_map_test.npy", mean_attn_map)

# %% [markdown]
# ## 11. Baseline configuration and dependencies
#
# Cấu hình các mô hình đối chuẩn (ARIMA_ROBUST, PROPHET, LSTM) và hiệu chỉnh phân cấp `TFT_L12_SHRINK`.

# %%
import importlib.util
import importlib.metadata
import logging
from joblib import Parallel, delayed

required_baseline_vars = [
    "SALES", "FEAT", "FEATURE_COLS", "train_ds", "val_ds", "test_ds",
    "VAL_EVAL_CONTEXTS", "TEST_EVAL_CONTEXTS", "test_variants", "best_tag",
]
missing_vars = [x for x in required_baseline_vars if x not in globals()]
if missing_vars:
    raise RuntimeError("Chạy các phần 1–10 trước: thiếu " + ", ".join(missing_vars))

BASELINE_MODELS = tuple(
    x.strip().upper()
    for x in os.getenv("BASELINE_MODELS", "LSTM" if SMOKE else "ARIMA,PROPHET,LSTM").split(",")
    if x.strip()
)
if not set(BASELINE_MODELS) <= {"ARIMA", "PROPHET", "LSTM"}:
    raise ValueError("BASELINE_MODELS chỉ nhận ARIMA,PROPHET,LSTM")
for tag, pkg in [("ARIMA", "statsmodels"), ("PROPHET", "prophet")]:
    if tag in BASELINE_MODELS and importlib.util.find_spec(pkg) is None:
        raise ImportError(f"Thiếu {pkg}; chạy %pip install statsmodels prophet joblib")

BASELINE_HISTORY_DAYS = int(os.getenv("BASELINE_HISTORY_DAYS", "0"))
BASELINE_N_JOBS = int(os.getenv("BASELINE_N_JOBS", "2"))
BASELINE_REFIT_TEST = os.getenv("BASELINE_REFIT_TEST", "0") == "1"
ARIMA_ORDER = tuple(int(v) for v in os.getenv("ARIMA_ORDER", "1,1,1").split(","))
ARIMA_MAXITER = int(os.getenv("ARIMA_MAXITER", "200"))
LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", "2" if SMOKE else "15"))
LSTM_HIDDEN = int(os.getenv("LSTM_HIDDEN", str(D_MODEL)))
LSTM_PATIENCE = int(os.getenv("LSTM_PATIENCE", "3"))
LSTM_LR = float(os.getenv("LSTM_LR", "0.001"))

BASELINE_DIR = OUT_DIR / "baselines"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
BASELINE_VAL_TRUE = SALES[:, VAL_START - 1:VAL_END].copy()
BASELINE_TEST_TRUE = SALES[:, TEST_START - 1:TEST_END].copy()
BASELINE_VAL_PRED, BASELINE_TEST_PRED = {}, {}
BASELINE_STATUS = []
print(
    "Baseline models:", BASELINE_MODELS,
    "| series:", n_series,
    "| statistical refit before test:", BASELINE_REFIT_TEST,
)

# %% [markdown]
# ## 12. ARIMA_ROBUST and Prophet local models

# %%
_bcal = pd.read_csv(DATA_ROOT / "calendar.csv")
_bcal["d_num"] = _bcal["d"].str[2:].astype(int)
BASELINE_DATES = pd.to_datetime(_bcal.set_index("d_num").loc[np.arange(1, n_days + 1), "date"]).to_numpy()
del _bcal

PROPHET_REGRESSORS = ("snap", "has_event")
_reg_idx = [FEATURE_COLS.index(x) for x in PROPHET_REGRESSORS]
BASELINE_REG = (FEAT[:, :, _reg_idx] * FEAT_STD[_reg_idx] + FEAT_MEAN[_reg_idx]).astype(np.float32)
BASELINE_REG = np.rint(BASELINE_REG)


def baseline_positive_prediction(values, horizon):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape != (horizon,) or not np.isfinite(values).all():
        raise ValueError("Forecast không đúng shape hoặc có NaN/Inf")
    return np.maximum(values, 0).astype(np.float32)


def _fit_local_baseline_once(
    tag, y, dates, regressors, train_end, val_end, test_end,
    history_days=0, refit_test=False, order=(1, 1, 1), maxiter=200,
):
    from time import perf_counter
    started = perf_counter()
    hv, ht = val_end - train_end, test_end - val_end
    y = np.asarray(y, dtype=np.float64)
    if len(y) != val_end:
        raise ValueError("y phải kết thúc ở validation cutoff, không chứa test")
    details = []

    def predict_from_cutoff(cutoff, steps):
        start = max(0, cutoff - history_days) if history_days else 0
        history = y[start:cutoff]
        if not len(history) or not np.isfinite(history).all():
            raise ValueError("History rỗng hoặc không hữu hạn")
        if np.ptp(history) < 1e-12:
            details.append("constant_history")
            return np.full(steps, history[-1], dtype=np.float64), None
        if tag == "ARIMA":
            from statsmodels.tsa.arima.model import ARIMA
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                fitted = ARIMA(history, order=order).fit(method_kwargs={"maxiter": maxiter})
            details.extend(str(w.message)[:250] for w in captured)
            if not fitted.mle_retvals.get("converged", True):
                raise RuntimeError("ARIMA optimizer did not converge")
            return fitted.forecast(steps), fitted
        if tag != "PROPHET":
            raise ValueError(tag)
        from prophet import Prophet
        frame = pd.DataFrame({"ds": dates[start:cutoff], "y": history})
        future = pd.DataFrame({"ds": dates[cutoff:cutoff + steps]})
        fitted = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=len(history) >= 730,
            daily_seasonality=False,
            uncertainty_samples=0,
            changepoint_prior_scale=0.05,
        )
        for j, name in enumerate(PROPHET_REGRESSORS):
            if np.ptp(regressors[start:cutoff, j]) > 1e-8:
                fitted.add_regressor(name, standardize=False)
                frame[name] = regressors[start:cutoff, j]
                future[name] = regressors[cutoff:cutoff + steps, j]
        fitted.fit(frame)
        return fitted.predict(future)["yhat"].to_numpy(), fitted

    try:
        steps = hv + ht if tag == "PROPHET" and not refit_test else hv
        first, fitted = predict_from_cutoff(train_end, steps)
        vp = first[:hv]
        if refit_test:
            tp, _ = predict_from_cutoff(val_end, ht)
        elif tag == "PROPHET":
            tp = first[hv:hv + ht]
        elif fitted is None:
            tp = np.full(ht, y[train_end - 1])
            details.append("frozen_constant_train_model")
        else:
            updated = fitted.append(y[train_end:val_end], refit=False)
            tp = updated.forecast(ht)
        return (
            baseline_positive_prediction(vp, hv),
            baseline_positive_prediction(tp, ht),
            {"status": "ok", "details": " | ".join(details), "seconds": perf_counter() - started},
        )
    except Exception as exc:
        return (
            None,
            None,
            {"status": "failed", "details": f"{type(exc).__name__}: {exc}", "seconds": perf_counter() - started},
        )


def fit_local_baseline(
    tag, y, dates, regressors, train_end, val_end, test_end,
    history_days=0, refit_test=False, order=(1, 1, 1), maxiter=200,
):
    if tag != "ARIMA":
        return _fit_local_baseline_once(
            tag, y, dates, regressors, train_end, val_end, test_end,
            history_days, refit_test, order, maxiter,
        )
    started = time.time()
    y = np.asarray(y, dtype=np.float64)
    if len(y) != val_end or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("ARIMA history must be finite, nonnegative and stop at val_end")
    attempts = [
        (tuple(order), maxiter),
        (tuple(order), max(500, maxiter)),
        ((1, 0, 0), 500),
        ((0, 1, 1), 500),
        ((0, 1, 0), 100),
    ]
    seen, errors = set(), []
    for candidate, iterations in attempts:
        if (candidate, iterations) in seen:
            continue
        seen.add((candidate, iterations))
        v, t, info = _fit_local_baseline_once(
            tag, y, dates, regressors, train_end, val_end, test_end,
            history_days, refit_test, candidate, iterations,
        )
        if info["status"] == "ok":
            info.update(
                effective_order=str(candidate),
                attempts=len(seen),
                used_fallback=False,
                fallback_method="",
                retry_errors=" | ".join(errors),
                seconds=time.time() - started,
            )
            return v, t, info
        errors.append(f"order={candidate}, maxiter={iterations}: {info['details']}")

    def fallback(cutoff, horizon):
        window = min(28, cutoff)
        if window >= 7:
            window = (window // 7) * 7
            pattern = y[cutoff - window:cutoff].reshape(-1, 7).mean(axis=0)
            return np.tile(pattern, int(np.ceil(horizon / 7)))[:horizon].astype(np.float32)
        return np.full(horizon, y[:cutoff].mean(), dtype=np.float32)

    return (
        fallback(train_end, val_end - train_end),
        fallback(val_end, test_end - val_end),
        dict(
            status="ok",
            details="All ARIMA fits failed; explicit seasonal fallback",
            effective_order="none",
            attempts=len(seen),
            used_fallback=True,
            fallback_method="WEEKDAY_MEAN_4",
            retry_errors=" | ".join(errors),
            seconds=time.time() - started,
        ),
    )


def run_local_panel(tag):
    vp = np.full((n_series, H), np.nan, np.float32)
    tp = np.full_like(vp, np.nan)
    chunk_size = max(8, 4 * max(1, BASELINE_N_JOBS))
    for start in range(0, n_series, chunk_size):
        stop = min(start + chunk_size, n_series)
        results = Parallel(n_jobs=BASELINE_N_JOBS, prefer="threads")(
            delayed(fit_local_baseline)(
                tag, SALES[s, :VAL_END], BASELINE_DATES[:TEST_END],
                BASELINE_REG[s, :TEST_END], TRAIN_END, VAL_END, TEST_END,
                BASELINE_HISTORY_DAYS, BASELINE_REFIT_TEST, ARIMA_ORDER, ARIMA_MAXITER,
            )
            for s in range(start, stop)
        )
        for s, (v, t, status) in zip(range(start, stop), results):
            BASELINE_STATUS.append({"model": tag, "series_id": series_ids[s], **status})
            if v is not None:
                vp[s], tp[s] = v, t
        pd.DataFrame(BASELINE_STATUS).to_csv(BASELINE_DIR / "fit_status.csv", index=False)
        print(f"{tag}: {stop}/{n_series}", flush=True)
    np.savez_compressed(
        BASELINE_DIR / f"{tag.lower()}_predictions.npz",
        ids=np.asarray(series_ids), val_pred=vp, test_pred=tp,
    )
    if not (np.isfinite(vp).all() and np.isfinite(tp).all()):
        print(f"{tag}: có fit thất bại; KHÔNG tính metric bằng cách bỏ chuỗi.")
        return
    output_tag = "ARIMA_ROBUST" if tag == "ARIMA" else tag
    BASELINE_VAL_PRED[output_tag], BASELINE_TEST_PRED[output_tag] = vp, tp


for _tag in BASELINE_MODELS:
    if _tag in {"ARIMA", "PROPHET"}:
        run_local_panel(_tag)

# %% [markdown]
# ## 13. Global encoder-decoder LSTM baseline

# %%
class ExogenousLSTMBaseline(nn.Module):
    def __init__(self, n_past, n_future, cards, hidden=64, dropout=0.1):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, 8) for card in cards])
        static_dim = 8 * len(cards)
        self.encoder = nn.LSTM(n_past + static_dim, hidden, batch_first=True)
        self.decoder = nn.LSTM(n_future + static_dim, hidden, batch_first=True)
        self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, past, future, static):
        context = torch.cat([emb(static[:, j]) for j, emb in enumerate(self.embeddings)], dim=-1)
        enc_input = torch.cat([past, context[:, None].expand(-1, past.size(1), -1)], dim=-1)
        dec_input = torch.cat([future, context[:, None].expand(-1, future.size(1), -1)], dim=-1)
        _, state = self.encoder(enc_input)
        decoded, _ = self.decoder(dec_input, state)
        return F.softplus(self.output(decoded).squeeze(-1))


def train_lstm_baseline():
    seed_everything(SEED)
    net = ExogenousLSTMBaseline(
        len(MODEL_PAST_COLS), len(FUT_COLS), STATIC_CARD, LSTM_HIDDEN, DROPOUT
    ).to(DEVICE)
    loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=USE_AMP
    )
    optimizer_lstm = torch.optim.AdamW(net.parameters(), lr=LSTM_LR, weight_decay=WEIGHT_DECAY)
    best, best_weights, wait = float("inf"), None, 0
    records = []
    for epoch in range(1, LSTM_EPOCHS + 1):
        started = time.time()
        net.train()
        sum_loss, sum_weight = 0.0, 0.0
        for past, future, static, target, weight, _, _ in loader:
            past, future, static, target, weight = [
                x.to(DEVICE, non_blocking=True) for x in (past, future, static, target, weight)
            ]
            optimizer_lstm.zero_grad(set_to_none=True)
            forecast = net(past, future, static)
            loss = (((forecast - target) ** 2) * weight).sum() / weight.sum().clamp_min(1)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer_lstm.step()
            sum_loss += loss.item() * weight.sum().item()
            sum_weight += weight.sum().item()
        vpred, truth, _ = predict_ds(net, val_ds)
        frame, wr = eval_all(truth, vpred, VAL_EVAL_CONTEXTS, "LSTM")
        score = float(frame.loc[frame.level.eq("L12"), "WRMSSE"].iloc[0]) if BEST_METRIC == "l12" else wr
        records.append({
            "epoch": epoch,
            "train_weighted_mse": sum_loss / max(sum_weight, 1),
            "val_wrmsse5": wr,
            "selection_score": score,
            "seconds": time.time() - started,
        })
        pd.DataFrame(records).to_csv(BASELINE_DIR / "lstm_history.csv", index=False)
        print(records[-1], flush=True)
        if score < best - 1e-4:
            best, wait = score, 0
            best_weights = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
        if wait >= LSTM_PATIENCE:
            break
    if best_weights is not None:
        net.load_state_dict(best_weights)
    vp, _, _ = predict_ds(net, val_ds)
    tp, _, _ = predict_ds(net, test_ds)
    np.savez_compressed(
        BASELINE_DIR / "lstm_predictions.npz",
        ids=np.asarray(series_ids), val_pred=vp, test_pred=tp,
    )
    return vp, tp


if "LSTM" in BASELINE_MODELS:
    BASELINE_VAL_PRED["LSTM"], BASELINE_TEST_PRED["LSTM"] = train_lstm_baseline()

# %% [markdown]
# ## 14. Hierarchical Shrinkage Calibration (`TFT_L12_SHRINK` & `TFT_HIER_SHRINK`) & Full Model Comparison

# %%
L12_STORE_TOLERANCE = float(os.getenv("L12_STORE_TOLERANCE", "0.02"))
cal_h = CALIBRATION_DAYS
base_select, _ = eval_all(val_true[:, cal_h:], val_pred[:, cal_h:], CAL_SELECT_EVAL_CONTEXTS, "BU")
base_store = float(base_select.loc[base_select.level.eq("L3"), "WRMSSE"].iloc[0])

# Evaluate shrinkage candidates starting from both BU and HIER_L3_L9_CAL
rows = []
for base_mode, val_cal_base, val_sel_base, test_base in [
    ("BU", val_pred[:, :cal_h], val_pred[:, cal_h:], test_variants["BU"]),
    ("HIER_L3_L9", apply_factors(cal_store_aligned, l9_groups, selection_l9_factors), selection_l3_l9, test_variants["HIER_L3_L9_CAL"]),
]:
    for k in [5.0, 20.0, 100.0]:
        factors = fit_series_factors(val_cal_base, val_true[:, :cal_h], k=k)
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pred = val_sel_base * (1.0 + alpha * (factors - 1.0))[:, None]
            frame, overall = eval_all(val_true[:, cal_h:], pred, CAL_SELECT_EVAL_CONTEXTS, "candidate")
            sku = float(frame.loc[frame.level.eq("L12"), "WRMSSE"].iloc[0])
            store = float(frame.loc[frame.level.eq("L3"), "WRMSSE"].iloc[0])
            rows.append(dict(
                base_mode=base_mode,
                k=k,
                alpha=alpha,
                L12_WRMSSE=sku,
                L3_WRMSSE=store,
                WRMSSE_5levels=overall,
                eligible=store <= base_store * (1.0 + L12_STORE_TOLERANCE) + 1e-12,
            ))

l12_selection = pd.DataFrame(rows)
eligible_df = l12_selection[l12_selection.eligible]
if len(eligible_df) == 0:
    eligible_df = l12_selection
l12_choice = eligible_df.sort_values(["WRMSSE_5levels", "L12_WRMSSE", "alpha", "k"]).iloc[0]

chosen_val_base = val_l3_l9_full if l12_choice.base_mode == "HIER_L3_L9" else val_pred
chosen_test_base = test_variants["HIER_L3_L9_CAL"] if l12_choice.base_mode == "HIER_L3_L9" else test_variants["BU"]
l12_factors = fit_series_factors(chosen_val_base, val_true, k=float(l12_choice.k))
l12_factors = 1.0 + float(l12_choice.alpha) * (l12_factors - 1.0)
l12_test = chosen_test_base * l12_factors[:, None]

BASELINE_TEST_PRED["TFT_L12_SHRINK"] = l12_test
l12_selection.to_csv(BASELINE_DIR / "l12_calibration_selection.csv", index=False)
np.savez_compressed(
    BASELINE_DIR / "l12_shrink_predictions.npz",
    ids=np.asarray(series_ids), test_pred=l12_test, factors=l12_factors,
)
with open(BASELINE_DIR / "l12_calibration_config.json", "w") as f:
    json.dump(
        dict(
            base_mode=str(l12_choice.base_mode),
            alpha=float(l12_choice.alpha),
            k=float(l12_choice.k),
            store_tolerance=L12_STORE_TOLERANCE,
            selection_days=[VAL_START + cal_h, VAL_END],
            selection_metric="WRMSSE_5levels",
        ),
        f,
        indent=2,
    )
print("L12 calibration chosen on validation:", l12_choice.to_dict())

for _tag, _builder in [
    ("NAIVE_7", lambda start: seasonal_naive(SALES, start, 7, H)),
    ("NAIVE_28", lambda start: seasonal_naive(SALES, start, 28, H)),
    ("WEEKDAY_MEAN_4", lambda start: weekday_mean_naive(SALES, start, H, 4)),
]:
    BASELINE_VAL_PRED[_tag] = _builder(VAL_START)
    BASELINE_TEST_PRED[_tag] = _builder(TEST_START)

BASELINE_VAL_PRED["TFT_BU"] = val_pred.copy()
BASELINE_TEST_PRED["TFT_BU"] = test_variants["BU"].copy()
BASELINE_TEST_PRED["TFT_HIER_L3_L9_CAL"] = test_variants["HIER_L3_L9_CAL"].copy()
if best_tag != "BU":
    BASELINE_TEST_PRED["TFT_SELECTED_" + best_tag] = test_variants[best_tag].copy()

comparison_frames = []
for split, truth, predictions, contexts in [
    ("validation", BASELINE_VAL_TRUE, BASELINE_VAL_PRED, VAL_EVAL_CONTEXTS),
    ("test", BASELINE_TEST_TRUE, BASELINE_TEST_PRED, TEST_EVAL_CONTEXTS),
]:
    for tag, pred in predictions.items():
        frame, _ = eval_all(truth, pred, contexts, tag)
        frame.insert(0, "split", split)
        frame["MAPE_nonzero_pct"] = 100.0 * frame["MAPE"]
        frame["WMAPE_pct"] = 100.0 * frame["WMAPE"]
        frame["aggregation_scope"] = "full_M5" if n_series == 30490 else "selected_series_only"
        comparison_frames.append(frame)

comparison_df = pd.concat(comparison_frames, ignore_index=True)
comparison_df.to_csv(BASELINE_DIR / "comparison_all_levels.csv", index=False)
main_table = comparison_df[comparison_df.level.isin(["L12", "L3"])][
    ["split", "model", "level", "n_keys", "RMSE", "MAE", "MAPE_nonzero_pct", "WMAPE_pct", "WRMSSE", "Bias", "aggregation_scope"]
]
main_table.to_csv(BASELINE_DIR / "comparison_sku_store.csv", index=False)
print(main_table.to_string(index=False))

summary_baselines = (
    comparison_df.groupby(["split", "model"], sort=False)["WRMSSE"]
    .mean()
    .reset_index(name="WRMSSE_5levels")
)
summary_baselines.to_csv(BASELINE_DIR / "comparison_summary.csv", index=False)
print("\nSummary WRMSSE-5 across models:")
print(summary_baselines.to_string(index=False))
