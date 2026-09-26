"""
Ablation: so sánh độ dài cửa sổ quá khứ seq_len ∈ {28, 56, 84} cho TFT.
- Cùng dữ liệu store-level, cùng split, cùng horizon=28, cùng features.
- Mô hình thu nhỏ (d_model=16, epochs=10), cố định seed → so sánh công bằng, chạy nhanh.
- Nhận seq_len từ argv, ghi kết quả append vào data/processed/ablation_window.csv
Usage: python scripts/ablate_window.py 56
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import torch

from src.utils.metrics import evaluate_forecast
from src.models.tft_model import TFTForecaster

SEED = 42
EPOCHS = 10


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def main():
    seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 56
    set_seed(SEED)

    store_df = pd.read_parquet("data/processed/m5_store_level_features.parquet")
    target_store = store_df["store_id"].unique()[0]
    df_store = store_df[store_df["store_id"] == target_store].sort_values("d_num").reset_index(drop=True)

    past_feature_cols = ["total_sales", "sales_lag_28", "sales_lag_35",
                         "sales_rolling_mean_7", "sales_rolling_mean_28"]
    future_feature_cols = ["mean_price", "avg_discount", "is_weekend", "sin_wday",
                           "cos_wday", "sin_month", "cos_month", "snap_active",
                           "has_event", "event_count", "evt_Cultural", "evt_Religious",
                           "evt_Sporting", "evt_National", "days_to_next_event",
                           "days_since_last_event"]

    max_d = df_store["d_num"].max()
    test_split_d = max_d - 28
    val_split_d = test_split_d - 28
    train_df = df_store[df_store["d_num"] <= val_split_d].copy()
    val_df = df_store[(df_store["d_num"] > val_split_d) & (df_store["d_num"] <= test_split_d)].copy()
    test_df = df_store[df_store["d_num"] > test_split_d].copy()
    y_test = test_df["total_sales"].values

    tft = TFTForecaster(
        past_feature_cols=past_feature_cols,
        future_feature_cols=future_feature_cols,
        target_col="total_sales",
        seq_len=seq_len,
        horizon=28,
        d_model=16,
        num_heads=2,
        quantiles=[0.1, 0.5, 0.9],
        lr=1e-3,
        batch_size=64,
        epochs=EPOCHS,
    )
    tft.fit(train_df, val_df)

    last_seq_df = pd.concat([train_df, val_df]).iloc[-seq_len:]
    q_preds, _ = tft.predict_quantiles(last_seq_df, test_df)
    p50 = q_preds[:, 1]
    metrics = evaluate_forecast(y_test, p50)

    wmape = None
    for k, v in metrics.items():
        if "wmape" in k.lower():
            wmape = float(v)
    print(f"[seq_len={seq_len}] metrics={metrics} -> WMAPE={wmape}")

    out = "data/processed/ablation_window.csv"
    row = pd.DataFrame([{"seq_len": seq_len, "wmape": wmape,
                         "d_model": 16, "epochs": EPOCHS, "seed": SEED}])
    if os.path.exists(out):
        old = pd.read_csv(out)
        old = old[old["seq_len"] != seq_len]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(out, index=False)
    print(f"[✓] Ghi {out}")


if __name__ == "__main__":
    main()
