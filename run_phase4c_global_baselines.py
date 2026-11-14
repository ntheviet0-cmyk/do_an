"""
Script: run_phase4c_global_baselines.py
Mục đích: Huấn luyện song song 3 mô hình ARIMA, LSTM Global, TFT Global trên
           100 chuỗi SKU (50 items × 2 stores), đánh giá RMSE/MAE/MAPE/WMAPE
           trên 3 cấp độ tổng hợp (SKU, Store, National).
"""

import os
import numpy as np
import pandas as pd

from src.data.m5_pipeline import M5DataPipeline
from src.utils.metrics import evaluate_forecast
from src.utils.multi_level_evaluation import evaluate_hierarchical
from src.models.arima_baseline import ARIMABaseline
from src.models.lstm_global import LSTMGlobalForecaster
from src.models.tft_global import TFTGlobalForecaster

PROC = "data/processed"


def load_data():
    sku_file = os.path.join(PROC, "m5_sku_level_features.parquet")
    if not os.path.exists(sku_file):
        print("[!] Thiếu processed; chạy pipeline Phase 2...")
        p = M5DataPipeline(data_dir="data/raw")
        sku = p.create_sku_level_data(start_day=1500, end_day=1913)
        p.save_processed_data(sku, "m5_sku_level_features.parquet", output_dir=PROC)
    return pd.read_parquet(sku_file)


def split_series(df, group_col):
    out = {}
    for sid, g in df.groupby(group_col):
        g = g.sort_values("d_num").reset_index(drop=True)
        max_d = g["d_num"].max()
        test_split = max_d - 28
        val_split = test_split - 28
        out[sid] = (
            g[g["d_num"] <= val_split].copy(),
            g[(g["d_num"] > val_split) & (g["d_num"] <= test_split)].copy(),
            g[g["d_num"] > test_split].copy(),
        )
    return out


def main():
    print("=" * 80)
    print("PHASE 4C: SO SÁNH 3 MÔ HÌNH ARIMA — LSTM GLOBAL — TFT GLOBAL (100 SKU)")
    print("=" * 80)

    sku_df = load_data()
    sku_df["snap_active"] = sku_df.apply(
        lambda r: r[f"snap_{r['state_id']}"], axis=1
    ).astype(np.int8)
    sku_df["mean_price"] = sku_df["sell_price"]
    sku_df["avg_discount"] = sku_df["price_discount"]

    print(f"[+] SKU: {sku_df['item_id'].nunique()} items × {sku_df['store_id'].nunique()} stores = {sku_df['id'].nunique()} chuỗi")

    past_cols = ["sales", "sales_lag_28", "sales_lag_35",
                 "sales_rolling_mean_7", "sales_rolling_mean_28",
                 "sell_price_lag_28", "price_discount_lag_28",
                 "price_momentum_lag_28", "sell_price_rolling_mean_7"]
    # PHUONG AN A: FUTURE chi lich/SNAP/su kien (gia la unknown future)
    future_cols = [
        "is_weekend",
        "sin_wday", "cos_wday", "sin_month", "cos_month",
        "snap_active", "has_event", "event_count",
        "evt_Cultural", "evt_Religious", "evt_Sporting", "evt_National",
        "days_to_next_event", "days_since_last_event"
    ]
    static_cols = ["store_id", "state_id", "cat_id", "dept_id", "item_id"]

    sku_split = split_series(sku_df, "id")
    all_ids = list(sku_split.keys())
    train_dfs = [sku_split[k][0] for k in all_ids]
    print(f"[+] Tổng chuỗi: {len(all_ids)} | Train windows mỗi chuỗi: ~{len(train_dfs[0])} ngày")

    predictions = {"ARIMA": {}, "LSTM_GLOBAL": {}, "TFT_GLOBAL": {}}

    # 1. ARIMA
    print("\n[1/3] Đang huấn luyện ARIMA(7,1,1) trên 100 SKU...")
    for sid in all_ids:
        tr, va, te = sku_split[sid]
        arima = ARIMABaseline(order=(7, 1, 1))
        arima.fit(pd.concat([tr, va])["sales"].values)
        pred = arima.predict(horizon=28)
        predictions["ARIMA"][sid] = (te["sales"].values, pred)
    print("  ✓ ARIMA hoàn thành!")

    # 2. LSTM Global
    print("\n[2/3] Đang huấn luyện LSTM Global trên 100 SKU...")
    lstm = LSTMGlobalForecaster(
        past_feature_cols=past_cols,
        static_cat_cols=static_cols,
        target_col="sales",
        seq_len=56, horizon=28,
        hidden_dim=32, num_layers=1,
        lr=2e-3, batch_size=64, epochs=15
    )
    lstm.fit(train_dfs)

    print("  Dự báo 100 SKU...")
    for sid in all_ids:
        tr, va, te = sku_split[sid]
        last_seq = pd.concat([tr, va]).iloc[-56:]
        static_row = te.head(1)
        pred = lstm.predict(last_seq, static_row)
        predictions["LSTM_GLOBAL"][sid] = (te["sales"].values, pred)
    print("  ✓ LSTM Global hoàn thành!")

    # 3. TFT Global
    print("\n[3/3] Đang huấn luyện TFT Global trên 100 SKU...")
    tft = TFTGlobalForecaster(
        past_feature_cols=past_cols,
        future_feature_cols=future_cols,
        static_cat_cols=static_cols,
        target_col="sales",
        seq_len=56, horizon=28,
        d_model=16, num_heads=2,
        lr=1e-3, batch_size=64, epochs=10
    )
    tft.fit(train_dfs)

    print("  Dự báo 100 SKU...")
    for sid in all_ids:
        tr, va, te = sku_split[sid]
        last_seq = pd.concat([tr, va]).iloc[-56:]
        q = tft.predict_quantiles(last_seq, te)
        pred = q[:, 1]
        predictions["TFT_GLOBAL"][sid] = (te["sales"].values, pred)
    print("  ✓ TFT Global hoàn thành!")

    # Đánh giá đa cấp độ
    print("\n" + "=" * 80)
    print("ĐÁNH GIÁ ĐA CẤP ĐỘ (SKU / STORE / NATIONAL)")
    print("=" * 80)

    all_results = []
    for model_name, preds in predictions.items():
        y_true_dict = {k: v[0] for k, v in preds.items()}
        y_pred_dict = {k: v[1] for k, v in preds.items()}
        df_metrics = evaluate_hierarchical(y_true_dict, y_pred_dict, sku_df, model_name)
        all_results.append(df_metrics)

    results_df = pd.concat(all_results, ignore_index=True)

    for level in ["SKU", "Store", "National"]:
        print(f"\n--- {level.upper()} LEVEL ---")
        level_df = results_df[results_df["Level"] == level].copy()
        level_df = level_df.drop(columns=["Level"])
        level_df = level_df.round(3)
        print(level_df.to_string(index=False))

    out_path = os.path.join(PROC, "3model_hierarchical_metrics.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n[✓] Đã lưu: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
