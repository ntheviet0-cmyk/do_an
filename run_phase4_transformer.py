"""
Script: run_phase4_transformer.py
Mục đích: Huấn luyện mô hình Đề xuất Temporal Fusion Transformer (TFT) trên M5 Dataset,
          đánh giá độ chính xác (P10, P50, P90) và lưu lại các ma trận trọng số Attention & Feature Importance.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch

from src.data.m5_pipeline import M5DataPipeline
from src.utils.metrics import evaluate_forecast
from src.models.tft_model import TFTForecaster


def run_phase4_experiment():
    print("=" * 80)
    print("PHASE 4: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH ĐỀ XUẤT TEMPORAL FUSION TRANSFORMER (TFT)")
    print("=" * 80)

    # 1. Load dữ liệu đã tiền xử lý
    processed_dir = "data/processed"
    store_file = os.path.join(processed_dir, "m5_store_level_features.parquet")

    if not os.path.exists(store_file):
        print("[!] Không tìm thấy dữ liệu parquet. Đang tự động chạy pipeline Phase 2...")
        pipeline = M5DataPipeline(data_dir="data/raw")
        sku_df = pipeline.create_sku_level_data(start_day=1500, end_day=1913)
        pipeline.save_processed_data(sku_df, "m5_sku_level_features.parquet", output_dir=processed_dir)
        store_df = pipeline.create_store_level_data(sku_df)
        pipeline.save_processed_data(store_df, "m5_store_level_features.parquet", output_dir=processed_dir)
    else:
        print(f"[+] Đang load dữ liệu từ: {store_file}")
        store_df = pd.read_parquet(store_file)

    target_store = store_df["store_id"].unique()[0]
    print(f"\n--- Huấn luyện TFT trên chuỗi Store-Level: {target_store} ---")
    df_store = store_df[store_df["store_id"] == target_store].sort_values("d_num").reset_index(drop=True)

    # 2. Định nghĩa các nhóm đặc trưng cho TFT
    # A. Past Features (PHUONG AN A: gia chi dung ban lagged <= t, khong dung gia tuong lai)
    past_feature_cols = [
        "total_sales",
        "sales_lag_28",
        "sales_lag_35",
        "sales_rolling_mean_7",
        "sales_rolling_mean_28",
        "mean_price_lag_28",
        "avg_discount_lag_28",
        "mean_price_rolling_mean_7",
    ]

    # B. Known-Future Features (chi lich + SNAP + su kien da biet truoc; KHONG co gia)
    future_feature_cols = [
        "is_weekend",
        "sin_wday",
        "cos_wday",
        "sin_month",
        "cos_month",
        "snap_active",
        "has_event",
        "event_count",
        "evt_Cultural",
        "evt_Religious",
        "evt_Sporting",
        "evt_National",
        "days_to_next_event",
        "days_since_last_event"
    ]

    # 3. Train / Validation / Test Split
    test_days = 28
    val_days = 28
    
    max_d = df_store["d_num"].max()
    test_split_d = max_d - test_days
    val_split_d = test_split_d - val_days

    train_df = df_store[df_store["d_num"] <= val_split_d].copy()
    val_df = df_store[(df_store["d_num"] > val_split_d) & (df_store["d_num"] <= test_split_d)].copy()
    test_df = df_store[df_store["d_num"] > test_split_d].copy()

    y_test = test_df["total_sales"].values
    print(f"[-] Số lượng mẫu: Train={len(train_df)} ngày | Val={len(val_df)} ngày | Test={len(test_df)} ngày")
    print(f"[-] Past Features ({len(past_feature_cols)}): {past_feature_cols}")
    print(f"[-] Future Features ({len(future_feature_cols)}): {future_feature_cols}")

    # 4. Khởi tạo và Fit TFTForecaster
    tft = TFTForecaster(
        past_feature_cols=past_feature_cols,
        future_feature_cols=future_feature_cols,
        target_col="total_sales",
        seq_len=56,
        horizon=28,
        d_model=64,
        num_heads=4,
        quantiles=[0.1, 0.5, 0.9],
        lr=1e-3,
        batch_size=32,
        epochs=35
    )

    tft.fit(train_df, val_df)

    # 5. Dự báo trên tập Test
    last_seq_df = pd.concat([train_df, val_df]).iloc[-56:]
    q_preds, interp_dict = tft.predict_quantiles(last_seq_df, test_df)

    p10_preds = q_preds[:, 0]
    p50_preds = q_preds[:, 1] # Median forecast dùng để tính point metrics
    p90_preds = q_preds[:, 2]

    # 6. Đánh giá hiệu năng Median (P50)
    tft_metrics = evaluate_forecast(y_test, p50_preds)

    print("\n" + "=" * 80)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ĐỀ XUẤT TFT (P50 MEDIAN FORECAST vs ACTUAL TEST)")
    print("=" * 80)
    for k, v in tft_metrics.items():
        print(f"  {k:<15}: {v:.4f}")

    # 7. Lưu lại kết quả dự báo và trọng số giải thích cho Phase 5
    export_data = {
        "dates": test_df["date"].values,
        "y_true": y_test,
        "p10": p10_preds,
        "p50": p50_preds,
        "p90": p90_preds,
        "past_features": past_feature_cols,
        "future_features": future_feature_cols,
        "past_vsn_weights": interp_dict["past_vsn_weights"],
        "future_vsn_weights": interp_dict["future_vsn_weights"],
        "temporal_attention": interp_dict["temporal_attention"]
    }

    save_path = os.path.join(processed_dir, "tft_results_and_interpretability.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(export_data, f)

    print(f"\n[✓] Đã lưu kết quả dự báo và trọng số giải thích vào: {save_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_phase4_experiment()
