"""
Script: run_phase3_baselines.py
Mục đích: Huấn luyện và đánh giá các mô hình Baseline (ARIMA, Prophet, LSTM)
          trên tập dữ liệu kiểm thử 28 ngày (M5 Forecasting Horizon).
"""

import os
import numpy as np
import pandas as pd

from src.data.m5_pipeline import M5DataPipeline
from src.utils.metrics import evaluate_forecast
from src.models.arima_baseline import ARIMABaseline
from src.models.prophet_baseline import ProphetBaseline, PROPHET_AVAILABLE
from src.models.lstm_model import LSTMBaseline


def run_phase3_experiment():
    print("=" * 75)
    print("PHASE 3: HUẤN LUYỆN VÀ ĐỐI CHUẨN BASELINE MODELS (ARIMA, PROPHET, LSTM)")
    print("=" * 75)

    # 1. Đọc dữ liệu đã xử lý từ Phase 2
    processed_dir = "data/processed"
    store_file = os.path.join(processed_dir, "m5_store_level_features.parquet")
    sku_file = os.path.join(processed_dir, "m5_sku_level_features.parquet")

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

    # 2. Chuẩn bị tập Train / Val / Test Split (Horizon H = 28 ngày)
    # Test set: 28 ngày cuối cùng (d_1886 đến d_1913)
    # Val set: 28 ngày trước đó (d_1858 đến d_1885)
    # Train set: Các ngày trước d_1858
    target_store = store_df["store_id"].unique()[0]
    print(f"\n--- Thực hiện đánh giá trên chuỗi Store-Level: {target_store} ---")
    
    df_store = store_df[store_df["store_id"] == target_store].sort_values("d_num").reset_index(drop=True)
    
    test_days = 28
    val_days = 28
    
    max_d = df_store["d_num"].max()
    test_split_d = max_d - test_days
    val_split_d = test_split_d - val_days

    train_df = df_store[df_store["d_num"] <= val_split_d].copy()
    val_df = df_store[(df_store["d_num"] > val_split_d) & (df_store["d_num"] <= test_split_d)].copy()
    test_df = df_store[df_store["d_num"] > test_split_d].copy()

    y_test = test_df["total_sales"].values
    print(f"[-] Train set: {len(train_df)} ngày | Val set: {len(val_df)} ngày | Test set: {len(test_df)} ngày")

    results = {}

    # -------------------------------------------------------------
    # 1. BASELINE 1: ARIMA
    # -------------------------------------------------------------
    print("\n[1/3] Đang huấn luyện mô hình ARIMA(7, 1, 1)...")
    try:
        arima = ARIMABaseline(order=(7, 1, 1))
        # Fit trên toàn bộ dữ liệu trước test set (Train + Val)
        train_val_sales = pd.concat([train_df, val_df])["total_sales"].values
        arima.fit(train_val_sales)
        arima_preds = arima.predict(horizon=test_days)
        results["ARIMA (7,1,1)"] = evaluate_forecast(y_test, arima_preds)
        print("  ✓ ARIMA hoàn thành!")
    except Exception as e:
        print(f"  ✗ Lỗi ARIMA: {e}")

    # -------------------------------------------------------------
    # 2. BASELINE 2: Facebook Prophet
    # -------------------------------------------------------------
    print("\n[2/3] Đang huấn luyện mô hình Facebook Prophet (có Exogenous)...")
    if PROPHET_AVAILABLE:
        try:
            # PHUONG AN A: Prophet exog chi lich/SNAP/su kien (gia la unknown future)
            exog_cols = ["is_weekend", "snap_active", "has_event"]
            prophet_model = ProphetBaseline(exogenous_cols=exog_cols)
            
            # Format train dataframe cho Prophet
            train_prophet_df = pd.concat([train_df, val_df]).rename(columns={"total_sales": "sales"})
            prophet_model.fit(train_prophet_df)
            
            prophet_preds = prophet_model.predict(test_df)
            results["Facebook Prophet"] = evaluate_forecast(y_test, prophet_preds)
            print("  ✓ Prophet hoàn thành!")
        except Exception as e:
            print(f"  ✗ Lỗi Prophet: {e}")
    else:
        print("  (-) Bỏ qua Prophet (Chưa cài đặt thư viện 'prophet').")

    # -------------------------------------------------------------
    # 3. BASELINE 3: Multivariate LSTM (PyTorch)
    # -------------------------------------------------------------
    print("\n[3/3] Đang huấn luyện mô hình Multivariate LSTM (PyTorch)...")
    # PHUONG AN A: LSTM baseline chi dung qua khu (gia ban lagged, khong dung gia tuong lai)
    feature_cols = [
        "total_sales", "mean_price_lag_28", "avg_discount_lag_28", "is_weekend",
        "sin_wday", "cos_wday", "sin_month", "cos_month", "snap_active", "has_event",
        "event_count", "evt_Cultural", "evt_Religious", "evt_Sporting", "evt_National",
        "days_to_next_event", "days_since_last_event",
        "sales_lag_28", "sales_rolling_mean_7", "sales_rolling_mean_28"
    ]
    
    lstm_pipeline = LSTMBaseline(
        feature_cols=feature_cols,
        target_col="total_sales",
        seq_len=56,
        horizon=28,
        hidden_dim=64,
        num_layers=2,
        lr=2e-3,
        batch_size=32,
        epochs=40
    )
    
    lstm_pipeline.fit(train_df, val_df)
    
    # Input cho dự báo là 56 ngày gần nhất trước Test set
    last_seq_df = pd.concat([train_df, val_df]).iloc[-56:]
    lstm_preds = lstm_pipeline.predict(last_seq_df)
    results["Multivariate LSTM"] = evaluate_forecast(y_test, lstm_preds)
    print("  ✓ Multivariate LSTM hoàn thành!")

    # -------------------------------------------------------------
    # TỔNG HỢP VÀ SO SÁNH KẾT QUẢ
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("BẢNG SO SÁNH HIỆU SUẤT CÁC MÔ HÌNH BASELINE (TEST HORIZON = 28 DAYS)")
    print("=" * 75)
    
    res_df = pd.DataFrame(results).T
    print(res_df.to_string())
    print("=" * 75)
    
    return res_df


if __name__ == "__main__":
    run_phase3_experiment()
