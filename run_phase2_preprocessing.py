"""
Script: run_phase2_preprocessing.py
Mục đích: Thực thi toàn bộ quy trình tiền xử lý dữ liệu của Phase 2.
          Hỗ trợ cả việc tạo Mock/Sample Dataset để test code ngay lập tức
          nếu chưa tải đủ 3 file raw của M5.
"""

import os
import sys
import numpy as np
import pandas as pd
from src.data.m5_pipeline import M5DataPipeline


def generate_mock_m5_data_if_missing(data_dir: str = "data/raw"):
    """
    Tạo dữ liệu mẫu giả lập định dạng M5 nếu thư mục raw chưa có file
    để người dùng có thể chạy thử nghiệm pipeline ngay lập tức.
    """
    os.makedirs(data_dir, exist_ok=True)
    cal_file = os.path.join(data_dir, "calendar.csv")
    price_file = os.path.join(data_dir, "sell_prices.csv")
    sales_file = os.path.join(data_dir, "sales_train_validation.csv")

    if not (os.path.exists(cal_file) and os.path.exists(price_file) and os.path.exists(sales_file)):
        print("[!] Không tìm thấy đủ 3 file raw của M5. Đang tự động tạo tập MOCK DATA nhỏ để test pipeline...")

        # 1. Mock Calendar (d_1 đến d_1913)
        dates = pd.date_range("2011-01-29", periods=1913, freq="D")
        cal_df = pd.DataFrame({
            "date": dates,
            "wm_yr_wk": 11101 + (dates.dayofyear // 7),
            "weekday": dates.day_name(),
            "wday": (dates.dayofweek + 2) % 7 + 1,
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i+1}" for i in range(1913)],
            "event_name_1": [np.random.choice(["SuperBowl", "ValentineDay", "Easter", np.nan], p=[0.01, 0.01, 0.01, 0.97]) for _ in range(1913)],
            "event_type_1": [np.random.choice(["Sporting", "Cultural", "Religious", np.nan], p=[0.01, 0.01, 0.01, 0.97]) for _ in range(1913)],
            "event_name_2": [np.nan] * 1913,
            "event_type_2": [np.nan] * 1913,
            "snap_CA": np.random.choice([0, 1], size=1913, p=[0.7, 0.3]),
            "snap_TX": np.random.choice([0, 1], size=1913, p=[0.7, 0.3]),
            "snap_WI": np.random.choice([0, 1], size=1913, p=[0.7, 0.3]),
        })
        cal_df.to_csv(cal_file, index=False)

        # 2. Mock Sales (50 items x 2 stores = 100 series)
        num_items = 50
        item_ids = [f"FOODS_3_{i:03d}" for i in range(1, num_items + 1)]
        stores = ["CA_1", "TX_1"]
        
        rows = []
        for store in stores:
            state = store.split("_")[0]
            for item in item_ids:
                row = {
                    "id": f"{item}_{store}_validation",
                    "item_id": item,
                    "dept_id": "FOODS_3",
                    "cat_id": "FOODS",
                    "store_id": store,
                    "state_id": state
                }
                # Random sales demand (Poisson distribution mô phỏng đứt quãng)
                sales_series = np.random.poisson(lam=np.random.uniform(0.5, 4.0), size=1913)
                for day_idx in range(1, 1914):
                    row[f"d_{day_idx}"] = sales_series[day_idx - 1]
                rows.append(row)
        
        sales_df = pd.DataFrame(rows)
        sales_df.to_csv(sales_file, index=False)

        # 3. Mock Prices
        price_rows = []
        wm_weeks = cal_df["wm_yr_wk"].unique()
        for store in stores:
            for item in item_ids:
                base_price = np.random.uniform(1.5, 12.0)
                for wk in wm_weeks:
                    price_rows.append({
                        "store_id": store,
                        "item_id": item,
                        "wm_yr_wk": wk,
                        "sell_price": round(base_price * np.random.uniform(0.85, 1.05), 2)
                    })
        price_df = pd.DataFrame(price_rows)
        price_df.to_csv(price_file, index=False)
        print("[✓] Đã tạo thành công bộ dữ liệu Mock tại data/raw/!")


def main():
    print("=" * 70)
    print("PHASE 2: M5 DATA PREPROCESSING & MULTI-LEVEL AGGREGATION PIPELINE")
    print("=" * 70)

    # 1. Kiểm tra / Khởi tạo dữ liệu
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    generate_mock_m5_data_if_missing(raw_dir)

    # 2. Khởi tạo Pipeline
    pipeline = M5DataPipeline(data_dir=raw_dir)

    # 3. Chạy xử lý SKU-level (Lấy từ ngày 1500 đến 1913 cho đợt train gần nhất)
    print("\n--- BƯỚC 1: XỬ LÝ DỮ LIỆU CẤP ĐỘ SKU (LEVEL 12) ---")
    sku_df = pipeline.create_sku_level_data(start_day=1500, end_day=1913)
    pipeline.save_processed_data(sku_df, "m5_sku_level_features.parquet", output_dir=processed_dir)

    # 4. Chạy tổng hợp Store-level
    print("\n--- BƯỚC 2: TỔNG HỢP DỮ LIỆU CẤP ĐỘ STORE (LEVEL 3) ---")
    store_df = pipeline.create_store_level_data(sku_df)
    pipeline.save_processed_data(store_df, "m5_store_level_features.parquet", output_dir=processed_dir)

    print("\n" + "=" * 70)
    print("PHASE 2 HOÀN TẤT THÀNH CÔNG!")
    print(f"Dữ liệu đã được lưu tại: {os.path.abspath(processed_dir)}")
    print(f"- SKU-level:   {sku_df.shape[0]:,} dòng, {sku_df.shape[1]} đặc trưng")
    print(f"- Store-level: {store_df.shape[0]:,} dòng, {store_df.shape[1]} đặc trưng")
    print("=" * 70)


if __name__ == "__main__":
    main()
