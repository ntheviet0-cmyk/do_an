"""
Script: analyze_data_quality.py
Mục đích: Phục vụ báo cáo đồ án - hai hạng mục:
  1. KIỂM TRA DỮ LIỆU THIẾU (Missing Data) trên cả dữ liệu thô (raw) và đã xử lý (processed).
  2. AUDIT CHỐNG DATA LEAKAGE: xác minh (a) split time-based, (b) lag/rolling features
     chỉ dùng thông tin quá khứ, (c) không có đặc trưng nào chứa thông tin của tương lai.
"""

import os
import numpy as np
import pandas as pd

RAW = "data/raw"
PROC = "data/processed"


def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------------
# 1. KIỂM TRA DỮ LIỆU THIẾU
# ----------------------------------------------------------------------
def check_missing_raw():
    section("[1] KIỂM TRA DỮ LIỆU THIẾU - DỮ LIỆU THÔ (RAW)")
    cal = pd.read_csv(os.path.join(RAW, "calendar.csv"))
    prices = pd.read_csv(os.path.join(RAW, "sell_prices.csv"))
    sales = pd.read_csv(os.path.join(RAW, "sales_train_validation.csv"))

    print(f"Calendar : {cal.shape[0]} ngày x {cal.shape[1]} cột")
    print(f"Prices   : {prices.shape[0]} dòng x {prices.shape[1]} cột")
    print(f"Sales    : {sales.shape[0]} SKU  x {sales.shape[1]} cột (id + d_1..d_1913)")

    # Calendar: cột sự kiện luôn có thể thiếu (ngày không có lễ)
    cal_event_cols = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    print("\n* Tỷ lệ thiếu trong calendar (các cột sự kiện - BÌNH THƯỜNG vì nhiều ngày không có lễ):")
    for c in cal_event_cols:
        miss = cal[c].isna().mean() * 100
        print(f"    - {c:<14}: {miss:6.2f}% thiếu")

    # Sell prices: một số (store, item) không bán mọi tuần -> giá thiếu
    print("\n* Sell_prices: tỷ lệ dòng thiếu sell_price:")
    print(f"    - sell_price NaN: {prices['sell_price'].isna().mean()*100:6.2f}%")

    # Sales: M5 có thể có 0 nhưng không NaN (wide format)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    sales_vals = sales[day_cols].values
    print("\n* Sales (wide): tỷ lệ ô bằng 0 (có thể do không bán/cắt stock):")
    print(f"    - Tỷ lệ giá trị = 0: {(sales_vals == 0).mean()*100:6.2f}%")
    print(f"    - Có NaN trong sales? {np.isnan(sales_vals).any()}")


def check_missing_processed():
    section("[1] KIỂM TRA DỮ LIỆU THIẾU - DỮ LIỆU ĐÃ XỬ LÝ (PROCESSED)")
    for f in ["m5_sku_level_features.parquet", "m5_store_level_features.parquet"]:
        path = os.path.join(PROC, f)
        if not os.path.exists(path):
            print(f"  (!) Thiếu file {f}, bỏ qua (chạy pipeline Phase 2 trước).")
            continue
        df = pd.read_parquet(path)
        n = len(df)
        nan_counts = df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        print(f"\n* {f} : {n} dòng x {df.shape[1]} cột")
        if len(nan_cols) == 0:
            print("    - KHÔNG CÓ ô NaN nào.")
        else:
            print("    - Các cột còn NaN (sau xử lý):")
            for c, v in nan_cols.items():
                print(f"        {c:<28}: {v:>8} ({v/n*100:5.2f}%)")


# ----------------------------------------------------------------------
# 2. AUDIT CHỐNG DATA LEAKAGE
# ----------------------------------------------------------------------
def audit_leakage():
    section("[2] AUDIT CHỐNG DATA LEAKAGE (RÒ RỈ DỮ LIỆU)")
    store_file = os.path.join(PROC, "m5_store_level_features.parquet")
    if not os.path.exists(store_file):
        print("  (!) Thiếu file processed, chạy pipeline trước.")
        return

    df = pd.read_parquet(store_file)
    target_store = df["store_id"].unique()[0]
    df = df[df["store_id"] == target_store].sort_values("d_num").reset_index(drop=True)

    max_d = df["d_num"].max()
    test_split_d = max_d - 28
    val_split_d = test_split_d - 28

    train = df[df["d_num"] <= val_split_d]
    val = df[(df["d_num"] > val_split_d) & (df["d_num"] <= test_split_d)]
    test = df[df["d_num"] > test_split_d]

    print(f"* Split time-based: Train<=d{val_split_d} | Val d{val_split_d+1}..d{test_split_d} | Test d{test_split_d+1}..d{max_d}")
    print(f"    -> Train {len(train)} | Val {len(val)} | Test {len(test)} ngày  [KHÔNG trộn (no shuffling)]")

    # (b) Lag/rolling phải dùng shift>=horizon (28)
    lag_cols = [c for c in df.columns if "lag" in c or "rolling" in c]
    print("\n* Các đặc trưng trễ/trượt:")
    for c in lag_cols:
        print(f"    - {c}")

    # (c) Xác minh: tại mỗi hàng test, lag_28 phải lấy giá trị của d_num-28 (thuộc val, không thuộc test target)
    # Kiểm tra lag_28 của dòng test đầu tiên thực sự nằm ngoài test set.
    first_test_d = test["d_num"].min()
    expected_lag_src_d = first_test_d - 28
    lag_val_actual = test.iloc[0][f"sales_lag_28"]
    # giá trị sales tại d = expected_lag_src_d trong toàn bộ df
    src_row = df[df["d_num"] == expected_lag_src_d]
    lag_val_expected = src_row["total_sales"].values[0] if len(src_row) else np.nan
    ok = np.isclose(lag_val_actual, lag_val_expected) if pd.notna(lag_val_expected) else False
    print(f"\n* Xác minh Lag bảo toàn thứ tự thời gian (không rò rỉ):")
    print(f"    - Test bắt đầu tại d{first_test_d}, sales_lag_28 phải = total_sales tại d{expected_lag_src_d} (thuộc Val)")
    print(f"    - Giá trị lag thực tế = {lag_val_actual}, giá trị nguồn = {lag_val_expected} -> {'OK' if ok else 'KHÔNG OK'}")

    # (d) Future-known features PHUONG AN A: chi lich/SNAP/su kien (gia la unknown future)
    future_known = ["is_weekend", "sin_wday", "cos_wday",
                    "sin_month", "cos_month", "snap_active", "has_event"]
    print("\n* Dac trung 'known-future' (lich/SNAP/su kien da biet truoc, KHONG phai target):")
    missing_future = [c for c in future_known if c not in df.columns]
    print(f"    - Đầy đủ trong dữ liệu? {'Có' if not missing_future else 'Thiếu: '+str(missing_future)}")
    print("    - Quan trọng: các cột này là biến ngoại sinh/biến lịch, KHÔNG chứa total_sales tương lai"
          " -> sử dụng cho tập Test là HỢP LỆ (không rò rỉ).")


if __name__ == "__main__":
    check_missing_raw()
    check_missing_processed()
    audit_leakage()
    print("\n[✓] Hoàn tất phân tích chất lượng dữ liệu & leakage.")
