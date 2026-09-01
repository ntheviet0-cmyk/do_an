"""
Demo: Thế nào là DATA LEAK trong time series & cách phát hiện trên dữ liệu thật.
Chạy: python scripts/demo_leakage.py
"""
import numpy as np
import pandas as pd

df = pd.read_parquet("data/processed/m5_store_level_features.parquet")
df = df[df["store_id"] == "CA_1"].sort_values("d_num").reset_index(drop=True)
y = df["total_sales"].values
print("=== 5 dòng đầu tiên: total_sales (target) ===")
print(df[["d_num", "total_sales", "sales_lag_28"]].head(5).to_string(index=False))

# ---------------------------------------------------------------
# 1) FEATURE CHUẨN (không leak): sales_lag_28 đã shift(28) -> lấy quá khứ
# 2) FEATURE BỊ LEAK: lấy luôn doanh số TƯƠNG LAI (shift âm)
# ---------------------------------------------------------------
df["LEAK_future28"] = df["total_sales"].shift(-28)        # ngày t biết doanh số ngày t+28
df["GOOD_lag28"] = df["total_sales"].shift(28)           # ngày t biết doanh số ngày t-28

# ---------------------------------------------------------------
# CÁCH PHÁT HIỆN: với mỗi feature, đo tương quan với target TƯƠNG LAI
#   - feature chuẩn: tương quan với tương lai phải THẤP
#   - feature leak : tương quan với tương lai phải ~1.0 (như biết trước)
# ---------------------------------------------------------------
def leak_score(feature, target, horizons=(1, 7, 28)):
    """Điểm leak = trị tuyệt đối tương quan lớn nhất giữa feature[t] và target[t+h]."""
    a = feature.values.astype(float)
    b = target.values.astype(float)
    best = 0.0
    for h in horizons:
        x = a[:-h]
        y = b[h:]
        m = ~np.isnan(x)
        if m.sum() < 10:
            continue
        c = np.corrcoef(x[m], y[m])[0, 1]
        if not np.isnan(c):
            best = max(best, abs(c))
    return float(best)

print("\n=== ĐIỂM LEAK (|tương quan| lớn nhất của feature[t] vs target[t+1,t+7,t+28]) ===")
print(f"  sales_lag_28 (chuẩn)   : {leak_score(df['sales_lag_28'], df['total_sales']):.3f}  -> thấp, AN TOÀN")
print(f"  LEAK_future28 (giả)    : {leak_score(df['LEAK_future28'], df['total_sales']):.3f}  -> ~1.0, PHÁT HIỆN LEAK!")

# ---------------------------------------------------------------
# Trường hợp tinh vi: rolling mean lấy CẢ tương lai (shift âm)
# ---------------------------------------------------------------
df["LEAK_roll7_wrong"] = df["total_sales"].shift(-6).rolling(7).mean()  # gồm t..t+6 (có tương lai)
df["GOOD_roll7_right"] = df["total_sales"].shift(28).rolling(7).mean()  # chỉ quá khứ (t-34..t-28)
print(f"  LEAK_roll7_wrong       : {leak_score(df['LEAK_roll7_wrong'], df['total_sales']):.3f}  -> cao, LEAK")
print(f"  GOOD_roll7_right       : {leak_score(df['GOOD_roll7_right'], df['total_sales']):.3f}  -> thấp, AN TOÀN")

print("\n=== QUY TẮC VÀNG ===")
print("  Neu feature[t] tuong quan manh voi target tai VI TRI SAU DO (t+1, t+28...)")
print("  => no dang 'chua' thong tin tuong lai => DAT LEAK.")
