"""
Script: check_leakage_before_training.py
Chạy TRƯỚC KHI TRAIN để làm "cổng kiểm soát" rò rỉ dữ liệu.
  1. Kiểm tra cấu trúc: sort theo thời gian, split time-based, scaler chỉ fit trên train.
  2. Quét từng cột: đo |tương quan| lớn nhất giữa feature[t] và target[t+h]
     (h = 1, 7, 28). Cột nhóm quá khứ mà điểm cao -> nghi ngờ leak.
  3. Whitelist nhóm known-future (lịch/giá) -> bỏ qua (hợp lệ).
Usage: python scripts/check_leakage_before_training.py
"""
import numpy as np
import pandas as pd

TARGET = "total_sales"
HORIZONS = (1, 7, 28)
THRESHOLD = 0.30

# Nhóm known-future (lấy từ calendar/giá, đã biết trước) -> WHITELIST, không tính leak
WHITELIST_PREFIX = ("sin_", "cos_", "is_weekend", "has_event", "event_", "evt_",
                    "days_to_", "days_since_", "snap_", "mean_price", "avg_discount",
                    "price_", "weekday", "wday", "month", "day")


def leak_score(feature_vals, target_vals, horizons):
    a = np.asarray(feature_vals, dtype=float)
    b = np.asarray(target_vals, dtype=float)
    best = 0.0
    for h in horizons:
        if len(a) - h < 10:
            continue
        x, y = a[:-h], b[h:]
        m = ~np.isnan(x)
        if m.sum() < 10:
            continue
        c = np.corrcoef(x[m], y[m])[0, 1]
        if not np.isnan(c):
            best = max(best, abs(c))
    return float(best)


def main():
    df = pd.read_parquet("data/processed/m5_store_level_features.parquet")
    df = df[df["store_id"] == sorted(df["store_id"].unique())[0]].sort_values("d_num").reset_index(drop=True)
    y = df[TARGET].values

    print("=" * 78)
    print("KIỂM TRA RÒ RỈ DỮ LIỆU TRƯỚC KHI TRAIN")
    print("=" * 78)

    # ---- 1. Kiểm tra cấu trúc ----
    print("\n[1] KIỂM TRA CẤU TRÚC")
    mono = df["d_num"].is_monotonic_increasing
    print(f"  - Dữ liệu đã sắp xếp theo thời gian (d_num tăng dần): {'OK' if mono else 'LỖI'}")
    print("  - Split phải theo thời gian (test = ngày cuối). Hãy đảm bảo code dùng:")
    print("        test = df[df['d_num'] > test_split]   # KHÔNG dùng train_test_split ngẫu nhiên")
    print("  - Scaler: chỉ .fit(train), không .fit(toàn bộ). (Kiểm tra thủ công trong code)")

    # ---- 2. Quét từng cột ----
    print(f"\n[2] QUÉT TỪNG CỘT (|corr| feature[t] vs target[t+{HORIZONS}] > {THRESHOLD} => nghi ngờ)")
    print(f"  {'FEATURE':<28} {'|corr|':>7}  {'NHÓM':<10} KẾT LUẬN")
    print("  " + "-" * 70)
    flagged = []
    for col in df.columns:
        if col in (TARGET, "d_num", "date", "store_id", "state_id", "id"):
            continue
        if col.startswith(WHITELIST_PREFIX) or any(col == w for w in WHITELIST_PREFIX):
            grp = "known-future"
            note = "whitelist (hợp lệ)"
        else:
            grp = "past/lag"
            sc = leak_score(df[col], y, HORIZONS)
            if sc > THRESHOLD:
                note = ">>> NGHI NGỜ LEAK - xem lại công thức"
                flagged.append((col, sc))
            else:
                note = "an toàn"
        sc = leak_score(df[col], y, HORIZONS) if grp == "past/lag" else float("nan")
        print(f"  {col:<28} {('%.3f'%sc if grp=='past/lag' else '  -  '):>7}  {grp:<10} {note}")

    # ---- 3. Kết luận ----
    print("\n[3] KẾT LUẬN")
    if flagged:
        print("  >>> CÓ cột nghi ngờ leak, SỬA trước khi train:")
        for c, s in flagged:
            print(f"      - {c} (điểm {s:.3f}): kiểm tra xem có dùng shift âm / rolling không shift không")
    else:
        print("  >>> KHÔNG phát hiện leak ở nhóm quá khứ. Có thể train.")
    print("=" * 78)


if __name__ == "__main__":
    main()
