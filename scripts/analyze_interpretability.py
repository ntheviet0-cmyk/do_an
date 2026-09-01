"""
Script: analyze_interpretability.py
Mục đích: Phân tích khả năng diễn giải của TFT đối với CÁC YẾU TỐ NGOẠI SINH & TÍNH MÙA VỤ.
  - Dùng trọng số VSN (Variable Selection Network) để xếp hạng mức độ quan trọng
    của từng đặc trưng (đặc biệt nhóm tính mùa vụ: thứ, tháng, cuối tuần, lễ, SNAP).
  - Dùng Temporal Attention để xem mô hình tập trung vào các điểm quá khứ nào
    khi dự báo 28 ngày tới.
Kết quả in ra bảng xếp hạng và lưu bảng vào data/processed/feature_importance_table.csv.
"""

import os
import pickle
import numpy as np
import pandas as pd

PKL = "data/processed/tft_results_and_interpretability.pkl"

# Phân nhóm đặc trưng phục vụ báo cáo
SEASONAL = {
    "is_weekend", "sin_wday", "cos_wday", "sin_month", "cos_month",
    "snap_active", "has_event", "event_count",
    "evt_Cultural", "evt_Religious", "evt_Sporting", "evt_National",
    "days_to_next_event", "days_since_last_event",
}
PRICE_PROMO = {"mean_price", "avg_discount"}


def main():
    with open(PKL, "rb") as f:
        data = pickle.load(f)

    future_cols = data["future_features"]
    future_w = data["future_vsn_weights"]      # [T, num_future_vars] (batch đã squeeze)
    past_cols = data["past_features"]
    past_w = data["past_vsn_weights"]          # [seq_len, num_past_vars]
    attn = data["temporal_attention"]          # [T, T]

    # ---- 1. XẾP HẠNG QUAN TRỌNG ĐẶC TRƯNG TỪ VSN ----
    # Trung bình trọng số theo thời gian (mỗi đặc trưng đã được softmax => tổng=1)
    fut_imp = future_w.mean(axis=0)            # [num_future_vars]
    past_imp = past_w.mean(axis=0)             # [num_past_vars]

    fut_rank = pd.DataFrame({
        "feature": future_cols,
        "importance": fut_imp,
        "group": ["Mùa vụ/Ngoại sinh" if c in SEASONAL else ("Giá/Khuyến mãi" if c in PRICE_PROMO else "Khác")
                  for c in future_cols]
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    past_rank = pd.DataFrame({
        "feature": past_cols,
        "importance": past_imp,
        "group": "Lịch sử (past)"
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("=" * 78)
    print("BẢNG XẾP HẠNG QUAN TRỌNG ĐẶC TRƯNG NGOẠI SINH (Future VSN Weights)")
    print("=" * 78)
    print(fut_rank.to_string(index=False))
    print(f"\nTỷ trọng nhóm Future: "
          f"Mùa vụ/Ngoại sinh = {fut_rank.groupby('group')['importance'].sum().get('Mùa vụ/Ngoại sinh',0)*100:.1f}% | "
          f"Giá/Khuyến mãi = {fut_rank.groupby('group')['importance'].sum().get('Giá/Khuyến mãi',0)*100:.1f}%")

    print("\n" + "=" * 78)
    print("BẢNG XẾP HẠNG ĐẶC TRƯNG QUÁ KHỨ (Past VSN Weights)")
    print("=" * 78)
    print(past_rank.to_string(index=False))

    # ---- 2. TEMPORAL ATTENTION: mô hình nhìn vào quá khứ nào? ----
    seq_len = 56
    horizon = 28
    total = seq_len + horizon
    # Query là các vị trí horizon (28 vị trí cuối), trung bình attention lên toàn bộ chuỗi
    horizon_queries = attn[seq_len:total, :]              # [horizon, total]
    attn_on_past = horizon_queries[:, :seq_len].mean(axis=0)   # [seq_len] -> chú ý vào quá khứ
    # Ánh xạ vị trí quá khứ -> số ngày trước (lag): vị trí i tương ứng lag = seq_len - i
    lags = np.arange(seq_len, 0, -1)
    attn_past_df = pd.DataFrame({"lag_days": lags, "attention_weight": attn_on_past})
    attn_past_df = attn_past_df.sort_values("attention_weight", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 78)
    print("TEMPORAL ATTENTION: 10 điểm quá khứ được chú ý nhất khi dự báo tương lai")
    print("=" * 78)
    print(attn_past_df.head(10).to_string(index=False))

    # Tỷ trọng attention vào các vùng trễ chu kỳ (28, 56 ngày)
    def attn_at(lag):
        idx = seq_len - lag
        return attn_on_past[idx] if 0 <= idx < seq_len else 0.0
    print(f"\nChú ý vào lag ≈ 7 ngày (1 tuần)  : {attn_at(7):.4f}")
    print(f"Chú ý vào lag ≈ 28 ngày (4 tuần) : {attn_at(28):.4f}")
    print(f"Chú ý vào lag ≈ 56 ngày (8 tuần) : {attn_at(56):.4f}")

    # Lưu bảng để đưa vào báo cáo
    os.makedirs("data/processed", exist_ok=True)
    fut_rank.to_csv("data/processed/feature_importance_table.csv", index=False)
    attn_past_df.to_csv("data/processed/temporal_attention_profile.csv", index=False)
    print("\n[✓] Đã lưu feature_importance_table.csv & temporal_attention_profile.csv")


if __name__ == "__main__":
    main()
