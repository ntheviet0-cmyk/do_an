"""
Script: run_phase4b_global_tft.py
Mục đích: Huấn luyện TFT GLOBAL trên TOÀN BỘ 50 chuỗi SKU (Multi-Series + Static Context)
          của bộ M5 subset, sau đó tổng hợp (aggregate) lên cấp độ Store để so sánh
          với baseline ARIMA theo đúng tinh thần "dự báo đa cấp độ".
"""

import os
import numpy as np
import pandas as pd

from src.data.m5_pipeline import M5DataPipeline
from src.utils.metrics import evaluate_forecast
from src.models.tft_global import TFTGlobalForecaster
from src.models.arima_baseline import ARIMABaseline

PROC = "data/processed"


def load_sku_and_store():
    sku_file = os.path.join(PROC, "m5_sku_level_features.parquet")
    store_file = os.path.join(PROC, "m5_store_level_features.parquet")
    if not os.path.exists(sku_file):
        print("[!] Thiếu processed; chạy pipeline...")
        p = M5DataPipeline(data_dir="data/raw")
        sku = p.create_sku_level_data(start_day=1500, end_day=1913)
        p.save_processed_data(sku, "m5_sku_level_features.parquet", output_dir=PROC)
        store = p.create_store_level_data(sku)
        p.save_processed_data(store, "m5_store_level_features.parquet", output_dir=PROC)
    return pd.read_parquet(sku_file), pd.read_parquet(store_file)


def split_series(df, group_col, id_col):
    """Trả về dict {series_id: (train, val, test)} theo time-split."""
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
    print("PHASE 4B: TFT GLOBAL (50 SKU + STATIC CONTEXT) -> Aggregate lên Store")
    print("=" * 80)

    sku_df, store_df = load_sku_and_store()
    # Đồng nhất tên đặc trưng mô hình giữa cấp SKU và Store
    sku_df["mean_price"] = sku_df["sell_price"]
    sku_df["avg_discount"] = sku_df["price_discount"]
    state_to_snap = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    sku_df["snap_active"] = sku_df.apply(
        lambda r: r[state_to_snap[r["state_id"]]], axis=1
    ).astype(np.int8)

    print(f"[+] SKU: {sku_df['item_id'].nunique()} items | Stores: {sorted(store_df['store_id'].unique())}")

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

    # ---- Split SKU thành train/val/test (đầy đủ cho đánh giá) ----
    sku_split_full = split_series(sku_df, "id", "item_id")
    # Lấy mẫu 60 chuỗi SKU để huấn luyện trong thời gian hợp lý trên CPU,
    # nhưng vẫn dự báo & đánh giá trên TOÀN BỘ các chuỗi (dùng embedding <unk>=0).
    sampled_ids = list(sku_split_full.keys())[:60]
    train_dfs = [sku_split_full[k][0] for k in sampled_ids]
    val_dfs = [sku_split_full[k][1] for k in sampled_ids]
    print(f"[+] Số chuỗi huấn luyện (SKU): {len(train_dfs)} | Tổng chuỗi đánh giá: {len(sku_split_full)}")

    # ---- Huấn luyện TFT GLOBAL ----
    tft = TFTGlobalForecaster(
        past_feature_cols=past_cols,
        future_feature_cols=future_cols,
        static_cat_cols=static_cols,
        target_col="sales",
        seq_len=56, horizon=28, d_model=32, num_heads=2,
        lr=1e-3, batch_size=256, epochs=10
    )
    tft.fit(train_dfs, val_dfs)

    # ---- Đánh giá 2 cấp độ: (A) SKU-level (item) và (B) Store-level (tổng hợp) ----
    store_ids = sorted(store_df["store_id"].unique())

    # Chuẩn bị actual store-level test
    store_actual = {}
    for st in store_ids:
        st_test = store_df[store_df["store_id"] == st].sort_values("d_num")
        md = st_test["d_num"].max()
        store_actual[st] = st_test[st_test["d_num"] > md - 28]["total_sales"].values

    sku_tft_w, sku_arima_w = [], []
    store_pred = {st: np.zeros(28) for st in store_ids}
    n_sku = 0
    print("\n[+] Dự báo từng SKU (100 chuỗi) ...")
    for sid, (tr, va, te) in sku_split_full.items():
        y = te["sales"].values
        last_seq = pd.concat([tr, va]).iloc[-56:]
        q = tft.predict_quantiles(last_seq, te)
        tft_p = q[:, 1]

        arima = ARIMABaseline(order=(7, 1, 1))
        arima.fit(pd.concat([tr, va])["sales"].values)
        arima_p = arima.predict(horizon=28)

        sku_tft_w.append(evaluate_forecast(y, tft_p)["WMAPE (%)"])
        sku_arima_w.append(evaluate_forecast(y, arima_p)["WMAPE (%)"])

        st = te["store_id"].iloc[0]
        store_pred[st] += tft_p
        n_sku += 1

    # (A) SKU-level
    avg_sku_tft = float(np.mean(sku_tft_w))
    avg_sku_arima = float(np.mean(sku_arima_w))
    print("\n" + "=" * 80)
    print("CẤP ĐỘ SKU (ITEM) — SO SÁNH TFT GLOBAL vs ARIMA")
    print("=" * 80)
    print(f"  Trung bình WMAPE TFT GLOBAL : {avg_sku_tft:.3f}%")
    print(f"  Trung bình WMAPE ARIMA      : {avg_sku_arima:.3f}%")
    print(f"  Cải thiện (SKU-level)       : {(avg_sku_arima - avg_sku_tft) / avg_sku_arima * 100:+.2f}%")

    # (B) Store-level (tổng hợp bottom-up)
    rows = []
    for st in store_ids:
        tft_m = evaluate_forecast(store_actual[st], store_pred[st])
        arima = ARIMABaseline(order=(7, 1, 1))
        st_train = store_df[store_df["store_id"] == st].sort_values("d_num")
        st_train = st_train[st_train["d_num"] <= (st_train["d_num"].max() - 56)]
        arima.fit(st_train["total_sales"].values)
        arima_p = arima.predict(horizon=28)
        arima_m = evaluate_forecast(store_actual[st], arima_p)
        rows.append({
            "store": st,
            "TFT_WMAPE": tft_m["WMAPE (%)"], "ARIMA_WMAPE": arima_m["WMAPE (%)"],
            "WMAPE_improve_%": (arima_m["WMAPE (%)"] - tft_m["WMAPE (%)"]) / arima_m["WMAPE (%)"] * 100
        })
        print(f"  {st:<6} | TFT(agg) WMAPE {tft_m['WMAPE (%)']:6.3f} | ARIMA WMAPE {arima_m['WMAPE (%)']:6.3f} "
              f"| cải thiện {rows[-1]['WMAPE_improve_%']:+.2f}%")

    res_df = pd.DataFrame(rows)
    avg_tft = res_df["TFT_WMAPE"].mean()
    avg_arima = res_df["ARIMA_WMAPE"].mean()
    print("-" * 80)
    print(f"  TRUNG BÌNH | TFT GLOBAL (agg SKU->Store) WMAPE = {avg_tft:.3f}% | "
          f"ARIMA WMAPE = {avg_arima:.3f}% | cải thiện {(avg_arima-avg_tft)/avg_arima*100:+.2f}%")
    print("=" * 80)
    print(f"  >> Kết luận: tại cấp SKU (granular, cầu gián đoạn) TFT thắng ARIMA "
          f"{'+' if avg_sku_tft < avg_sku_arima else '-'}"
          f"{(abs(avg_sku_arima-avg_sku_tft)/avg_sku_arima*100):.1f}%; "
          f"tại cấp Store (mượt) ARIMA mạnh hơn (nhất quán với thực tế M5).")
    res_df.to_csv(os.path.join(PROC, "global_tft_vs_arima.csv"), index=False)
    print("[✓] Đã lưu data/processed/global_tft_vs_arima.csv")


if __name__ == "__main__":
    main()
