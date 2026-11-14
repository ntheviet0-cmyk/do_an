"""
Module: m5_pipeline.py
Mục đích: Pipeline tiền xử lý hoàn chỉnh cho dataset Walmart M5 Forecasting.
Bao gồm:
  1. Load và tối ưu hóa 3 bảng dữ liệu (calendar, sell_prices, sales_train_validation).
  2. Melt từ dạng Wide sang Long format.
  3. Xử lý Missing values và làm sạch dữ liệu.
  4. Trích xuất đặc trưng thời gian (Cyclical encoding, sự kiện, SNAP).
  5. Trích xuất đặc trưng giá (Discount, Price momentum, Price rolling stats).
  6. Tạo đặc trưng trễ (Lags) và thống kê trượt (Rolling Stats) CHỐNG RÒ RỈ DỮ LIỆU.
  7. Tổng hợp dữ liệu đa cấp độ (Hierarchical Aggregation: SKU-level và Store-level).
"""

import os
import gc
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from .memory_optimizer import reduce_mem_usage


class M5DataPipeline:
    def __init__(self, data_dir: str = "data/raw"):
        """
        Khởi tạo Pipeline với đường dẫn thư mục chứa dữ liệu thô.

        Args:
            data_dir (str): Thư mục chứa calendar.csv, sell_prices.csv, sales_train_validation.csv
        """
        self.data_dir = data_dir
        self.calendar_path = os.path.join(data_dir, "calendar.csv")
        self.prices_path = os.path.join(data_dir, "sell_prices.csv")
        self.sales_path = os.path.join(data_dir, "sales_train_validation.csv")

    def load_calendar(self) -> pd.DataFrame:
        """
        Load và kỹ nghệ đặc trưng cho bảng calendar.csv
        """
        print("[+] Đang load và xử lý calendar.csv...")
        calendar = pd.read_csv(self.calendar_path)
        calendar = reduce_mem_usage(calendar, verbose=False)

        # Chuyển đổi cột ngày
        calendar["date"] = pd.to_datetime(calendar["date"])
        calendar["day"] = calendar["date"].dt.day.astype(np.int8)
        calendar["week"] = calendar["date"].dt.isocalendar().week.astype(np.int8)
        calendar["month"] = calendar["date"].dt.month.astype(np.int8)
        calendar["year"] = calendar["date"].dt.year.astype(np.int16)
        calendar["is_weekend"] = calendar["wday"].isin([1, 2]).astype(np.int8)

        # 1. Cyclical Encoding cho ngày trong tuần và tháng
        calendar["sin_wday"] = np.sin(2 * np.pi * calendar["wday"] / 7).astype(np.float32)
        calendar["cos_wday"] = np.cos(2 * np.pi * calendar["wday"] / 7).astype(np.float32)
        calendar["sin_month"] = np.sin(2 * np.pi * calendar["month"] / 12).astype(np.float32)
        calendar["cos_month"] = np.cos(2 * np.pi * calendar["month"] / 12).astype(np.float32)

        # 2. Xử lý Missing Values cho các sự kiện
        calendar["event_name_1"] = calendar["event_name_1"].fillna("None").astype("category")
        calendar["event_type_1"] = calendar["event_type_1"].fillna("None").astype("category")
        calendar["event_name_2"] = calendar["event_name_2"].fillna("None").astype("category")
        calendar["event_type_2"] = calendar["event_type_2"].fillna("None").astype("category")

        # Cờ đánh dấu có sự kiện trong ngày
        calendar["has_event"] = (calendar["event_name_1"] != "None").astype(np.int8)

        # ---------------------------------------------------------------
        # CẢI TIẾN BIỂU DIỄN NGÀY LỄ CHO TRANSFORMER
        # Binary 'has_event' quá nghèo: không phân biệt LOẠI lễ (Văn hóa/
        # Thể thao/Tôn giáo/Quốc gia) và không nắm được HIỆU ỨNG LAN
        # (cầu tăng trước & sau lễ). Ta bổ sung:
        #   1. One-hot theo event_type_1 (loại sự kiện).
        #   2. event_count: số sự kiện trong ngày (0/1/2).
        #   3. days_to_next_event / days_since_last_event: khoảng cách
        #      (ngày) đến sự kiện gần nhất -> mô hình học được tiền/hậu
        #      kỳ của ngày lễ (ví dụ: mua sắm tăng mạnh trước Christmas).
        # ---------------------------------------------------------------
        # 1. One-hot event_type (bỏ 'None' vì đã có has_event)
        evt_types = ["Cultural", "Religious", "Sporting", "National"]
        for t in evt_types:
            calendar[f"evt_{t}"] = (calendar["event_type_1"] == t).astype(np.int8)

        # 2. Số lượng sự kiện (event_name_1 và event_name_2)
        calendar["event_count"] = (
            (calendar["event_name_1"] != "None").astype(np.int8)
            + (calendar["event_name_2"] != "None").astype(np.int8)
        )

        # 3. Khoảng cách đến sự kiện gần nhất (dùng event flag sinh bởi has_event)
        is_evt = calendar["has_event"].values
        n = len(calendar)
        days_to_next = np.full(n, 999, dtype=np.int16)
        days_since_last = np.full(n, 999, dtype=np.int16)
        # Tìm index các ngày có sự kiện
        evt_idx = np.where(is_evt == 1)[0]
        if len(evt_idx) > 0:
            # days_to_next_event: khoảng cách đến sự kiện tiếp theo
            next_evt = np.searchsorted(evt_idx, np.arange(n))
            # searchsorted trả về vị trí; clamp
            valid = next_evt < len(evt_idx)
            days_to_next[valid] = (evt_idx[next_evt[valid]] - np.arange(n)[valid]).astype(np.int16)
            days_to_next[~valid] = 999
            # days_since_last_event: khoảng cách từ sự kiện gần nhất trước đó
            prev_evt = np.searchsorted(evt_idx, np.arange(n), side="right") - 1
            valid_p = prev_evt >= 0
            days_since_last[valid_p] = (np.arange(n)[valid_p] - evt_idx[prev_evt[valid_p]]).astype(np.int16)
            days_since_last[~valid_p] = 999

        # Clip khoảng cách: quá 60 ngày coi như "không gần lễ" (tránh giá trị
        # ngoại lai 999 phá vỡ chuẩn hóa StandardScaler).
        calendar["days_to_next_event"] = np.clip(days_to_next, 0, 60).astype(np.int16)
        calendar["days_since_last_event"] = np.clip(days_since_last, 0, 60).astype(np.int16)

        return calendar

    def load_prices(self, train_end_wk: Optional[int] = None) -> pd.DataFrame:
        """
        Load và tạo đặc trưng cho bảng sell_prices.csv (Option A: frozen train stats).

        Args:
            train_end_wk: wm_yr_wk cutoff của train. price_max/min/mean/std chỉ tính
                trên prices[wm_yr_wk <= train_end_wk] để chống leak tương lai.
                Nếu None -> tính trên full (fallback test nhanh, có warning leak).
        """
        print("[+] Đang load và xử lý sell_prices.csv...")
        prices = pd.read_csv(self.prices_path)
        prices = reduce_mem_usage(prices, verbose=False)

        # Tạo đặc trưng động về giá (Price Dynamics) - FROZEN theo train
        # 1. Giá trung bình và giá cao nhất lịch sử theo store_id & item_id (chỉ trên train)
        if train_end_wk is not None:
            train_prices = prices[prices["wm_yr_wk"] <= train_end_wk]
            print(f"[-] Frozen price stats trên train (wm_yr_wk <= {train_end_wk}, "
                  f"{len(train_prices):,} dòng train / {len(prices):,} dòng full)...")
            price_stats = train_prices.groupby(["store_id", "item_id"])["sell_price"].agg(
                price_max="max",
                price_min="min",
                price_mean="mean",
                price_std="std"
            ).reset_index()
        else:
            print("[!] WARNING: train_end_wk=None -> tính stats trên full (nguy cơ leak). "
                  "Chỉ dùng để test nhanh, không dùng cho train/val chính thức.")
            price_stats = prices.groupby(["store_id", "item_id"])["sell_price"].agg(
                price_max="max",
                price_min="min",
                price_mean="mean",
                price_std="std"
            ).reset_index()

        prices = prices.merge(price_stats, on=["store_id", "item_id"], how="left")

        # 2. Tỷ lệ giảm giá (Discount Ratio: giá hiện tại / giá cao nhất train).
        # Sale sâu chưa từng thấy ở train -> discount âm -> clip về [0, 1].
        prices["price_discount"] = (1.0 - prices["sell_price"] / prices["price_max"]).fillna(0).astype(np.float32)
        prices["price_discount"] = prices["price_discount"].clip(0.0, 1.0).astype(np.float32)

        # 3. Tỷ lệ biến động giá so với trung bình train
        prices["price_norm_mean"] = (prices["sell_price"] / prices["price_mean"]).fillna(1.0).astype(np.float32)

        # 4. Momentum giá (So với tuần trước, chỉ dùng quá khứ -> không leak)
        prices["price_momentum"] = prices.groupby(["store_id", "item_id"])["sell_price"].pct_change().fillna(0).astype(np.float32)

        prices = reduce_mem_usage(prices, verbose=False)
        return prices

    def create_sku_level_data(
        self,
        start_day: int = 1000,
        end_day: int = 1913,
        items_subset: Optional[List[str]] = None,
        train_end_day: int = 1885
    ) -> pd.DataFrame:
        """
        Melt và merge toàn bộ dữ liệu ở cấp độ SKU (Level 12) - Option A chống leak.

        Args:
            start_day (int): Ngày bắt đầu lấy (mặc định lấy từ ngày 1000 để tối ưu RAM và loại bỏ drift cũ).
            end_day (int): Ngày kết thúc (1913 cho train/validation).
            items_subset (List[str], optional): Danh sách ID muốn lọc (nếu muốn train thử nghiệm nhanh).
            train_end_day (int): Ngày kết thúc train (mặc định 1885, 28 ngày cuối
                1886-1913 làm val giả lập H=28). price_max/mean chỉ tính đến ngày này.

        Returns:
            pd.DataFrame: Bảng dữ liệu long format hoàn chỉnh cấp độ SKU.
        """
        print(f"[+] Đang xử lý sales_train_validation.csv (Từ d_{start_day} đến d_{end_day}, "
              f"train_end=d_{train_end_day})...")
        sales = pd.read_csv(self.sales_path)
        
        if items_subset is not None:
            sales = sales[sales["id"].isin(items_subset)]

        # Danh sách các cột ngày cần unpivot
        day_cols = [f"d_{i}" for i in range(start_day, end_day + 1)]
        id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

        # Melt: Chuyển đổi từ Wide sang Long format
        print("[-] Unpivoting (Melt) dữ liệu...")
        df = pd.melt(
            sales,
            id_vars=id_cols,
            value_vars=day_cols,
            var_name="d",
            value_name="sales"
        )
        del sales
        gc.collect()

        df = reduce_mem_usage(df, verbose=False)

        # Load calendar trước để suy ra train_end_wk (chống leak giá)
        calendar = self.load_calendar()
        cal_day_map = calendar.set_index("d")["wm_yr_wk"]
        # train_end_day dạng số -> key "d_{n}"
        train_end_key = f"d_{train_end_day}"
        if train_end_key in cal_day_map.index:
            train_end_wk = int(cal_day_map.loc[train_end_key])
        else:
            # fallback: end_day nhỏ hơn train_end (chạy test nhanh) -> dùng max hiện có
            train_end_wk = None
            print(f"[!] train_end_day d_{train_end_day} ngoài phạm vi calendar của batch "
                  f"(end_day=d_{end_day}). Dùng fallback full stats + warning.")
        prices = self.load_prices(train_end_wk=train_end_wk)

        # Merge Calendar
        print("[-] Merging với Calendar features...")
        df = df.merge(calendar, on="d", how="left")

        # Merge Prices
        print("[-] Merging với Price features...")
        df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

        # Xử lý missing giá (left-censoring: chưa lên kệ -> sell_price NaN).
        # Không fill 0 (giá 0 không tồn tại). Thêm cờ is_listed + fill median dept.
        # PHUONG AN A (strict, chong leak): median fill CHI tinh tren train
        # (d_num <= train_end_day), khong dung thong ke tuong lai (val/test).
        df["is_listed"] = df["sell_price"].notna().astype(np.int8)
        df["d_num"] = df["d"].apply(lambda x: int(x.split("_")[1])).astype(np.int16)
        train_mask = df["d_num"] <= train_end_day
        if train_mask.any():
            dept_median_train = df.loc[train_mask].groupby("dept_id")["sell_price"].median()
            global_median_train = float(df.loc[train_mask, "sell_price"].median())
        else:  # fallback batch test nhanh ngoai pham vi train
            dept_median_train = df.groupby("dept_id")["sell_price"].median()
            global_median_train = float(df["sell_price"].median()) if df["sell_price"].notna().any() else 0.0
            print("[!] train_mask rong -> dung fallback median (chi cho test nhanh).")
        df["sell_price"] = (
            df["sell_price"]
            .fillna(df["dept_id"].map(dept_median_train))
            .fillna(global_median_train)
            .astype(np.float32)
        )
        df["price_discount"] = df["price_discount"].fillna(0).astype(np.float32)
        df["price_norm_mean"] = df["price_norm_mean"].fillna(1.0).astype(np.float32)
        df["price_momentum"] = df["price_momentum"].fillna(0).astype(np.float32)

        # 3. Kỹ nghệ đặc trưng Lag & Rolling (CHỐNG LEAKAGE)
        print("[-] Tạo Lag và Rolling features (Độ trễ tối thiểu >= 28 ngày)...")
        df = self._add_lag_and_rolling_features(df)

        # Bỏ warmup rows chưa đủ 28 ngày lịch sử (lag_28 NaN -> model không học được)
        before = len(df)
        df = df.dropna(subset=["sales_lag_28"]).reset_index(drop=True)
        print(f"[-] Drop warmup thiếu lag_28: {before - len(df):,} dòng "
              f"(còn {len(df):,} dòng).")

        # Assert chống leak giá: price_max phải frozen theo train
        if train_end_wk is not None:
            print(f"[-] Verify frozen stats: train_end_wk={train_end_wk} "
                  f"(price_max/min/mean tính đến tuần này, val dùng lại).")

        df = reduce_mem_usage(df, verbose=True)
        print("[✓] Hoàn thành tiền xử lý dữ liệu SKU-Level!")
        return df

    def _add_lag_and_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toán các đặc trưng trễ và thống kê trượt theo từng chuỗi 'id'.
        Lưu ý: Mọi lag đều bắt đầu từ 28 ngày trở đi để phục vụ dự báo 28 ngày tới.
        PHUONG AN A: gia (sell_price/discount/momentum) CHI dung ban lagged
        (qua khu <= t), khong bao gio dung gia realtime tuong lai.
        """
        # Đảm bảo dữ liệu được sắp xếp đúng thứ tự thời gian
        df = df.sort_values(by=["id", "d_num"]).reset_index(drop=True)

        # Các mức lag an toàn: 28, 35, 42, 56
        lags = [28, 35, 42, 56]
        for lag in lags:
            df[f"sales_lag_{lag}"] = df.groupby("id")["sales"].shift(lag).astype(np.float32)

        # Rolling statistics tính trên chuỗi đã shift 28 ngày
        # 1. Rolling Mean & Std trong 7 ngày (1 tuần)
        df["sales_rolling_mean_7"] = (
            df.groupby("id")["sales"]
            .transform(lambda x: x.shift(28).rolling(window=7, min_periods=1).mean())
            .astype(np.float32)
        )
        df["sales_rolling_std_7"] = (
            df.groupby("id")["sales"]
            .transform(lambda x: x.shift(28).rolling(window=7, min_periods=1).std())
            .fillna(0)
            .astype(np.float32)
        )

        # 2. Rolling Mean & Std trong 28 ngày (4 tuần)
        df["sales_rolling_mean_28"] = (
            df.groupby("id")["sales"]
            .transform(lambda x: x.shift(28).rolling(window=28, min_periods=1).mean())
            .astype(np.float32)
        )
        df["sales_rolling_std_28"] = (
            df.groupby("id")["sales"]
            .transform(lambda x: x.shift(28).rolling(window=28, min_periods=1).std())
            .fillna(0)
            .astype(np.float32)
        )

        # --- PHUONG AN A: dac trung gia lagged (chi qua khu, dung cho PAST) ---
        # sell_price / price_discount / price_momentum realtime la unknown future
        # -> model chi duoc thay ban shift >= 28.
        for lag in lags:
            df[f"sell_price_lag_{lag}"] = df.groupby("id")["sell_price"].shift(lag).astype(np.float32)
            df[f"price_discount_lag_{lag}"] = df.groupby("id")["price_discount"].shift(lag).astype(np.float32)
            df[f"price_momentum_lag_{lag}"] = df.groupby("id")["price_momentum"].shift(lag).astype(np.float32)
        df["sell_price_rolling_mean_7"] = (
            df.groupby("id")["sell_price"]
            .transform(lambda x: x.shift(28).rolling(window=7, min_periods=1).mean())
            .astype(np.float32)
        )
        df["sell_price_rolling_mean_28"] = (
            df.groupby("id")["sell_price"]
            .transform(lambda x: x.shift(28).rolling(window=28, min_periods=1).mean())
            .astype(np.float32)
        )
        # Alias giu tuong thich voi ten mean_price/avg_discount cu (dang lagged)
        df["mean_price_lag_28"] = df["sell_price_lag_28"]
        df["avg_discount_lag_28"] = df["price_discount_lag_28"]

        return df

    def create_store_level_data(self, sku_df: pd.DataFrame, train_end_day: int = 1885) -> pd.DataFrame:
        """
        Tổng hợp dữ liệu từ SKU-level lên Store-level (Level 3 - 10 chuỗi thời gian).
        
        Args:
            sku_df (pd.DataFrame): Dữ liệu SKU-level đã có các đặc trưng.

        Returns:
            pd.DataFrame: Dữ liệu chuỗi thời gian tổng hợp cấp Store.
        """
        print("[+] Đang tổng hợp dữ liệu lên Store-Level (Level 3)...")

        # mean_price bỏ giá fill/missing (sell_price<=0 do fill median lỗi) -> dùng median
        # của hàng listed để không kéo mean xuống oan. listed_ratio cho TFT biết độ phủ.
        if "is_listed" in sku_df.columns:
            sku_df["_price_listed"] = sku_df["sell_price"].where(sku_df["is_listed"] == 1)
        else:
            sku_df["_price_listed"] = sku_df["sell_price"].where(sku_df["sell_price"] > 0)

        # Nhóm theo store_id và ngày (date, d_num)
        store_agg = sku_df.groupby(["store_id", "date", "d_num"]).agg(
            total_sales=("sales", "sum"),
            mean_price=("_price_listed", "median"),
            avg_discount=("price_discount", "mean"),
            listed_ratio=("is_listed", "mean") if "is_listed" in sku_df.columns else ("sales", "size"),
            is_weekend=("is_weekend", "first"),
            sin_wday=("sin_wday", "first"),
            cos_wday=("cos_wday", "first"),
            sin_month=("sin_month", "first"),
            cos_month=("cos_month", "first"),
            has_event=("has_event", "first"),
            event_name_1=("event_name_1", "first"),
            state_id=("state_id", "first")
        ).reset_index()

        # Map calendar SNAP tương ứng theo bang (CA/TX/WI riêng)
        calendar = self.load_calendar()
        store_agg = store_agg.merge(calendar[["date", "snap_CA", "snap_TX", "snap_WI"]], on="date", how="left")
        
        store_agg["snap_active"] = np.where(
            store_agg["state_id"] == "CA", store_agg["snap_CA"],
            np.where(store_agg["state_id"] == "TX", store_agg["snap_TX"], store_agg["snap_WI"])
        ).astype(np.int8)

        # Bỏ các cột snap riêng lẻ
        store_agg.drop(columns=["snap_CA", "snap_TX", "snap_WI"], inplace=True)

        # Merge các đặc trưng ngày lễ mới (date-specific, giống nhau mọi store).
        # Loại 'has_event'/'event_name_1' vì đã có sẵn từ bước groupby phía trên.
        cal_event_cols = [
            "date", "event_count",
            "evt_Cultural", "evt_Religious", "evt_Sporting", "evt_National",
            "days_to_next_event", "days_since_last_event"
        ]
        store_agg = store_agg.merge(calendar[cal_event_cols], on="date", how="left")


        # Thêm Lags cho Store Total Sales + gia lagged (PHUONG AN A:
        # mean_price/avg_discount realtime la unknown future -> chi dung ban lagged)
        store_agg = store_agg.sort_values(by=["store_id", "d_num"]).reset_index(drop=True)
        for lag in [28, 35, 42, 56]:
            store_agg[f"sales_lag_{lag}"] = store_agg.groupby("store_id")["total_sales"].shift(lag).astype(np.float32)
            store_agg[f"mean_price_lag_{lag}"] = store_agg.groupby("store_id")["mean_price"].shift(lag).astype(np.float32)
            store_agg[f"avg_discount_lag_{lag}"] = store_agg.groupby("store_id")["avg_discount"].shift(lag).astype(np.float32)

        store_agg["sales_rolling_mean_7"] = (
            store_agg.groupby("store_id")["total_sales"]
            .transform(lambda x: x.shift(28).rolling(window=7, min_periods=1).mean())
            .astype(np.float32)
        )
        store_agg["sales_rolling_mean_28"] = (
            store_agg.groupby("store_id")["total_sales"]
            .transform(lambda x: x.shift(28).rolling(window=28, min_periods=1).mean())
            .astype(np.float32)
        )
        store_agg["mean_price_rolling_mean_7"] = (
            store_agg.groupby("store_id")["mean_price"]
            .transform(lambda x: x.shift(28).rolling(window=7, min_periods=1).mean())
            .astype(np.float32)
        )

        # Fill mean_price còn NaN (ngày không có hàng listed nào) + drop warmup lag
        # PHUONG AN A: median CHI tinh tren train (d_num <= train_end_day).
        train_med = store_agg.loc[store_agg["d_num"] <= train_end_day, "mean_price"].median()
        if pd.isna(train_med):
            train_med = store_agg["mean_price"].median()
        store_agg["mean_price"] = store_agg["mean_price"].fillna(train_med).astype(np.float32)
        if "listed_ratio" in store_agg.columns:
            store_agg["listed_ratio"] = store_agg["listed_ratio"].fillna(1.0).astype(np.float32)
        before = len(store_agg)
        store_agg = store_agg.dropna(subset=["sales_lag_28"]).reset_index(drop=True)
        print(f"[-] Store warmup drop: {before - len(store_agg):,} dòng.")

        # Dọn cột phụ
        if "_price_listed" in sku_df.columns:
            sku_df.drop(columns=["_price_listed"], inplace=True, errors="ignore")

        store_agg = reduce_mem_usage(store_agg, verbose=True)
        print("[✓] Hoàn thành tổng hợp dữ liệu Store-Level!")
        return store_agg

    def save_processed_data(self, df: pd.DataFrame, file_name: str, output_dir: str = "data/processed"):
        """
        Lưu DataFrame dưới định dạng Parquet nén (Snappy) để load cực nhanh.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
        print(f"[-] Đang lưu file vào: {output_path}...")
        df.to_parquet(output_path, compression="snappy", index=False)
        print(f"[✓] Đã lưu thành công! Dung lượng file: {os.path.getsize(output_path) / (1024**2):.2f} MB")
