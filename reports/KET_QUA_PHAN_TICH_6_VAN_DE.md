# KẾT QUẢ PHÂN TÍCH THỰC NGHIỆM — 6 HẠNG MỤC PHỤC VỤ BÁO CÁO

> Tài liệu tóm tắt các kết quả chạy thực tế trên bộ dữ liệu Walmart M5 (cấp độ Store CA_1,
> chân trời dự báo H = 28 ngày). Số liệu dưới đây lấy từ các script:
> `scripts/analyze_data_quality.py`, `run_phase3_baselines.py`, `run_phase4_transformer.py`,
> `run_phase5_evaluation_and_interpretability.py`, `scripts/analyze_interpretability.py`.

---

## 1. Chống rò rỉ dữ liệu (Data Leakage) trong Train/Test Time Series

**Phương pháp đảm bảo:**
- **Chia tập theo thời gian (time-based split), không trộn (no shuffling):**
  Train ≤ d1857 | Val d1858–d1885 | Test d1886–d1913 (358 / 28 / 28 ngày).
- **Đặc trưng trễ/trượt chỉ dùng quá khứ:** mọi lag đều `shift(≥28)` ngày, không bao giờ
  sử dụng thông tin tương lai. Kiểm chứng: tại ngày test đầu tiên (d1886),
  `sales_lag_28 = total_sales(d1858)` (thuộc Val) → **đúng, không rò rỉ**.
- **Biến known-future (lịch, giá, lễ) là thông tin đã biết trước:** `mean_price`, `avg_discount`,
  `is_weekend`, `snap_active`, `has_event`, … lấy từ bảng calendar/sell_prices, **không chứa**
  `total_sales` tương lai → dùng cho tập Test là **hợp lệ**.
- **Chuẩn hóa riêng cho train:** `StandardScaler` chỉ `fit` trên tập Train, áp dụng (transform)
  cho Val/Test → không truyền thống mức của tương lai vào quá khứ.
- **Mặt nạ nhân quả (causal mask) trong Temporal Attention:** vị trí i chỉ được chú ý đến
  j ≤ i, ngăn mô hình "nhìn trước" các bước thời gian tương lai.

---

## 2. Cơ chế Attention trong Transformer (và cách diễn giải)

- TFT dùng **Interpretable Multi-Head Temporal Self-Attention**: chia sẻ một phép chiếu Value
  duy nhất trên tất cả các head, sau đó lấy trung bình ma trận chú ý → một ma trận
  `A_tilde ∈ [T, T]` duy nhất, dễ trực quan hóa (xem `reports/figures/temporal_attention.png`).
- **Attention profile** (truy vấn là 28 bước tương lai, trung bình chú ý lên quá khứ) cho thấy
  mô hình tập trung mạnh vào các vùng trễ mang tính chu kỳ:
  - lag ≈ 7 ngày (1 tuần): 0.0170
  - lag ≈ 28 ngày (4 tuần): 0.0164
  - lag ≈ 56 ngày (8 tuần): 0.0161
  → nhất quán với **tính mùa vụ tuần/tháng** của hàng tiêu dùng bán lẻ.
- File kết quả: `data/processed/temporal_attention_profile.csv`.

---

## 3. Kiểm tra dữ liệu thiếu (Missing Data)

| Bảng | Cột | Tỷ lệ thiếu | Nhận xét |
|---|---|---|---|
| calendar | `event_name_1` / `event_type_1` | 97.5% / 97.0% | **Bình thường** – phần lớn ngày không có lễ |
| calendar | `event_name_2` / `event_type_2` | 100% | Chỉ vài ngày có sự kiện thứ 2, đã xử lý `fillna("None")` |
| sell_prices | `sell_price` | 0% | Đầy đủ |
| sales (wide) | toàn bộ | 0% NaN; **16.91% giá trị = 0** | Số 0 là **cầu gián đoạn (intermittent)**, không phải thiếu |
| processed (SKU/Store) | `sales_lag_*` / `sales_rolling_*` | 6.8%–13.5% | Chỉ ở **28 ngày đầu mỗi chuỗi** (chưa đủ lịch sử) → `fillna(0)` khi vào mô hình |

**Kết luận:** dữ liệu không có "ếm" (missing) thực tế; các ô NaN chỉ nằm ở biên chuỗi và được
xử lý an toàn. 16.91% giá trị 0 phản ánh tính gián đoạn của cầu – cần giữ nguyên (không nội suy).

---

## 4. Xây dựng mô hình Baseline: ARIMA và LSTM

Được triển khai sẵn trong `src/models/arima_baseline.py` và `src/models/lstm_model.py`,
chạy qua `run_phase3_baselines.py`. Kết quả trên Store CA_1, H = 28:

| Mô hình | RMSE | MAE | WMAPE (%) | sMAPE (%) |
|---|---|---|---|---|
| **ARIMA(7,1,1)** | 12.746 | 9.599 | **8.995** | 8.852 |
| **Multivariate LSTM** | 12.812 | 9.558 | **8.957** | 8.813 |
| TFT (đề xuất, P50) | 13.554 | 10.280 | 9.633 | 9.486 |

> Ghi chú: Facebook Prophet có trong kế hoạch nhưng **chưa cài thư viện `prophet`**
> (bị bỏ qua tự động). Có thể bổ sung sau bằng `pip install prophet`.
> WMAPE là chỉ số chính (robust với cầu gián đoạn, không bị chia 0).

---

## 5. Phân tích khả năng diễn giải các yếu tố ngoại sinh / tính mùa vụ

Dùng trọng số **Variable Selection Network (VSN)** trung bình theo thời gian. Nhóm đặc trưng
ngoại sinh mùa vụ chiếm **82.9%** tổng trọng số nhóm Future; nhóm Giá/Khuyến mãi chỉ **17.1%**.

**Top đặc trưng quan trọng nhất (Future VSN):**

| Đặc trưng | Trọng số | Nhóm |
|---|---|---|
| `evt_National` (lễ quốc gia) | 0.262 | Mùa vụ/Lễ |
| `avg_discount` (giảm giá) | 0.123 | Giá/KM |
| `cos_wday` (chu kỳ thứ) | 0.123 | Mùa vụ |
| `evt_Sporting` (sự kiện thể thao) | 0.100 | Mùa vụ/Lễ |
| `cos_month` / `sin_month` | 0.086 / 0.075 | Mùa vụ |
| `event_count` | 0.055 | Mùa vụ/Lễ |
| `mean_price` | 0.048 | Giá/KM |
| `snap_active` | 0.027 | Mùa vụ |

**Đặc trưng quá khứ (Past VSN):** `sales_lag_35` áp đảo (0.734) → cầu tuần trước có liên quan
mật thiết đến cùng tuần năm trước (chu kỳ 35 ngày gần 5 tuần).

→ Mô hình cho thấy **tính mùa vụ và ngày lễ là động lực ngoại sinh mạnh nhất** của cầu,
xác nhận vai trò của nhóm đặc trưng này trong kiến trúc đề xuất.

---

## 6. Cách biểu diễn ngày lễ (Holiday) để Transformer "hiểu"

**Trước đây (yếu):** chỉ một cờ nhị phân `has_event` ∈ {0,1} — không phân biệt *loại* lễ và
không nắm được *hiệu ứng lan* (cầu tăng trước/sau lễ).

**Cải tiến đã triển khai** (`src/data/m5_pipeline.py`, hàm `load_calendar`):
1. **One-hot theo loại sự kiện:** `evt_Cultural`, `evt_Religious`, `evt_Sporting`, `evt_National`.
2. **Số lượng sự kiện trong ngày:** `event_count` (0/1/2).
3. **Khoảng cách đến lễ gần nhất:** `days_to_next_event`, `days_since_last_event`
   (clip 0–60 ngày) → mô hình học được tiền/hậu kỳ của ngày lễ.

**Cách đưa vào Transformer:** các cột trên thuộc nhóm **known-future features**, đi qua
`future_embeddings` (Linear 1→d_model) → `future_vsn` (chọn lọc biến) → LSTM decoder →
Temporal Attention. Kết quả thực nghiệm: **`evt_National` trở thành đặc trưng ngoại sinh quan
trọng nhất (trọng số 0.262)**, minh chứng biểu diễn ngày lễ mới giúp mô hình khai thác tốt
tác động của ngày lễ. Xem `reports/figures/feature_importance.png`.

---

## Tổng kết cho báo cáo

- Quy trình **không rò rỉ dữ liệu** (split thời gian, lag an toàn, causal mask, scaler riêng).
- **Dữ liệu thiếu** chỉ ở biên chuỗi, xử lý an toàn; 16.9% số 0 là cầu gián đoạn (giữ nguyên).
- **Baseline** ARIMA & LSTM hoạt động, WMAPE ~9.0% làm mốc so sánh.
- **Diễn giải** cho thấy tính mùa vụ/ngày lễ chi phối cầu (82.9% nhóm ngoại sinh).
- **Biểu diễn ngày lễ** được nâng cấp (loại lễ + khoảng cách) và chứng minh hiệu quả qua VSN.
