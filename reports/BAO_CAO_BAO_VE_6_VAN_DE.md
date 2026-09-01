# BÁO CÁO CHI TIẾT 6 NỘI DUNG TRÌNH BÀY

**Đề tài:** Dự báo nhu cầu tồn kho đa cấp độ bằng Transformer có yếu tố ngoại sinh (M5 Walmart subset)
**Dữ liệu:** 100 SKU (50 mặt hàng × 2 cửa hàng CA_1, TX_1), 2 bang, 1 ngành FOODS, horizon H = 28 ngày.

---

## 1. Tránh rò rỉ dữ liệu (Data Leakage) cho train/test chuỗi thời gian

### 1.1. Khái niệm
Data leakage xảy ra khi thông tin của tập **test** lọt vào **train**, làm kết quả đánh giá quá lạc quan và không phản ánh thực tế triển khai. Với chuỗi thời gian, rò rỉ thường do 3 nguyên nhân:
- **Trộn ngẫu nhiên (random split)** giữa train/test → mẫu test nằm giữa train, mô hình "nhìn thấy tương lai".
- **Đặc trưng trễ (lag)** được tính trên toàn bộ chuỗi (gồm cả tương lai) → lag của ngày test chứa sales ngày tương lai.
- **Chuẩn hóa (scaler)** fit trên toàn bộ dữ liệu → trung bình/độ lệch chuẩn bị nhiễm từ test.

### 1.2. Cách thực hiện trong đồ án
| Biện pháp | Thực hiện cụ thể |
|---|---|
| **Chia theo thời gian** | Train = `[d0, d_t]`, Test = `[d_t+1, …]`. Không shuffle. |
| **Lag an toàn** | Đặc trưng lag/rolling được tạo với `shift >= H` (H=28), đảm bảo lag của ngày test chỉ dùng dữ liệu ≤ ngày train. Đã kiểm chứng trong `scripts/analyze_data_quality.py`. |
| **Scaler fit trên train** | `StandardScaler().fit(train)` rồi `transform(test)`; không fit trên toàn bộ. |
| **Known-future chỉ dùng lịch** | Đặc trưng tương lai chỉ gồm calendar (giá, lễ, SNAP, chu kỳ) — không dùng sales tương lai. |
| **Quét rò rỉ trước train** | `scripts/check_leakage_before_training.py` quét mọi cột: đặc trưng `future_known`/`static` được whitelist; bất kỳ cột lag có `shift < H` hoặc chứa `sales` tương lai đều bị báo động đỏ. |

### 1.3. Kết quả
Quét thực tế: toàn bộ cột đặc trưng quá khứ (lag ≥ 28) và known-future **đều qua whitelist** (không có cờ đỏ). Mô hình vì thế đánh giá đúng (WMAPE store-level ~9%, không bị ảo).

### 1.4. Câu trả lời cho báo cáo
> *"Chuỗi thời gian được chia theo trục thời gian (train trước, test sau), đặc trưng trễ được dịch đủ 28 ngày để không chứa thông tin tương lai, scaler chỉ fit trên tập train, và một bộ quét rò rỉ chạy trước khi huấn luyện để đảm bảo không có đặc trưng dựa trên sales tương lai. Nhờ đó kết quả không bị đánh giá quá lạc quan."*

---

## 2. Diễn giải Attention trong Transformer

### 2.1. Attention là gì
Self-attention cho phép mỗi bước thời gian "nhìn" các bước khác để cân bằng thông tin. Công thức: `Attention(Q,K,V)=softmax(QKᵀ/√d)·V`.

### 2.2. Cơ chế Interpretable trong code
Trong `src/models/transformer_components.py:144–210`, **InterpretableMultiHeadAttention** dùng chung một phép chiếu `V` cho mọi head, sau đó trung bình trọng số các head thành ma trận **[B, T, T]** (`mean_attn_weights`):
```python
mean_attn_weights = attn_weights_heads.mean(dim=1)   # [B, T, T]
```
- Ô `[i, j]` = mức độ bước i **chú ý** đến bước j.
- **Causal mask** (`tft_model.py`) chặn nhìn vào tương lai: mô hình chỉ được attend quá khứ → hợp lệ cho dự báo.

### 2.3. Ý nghĩa giải thích
- Đọc **hàng** tương ứng ngày cần dự báo → biết mô hình dựa vào những ngày quá khứ nào.
- Kết quả `temporal_attention_profile.csv`: attention tập trung vào **lag 7, 28, 56 ngày** → phản ánh tính tuần (7) và tháng (28/56). Đây là bằng chứng định lượng rằng mô hình học được mùa vụ.

### 2.4. So với LSTM
LSTM nén mọi thứ thành vector ẩn vô danh; Attention sinh ma trận có **tên (ngày i ↔ ngày j)** → đọc được.

### 2.5. Câu trả lời cho báo cáo
> *"Attention trong TFT được làm 'có thể diễn giải' bằng cách dùng chung phép chiếu V và trung bình trọng số các head, thu được ma trận chú ý T×T. Kết hợp causal mask, mô hình chỉ chú ý quá khứ. Phân tích ma trận này cho thấy mô hình tập trung vào các lag 7/28/56 ngày — tức là tính mùa vụ tuần và tháng."*

---

## 3. Cách kiểm tra dữ liệu có thiếu không

### 3.1. Phương pháp
Chạy `scripts/analyze_data_quality.py` để khảo sát tỉ lệ NaN và zero trên từng bảng (`calendar`, `sell_prices`, `sales`):
- **NaN** → dữ liệu bị khuyết.
- **Zero** → cần phân biệt "khuyết thật" vs "không bán" (intermittent demand).

### 3.2. Kết quả trên tập M5 subset
| Bảng / cột | Tình trạng | Nhận định |
|---|---|---|
| `calendar` event fields | ~97% NaN | **Bình thường** — chỉ ngày có lễ mới có tên lễ |
| `sell_prices` | 0% NaN | đầy đủ |
| `sales` | 16.9% giá trị = 0 | nhu cầu gián đoạn (intermittent) — **không phải thiếu**, giữ nguyên |

Sau tiền xử lý, chỉ còn NaN tại **đầu mỗi chuỗi** do lag/rolling chưa đủ cửa sổ — được giữ lại làm bước "warm-up" và loại khi tạo sample.

### 3.3. Chiến lược xử lý
- **Không impute** các giá trị 0 của sales (đó là tín hiệu thực).
- Chỉ loại/bỏ qua NaN đầu chuỗi do lag.
- Với calendar, giữ nguyên (dùng one-hot lễ, ngày không lễ = all-zero).

### 3.4. Câu trả lời cho báo cáo
> *"Kiểm tra cho thấy dữ liệu giá đầy đủ; tỉ lệ NaN lớn ở các trường lễ là do chỉ ngày có lễ mới có thông tin (bình thường); 16.9% sales bằng 0 phản ánh nhu cầu gián đoạn chứ không phải mất dữ liệu, nên được giữ nguyên. Quá trình tiền xử lý chỉ sinh NaN tại đầu chuỗi do cửa sổ trễ và được xử lý riêng."*

---

## 4. Baseline: xây dựng ARIMA và LSTM

### 4.1. ARIMA
- File: `src/models/arima_baseline.py`, chạy bởi `run_phase3_baselines.py`.
- Cấu hình: **ARIMA(7,1,1)** trên chuỗi tổng hợp cửa hàng CA_1 (chuỗi ổn định, ít zero hơn SKU).
- Đặc trưng: chỉ dùng lịch sử sales (mô hình thống kê, không dùng ngoại sinh phức tạp).
- Kết quả: **WMAPE 8.99%** (store CA_1). SKU-level: **60.36%**.

### 4.2. LSTM
- File: `src/models/lstm_model.py` (`MultivariateLSTMNet` + `LSTMBaseline`).
- Cấu hình: cửa sổ 56 ngày, `hidden_dim=64`, 2 lớp LSTM, đầu ra điểm.
- Đặc trưng: có dùng ngoại sinh (giá, lễ, SNAP).
- Kết quả: **WMAPE ~8.96%** (store). Tuy nhiên là **hộp đen** — không giải thích được.

### 4.3. So sánh với TFT
| Mô hình | WMAPE store | WMAPE SKU | Diễn giải |
|---|---|---|---|
| ARIMA(7,1,1) | 8.99% | 60.36% | Thống kê, yếu với ngoại sinh |
| LSTM | 8.96% | — | Hộp đen |
| **TFT (toàn cục)** | **8.92%** | **58.74%** | **Có (VSN + Attention)** |

→ Tại store-level, cả 3 xấp xỉ nhau (~9%), chứng tỏ baseline đã tốt; TFT thắng nhẹ và **quan trọng hơn: có khả năng diễn giải**. Tại SKU-level, TFT (58.74%) thắng ARIMA (60.36%) **2.67% tuyệt đối**.

### 4.4. Câu trả lời "anh xây dựng ARIMA và LSTM thế nào?"
> *"ARIMA(7,1,1) được xây dựng trên chuỗi tổng hợp cửa hàng, chỉ dùng lịch sử bán hàng và ước lượng bằng hàm tích hợp tối ưu AIC. LSTM gồm 2 lớp với cửa sổ 56 ngày, hidden 64, nhận cả đặc trưng ngoại sinh, huấn luyện bằng MSE. Cả hai làm baseline: ARIMA đại diện thống kê tuyến tính, LSTM đại diện mạng hồi quy phi tuyến nhưng là hộp đen."*

---

## 5. Phân tích khả năng diễn giải của yếu tố ngoại sinh (tính mùa vụ)

### 5.1. Phương pháp
Trọng số **Variable Selection Network (VSN)** thể hiện mức độ mô hình tin dùng từng đặc trưng. Lấy trung bình theo thời gian → bảng xếp hạng (tổng 100%). Kết quả từ `data/processed/feature_importance_table.csv`.

### 5.2. Bảng xếp hạng đặc trưng ngoại sinh
| Hạng | Đặc trưng | Trọng số | Nhóm |
|---|---|---|---|
| 1 | `evt_National` | **26.2%** | Mùa vụ/Lễ |
| 2 | `avg_discount` | 12.3% | Giá/KM |
| 3 | `cos_wday` | 12.3% | Mùa vụ |
| 4 | `evt_Sporting` | 10.0% | Mùa vụ/Lễ |
| 5 | `cos_month` | 8.6% | Mùa vụ |
| 6 | `sin_month` | 7.5% | Mùa vụ |
| 7 | `event_count` | 5.5% | Mùa vụ/Lễ |
| 8 | `mean_price` | 4.8% | Giá/KM |
| 9 | `snap_active` | 2.7% | Mùa vụ |
| 10 | `has_event` | 1.8% | Mùa vụ/Lễ |
| 11 | `is_weekend` | 1.6% | Mùa vụ |
| 12–16 | `days_to_next`, `evt_Cultural`, `days_since`, `sin_wday`, `evt_Religious` | 1.0–1.5% | Mùa vụ/Lễ |

**Phân nhóm:** Mùa vụ/Ngoại sinh = **82.9%**, Giá/Khuyến mãi = **17.1%**.

### 5.3. Phân tích
- **Ngày lễ là động lực mạnh nhất**: `evt_National` (26.2%) đứng đầu mọi đặc trưng, vượt cả giảm giá. `evt_Sporting` (10.0%) và `event_count` (5.5%) cũng cao → **loại lễ** quan trọng hơn sự hiện diện lễ (`has_event` chỉ 1.8%).
- **Chu kỳ tuần/tháng chi phối**: `cos_wday` (12.3%) + `sin/cos_month` (~16%) + `is_weekend` (1.6%) → tính mùa vụ rõ. Kết hợp attention (lag 7/28/56) tạo hai luồng bằng chứng độc lập.
- **Giá/khuyến mãi**: `avg_discount` (12.3%) gấp 2.6 lần `mean_price` (4.8%) → người tiêu dùng nhạy cảm với "đang sale".
- **Lịch sử vẫn là nền tảng**: Past VSN cho thấy `sales_lag_35` chiếm **73.4%** → ngoại sinh đóng vai trò bổ trợ.

### 5.4. Câu trả lời cho báo cáo
> *"Trích xuất từ VSN cho thấy nhóm mùa vụ/ngoại sinh chiếm 82.9% trọng số, áp đảo nhóm giá (17.1%). Đặc biệt `evt_National` đạt 26.2% — cao nhất — khẳng định ngày lễ và tính chu kỳ (cos/sin thứ, tháng) là động lực cầu thực phẩm. Điều này chứng minh TFT không chỉ dự báo tốt mà còn chỉ ra được nguyên nhân."*

---

## 6. Biểu diễn ngày lễ (Holiday) trong Transformer để mô hình hiểu

### 6.1. Vấn đề
Dữ liệu lễ thô là **văn bản** (`event_name`, `event_type`). Mô hình chỉ học được với số. Cờ nhị phân `has_event` quá nghèo (không nói lễ gì, không bắt hiệu ứng trước/sau lễ).

### 6.2. Cách mã hóa (`src/data/m5_pipeline.py`, hàm `load_calendar`)
Tạo 4 nhóm đặc trưng số, thuộc nhóm **known-future**:
| Đặc trưng | Cách tính | Ý nghĩa |
|---|---|---|
| `has_event` | 1/0 | có lễ không (giữ để so sánh) |
| `evt_Cultural/Religious/Sporting/National` | **one-hot theo loại** | phân biệt loại lễ |
| `event_count` | 0/1/2 | số lễ/ngày |
| `days_to_next_event`, `days_since_last_event` | khoảng cách (ngày), clip 0–60 | hiệu ứng lan trước/sau lễ |

Ví dụ ngày Quốc khánh: `has_event=1, evt_National=1, event_count=1, days_to_next_event=0`.

### 6.3. Đưa vào mô hình
Các cột known-future qua `Linear(1 → d_model=32)` (`tft_model.py` / `tft_global.py`) → mỗi giá trị thành vector 32 chiều → VSN gán trọng số → Attention kết hợp.

### 6.4. Minh chứng cách mã hóa ĐÚNG
Kết quả VSN: `evt_National` = **26.2%** (top), trong khi `has_event` cũ chỉ **1.8%**. Nếu chỉ dùng cờ nhị phân, mô hình không phân biệt được lễ Quốc gia → không thể phát hiện tác động lớn này. Cách mã hóa one-hot theo loại lễ đã giúp Transformer "hiểu" và khai thác đúng.

### 6.5. Câu trả lời cho báo cáo
> *"Ngày lễ không dùng cờ nhị phân đơn lẻ, mà được biểu diễn bằng one-hot theo loại lễ, số lượng lễ, và khoảng cách đến lễ gần nhất. Các đặc trưng này thuộc known-future, qua lớp embedding rồi VSN. Kết quả diễn giải cho thấy `evt_National` đạt 26.2% — cao nhất nhóm ngoại sinh — minh chứng cách mã hóa này giúp mô hình nhận diện chính xác tác động ngày lễ."*

---

## TÀI LIỆU KÈM THEO (hình ảnh đã tạo)
- `reports/figures/diagram_arima.png` — khối ARIMA
- `reports/figures/diagram_lstm.png` — khối LSTM
- `reports/figures/diagram_holiday.png` — luồng biểu diễn ngày lễ
- `reports/figures/diagram_tft.png` — kiến trúc TFT
- `reports/figures/feature_importance.png` — xếp hạng VSN
- `reports/figures/temporal_attention.png` — ma trận chú ý (lag 7/28/56)
- `reports/figures/forecast_quantiles.png` — dự báo P10/P50/P90 vs thực tế

## TÓM TẮT NHẤN MẠNH KHI BẢO VỆ
1. **Không rò rỉ**: chia theo thời gian, lag an toàn, scaler fit trên train, có bộ quét rò rỉ.
2. **Attention đọc được** nhờ shared-V + causal mask → lag 7/28/56.
3. **Dữ liệu**: giá đủ, lễ NaN là bình thường, 0 của sales là nhu cầu gián đoạn (giữ nguyên).
4. **Baseline**: ARIMA(7,1,1) 8.99% và LSTM 8.96% (store) — TFT thắng nhẹ, quan trọng là có diễn giải.
5. **Ngoại sinh/Mùa vụ**: chiếm 82.9%, `evt_National` top (26.2%) → giải thích được nguyên nhân.
6. **Holiday**: mã hóa one-hot loại lễ + khoảng cách → mô hình hiểu đúng, `evt_National` nổi lên nhờ đó.
