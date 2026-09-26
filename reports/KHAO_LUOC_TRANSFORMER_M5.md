# Khảo lược các mô hình Transformer cho dự báo nhu cầu bán lẻ trên bộ dữ liệu M5

**Phạm vi:** chỉ các nghiên cứu dùng kiến trúc Transformer trên dataset M5 (Walmart) và tiền thân lý thuyết của chúng, giai đoạn 2020–2026.
**Mục đích:** trả lời 3 câu hỏi — (a) Transformer nào tốt nhất trên M5? (b) Biến ngoại sinh giúp được bao nhiêu? (c) Diễn giải được đến đâu?

---

## 1. Mở đầu

Cuộc thi M5 (Makridakis et al., 2022) đặt ra bài toán dự báo 42.840 chuỗi doanh số phân cấp của Walmart, gồm hai thử thách song song: Accuracy (điểm dự báo) và Uncertainty (khoảng tin cậy). Kết quả chính thức cho thấy **LightGBM** — chứ không phải deep learning — thống trị top 50 (đội thắng YJ_STU dùng 220 mô hình LightGBM với Tweedie loss; hạng hai kết hợp LightGBM + N-BEATS; hạng ba dùng 43 mạng LSTM). Điều này đặt ra câu hỏi: các kiến trúc Transformer — vốn thành công vang dội trong NLP — có thực sự vượt trội trên dữ liệu bán lẻ phân cấp, gián đoạn, nhiều biến ngoại sinh như M5 hay không? Loạt nghiên cứu hậu-M5 (2024–2026) đã trả lời có hệ thống, và bài khảo lược này tổng hợp chúng.

---

## 2. Bối cảnh dữ liệu M5

M5 gồm 3.049 sản phẩm × 10 cửa hàng (bang CA, TX, WI), doanh số ngày từ 29/01/2011 đến 19/06/2016 (1.969 ngày), tổng cộng 30.490 chuỗi ở cấp SKU–cửa hàng, kèm biến ngoại sinh: lịch (ngày lễ, thứ), giá bán, khuyến mãi, SNAP. Metric chuẩn: WRMSSE (accuracy phân cấp) và WQL/pinball loss (uncertainty); các bài benchmark Transformer thường báo thêm MASE và NRMSE. Đặc trưng gây khó: nhu cầu gián đoạn (intermittent, nhiều zero), tính mùa vụ đa chu kỳ, và phụ thuộc mạnh vào sự kiện/giá.

---

## 3. Họ mô hình Transformer đánh giá trên M5

| Mô hình | Ý tưởng cốt lõi | Điểm mạnh / yếu trên M5 |
|---|---|---|
| **Vanilla Transformer** | Self-attention đầy đủ O(L²), encoder–decoder | Mạnh toàn diện, nhưng tốn tính toán với chuỗi dài |
| **Informer** (Zhou et al., 2021) | ProbSparse attention O(L log L) + distilling + decoder sinh 1 lần | Top đầu M5, nhưng train lâu nhất |
| **Autoformer** | Tách trend–seasonal + auto-correlation | Vượt baseline nhưng thấp hơn nhóm đầu, cần tune thêm |
| **PatchTST** | Chia chuỗi thành patch như vision token | Hiệu quả tính toán nhất, accuracy hơi thấp hơn |
| **ETSFormer / NSTransformer / Reformer** | Tích hợp ETS / chuẩn hóa stationarity / hashing attention | Được khảo sát (Caetano 2025), không vượt nhóm dẫn đầu |
| **TFT** (Lim et al., 2021) | VSN + GRN + LSTM cục bộ + interpretable attention + quantile output | Top đầu M5 **và** diễn giải được — lựa chọn của đồ án này |

---

## 4. So sánh thực nghiệm chi tiết trên M5

### 4.1. Benchmark 5 Transformer vs baseline truyền thống (Oliveira & Ramos, 2024)
- **Thiết lập:** M5 đầy đủ (30.490 chuỗi, 10 cửa hàng); baseline AutoARIMA, AutoETS, seasonal Naïve, Naïve; metric MASE và WQL trên nhiều horizon.
- **Kết quả MASE:** Transformer, Informer và **TFT** dẫn đầu ổn định, cải thiện **26–29%** so với seasonal Naïve trên mọi horizon. Autoformer và PatchTST cũng vượt baseline nhưng thấp hơn, cần tinh chỉnh thêm. AutoARIMA/AutoETS hơn seasonal Naïve nhưng thua xa nhóm Transformer; Naïve tệ nhất.
- **Kết quả WQL (probabilistic):** TFT và Transformer dẫn đầu, giảm tới **34%** so với seasonal Naïve, đặc biệt ở horizon ngắn (1–7 ngày). Informer tương đương, hơi thấp hơn. Autoformer/PatchTST cải thiện ít ở WQL — gợi ý chúng kém hợp với quantile loss.
- **Chi phí:** Informer train lâu nhất; PatchTST hiệu quả nhất; đáng chú ý **AutoARIMA train > 24 giờ** — chậm hơn cả Transformer mà accuracy kém hơn.

### 4.2. Vai trò biến ngoại sinh (Caetano, Oliveira & Ramos, 2025)
- **Thiết lập:** 6 kiến trúc (Vanilla, Informer, Autoformer, ETSFormer, NSTransformer, Reformer) trên M5, so hai chế độ: chỉ dùng quá khứ sales vs thêm biến giải thích (lịch, giá, SNAP, socio-economic).
- **Kết quả:** thêm biến ngoại sinh giảm **tới 12.4% NRMSE** và cải thiện **2.9% MASE**. Dự báo phân vị giúp lượng hóa bất định, hỗ trợ ra quyết định tồn kho theo rủi ro.
- **Ý nghĩa cho đồ án:** đây là bằng chứng văn hiến trực tiếp cho giả thiết "ngoại sinh chi phối cầu", khớp với kết quả VSN của đồ án (nhóm mùa vụ/lễ chiếm ưu thế).

### 4.3. TFT gốc và tính diễn giải (Lim et al., 2019/2021)
- Trên 4 dataset (Electricity, Traffic, Retail-Favorita, Volatility), TFT giảm **7% P50 và 9% P90 loss** so với mô hình tốt thứ hai.
- Ba use-case diễn giải: (i) biến quan trọng toàn cục, (ii) mẫu thời gian bền vững (seasonality), (iii) phát hiện đổi chế độ — đúng 3 phân tích mà đồ án tái hiện (bảng VSN, attention lag 7/28/56).

### 4.4. Mở rộng đa cấp: HTF (2026)
- Hierarchical Temporal Fusion thêm structured embedding + coherence loss vào TFT, đánh giá trên M5: **WRMSSE 0.605 so với flat TFT 0.691 (cải thiện 12%)**, coherence error 0.045 so với 0.121 (giảm > 60%), pinball 0.158 so với 0.172.
- Ý nghĩa: dự báo độc lập từng cấp (như đồ án hiện tại: SKU riêng, store riêng) chưa đảm bảo cộng khớp phân cấp — đây là hướng phát triển tự nhiên.

### 4.5. TFT trên dữ liệu Walmart khác (Punati et al., 2025)
- Doanh số tuần 45 cửa hàng Walmart (2010–2012), encoder 52 tuần → dự báo 5 tuần, QuantileLoss(0.1, 0.5, 0.9), hidden 64, 4 heads: **RMSE ≈ 57.9k USD/store-week, R² = 0.9875**; CV 5-fold trung bình R² = 0.9844, thắng XGB/CNN/LSTM/CNN-LSTM.
- Củng cố: cùng họ siêu tham số với đồ án (quantile 0.1/0.5/0.9, static store embedding) cho kết quả tốt trên dữ liệu Walmart.

### Bảng tổng hợp

| Nghiên cứu | Mô hình so sánh | Phát hiện chính trên M5 / Walmart |
|---|---|---|
| Oliveira & Ramos 2024 | Vanilla, Informer, Autoformer, PatchTST, TFT vs AutoARIMA/ETS | Nhóm Transformer/Informer/TFT tốt nhất: MASE +26–29%, WQL −34% |
| Caetano et al. 2025 | 6 Transformer, có/không biến ngoại sinh | Thêm ngoại sinh: NRMSE −12.4%, MASE +2.9% |
| Lim et al. 2021 | TFT vs DeepAR/DSSM/ConvTrans | TFT: P50 −7%, P90 −9%; 3 use-case diễn giải |
| HTF 2026 | HTF vs flat TFT/MinT/Bottom-Up | WRMSSE 0.605 vs 0.691; coherence error −60% |
| Punati et al. 2025 | TFT vs XGB/CNN/LSTM/CNN-LSTM (Walmart tuần) | TFT thắng, R² = 0.9875 |

---

## 5. Trả lời 3 câu hỏi khảo lược

**(a) Transformer nào tốt nhất trên M5?** Bộ ba Vanilla Transformer – Informer – TFT dẫn đầu ổn định ở cả MASE và WQL; PatchTST thắng về hiệu quả tính toán; Autoformer cần tune thêm. Không có một "nhà vô địch tuyệt đối" — lựa chọn phụ thuộc trade-off accuracy/compute/interpretability, và TFT là lựa chọn duy nhất mạnh cả ba mặt.

**(b) Biến ngoại sinh giúp bao nhiêu?** Tới 12.4% NRMSE (Caetano 2025) — đủ lớn để coi ngoại sinh là thành phần bắt buộc, không phải tùy chọn, trong dự báo bán lẻ.

**(c) Diễn giải được đến đâu?** Chỉ TFT (và hậu duệ HTF) cho diễn giải hệ thống: trọng số biến (VSN), mẫu thời gian (attention), phát hiện sự kiện. Các Transformer còn lại vẫn là hộp đen như LSTM.

---

## 6. Hạn chế chung và khoảng trống

- **Compute:** Transformer (đặc biệt Informer) train nặng; AutoARIMA còn nặng hơn (> 24h) mà kém chính xác hơn.
- **LightGBM vẫn cạnh tranh:** quán quân M5 là LightGBM — Transformer thắng trong các benchmark hậu-M5 có tune kỹ, nên mọi claim "vượt trội" cần kèm điều kiện thực nghiệm.
- **Coherence phân cấp:** hầu hết benchmark dự báo độc lập từng chuỗi; HTF (2026) mới bắt đầu giải quyết cộng khớp — còn là khoảng trống lớn.
- **Validation:** Caetano (2024) dùng rolling CV 3 cửa sổ 28 ngày — chuẩn nên áp dụng thay vì single holdout.

---

## 7. Kết luận

Văn hiến 2024–2026 hội tụ: (1) Transformer — đặc biệt TFT — vượt baseline truyền thống trên M5 ở cả điểm và phân vị; (2) biến ngoại sinh (lịch, giá, SNAP) là nguồn gain lớn nhất nhì; (3) TFT nổi bật nhờ diễn giải được. Ba kết luận này chính là nền tảng văn hiến cho đồ án: chọn TFT, kỹ nghệ ngoại sinh/ngày lễ, và phân tích VSN + attention.

---

## Tài liệu tham khảo

1. Lim, B. et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *Int. J. Forecasting* 37(4). (arXiv:1912.09363)
2. Oliveira, J. M. & Ramos, P. (2024). Evaluating the Effectiveness of Time Series Transformers for Demand Forecasting in Retail. *Mathematics* 12(17):2728.
3. Caetano, R., Oliveira, J. M. & Ramos, P. (2025). Transformer-Based Models for Probabilistic Time Series Forecasting with Explanatory Variables. Preprints 2025.
4. Caetano, R. J. T. (2024). Transformer-based deep learning models for retail forecasting. Master's thesis, Porto Polytechnic.
5. Hierarchical Temporal Fusion (2026). Improving Coherence in Hierarchical Time Series Forecasting using Structured Temporal Fusion. arXiv:2606.28553.
6. Punati, S. B. et al. (2025). Temporal Fusion Transformer for Multi-Horizon Probabilistic Forecasting of Weekly Retail Sales. IEEE ICDABI.
7. Makridakis, S. et al. (2022). The M5 competition: Background, organization, and implementation. *Int. J. Forecasting* 38(4).
8. Makridakis, S. et al. (2022). M5 accuracy competition: Results, findings, and conclusions. *Int. J. Forecasting* 38(4).
9. Zhou, H. et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI.
10. Kunz, M. (2025). Hierarchical Time Series Forecasting Via Latent Mean Encoding. (So sánh TFT/DeepAR/TSMixer trên M5: WRMSSE TFT 0.670.)
