# BÁO CÁO PHÂN TÍCH CHUYÊN SÂU BỘ DỮ LIỆU WALMART M5 FORECASTING

---

### **ĐỀ TÀI NGHIÊN CỨU:**
# XÂY DỰNG MÔ HÌNH DỰ BÁO NHU CẦU TỒN KHO ĐA CẤP ĐỘ SỬ DỤNG MÔ HÌNH TRANSFORMER CÓ YẾU TỐ NGOẠI SINH

---

```
                                  BỘ DỮ LIỆU WALMART M5
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        ▼                                   ▼                                   ▼
┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
│ sales_train_val.csv   │       │     calendar.csv      │       │    sell_prices.csv    │
│ 30,490 chuỗi thời gian│       │ 1,969 ngày quan sát   │       │ ~6.84 triệu bản ghi   │
│ 1,913 ngày lịch sử    │       │ Sự kiện, SNAP, Mùa vụ │       │ Giá niêm yết theo tuần│
└───────────────────────┘       └───────────────────────┘       └───────────────────────┘
```

---

## 1. NGUỒN GỐC VÀ QUY MÔ BỘ DỮ LIỆU

*   **Đơn vị cung cấp:** Tập đoàn bán lẻ hàng đầu thế giới **Walmart Inc. (Hoa Kỳ)** hợp tác cùng Trung tâm Dự báo Nghiên cứu M-Competitions do Giáo sư **Spyros Makridakis** (Đại học Nicosia) chủ trì.
*   **Mục tiêu bài toán:** Dự báo chính xác số lượng hàng bán ra hàng ngày (Daily Unit Sales) cho **28 ngày tiếp theo** ($H = 28$).
*   **Phạm vi không gian:** 10 đại siêu thị (Supercenters) phân bổ trên **3 bang lớn** của Mỹ:
    *   *California (CA):* 4 cửa hàng (`CA_1`, `CA_2`, `CA_3`, `CA_4`).
    *   *Texas (TX):* 3 cửa hàng (`TX_1`, `TX_2`, `TX_3`).
    *   *Wisconsin (WI):* 3 cửa hàng (`WI_1`, `WI_2`, `WI_3`).
*   **Phạm vi ngành hàng:** 3,049 sản phẩm độc nhất thuộc **3 danh mục lớn** (Categories) chia thành **7 phân ngành** (Departments):
    *   `FOODS` (Thực phẩm & Đồ tươi sống): `FOODS_1`, `FOODS_2`, `FOODS_3`.
    *   `HOUSEHOLD` (Đồ gia dụng & Hóa mỹ phẩm): `HOUSEHOLD_1`, `HOUSEHOLD_2`.
    *   `HOBBIES` (Đồ chơi, Sở thích & Giải trí): `HOBBIES_1`, `HOBBIES_2`.
*   **Phạm vi thời gian:** Kéo dài từ ngày **29/01/2011 đến ngày 19/06/2016** (1,913 ngày cho tập huấn luyện và xác thực ban đầu).

---

## 2. CẤU TRÚC CHI TIẾT 3 TỆP DỮ LIỆU NGUỒN

### 2.1. Tệp Doanh số: `sales_train_validation.csv`
*   **Dung lượng & Kích thước:** 30,490 dòng $\times$ 1,919 cột (ở dạng bảng rộng - Wide Format).
*   **Cấu trúc thuộc tính:**
    *   `id`: Khóa định danh duy nhất của chuỗi (ví dụ: `FOODS_3_001_CA_1_validation`).
    *   `item_id`: Mã sản phẩm (3,049 mã duy nhất).
    *   `dept_id`: Mã phân ngành (7 mã).
    *   `cat_id`: Mã danh mục chính (3 mã).
    *   `store_id`: Mã cửa hàng (10 mã).
    *   `state_id`: Mã bang (`CA`, `TX`, `WI`).
    *   `d_1` $\to$ `d_1913`: Số lượng sản phẩm bán được từ ngày thứ 1 đến ngày thứ 1,913.

### 2.2. Tệp Lịch và Sự kiện Ngoại sinh: `calendar.csv`
*   **Dung lượng & Kích thước:** 1,969 dòng $\times$ 14 cột.
*   **Cấu trúc thuộc tính:**
    *   `date`: Ngày dương lịch (`YYYY-MM-DD`).
    *   `wm_yr_wk`: Mã tuần tài chính của Walmart (ví dụ: `11101`).
    *   `weekday`, `wday`: Tên thứ trong tuần và mã thứ ($\text{wday} \in [1, 7]$, với 1 = Thứ Bảy theo quy ước tài chính của Walmart).
    *   `month`, `year`: Tháng và năm.
    *   `d`: Nhãn ánh xạ tương ứng với bảng sales (`d_1` $\to$ `d_1969`).
    *   `event_name_1`, `event_type_1`: Tên và phân loại của sự kiện chính trong ngày (gồm 4 nhóm: *Sporting, Cultural, National, Religious*).
    *   `event_name_2`, `event_type_2`: Tên và phân loại sự kiện phụ (xảy ra khi một ngày trùng hai sự kiện lớn, ví dụ: Lễ Phục sinh trùng ngày Cá tháng Tư).
    *   `snap_CA`, `snap_TX`, `snap_WI`: Cờ nhị phân ($1/0$) thể hiện ngày đó bang có đang trong đợt phát trợ cấp lương thực SNAP hay không.

### 2.3. Tệp Giá bán: `sell_prices.csv`
*   **Dung lượng & Kích thước:** 6,841,121 dòng $\times$ 4 cột.
*   **Cấu trúc thuộc tính:**
    *   `store_id`: Cửa hàng áp dụng giá.
    *   `item_id`: Sản phẩm áp dụng giá.
    *   `wm_yr_wk`: Tuần tài chính áp dụng.
    *   `sell_price`: Giá bán niêm yết (USD). Giá được duy trì cố định theo từng tuần nhưng có thể thay đổi giữa các tuần khi có khuyến mãi hoặc điều chỉnh chính sách giá.

---

## 3. CÂY PHÂN CẤP 12 CẤP ĐỘ CỦA M5 (12-LEVEL HIERARCHY)

Một trong những nét đặc trưng học thuật quan trọng nhất của M5 là cấu trúc phân cấp đa tầng (Hierarchical Structure). Dự báo không chỉ thực hiện ở cấp độ chi tiết nhất mà phải thỏa mãn tính nhất quán trên toàn bộ 12 cấp độ:

```
                                 [ Level 1: Toàn quốc (1 chuỗi) ]
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
      [ Level 2: Bang CA ]            [ Level 2: Bang TX ]           [ Level 2: Bang WI ]
                 │                              │                              │
        ┌────────┴────────┐            ┌────────┴────────┐            ┌────────┴────────┐
        ▼                 ▼            ▼                 ▼            ▼                 ▼
   [Store CA_1]      [Store CA_4] [Store TX_1]      [Store TX_3] [Store WI_1]      [Store WI_3]
   (Level 3: 10 chuỗi Cửa hàng)
        │
        ├──────────────────────────────┬──────────────────────────────┐
        ▼                              ▼                              ▼
 [ HOBBIES Dept ]               [ FOODS Dept ]                [ HOUSEHOLD Dept ]
 (Level 4: 3 Danh mục | Level 5: 7 Phân ngành | Level 8 & 9: Cửa hàng x Ngành)
        │
        ▼
 [ 30,490 chuỗi cấp độ thấp nhất: Từng Mặt hàng tại từng Cửa hàng (Level 12 - SKU Level) ]
```

### Bảng Chi tiết 12 Cấp độ Phân cấp trong M5:

| Cấp độ (Level) | Tên Phân cấp (Aggregation Level) | Số lượng Chuỗi ($N$) | Ý nghĩa trong Quản trị Chuỗi Cung ứng |
| :---: | :--- | :---: | :--- |
| **Level 1** | Tổng toàn hệ thống (Total National) | 1 | Kế hoạch tài chính & Ngân sách cấp tập đoàn |
| **Level 2** | Tổng theo Bang (State Total) | 3 | Quy hoạch Trung tâm Phân phối Vùng (Regional DC) |
| **Level 3** | Tổng theo Cửa hàng (Store Total) | 10 | Điều phối nhân sự, diện tích kho bãi siêu thị |
| **Level 4** | Tổng theo Danh mục chính (Category) | 3 | Chiến lược ngành hàng (Category Management) |
| **Level 5** | Tổng theo Phân ngành (Department) | 7 | Đàm phán nhà cung ứng theo nhóm sản phẩm |
| **Level 6** | Bang $\times$ Danh mục chính | 9 | Phân bổ hàng hóa vùng theo thị hiếu |
| **Level 7** | Bang $\times$ Phân ngành | 21 | Lập kế hoạch bổ sung hàng cấp vùng |
| **Level 8** | Cửa hàng $\times$ Danh mục chính | 30 | Tối ưu hóa không gian trưng bày cửa hàng |
| **Level 9** | Cửa hàng $\times$ Phân ngành | 70 | Đặt hàng định kỳ cho từng quầy hàng |
| **Level 10** | Toàn quốc $\times$ Từng Mặt hàng | 3,049 | Đánh giá vòng đời sản phẩm (Product Life Cycle) |
| **Level 11** | Bang $\times$ Từng Mặt hàng | 9,147 | Kế hoạch sản xuất & Nhập khẩu cấp vùng |
| **Level 12** | **Cửa hàng $\times$ Từng Mặt hàng (SKU)** | **30,490** | **Bổ sung tồn kho tự động & Tính Safety Stock** |

---

## 4. BỐN ĐẶC TRƯNG THỐNG KÊ VÀ THÁCH THỨC TOÁN HỌC

### 4.1. Nhu cầu Ngắt quãng và Tỷ lệ Bằng Không Lớn (Zero-Inflation & Sparsity)
*   **Thực trạng:** Ở cấp độ Level 12 (SKU-level), trung bình **hơn 60% số ngày có doanh số bán ra bằng 0**. Nhiều sản phẩm thuộc nhóm `HOBBIES` hoặc `HOUSEHOLD_2` có thể không bán được chiếc nào trong 2-3 tuần liên tiếp, sau đó đột ngột xuất hiện giao dịch mua 5-10 sản phẩm trong một ngày.
*   **Thách thức toán học:**
    *   Phá vỡ giả định sai số chuẩn $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ của các mô hình hồi quy tuyến tính.
    *   Hàm mất mát $MAPE$ truyền thống bị vô hiệu hóa vì mẫu số $y_t = 0$ dẫn đến phân kỳ $\infty$.
    *   Mô hình dễ rơi vào cực tiểu cục bộ "luôn dự báo bằng 0" (Trivial Zero Prediction) nếu dùng hàm mất mát MSE thuần túy.

```
MÔ PHỎNG PHÂN PHỐI NHU CẦU CẤP ĐỘ SKU (LEVEL 12):
Số ngày
  ^
  |  [========================================] (Doanh số = 0: chiếm ~60-70%)
  |
  |  [======] (Doanh số = 1)
  |  [===]    (Doanh số = 2)
  |  [=]      (Doanh số >= 3)
  +------------------------------------------------------------> Doanh số (Units)
```

### 4.2. Hiện tượng Kiểm duyệt Dữ liệu Trước Mở bán (Left-Censoring / Unlisted Items)
*   **Thực trạng:** Trong 1,913 ngày lịch sử, không phải sản phẩm nào cũng có mặt trên kệ từ ngày đầu tiên ($d_1$). Rất nhiều sản phẩm chỉ mới bắt đầu được nhập về và niêm yết bán từ ngày $d_{500}$ hoặc $d_{1000}$.
*   **Thách thức:** Trong giai đoạn trước khi mở bán, doanh số ghi nhận là 0 nhưng không phải vì "nhu cầu bằng 0" mà vì "chưa có hàng để bán". Nếu đưa toàn bộ giai đoạn này vào huấn luyện mà không xử lý biến giá (`sell_price` bị thiếu), mô hình sẽ học sai lệch phân phối nhu cầu thực tế của sản phẩm.

### 4.3. Tác động của Chính sách Trợ cấp Xã hội SNAP (Food Stamps Effect)
*   **Cơ chế:** Chương trình SNAP cấp ngân sách mua thực phẩm cho các hộ gia đình khó khăn vào các ngày cố định:
    *   *California (CA):* Cấp vào 10 ngày đầu tiên của tháng (ngày 1 đến ngày 10).
    *   *Texas (TX):* Cấp rải rác theo các ngày làm việc từ ngày 1 đến ngày 15.
    *   *Wisconsin (WI):* Cấp vào các ngày chẵn/lẻ tùy thuộc vào mã hồ sơ.
*   **Tác động lên dữ liệu:** Khi cờ `snap == 1`, doanh số của ngành hàng `FOODS` tại các siêu thị tương ứng tăng vọt từ **20% đến 45%** so với những ngày bình thường.

### 4.4. Tác động Cộng hưởng của Giá, Khuyến mãi và Sự kiện
*   **Độ co giãn của cầu theo giá:** Giá bán của Walmart kết thúc bằng đuôi lẻ (`.98`, `.97`, `.94` biểu thị các mức giảm giá khuyến mãi). Khi tỷ lệ giảm giá $\text{price\_discount} > 15\%$, lượng bán ra có xu hướng tăng đột biến phi tuyến tính.
*   **Ngày đóng cửa duy nhất:** Walmart đóng cửa toàn bộ hệ thống vào ngày **Lễ Giáng sinh (25/12)** hàng năm $\to$ doanh số $y_t = 0$ tuyệt đối trên tất cả các chuỗi.
*   **Sự kiện thể thao & Văn hóa:** Ngày diễn ra **Super Bowl** làm bùng nổ doanh số đồ ăn nhanh, bia, nước ngọt (`FOODS_3`) và đồ uống gia dụng trước đó 1-2 ngày.

---

## 5. TÁC ĐỘNG CỦA BẢN CHẤT DATASET ĐẾN THIẾT KẾ MÔ HÌNH TRANSFORMER (TFT)

Những đặc tính phức tạp nêu trên của bộ dữ liệu M5 chính là lý do quyết định tại sao kiến trúc **Temporal Fusion Transformer (TFT)** là sự lựa chọn tối ưu vượt trội:

```
ĐẶC TÍNH BỘ DỮ LIỆU WALMART M5             THIẾT KẾ KIẾN TRÚC TƯƠNG ỨNG TRONG TFT
──────────────────────────────────────     ────────────────────────────────────────────────
1. Dữ liệu tĩnh đa tầng                 ──> Static Entity Embeddings & Static Context GRN
   (Store_ID, Item_ID, Dept_ID, State)      (Mã hóa bản chất thực thể vào không gian ẩn)

2. Tác động ngoại sinh đan xen          ──> Variable Selection Network (VSN)
   (Discount, SNAP, Sự kiện, Cuối tuần)     (Tự động gán trọng số đóng góp động cho từng biến)

3. Nhu cầu ngắt quãng, nhiều rủi ro     ──> Quantile Loss (Pinball Loss)
   (Biến động khó lường)                    (Dự báo đa phân vị P10, P50, P90 thay vì điểm đơn)

4. Chu kỳ mùa vụ tuần rõ rệt            ──> Interpretable Multi-Head Temporal Self-Attention
   (Đỉnh mua sắm thứ Bảy & Chủ Nhật)       (Tập trung trọng số chú ý vào các lag t-7, t-14, t-28)
```

1.  **Cần phân luồng biến đa dạng:** Dữ liệu M5 bắt buộc mô hình phải phân tách rõ: Biến tĩnh (Static), Biến quan sát quá khứ (Past Sales, Lags), và Biến ngoại sinh tương lai đã biết trước (Giá niêm yết, Lịch SNAP, Ngày lễ). TFT là kiến trúc duy nhất có các luồng VSN riêng biệt cho từng loại biến này.
2.  **Yêu cầu Dự báo Bất định (Uncertainty Estimation):** Với dữ liệu có độ lệch chuẩn lớn như M5, dự báo đơn điểm (Point Forecast) không thể phản ánh rủi ro đứt hàng. Hàm mất mát **Quantile Pinball Loss** của TFT sinh ra khoảng tin cậy $[P_{10}, P_{90}]$, cho phép chuyển đổi trực tiếp kết quả mô hình thành quyết định **Tồn kho an toàn (Safety Stock)** trong nghiệp vụ bán lẻ thực tế.

---

## 6. TỔNG KẾT

Bộ dữ liệu **Walmart M5 Forecasting** không đơn thuần là một chuỗi số liệu thời gian lý thuyết, mà là một bức tranh toàn cảnh sống động về hành vi tiêu dùng, kinh tế học bán lẻ và quản trị chuỗi cung ứng hiện đại. Việc thấu hiểu sâu sắc cấu trúc 3 bảng dữ liệu, cây phân cấp 12 cấp độ và các động lực ngoại sinh (Giá, Khuyến mãi, SNAP, Lễ hội) chính là chìa khóa then chốt giúp xây dựng pipeline tiền xử lý chống rò rỉ dữ liệu và cấu hình thành công mô hình học sâu **Temporal Fusion Transformer** đạt hiệu năng vượt trội trong đề tài này.
