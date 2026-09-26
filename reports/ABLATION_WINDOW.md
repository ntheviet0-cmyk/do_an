# Thực nghiệm ablation: độ dài cửa sổ quá khứ (28 / 56 / 84 ngày)

**Ngày chạy:** từ script `scripts/ablate_window.py` (seed=42 cố định).
**Thiết lập công bằng:** cùng dữ liệu store-level, cùng split Train/Val/Test,
cùng horizon=28, cùng bộ đặc trưng, cùng mô hình TFT thu nhỏ
(`d_model=16`, `num_heads=2`, `epochs=10`, `lr=1e-3`, `batch=64`).
Chỉ thay đổi duy nhất `seq_len`. Chỉ số: **WMAPE trên tập Test**.

## Kết quả

| Cửa sổ quá khứ | WMAPE (Test) | RMSE | MAE |
|---|---|---|---|
| 28 ngày | 9.130% | 13.279 | 9.743 |
| **56 ngày** | **9.149%** | 13.153 | 9.764 |
| 84 ngày | 9.085% | 13.253 | 9.695 |

Chi tiết đầy đủ: `data/processed/ablation_window.csv`.

## Đọc kết quả trung thực
- Chênh lệch tối đa giữa 3 cấu hình là **0.06pp** — dưới ngưỡng nhiễu của
  mô hình thu nhỏ (10 epoch, 1 seed, 1 chuỗi store). **Không có bằng chứng
  rằng 56 tối ưu về độ chính xác**, và cũng không có bằng chứng nó tệ hơn.
- Kết luận đúng: **mô hình ỔN ĐỊNH với mọi độ dài cửa sổ** trong khoảng 28–84.
- Vì vậy 56 được giữ bằng 3 căn cứ độc lập với accuracy:
  1. **Kỹ thuật:** lag lớn nhất trong bộ đặc trưng là `sales_lag_56`
     (`m5_pipeline.py:227`) → cửa sổ phải ≥ 56 mới dùng trọn bộ đặc trưng.
  2. **Thực chứng:** attention của mô hình đã train có peak tại lag 56 và
     `sales_lag_35` chiếm ~73% trọng số quá khứ → mô hình thực sự khai thác
     thông tin xa, không thể cắt ngắn tùy tiện.
  3. **Domain:** 56 ngày = 8 tuần tròn, khớp chu kỳ bán lẻ; = 2× horizon 28.

## Giới hạn & hướng phát triển
- Mô hình thu nhỏ + 10 epoch + 1 seed + chỉ 1 chuỗi store → ablation mang
  tính chỉ báo, chưa đủ sức khẳng định tối ưu toàn cục.
- Muốn chặt chẽ hơn: lặp lại với `d_model=64, epochs=35`, nhiều seed,
  và trên cả cấp SKU (tốn kém tính toán, để dành làm hướng phát triển).

## Câu trả lời bảo vệ
> "Em đã chạy ablation so 3 cửa sổ 28/56/84 trên cùng mô hình và dữ liệu:
> WMAPE lần lượt 9.13% / 9.15% / 9.09% — chênh lệch dưới 0.1pp, tức mô hình
> ổn định với mọi cửa sổ. Vì accuracy không phân biệt được, em giữ 56 ngày
> bằng ba căn cứ khác: đó là lag lớn nhất trong bộ đặc trưng nên là mức tối
> thiểu kỹ thuật; attention thực chứng cho thấy mô hình dùng thông tin tới
> lag 56; và 56 ngày là 8 tuần tròn khớp chu kỳ bán lẻ."
