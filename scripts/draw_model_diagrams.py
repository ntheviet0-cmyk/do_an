"""
Script: draw_model_diagrams.py
Vẽ sơ đồ khối chi tiết cho 2 baseline: ARIMA(7,1,1) và Multivariate LSTM.
Output: reports/figures/diagram_arima.png, reports/figures/diagram_lstm.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- helpers ----------
def box(ax, x, y, w, h, title, sub=None, color="#DCE6F1", tcolor="#1F3864"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.02",
                 linewidth=1.5, edgecolor="#2E5496", facecolor=color))
    ax.text(x + w/2, y + h*0.62, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color=tcolor, wrap=True)
    if sub:
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center",
                fontsize=7.5, color="#333333")

def arrow(ax, x1, y1, x2, y2, label=None, color="#2E5496"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                  arrowstyle="-|>", mutation_scale=14, linewidth=1.6, color=color))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.012, label, ha="center", va="bottom",
                fontsize=7.5, color="#7F3F00", fontweight="bold")

# ============================================================
# 1) ARIMA(7,1,1)
# ============================================================
def draw_arima():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("KIẾN TRÚC BASELINE — ARIMA(7,1,1)  (đơn biến, không ngoại sinh)",
                 fontsize=13, fontweight="bold", color="#1F3864", pad=14)

    box(ax, 0.02, 0.42, 0.17, 0.26, "Chuỗi doanh số y_t", "train + val\n(store CA_1)", "#FCE4D6")
    box(ax, 0.24, 0.42, 0.17, 0.26, "Sai phân bậc 1\n(d = 1)", "∇y_t = y_t − y_{t-1}\n→ chuỗi đứng yên", "#FFF2CC")
    box(ax, 0.46, 0.30, 0.22, 0.50, "MÔ HÌNH ARMA", "kết hợp AR + MA", "#DDEBF7")
    box(ax, 0.50, 0.66, 0.14, 0.18, "AR(p=7)", "hồi quy 7 bước\n→ bắt chu kỳ TUẦN", "#E2EFDA")
    box(ax, 0.50, 0.36, 0.14, 0.18, "MA(q=1)", "sai số trễ 1 bước", "#E2EFDA")
    box(ax, 0.73, 0.42, 0.17, 0.26, "Dự báo 28 ngày", "forecast(steps=28)", "#FCE4D6")
    box(ax, 0.90, 0.42, 0.08, 0.26, "Clip ≥ 0", "nhu cầu\nkhông âm", "#F2F2F2")

    arrow(ax, 0.19, 0.55, 0.24, 0.55)
    arrow(ax, 0.41, 0.55, 0.46, 0.55, "∇y_t")
    arrow(ax, 0.57, 0.70, 0.68, 0.55, "", color="#548235")
    arrow(ax, 0.57, 0.40, 0.68, 0.50, "", color="#548235")
    arrow(ax, 0.68, 0.55, 0.73, 0.55)
    arrow(ax, 0.90, 0.55, 0.98, 0.55)

    ax.text(0.5, 0.06, "Đặc trưng: CHỈ dùng lịch sử y_t (không có giá/lễ/SNAP…)  •  Học bằng tối ưu hợp lý (MLE)  •  WMAPE ≈ 8.99%",
            ha="center", fontsize=8.5, color="#333333", style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "diagram_arima.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] Đã lưu reports/figures/diagram_arima.png")

# ============================================================
# 2) Multivariate LSTM
# ============================================================
def draw_lstm():
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("KIẾN TRÚC BASELINE — MULTIVARIATE LSTM (đa biến, dự báo thẳng 28 bước)",
                 fontsize=13, fontweight="bold", color="#1F3864", pad=14)

    box(ax, 0.01, 0.40, 0.16, 0.30, "Cửa sổ 56 ngày",
        "sliding window\nmỗi ngày = 1 vector", "#FCE4D6")
    feats = ("Đặc trưng/ngày (20):\n"
             "sales, mean_price, avg_discount,\n"
             "is_weekend, sin/cos_wday,\n"
             "sin/cos_month, snap_active,\n"
             "has_event, evt_*, days_to_event,\n"
             "sales_lag_28, rolling_mean_7/28")
    box(ax, 0.20, 0.30, 0.20, 0.50, "Vector đầu vào x_t",
        feats, "#FFF2CC")
    box(ax, 0.44, 0.40, 0.15, 0.30, "LSTM Layer 1",
        "hidden = 64\n(num_layers=2)", "#DDEBF7")
    box(ax, 0.63, 0.40, 0.15, 0.30, "LSTM Layer 2",
        "hidden = 64\n(đầu ra h_T)", "#DDEBF7")
    box(ax, 0.81, 0.42, 0.17, 0.26, "FC Head",
        "Linear 64→28\n→ 28 bước dự báo", "#E2EFDA")
    box(ax, 0.90, 0.70, 0.09, 0.18, "ŷ (28)", "", "#F2F2F2")

    arrow(ax, 0.17, 0.55, 0.20, 0.55)
    arrow(ax, 0.40, 0.55, 0.44, 0.55)
    arrow(ax, 0.59, 0.55, 0.63, 0.55)
    arrow(ax, 0.78, 0.55, 0.81, 0.55)
    arrow(ax, 0.98, 0.78, 0.90, 0.80, "", color="#548235")

    # loss + training note
    ax.text(0.5, 0.12,
            "Huấn luyện: cửa sổ trượt (seq_len=56) • Hàm lỗi MSE • Early-stop qua Val Loss • Clip ≥ 0\n"
            "WMAPE ≈ 8.97%  •  Multi-horizon: 1 lần dự báo thẳng 28 ngày từ hidden state cuối",
            ha="center", fontsize=8.5, color="#333333", style="italic")

    # little legend for "t" flow
    ax.annotate("", xy=(0.17, 0.85), xytext=(0.01, 0.85),
                arrowprops=dict(arrowstyle="->", color="#C55A11"))
    ax.text(0.09, 0.88, "trục thời gian t (quá khứ → hiện tại)", ha="center",
            fontsize=7.5, color="#C55A11")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "diagram_lstm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] Đã lưu reports/figures/diagram_lstm.png")

if __name__ == "__main__":
    draw_arima()
    draw_lstm()
