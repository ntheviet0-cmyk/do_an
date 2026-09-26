"""
Vẽ lại 3 sơ đồ khối GỌN, TRỰC QUAN (ghi đè 3 file cũ):
  reports/figures/diagram_arima.png
  reports/figures/diagram_lstm.png
  reports/figures/diagram_holiday.png
Quy tắc: 5 hộp, 1 luồng thẳng trái->phải, chữ to, mỗi hộp tối đa 2 dòng phụ.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

NAVY = "#1F3864"
EDGE = "#2E5496"
C_IN = "#FCE4D6"     # input: cam nhạt
C_PROC = "#FFF2CC"   # xử lý: vàng nhạt
C_MODEL = "#DDEBF7"  # mô hình: xanh dương nhạt
C_OUT = "#E2EFDA"    # output: xanh lá nhạt


def box(ax, x, y, w, h, title, sub=None, color=C_MODEL):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=1.6, edgecolor=EDGE, facecolor=color))
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=NAVY)
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=8.5, color="#333333")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>",
        mutation_scale=20, linewidth=2.4, color=EDGE,
        shrinkA=0, shrinkB=2))


def flow5(ax, boxes, y=0.30, h=0.34):
    """5 hộp thẳng hàng trái->phải + mũi tên. boxes: list[(title, sub, color)]."""
    xs = [0.015, 0.215, 0.415, 0.615, 0.815]
    w = 0.165
    for (title, sub, color), x in zip(boxes, xs):
        box(ax, x, y, w, h, title, sub, color)
    yc = y + h / 2
    for i in range(4):
        arrow(ax, xs[i] + w, yc, xs[i + 1], yc)


def draw_arima():
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("ARIMA(7,1,1) — baseline đơn biến", fontsize=14,
                 fontweight="bold", color=NAVY, pad=10)
    flow5(ax, [
        ("Quá khứ", "doanh số y_t\ntrain + val", C_IN),
        ("Sai phân d=1", "y_t − y(t−1)\nchuỗi đứng yên", C_PROC),
        ("ARMA(7, 1)", "AR(7): chu kỳ tuần\nMA(1): sai số", C_MODEL),
        ("Dự báo", "28 ngày tới", C_OUT),
        ("Clip ≥ 0", "nhu cầu\nkhông âm", C_OUT),
    ])
    ax.text(0.5, 0.10, "Chỉ dùng lịch sử y_t (không giá / lễ / SNAP)  •  WMAPE ≈ 8,99%",
            ha="center", fontsize=9, color="#333333", style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "diagram_arima.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] diagram_arima.png")


def draw_lstm():
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("LSTM — baseline đa biến", fontsize=14,
                 fontweight="bold", color=NAVY, pad=10)
    flow5(ax, [
        ("Cửa sổ 56 ngày", "mỗi ngày = 1 vector", C_IN),
        ("20 đặc trưng/ngày", "sales, giá, lễ,\nlag, rolling…", C_IN),
        ("LSTM × 2 lớp", "hidden = 64", C_MODEL),
        ("FC 64 → 28", "1 lần ra 28 ngày", C_MODEL),
        ("Dự báo điểm", "hàm lỗi MSE\nWMAPE ≈ 8,96%", C_OUT),
    ])
    ax.text(0.5, 0.10, "Dùng được ngoại sinh, nhưng HỘP ĐEN (trạng thái ẩn không đọc được)",
            ha="center", fontsize=9, color="#333333", style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "diagram_lstm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] diagram_lstm.png")


def draw_holiday():
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Ngày lễ đưa vào Transformer như thế nào", fontsize=14,
                 fontweight="bold", color=NAVY, pad=10)
    flow5(ax, [
        ("Lịch thô", "tên lễ\nvd: Super Bowl", C_IN),
        ("Mã hoá lễ", "has_event, one-hot loại,\nevent_count, days_to/from", C_PROC),
        ("Known-future", "lịch biết trước\ncả 84 bước", C_PROC),
        ("VSN", "mỗi đặc trưng → 1 trọng số\n(tổng = 1)", C_MODEL),
        ("Dự báo 28 ngày", "P10 / P50 / P90", C_OUT),
    ], y=0.32)
    ax.text(0.5, 0.10, "Mã hoá theo LOẠI lễ (Thể thao / Văn hoá / Tôn giáo) + khoảng cách đến lễ",
            ha="center", fontsize=9, color="#333333", style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "diagram_holiday.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] diagram_holiday.png")


def draw_tft():
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("TFT — mô hình đề xuất", fontsize=14,
                 fontweight="bold", color=NAVY, pad=10)

    # 3 đầu vào
    box(ax, 0.03, 0.74, 0.28, 0.15, "Quá khứ", "56 ngày: sales,\nlag, rolling", C_IN)
    box(ax, 0.36, 0.74, 0.28, 0.15, "Tương lai known", "84 bước: giá, lễ,\nchu kỳ", C_IN)
    box(ax, 0.69, 0.74, 0.28, 0.15, "Tĩnh", "cửa hàng, bang,\nmặt hàng", C_IN)

    # 1 luồng thẳng
    box(ax, 0.18, 0.55, 0.64, 0.14, "VSN — chọn biến",
        "mỗi đặc trưng → 1 trọng số (tổng = 1)", C_PROC)
    box(ax, 0.18, 0.38, 0.64, 0.14, "LSTM Encoder – Decoder",
        "bộ nhớ chuỗi: quá khứ → tương lai", C_MODEL)
    box(ax, 0.18, 0.21, 0.64, 0.14, "Interpretable Attention",
        "causal mask • ma trận chú ý T×T đọc được", C_MODEL)
    box(ax, 0.18, 0.06, 0.64, 0.12, "Dự báo 28 ngày",
        "P10 / P50 / P90", C_OUT)

    for x in (0.17, 0.50, 0.83):
        arrow(ax, x, 0.74, x, 0.69)
    arrow(ax, 0.50, 0.55, 0.50, 0.52)
    arrow(ax, 0.50, 0.38, 0.50, 0.35)
    arrow(ax, 0.50, 0.21, 0.50, 0.18)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "diagram_tft.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] diagram_tft.png")


if __name__ == "__main__":
    draw_arima()
    draw_lstm()
    draw_holiday()
    draw_tft()
