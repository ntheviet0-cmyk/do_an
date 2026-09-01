"""
Script: draw_holiday_diagram.py
Vẽ sơ đồ khối: cách ngày lễ được biểu diễn và đưa vào Transformer (TFT).
Output: reports/figures/diagram_holiday.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

def box(ax, x, y, w, h, title, sub=None, color="#DCE6F1", tcolor="#1F3864", fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.02",
                 linewidth=1.5, edgecolor="#2E5496", facecolor=color))
    ax.text(x + w/2, y + h*0.60, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tcolor)
    if sub:
        ax.text(x + w/2, y + h*0.27, sub, ha="center", va="center",
                fontsize=7.4, color="#333333")

def arrow(ax, x1, y1, x2, y2, color="#2E5496"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                  arrowstyle="-|>", mutation_scale=14, linewidth=1.6, color=color))

def draw_holiday():
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("BIỂU DIỄN NGÀY LỄ TRONG TRANSFORMER (TFT) — từ text thô đến trọng số diễn giải",
                 fontsize=12.5, fontweight="bold", color="#1F3864", pad=12)

    # 1. raw
    box(ax, 0.02, 0.80, 0.22, 0.16, "Dữ liệu thô (calendar)",
        "event_name_1='Christmas'\nevent_type_1='National'", "#FCE4D6", fs=9)
    # 2. feature engineering
    box(ax, 0.30, 0.72, 0.26, 0.24, "Kỹ nghệ đặc trưng",
        "has_event (0/1)\nOne-hot loại lễ: evt_National,\nevt_Sporting, evt_Cultural, evt_Religious\n"
        "event_count (0/1/2)\ndays_to/from_event (clip 0–60)", "#FFF2CC", fs=8.5)
    # 3. known-future group
    box(ax, 0.62, 0.74, 0.20, 0.20, "Nhóm known-future",
        "đặc trưng lịch\nđã biết trước", "#DDEBF7", fs=9)
    # 4. embedding
    box(ax, 0.62, 0.46, 0.20, 0.20, "Embedding",
        "Linear(1 → d_model=32)\nmỗi giá trị -> vector 32 chiều", "#DDEBF7", fs=8.5)
    # 5. VSN
    box(ax, 0.36, 0.20, 0.24, 0.22, "Variable Selection Network (VSN)",
        "gán trọng số cho từng đặc trưng\nvd: evt_National = 0.262", "#E2EFDA", fs=9)
    # 6. attention + forecast
    box(ax, 0.04, 0.20, 0.24, 0.22, "LSTM + Interpretable Attention",
        "kết hợp chuỗi quá khứ &\nknown-future", "#E2EFDA", fs=8.5)
    box(ax, 0.04, 0.78, 0.20, 0.16, "Dự báo 28 ngày",
        "P10 / P50 / P90", "#FCE4D6", fs=9)

    # arrows main flow (top->down zigzag)
    arrow(ax, 0.24, 0.86, 0.30, 0.86)
    arrow(ax, 0.56, 0.84, 0.62, 0.84)          # eng -> known-future
    arrow(ax, 0.72, 0.74, 0.72, 0.66)          # known-future -> embedding
    arrow(ax, 0.72, 0.46, 0.60, 0.34)          # embedding -> vsn
    arrow(ax, 0.36, 0.31, 0.28, 0.31)          # vsn -> attention
    arrow(ax, 0.16, 0.31, 0.16, 0.20)          # attention -> forecast (up? no, forecast below left)
    # instead route attention -> forecast upward on left
    arrow(ax, 0.16, 0.42, 0.16, 0.78)

    # example box (right)
    box(ax, 0.84, 0.44, 0.15, 0.30, "Ví dụ",
        "Quốc khánh:\nhas_event=1\nevt_National=1\nevent_count=1\ndays_to_next=0\n\nNgày thường:\nhas_event=0\nevt_*=0\ndays_to=12", "#F2F2F2", fs=7.6)
    arrow(ax, 0.82, 0.55, 0.72, 0.56, color="#7F3F00")

    # bottom note
    ax.text(0.5, 0.06,
            "Minh chứng: evt_National đạt trọng số 26.2% (cao nhất nhóm ngoại sinh) → cách mã hóa theo LOẠI lễ "
            "giúp Transformer nhận diện đúng tác động ngày lễ (has_event cũ chỉ 1.8%).",
            ha="center", fontsize=8.2, color="#333333", style="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "diagram_holiday.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] Đã lưu reports/figures/diagram_holiday.png")

if __name__ == "__main__":
    draw_holiday()
