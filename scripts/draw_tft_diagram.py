"""
Script: draw_tft_diagram.py
Vẽ sơ đồ khối kiến trúc TFT: Past/Future/Static input -> VSN -> LSTM -> Static Enrichment
-> Interpretable Attention (causal mask) -> Quantile Head, kèm nhánh diễn giải.
Output: reports/figures/diagram_tft.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

def box(ax, x, y, w, h, title, sub=None, color="#DCE6F1", tcolor="#1F3864", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.02",
                 linewidth=1.5, edgecolor="#2E5496", facecolor=color))
    ax.text(x + w/2, y + h*0.62, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tcolor)
    if sub:
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center",
                fontsize=7.2, color="#333333")

def arrow(ax, x1, y1, x2, y2, color="#2E5496", style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                  arrowstyle=style, mutation_scale=13, linewidth=1.5, color=color))

def draw_tft():
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("KIẾN TRÚC TFT (Temporal Fusion Transformer) — MÔ HÌNH ĐỀ XUẤT",
                 fontsize=13, fontweight="bold", color="#1F3864", pad=12)

    # ---- Inputs (top) ----
    box(ax, 0.02, 0.82, 0.22, 0.15, "PAST input",
        "56 ngày quá khứ\nsales, lag, rolling", "#FCE4D6")
    box(ax, 0.39, 0.82, 0.22, 0.15, "FUTURE input (known)",
        "84 ngày (quá khứ+tương lai)\ngiá, lễ, SNAP, chu kỳ", "#FCE4D6")
    box(ax, 0.80, 0.82, 0.18, 0.15, "STATIC context",
        "store/state/cat/\ndept/item -> emb", "#FFF2CC")

    # ---- VSN ----
    box(ax, 0.02, 0.58, 0.22, 0.16, "Past VSN",
        "chọn biến quá khứ\n+ trọng số", "#DDEBF7")
    box(ax, 0.39, 0.58, 0.22, 0.16, "Future VSN",
        "chọn biến ngoại sinh\n+ trọng số", "#DDEBF7")

    # ---- LSTM ----
    box(ax, 0.02, 0.38, 0.22, 0.15, "LSTM Encoder", "từ Past VSN", "#E2EFDA")
    box(ax, 0.39, 0.38, 0.22, 0.15, "LSTM Decoder", "từ Future VSN", "#E2EFDA")

    # ---- Combine + Static Enrichment ----
    box(ax, 0.22, 0.18, 0.24, 0.15, "Static Enrichment (GRN)",
        "làm giàu bằng\nstatic_emb", "#E2EFDA")

    # ---- Attention ----
    box(ax, 0.50, 0.18, 0.26, 0.15, "Interpretable Multi-Head Attention",
        "có causal mask\n(trọng số T x T)", "#DDEBF7")

    # ---- Output ----
    box(ax, 0.80, 0.30, 0.18, 0.15, "Quantile Head",
        "Linear -> P10/P50/P90", "#FCE4D6")
    box(ax, 0.80, 0.10, 0.18, 0.15, "Dự báo 28 ngày", "phân vị + điểm", "#FCE4D6")

    # ---- Interpretability (right) ----
    box(ax, 0.80, 0.56, 0.18, 0.26, "KHẢ NĂNG DIỄN GIẢI",
        "past_vsn_weights\nfuture_vsn_weights\ntemporal_attention",
        "#F2F2F2", fs=8.5)

    # ---- arrows main flow ----
    arrow(ax, 0.13, 0.82, 0.13, 0.74)          # past -> past vsn
    arrow(ax, 0.50, 0.82, 0.50, 0.74)          # future -> future vsn
    arrow(ax, 0.13, 0.58, 0.13, 0.53)          # past vsn -> lstm enc
    arrow(ax, 0.50, 0.58, 0.50, 0.53)          # future vsn -> lstm dec
    arrow(ax, 0.13, 0.38, 0.28, 0.28)          # enc -> enrich
    arrow(ax, 0.50, 0.38, 0.44, 0.28)          # dec -> enrich
    arrow(ax, 0.46, 0.255, 0.50, 0.255)        # enrich -> attention
    arrow(ax, 0.76, 0.255, 0.80, 0.40)         # attention -> quantile head
    arrow(ax, 0.89, 0.30, 0.89, 0.25)          # quantile -> forecast

    # static context feeds VSNs and enrichment
    arrow(ax, 0.80, 0.82, 0.55, 0.70, color="#7F3F00")   # -> future vsn
    arrow(ax, 0.80, 0.82, 0.24, 0.70, color="#7F3F00")   # -> past vsn
    arrow(ax, 0.89, 0.74, 0.40, 0.30, color="#7F3F00")  # -> enrichment

    # interpretability arrows
    arrow(ax, 0.24, 0.66, 0.80, 0.74, color="#548235", style="-|>")
    arrow(ax, 0.61, 0.66, 0.80, 0.68, color="#548235", style="-|>")
    arrow(ax, 0.76, 0.30, 0.80, 0.56, color="#548235", style="-|>")

    ax.text(0.5, 0.03,
            "Luồng dự báo: Past/Future -> VSN -> LSTM -> Static Enrichment -> Attention -> Quantile Head.\n"
            "Nhánh xanh: VSN và Attention đồng thời sinh trọng số giải thích (khắc phục hộp đen của LSTM).",
            ha="center", fontsize=8.2, color="#333333", style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "diagram_tft.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] Đã lưu reports/figures/diagram_tft.png")

if __name__ == "__main__":
    draw_tft()
