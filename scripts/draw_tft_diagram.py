# -*- coding: utf-8 -*-
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def box(ax, cx, cy, w, h, title, sub=None, color="#DCE6F1", fs=8.5):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.02",
                 linewidth=2, edgecolor="#2E5496", facecolor=color, zorder=2))
    ax.text(cx, cy+h*0.30, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="#1F3864", zorder=3)
    if sub:
        ax.text(cx, cy-h*0.20, sub, ha="center", va="center",
                fontsize=7, color="#444444", zorder=3)

def arrow(ax, x1, y1, x2, y2, color="#2E5496", lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                  arrowstyle=style, mutation_scale=16, linewidth=lw, color=color, zorder=1))

def draw_tft():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(-1, 10); ax.axis("off")
    ax.set_title("KIẾN TRÚC TFT (Temporal Fusion Transformer)",
                 fontsize=17, fontweight="bold", color="#1F3864", pad=25)

    # ======== INPUTS ========
    box(ax, 2.5, 9, 2.5, 1.0, "PAST INPUT", "56 ngày", "#FCE4D6")
    box(ax, 6, 9, 2.5, 1.0, "FUTURE INPUT", "84 ngày", "#FCE4D6")
    box(ax, 9.5, 9, 2.5, 1.0, "STATIC", "store, item", "#FFF2CC")

    # ======== VSN ========
    box(ax, 2.5, 7, 2.5, 1.0, "PAST VSN", "chọn biến", "#DDEBF7")
    box(ax, 6, 7, 2.5, 1.0, "FUTURE VSN", "chọn ngoại sinh", "#DDEBF7")

    # ======== LSTM ========
    box(ax, 2.5, 5, 2.5, 1.0, "LSTM ENCODER", "mã hóa", "#E2EFDA")
    box(ax, 6, 5, 2.5, 1.0, "LSTM DECODER", "mã hóa", "#E2EFDA")

    # ======== ENRICHMENT ========
    box(ax, 4, 3, 3, 1.0, "ENRICHMENT", "GRN + static_emb", "#E2EFDA")

    # ======== ATTENTION ========
    box(ax, 9.5, 3, 3, 1.0, "INTERPRETABLE\nATTENTION", "causal mask", "#DDEBF7")

    # ======== QUANTILE HEAD ========
    box(ax, 9.5, 1.5, 3, 1.0, "QUANTILE HEAD", "P10/P50/P90", "#FCE4D6")

    # ======== FORECAST ========
    box(ax, 9.5, 0.3, 3, 1.0, "DỰ BÁO 28 NGÀY", "phân vị", "#FCE4D6")

    # ======== INTERPRETABILITY PANEL ========
    box(ax, 12, 6.5, 2.5, 3.0, "DIỄN GIẢI",
        "past_vsn_weights\nfuture_vsn_weights\ntemporal_attention", "#F2F2F2")

    # ======== ARROWS — Main flow (blue) ========
    # Input → VSN
    arrow(ax, 2.5, 8.5, 2.5, 7.5)
    arrow(ax, 6, 8.5, 6, 7.5)
    # Static → VSN
    arrow(ax, 9.5, 8.5, 2.5, 7.5, color="#7F3F00", lw=1.3)
    arrow(ax, 9.5, 8.5, 6, 7.5, color="#7F3F00", lw=1.3)
    # VSN → LSTM
    arrow(ax, 2.5, 6.5, 2.5, 5.5)
    arrow(ax, 6, 6.5, 6, 5.5)
    # LSTM → Enrichment
    arrow(ax, 2.5, 4.5, 4, 3.5)
    arrow(ax, 6, 4.5, 4, 3.5)
    # Enrichment → Attention
    arrow(ax, 5.5, 3, 8, 3)
    # Attention → Quantile
    arrow(ax, 9.5, 2.5, 9.5, 2.0)
    # Quantile → Forecast
    arrow(ax, 9.5, 1.0, 9.5, 0.8)

    # ======== INTERPRETABILITY ARROWS (green) ========
    arrow(ax, 2.5, 7.5, 10.75, 7.8, color="#548235", style="-|>", lw=1.3)
    arrow(ax, 9.5, 7.5, 10.75, 7.8, color="#548235", style="-|>", lw=1.3)
    arrow(ax, 9.5, 3, 10.75, 7.0, color="#548235", style="-|>", lw=1.3)

    # Footer
    ax.text(7, -0.3,
            "Luồng: Past/Future → VSN → LSTM → Enrichment → Attention → Quantile Head\n"
            "Nhánh xanh lá: VSN & Attention sinh trọng số giải thích",
            ha="center", fontsize=8, color="#333333", style="italic")

    fig.savefig(os.path.join(FIG_DIR, "diagram_tft.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[+] Saved reports/figures/diagram_tft.png")

if __name__ == "__main__":
    draw_tft()
