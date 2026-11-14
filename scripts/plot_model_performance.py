#!/usr/bin/env python3
"""Create a slide-ready pastel comparison of baselines and the selected TFT model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
OUT_PATH = ROOT / "reports" / "figures" / "model_performance_comparison_pastel.png"

WHITE = "#FFFFFF"
TEXT = "#252525"
EDGE = "#444444"
GRID = "#D9D9D9"
BASELINE = "#B9C7D8"
BASELINE_DARK = "#7890AA"
TFT = "#C9B8E0"
TFT_DARK = "#765B98"
CYAN = "#A8E0E3"
BEIGE = "#FDE5C0"
RED = "#D94A45"


def add_panel_title(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=TEXT, pad=18)
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=8.5, color="#666666")


def read_results():
    summary = pd.read_csv(RESULT_DIR / "comparison_summary.csv")
    levels = pd.read_csv(RESULT_DIR / "comparison_all_levels.csv")
    return summary, levels


def main():
    summary, levels = read_results()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    display_names = {
        "ARIMA_ROBUST": "ARIMA Robust",
        "PROPHET": "Prophet",
        "LSTM": "LSTM",
        "NAIVE_7": "Naive-7",
        "NAIVE_28": "Naive-28",
        "WEEKDAY_MEAN_4": "Weekday Mean-4",
        "TFT_BU": "TFT Bottom-up",
        "TFT_SELECTED_STORE_CAL": "TFT Store_CAL",
    }
    overall_models = [
        "ARIMA_ROBUST", "NAIVE_7", "NAIVE_28", "PROPHET", "LSTM",
        "WEEKDAY_MEAN_4", "TFT_SELECTED_STORE_CAL",
    ]
    overall = summary[summary["split"].eq("test")].set_index("model")["WRMSSE_5levels"]
    overall = overall.rename(index=display_names).reindex([display_names[m] for m in overall_models])

    fig = plt.figure(figsize=(16, 9), facecolor=WHITE)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.08, 0.92], width_ratios=[1.05, 0.95], hspace=0.48, wspace=0.30)
    ax_overall = fig.add_subplot(grid[0, 0])
    ax_levels = fig.add_subplot(grid[0, 1])
    ax_effect = fig.add_subplot(grid[1, :])
    fig.suptitle("Model Performance Comparison: Baselines vs Selected TFT", fontsize=19, fontweight="bold", color=TEXT, y=0.97)
    fig.text(0.5, 0.932, "M5 full profile | 28-day test horizon | Lower WRMSSE is better", ha="center", fontsize=10, color="#666666")

    # Panel A: overall comparison.
    add_panel_title(ax_overall, "A. Overall Test WRMSSE-5", "Selected TFT model is highlighted in purple")
    y = np.arange(len(overall))
    colors = [TFT if name == "TFT Store_CAL" else BASELINE for name in overall.index]
    edges = [TFT_DARK if name == "TFT Store_CAL" else "none" for name in overall.index]
    bars = ax_overall.barh(y, overall.values, color=colors, edgecolor=edges, linewidth=1.8, height=0.62)
    ax_overall.set_yticks(y, overall.index, fontsize=9)
    ax_overall.invert_yaxis()
    ax_overall.set_xlim(0, 1.2)
    ax_overall.set_xlabel("WRMSSE-5", fontsize=9)
    ax_overall.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax_overall.set_axisbelow(True)
    for spine in ax_overall.spines.values():
        spine.set_visible(False)
    ax_overall.tick_params(axis="y", length=0)
    for bar, value, name in zip(bars, overall.values, overall.index):
        suffix = "  Selected" if name == "TFT Store_CAL" else ""
        weight = "bold" if name == "TFT Store_CAL" else "normal"
        ax_overall.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}{suffix}", va="center", fontsize=8.5, fontweight=weight, color=TEXT)

    weekday = float(overall["Weekday Mean-4"])
    store_cal = float(overall["TFT Store_CAL"])
    gain_baseline = (weekday - store_cal) / weekday * 100
    ax_overall.annotate(
        f"{gain_baseline:.1f}% lower error than\nstrongest simple baseline",
        xy=(store_cal, len(overall) - 1), xytext=(1.04, len(overall) - 2.35),
        fontsize=8, color=RED, ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.7),
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF4F2", ec=RED, lw=1),
    )

    # Panel B: hierarchy trade-off.
    add_panel_title(ax_levels, "B. Test WRMSSE by Hierarchy Level", "TFT Store_CAL is strongest at aggregate levels; LSTM is slightly better at L12")
    selected = {
        "LSTM": ("LSTM", BASELINE_DARK),
        "WEEKDAY_MEAN_4": ("Weekday Mean-4", BASELINE),
        "TFT_SELECTED_STORE_CAL": ("TFT Store_CAL", TFT),
    }
    level_order = ["L1", "L2", "L3", "L9", "L12"]
    x = np.arange(len(level_order))
    width = 0.23
    for idx, (model, (label, color)) in enumerate(selected.items()):
        frame = levels[(levels["split"].eq("test")) & (levels["model"].eq(model))].set_index("level")
        values = frame.loc[level_order, "WRMSSE"].to_numpy()
        bars = ax_levels.bar(x + (idx - 1) * width, values, width, label=label, color=color,
                             edgecolor=TFT_DARK if model == "TFT_SELECTED_STORE_CAL" else "none",
                             linewidth=1.3)
        for bar, value in zip(bars, values):
            ax_levels.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    ax_levels.set_xticks(x, level_order)
    ax_levels.set_ylim(0, 1.05)
    ax_levels.set_xlabel("Hierarchy level", fontsize=9)
    ax_levels.set_ylabel("WRMSSE", fontsize=9)
    ax_levels.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax_levels.set_axisbelow(True)
    for spine in ax_levels.spines.values():
        spine.set_visible(False)
    ax_levels.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.19), frameon=False, fontsize=8)

    # Panel C: calibration effect, comparing only raw TFT to the selected calibration.
    add_panel_title(ax_effect, "C. Store Calibration Effect", "Calibration corrects aggregate store bias while leaving SKU-level performance almost unchanged")
    effect_levels = [("Overall WRMSSE-5", None), ("L3 Store", "L3"), ("L12 SKU", "L12")]
    bu_overall = float(summary[(summary["split"].eq("test")) & (summary["model"].eq("TFT_BU"))]["WRMSSE_5levels"].iloc[0])
    store_overall = float(summary[(summary["split"].eq("test")) & (summary["model"].eq("TFT_SELECTED_STORE_CAL"))]["WRMSSE_5levels"].iloc[0])
    bu_levels = levels[(levels["split"].eq("test")) & (levels["model"].eq("TFT_BU"))].set_index("level")["WRMSSE"]
    store_levels = levels[(levels["split"].eq("test")) & (levels["model"].eq("TFT_SELECTED_STORE_CAL"))].set_index("level")["WRMSSE"]
    raw_values = [bu_overall, float(bu_levels["L3"]), float(bu_levels["L12"])]
    calibrated_values = [store_overall, float(store_levels["L3"]), float(store_levels["L12"])]
    effect_y = np.arange(len(effect_levels))[::-1]
    ax_effect.set_xlim(0.48, 0.90)
    ax_effect.set_ylim(-0.65, 2.65)
    ax_effect.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax_effect.set_axisbelow(True)
    for spine in ax_effect.spines.values():
        spine.set_visible(False)
    ax_effect.set_yticks(effect_y, [label for label, _ in effect_levels], fontsize=10)
    ax_effect.tick_params(axis="y", length=0)
    ax_effect.set_xlabel("WRMSSE", fontsize=9)
    for y_pos, raw, calibrated in zip(effect_y, raw_values, calibrated_values):
        ax_effect.plot([raw, calibrated], [y_pos, y_pos], color="#989898", linewidth=2.4, zorder=1)
        ax_effect.scatter(raw, y_pos, s=82, color=CYAN, edgecolor=EDGE, linewidth=1, zorder=2, label="TFT Bottom-up" if y_pos == 2 else None)
        ax_effect.scatter(calibrated, y_pos, s=92, color=TFT, edgecolor=TFT_DARK, linewidth=1.4, zorder=3, label="TFT Store_CAL" if y_pos == 2 else None)
        reduction = (raw - calibrated) / raw * 100
        ax_effect.text(raw + 0.008, y_pos + 0.13, f"{raw:.3f}", fontsize=8, color="#555555")
        ax_effect.text(calibrated - 0.008, y_pos - 0.23, f"{calibrated:.3f}", fontsize=8, color=TFT_DARK, ha="right", fontweight="bold")
        label_y = y_pos + (0.34 if reduction < 1 else 0.13)
        ax_effect.text((raw + calibrated) / 2, label_y, f"{reduction:.1f}% lower", fontsize=8, color=RED, ha="center", fontweight="bold")
    ax_effect.legend(loc="lower right", frameon=False, fontsize=9)
    ax_effect.text(0.49, -0.52, "Key result: Store calibration reduces overall error by 17.2% and L3 error by 23.8%, but is not designed to fix individual SKU errors.",
                   fontsize=8.5, color=TEXT, bbox=dict(boxstyle="round,pad=0.45", fc=BEIGE, ec=EDGE, lw=0.8))

    fig.savefig(OUT_PATH, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
