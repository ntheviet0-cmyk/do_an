#!/usr/bin/env python3
"""Create one slide-readable hierarchy and metric dashboard for each M5 model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_forecast_vs_actual import COLORS, HORIZON, RESULT_DIR, TEXT, WHITE, aggregate, hierarchy_keys, load_predictions


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "figures"
GRID = "#D9D9D9"
ACTUAL = "#242424"
LEVELS = ["L1", "L2", "L3", "L9", "L12"]

# One fixed lineage makes all model dashboards visually comparable from aggregate to SKU-store.
LINEAGE = {
    "L1": "All M5",
    "L2": "TX",
    "L3": "TX_1",
    "L9": "TX_1 | FOODS_3",
    "L12": "FOODS_3_586_TX_1_evaluation",
}
MODELS = {
    "TFT Store_CAL": {"metric_name": "TFT_SELECTED_STORE_CAL", "color": COLORS["TFT Store_CAL"], "filename": "dashboard_TFT_STORE_CAL.png"},
    "Prophet": {"metric_name": "PROPHET", "color": COLORS["Prophet"], "filename": "dashboard_Prophet.png"},
    "LSTM": {"metric_name": "LSTM", "color": COLORS["LSTM"], "filename": "dashboard_LSTM.png"},
    "ARIMA": {"metric_name": "ARIMA_ROBUST", "color": COLORS["ARIMA"], "filename": "dashboard_ARIMA.png"},
}


def load_lineage(arrays, ids):
    keys = hierarchy_keys(ids)
    selected = {}
    for level, label in LINEAGE.items():
        labels, grouped = aggregate(arrays, keys[level])
        if label not in labels:
            raise ValueError(f"Expected lineage key missing at {level}: {label}")
        selected[level] = grouped[label]
    return selected


def style_forecast_axis(ax, level, label):
    ax.set_title(f"{level}  |  {label}", loc="left", fontsize=10, fontweight="bold", color=TEXT, pad=6)
    ax.set_xlim(1, HORIZON)
    ax.set_xticks([1, 7, 14, 21, 28])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_metric_bar(ax, metrics, column, title, formatter, signed=False):
    values = metrics[column].to_numpy(dtype=float)
    colors = ["#C9B8E0"] * len(LEVELS)
    bars = ax.bar(LEVELS, values, color=colors, edgecolor="#765B98", linewidth=0.8)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=TEXT, pad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if signed:
        extent = max(np.abs(values).max() * 1.32, 0.01)
        ax.set_ylim(-extent, extent)
        label_offset = extent * 0.055
    else:
        ymax = max(values.max() * 1.24, 0.01)
        ax.set_ylim(0, ymax)
        label_offset = ymax * 0.035
    for bar, value in zip(bars, values):
        y = value + label_offset if value >= 0 else value - label_offset
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y, formatter(value),
                ha="center", va=va, fontsize=7.5, color=TEXT)


def draw_metric_table(ax, metrics):
    ax.axis("off")
    display = metrics[["level", "RMSE", "MAE", "WMAPE", "Bias"]].copy()
    display["RMSE"] = display["RMSE"].map(lambda value: f"{value:.2f}")
    display["MAE"] = display["MAE"].map(lambda value: f"{value:.2f}")
    display["WMAPE"] = display["WMAPE"].map(lambda value: f"{value * 100:.1f}%")
    display["Bias"] = display["Bias"].map(lambda value: f"{value * 100:+.1f}%")
    table = ax.table(
        cellText=display.values,
        colLabels=["Level", "RMSE", "MAE", "WMAPE", "Bias"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#E1E1E1")
        if row == 0:
            cell.set_facecolor("#EDE6F5")
            cell.set_text_props(weight="bold", color=TEXT)
        elif row % 2:
            cell.set_facecolor("#F8F8F8")
    ax.set_title("Test Metrics by Hierarchy Level", loc="left", fontsize=10, fontweight="bold", color=TEXT, pad=8)


def render_dashboard(model_name, model_config, lineage, metrics):
    color = model_config["color"]
    fig = plt.figure(figsize=(16, 12), facecolor=WHITE)
    grid = fig.add_gridspec(5, 2, width_ratios=[1.55, 1.0], hspace=0.64, wspace=0.28)
    fig.suptitle(f"{model_name}: Forecast vs Actual and Test Metrics", fontsize=19, fontweight="bold", color=TEXT, y=0.985)
    fig.text(0.5, 0.958, "M5 full profile | 28-day test horizon | Fixed hierarchy lineage for comparable dashboards",
             ha="center", fontsize=9, color="#666666")

    forecast_axes = []
    days = np.arange(1, HORIZON + 1)
    for row, level in enumerate(LEVELS):
        ax = fig.add_subplot(grid[row, 0])
        values = lineage[level]
        ax.plot(days, values["Actual"], color=ACTUAL, linewidth=2.4, label="Actual", zorder=3)
        ax.plot(days, values[model_name], color=color, linewidth=2.0, label=model_name, zorder=2)
        style_forecast_axis(ax, level, LINEAGE[level])
        if row == len(LEVELS) - 1:
            ax.set_xlabel("Test horizon day", fontsize=8)
        ax.set_ylabel("Demand", fontsize=8)
        forecast_axes.append(ax)
    forecast_axes[0].legend(loc="upper right", frameon=False, fontsize=8)

    right = grid[:, 1].subgridspec(4, 1, height_ratios=[1, 1, 1, 1.25], hspace=0.64)
    ax_wrmsse = fig.add_subplot(right[0])
    ax_wmape = fig.add_subplot(right[1])
    ax_bias = fig.add_subplot(right[2])
    ax_table = fig.add_subplot(right[3])
    draw_metric_bar(ax_wrmsse, metrics, "WRMSSE", "WRMSSE (lower is better)", lambda value: f"{value:.3f}")
    draw_metric_bar(ax_wmape, metrics, "WMAPE", "WMAPE (lower is better)", lambda value: f"{value * 100:.1f}%")
    draw_metric_bar(ax_bias, metrics, "Bias", "Bias (negative = under-forecast)", lambda value: f"{value * 100:+.1f}%", signed=True)
    ax_bias.axhline(0, color="#777777", linewidth=0.9)
    draw_metric_table(ax_table, metrics)

    fig.subplots_adjust(top=0.91, bottom=0.06, left=0.06, right=0.98)
    out_path = OUT_DIR / model_config["filename"]
    fig.savefig(out_path, dpi=250, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids, arrays = load_predictions()
    lineage = load_lineage(arrays, ids)
    all_metrics = pd.read_csv(RESULT_DIR / "comparison_all_levels.csv")

    paths = []
    for model_name, config in MODELS.items():
        metrics = all_metrics[
            all_metrics["split"].eq("test") & all_metrics["model"].eq(config["metric_name"])
        ].set_index("level").loc[LEVELS].reset_index()
        if len(metrics) != len(LEVELS):
            raise ValueError(f"Missing hierarchy metrics for {model_name}.")
        paths.append(render_dashboard(model_name, config, lineage, metrics))

    print("Created:")
    for path in paths:
        print(path.name)


if __name__ == "__main__":
    main()
