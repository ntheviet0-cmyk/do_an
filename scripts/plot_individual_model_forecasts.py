#!/usr/bin/env python3
"""Create separate Actual vs Forecast images, one per model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_forecast_vs_actual import COLORS, HORIZON, TEXT, WHITE, aggregate, hierarchy_keys, load_predictions


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "figures"
GRID = "#D9D9D9"
ACTUAL = "#242424"

LINEAGE = {
    "L1": "All M5",
    "L2": "TX",
    "L3": "TX_1",
    "L9": "TX_1 | FOODS_3",
    "L12": "FOODS_3_586_TX_1_evaluation",
}
MODELS = {
    "TFT Store_CAL": {
        "filename": "forecast_actual_TFT.png",
        "color": COLORS["TFT Store_CAL"],
    },
    "LSTM": {
        "filename": "forecast_actual_LSTM.png",
        "color": COLORS["LSTM"],
    },
    "Prophet": {
        "filename": "forecast_actual_Prophet.png",
        "color": COLORS["Prophet"],
    },
    "ARIMA": {
        "filename": "forecast_actual_ARIMA.png",
        "color": COLORS["ARIMA"],
    },
}


def style_axis(ax, level, label):
    ax.set_title(f"{level}: {label}", loc="left", fontsize=10, fontweight="bold", color=TEXT, pad=7)
    ax.set_xlim(1, HORIZON)
    ax.set_ylim(ax.get_ylim())
    ax.set_xticks([1, 7, 14, 21, 28])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def get_shared_y_limits(lineages):
    """Use one y-range per level for all dashboards to avoid visual exaggeration."""
    limits = {}
    for level in LINEAGE:
        all_actual = []
        all_forecasts = []
        for model_name, lineage in lineages.items():
            all_actual.append(lineage[level]["Actual"])
            all_forecasts.append(lineage[level][model_name])
        low = min(np.min(all_actual), np.min(all_forecasts))
        high = max(np.max(all_actual), np.max(all_forecasts))
        span = max(high - low, 1.0)
        limits[level] = (low - span * 0.06, high + span * 0.06)
    return limits


def render_image(model_name, model_config, lineage, y_limits):
    days = np.arange(1, HORIZON + 1)
    fig = plt.figure(figsize=(16, 9), facecolor=WHITE)
    fig.suptitle(f"Actual vs {model_name} Forecast", fontsize=19, fontweight="bold", color=TEXT, y=0.975)
    fig.text(
        0.5,
        0.935,
        "M5 hierarchy lineage | 28-day test horizon | Actual shown in black; forecast in color",
        ha="center",
        fontsize=9,
        color="#666666",
    )

    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], width_ratios=[1, 1], hspace=0.55, wspace=0.24)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, :]),
    ]

    levels = list(LINEAGE)
    for ax, level in zip(axes, levels):
        values = lineage[level]
        ax.plot(days, values["Actual"], color=ACTUAL, linewidth=2.6, label="Actual", zorder=3)
        ax.plot(days, values[model_name], color=model_config["color"], linewidth=2.3, label=model_name, zorder=2)
        style_axis(ax, level, LINEAGE[level])
        ax.set_ylim(y_limits[level])
        ax.set_ylabel("Demand", fontsize=8)
        ax.set_xlabel("Test horizon day", fontsize=8)
        ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.subplots_adjust(top=0.89, bottom=0.08, left=0.06, right=0.98)
    out_path = OUT_DIR / model_config["filename"]
    fig.savefig(out_path, dpi=250, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ids, arrays = load_predictions()
    keys = hierarchy_keys(ids)
    lineages = {}
    for model_name, model_config in MODELS.items():
        model_arrays = {"Actual": arrays["Actual"], model_name: arrays[model_name]}
        lineage = {}
        for level, label in LINEAGE.items():
            labels, grouped = aggregate(model_arrays, keys[level])
            lineage[level] = grouped[label]
        lineages[model_name] = lineage

    y_limits = get_shared_y_limits(lineages)
    paths = []
    for model_name, model_config in MODELS.items():
        paths.append(render_image(model_name, model_config, lineages[model_name], y_limits))

    print("Created:")
    for path in paths:
        print(path.name)


if __name__ == "__main__":
    main()
