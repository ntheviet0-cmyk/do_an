#!/usr/bin/env python3
"""Create slide-ready actual-versus-forecast plots for M5 hierarchy levels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
OUT_DIR = ROOT / "reports" / "figures"
HORIZON = 28

WHITE = "#FFFFFF"
TEXT = "#252525"
GRID = "#D9D9D9"
ACTUAL = "#242424"
COLORS = {
    "TFT Store_CAL": "#765B98",
    "Prophet": "#62A67C",
    "LSTM": "#4C91C9",
    "ARIMA": "#E59654",
}


def load_predictions():
    """Load each model and prove that its rows refer to the same L12 series."""
    tft = np.load(RESULT_DIR / "predictions_test (2).npz", allow_pickle=False)
    ids = tft["ids"]
    arrays = {
        "TFT Store_CAL": tft["y_pred_store_cal"].astype(np.float64),
        "Actual": tft["y_true"].astype(np.float64),
    }
    for label, filename in {
        "Prophet": "prophet_predictions.npz",
        "LSTM": "lstm_predictions.npz",
        "ARIMA": "arima_predictions.npz",
    }.items():
        artifact = np.load(RESULT_DIR / filename, allow_pickle=False)
        if not np.array_equal(ids, artifact["ids"]):
            raise ValueError(f"series_id order differs for {label}; forecasts cannot be compared safely.")
        arrays[label] = artifact["test_pred"].astype(np.float64)

    for label, values in arrays.items():
        if values.shape != (len(ids), HORIZON):
            raise ValueError(f"Unexpected shape for {label}: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values found in {label}.")
    return ids, arrays


def hierarchy_keys(ids):
    """Derive M5 L2/L3/L9 keys from ids such as HOBBIES_1_001_CA_1_evaluation."""
    states = []
    stores = []
    store_departments = []
    for series_id in ids:
        parts = series_id.split("_")
        if len(parts) < 6:
            raise ValueError(f"Unexpected M5 series_id: {series_id}")
        state = parts[-3]
        store = "_".join(parts[-3:-1])
        department = "_".join(parts[:2])
        states.append(state)
        stores.append(store)
        store_departments.append(f"{store} | {department}")
    return {
        "L1": np.full(len(ids), "All M5", dtype=object),
        "L2": np.asarray(states, dtype=object),
        "L3": np.asarray(stores, dtype=object),
        "L9": np.asarray(store_departments, dtype=object),
        "L12": np.asarray(ids, dtype=object),
    }


def aggregate(arrays, keys):
    """Bottom-up aggregate actual and every model for one hierarchy level."""
    labels = sorted(np.unique(keys).tolist())
    grouped = {}
    for label in labels:
        mask = keys == label
        grouped[label] = {name: values[mask].sum(axis=0) for name, values in arrays.items()}
    return labels, grouped


def style_axis(ax, title):
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=TEXT, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(1, HORIZON)
    ax.set_xticks([1, 7, 14, 21, 28])
    ax.tick_params(labelsize=8)


def draw_series(ax, values, title):
    days = np.arange(1, HORIZON + 1)
    ax.plot(days, values["Actual"], color=ACTUAL, linewidth=2.5, label="Actual", zorder=5)
    for name, color in COLORS.items():
        ax.plot(days, values[name], color=color, linewidth=1.7, label=name, alpha=0.95)
    style_axis(ax, title)


def add_legend(fig):
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))


def save_grid(level, labels, grouped, ncols, filename):
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.25 * nrows), squeeze=False)
    fig.patch.set_facecolor(WHITE)
    fig.suptitle(f"Actual vs Forecast: {level}", fontsize=16, fontweight="bold", color=TEXT, y=0.985)
    for ax, label in zip(axes.flat, labels):
        draw_series(ax, grouped[label], label)
    for ax in axes.flat[len(labels):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Test horizon day", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Demand", fontsize=8)
    add_legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    fig.savefig(OUT_DIR / filename, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def representative_l9(labels, grouped):
    """Show the six L9 groups with the largest observed test demand."""
    return sorted(labels, key=lambda label: grouped[label]["Actual"].sum(), reverse=True)[:6]


def representative_l12(ids, arrays):
    """Select deterministic examples for intermittent, stable, and spike demand."""
    actual = arrays["Actual"]
    total = actual.sum(axis=1)
    zero_rate = (actual == 0).mean(axis=1)
    mean = actual.mean(axis=1)
    std = actual.std(axis=1)
    coefficient_of_variation = std / np.maximum(mean, 1e-6)
    peak_ratio = actual.max(axis=1) / np.maximum(mean, 1e-6)

    nonzero = total > 0
    intermittent = np.where(nonzero & (zero_rate >= 0.70))[0]
    stable = np.where((zero_rate <= 0.15) & (total >= np.quantile(total, 0.75)))[0]
    spike = np.where((total >= np.quantile(total, 0.75)) & (zero_rate <= 0.50))[0]
    if not len(intermittent) or not len(stable) or not len(spike):
        raise ValueError("Could not find representative L12 series for all three demand patterns.")

    selected = [
        ("Intermittent / zero-heavy", intermittent[np.argmax(total[intermittent])]),
        ("Stable demand", stable[np.argmin(coefficient_of_variation[stable])]),
        ("Demand spike", spike[np.argmax(peak_ratio[spike])]),
    ]
    titles = []
    grouped = {}
    for pattern, index in selected:
        label = str(ids[index])
        titles.append(f"{pattern}\n{label}")
        grouped[titles[-1]] = {name: values[index] for name, values in arrays.items()}
    return titles, grouped


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids, arrays = load_predictions()
    keys = hierarchy_keys(ids)

    l1_labels, l1 = aggregate(arrays, keys["L1"])
    l2_labels, l2 = aggregate(arrays, keys["L2"])
    l3_labels, l3 = aggregate(arrays, keys["L3"])
    l9_labels, l9 = aggregate(arrays, keys["L9"])

    save_grid("L1: Total M5 Demand", l1_labels, l1, 1, "forecast_vs_actual_L1.png")
    save_grid("L2: State Demand", l2_labels, l2, 3, "forecast_vs_actual_L2_states.png")
    save_grid("L3: Store Demand", l3_labels, l3, 5, "forecast_vs_actual_L3_stores.png")

    selected_l9 = representative_l9(l9_labels, l9)
    save_grid("L9: Store x Department Demand", selected_l9, l9, 3,
              "forecast_vs_actual_L9_representative.png")

    l12_titles, l12 = representative_l12(ids, arrays)
    save_grid("L12: Representative SKU-Store Demand", l12_titles, l12, 3,
              "forecast_vs_actual_L12_representative.png")

    print("Created:")
    for name in [
        "forecast_vs_actual_L1.png",
        "forecast_vs_actual_L2_states.png",
        "forecast_vs_actual_L3_stores.png",
        "forecast_vs_actual_L9_representative.png",
        "forecast_vs_actual_L12_representative.png",
    ]:
        print(name)


if __name__ == "__main__":
    main()
