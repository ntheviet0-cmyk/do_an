"""Leakage-safe WRMSSE evaluation for M5 levels L1/L2/L3/L9/L12."""

from __future__ import annotations

from typing import Mapping

import numpy as np


LEVELS = ("L1", "L2", "L3", "L9", "L12")


def group_sum(values: np.ndarray, groups: np.ndarray, n_groups: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("values must have shape [series, time]")
    if groups.ndim != 1 or len(groups) != len(values):
        raise ValueError("groups must contain one code per series")
    if n_groups is None:
        n_groups = int(groups.max()) + 1
    out = np.zeros((n_groups, values.shape[1]), dtype=np.float64)
    np.add.at(out, groups, values)
    return out


def rmsse_scale(y_history: np.ndarray) -> np.ndarray:
    """Return M5 RMSSE denominators after removing pre-launch leading zeros."""
    y_history = np.asarray(y_history, dtype=np.float64)
    if y_history.ndim != 2:
        raise ValueError("y_history must have shape [series, time]")

    scales = np.full(len(y_history), np.nan, dtype=np.float64)
    for row_index, row in enumerate(y_history):
        nonzero = np.flatnonzero(row != 0)
        if not len(nonzero):
            continue
        active = row[nonzero[0]:]
        if len(active) < 2:
            continue
        scales[row_index] = np.mean(np.diff(active) ** 2)
    return scales


def build_contexts(
    groups_by_level: Mapping[str, np.ndarray],
    sales_history: np.ndarray,
    price_history: np.ndarray,
    weight_window: int = 28,
) -> dict[str, dict[str, np.ndarray | int | str]]:
    """Precompute scale and pre-forecast dollar weights for the five levels."""
    sales_history = np.asarray(sales_history)
    price_history = np.asarray(price_history)
    if sales_history.shape != price_history.shape:
        raise ValueError("sales_history and price_history must have the same shape")
    if sales_history.ndim != 2 or sales_history.shape[1] == 0:
        raise ValueError("history must have shape [series, positive_time]")
    if weight_window <= 0:
        raise ValueError("weight_window must be positive")

    start = max(0, sales_history.shape[1] - weight_window)
    recent_sales = sales_history[:, start:].astype(np.float64)
    recent_price = np.nan_to_num(price_history[:, start:], nan=0.0).astype(np.float64)
    series_revenue = np.sum(recent_sales * recent_price, axis=1)
    contexts = {}
    for level in LEVELS:
        groups = np.asarray(groups_by_level[level], dtype=np.int64)
        n_groups = int(groups.max()) + 1
        grouped_history = group_sum(sales_history, groups, n_groups)
        scale = rmsse_scale(grouped_history)
        revenue = np.bincount(groups, weights=series_revenue, minlength=n_groups).astype(np.float64)
        valid = np.isfinite(scale) & (scale > 1e-12)
        weights = np.where(valid, np.maximum(revenue, 0.0), 0.0)
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weight_sum
        elif valid.any():
            weights[valid] = 1.0 / valid.sum()

        contexts[level] = {
            "level": level,
            "groups": groups,
            "n_groups": n_groups,
            "scale": scale,
            "weights": weights,
            "valid": valid,
        }
    return contexts


def evaluate_level(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    context: Mapping[str, np.ndarray | int | str],
) -> dict[str, float | int | str]:
    """Evaluate one hierarchy level using a pre-forecast evaluation context."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    groups = np.asarray(context["groups"], dtype=np.int64)
    n_groups = int(context["n_groups"])
    scale = np.asarray(context["scale"], dtype=np.float64)
    weights = np.asarray(context["weights"], dtype=np.float64)
    valid = np.asarray(context["valid"], dtype=bool)
    yt = group_sum(y_true, groups, n_groups)
    yp = group_sum(y_pred, groups, n_groups)

    mse = np.mean((yt - yp) ** 2, axis=1)
    rmsse = np.full(n_groups, np.nan, dtype=np.float64)
    rmsse[valid] = np.sqrt(mse[valid] / scale[valid])
    wrmsse = float(np.sum(weights[valid] * rmsse[valid]))

    error = np.abs(yt - yp)
    actual_by_group = np.abs(yt).sum(axis=1)
    group_ok = actual_by_group > 1e-12
    group_bias = np.full(n_groups, np.nan, dtype=np.float64)
    group_bias[group_ok] = (yp - yt).sum(axis=1)[group_ok] / actual_by_group[group_ok]
    positive = np.abs(yt) > 0

    return {
        "level": str(context["level"]),
        "n_keys": n_groups,
        "n_valid_scales": int(valid.sum()),
        "n_obs": int(yt.size),
        "WRMSSE": wrmsse,
        "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "MAE": float(error.mean()),
        "WMAPE": float(error.sum() / max(np.abs(yt).sum(), 1e-12)),
        "MAPE": float(np.mean(error[positive] / np.abs(yt[positive]))) if positive.any() else np.nan,
        "Bias": float((yp - yt).sum() / max(np.abs(yt).sum(), 1e-12)),
        "Ratio": float(yp.sum() / max(yt.sum(), 1e-12)),
        "MeanAbsGroupBias": float(np.nanmean(np.abs(group_bias))) if group_ok.any() else np.nan,
        "P90AbsGroupBias": float(np.nanpercentile(np.abs(group_bias), 90)) if group_ok.any() else np.nan,
    }


def evaluate_wrmsse5(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    contexts: Mapping[str, Mapping[str, np.ndarray | int | str]],
) -> tuple[list[dict[str, float | int | str]], float]:
    rows = [evaluate_level(y_true, y_pred, contexts[level]) for level in LEVELS]
    return rows, float(np.mean([float(row["WRMSSE"]) for row in rows]))

