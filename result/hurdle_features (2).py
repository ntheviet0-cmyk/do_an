"""Leakage-safe local and hierarchical features for the M5 HurdleTFT."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


LOCAL_STATE_COLS = (
    "nonzero_rate_7",
    "nonzero_rate_28",
    "nonzero_rate_56",
    "log_zero_run",
    "has_ever_sold",
    "log_age_since_first_sale",
    "log_mean_positive_28",
    "listed_rate_28",
    "log_last_positive_units",
)

HIER_CHANNELS = (
    "log_sales",
    "log_roll_mean_7",
    "log_roll_mean_28",
    "trend_7_28",
    "listed_ratio",
)


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Inclusive causal rolling sum along time for a two-dimensional panel."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("values must have shape [series, time]")
    if window <= 0:
        raise ValueError("window must be positive")

    cumulative = np.cumsum(values, axis=1, dtype=np.float32)
    result = cumulative.copy()
    if window < values.shape[1]:
        result[:, window:] -= cumulative[:, :-window]
    return result


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = rolling_sum(values, window)
    counts = np.minimum(np.arange(1, values.shape[1] + 1), window).astype(np.float32)
    result /= counts[None, :]
    return result


def build_local_state_features(
    sales: np.ndarray,
    price_available: np.ndarray,
) -> np.ndarray:
    """Build L12 occurrence, lifecycle and past-listing features.

    Every value at day ``t`` depends only on observations at or before ``t``.
    Float16 storage keeps the full 30,490-series panel practical; windows are
    converted back to float32 by the dataset.
    """
    sales = np.asarray(sales, dtype=np.float32)
    price_available = np.asarray(price_available, dtype=bool)
    if sales.ndim != 2 or price_available.shape != sales.shape:
        raise ValueError("sales and price_available must share shape [series, time]")
    if np.any(sales < 0):
        raise ValueError("sales must be non-negative")

    n_series, n_days = sales.shape
    output = np.empty((n_series, n_days, len(LOCAL_STATE_COLS)), dtype=np.float16)
    positive = sales > 0
    positive_float = positive.astype(np.float32)

    for name, window in (
        ("nonzero_rate_7", 7),
        ("nonzero_rate_28", 28),
        ("nonzero_rate_56", 56),
    ):
        output[:, :, LOCAL_STATE_COLS.index(name)] = rolling_mean(
            positive_float, window
        ).astype(np.float16)

    run = np.zeros(n_series, dtype=np.int16)
    first_sale = np.full(n_series, -1, dtype=np.int16)
    last_positive = np.zeros(n_series, dtype=np.float32)
    zero_run_idx = LOCAL_STATE_COLS.index("log_zero_run")
    ever_idx = LOCAL_STATE_COLS.index("has_ever_sold")
    age_idx = LOCAL_STATE_COLS.index("log_age_since_first_sale")
    last_positive_idx = LOCAL_STATE_COLS.index("log_last_positive_units")
    for day in range(n_days):
        sold = positive[:, day]
        first_sale[(first_sale < 0) & sold] = day
        run = np.where(sold, 0, np.minimum(run + 1, 364)).astype(np.int16)
        last_positive = np.where(sold, sales[:, day], last_positive)
        seen = first_sale >= 0
        age = np.where(seen, day - first_sale + 1, 0)
        output[:, day, zero_run_idx] = np.log1p(run).astype(np.float16)
        output[:, day, ever_idx] = seen.astype(np.float16)
        output[:, day, age_idx] = np.log1p(age).astype(np.float16)
        output[:, day, last_positive_idx] = np.log1p(last_positive).astype(np.float16)

    positive_count_28 = rolling_sum(positive_float, 28)
    positive_sales_28 = rolling_sum(sales, 28)
    mean_positive_28 = np.divide(
        positive_sales_28,
        positive_count_28,
        out=np.zeros_like(positive_sales_28),
        where=positive_count_28 > 0,
    )
    output[:, :, LOCAL_STATE_COLS.index("log_mean_positive_28")] = np.log1p(
        mean_positive_28
    ).astype(np.float16)
    output[:, :, LOCAL_STATE_COLS.index("listed_rate_28")] = rolling_mean(
        price_available.astype(np.float32), 28
    ).astype(np.float16)
    return output


def group_sum(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    groups = np.asarray(groups, dtype=np.int64)
    if values.ndim != 2 or groups.ndim != 1 or len(groups) != len(values):
        raise ValueError("invalid values/groups shape")
    if len(groups) == 0 or np.any(groups < 0):
        raise ValueError("groups must contain non-negative codes")
    result = np.zeros((int(groups.max()) + 1, values.shape[1]), dtype=np.float32)
    np.add.at(result, groups, values)
    return result


def build_hierarchy_context(
    sales: np.ndarray,
    price_available: np.ndarray,
    groups_by_level: Mapping[str, np.ndarray],
    levels: tuple[str, ...] = ("L1", "L2", "L3", "L9"),
) -> dict[str, np.ndarray]:
    """Build compact parent histories without broadcasting them to L12."""
    sales = np.asarray(sales, dtype=np.float32)
    price_available = np.asarray(price_available, dtype=bool)
    if sales.ndim != 2 or price_available.shape != sales.shape:
        raise ValueError("sales and price_available must share shape [series, time]")

    contexts = {}
    for level in levels:
        groups = np.asarray(groups_by_level[level], dtype=np.int64)
        aggregate = group_sum(sales, groups)
        roll_7 = rolling_mean(aggregate, 7)
        roll_28 = rolling_mean(aggregate, 28)
        trend = np.clip((roll_7 - roll_28) / (roll_28 + 1.0), -5.0, 5.0)
        listed_count = group_sum(price_available.astype(np.float32), groups)
        group_size = np.bincount(groups, minlength=aggregate.shape[0]).astype(np.float32)
        listed_ratio = listed_count / np.maximum(group_size[:, None], 1.0)
        contexts[level] = np.stack(
            [
                np.log1p(aggregate),
                np.log1p(roll_7),
                np.log1p(roll_28),
                trend,
                listed_ratio,
            ],
            axis=-1,
        ).astype(np.float32)
    return contexts


def fit_feature_scaler(values: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature-wise moments strictly before the one-based train cutoff."""
    values = np.asarray(values)
    if values.ndim < 2 or not 0 < train_end <= values.shape[1]:
        raise ValueError("invalid train_end")
    axes = tuple(range(values.ndim - 1))
    train = values[:, :train_end].astype(np.float32)
    mean = train.mean(axis=axes, dtype=np.float64).astype(np.float32)
    std = train.std(axis=axes, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def hierarchy_feature_names(
    levels: tuple[str, ...] = ("L1", "L2", "L3", "L9"),
) -> list[str]:
    return [f"{level}_{channel}" for level in levels for channel in HIER_CHANNELS]
