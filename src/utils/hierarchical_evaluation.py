"""
Module: hierarchical_evaluation.py
Mục đích: Đánh giá đa cấp độ (Hierarchical Evaluation) cho bài toán M5:
           RMSE, MAE, MAPE, WMAPE, sMAPE, RMSSE, WRMSSE.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from .metrics import (
    calc_rmse,
    calc_mae,
    calc_mape,
    calc_wmape,
    calc_smape,
    calc_rmsse,
    forecast_metrics,
)


def evaluate_hierarchical_level(
    y_true_mat,
    y_pred_mat,
    y_train_mat=None,
    weights=None,
) -> Dict[str, float]:
    y = np.asarray(y_true_mat, dtype=np.float64)
    p = np.clip(np.asarray(y_pred_mat, dtype=np.float64), 0, None)
    n_keys = y.shape[0]

    ms = []
    for i in range(n_keys):
        tr = None if y_train_mat is None else np.asarray(y_train_mat, dtype=np.float64)[i]
        ms.append(forecast_metrics(y[i], p[i], y_train=tr))

    out: Dict[str, float] = {}
    for k in ("RMSE", "MAE", "sMAPE (%)", "Bias"):
        out[k] = float(np.mean([m[k] for m in ms]))

    mapes = np.asarray([m["MAPE (%)"] for m in ms], dtype=np.float64)
    nan_m = int(np.isnan(mapes).sum())
    out["n_zero_mape_keys"] = nan_m
    out["MAPE (%)"] = float(np.nanmean(mapes)) if nan_m < len(mapes) else float("nan")

    wmapes = np.asarray([m["WMAPE (%)"] for m in ms], dtype=np.float64)
    nan_w = int(np.isnan(wmapes).sum())
    out["n_zero_keys"] = nan_w
    out["n_keys_scored"] = int(len(wmapes) - nan_w)
    out["WMAPE (%)"] = float(np.nanmean(wmapes)) if nan_w < len(wmapes) else float("nan")

    if y_train_mat is not None:
        rmsses = np.asarray([m["RMSSE"] for m in ms], dtype=np.float64)
        out["RMSSE"] = float(np.nanmean(rmsses)) if np.any(~np.isnan(rmsses)) else float("nan")

        r = np.asarray([m["RMSSE"] for m in ms], dtype=np.float64)
        if weights is None:
            w = np.ones(n_keys, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.shape[0] != n_keys:
                raise ValueError("weights must have one value per series")

        valid = ~np.isnan(r)
        idx = np.where(valid)[0]
        raw_w = w[idx]
        denom = raw_w.sum()
        norm_w = (raw_w / denom) if denom > 0 else raw_w
        contrib = np.where(raw_w > 0, r[idx] * norm_w, 0.0)
        out["WRMSSE"] = float(np.sum(contrib))
    else:
        out["RMSSE"] = float("nan")
        out["WRMSSE"] = float("nan")

    y64 = y.reshape(-1)
    p64 = p.reshape(-1)
    out["pooled_WMAPE (%)"] = calc_wmape(y64, p64)
    out["pooled_MAPE (%)"] = calc_mape(y64, p64)
    out["pooled_MAE"] = calc_mae(y64, p64)
    out["total_actual"] = float(y64.sum())
    out["total_pred"] = float(p64.sum())
    out["pred_actual_ratio"] = float(p64.sum() / y64.sum()) if y64.sum() != 0 else float("nan")
    out["zero_true_keys"] = int((y.sum(axis=1) == 0).sum())
    out["n_keys"] = int(n_keys)
    return out


def evaluate_hierarchical_forecasts(
    y_true_sku: np.ndarray,
    y_pred_sku: np.ndarray,
    y_true_store: np.ndarray,
    y_pred_store: np.ndarray,
) -> pd.DataFrame:
    results = {}

    y_t_sku_flat = np.asarray(y_true_sku).reshape(-1)
    y_p_sku_flat = np.clip(np.asarray(y_pred_sku).reshape(-1), 0, None)
    results["Level 12 (SKU - Item/Store)"] = {
        "RMSE": calc_rmse(y_t_sku_flat, y_p_sku_flat),
        "MAE": calc_mae(y_t_sku_flat, y_p_sku_flat),
        "MAPE (%)": calc_mape(y_t_sku_flat, y_p_sku_flat),
        "WMAPE (%)": calc_wmape(y_t_sku_flat, y_p_sku_flat),
        "sMAPE (%)": calc_smape(y_t_sku_flat, y_p_sku_flat),
    }

    y_t_store_flat = np.asarray(y_true_store).reshape(-1)
    y_p_store_flat = np.clip(np.asarray(y_pred_store).reshape(-1), 0, None)
    results["Level 3 (Store Level)"] = {
        "RMSE": calc_rmse(y_t_store_flat, y_p_store_flat),
        "MAE": calc_mae(y_t_store_flat, y_p_store_flat),
        "MAPE (%)": calc_mape(y_t_store_flat, y_p_store_flat),
        "WMAPE (%)": calc_wmape(y_t_store_flat, y_p_store_flat),
        "sMAPE (%)": calc_smape(y_t_store_flat, y_p_store_flat),
    }

    y_t_national = np.sum(y_true_store, axis=0) if np.asarray(y_true_store).ndim > 1 else y_true_store
    y_p_national = np.sum(y_pred_store, axis=0) if np.asarray(y_pred_store).ndim > 1 else y_pred_store
    y_t_national = np.asarray(y_t_national).reshape(-1)
    y_p_national = np.clip(np.asarray(y_p_national).reshape(-1), 0, None)
    results["Level 1 (Total System / State)"] = {
        "RMSE": calc_rmse(y_t_national, y_p_national),
        "MAE": calc_mae(y_t_national, y_p_national),
        "MAPE (%)": calc_mape(y_t_national, y_p_national),
        "WMAPE (%)": calc_wmape(y_t_national, y_p_national),
        "sMAPE (%)": calc_smape(y_t_national, y_p_national),
    }

    return pd.DataFrame(results).T
