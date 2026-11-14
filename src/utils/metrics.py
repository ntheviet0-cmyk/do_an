"""
Module: metrics.py
Mục đích: Cung cấp các độ đo đánh giá độ chính xác chuẩn cho bài toán dự báo chuỗi thời gian bán lẻ:
           RMSE, MAE, MAPE, WMAPE, sMAPE, RMSSE.
"""

import numpy as np
from typing import Dict


def _as_float(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).ravel()


def calc_rmse(y_true, y_pred) -> float:
    a, b = _as_float(y_true), _as_float(y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else float("nan")


def calc_mae(y_true, y_pred) -> float:
    a, b = _as_float(y_true), _as_float(y_pred)
    return float(np.mean(np.abs(a - b))) if a.size else float("nan")


def calc_mape(y_true, y_pred) -> float:
    a, p = _as_float(y_true), _as_float(y_pred)
    mask = (a != 0) & ~np.isnan(a)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100.0)


def calc_wmape(y_true, y_pred) -> float:
    a, b = _as_float(y_true), _as_float(y_pred)
    den = np.abs(a).sum()
    return float("nan") if den == 0 else float(np.abs(a - b).sum() / den * 100)


def calc_bias(y_true, y_pred) -> float:
    a, b = _as_float(y_true), _as_float(y_pred)
    return float(np.mean(b - a)) if a.size else float("nan")


def calc_smape(y_true, y_pred, eps: float = 1e-8) -> float:
    a, b = _as_float(y_true), _as_float(y_pred)
    den = (np.abs(a) + np.abs(b)) / 2.0 + eps
    return float(np.mean(np.abs(b - a) / den) * 100) if a.size else float("nan")


def calc_rmsse(y_true, y_pred, y_train) -> float:
    a, p, tr = _as_float(y_true), _as_float(y_pred), _as_float(y_train)
    nz = np.where(tr != 0)[0]
    if len(nz) == 0:
        scale_sq = 0.0
    else:
        tr_act = tr[nz[0]:]
        scale_sq = float(np.mean(np.diff(tr_act) ** 2)) if len(tr_act) > 1 else 0.0
    rmse = float(np.sqrt(np.mean((a - p) ** 2))) if a.size else float("nan")
    if scale_sq <= 0 or np.isnan(scale_sq):
        return 0.0 if rmse == 0 else float("inf")
    return float(rmse / np.sqrt(scale_sq))


def forecast_metrics(y_true, y_pred, y_train=None, label: str = "") -> Dict[str, float]:
    a, p = _as_float(y_true), _as_float(y_pred)
    if a.shape != p.shape:
        raise ValueError(f"{label}: shape mismatch {a.shape} vs {p.shape}")
    den = np.abs(a).sum()
    ratio = float(p.sum() / a.sum()) if a.sum() != 0 else float("nan")
    res = {
        "n_obs": int(a.size),
        "total_actual": float(a.sum()),
        "total_pred": float(p.sum()),
        "pred_actual_ratio": ratio,
        "WMAPE (%)": calc_wmape(a, p),
        "MAPE (%)": calc_mape(a, p),
        "MAE": calc_mae(a, p),
        "RMSE": calc_rmse(a, p),
        "Bias": calc_bias(a, p),
        "sMAPE (%)": calc_smape(a, p),
        "zero_actual_frac": float(np.mean(a == 0)) if a.size else float("nan"),
        "zero_denominator": bool(den == 0),
    }
    if y_train is not None:
        res["RMSSE"] = calc_rmsse(a, p, y_train)
    return res


def evaluate_forecast(y_true, y_pred, y_train=None) -> Dict[str, float]:
    m = forecast_metrics(y_true, np.clip(np.asarray(y_pred, dtype=np.float64), 0, None), y_train=y_train)
    keys = ["RMSE", "MAE", "MAPE (%)", "WMAPE (%)", "sMAPE (%)"]
    if y_train is not None:
        keys.append("RMSSE")
    return {k: m[k] for k in keys}
