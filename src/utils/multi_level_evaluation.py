"""
Module: multi_level_evaluation.py
Mục đích: Đánh giá RMSE, MAE, MAPE, WMAPE, sMAPE trên nhiều cấp độ tổng hợp
           (SKU-level, Store-level, National-level) theo hướng bottom-up.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.utils.metrics import evaluate_forecast


def aggregate_bottom_up(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    sku_df: pd.DataFrame,
    group_col: Optional[str] = None
) -> tuple:
    sku_to_group = sku_df.set_index("id")[group_col].to_dict() if group_col else None

    groups: Dict[str, List[str]] = {}
    for sku_id in y_true_dict.keys():
        g = sku_to_group[sku_id] if sku_to_group else "NATIONAL"
        groups.setdefault(g, []).append(sku_id)

    y_true_agg, y_pred_agg, labels = [], [], []
    for g, sku_ids in groups.items():
        yt = np.sum([y_true_dict[s] for s in sku_ids], axis=0)
        yp = np.sum([y_pred_dict[s] for s in sku_ids], axis=0)
        y_true_agg.append(yt)
        y_pred_agg.append(yp)
        labels.append(g)

    return np.array(y_true_agg), np.array(y_pred_agg), labels


def evaluate_hierarchical(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    sku_df: pd.DataFrame,
    model_name: str
) -> pd.DataFrame:
    rows = []

    levels = [
        ("SKU", None),
        ("Store", "store_id"),
        ("National", None),
    ]

    for level_name, group_col in levels:
        if level_name == "SKU":
            yt_list = [y_true_dict[k] for k in y_true_dict]
            yp_list = [y_pred_dict[k] for k in y_pred_dict]
            n = len(yt_list)
        else:
            yt_arr, yp_arr, labels = aggregate_bottom_up(
                y_true_dict, y_pred_dict, sku_df, group_col
            )
            yt_list = [yt_arr[i] for i in range(len(labels))]
            yp_list = [yp_arr[i] for i in range(len(labels))]
            n = len(labels)

        metrics_all = [evaluate_forecast(yt, yp) for yt, yp in zip(yt_list, yp_list)]

        rows.append({
            "Level": level_name,
            "Model": model_name,
            "N_Series": n,
            "RMSE": float(np.mean([m["RMSE"] for m in metrics_all])),
            "MAE": float(np.mean([m["MAE"] for m in metrics_all])),
            "MAPE (%)": float(np.nanmean([m["MAPE (%)"] for m in metrics_all])),
            "WMAPE": float(np.nanmean([m["WMAPE (%)"] for m in metrics_all])),
            "sMAPE (%)": float(np.mean([m["sMAPE (%)"] for m in metrics_all])),
        })

    return pd.DataFrame(rows)
