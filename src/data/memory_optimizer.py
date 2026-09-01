"""
Module: memory_optimizer.py
Mục đích: Tối ưu hóa dung lượng RAM cho DataFrame bằng cách ép kiểu dữ liệu
          (Downcasting numeric types: float64 -> float32/float16, int64 -> int16/int8)
"""

import numpy as np
import pandas as pd
from typing import Tuple


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Duyệt qua tất cả các cột của DataFrame và giảm kiểu dữ liệu số
    để tối ưu bộ nhớ RAM mà không làm mất mát độ chính xác dữ liệu.

    Args:
        df (pd.DataFrame): DataFrame đầu vào cần tối ưu.
        verbose (bool): In thông tin trước và sau khi giảm dung lượng.

    Returns:
        pd.DataFrame: DataFrame sau khi đã tối ưu kiểu dữ liệu.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f"[-] Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB (Giảm: {reduction:.1f}%)")

    return df
