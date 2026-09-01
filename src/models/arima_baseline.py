"""
Module: arima_baseline.py
Mục đích: Xây dựng mô hình Baseline ARIMA / SARIMA cho bài toán dự báo chuỗi thời gian đơn biến.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA


class ARIMABaseline:
    def __init__(self, order: Tuple[int, int, int] = (7, 1, 1)):
        """
        Khởi tạo mô hình ARIMA.
        
        Args:
            order (Tuple[int, int, int]): (p, d, q) - Mặc định (7, 1, 1) để bắt tính chu kỳ tuần.
        """
        self.order = order
        self.model_fitted = None

    def fit(self, train_series: np.ndarray) -> "ARIMABaseline":
        """
        Fit mô hình ARIMA trên chuỗi thời gian lịch sử.
        
        Args:
            train_series (np.ndarray): Mảng 1D chuỗi doanh số huấn luyện.
        """
        # Fit mô hình sử dụng statsmodels
        model = ARIMA(train_series, order=self.order)
        self.model_fitted = model.fit()
        return self

    def predict(self, horizon: int = 28) -> np.ndarray:
        """
        Dự báo cho H bước thời gian tiếp theo.
        
        Args:
            horizon (int): Độ dài khoảng dự báo (H=28 ngày).

        Returns:
            np.ndarray: Giá trị dự báo 1D.
        """
        if self.model_fitted is None:
            raise ValueError("Mô hình chưa được fit! Vui lòng gọi fit() trước.")
        
        forecast = self.model_fitted.forecast(steps=horizon)
        # Nhu cầu tồn kho không âm
        return np.clip(np.asarray(forecast), 0, None)
