"""
Module: prophet_baseline.py
Mục đích: Xây dựng mô hình Baseline Facebook Prophet hỗ trợ các yếu tố ngoại sinh (Exogenous Regressors).
"""

import numpy as np
import pandas as pd
from typing import List, Optional

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetBaseline:
    def __init__(
        self,
        exogenous_cols: Optional[List[str]] = None,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False
    ):
        """
        Khởi tạo Facebook Prophet với các biến ngoại sinh.

        Args:
            exogenous_cols (List[str], optional): Danh sách các cột biến ngoại sinh (vd: sell_price, snap, event).
            yearly_seasonality (bool): Bắt chu kỳ năm.
            weekly_seasonality (bool): Bắt chu kỳ tuần.
        """
        if not PROPHET_AVAILABLE:
            raise ImportError(
                "Thư viện 'prophet' chưa được cài đặt. "
                "Bạn có thể cài đặt bằng lệnh: pip install prophet"
            )
        
        self.exogenous_cols = exogenous_cols or []
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            interval_width=0.95
        )
        
        # Đăng ký các biến ngoại sinh vào mô hình Prophet
        for col in self.exogenous_cols:
            self.model.add_regressor(col)
            
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "ProphetBaseline":
        """
        Fit mô hình Prophet.

        Args:
            train_df (pd.DataFrame): DataFrame chứa cột 'ds' (ngày), 'y' (target) và các cột exogenous.
        """
        prophet_df = train_df[["date", "sales"] + self.exogenous_cols].copy()
        prophet_df.rename(columns={"date": "ds", "sales": "y"}, inplace=True)
        
        self.model.fit(prophet_df)
        self.is_fitted = True
        return self

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        """
        Dự báo tương lai với các biến ngoại sinh đã biết trước trong tập test.

        Args:
            future_df (pd.DataFrame): DataFrame chứa cột 'date' và các cột exogenous cho 28 ngày tương lai.

        Returns:
            np.ndarray: Mảng giá trị dự báo yhat (đã clip >= 0).
        """
        if not self.is_fitted:
            raise ValueError("Mô hình chưa được fit! Vui lòng gọi fit() trước.")

        future_prophet = future_df[["date"] + self.exogenous_cols].copy()
        future_prophet.rename(columns={"date": "ds"}, inplace=True)
        
        forecast = self.model.predict(future_prophet)
        yhat = forecast["yhat"].values
        return np.clip(yhat, 0, None)
