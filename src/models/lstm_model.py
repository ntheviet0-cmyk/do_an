"""
Module: lstm_model.py
Mục đích: Xây dựng mô hình Deep Learning Multivariate LSTM bằng PyTorch.
          Hỗ trợ đầu vào đa biến (Sales + Exogenous Variables) và dự báo trực tiếp H bước (Direct Multi-Horizon Forecasting).
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict


class TimeSeriesSlidingWindowDataset(Dataset):
    """
    Dataset PyTorch tạo các cửa sổ trượt (Sliding Windows) cho bài toán dự báo chuỗi thời gian đa biến.
    Lookback Window = L (quá khứ), Horizon = H (tương lai cần dự báo).
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 56, horizon: int = 28):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.total_samples = len(features) - seq_len - horizon + 1

    def __len__(self) -> int:
        return max(0, self.total_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # X: [seq_len, num_features]
        x = self.features[idx : idx + self.seq_len]
        # Y: [horizon]
        y = self.targets[idx + self.seq_len : idx + self.seq_len + self.horizon]
        return x, y


class MultivariateLSTMNet(nn.Module):
    """
    Kiến trúc mạng Muti-Layer LSTM cho bài toán Multivariate Time Series Forecasting.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, horizon: int = 28, dropout: float = 0.2):
        super(MultivariateLSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully-connected head để map từ hidden state cuối cùng sang vector H bước dự báo
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch_size, Seq_len, Input_dim]
        lstm_out, (hn, cn) = self.lstm(x)
        # Lấy hidden state của bước thời gian cuối cùng: hn[-1] -> [Batch_size, Hidden_dim]
        last_hidden = hn[-1]
        out = self.fc(last_hidden) # [Batch_size, Horizon]
        return out


class LSTMBaseline:
    """
    Pipeline đóng gói huấn luyện và dự báo cho Multivariate LSTM.
    """
    def __init__(
        self,
        feature_cols: List[str],
        target_col: str = "sales",
        seq_len: int = 56,
        horizon: int = 28,
        hidden_dim: int = 64,
        num_layers: int = 2,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        device: Optional[str] = None
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.model: Optional[MultivariateLSTMNet] = None

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "LSTMBaseline":
        """
        Fit mô hình LSTM với chuẩn hóa dữ liệu và Early Stopping.
        """
        print(f"[+] Huấn luyện LSTM trên thiết bị: {self.device}")
        
        # 1. Chuẩn hóa dữ liệu đầu vào
        train_feats = self.scaler.fit_transform(train_df[self.feature_cols].fillna(0).values)
        train_targets = self.target_scaler.fit_transform(train_df[[self.target_col]].fillna(0).values).flatten()

        train_dataset = TimeSeriesSlidingWindowDataset(train_feats, train_targets, self.seq_len, self.horizon)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if val_df is not None:
            # Ghép phần đuôi của train để tạo đủ seq_len cho val
            combined_val = pd.concat([train_df.iloc[-self.seq_len:], val_df], ignore_index=True)
            val_feats = self.scaler.transform(combined_val[self.feature_cols].fillna(0).values)
            val_targets = self.target_scaler.transform(combined_val[[self.target_col]].fillna(0).values).flatten()
            val_dataset = TimeSeriesSlidingWindowDataset(val_feats, val_targets, self.seq_len, self.horizon)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # 2. Khởi tạo mạng và Optimizer
        input_dim = len(self.feature_cols)
        self.model = MultivariateLSTMNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            horizon=self.horizon
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

        best_loss = float("inf")
        best_weights = None

        # 3. Training Loop
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)

            train_loss /= len(train_dataset)

            # Validation step
            val_loss = train_loss
            if val_loader is not None and len(val_dataset) > 0:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        preds = self.model(batch_x)
                        loss = criterion(preds, batch_y)
                        val_loss += loss.item() * len(batch_x)
                val_loss /= len(val_dataset)

            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())

            if epoch % 10 == 0 or epoch == self.epochs:
                print(f"Epoch [{epoch:02d}/{self.epochs:02d}] - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        print("[✓] Huấn luyện LSTM hoàn tất!")
        return self

    def predict(self, last_seq_df: pd.DataFrame) -> np.ndarray:
        """
        Dự báo H bước tương lai dựa trên chuỗi quan sát gần nhất (seq_len bước cuối cùng).

        Args:
            last_seq_df (pd.DataFrame): DataFrame chứa đúng seq_len bước thời gian gần nhất.

        Returns:
            np.ndarray: Mảng 1D độ dài horizon dự báo.
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được fit!")

        self.model.eval()
        feats = self.scaler.transform(last_seq_df[self.feature_cols].fillna(0).values)
        input_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self.device) # [1, Seq_len, Input_dim]

        with torch.no_grad():
            pred_scaled = self.model(input_tensor).cpu().numpy().flatten()

        # Inverse transform về thang đo ban đầu
        pred = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        return np.clip(pred, 0, None)
