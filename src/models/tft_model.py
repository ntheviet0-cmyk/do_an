"""
Module: tft_model.py
Mục đích: Triển khai hoàn chỉnh kiến trúc mô hình đề xuất: Temporal Fusion Transformer (TFT) bằng PyTorch.
          Bao gồm:
          1. TFTTimeSeriesDataset chuẩn hóa và tạo batch cho Static, Past, và Known-Future features.
          2. TemporalFusionTransformer (Mạng PyTorch hoàn chỉnh).
          3. TFTForecaster (Pipeline huấn luyện, dự báo phân vị và trích xuất trọng số giải thích).
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

from .transformer_components import (
    GLU,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
    QuantileLoss
)


class TFTTimeSeriesDataset(Dataset):
    """
    Dataset PyTorch chuyên biệt cho kiến trúc Temporal Fusion Transformer:
    - past_inputs: [Seq_len, num_past_features] (Doanh số lịch sử, Lags, Rolling stats)
    - future_inputs: [Seq_len + Horizon, num_future_features] (Biến biết trước: Giá, Giảm giá, SNAP, Lễ hội, Chu kỳ)
    - targets: [Horizon] (Doanh số cần dự báo)
    """
    def __init__(
        self,
        past_features: np.ndarray,
        future_features: np.ndarray,
        targets: np.ndarray,
        seq_len: int = 56,
        horizon: int = 28
    ):
        self.past_features = torch.tensor(past_features, dtype=torch.float32)
        self.future_features = torch.tensor(future_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
        self.seq_len = seq_len
        self.horizon = horizon
        self.total_len = seq_len + horizon
        self.num_samples = len(past_features) - self.total_len + 1

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Past inputs trong cửa sổ [t - seq_len : t]
        past_x = self.past_features[idx : idx + self.seq_len]
        
        # 2. Known-Future inputs trong toàn bộ cửa sổ [t - seq_len : t + horizon]
        future_x = self.future_features[idx : idx + self.total_len]
        
        # 3. Target trong tương lai [t : t + horizon]
        y = self.targets[idx + self.seq_len : idx + self.total_len]
        
        return past_x, future_x, y


class TemporalFusionTransformer(nn.Module):
    """
    Kiến trúc Temporal Fusion Transformer (TFT) cốt lõi:
    - Variable Selection Network cho Past & Future features
    - LSTM Sequence Encoder-Decoder
    - Interpretable Multi-Head Temporal Self-Attention
    - Quantile Output Projection (P10, P50, P90)
    """
    def __init__(
        self,
        num_past_vars: int,
        num_future_vars: int,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        seq_len: int = 56,
        horizon: int = 28
    ):
        super(TemporalFusionTransformer, self).__init__()
        self.num_past_vars = num_past_vars
        self.num_future_vars = num_future_vars
        self.d_model = d_model
        self.seq_len = seq_len
        self.horizon = horizon
        self.quantiles = quantiles

        # 1. Input Linear Embeddings cho từng biến số
        self.past_embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_past_vars)])
        self.future_embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_future_vars)])

        # 2. Variable Selection Networks (VSN)
        self.past_vsn = VariableSelectionNetwork(
            num_vars=num_past_vars,
            input_dim=d_model,
            hidden_dim=d_model,
            dropout=dropout
        )
        
        self.future_vsn = VariableSelectionNetwork(
            num_vars=num_future_vars,
            input_dim=d_model,
            hidden_dim=d_model,
            dropout=dropout
        )

        # 3. LSTM Sequence Encoder & Decoder
        self.lstm_encoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.lstm_decoder = nn.LSTM(d_model, d_model, batch_first=True)

        # 4. Gated Residual Connections xung quanh LSTM
        self.post_lstm_glu = GLU(d_model, d_model)
        self.post_lstm_norm = nn.LayerNorm(d_model)

        # 5. Interpretable Multi-Head Temporal Self-Attention
        self.multihead_attn = InterpretableMultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout)
        self.post_attn_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)

        # 6. Quantile Output Heads
        self.output_head = nn.Linear(d_model, len(quantiles))

    def forward(
        self,
        past_x: torch.Tensor,
        future_x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass với khả năng trích xuất toàn bộ Attention weights và Variable Importance.

        Args:
            past_x: [Batch, Seq_len, Num_past_vars]
            future_x: [Batch, Seq_len + Horizon, Num_future_vars]

        Returns:
            quantiles_out: [Batch, Horizon, Num_quantiles]
            interpretability_dict: Chứa Past VSN weights, Future VSN weights, và Temporal Attention map
        """
        batch_size = past_x.size(0)

        # --- A. BIẾN ĐỔI VÀ CHỌN LỌC BIẾN (VSN) ---
        # 1. Past VSN
        past_embeds = [emb(past_x[:, :, i:i+1]) for i, emb in enumerate(self.past_embeddings)]
        past_flat = torch.cat(past_embeds, dim=-1)
        past_vsn_out, past_vsn_weights = self.past_vsn(past_flat, past_embeds) # [Batch, Seq_len, d_model]

        # 2. Future VSN
        future_embeds = [emb(future_x[:, :, i:i+1]) for i, emb in enumerate(self.future_embeddings)]
        future_flat = torch.cat(future_embeds, dim=-1)
        future_vsn_out, future_vsn_weights = self.future_vsn(future_flat, future_embeds) # [Batch, Seq_len + Horizon, d_model]

        # --- B. LSTM ENCODER-DECODER ---
        enc_out, (hn, cn) = self.lstm_encoder(past_vsn_out)
        dec_out, _ = self.lstm_decoder(future_vsn_out[:, self.seq_len:], (hn, cn))

        # Ghép chuỗi biểu diễn toàn thời gian
        lstm_combined = torch.cat([enc_out, dec_out], dim=1) # [Batch, Seq_len + Horizon, d_model]
        vsn_combined = torch.cat([past_vsn_out, future_vsn_out[:, self.seq_len:]], dim=1)

        # Gated Skip connection quanh LSTM
        lstm_gated = self.post_lstm_glu(lstm_combined)
        temporal_features = self.post_lstm_norm(vsn_combined + lstm_gated)

        # --- C. TEMPORAL SELF-ATTENTION ---
        # Áp dụng Attention trên toàn bộ chuỗi thời gian.
        # Sử dụng mặt nạ nhân quả (causal mask): vị trí i chỉ được phép chú ý
        # đến các vị trí j <= i, ngăn việc "nhìn trước" thông tin tương lai
        # (chống rò rỉ dữ liệu trong khâu Attention).
        total_len = temporal_features.size(1)
        causal_mask = torch.tril(torch.ones(total_len, total_len, device=temporal_features.device))

        attn_out, attn_weights = self.multihead_attn(
            q=temporal_features,
            k=temporal_features,
            v=temporal_features,
            mask=causal_mask
        )
        
        # Skip connection qua Attention + GRN
        enriched_features = self.post_attn_grn(temporal_features + attn_out)

        # --- D. DỰ BÁO PHÂN VỊ (CHỈ LẤY PHẦN HORIZON) ---
        horizon_features = enriched_features[:, self.seq_len:] # [Batch, Horizon, d_model]
        quantiles_out = self.output_head(horizon_features)     # [Batch, Horizon, Num_quantiles]

        # Lưu lại thông tin giải thích mô hình
        interpretability_dict = {
            "past_vsn_weights": past_vsn_weights,       # [Batch, Seq_len, Num_past_vars]
            "future_vsn_weights": future_vsn_weights,   # [Batch, Seq_len + Horizon, Num_future_vars]
            "temporal_attention": attn_weights          # [Batch, Seq_len + Horizon, Seq_len + Horizon]
        }

        return quantiles_out, interpretability_dict


class TFTForecaster:
    """
    Pipeline huấn luyện và suy diễn của Temporal Fusion Transformer.
    """
    def __init__(
        self,
        past_feature_cols: List[str],
        future_feature_cols: List[str],
        target_col: str = "total_sales",
        seq_len: int = 56,
        horizon: int = 28,
        d_model: int = 64,
        num_heads: int = 4,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 40,
        device: Optional[str] = None
    ):
        self.past_cols = past_feature_cols
        self.future_cols = future_feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.horizon = horizon
        self.d_model = d_model
        self.num_heads = num_heads
        self.quantiles = quantiles
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.past_scaler = StandardScaler()
        self.future_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.model: Optional[TemporalFusionTransformer] = None

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "TFTForecaster":
        print(f"[+] Huấn luyện Temporal Fusion Transformer (TFT) trên thiết bị: {self.device}")

        # 1. Chuẩn hóa dữ liệu
        past_train = self.past_scaler.fit_transform(train_df[self.past_cols].fillna(0).values)
        future_train = self.future_scaler.fit_transform(train_df[self.future_cols].fillna(0).values)
        target_train = self.target_scaler.fit_transform(train_df[[self.target_col]].fillna(0).values).flatten()

        train_dataset = TFTTimeSeriesDataset(
            past_features=past_train,
            future_features=future_train,
            targets=target_train,
            seq_len=self.seq_len,
            horizon=self.horizon
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if val_df is not None:
            combined_val = pd.concat([train_df.iloc[-self.seq_len:], val_df], ignore_index=True)
            past_val = self.past_scaler.transform(combined_val[self.past_cols].fillna(0).values)
            future_val = self.future_scaler.transform(combined_val[self.future_cols].fillna(0).values)
            target_val = self.target_scaler.transform(combined_val[[self.target_col]].fillna(0).values).flatten()
            val_dataset = TFTTimeSeriesDataset(past_val, future_val, target_val, self.seq_len, self.horizon)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # 2. Khởi tạo mạng TFT
        self.model = TemporalFusionTransformer(
            num_past_vars=len(self.past_cols),
            num_future_vars=len(self.future_cols),
            d_model=self.d_model,
            num_heads=self.num_heads,
            quantiles=self.quantiles,
            seq_len=self.seq_len,
            horizon=self.horizon
        ).to(self.device)

        criterion = QuantileLoss(quantiles=self.quantiles)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

        best_val_loss = float("inf")
        best_state = None

        # 3. Training Loop
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            for p_x, f_x, y in train_loader:
                p_x, f_x, y = p_x.to(self.device), f_x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                q_preds, _ = self.model(p_x, f_x)
                loss = criterion(q_preds, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * len(p_x)

            train_loss /= len(train_dataset)

            # Validation
            val_loss = train_loss
            if val_loader is not None and len(val_dataset) > 0:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for p_x, f_x, y in val_loader:
                        p_x, f_x, y = p_x.to(self.device), f_x.to(self.device), y.to(self.device)
                        q_preds, _ = self.model(p_x, f_x)
                        loss = criterion(q_preds, y)
                        val_loss += loss.item() * len(p_x)
                val_loss /= len(val_dataset)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())

            if epoch % 5 == 0 or epoch == self.epochs:
                print(f"Epoch [{epoch:02d}/{self.epochs:02d}] - Quantile Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print("[✓] Huấn luyện TFT hoàn tất!")
        return self

    def predict_quantiles(
        self,
        last_seq_df: pd.DataFrame,
        future_df: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Dự báo các phân vị (P10, P50, P90) và trích xuất trọng số giải thích.

        Args:
            last_seq_df: Seq_len ngày quan sát gần nhất (chứa cả past_cols và future_cols)
            future_df: Horizon ngày tương lai cần dự báo (chứa future_cols)

        Returns:
            quantiles_pred: Mảng [Horizon, 3] tương ứng (P10, P50_Median, P90)
            interpretability_dict: Dict chứa attention weights và VSN weights
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được fit!")

        self.model.eval()

        # Chuẩn bị Past X: [1, Seq_len, Num_past_vars]
        p_scaled = self.past_scaler.transform(last_seq_df[self.past_cols].fillna(0).values)
        past_tensor = torch.tensor(p_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Chuẩn bị Future X: [1, Seq_len + Horizon, Num_future_vars]
        full_future_df = pd.concat([last_seq_df, future_df], ignore_index=True)
        f_scaled = self.future_scaler.transform(full_future_df[self.future_cols].fillna(0).values)
        future_tensor = torch.tensor(f_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_out_scaled, interp = self.model(past_tensor, future_tensor)
            # q_out_scaled: [1, Horizon, Num_quantiles]
            q_out_scaled = q_out_scaled.squeeze(0).cpu().numpy()

        # Inverse transform từng phân vị
        q_preds = np.zeros_like(q_out_scaled)
        for i in range(len(self.quantiles)):
            q_preds[:, i] = self.target_scaler.inverse_transform(q_out_scaled[:, i:i+1]).flatten()

        q_preds = np.clip(q_preds, 0, None)

        # Chuyển đổi tensor giải thích sang numpy
        interp_numpy = {
            "past_vsn_weights": interp["past_vsn_weights"].squeeze(0).cpu().numpy(),
            "future_vsn_weights": interp["future_vsn_weights"].squeeze(0).cpu().numpy(),
            "temporal_attention": interp["temporal_attention"].squeeze(0).cpu().numpy()
        }

        return q_preds, interp_numpy
