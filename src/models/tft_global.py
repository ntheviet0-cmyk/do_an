"""
Module: tft_global.py
Mục đích: Mở rộng TFT thành mô hình GLOBAL (huấn luyện đồng thời nhiều chuỗi thời gian)
          với STATIC CONTEXT (embedding cửa hàng/bang/ngành) — đúng tinh thần
          "dự báo đa cấp độ có yếu tố ngoại sinh". Đây là bước làm TFT vượt baseline.

So với TFT单 chuỗi (tft_model.py):
  - Thêm StaticContext: embedding các biến tĩnh (store_id, state_id, cat_id...) -> vector [d_model].
  - Static context điều kiện hóa:
      (1) Variable Selection Network (VSN) -> chọn lọc biến khác nhau theo từng chuỗi.
      (2) Static Enrichment GRN -> làm giàu đặc trưng thời gian trước Attention.
  - Dataset lấy cửa sổ trượt từ TẤT CẢ các chuỗi (store), gộp chung thành 1 batch huấn luyện.
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


class StaticContext(nn.Module):
    """
    Ánh xạ các biến tĩnh (phân loại) thành embedding rồi cộng gộp thành 1 vector.
    Ví dụ: store_id=CA_1, state_id=CA -> hai embedding [d_model] -> tổng [d_model].
    """
    def __init__(self, cardinalities: Dict[str, int], d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(card + 1, d_model, padding_idx=0)
            for name, card in cardinalities.items()
        })

    def forward(self, static_idx: Dict[str, torch.Tensor]) -> torch.Tensor:
        # static_idx[name]: [Batch]
        embs = [self.embeddings[name](static_idx[name]) for name in self.embeddings]
        return sum(embs)  # [Batch, d_model]


class TFTGlobalDataset(Dataset):
    """
    Dataset lấy cửa sổ từ NHIỀU chuỗi. Mỗi mẫu: (past, future, static_idx, target).
    """
    def __init__(self, past: np.ndarray, future: np.ndarray, static: np.ndarray,
                 targets: np.ndarray):
        # past:   [N, L, P]
        # future: [N, L+H, F]
        # static: [N, S] (chỉ số categorical)
        # targets:[N, H]
        self.past = torch.tensor(past, dtype=torch.float32)
        self.future = torch.tensor(future, dtype=torch.float32)
        self.static = torch.tensor(static, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.past)

    def __getitem__(self, idx):
        return self.past[idx], self.future[idx], self.static[idx], self.targets[idx]


class TemporalFusionTransformerGlobal(nn.Module):
    """
    TFT có static context. Khác TFT đơn chuỗi ở chỗ:
      - nhận thêm đầu vào static_idx (batch các chỉ số categorical).
      - static_emb điều kiện hóa VSN (chọn biến) và Static-Enrichment GRN.
    """
    def __init__(
        self,
        num_past_vars: int,
        num_future_vars: int,
        static_cardinalities: Dict[str, int],
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        seq_len: int = 56,
        horizon: int = 28
    ):
        super().__init__()
        self.num_past_vars = num_past_vars
        self.num_future_vars = num_future_vars
        self.d_model = d_model
        self.seq_len = seq_len
        self.horizon = horizon
        self.quantiles = quantiles

        self.static_ctx = StaticContext(static_cardinalities, d_model)

        self.past_embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_past_vars)])
        self.future_embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_future_vars)])

        self.past_vsn = VariableSelectionNetwork(
            num_vars=num_past_vars, input_dim=d_model, hidden_dim=d_model,
            dropout=dropout, context_dim=d_model
        )
        self.future_vsn = VariableSelectionNetwork(
            num_vars=num_future_vars, input_dim=d_model, hidden_dim=d_model,
            dropout=dropout, context_dim=d_model
        )

        self.lstm_encoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.lstm_decoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.post_lstm_glu = GLU(d_model, d_model)
        self.post_lstm_norm = nn.LayerNorm(d_model)

        # Static enrichment GRN (điều kiện hóa bởi static_emb)
        self.static_enrichment = GatedResidualNetwork(
            input_dim=d_model, hidden_dim=d_model, output_dim=d_model,
            dropout=dropout, context_dim=d_model
        )

        self.multihead_attn = InterpretableMultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout=dropout)
        self.post_attn_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)

        self.output_head = nn.Linear(d_model, len(quantiles))

    def forward(self, past_x, future_x, static_idx):
        batch_size = past_x.size(0)
        static_emb = self.static_ctx(static_idx)  # [B, d_model]

        past_embeds = [emb(past_x[:, :, i:i+1]) for i, emb in enumerate(self.past_embeddings)]
        past_flat = torch.cat(past_embeds, dim=-1)
        past_vsn_out, past_vsn_weights = self.past_vsn(past_flat, past_embeds, static_emb)

        future_embeds = [emb(future_x[:, :, i:i+1]) for i, emb in enumerate(self.future_embeddings)]
        future_flat = torch.cat(future_embeds, dim=-1)
        future_vsn_out, future_vsn_weights = self.future_vsn(future_flat, future_embeds, static_emb)

        enc_out, (hn, cn) = self.lstm_encoder(past_vsn_out)
        dec_out, _ = self.lstm_decoder(future_vsn_out[:, self.seq_len:], (hn, cn))

        lstm_combined = torch.cat([enc_out, dec_out], dim=1)
        vsn_combined = torch.cat([past_vsn_out, future_vsn_out[:, self.seq_len:]], dim=1)

        lstm_gated = self.post_lstm_glu(lstm_combined)
        temporal_features = self.post_lstm_norm(vsn_combined + lstm_gated)

        # Static enrichment (làm giàu bằng thông tin tĩnh của chuỗi)
        enriched = self.static_enrichment(temporal_features, static_emb)

        total_len = enriched.size(1)
        causal_mask = torch.tril(torch.ones(total_len, total_len, device=enriched.device))
        attn_out, attn_weights = self.multihead_attn(
            q=enriched, k=enriched, v=enriched, mask=causal_mask
        )
        final_features = self.post_attn_grn(enriched + attn_out)

        horizon_features = final_features[:, self.seq_len:]
        quantiles_out = self.output_head(horizon_features)

        interp = {
            "past_vsn_weights": past_vsn_weights,
            "future_vsn_weights": future_vsn_weights,
            "temporal_attention": attn_weights,
            "static_emb": static_emb
        }
        return quantiles_out, interp


class TFTGlobalForecaster:
    """
    Pipeline huấn luyện TFT GLOBAL trên nhiều chuỗi (store/SKU).
    """
    def __init__(
        self,
        past_feature_cols: List[str],
        future_feature_cols: List[str],
        static_cat_cols: List[str],
        target_col: str = "total_sales",
        seq_len: int = 56,
        horizon: int = 28,
        d_model: int = 64,
        num_heads: int = 4,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 40,
        device: Optional[str] = None
    ):
        self.past_cols = past_feature_cols
        self.future_cols = future_feature_cols
        self.static_cols = static_cat_cols
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
        self.cat_maps: Dict[str, Dict[str, int]] = {}
        self.cardinalities: Dict[str, int] = {}
        self.model: Optional[TemporalFusionTransformerGlobal] = None

    # ---------- helpers ----------
    def _build_windows(self, df: pd.DataFrame):
        """Trả về (past_arr, future_arr, target_arr) cho 1 chuỗi đã chuẩn hóa."""
        p = self.past_scaler.transform(df[self.past_cols].fillna(0).values)
        f = self.future_scaler.transform(df[self.future_cols].fillna(0).values)
        t = self.target_scaler.transform(df[[self.target_col]].fillna(0).values).flatten()
        L, H = self.seq_len, self.horizon
        n = len(df)
        n_samples = n - L - H + 1
        past_w, future_w, target_w = [], [], []
        for s in range(n_samples):
            past_w.append(p[s:s+L])
            future_w.append(f[s:s+L+H])
            target_w.append(t[s+L:s+L+H])
        return np.array(past_w), np.array(future_w), np.array(target_w)

    def _static_indices(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.static_cols:
            mp = self.cat_maps[c]
            cols.append(df[c].map(mp).fillna(0).astype(int).values)
        return np.stack(cols, axis=1)  # [n, S]

    # ---------- fit ----------
    def fit(self, train_dfs: List[pd.DataFrame], val_dfs: Optional[List[pd.DataFrame]] = None) -> "TFTGlobalForecaster":
        print(f"[+] Huấn luyện TFT GLOBAL ({len(train_dfs)} chuỗi) trên thiết bị: {self.device}")

        # 1. Xây dựng ánh xạ categorical từ tập train
        all_train = pd.concat(train_dfs, ignore_index=True)
        for c in self.static_cols:
            uniques = sorted(all_train[c].astype(str).unique().tolist())
            # index 0 = <pad>/unknown, bắt đầu từ 1
            self.cat_maps[c] = {v: i + 1 for i, v in enumerate(uniques)}
            self.cardinalities[c] = len(uniques)

        # 2. Fit scaler trên toàn bộ train (chuẩn hóa chung giữa các chuỗi)
        self.past_scaler.fit(all_train[self.past_cols].fillna(0).values)
        self.future_scaler.fit(all_train[self.future_cols].fillna(0).values)
        self.target_scaler.fit(all_train[[self.target_col]].fillna(0).values)

        # 3. Tạo cửa sổ từ TẤT CẢ chuỗi train
        past_list, future_list, target_list, static_list = [], [], [], []
        for df in train_dfs:
            df = df.sort_values("d_num").reset_index(drop=True)
            pw, fw, tw = self._build_windows(df)
            past_list.append(pw); future_list.append(fw); target_list.append(tw)
            sw = np.tile(self._static_indices(df)[0:1], (len(pw), 1))
            static_list.append(sw)
        past_arr = np.concatenate(past_list, axis=0)
        future_arr = np.concatenate(future_list, axis=0)
        target_arr = np.concatenate(target_list, axis=0)
        static_arr = np.concatenate(static_list, axis=0)
        print(f"[-] Tổng số cửa sổ huấn luyện: {len(past_arr)}")

        train_dataset = TFTGlobalDataset(past_arr, future_arr, static_arr, target_arr)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # 4. Khởi tạo mô hình
        self.model = TemporalFusionTransformerGlobal(
            num_past_vars=len(self.past_cols),
            num_future_vars=len(self.future_cols),
            static_cardinalities=self.cardinalities,
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
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            for p_x, f_x, s_x, y in train_loader:
                p_x, f_x, y = p_x.to(self.device), f_x.to(self.device), y.to(self.device)
                s_x = {c: s_x[:, i].to(self.device) for i, c in enumerate(self.static_cols)}
                optimizer.zero_grad()
                q_preds, _ = self.model(p_x, f_x, s_x)
                loss = criterion(q_preds, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(p_x)
            train_loss /= len(train_dataset)

            val_loss = train_loss
            if val_dfs is not None:
                # Dùng đuôi tập train (28 forecast-origin cuối) làm validation
                # vì tập val chỉ dài 28 ngày, không đủ tạo cửa sổ 84 ngày.
                val_loss = self._eval_loss(train_dfs, criterion)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
            if epoch % 5 == 0 or epoch == self.epochs:
                print(f"Epoch [{epoch:02d}/{self.epochs:02d}] - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print("[✓] Huấn luyện TFT GLOBAL hoàn tất!")
        return self

    def _eval_loss(self, dfs, criterion):
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for df in dfs:
                df = df.sort_values("d_num").reset_index(drop=True)
                pw, fw, tw = self._build_windows(df)
                if len(pw) == 0:
                    continue
                # Chỉ lấy 28 cửa sổ cuối (các "forecast origin" nằm trong 28 ngày
                # cuối của tập train) để làm validation không rò rỉ.
                k = min(28, len(pw))
                pw, fw, tw = pw[-k:], fw[-k:], tw[-k:]
                sw = np.tile(self._static_indices(df)[0:1], (len(pw), 1))
                s_x = {c: torch.tensor(sw[:, i], dtype=torch.long).to(self.device)
                       for i, c in enumerate(self.static_cols)}
                p_t = torch.tensor(pw, dtype=torch.float32).to(self.device)
                f_t = torch.tensor(fw, dtype=torch.float32).to(self.device)
                t_t = torch.tensor(tw, dtype=torch.float32).to(self.device)
                q, _ = self.model(p_t, f_t, s_x)
                loss = criterion(q, t_t)
                total += loss.item() * len(pw); count += len(pw)
        return total / max(count, 1)

    # ---------- predict ----------
    def predict_quantiles(self, last_seq_df: pd.DataFrame, future_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Mô hình chưa fit!")
        self.model.eval()
        df_full = pd.concat([last_seq_df, future_df], ignore_index=True).sort_values("d_num").reset_index(drop=True)
        p_scaled = self.past_scaler.transform(df_full[self.past_cols].fillna(0).values)
        f_scaled = self.future_scaler.transform(df_full[self.future_cols].fillna(0).values)

        p_t = torch.tensor(p_scaled[:self.seq_len], dtype=torch.float32).unsqueeze(0).to(self.device)
        f_t = torch.tensor(f_scaled[:self.seq_len + self.horizon], dtype=torch.float32).unsqueeze(0).to(self.device)
        # static của chuỗi này
        tmp = last_seq_df.head(1)
        sw = self._static_indices(tmp)  # [1, S]
        s_x = {c: torch.tensor(sw[:, i], dtype=torch.long).to(self.device)
               for i, c in enumerate(self.static_cols)}

        with torch.no_grad():
            q_out, _ = self.model(p_t, f_t, s_x)
            q_out = q_out.squeeze(0).cpu().numpy()

        q_preds = np.zeros_like(q_out)
        for i in range(len(self.quantiles)):
            q_preds[:, i] = self.target_scaler.inverse_transform(q_out[:, i:i+1]).flatten()
        return np.clip(q_preds, 0, None)
