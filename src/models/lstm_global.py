"""
Module: lstm_global.py
Mục đích: Mở rộng LSTM thành mô hình GLOBAL (huấn luyện đồng thời nhiều chuỗi SKU)
           với STATIC CONTEXT (embedding cửa hàng/sản phẩm) — tương đương TFT Global.
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict


class StaticContextLSTM(nn.Module):
    def __init__(self, cardinalities: Dict[str, int], hidden_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(card + 1, hidden_dim, padding_idx=0)
            for name, card in cardinalities.items()
        })

    def forward(self, static_idx: Dict[str, torch.Tensor]) -> torch.Tensor:
        embs = [self.embeddings[name](static_idx[name]) for name in self.embeddings]
        return sum(embs)


class GlobalLSTMNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        horizon: int = 28,
        dropout: float = 0.2,
        static_cardinalities: Optional[Dict[str, int]] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        self.static_ctx = None
        if static_cardinalities:
            self.static_ctx = StaticContextLSTM(static_cardinalities, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        fc_input = hidden_dim * 2 if static_cardinalities else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon)
        )

    def forward(self, x: torch.Tensor, static_idx: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        lstm_out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]

        if self.static_ctx is not None and static_idx is not None:
            static_emb = self.static_ctx(static_idx)
            combined = torch.cat([last_hidden, static_emb], dim=-1)
        else:
            combined = last_hidden

        return self.fc(combined)


class LSTMGlobalDataset(Dataset):
    def __init__(self, features: np.ndarray, static: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.static = torch.tensor(static, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.static[idx], self.targets[idx]


class LSTMGlobalForecaster:
    def __init__(
        self,
        past_feature_cols: List[str],
        static_cat_cols: List[str],
        target_col: str = "sales",
        seq_len: int = 56,
        horizon: int = 28,
        hidden_dim: int = 64,
        num_layers: int = 2,
        lr: float = 2e-3,
        batch_size: int = 128,
        epochs: int = 20,
        device: Optional[str] = None
    ):
        self.past_cols = past_feature_cols
        self.static_cols = static_cat_cols
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
        self.cat_maps: Dict[str, Dict[str, int]] = {}
        self.cardinalities: Dict[str, int] = {}
        self.model: Optional[GlobalLSTMNet] = None

    def _build_windows(self, df: pd.DataFrame):
        f = self.scaler.transform(df[self.past_cols].fillna(0).values)
        t = self.target_scaler.transform(df[[self.target_col]].fillna(0).values).flatten()
        L, H = self.seq_len, self.horizon
        n = len(df)
        n_samples = n - L - H + 1
        windows, targets = [], []
        for s in range(n_samples):
            windows.append(f[s:s+L])
            targets.append(t[s+L:s+L+H])
        return np.array(windows), np.array(targets)

    def _static_indices(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.static_cols:
            mp = self.cat_maps[c]
            cols.append(df[c].map(mp).fillna(0).astype(int).values)
        return np.stack(cols, axis=1)

    def fit(self, train_dfs: List[pd.DataFrame], val_dfs: Optional[List[pd.DataFrame]] = None) -> "LSTMGlobalForecaster":
        print(f"[+] Huấn luyện LSTM GLOBAL ({len(train_dfs)} chuỗi) trên thiết bị: {self.device}")

        all_train = pd.concat(train_dfs, ignore_index=True)
        for c in self.static_cols:
            uniques = sorted(all_train[c].astype(str).unique().tolist())
            self.cat_maps[c] = {v: i + 1 for i, v in enumerate(uniques)}
            self.cardinalities[c] = len(uniques)

        self.scaler.fit(all_train[self.past_cols].fillna(0).values)
        self.target_scaler.fit(all_train[[self.target_col]].fillna(0).values)

        window_list, target_list, static_list = [], [], []
        for df in train_dfs:
            df = df.sort_values("d_num").reset_index(drop=True)
            w, t = self._build_windows(df)
            window_list.append(w)
            target_list.append(t)
            sw = np.tile(self._static_indices(df)[0:1], (len(w), 1))
            static_list.append(sw)

        windows_arr = np.concatenate(window_list, axis=0)
        targets_arr = np.concatenate(target_list, axis=0)
        static_arr = np.concatenate(static_list, axis=0)
        print(f"[-] Tổng số cửa sổ huấn luyện: {len(windows_arr)}")

        train_dataset = LSTMGlobalDataset(windows_arr, static_arr, targets_arr)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model = GlobalLSTMNet(
            input_dim=len(self.past_cols),
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            horizon=self.horizon,
            static_cardinalities=self.cardinalities if self.static_cols else None
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

        best_loss = float("inf")
        best_state = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            for x, s, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                s = {c: s[:, i].to(self.device) for i, c in enumerate(self.static_cols)}

                optimizer.zero_grad()
                preds = self.model(x, s)
                loss = criterion(preds, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(x)

            train_loss /= len(train_dataset)
            scheduler.step(train_loss)

            if train_loss < best_loss:
                best_loss = train_loss
                best_state = copy.deepcopy(self.model.state_dict())

            if epoch % 5 == 0 or epoch == self.epochs:
                print(f"Epoch [{epoch:02d}/{self.epochs}] - Train Loss: {train_loss:.5f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print("[✓] Huấn luyện LSTM GLOBAL hoàn tất!")
        return self

    def predict(self, last_seq_df: pd.DataFrame, static_row: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Mô hình chưa fit!")

        self.model.eval()
        f_scaled = self.scaler.transform(last_seq_df[self.past_cols].fillna(0).values)
        x_t = torch.tensor(f_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        sw = self._static_indices(static_row)
        s_x = {c: torch.tensor(sw[:, i], dtype=torch.long).to(self.device) for i, c in enumerate(self.static_cols)}

        with torch.no_grad():
            pred_scaled = self.model(x_t, s_x).cpu().numpy().flatten()

        pred = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        return np.clip(pred, 0, None)
