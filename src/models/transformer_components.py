"""
Module: transformer_components.py
Mục đích: Cung cấp các khối kiến trúc nền tảng của mô hình Temporal Fusion Transformer (TFT):
          1. Gated Linear Unit (GLU)
          2. Gated Residual Network (GRN)
          3. Variable Selection Network (VSN)
          4. Interpretable Multi-Head Attention
          5. Quantile Loss (Pinball Loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


class GLU(nn.Module):
    """
    Gated Linear Unit (GLU):
    GLU(gamma) = Linear(gamma) * Sigmoid(Linear(gamma))
    Cung cấp cơ chế đóng/mở cổng thích ứng để kiểm soát lượng thông tin được truyền qua.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(GLU, self).__init__()
        self.fc_val = nn.Linear(input_dim, output_dim)
        self.fc_gate = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val = self.fc_val(x)
        gate = torch.sigmoid(self.fc_gate(x))
        return val * gate


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN):
    Khối cốt lõi trong TFT cho phép mô hình tự động điều chỉnh độ phi tuyến tính
    hoặc bỏ qua biến đổi (Skip connection) nếu đặc trưng không cần thiết.
    
    Công thức:
    eta_2 = ELU(W_2 * a + W_3 * c + b_2)
    eta_1 = W_1 * eta_2 + b_1
    GRN(a, c) = LayerNorm(a + GLU(eta_1))
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1, context_dim: Optional[int] = None):
        super(GatedResidualNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim is not None:
            self.fc_context = nn.Linear(context_dim, hidden_dim, bias=False)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Skip connection projection nếu số chiều input != output
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [..., input_dim]
        # context: [..., context_dim]
        hidden = self.fc1(x)
        if context is not None and self.context_dim is not None:
            # Điều kiện hóa bởi vector ngữ cảnh tĩnh (Static Context).
            # Nếu hidden có chiều thời gian [B, T, H], mở rộng context thành [B, 1, C]
            # để phát sóng (broadcast) đúng qua trục thời gian.
            ctx = context.unsqueeze(1) if hidden.dim() == 3 else context
            hidden = hidden + self.fc_context(ctx)
            
        hidden = F.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        gated = self.glu(hidden)

        skip = self.skip_proj(x)
        return self.layer_norm(skip + gated)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN):
    Tự động gán trọng số tầm quan trọng (Attention-based feature selection) cho từng biến đầu vào
    tại mỗi bước thời gian.
    
    Trả về:
    1. Vector đặc trưng tổng hợp sau khi chọn lọc (Transformed features)
    2. Trọng số đóng góp của từng biến (Variable Importance Weights) để phục vụ diễn giải mô hình.
    """
    def __init__(self, num_vars: int, input_dim: int, hidden_dim: int, dropout: float = 0.1, context_dim: Optional[int] = None):
        super(VariableSelectionNetwork, self).__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        # Mạng GRN riêng cho từng biến đầu vào
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_vars)
        ])

        # Mạng GRN tổng để sinh trọng số chọn biến (Softmax weights)
        self.weight_grn = GatedResidualNetwork(
            input_dim=num_vars * input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_vars,
            dropout=dropout,
            context_dim=context_dim
        )

    def forward(self, flattened_vars: torch.Tensor, individual_vars: List[torch.Tensor], context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            flattened_vars: [Batch, (Time), num_vars * input_dim] - Các biến nối lại
            individual_vars: Danh sách các tensor [Batch, (Time), input_dim] cho từng biến
            context: [Batch, (Time), context_dim] - Ngữ cảnh tĩnh (tùy chọn)

        Returns:
            vsn_output: [Batch, (Time), hidden_dim] - Biểu diễn tổng hợp
            weights: [Batch, (Time), num_vars] - Trọng số quan trọng của từng biến
        """
        # 1. Tính trọng số chọn biến qua GRN + Softmax
        weights_logits = self.weight_grn(flattened_vars, context)
        weights = F.softmax(weights_logits, dim=-1) # [Batch, (Time), num_vars]

        # 2. Xử lý từng biến qua GRN tương ứng
        transformed_vars = [grn(var) for grn, var in zip(self.var_grns, individual_vars)]
        # Stack lại: [Batch, (Time), num_vars, hidden_dim]
        stacked = torch.stack(transformed_vars, dim=-2)

        # 3. Nhân trọng số và tính tổng có trọng số
        # weights: [Batch, (Time), num_vars, 1]
        expanded_weights = weights.unsqueeze(-1)
        vsn_output = torch.sum(stacked * expanded_weights, dim=-2) # [Batch, (Time), hidden_dim]

        return vsn_output, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention:
    Khác với Transformer thông thường (mỗi head có một phép chiếu Value riêng biệt),
    TFT sử dụng chung một phép chiếu Value matrix W_V trên toàn bộ các Head.
    
    Điều này cho phép gộp các ma trận Attention qua các Head thành một ma trận chú ý duy nhất:
    A_tilde = Mean(A_h)
    giúp trực quan hóa và diễn giải chính xác tầm quan trọng của các mốc thời gian quá khứ.
    """
    def __init__(self, num_heads: int, d_model: int, dropout: float = 0.1):
        super(InterpretableMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        # Value projection duy nhất dùng chung
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: [Batch, Seq_len_q, d_model]
            k: [Batch, Seq_len_k, d_model]
            v: [Batch, Seq_len_v, d_model]
            mask: Causal mask [Seq_len_q, Seq_len_k]

        Returns:
            output: [Batch, Seq_len_q, d_model]
            attention_weights: [Batch, Seq_len_q, Seq_len_k] - Ma trận chú ý trung bình để diễn giải
        """
        batch_size = q.size(0)

        # 1. Chiếu Query và Key sang multi-head
        # [Batch, Heads, Seq_len, d_k]
        q_heads = self.q_proj(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k_heads = self.k_proj(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Chiếu Value chung (Shared Value)
        # [Batch, Seq_len, d_model] -> [Batch, 1, Seq_len, d_model]
        v_shared = self.v_proj(v).unsqueeze(1)

        # 3. Scaled Dot-Product Attention cho từng head
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights_heads = F.softmax(scores, dim=-1) # [Batch, Heads, Seq_q, Seq_k]
        attn_weights_heads = self.dropout(attn_weights_heads)

        # 4. Tính ma trận Attention trung bình trên các Head (Interpretable Attention Map)
        mean_attn_weights = attn_weights_heads.mean(dim=1) # [Batch, Seq_q, Seq_k]

        # 5. Tính đầu ra bằng ma trận Attention trung bình nhân với Value chung
        # [Batch, Seq_q, Seq_k] x [Batch, Seq_k, d_model] -> [Batch, Seq_q, d_model]
        context = torch.matmul(mean_attn_weights, v_shared.squeeze(1))
        output = self.out_proj(context)

        return output, mean_attn_weights


class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss) cho bài toán dự báo bất định (Probabilistic Forecasting):
    L_q(y, y_hat) = max( q * (y - y_hat), (1 - q) * (y_hat - y) )
    
    Hỗ trợ dự báo đồng thời nhiều phân vị (mặc định: P10, P50 (Median), P90).
    """
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: [Batch, Horizon, Num_quantiles] - Giá trị dự báo các phân vị
            y_true: [Batch, Horizon] - Giá trị thực tế

        Returns:
            loss: Giá trị loss trung bình trên toàn bộ các phân vị
        """
        losses = []
        y_true = y_true.unsqueeze(-1) # [Batch, Horizon, 1]

        for i, q in enumerate(self.quantiles):
            errors = y_true - y_pred[..., i:i+1]
            q_loss = torch.max((q - 1) * errors, q * errors)
            losses.append(q_loss)

        total_loss = torch.mean(torch.cat(losses, dim=-1))
        return total_loss
