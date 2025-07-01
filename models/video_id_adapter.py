import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

class PositionalEncoding(nn.Module):
    """
    Sinusoidal PE，遇到更长序列时自动扩展，不再固定 262 144。
    内部缓存总在 CPU / fp32，前向时按需 to(device, dtype)。
    """
    def __init__(self, dim, init_len: int = 262_144):
        super().__init__()
        self.dim = dim
        self.register_buffer("pe", self._build(1024), persistent=False)

    def _build(self, length: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)      # [L,1]
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32)
                             * (-math.log(10_000.0) / self.dim))               # [D/2]
        pe = torch.zeros(length, self.dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe        # remains on CPU / fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        """
        L = x.size(1)
        if L > self.pe.size(0):           # 需要更长的表时，扩容一倍直到够用
            new_len = 2 ** math.ceil(math.log2(L))
            self.pe = self._build(new_len)

        # cast + device 转移只对前 L 行
        pe = self.pe[:L].to(dtype=x.dtype, device=x.device)
        return x + pe


class VideoIDAdapter(nn.Module):
    """
    时序 Transformer，把 Wan-VAE latent 序列 → N 个 ID-tokens
    """

    def __init__(
        self,
        *,
        hidden_size: int,   # Wan cross-attn dim  (5120  for 14B)
        num_id_tokens: int = 16,
        num_layers: int = 4,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        proj_in_channels: Optional[int] = None,  # 若 None, 用 LazyLinear
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_id_tokens = num_id_tokens
        self.hidden_size = hidden_size

        # 4/8/16 → hidden_size 线性投影（Lazy 自动推断通道）
        if proj_in_channels is None:
            self.in_proj = nn.LazyLinear(hidden_size, bias=False)
        else:
            self.in_proj = nn.Linear(proj_in_channels, hidden_size, bias=False)

        # learnable [ID] tokens
        self.id_tokens = nn.Parameter(
            torch.randn(1, num_id_tokens, hidden_size) / math.sqrt(hidden_size)
        )

        # Position encoding（时空展平后一维绝对 PE）
        self.pos_enc = PositionalEncoding(hidden_size)

        # 标准 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: [B, 4, F, H, W] – Wan-VAE 输出 (未加噪)
        return id_tokens: [B, N, D]
        """
        B, C, F, H, W = latents.shape               # 4 / 17 / 160 / 90
        # 时空展平 -> L = F*H*W
        # x = latents.permute(0, 2, 3, 4, 1).reshape(B, F * H * W, C)
        # ① 先对空间 (H,W) 做平均池化，得到帧级特征
        # x = latents.mean(dim=[3, 4]).permute(0, 2, 1)                # [B, F, C] 17 token
        g_h, g_w = 4, 4
        h_idx = torch.linspace(0, latents.size(3)-1, g_h).long()
        w_idx = torch.linspace(0, latents.size(4)-1, g_w).long()
        lat_sel = latents[:, :, :, h_idx][:, :, :, :, w_idx]   # [B,C,F,4,4]
        x = lat_sel.permute(0, 2, 3, 4, 1).reshape(B, F*g_h*g_w, C)  # [B, L, C]    
        L = F  # 序列长度
        x = self.in_proj(x)                         # [B, L, D]
        x = self.pos_enc(x)

        # prepend learnable ID tokens
        id_tok = self.id_tokens.expand(B, -1, -1)   # [B, N, D]
        x = torch.cat([id_tok, x], dim=1)           # [B, N+L, D]

        x = self.transformer(x)                     # same shape

        # 取前 N 个
        return x[:, :self.num_id_tokens, :]
