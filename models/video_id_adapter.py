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
        adapter_dim: int = 1024,     # 内部瓶颈宽度
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_id_tokens = num_id_tokens
        self.hidden_size = hidden_size
        self.adapter_dim = adapter_dim

        # 4/16 → adapter_dim
        if proj_in_channels is None:
            self.in_proj = nn.LazyLinear(adapter_dim, bias=False)
        else:
            self.in_proj = nn.Linear(proj_in_channels, adapter_dim, bias=False)

        # learnable [ID] tokens
        self.id_tokens = nn.Parameter(
            torch.randn(1, num_id_tokens, adapter_dim) / math.sqrt(adapter_dim)
        )

        # Position encoding（时空展平后一维绝对 PE）
        self.pos_enc = PositionalEncoding(adapter_dim)

        # 标准 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=adapter_dim,
            nhead=num_heads,
            dim_feedforward=adapter_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.out_proj = nn.Linear(adapter_dim, hidden_size, bias=True)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: [B, C, F, H, W] – Wan-VAE 输出 (未加噪)
        return id_tokens: [B, N, D]
        """
        B, C, F, H, W = latents.shape               # 16 / 17 / 160 / 90
        g_h, g_w = 4, 4
        h_idx = torch.linspace(0, latents.size(3)-1, g_h).long()
        w_idx = torch.linspace(0, latents.size(4)-1, g_w).long()
        lat_sel = latents[:, :, :, h_idx][:, :, :, :, w_idx]   # [B,C,F,4,4]
        x = lat_sel.permute(0, 2, 3, 4, 1).reshape(B, F*g_h*g_w, C)  # 把 [B,C,F,4,4] → [B, F, 4, 4, C] → [B, L, C]，形成每个采样点一个 token 的时空序列  
        L = F  # 序列长度
        x = self.in_proj(x)                         # [B, L, D] in_proj: C → adapter_dim
        x = self.pos_enc(x)                         # 为序列中每个位置添加可微分的绝对正弦余弦位置信息，使 Transformer 能区分 token 在时间/空间序列上的先后

        # prepend learnable ID tokens
        id_tok = self.id_tokens.expand(B, -1, -1)   # [B, N, D]
        x = torch.cat([id_tok, x], dim=1)           # [B, N+L, D] 把它们“ prepend” 到时空 token 序列前面，让 Transformer 可以用后续的自注意力把时空信息汇聚到这些 ID token 上

        x = self.transformer(x)                     # same shape

        out = x[:, :self.num_id_tokens, :]          # [B,N, adapter_dim] 取出经 Transformer 更新后的前 N 个 “身份” token
        # 映射回 Wan hidden_size=5120 供 cross-attn 用一个全连接把 adapter_dim 投回与 Wan cross-attn 相同的维度（5120），以便后续注入到主模型的 cross-attn 中
        return self.out_proj(out)

    # output projection weights（trainable）
    # out_proj_weight: torch.Tensor
    # out_proj_bias:   torch.Tensor
    def _init_output_proj(self, hidden_size, adapter_dim):
        self.out_proj_weight = nn.Parameter(torch.empty(hidden_size, adapter_dim))
        self.out_proj_bias   = nn.Parameter(torch.zeros(hidden_size))               
        nn.init.kaiming_uniform_(self.out_proj_weight, a=math.sqrt(5))

    def __post_init__(self):
        self._init_output_proj(self.hidden_size, self.adapter_dim)
