import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """
    Sinusoidal PE，遇到更长序列时自动扩展，不再固定 262 144。
    内部缓存总在 CPU / fp32，前向时按需 to(device, dtype)。
    """
    def __init__(self, dim, init_len: int = 1024):
        super().__init__()
        self.dim = dim
        self.register_buffer("pe", self._build(init_len), persistent=False)

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

from einops import rearrange
from timm.layers import Mlp, DropPath

class TemporalSelfAttention(nn.Module):
    """(B, L, D) → (B, L, D)，仅在时间维做全局 self-attn，空间内用分组"""
    def __init__(self, dim, num_heads, num_id_tokens: int, dropout: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
    # <<< 让 block 在构造时写入 num_id_tokens >>>
        self.num_id_tokens = num_id_tokens

    def forward(self, x, T):
        """
        Args:
            x: (B, L_total, D)  (含 ID‑token)
            T: 帧数
        """
        n_id = self.num_id_tokens     # 0 or 16

        # id_tok, seq_tok = (x[:, :n_id, :], x[:, n_id:, :]) if n_id else (None, x)
        if n_id:
            id_tok, seq_tok = x[:, :n_id, :], x[:, n_id:, :]
        else:
            id_tok, seq_tok = None, x

        B, L_seq, D = seq_tok.shape
        N = L_seq // T               # token per frame
        seq_tok = seq_tok.view(B * N, T, D)       # (B·N, T, D)
        qkv = self.qkv(seq_tok).reshape(B * N, T, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, B*N, H, T, d)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        seq_out = (attn @ v).transpose(1, 2).reshape(B * N, T, D)
        seq_out = self.proj(seq_out)
        seq_out = self.proj_drop(seq_out)
        seq_out = seq_out.view(B, L_seq, D)
        return torch.cat([id_tok, seq_out], dim=1) if n_id else seq_out

class VideoIDAdapter(nn.Module):
    """
    时序 Transformer，把 Wan-VAE latent 序列 → N 个 ID-tokens
    """
    # -----------------------------------------------------------
    # VideoIDAdapter v2
    # -----------------------------------------------------------
    # ① 支持多尺度 / ROI 网格；② 引入“Temporal-Self-Attention (TSA)”
    # ③ 输出端用 token-wise gating α_i，自动权衡各 ID-token 贡献
    # -----------------------------------------------------------

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
        grid_size: Tuple[int, int] = (8, 8),     # 空间采样网格 (g_h, g_w)
        pool_mode: str = "grid",                 # "grid" | "mean" | "center"
    ):
        super().__init__()
        self.num_id_tokens = num_id_tokens
        # ----- 控制采样策略 -----
        self.grid_h, self.grid_w = grid_size      # 默认 8 × 8 网格
        self.pool_mode = pool_mode.lower()                 # "grid" | ["grid","mean",...]
        self.multi_scales: Tuple[int]=(8,4)          # extra grid sizes, e.g. (8,4)
        
        # -----  
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

        # ---------------- Transformer Encoder ----------------
        blocks = []
        for _ in range(num_layers):
            blocks.append(nn.ModuleDict({
                "tsa":       TemporalSelfAttention(adapter_dim, num_heads,
                                                   num_id_tokens, dropout),
                "mlp":       Mlp(adapter_dim, int(adapter_dim * mlp_ratio),
                                 drop=dropout),
                "norm1":     nn.LayerNorm(adapter_dim),
                "norm2":     nn.LayerNorm(adapter_dim),
                "drop_path": DropPath(dropout) if dropout > 0 else nn.Identity(),
            }))
        self.transformer = nn.ModuleList(blocks)
        
        # ========================= NEW =========================
        # 1. 每‑token 独立线性映射  W_i, b_i
        self.token_proj_weight = nn.Parameter(
            torch.empty(num_id_tokens, hidden_size, adapter_dim)
        )                               # (N, H, D)
        self.token_proj_bias = nn.Parameter(
            torch.zeros(num_id_tokens, hidden_size)
        )                               # (N, H)
        nn.init.kaiming_uniform_(self.token_proj_weight, a=math.sqrt(5))

        # 2. gating α_i（初始化为 0，相当于均匀）
        self.alpha = nn.Parameter(torch.zeros(num_id_tokens))
        # =======================================================

    def forward(self, 
                latents: torch.Tensor,
                ) -> torch.Tensor:
        """
        latents: [B, C, F, H, W] – Wan-VAE 输出 (未加噪)
        return id_tokens: [B, N, D]
        """
        B, C, F, H, W = latents.shape               # 16 / 17 / 160 / 90
        # ------------------------------------------------------------------
        # 先把 VAE latent 变成时序 token 序列  (B, seq_len, C)
        # ------------------------------------------------------------------
        seqs = []
        # 1. 主网格
        if self.pool_mode == "grid" or isinstance(self.pool_mode, list) and "grid" in self.pool_mode:
            g_h, g_w = self.grid_h, self.grid_w
            # 在高、宽两个方向上取均匀网格中心的索引
            h_idx = torch.linspace(0, H - 1, g_h, device=latents.device).long()
            w_idx = torch.linspace(0, W - 1, g_w, device=latents.device).long()
            lat_sel = (
                latents.index_select(-2, h_idx)          # 选高度
                .index_select(-1, w_idx)          # 选宽度
            )                                            # [B,C,F,g_h,g_w]
            seqs.append(
                rearrange(lat_sel, "b c f gh gw -> b (f gh gw) c")
            )
            
            # 2. 可选多尺度
            for g in self.multi_scales:
                if g == self.grid_h: continue
                h_idx = torch.linspace(0, H - 1, g, device=latents.device).long()
                w_idx = torch.linspace(0, W - 1, g, device=latents.device).long()
                sel = latents.index_select(-2, h_idx).index_select(-1, w_idx)
                seqs.append(rearrange(sel, "b c f gh gw -> b (f gh gw) c"))

        elif self.pool_mode == "mean" or isinstance(self.pool_mode, list) and "mean" in self.pool_mode:
            # 对 H、W 做全局平均池化
            x = latents.mean(dim=(-2, -1))               # [B,C,F]
            x = x.permute(0, 2, 1)                      # [B,F,C]

        elif self.pool_mode == "center" or isinstance(self.pool_mode, list) and "center" in self.pool_mode:
            # 直接取中心像素
            x = latents[..., H // 2, W // 2]             # [B,C,F]
            x = x.permute(0, 2, 1)                       # [B,F,C]

        # ------------------------------------------------------------------
        # 线性降维到 adapter_dim 并加绝对 PE
        # ------------------------------------------------------------------
        x = torch.cat(seqs, dim=1)                  # concat all sequences
        x = self.in_proj(x)
        x = self.pos_enc(x)                         # 为序列中每个位置添加可微分的绝对正弦余弦位置信息，使 Transformer 能区分 token 在时间/空间序列上的先后

        # prepend learnable ID tokens
        id_tok = self.id_tokens.expand(B, -1, -1)   # [B, N, D]
        x = torch.cat([id_tok, x], dim=1)           # [B, N+L, D] 把它们“ prepend” 到时空 token 序列前面，让 Transformer 可以用后续的自注意力把时空信息汇聚到这些 ID token 上

        T = F                                        # 注意时间长度
        for blk in self.transformer:
            x = x + blk['drop_path'](blk['tsa'](blk['norm1'](x), T))
            x = x + blk['drop_path'](blk['mlp'](blk['norm2'](x)))
        # ------------------------------------------------------------------
        # 取出前 N 个 ‑[ID]‑token → 投回 Wan hidden_size=5120
        # ------------------------------------------------------------------
        id_feat = x[:, :self.num_id_tokens, :]      # [B,N,D]
        
        # ---------- 1. token‑wise全连接：D_a → hidden_size ----------
        proj_out = torch.einsum("bnd,nhd->bnh",
                                id_feat,                 # (B,N,D_a)
                                self.token_proj_weight   # (N,H,D_a)
                               )                         # → (B,N,H)
        proj_out = proj_out + self.token_proj_bias.unsqueeze(0)   # broadcast bias

        # ---------- 2. αᵢ gating ----------
        w = torch.softmax(self.alpha, dim=0)                             # (N,)
        proj_out = proj_out * w.unsqueeze(0).unsqueeze(-1)               # (B,N,H)

        # ---------- 3. 把加权后的 N 个 token 直接返回 ----------
        return proj_out
