import torch
from torch import nn
from utils.common import AUTOCAST_DTYPE

class AdapterLayer(nn.Module):
    """
    单独的一层，包裹 VideoIDAdapter。前向返回 (id_tokens, id_lens)
    给后续层使用。
    """
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter             # VideoIDAdapter 模块

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        # """
        # inputs: (latents, *others)
        # 只取 latents -> adapter -> id_tokens，其他原样传递
        # """
        # x, *rest = inputs                  # x == latents
        # id_tokens = self.adapter(x)        # [B,N,D]
        # id_lens   = id_tokens.new_full((x.size(0),), id_tokens.size(1), dtype=torch.long)
        # return (x, *rest, id_tokens, id_lens)   # append 两个新字段
        return inputs
