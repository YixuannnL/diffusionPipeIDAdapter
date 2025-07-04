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
        return inputs
