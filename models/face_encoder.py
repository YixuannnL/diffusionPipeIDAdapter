# models/face_encoder.py  （整文件替换，保持与旧接口一致）
import torch, torch.nn as nn, torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

class FaceEncoder(nn.Module):
    """
    img : (B,3,H,W)，RGB，数值范围 [-1, 1]
    out : (B,512)，L2-normed，梯度可回传到 img
    """

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # Facenet-pytorch 会自动下载 VGGFace2 预训练权重
        self.backbone = (
            InceptionResnetV1(pretrained='vggface2')
            .to(device)
            .eval()
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)      # 冻结，但保持可微

    @torch.autocast('cuda')
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # [-1,1] → [0,1]
        x = (img + 1) * 0.5
        # Facenet 期望 160×160
        x = F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
        # Normalize：mean=0.5, std=0.5  → [-1,1]
        x = (x - 0.5) / 0.5
        emb = self.backbone(x)           # (B,512)
        return F.normalize(emb, p=2, dim=-1)
