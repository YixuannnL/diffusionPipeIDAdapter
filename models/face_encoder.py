# models/face_encoder.py  （整文件替换，保持与旧接口一致）
import torch, torch.nn as nn, torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

class FaceEncoder(nn.Module):
    """
    img : (B,3,H,W)，RGB，数值范围 [-1, 1]
    out : (B,512)，L2-normed，梯度可回传到 img
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self._current_device = torch.device(device)

        # Facenet-pytorch 会自动下载 VGGFace2 预训练权重
        self.backbone = (
            InceptionResnetV1(pretrained='vggface2')
            .to(self._current_device)
            .eval()
        ).half()
        for p in self.backbone.parameters():
            p.requires_grad_(False)      # 冻结，但保持可微
            
    def _ensure_device(self, target: torch.device):
        if target != self._current_device:
            self.backbone.to(target, non_blocking=True)
            self._current_device = target

    @torch.autocast('cuda')
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        self._ensure_device(img.device)
        # [-1,1] → [0,1]
        x = (img + 1) * 0.5
        # Facenet 期望 160×160
        
        x = F.interpolate(x.squeeze(2), size=(160, 160), mode='bilinear', align_corners=False) # squeeze(2) 去掉 time 维度
        x = (x - 0.5) / 0.5
        emb = self.backbone(x)           # (B,512)
        return F.normalize(emb, p=2, dim=-1)
