# models/face_encoder.py  （整文件替换，保持与旧接口一致）
import torch, torch.nn as nn, torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN
from torchvision.transforms.functional import to_pil_image

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
        self.mtcnn = MTCNN(image_size=160, margin=0,device='cpu')
        for p in self.backbone.parameters():
            p.requires_grad_(False)      # 冻结，但保持可微
            
    def _ensure_device(self, target: torch.device):
        if target != self._current_device:
            # ① backbone 跟着主设备走
            self.backbone.to(target, non_blocking=True)

            # ② 把 MTCNN 也搬过去，并同步它的 device 字段
            #    （facenet‑pytorch 里 forward 时会用到 self.device）
            self.mtcnn.to(target, non_blocking=True)
            self.mtcnn.device = torch.device(target).type

            self._current_device = target

    @torch.autocast('cuda', dtype=torch.float16)
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        self._ensure_device(img.device)
        model_device = next(self.backbone.parameters()).device
        model_dtype = next(self.backbone.parameters()).dtype
       
        mean, std = 0.5, 0.5
        first_unnorm = img * std + mean
        if first_unnorm.ndim == 5:
            first_unnorm = first_unnorm[:, :, 0, ...].squeeze()
        img_recovered = to_pil_image(first_unnorm)
        
        with torch.cuda.amp.autocast(enabled=False):
            img_cropped = self.mtcnn(img_recovered, 
                                # save_path="mtcnn.png"
                                )
            
        if img_cropped is None:
            import torchvision.transforms.functional as TF
            img_cropped = TF.to_tensor(img_recovered)
            img_cropped = TF.center_crop(img_cropped, min(img_cropped.shape[-2:]))
            img_cropped = TF.resize(img_cropped, (160, 160), antialias=True)
            img_cropped = (img_cropped * 2.0 - 1.0).to(dtype=model_dtype)
            
        img_cropped = img_cropped.to(dtype=model_dtype, non_blocking=True)
        img_cropped = img_cropped.to(device=model_device,
                                 dtype=model_dtype,
                                 non_blocking=True)
        
        emb = self.backbone(img_cropped.unsqueeze(0))           # (B,512)
        return F.normalize(emb, p=2, dim=-1)
