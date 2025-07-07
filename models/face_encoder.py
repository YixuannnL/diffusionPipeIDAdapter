import torch, torchvision
import cv2, numpy as np, insightface
from torch import nn

class FaceEncoder(nn.Module):
    """
    提供 .get(image_tensor[B,3,H,W]) -> emb[B,512]
    image_tensor 范围 [-1,1] (Wan 默认)
    """
    def __init__(self, det_thresh=0.6, device='cuda'):
        super().__init__()
        self.device = device
        self.det_thresh = det_thresh

        model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        model.prepare(ctx_id=0, det_size=(640, 640))
        self.detector = model  # 带检测、对齐、embedding

    @torch.no_grad()
    def forward(self, img_t: torch.Tensor):
        bs = img_t.shape[0]
        out = torch.zeros(bs, 512, device=self.device)
        for i in range(bs):
            img = ((img_t[i].permute(1,2,0).cpu().numpy()+1)/2*255).astype(np.uint8)
            faces = self.detector.get(img)
            if len(faces)==0:
                continue
            face = max(faces, key=lambda f: f.bbox[2]-f.bbox[0])
            out[i] = torch.from_numpy(face.embedding).to(self.device)
        out = torch.nn.functional.normalize(out, dim=-1)
        return out  # (B,512)
