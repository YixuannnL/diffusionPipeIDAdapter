import argparse, os, json, math, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from pathlib import Path
import torch, safetensors.torch as safe, toml
from tqdm import tqdm
import imageio.v3 as iio

from utils.common import DTYPE_MAP, AUTOCAST_DTYPE as _AC
import utils.common as common         # 提前 import 公共模块
from models.wan import WanPipeline, vae_encode


# --------------------- CLI ------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', default="/data/bohong/diffusion-pipe/examples/wan_14b_min_vram.toml", help='Same TOML used for training')
parser.add_argument('--adapter', default="/data/bohong/diffusion-pipe/data/output/20250703_09-25-41/epoch1/video_id_adapter.safetensors", help='video_id_adapter.safetensors')
parser.add_argument('--ref_video', default="/data/bohong/diffusion-pipe/data/input/VIDEOS/video1.mp4", help='User video path')
parser.add_argument('--prompt', default="The video captures a woman walking along a city street, filmed in black and white on a classic 35mm camera. Her expression is thoughtful, her brow slightly furrowed as if she's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around her, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the woman's path, adding depth to the scene. The lighting highlights the woman's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel.",help='Text prompt')
parser.add_argument('--out_dir', default="/data/bohong/diffusion-pipe/data/inference_output", help='Output mp4 file')
parser.add_argument('--num_steps', type=int, default=1, help='DDIM / Euler steps')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cfg_scale', type=float, default=3.0,
                    help='classifier-free guidance scale (>=1.0)') # --cfg_scale=1.0 ⇒ 不使用 CFG；
                                                                    # 推荐 3-7 之间调节，过高会导致 ID 信息淡化或噪声放大
# block-swap：0 表示关闭；>0 表示需要搬运的 block 数
parser.add_argument('--blocks_to_swap', type=int, default=38,
                    help='Number of transformer blocks kept on CPU and swapped \
+back and forth during inference. Use 0 for fastest speed but highest VRAM.')
parser.add_argument('--cpu_only', action='store_true',
                     help='Run the entire transformer on CPU (slow but <10 GB RAM).')
args = parser.parse_args()

# ---------- Config ----------
with open(args.config) as f:
    cfg = toml.load(f)
cfg.setdefault("reentrant_activation_checkpointing", False)
model_cfg = cfg["model"]
model_cfg["dtype"] = DTYPE_MAP[model_cfg["dtype"]]
if isinstance(model_cfg.get("transformer_dtype", None), str):
    model_cfg["transformer_dtype"] = DTYPE_MAP[model_cfg["transformer_dtype"]]

# fp16 autocast  若想 bfloat16 改这里
common.AUTOCAST_DTYPE = torch.float16

# ---------- Reproducibility ----------
torch.manual_seed(args.seed)

# ---------- Load pipeline ----------
print('[*] loading Wan-2.1 …')
pipe = WanPipeline(cfg)                 # instantiate empty
pipe.load_diffusion_model()           # load weights
pipe.eval()         
dtype = model_cfg["dtype"]              # torch.bfloat16 mostly

if args.cpu_only:
    print('[*] running transformer on **CPU**')
    pipe.transformer.to('cpu')
    if pipe.video_id_adapter is not None:
        pipe.video_id_adapter.to('cpu')
else:
    pipe.transformer.to('cuda')
    print(f'[*] enabling block-swap - {args.blocks_to_swap} blocks on CPU')
    pipe.enable_block_swap(args.blocks_to_swap)
    pipe.prepare_block_swap_inference()           # forward-only模式

# ---------- Adapter ----------
print('[*] loading adapter weights …')
pipe.load_video_id_adapter_weights(args.adapter)

# ---------- Pre-encode reference video ----------
print('[*] encoding reference video …')
preprocess_fn = pipe.get_preprocess_media_file_fn()
H, W = 720, 1280
size_bucket = (W, H, -1)
tensor, _ = preprocess_fn(args.ref_video, None, size_bucket)[0]   # [C,F,H,W]
tensor = tensor.to(dtype=torch.float32) .unsqueeze(0)             # [1,C,F,H,W]

vae = pipe.vae 
vae.model.to('cuda', dtype=torch.float32)
vae.mean = vae.mean.to('cuda')
vae.std  = vae.std.to('cuda')
with torch.no_grad():
    latents = vae_encode(tensor.to('cuda'), vae)  # fp32 latents
    latents = latents.to(dtype) 
vae.model.to('cpu'); torch.cuda.empty_cache()

# ---------- Text encoding ----------
print('[*] encoding prompt …')
t5 = pipe.text_encoder
t5.model.to('cuda', dtype=dtype) 
txt_list  = t5([args.prompt], device='cuda') 
un_list  = t5([""], device='cuda')

def pad_list(lst):
    L = max(e.shape[0] for e in lst)
    out = torch.zeros((1, L, lst[0].shape[1]), dtype=lst[0].dtype, device='cuda')
    out[0, :lst[0].shape[0]] = lst[0]
    return out, torch.tensor([lst[0].shape[0]], device='cuda')

text_embs, seq_lens     = pad_list(txt_list)
un_text_embs, un_seq_lens = pad_list(un_list)

t5.model.to('cpu'); torch.cuda.empty_cache()

# ---------- ID tokens ----------
with torch.no_grad(), torch.autocast('cuda', _AC):
    id_tokens = pipe.video_id_adapter(latents)        # [1,N,5120]
id_lens   = torch.tensor([id_tokens.size(1)], device=id_tokens.device)

empty_tensor = torch.tensor([], device='cuda', dtype=_AC)

def build_feat(x, t_scalar, emb, lens):
    b = x.size(0)
    t = torch.full((b,), t_scalar, device=x.device, dtype=torch.float32)
    return (x, empty_tensor, t, emb, lens, empty_tensor, id_tokens, id_lens)

# ---------- Sampler ----------
def ddim_step(x, t, eps):
    alpha     = 1 - t
    sigma_inv = torch.rsqrt(1 - alpha**2)
    return (x - alpha.sqrt()*eps) * sigma_inv

x = torch.randn_like(latents)                              # start from noise
layers = pipe.to_layers()
DEVICE_RUN = 'cpu' if args.cpu_only else 'cuda'
ts = torch.linspace(1, 0, args.num_steps, device=DEVICE_RUN)
print('[*] sampling …')
for t in tqdm(ts):
    with torch.autocast(DEVICE_RUN, _AC), torch.inference_mode():
        eps_c, eps_u = None, None
        for cond, emb, lens in [(True,  text_embs,     seq_lens),
                                (False, un_text_embs, un_seq_lens)]:
            cur = build_feat(x, t*1000, emb, lens)
            for lyr in layers:
                cur = lyr(cur)
            if cond: eps_c = cur
            else:    eps_u = cur
        eps = eps_u + args.cfg_scale*(eps_c - eps_u)
    x = ddim_step(x, t, eps)

# ---------- Decode ----------
print('[*] decoding & saving …')
vae = pipe.vae
vae.model.to('cuda', dtype=torch.float32)
with torch.no_grad():
    latents = x / vae.config.scaling_factor
    if getattr(vae.config, 'shift_factor', None) is not None:
        latents = latents + vae.config.shift_factor
    video = vae.decode(latents).sample                      # [1,C,F,H,W]  [-1,1]
video = ((video.squeeze(0).permute(2,3,4,1) + 1) / 2).clamp(0,1)  # [F,H,W,C] 0-1

out_dir = Path(args.out_dir);  out_dir.mkdir(exist_ok=True, parents=True)
mp4_path = out_dir / 'result.mp4'
png_path = out_dir / 'first_frame.png'
iio.imwrite(out_dir/'result.mp4', (video*255).byte(), fps=pipe.framerate)
iio.imwrite(out_dir/'first_frame.png', (video[0]*255).byte())
print(f'finished. saved to {mp4_path}')
