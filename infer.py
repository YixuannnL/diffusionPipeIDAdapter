from __future__ import annotations

import os, sys, argparse, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from pathlib import Path
from typing import Tuple, Optional

ROOT_DIR      = os.path.abspath(os.path.dirname(__file__))
WAN_SUBMODULE = os.path.join(ROOT_DIR, "submodules", "Wan2_1")
if WAN_SUBMODULE not in sys.path:
    sys.path.insert(0, WAN_SUBMODULE)

import torch
import toml
import imageio.v3 as iio
from torchvision import transforms as T, transforms as Tv

# ───────── 性能开关 (无需 CLI) ─────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
try:                                    # 自动 SDPA 选 Flash‑Attn / MEM‑Eff
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    print("[Speed‑up] Flash‑Attention 2 / SDPA enabled ✅")
except Exception as e:
    import xformers.ops                 # noqa: F401
    print(f"[Speed‑up] SDPA fallback to xformers ({e})")

# ───────── 类型映射 ─────────
_DTYPE_MAP = {
    "float16": torch.float16, "fp16": torch.float16, "f16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float8": torch.float8_e4m3fn, "fp8":    torch.float8_e4m3fn,
}
def _resolve_dtype(val):
    if isinstance(val, torch.dtype) or val is None:
        return val
    return _DTYPE_MAP[str(val).lower()]

# ───────── 读取视频 ─────────
def _read_video_frames(
    path: Path | str,
    resize: Tuple[int, int] | None = None,
    max_frames: int | None = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    frames = []
    for i, f in enumerate(iio.imiter(path)):
        if max_frames and i >= max_frames:
            break
        t = torch.from_numpy(f).permute(2,0,1)
        if resize is not None:
            t = T.functional.resize(t, resize, antialias=True)
        frames.append(t)
    vid = torch.stack(frames).to(dtype=dtype).div_(127.5).sub_(1)   # ‑1‥1
    return vid.cuda(non_blocking=True)   

def _choose_wan_config(pipe):
    """根据 ckpt 的 JSON 自动选择 Wan 配置常量"""
    with open(pipe.original_model_config_path, "r") as f:
        meta = json.load(f)

    model_dim = meta["dim"]
    if pipe.i2v:
        if model_dim == 5120:
            return wan_cfgs.i2v_14B
        else:                       # 5120 以外目前只有 1.3B（实验用）
            cfg = wan_cfgs.t2v_1_3B
            # 官方 i2v‑1.3B 额外三行：保持和 WanPipeline.__init__ 一致
            cfg.clip_model      = 'clip_xlm_roberta_vit_h_14'
            cfg.clip_dtype      = torch.float16
            cfg.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
            cfg.clip_tokenizer  = 'xlm-roberta-large'
            return cfg
    elif pipe.flf2v:
        return wan_cfgs.flf2v_14B          # 目前只有 14B
    else:                                   # t2v
        return wan_cfgs.t2v_14B if model_dim == 5120 else wan_cfgs.t2v_1_3B

def _build_runner(pipe, device_id: int = 0):
    """创建 WanX2V 对象，并把已经加载好的子模块 *塞* 进去，避免重复占显存"""
    cfg = _choose_wan_config(pipe)
    ckpt_dir = pipe.model_config["ckpt_path"]

    if pipe.i2v:
        runner = wan.WanI2V(cfg, checkpoint_dir=ckpt_dir,
                            device_id=device_id, rank=0,
                            t5_fsdp=False, dit_fsdp=False,
                            use_usp=False, t5_cpu=False)
    elif pipe.flf2v:
        runner = wan.WanFLF2V(cfg, checkpoint_dir=ckpt_dir,
                              device_id=device_id, rank=0,
                              t5_fsdp=False, dit_fsdp=False,
                              use_usp=False, t5_cpu=False)
    else:
        runner = wan.WanT2V(cfg, checkpoint_dir=ckpt_dir,
                            device_id=device_id, rank=0,
                            t5_fsdp=False, dit_fsdp=False,
                            use_usp=False, t5_cpu=False)

    # ---- 重用训练时已在 GPU 的权重，包含 video‑id‑adapter ----
    runner.model        = pipe.transformer                 # DiT 主干
    runner.vae          = pipe.vae                         # Wan‑VAE
    runner.text_encoder = pipe.text_encoder                # UM‑T5
    if getattr(pipe, "video_id_adapter", None):
        runner.model.video_id_adapter = pipe.video_id_adapter
        runner.model.adapter_type     = "video_id"

    # 其余子模块（clip 等）WanX2V 自己内部会处理
    return runner, cfg# [F,3,H,W]

# ───────── Wan imports ─────────
import wan                                           
from wan import configs as wan_cfgs
from models.wan import WanPipeline                   


def build_model(cfg: dict, device: torch.device, dtype: torch.dtype, compile_graph: bool):
    """a) 默认不 compile b) 构建后整体搬 GPU."""
    from utils.common import DTYPE_MAP                # 原 mapping
    
    pipe = WanPipeline(cfg)
    pipe.load_diffusion_model()
    
    
    adapter_cfg = cfg.get("adapter", {})
    if adapter_cfg and adapter_cfg.get("type") == "video_id":
        pipe.configure_adapter(adapter_cfg)
    
    adapter_ckpt = cfg.get("adapter", {}).get("ckpt_path")
    if adapter_ckpt is None:
        raise ValueError("Please specify adapter.ckpt_path in the TOML.")
    
    from safetensors.torch import load_file
    full_sd = load_file(adapter_ckpt, device='cpu')
    
    # ───────────── 修复 LoRA 键名 ─────────────
    import re

    def _fix_lora_key(k: str) -> str:
        """把训练阶段存储时被 `_` 扁平化的 LoRA 参数名还原成带 `.` 的真实层级名"""
        if not k.startswith("blocks.") or "_lora_" not in k:
            # 非 LoRA / 非 blocks 开头的不需要动
            return k
        # model_ → model.
        k = k.replace("model_", "model.")
        # *_lora_[A|B|E]_* → *.lora_[A|B|E].*
        k = re.sub(r"_lora_([ABEinE])_", r".lora_\1.", k)
        # 末尾 _weight / _bias → .weight / .bias
        k = re.sub(r"_weight$", ".weight", k)
        k = re.sub(r"_bias$",   ".bias",   k)
        return k

    full_sd = {_fix_lora_key(k): v for k, v in full_sd.items()}
    
    # A  把 blocks.* 权重先尝试补进 transformer
    xattn_sd = {k: v for k, v in full_sd.items() if k.startswith("blocks.")}
    if xattn_sd:
        incomp, unexp = pipe.transformer.load_state_dict(xattn_sd, strict=False)
        print(f"unexpected={len(unexp)}")
    import pdb; pdb.set_trace()
    # B  再让适配器本体去加载自己那部分
    load_fn = (
        getattr(pipe, "load_video_id_adapter_weights", None)
        or getattr(pipe, "load_adapter_weights", None)
    )
    load_fn(adapter_ckpt)

    # --- 把需要的子模块搬到 GPU ---
    mods = [
        pipe.transformer,
        pipe.text_encoder.model,
        pipe.vae.model,
        pipe.video_id_adapter          # 可能是 None，但属性一定存在
    ]

    # 只有 i2v / flf2v 变体才有 clip
    if pipe.i2v or pipe.flf2v:
        mods.append(pipe.clip.model)   # clip 是包装器对象，里面的 nn.Module 才需要 .to()

    for m in mods:
        if m is not None:
            m.to(device=device, dtype=dtype, non_blocking=True)

    pipe.eval()
    
    if compile_graph:                                 
        try:
            pipe.transformer = torch.compile(
                pipe.transformer, mode="reduce-overhead",
                fullgraph=False, dynamic=True)
            print("[Speed‑up] torch.compile on transformer ✅")
        except Exception as e:
            print(f"[Speed‑up] compile skipped ({e})")
    return pipe

def generate_video(
    pipe,
    prompt: str,
    ref_vid: torch.Tensor,
    cfg: dict,
    seed: int,
    num_frames: int,
    guidance: float,
):
    torch.manual_seed(seed)
    device_id = torch.cuda.current_device()
    runner, _ = _build_runner(pipe, device_id)
    
    # ↓ 全程关闭 grad
    with torch.inference_mode(), torch.autocast("cuda", dtype=ref_vid.dtype):
        extra_kwargs: dict[str, object] = {}
        if pipe.i2v:
            first = (ref_vid[0].clamp(-1,1).add(1).mul_(127.5)
                     .to(torch.uint8).cpu())
            extra_kwargs["src_image"] = Tv.ToPILImage()(first)
        elif pipe.flf2v:
            extra_kwargs["first_frame"] = ref_vid[0]
            extra_kwargs["last_frame"]  = ref_vid[-1]

        # Wan 的 size 配置
        # SIZE_CONFIGS = {
        #     '720*1280': (720, 1280),
        #     '1280*720': (1280, 720),
        #     '480*832': (480, 832),
        #     '832*480': (832, 480),
        #     '1024*1024': (1024, 1024),
        # }

        size_str  = cfg.get("size", "832*480")
        size_tuple = wan_cfgs.SIZE_CONFIGS.get(size_str, wan_cfgs.SIZE_CONFIGS["832*480"])
        max_area   = wan_cfgs.MAX_AREA_CONFIGS[size_str]

        common = dict(frame_num=num_frames,
                      sample_solver=cfg.get("sample_solver", "unipc"),
                      sampling_steps=cfg.get("sample_steps", 50),
                      guide_scale=guidance,
                      seed=seed,
                      offload_model=False)

        if pipe.i2v:
            video = runner.generate(prompt, extra_kwargs["src_image"],
                                     max_area=max_area, **common)
        elif pipe.flf2v:
            video = runner.generate(prompt, extra_kwargs["first_frame"],
                                     extra_kwargs["last_frame"], max_area=max_area, **common)
        else:
            video = runner.generate(prompt, size=size_tuple, **common)

    video = (video.clamp_(-1,1).add_(1).mul_(127.5)
                   .to(torch.uint8).permute(1,0,2,3).contiguous())   # [F,3,H,W]
    return video

# ───────── CLI / entry‑point ─────────
def main():                                          # 保持原 argparse
    parser = argparse.ArgumentParser(description="Identity‑preserving text‑to‑video (Wan 2.1 + Video‑ID adapter)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--ref_video", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--name", type=str, default="sample_1")
    group = parser.add_argument_group("speed‑ups")
    group.add_argument("--fp16", action="store_true")
    group.add_argument("--flash", action="store_true")      # 仍保留但已自动启
    group.add_argument("--xformers", action="store_true")   # ditto
    group.add_argument("--compile", action="store_true")
    group_quality = parser.add_argument_group("generation params")
    group_quality.add_argument("--frames", type=int, default=49)
    group_quality.add_argument("--seed", type=int, default=42)
    group_quality.add_argument("--guidance", type=float, default=8.0)
    args = parser.parse_args()

    cfg_raw: dict = toml.load(args.config)
    # —— 原 _normalize_cfg() 复制过来 ↓↓↓
    def _normalize_cfg(cfg: dict) -> dict:
        m = cfg.get("model", {})
        if "dtype" in m:            m["dtype"]            = _resolve_dtype(m["dtype"])
        if "transformer_dtype" in m: m["transformer_dtype"] = _resolve_dtype(m["transformer_dtype"])
        a = cfg.get("adapter", {})
        if "dtype" in a:            a["dtype"]            = _resolve_dtype(a["dtype"])
        if "weights" in a and "ckpt_path" not in a:
            a["ckpt_path"] = a.pop("weights")
        return cfg
    cfg = _normalize_cfg(cfg_raw)

    device = torch.device("cuda")
    dtype  = torch.float16 if args.fp16 else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    print(f"Using dtype={dtype}, device={device}")

    pipe = build_model(cfg, device, dtype, compile_graph=args.compile)

    ref_vid = _read_video_frames(args.ref_video, dtype=dtype)

    print("Generating…")
    t0 = time.time()
    with torch.inference_mode():                 # 再保险
        video = generate_video(
            pipe, prompt=args.prompt, ref_vid=ref_vid,
            cfg=cfg, seed=args.seed, num_frames=args.frames, guidance=args.guidance)
    dt = time.time() - t0
    print(f"Done in {dt:.1f} s  (≈{video.shape[0]/dt:.1f} FPS) ✅")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.name}.mp4"
    iio.imwrite(out_path, video.permute(0,2,3,1).cpu().numpy(), fps=16)
    print("Saved →", out_path)

if __name__ == "__main__":
    main()
