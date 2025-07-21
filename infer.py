#!/usr/bin/env python
"""
Fast inference script for Wan 2.1 + Video‑ID adapter (trained with diffusion‑pipe).

Features
--------
* Loads the **base Wan 2.1 checkpoint** *and* the fine‑tuned **video_id_adapter.safetensors**
  (which also contains the per‑layer ID‑cross‑attention weights).
* Reads a **TOML configuration** (same schema as training) so you don’t have to keep the
  CLI huge.
* Supports several opt‑in **speed‑ups**:
  – fp16 / bf16 autocast
  – Flash‑Attention 2 (if installed)
  – `xformers` memory‑efficient attention (fallback if Flash‑Attn unavailable)
  – `torch.compile()` / Torch‑Dynamo (PyTorch 2.1+)  
  – CUDA graph warm‑up for constant shape workloads
* Multi‑GPU **model sharding** (identical to training) or single‑GPU inference.
* Simple **CLI** – pass prompt + reference video, get an MP4 in `--output_dir`.

> **NOTE**  The exact public API of `WanPipeline` is not published at the time of writing.
> The helper utilities below call only the *documented* methods used in `train.py`.
> If you changed those, adapt the marked TODOs.
"""
from __future__ import annotations
# ---- 让 submodules/Wan2_1/wan/ 成为可导入包 -------------------------
import os, sys
ROOT_DIR      = os.path.abspath(os.path.dirname(__file__))
WAN_SUBMODULE = os.path.join(ROOT_DIR, "submodules", "Wan2_1")
if WAN_SUBMODULE not in sys.path:
    sys.path.insert(0, WAN_SUBMODULE)
# -------------------------------------------------------------------

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import time
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import toml
from torchvision import transforms as T

import wan                                    
from wan import configs as wan_cfgs
from wan.configs import SIZE_CONFIGS, MAX_AREA_CONFIGS
from torchvision import transforms as Tv

# --------------------  dtype helpers  --------------------
_DTYPE_MAP = {
    # fp32/16
    "float32": torch.float32,  "fp32": torch.float32, "f32": torch.float32,
    "float16": torch.float16,  "fp16": torch.float16, "f16": torch.float16,
    # bf16
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    # fp8  (选常见 e4m3fn；如果你用 e5m2 就把键值改掉)
    "float8":  torch.float8_e4m3fn, "fp8": torch.float8_e4m3fn,
}

def _resolve_dtype(val):
    """str → torch.dtype（已是 dtype 就原样返回）"""
    if isinstance(val, torch.dtype) or val is None:
        return val
    try:
        return _DTYPE_MAP[str(val).lower()]
    except KeyError:         # noqa:  B902
        raise ValueError(
            f"Unknown dtype '{val}'. Valid keys: {list(_DTYPE_MAP)}"
        ) from None


# -----------------------------------------------------------------------------
# Optional accelerators
# -----------------------------------------------------------------------------

def _try_enable_flash_attn() -> bool:
    """Monkey‑patch torch MHA to use Flash‑Attn 2 if available."""
    try:
        from flash_attn import flash_attn_interface  # noqa: F401
        from flash_attn.modules.mha import MHA as FlashMHA
        from functools import partial
        import models.flash_patch as flash_patch  # hypothetical helper shipped with repo

        flash_patch.patch_wan_with_flash(MHA_cls=FlashMHA)  # registers custom attn
        print("[Speed‑up] Flash‑Attention enabled ✅")
        return True
    except Exception as e:  # pragma: no‑cover  # noqa: BLE001
        print(f"[Speed‑up] Flash‑Attention not available ({e}).")
        return False


def _try_enable_xformers() -> bool:
    """Fallback memory‑efficient attention (xformers)."""
    try:
        import xformers.ops  # noqa: F401
        from models.xformers_patch import patch_wan_with_xformers  # hypothetical helper

        patch_wan_with_xformers()
        print("[Speed‑up] xFormers attention enabled ✅")
        return True
    except Exception as e:  # pragma: no‑cover  # noqa: BLE001
        print(f"[Speed‑up] xFormers not available ({e}).")
        return False


# -----------------------------------------------------------------------------
# Video I/O helpers (very lightweight – rely on imageio / torchvision)
# -----------------------------------------------------------------------------

def _read_video_frames(
    path: Path | str,
    resize: Tuple[int, int] | None = None,
    max_frames: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Load a video file → float tensor in ‑1‥1, shape [F, 3, H, W]."""
    import imageio.v3 as iio

    ims = []
    for i, frame in enumerate(iio.imiter(path)):
        if max_frames and i >= max_frames:
            break
        im = torch.from_numpy(frame).permute(2, 0, 1)  # HWC → CHW
        if resize is not None:
            im = T.functional.resize(im, resize, antialias=True)
        ims.append(im)
    vid = torch.stack(ims).float() / 127.5 - 1  # 0‥255 → ‑1‥1
    if device is not None:
        vid = vid.to(device, dtype=dtype)
    return vid  # [F,3,H,W]

def _choose_wan_config(pipe):
    """根据 ckpt 的 JSON 自动选择 Wan 配置常量（完全照搬 train.py 里的逻辑）"""
    with open(pipe.original_model_config_path, "r") as f:
        meta = json.load(f)

    model_dim = meta["dim"]
    if pipe.i2v:
        if model_dim == 5120:
            return wan_cfgs.i2v_14B
        else:                       # 5120 以外目前只有 1.3B（实验用）
            cfg = wan_cfgs.t2v_1_3B
            # ↓↓↓ 官方 i2v‑1.3B 额外三行：保持和 WanPipeline.__init__ 一致
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
    return runner, cfg
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device, dtype: torch.dtype, compile_graph: bool):
    """Instantiate WanPipeline + adapter, load weights."""
    from utils.common import DTYPE_MAP  # same mapping used in training

    # 1) Create pipeline skeleton (no weights yet)
    from models import wan
    pipe = wan.WanPipeline(cfg)

    # -------------------------------------------------------------
    # 2) ── Diffusion / VAE 主干（基础权重）
    # -------------------------------------------------------------
    pipe.load_diffusion_model()   # 内部已把 transformer 塞进 pipe.transformer

    # === 找出 base ckpt 仍未提供的参数（被填 0 的） ===
    missing_after_base = []
    for n, p in pipe.transformer.named_parameters():
        if torch.all(p == 0):          # WanModelFromSafetensors 会用 0 占位
            missing_after_base.append(n)
    # -------------------------------------------------------------
    # 3) ── Video‑ID Adapter（包含 blocks.*.id_cross_attn / alpha_id 权重）
    # -------------------------------------------------------------
    adapter_cfg = cfg.get("adapter", {})
    if adapter_cfg and adapter_cfg.get("type") == "video_id":
        pipe.configure_adapter(adapter_cfg)
    
    adapter_ckpt = cfg.get("adapter", {}).get("ckpt_path")
    if adapter_ckpt is None:
        raise ValueError("Please specify adapter.ckpt_path in the TOML.")

    from safetensors.torch import load_file
    full_sd = load_file(adapter_ckpt, device='cpu')

    # 3‑A  把 blocks.* 权重先尝试补进 transformer
    xattn_sd = {k: v for k, v in full_sd.items() if k.startswith("blocks.")}

    if xattn_sd:
        incomp, unexp = pipe.transformer.load_state_dict(xattn_sd, strict=False)
        print(f"unexpected={len(unexp)}")

    # 3‑B  再让适配器本体去加载自己那部分
    load_fn = (
        getattr(pipe, "load_video_id_adapter_weights", None)
        or getattr(pipe, "load_adapter_weights", None)
    )
    load_fn(adapter_ckpt)

    # === 统计最终仍缺失的 key（仍是 0）并打印 ===
    still_missing = []
    for n, p in pipe.transformer.named_parameters():
        if torch.all(p == 0):
            still_missing.append(n)

    print(f"[adapter] ⚠️  missing after base‑ckpt: {len(missing_after_base)}, "
        f"after adapter补齐仍缺: {len(still_missing)}")
    if still_missing:
        for k in still_missing[:20]:          # 只打印前 20 个，够你快速定位
            print("   ", k)
    
    # Optional compilation —— **只编 transformer 子模块**
    if compile_graph and hasattr(torch, "compile"):
        try:
            pipe.transformer = torch.compile(
                pipe.transformer,                 # 仅编译 nn.Module
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=True,
            )
            print("[Speed-up] torch.compile on transformer ✅")
        except Exception as e:
            # 编译失败时退化为正常执行，避免整脚本崩溃
            print(f"[Speed‑up] torch.compile skipped ➜ fallback to eager ({e})")

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
    runner, wand_cfg = _build_runner(pipe, device_id)

    extra_kwargs = {}
    if pipe.i2v:
        first = (ref_vid[0].clamp(-1,1).add(1).mul_(127.5)
                 .to(torch.uint8).cpu())        # [3,H,W] uint8
        extra_kwargs["src_image"] = Tv.ToPILImage()(first)
    elif pipe.flf2v:
        extra_kwargs["first_frame"] = ref_vid[0]
        extra_kwargs["last_frame"]  = ref_vid[-1]
        
    # -------------------- 通用采样参数 --------------------
    # 如果 TOML 里有 size="1280*720" 就用，没有就默认 1280*720
    size_str = cfg.get("size", "1280*720")
    size_tuple = SIZE_CONFIGS.get(size_str, SIZE_CONFIGS["1280*720"])

    # I2V / FLF2V 用 max_area 而不是 size
    max_area = MAX_AREA_CONFIGS[size_str]

    common_args = dict(
        frame_num      = num_frames,
        shift          = cfg.get("sample_shift", 5.0),
        sample_solver  = cfg.get("sample_solver", "unipc"),
        sampling_steps = cfg.get("sample_steps", 50),
        guide_scale    = guidance,
        seed           = seed,
        offload_model  = False,
    )

    # ------------ 正式采样 ------------
    with torch.autocast("cuda", dtype=ref_vid.dtype):
        if pipe.i2v:
            video = runner.generate(
                prompt,
                extra_kwargs["src_image"],
                max_area=max_area,
                **common_args,
            )
        elif pipe.flf2v:
            video = runner.generate(
                prompt,
                extra_kwargs["first_frame"],
                extra_kwargs["last_frame"],
                max_area=max_area,
                **common_args,
            )
        else:  # t2v / t2i
            video = runner.generate(
                prompt,
                size=size_tuple,
                **common_args,
            )

    video = (video.clamp_(-1,1).add_(1).mul_(127.5)).to(torch.uint8)
    video = video.permute(1, 0, 2, 3).contiguous()      # -> [F,3,H,W]
    return video


# -----------------------------------------------------------------------------
# CLI / entry‑point
# -----------------------------------------------------------------------------

def main():  # noqa: C901 – high cyclomatic, fine for script
    parser = argparse.ArgumentParser(description="Identity‑preserving text‑to‑video (Wan 2.1 + Video‑ID adapter)")
    parser.add_argument("--config", type=Path, required=True, help="Path to inference TOML config")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--ref_video", type=Path, required=True, help="Reference video providing identity information")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--name", type=str, default="sample")

    # speed / quality knobs
    group = parser.add_argument_group("speed‑ups")
    group.add_argument("--fp16", action="store_true", help="Force fp16 autocast (default: bf16 if supported)")
    group.add_argument("--flash", action="store_true", help="Enable Flash‑Attention 2")
    group.add_argument("--xformers", action="store_true", help="Enable xformers attention to save memory, will slow down the inference")
    group.add_argument("--compile", action="store_true", help="torch.compile the whole pipeline")

    group_quality = parser.add_argument_group("generation params")
    group_quality.add_argument("--frames", type=int, default=81)
    group_quality.add_argument("--seed", type=int, default=42)
    group_quality.add_argument("--guidance", type=float, default=8.0)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    cfg_raw: dict = toml.load(args.config)

    def _normalize_cfg(cfg: dict) -> dict:
        # --- model ---
        m = cfg.get("model", {})
        if "dtype" in m:
            m["dtype"] = _resolve_dtype(m["dtype"])
        if "transformer_dtype" in m:
            m["transformer_dtype"] = _resolve_dtype(m["transformer_dtype"])

        # --- adapter ---
        a = cfg.get("adapter", {})
        if "dtype" in a:
            a["dtype"] = _resolve_dtype(a["dtype"])
        # 允许用 weights= 作为别名
        if "weights" in a and "ckpt_path" not in a:
            a["ckpt_path"] = a.pop("weights")
        return cfg

    cfg = _normalize_cfg(cfg_raw)

    # device & dtype ----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script currently requires a CUDA‑capable GPU.")

    dtype = torch.float16 if args.fp16 else torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype={dtype}, device={device}")

    # ------------------------------------------------------------------
    # Speed‑ups (patch before model build!)
    if args.flash:
        ok = _try_enable_flash_attn()
        if not ok and args.xformers:
            _try_enable_xformers()
    elif args.xformers:
        _try_enable_xformers()

    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 on A100/H100 → ~20% speed
    torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------
    # Build model + load weights
    pipe = build_model(cfg, device=device, dtype=dtype, compile_graph=args.compile)
    pipe.eval()

    # ------------------------------------------------------------------
    # Prepare reference video → tensor
    ref_vid = _read_video_frames(args.ref_video, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    print("Generating…")
    t0 = time.time()
    video = generate_video(
        pipe,
        prompt=args.prompt,
        ref_vid=ref_vid,
        cfg=cfg,
        seed=args.seed,
        num_frames=args.frames,
        guidance=args.guidance,
    )
    dt = time.time() - t0
    fps = video.shape[0] / dt
    print(f"Done in {dt:.2f}s  (≈{fps:.1f} FPS) ✅")

    # ------------------------------------------------------------------
    # Save MP4
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.name}.mp4"
    import imageio.v3 as iio

    iio.imwrite(out_path, video.permute(0, 2, 3, 1).numpy(), fps=16)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
