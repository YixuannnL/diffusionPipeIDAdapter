import sys
import json
import math
import re
import os.path
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/Wan2_1'))

import torch
from torch import nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
import torch.distributed as dist
import safetensors, os
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import insightface, functools

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE
from utils.offloading import ModelOffloader
from utils.common import is_main_process
import wan
from wan.modules.t5 import T5Encoder, T5Decoder, T5Model
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.vae import WanVAE
from wan.modules.model import (
    WanModel, sinusoidal_embedding_1d, WanLayerNorm, WanSelfAttention, WAN_CROSSATTENTION_CLASSES
)
from wan.modules.clip import CLIPModel
from wan import configs as wan_configs
from safetensors.torch import load_file

from .video_id_adapter import VideoIDAdapter


KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 'patch_embedding', 'text_embedding', 'time_embedding', 'time_projection', 'head', 'modulation', 'alpha_id']


class WanModelFromSafetensors(WanModel):
    @classmethod
    def from_pretrained(
        cls,
        weights_file,
        config_file,
        torch_dtype=torch.bfloat16,
        transformer_dtype=torch.bfloat16,
    ):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)

        with init_empty_weights():
            model = cls(**config)

        state_dict = load_file(weights_file, device='cpu')
        state_dict = {
            re.sub(r'^model\.diffusion_model\.', '', k): v for k, v in state_dict.items()
        }

        for name, param in model.named_parameters():
            dtype_to_use = torch_dtype if any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) else transformer_dtype
            if name in state_dict:
                set_module_tensor_to_device(model, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])
            else:
                set_module_tensor_to_device(model, name, device='cpu', dtype=dtype_to_use, value=torch.zeros(param.shape, dtype=dtype_to_use))

        return model

def vae_encode(tensor, vae):
    return vae.model.encode(tensor, vae.scale)

def umt5_keys_mapping_comfy(state_dict):
    import re
    # define key mappings rule
    def execute_mapping(original_key):
        # Token embedding mapping
        if original_key == "shared.weight":
            return "token_embedding.weight"

        # Final layer norm mapping
        if original_key == "encoder.final_layer_norm.weight":
            return "norm.weight"

        # Block layer mappings
        block_match = re.match(r"encoder\.block\.(\d+)\.layer\.(\d+)\.(.+)", original_key)
        if block_match:
            block_num = block_match.group(1)
            layer_type = int(block_match.group(2))
            rest = block_match.group(3)

            # self-attn layer（layer.0）
            if layer_type == 0:
                if "SelfAttention" in rest:
                    attn_part = rest.split(".")[1]
                    if attn_part in ["q", "k", "v", "o"]:
                        return f"blocks.{block_num}.attn.{attn_part}.weight"
                    elif attn_part == "relative_attention_bias":
                        return f"blocks.{block_num}.pos_embedding.embedding.weight"
                elif rest == "layer_norm.weight":
                    return f"blocks.{block_num}.norm1.weight"

            # FFN Layer（layer.1）
            elif layer_type == 1:
                if "DenseReluDense" in rest:
                    parts = rest.split(".")
                    if parts[1] == "wi_0":
                        return f"blocks.{block_num}.ffn.gate.0.weight"
                    elif parts[1] == "wi_1":
                        return f"blocks.{block_num}.ffn.fc1.weight"
                    elif parts[1] == "wo":
                        return f"blocks.{block_num}.ffn.fc2.weight"
                elif rest == "layer_norm.weight":
                    return f"blocks.{block_num}.norm2.weight"

        return None

    new_state_dict = {}
    unmapped_keys = []

    for key, value in state_dict.items():
        new_key = execute_mapping(key)
        if new_key:
            new_state_dict[new_key] = value
        else:
            unmapped_keys.append(key)

    print(f"Unmapped keys (usually safe to ignore): {unmapped_keys}")
    del state_dict
    return new_state_dict


def umt5_keys_mapping_kijai(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("attention.", "attn.")
        new_key = new_key.replace("final_norm.weight", "norm.weight")
        new_state_dict[new_key] = value
    del state_dict
    return new_state_dict

def umt5_keys_mapping(state_dict):
    if 'blocks.0.attn.k.weight' in state_dict:
        print("loading kijai warpper umt5 safetensors model...")
        return umt5_keys_mapping_kijai(state_dict)
    else:
        print("loading comfyui repacked umt5 safetensors model...")
        return umt5_keys_mapping_comfy(state_dict)

# We can load T5 a lot faster by copying some code so we can construct the model
# inside an init_empty_weights() context.

def _t5(name,
        encoder_only=False,
        decoder_only=False,
        return_tokenizer=False,
        tokenizer_kwargs={},
        dtype=torch.float32,
        device='cpu',
        **kwargs):
    # sanity check
    assert not (encoder_only and decoder_only)

    # params
    if encoder_only:
        model_cls = T5Encoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('encoder_layers')
        _ = kwargs.pop('decoder_layers')
    elif decoder_only:
        model_cls = T5Decoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('decoder_layers')
        _ = kwargs.pop('encoder_layers')
    else:
        model_cls = T5Model

    # init model
    with torch.device(device):
        model = model_cls(**kwargs)

    # init tokenizer
    if return_tokenizer:
        tokenizer = HuggingfaceTokenizer(f'google/{name}', **tokenizer_kwargs)
        return model, tokenizer
    else:
        return model


def umt5_xxl(**kwargs):
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        encoder_layers=24,
        decoder_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1)
    cfg.update(**kwargs)
    return _t5('umt5-xxl', **cfg)


class T5EncoderModel:

    def __init__(
        self,
        text_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None,
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        with init_empty_weights():
            model = umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=dtype,
                device=device).eval().requires_grad_(False)

        if checkpoint_path.endswith('.safetensors'):
            state_dict = load_file(checkpoint_path, device='cpu')
            state_dict = umt5_keys_mapping(state_dict)
        else:
            state_dict = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(state_dict, assign=True)
        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean='whitespace')

    def __call__(self, texts, device):
        ids, mask = self.tokenizer(
            texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]


# Wrapper to hold both VAE and CLIP, so we can move both to/from GPU together.
class VaeAndClip(nn.Module):
    def __init__(self, vae, clip):
        super().__init__()
        self.vae = vae
        self.clip = clip


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 use_id_tokens: bool = False,):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        
        # 专用 cross-attn for ID-tokens
        self.use_id_tokens = use_id_tokens # self.use_id_tokens 的默认值是 False；只有在显式指定要加IDToken的那几层时才会被改成 True
        if self.use_id_tokens:
            self.id_cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
                dim, num_heads, (-1, -1), qk_norm, eps
            )
            self.alpha_id = nn.Parameter(torch.full((dim,), 1e-3))
        else:
            self.id_cross_attn = None
            self.alpha_id      = None
        
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        id_tokens=None,        
        id_lens=None, 
        face_emb=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        x = x + y * e[2]

        # cross-attention & ffn function
        # def cross_attn_ffn(x, context, context_lens, e):
            # x = x + self.cross_attn(self.norm3(x), context, context_lens)
        def cross_attn_ffn(x, context, context_lens, id_tokens, id_lens, e):                      
            # 选一个“工作 dtype”：如果当前是 fp8，就升到 bf16，否则保持不变
            work_dtype = torch.bfloat16 if x.dtype in (
                torch.float8_e4m3fn, torch.float8_e5m2) else x.dtype
            
            # text cross-attn
            y_txt = self.cross_attn(self.norm3(x), context, context_lens).to(work_dtype)
            x_work = x.to(work_dtype)
            # id-token cross-attn
            if self.id_cross_attn is not None and id_tokens is not None:
                y_id = self.id_cross_attn(self.norm3(x), id_tokens, id_lens).to(work_dtype)
                x = x_work + y_txt + self.alpha_id.to(work_dtype) * y_id
            else:
                x = x_work + y_txt

            # 如果原来是 fp8，就降回去；否则保持
            if work_dtype != x_work.dtype:
                x = x.to(x_work.dtype)
            
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            x = x + y * e[5]
            return x

        # x = cross_attn_ffn(x, context, context_lens, e)
        x = cross_attn_ffn(x, context, context_lens, id_tokens, id_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        with torch.autocast('cuda', dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


# Patch these to remove some forced casting to float32, saving memory.
wan.modules.model.WanAttentionBlock = WanAttentionBlock
wan.modules.model.Head = Head


class WanPipeline(BasePipeline):
    name = 'wan'
    framerate = 16
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['WanAttentionBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        ckpt_dir = self.model_config['ckpt_path']
        dtype = self.model_config['dtype']
        self.dtype = dtype
        
        # SkyReels V2 uses 24 FPS. There seems to be no better way to autodetect this.
        if 'skyreels' in Path(ckpt_dir).name.lower():
            skyreels = True
            self.framerate = 24
            # FPS is different so make sure to use a new cache dir
            self.name = 'skyreels_v2'
        else:
            skyreels = False

        self.original_model_config_path = os.path.join(ckpt_dir, 'config.json')
        with open(self.original_model_config_path) as f:
            json_config = json.load(f)
        self.i2v = (json_config['model_type'] == 'i2v')
        self.flf2v = (json_config['model_type'] == 'flf2v')
        if self.i2v:
            if skyreels:
                self.name = 'skyreels_v2_i2v'
            else:
                self.name = 'wan_i2v'
        if self.flf2v:
            assert not skyreels
            self.name = 'wan_flf2v'
        model_dim = json_config['dim']
        if not self.i2v and model_dim == 1536:
            wan_config = wan_configs.t2v_1_3B
        elif self.i2v and model_dim == 1536: # There is no official i2v 1.3b model, but there is https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP
            # This is a hack,
            wan_config = wan_configs.t2v_1_3B
            # The following lines are taken from https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/wan_i2v_14B.py
            wan_config.clip_model = 'clip_xlm_roberta_vit_h_14'
            wan_config.clip_dtype = torch.float16
            wan_config.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
            wan_config.clip_tokenizer = 'xlm-roberta-large'
        elif self.i2v and model_dim == 5120:
            wan_config = wan_configs.i2v_14B
        elif self.flf2v and model_dim == 5120:
            wan_config = wan_configs.flf2v_14B
        elif not self.i2v and model_dim == 5120:
            wan_config = wan_configs.t2v_14B
        else:
            raise RuntimeError(f'Could not autodetect model variant. model_dim={model_dim}, i2v={self.i2v}, flf2v={self.flf2v}')

        # This is the outermost class, which isn't a nn.Module
        t5_model_path = self.model_config['llm_path'] if self.model_config.get('llm_path', None) else os.path.join(ckpt_dir, wan_config.t5_checkpoint)
        self.text_encoder = T5EncoderModel(
            text_len=wan_config.text_len,
            dtype=dtype,
            device='cpu',
            checkpoint_path=t5_model_path,
            tokenizer_path=os.path.join(ckpt_dir, wan_config.t5_tokenizer),
            shard_fn=None,
        )

        # Same here, this isn't a nn.Module.
        # TODO: by default the VAE is float32, and therefore so are the latents. Do we want to change that?
        self.vae = WanVAE(
            vae_pth=os.path.join(ckpt_dir, wan_config.vae_checkpoint),
            device='cpu',
        )
        self.video_id_adapter = None
        # These need to be on the device the VAE will be moved to during caching.
        self.vae.mean = self.vae.mean.to('cuda')
        self.vae.std = self.vae.std.to('cuda')
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]

        if self.i2v or self.flf2v:
            self.clip = CLIPModel(
                dtype=dtype,
                device='cpu',
                checkpoint_path=os.path.join(ckpt_dir, wan_config.clip_checkpoint),
                tokenizer_path=os.path.join(ckpt_dir, wan_config.clip_tokenizer)
            )
            
        if self.config.get("use_id_loss", False):
            from models.face_encoder import FaceEncoder
            self.face_encoder = FaceEncoder(device='cpu')
            
            self.id_loss_base          = self.config.get("id_loss_weight", 0.3)
            self.id_loss_warmup_steps  = self.config.get("id_loss_warmup_steps", 3_000)
            self._global_step          = 0
        else:
            self.face_encoder = None
            
        self._vae_on_cuda = False
        self._last_id_loss = None


    # delay loading transformer to save RAM
    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        if transformer_path := self.model_config.get('transformer_path', None):
            self.transformer = WanModelFromSafetensors.from_pretrained(
                transformer_path,
                self.original_model_config_path,
                torch_dtype=dtype,
                transformer_dtype=transformer_dtype,
            )
        else:
            ckpt_path = Path(self.model_config['ckpt_path'])
            with init_empty_weights():
                self.transformer = WanModel.from_config(ckpt_path / 'config.json')
            state_dict = {}
            for shard in ckpt_path.glob('*.safetensors'):
                with safetensors.safe_open(shard, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            for name, param in self.transformer.named_parameters():
                dtype_to_use = dtype if any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) else transformer_dtype
                if name in state_dict:
                    set_module_tensor_to_device(self.transformer, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])
                else:
                    set_module_tensor_to_device(self.transformer, name, device='cpu', dtype=dtype_to_use, value=torch.zeros(param.shape, dtype=dtype_to_use))

        self.transformer.train()
        # We'll need the original parameter name for saving, and the name changes once we wrap modules for pipeline parallelism,
        # so store it in an attribute here. Same thing below if we're training a lora and creating lora weights.
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    # called from train.py  (same signature as BasePipeline)
    def configure_adapter(self, adapter_config):
        adapter_type = adapter_config['type']
        self.adapter_type = adapter_type
        if adapter_type == 'lora':
            # 保持原有逻辑 – 调用父类
            return super().configure_adapter(adapter_config)

        if adapter_type != 'video_id':
            raise NotImplementedError(f'Adapter type {adapter_type} not implemented.')

        # 1) 构建 VideoIDAdapter
        hidden = self.transformer.dim        # 5120 (14B) or 4096 (1.3B)
        # 从 VAE 的 mean 向量推断潜空间通道数（Wan-VAE: 16）
        latent_ch = int(self.vae.mean.shape[0]) if hasattr(self.vae, "mean") else 4
        self.video_id_adapter = VideoIDAdapter(
            hidden_size       = hidden,
            num_id_tokens     = adapter_config.get('num_id_tokens', 16),
            num_layers        = adapter_config.get('num_layers', 4),
            adapter_dim        = adapter_config.get('adapter_dim', 1024),
            num_heads         = adapter_config.get('num_heads', 16),
            mlp_ratio         = adapter_config.get('mlp_ratio', 4),
            dropout           = adapter_config.get('dropout', 0.0),
            proj_in_channels  = latent_ch,
            grid_size         = (adapter_config.get('grid_size', 8)[0], adapter_config.get('grid_size', 8)[1]),
            pool_mode         = adapter_config.get('pool_mode', "grid"),
        ).to(self.model_config['dtype'])

        # 2) 冻结 Wan-Transformer
        for p in self.transformer.parameters():
            p.requires_grad_(False)
            
        # 3) 把指定层打上 use_id_tokens=True 并加 id_cross_attn
        inject_layers = adapter_config.get('layers', [])
        for idx in inject_layers:
            blk = self.transformer.blocks[idx]
            if not getattr(blk, "use_id_tokens", False):
                blk.use_id_tokens = True
                # 拿同一个 CrossAttention 类直接实例化
                cross_attn_cls = blk.cross_attn.__class__
                blk.id_cross_attn = cross_attn_cls(
                    blk.dim,
                    blk.num_heads,
                    (-1, -1),
                    blk.qk_norm,
                    blk.eps,
                ).to(                                       # 与原 cross-attn 对齐 dtype / device
                    blk.cross_attn.q.weight.device,
                    dtype=self.model_config["dtype"],      # 强制 bfloat16 / fp16
                )
            if getattr(blk, "alpha_id", None) is None:
                mean_val = blk.cross_attn.o.weight.float().mean().to(self.model_config["dtype"])
                blk.alpha_id = torch.nn.Parameter(mean_val.expand(blk.dim).clone(), requires_grad=True)
                # blk.alpha_id = torch.nn.Parameter(
                #     torch.full((blk.dim,), 1e-3, dtype=self.model_config["dtype"]),
                #     requires_grad=True,
                # )
                blk.alpha_id.original_name = f"blocks.{idx}.alpha_id"
                
                # 记录 warm‑up 目标值（0.05）——存在 buffer 里，训练循环再按比例写回
                blk.register_buffer("_alpha_target",
                                    torch.full((1,), 0.05,
                                               dtype=self.model_config["dtype"]),
                                    persistent=False)
                # 注册梯度放大 hook：所有 alpha_id 的 grad ×10
                blk.alpha_id.register_hook(lambda g: g * 10.0)
            # --- 给新建 cross-attn 的每个参数打 original_name ---
            for n, p in blk.id_cross_attn.named_parameters():
                p.original_name = f"blocks.{idx}.id_cross_attn.{n}"
                
        # 4) 给 Adapter 自身权重打 original_name
        for n, p in self.video_id_adapter.named_parameters():
            p.original_name = f"video_id_adapter.{n}"

        # 5) 把 Adapter 参数加入训练列表
        for p in self.video_id_adapter.parameters():
            p.requires_grad_(True)
            
        # ------------------------------------------------------------------
        # 在 selected blocks 上挂 4‑bit LoRA （v_proj / out_proj）
        # ------------------------------------------------------------------
        if adapter_config.get("enable_value_lora", True):
            from peft.tuners.lora import LoraConfig, LoraModel
            from peft import prepare_model_for_kbit_training
            from peft import get_peft_model

            # 哪几层挂 LoRA ＝ adapter_config['layers']
            lora_layers = adapter_config.get("layers", [])
            if not lora_layers:
                raise ValueError(
                    "[LoRA] adapter.layers 为空；请在 TOML 里显式指定要注入的 block 索引列表"
                )
            print(f"[LoRA] apply 4‑bit LoRA to blocks: {lora_layers}")

            # PEFT LoRA 配置
            lora_cfg = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=["v", "o"],   # Wan*Attention 命名
                task_type="FEATURE_EXTRACTION",
            )

            for idx in lora_layers:
                blk = self.transformer.blocks[idx]
                for name in ("self_attn", "cross_attn"):
                    attn = getattr(blk, name)
                    # ① 把内部 Linear 换成 4‑bit 权重（freeze）
                    attn = prepare_model_for_kbit_training(
                        attn, use_gradient_checkpointing=False
                    )
                    # ② 对 Linear 注入 LoRA
                    attn = get_peft_model(attn, lora_cfg, adapter_name=f"blk{idx}_{name}")
                    # ③ 打 original_name 便于 Saver 收集
                    for n, p in attn.named_parameters():
                        if p.requires_grad:
                            p.original_name = (
                                f"blocks.{idx}.{name}.{n.replace('.', '_')}"
                            )
            
        # ------------------------------------------------------------------
        #  LayerNorm γ / β 解冻
        # ------------------------------------------------------------------
        for m in self.transformer.modules():
            if isinstance(m, torch.nn.LayerNorm) and m.elementwise_affine:
                m.weight.requires_grad_(True)
                if m.bias is not None:
                    m.bias.requires_grad_(True)
                # 打 name
                for n, p in m.named_parameters(recurse=False):
                    p.original_name = (
                        f"blocks_layernorm.{id(m)}.{n}"  # 唯一即可
                    )

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def get_vae(self):
        vae = self.vae.model
        clip = self.clip.model if self.i2v or self.flf2v else None
        return VaeAndClip(vae, clip)

    def get_text_encoders(self):
        # Return the inner nn.Module
        return [self.text_encoder.model]

    def load_video_id_adapter_weights(self, weights_file: str):
        """
        自动：
        ① 若 video_id_adapter 尚未创建，则 **根据权重 shape** 推断超参并实例化
        ② 为需要注入的 blocks 创建 id_cross_attn
        ③ 加载权重
        """
        sd = load_file(weights_file, device='cpu')
        
        # ---------- 1.  若 adapter 不存在 → 根据 state_dict 推断并创建 ----------
        if getattr(self, "video_id_adapter", None) is None:
            # 1) 基础形状信息
            tok_key   = next(k for k in sd if k.startswith("video_id_adapter.id_tokens"))
            _, N, adapter_dim = sd[tok_key].shape

            # 2) 输入通道 from in_proj.weight
            inproj_key = next(k for k in sd if k.startswith("video_id_adapter.in_proj.weight"))
            in_channels = sd[inproj_key].shape[1]          # 16 in your ckpt

            # 3) num_layers
            layer_idxs = [int(m.group(1)) for k in sd
                        if (m := re.match(r"video_id_adapter\.transformer\.layers\.(\d+)\.", k))]
            num_layers = max(layer_idxs) + 1

            # 4) num_heads  – choose divisor of adapter_dim
            def pick_heads(dim):
                for h in (16, 12, 8, 4, 2, 1):
                    if dim % h == 0:
                        return h
                return 4
            num_heads = pick_heads(adapter_dim)

            hidden = self.transformer.dim  # 5120
            self.video_id_adapter = VideoIDAdapter(
                hidden_size        = hidden,
                num_id_tokens      = N,
                num_layers         = num_layers,
                num_heads          = num_heads,
                adapter_dim        = adapter_dim,
                proj_in_channels   = in_channels,   # ← auto-detected (16)
            ).to(dtype=self.model_config["dtype"], device='cuda')

            print(f"[adapter] instantiated: "
                f"N={N}, adapter_dim={adapter_dim}, in_ch={in_channels}, "
                f"layers={num_layers}, heads={num_heads}")

        # ---------- 2.  为每个 blocks.<idx>.id_cross_attn.* 创建模块 ----------
        inject_blocks = set(
            int(m.group(1)) for k in sd
            if (m := re.match(r"blocks\.(\d+)\.id_cross_attn\.", k))
        )
        for idx in inject_blocks:
            blk = self.transformer.blocks[idx]
            if not hasattr(blk, "id_cross_attn"):
                cross_cls = blk.cross_attn.__class__
                blk.id_cross_attn = cross_cls(
                    blk.dim, blk.num_heads, (-1,-1),
                    # blk.qk_norm, 
                    # blk.eps
                    getattr(blk.cross_attn, "model", blk.cross_attn).qk_norm,
                    getattr(blk.cross_attn, "model", blk.cross_attn).eps,
                ).to(dtype=self.model_config["dtype"], device='cuda')
                
        # ---------- 3.  load weights ----------
        # 3.1 adapter
        adpt_kv = {k.replace("video_id_adapter.", ""): v for k,v in sd.items()
                   if k.startswith("video_id_adapter.")}
        missing = self.video_id_adapter.load_state_dict(adpt_kv, strict=False)[0]
        print("adapter missing:", len(missing))

        # 3.2 id_cross_attn
        xattn_kv = {k:v for k,v in sd.items() if k.startswith("blocks.")}
        miss2, unexp = self.transformer.load_state_dict(xattn_kv, strict=False)
        print(f"id_cross missing={len(miss2)}, unexpected={len(unexp)}")
        
        # ---------- 4.  done ----------
        self.adapter_type = "video_id"
        
    def save_adapter(self, save_dir, state_dict):
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1) LoRA – 保持旧逻辑
        if getattr(self, "adapter_type", None) == "lora":
            self.peft_config.save_pretrained(save_dir)
            # ComfyUI format.
            peft_state_dict = {'diffusion_model.'+k: v for k, v in state_dict.items()}
            safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})
            return
        
        # 2) Video-ID Adapter – 仅需保存 adapter & id_cross_attn 权重
        if getattr(self, "adapter_type", None) == "video_id":
            # 直接把 state_dict 打包成 safetensors
            safetensors.torch.save_file(
                state_dict, save_dir / "video_id_adapter.safetensors",
                metadata={"format": "pt"}
            )
            
            # safetensors.torch.save_file(
            #     self.trainable_state_dict(),
            #     save_dir / "video_id_deltas.safetensors",
            #     metadata={"format": "pt"}
            # )
            return

        # 3) 其它类型暂不支持
        raise NotImplementedError(
            f"save_adapter not implemented for adapter_type={getattr(self, 'adapter_type', None)}"
        )        

    def save_model(self, save_dir, diffusers_sd):
        raise NotImplementedError()

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
            round_height=8,
            round_width=8,
            round_frames=4,
        )

    def get_call_vae_fn(self, vae_and_clip):
        def fn(tensor):
            vae = vae_and_clip.vae
            p = next(vae.parameters())
            tensor = tensor.to(p.device, p.dtype)           
            latents = vae_encode(tensor, self.vae)
            ret = {'latents': latents}

            # ====== Face-ID branch ======
            if self.face_encoder is not None:
                # 取首帧，范围保持 [-1,1]
                first = tensor[:, :, 0, ...]                 # (B,C,H,W)
                # InsightFace 推荐 112×112；这里 224×224 精度更好
                first_resized = torch.nn.functional.interpolate(
                    first, size=(224,224), mode='bilinear', align_corners=False
                )
                with torch.no_grad():
                    emb = self.face_encoder(first_resized)   # (B,512), 已经 L2-norm
                ret['face_emb'] = emb.to('cpu')              # 存 CPU，方便写 Arrow
            # ==================================
            clip = vae_and_clip.clip
            if clip is not None:
                assert tensor.ndim == 5, f'i2v/flf2v must train on videos, got tensor with shape {tensor.shape}'
                first_frame = tensor[:, :, 0:1, ...].clone()
                clip_context = self.clip.visual(first_frame.to(p.device, p.dtype))

                if self.flf2v:
                    last_frame = tensor[:, :, -1:, ...].clone()
                    # NOTE: dim=1 is a hack to pass clip_context without microbatching breaking the zeroth dim
                    clip_context = torch.cat([clip_context, self.clip.visual(last_frame.to(p.device, p.dtype))], dim=1)
                    tensor[:, :, 1:-1, ...] = 0
                else:
                    tensor[:, :, 1:, ...] = 0

                # Image conditioning. Same shame as latents, first frame is unchanged, rest is 0.
                # NOTE: encoding 0s with the VAE doesn't give you 0s in the latents, I tested this. So we need to
                # encode the whole thing here, we can't just extract the first frame from the latents later and make
                # the rest 0. But what happens if you do that? Probably things get fried, but might be worth testing.
                y = vae_encode(tensor, self.vae)
                ret['y'] = y
                ret['clip_context'] = clip_context
            return ret
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video):
            # Args are lists
            p = next(text_encoder.parameters())
            ids, mask = self.text_encoder.tokenizer(caption, return_mask=True, add_special_tokens=True)
            ids = ids.to(p.device)
            mask = mask.to(p.device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            with torch.autocast(device_type=p.device.type, dtype=p.dtype):
                text_embeddings = text_encoder(ids, mask)
                return {'text_embeddings': text_embeddings, 'seq_lens': seq_lens}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):      
        latents = inputs['latents'].float()
        # TODO: why does text_embeddings become float32 here? It's bfloat16 coming out of the text encoder.
        text_embeddings = inputs['text_embeddings']
        seq_lens = inputs['seq_lens']
        mask = inputs['mask']
        y = inputs['y'] if self.i2v or self.flf2v else None
        clip_context = inputs['clip_context'] if self.i2v or self.flf2v else None
        face_emb = inputs.get('face_emb', None)

        bs, channels, num_frames, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # make mask same number of dims as target

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        # timestep input to model needs to be in range [0, 1000]
        t = t * 1000
        
        # -------- Video-ID Adapter --------
        # id_tokens, id_lens = None, None # 现在由 AdapterLayer 生成
        # -------- Video-ID Adapter --------
        if self.video_id_adapter is not None:
            adap_dev = next(self.video_id_adapter.parameters()).device
            latents_for_adp = latents.to(adap_dev, dtype=self.model_config["dtype"])
            with torch.autocast(device_type=adap_dev.type,
                                dtype=self.model_config["dtype"]):
                id_tokens = self.video_id_adapter(latents_for_adp)   # [B,N,D]
            id_lens = torch.full((bs,), id_tokens.size(1),
                                dtype=torch.long, device=id_tokens.device)
        else:
            id_tokens, id_lens = None, None
        return (
            # (x_t, y, t, text_embeddings, seq_lens, clip_context),
            (x_t, y, t, text_embeddings, seq_lens, clip_context, id_tokens, id_lens, face_emb),
            (target, mask, face_emb, x_t.detach(), t.detach()),
        )

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        
        # —— 在 InitialLayer 之后插入 AdapterLayer ——
        if self.video_id_adapter is not None:
            from models.adapter_layer import AdapterLayer
            layers.append(AdapterLayer(self.video_id_adapter))
        
        for i, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        blocks = transformer.blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        transformer.blocks = None
        transformer.to('cuda')
        transformer.blocks = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()
        
    def get_loss_fn(self):
        mse = torch.nn.MSELoss(reduction="none")          # 与原实现一致
        
        def _decode_and_embed(z):
            """
            z : (B, C, H, W)  [-latent-]    (不含帧维度)
            返回 : (B, 512)     已 L2‑norm
            """
            rgb = self.vae.model.decode(z, self.vae.scale)            # [-1, 1]
            rgb = rgb.clamp_(-1, 1)
            emb = self.face_encoder(rgb)                         # (B,512)
            return emb

        # 通过闭包捕获 self
        def loss_fn(model_output, label):
            """
            model_output : (B,C,F,H,W)   — ε̂ 或 v̂
            label        : (target, mask [, ref_face])
            return       : scalar loss
            """
            # ----------- 1. 解析 label -----------
            target, mask, ref_face, x_t, t = label
            target = target.to(model_output.dtype)

            # ----------- 2. 扩散 MSE/V-pred 损失 -----------   
            diff_loss = mse(model_output, target)        
            if mask is not None and mask.numel():
                diff_loss = diff_loss * mask.to(diff_loss.dtype)
            diff_loss = diff_loss.mean()

            total_loss = diff_loss                         # 初始化总损失

            # ----------- 3. Face-ID 损失 -----------
            if (
                (self.face_encoder is not None) and        # 已启用人脸 Encoder
                (ref_face is not None) and                 # 数据里带 ref 向量
                ref_face.abs().sum() > 0                   # 不是全 0（= 未检出脸）
            ):
                t_exp = t.view(-1, 1, 1, 1, 1).to(model_output.dtype)
                x1_pred = x_t.to(model_output.dtype) - t_exp * model_output
                # i = torch.randint(0, model_output.size(2), (1,)).item() # 随机取一帧
                lat = x1_pred[:, :, 0:1, :, :]
                lat = lat.to(dtype=self.vae.dtype, memory_format=torch.contiguous_format)
                
                dev = lat.device
                if not self._vae_on_cuda or next(self.vae.model.parameters()).device != dev:
                    self.vae.model.to(dev)
                    self.vae.mean  = self.vae.mean.to(dev)
                    self.vae.std   = self.vae.std.to(dev)
                    self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]
                    self.face_encoder.to(dev)
                    self._vae_on_cuda = True


                emb_pred = torch.utils.checkpoint.checkpoint(
                    _decode_and_embed, lat, use_reentrant=False
                ).float()                        # ID 向量统一 float32

                # (d) cosine 相似度 → loss
                ref_face = ref_face.to(emb_pred.device, dtype=emb_pred.dtype)
                cos_sim  = torch.sum(emb_pred * ref_face, dim=1)    # (B,)
                id_loss  = (1 - cos_sim).mean().to(model_output.dtype)

                # (e) 逐步放大权重
                scale_w  = min(1.0, self._global_step / max(1, self.id_loss_warmup_steps))
                total_loss = total_loss + self.id_loss_base * scale_w * id_loss

                if 'id_loss_val' in locals():
                    self._last_id_loss = (id_loss.detach(), scale_w)  # 只存本地，先不通信
            # ----------- 4. 返回 -----------
            return total_loss

        return loss_fn

    def trainable_state_dict(self):
        """
        收集 **当前 rank** 持有且 requires_grad=True 的所有参数，
        不依赖 self.named_parameters()，避免 __getattr__ 递归。
        返回 {name: tensor.cpu()} 字典，可供 all‑gather 后保存。
        """
        state = {}
        seen = set()                             # 防止重复

        def _collect(prefix: str, mod: torch.nn.Module):
            for n, p in mod.named_parameters(recurse=True):
                if p.requires_grad and id(p) not in seen:
                    key = getattr(p, "original_name", f"{prefix}.{n}" if prefix else n)
                    state[key] = p.detach().cpu()
                    seen.add(id(p))

        # 2. Wan 主干里真正是 nn.Module 的字段，例如 transformer / vae 等
        for attr_name, attr_val in self.__dict__.items():
            if isinstance(attr_val, torch.nn.Module):
                _collect(attr_name, attr_val)

        return state      # 仅本 rank，外层 Saver 会负责 all‑gather & merge

    def eval(self):
        self.transformer.eval()
        if hasattr(self, "video_id_adapter") and self.video_id_adapter is not None:
            self.video_id_adapter.eval()
        return self
    
    # -------- 按 warm‑up 进度把 alpha_id 写成目标值 × scale --------
    def set_alpha_scale(self, scale: float):
        """在训练循环里每个 step 调一次, scale∈[0,1]"""
        for blk in self.transformer.blocks:
            if hasattr(blk, "alpha_id") and hasattr(blk, "_alpha_target"):
                blk.alpha_id.data.copy_(blk._alpha_target * scale)

class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.patch_embedding = model.patch_embedding
        self.time_embedding = model.time_embedding
        self.text_embedding = model.text_embedding
        self.time_projection = model.time_projection
        self.i2v = (model.model_type == 'i2v')
        self.flf2v = (model.model_type == 'flf2v')
        if self.i2v or self.flf2v:
            self.img_emb = model.img_emb
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
                
        if len(inputs) == 9:
            x, y, t, context, text_seq_lens, clip_fea, id_tokens, id_lens, face_emb = inputs
        else:  # 旧路径 (无 adapter)
            x, y, t, context, text_seq_lens, clip_fea = inputs
            id_tokens, id_lens = None, None

        bs, channels, f, h, w = x.shape
        if clip_fea.numel() == 0:
            clip_fea = None
        context = [emb[:length] for emb, length in zip(context, text_seq_lens)]

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if self.i2v or self.flf2v:
            mask = torch.zeros((bs, 4, f, h, w), device=x.device, dtype=x.dtype)
            mask[:, :, 0, ...] = 1
            if self.flf2v:
                mask[:, :, -1, ...] = 1
            y = torch.cat([mask, y], dim=1)
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        seq_len = seq_lens.max()
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(x.device, torch.float32))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if self.i2v or self.flf2v:
            assert clip_fea is not None
            if self.flf2v:
                self.img_emb.emb_pos.data = self.img_emb.emb_pos.data.to(clip_fea.device, torch.float32)
                clip_fea = clip_fea.view(-1, 257, 1280)
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        # pipeline parallelism needs everything on the GPU
        seq_lens = seq_lens.to(x.device)
        grid_sizes = grid_sizes.to(x.device)

        return make_contiguous(x, e, e0, seq_lens, grid_sizes, self.freqs, context, id_tokens, id_lens, face_emb)


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context, id_tokens, id_lens, face_emb = inputs

        self.offloader.wait_for_block(self.block_idx)
        # x = self.block(x, e0, seq_lens, grid_sizes, freqs, context, None)
        x = self.block(
            x, e0, seq_lens, grid_sizes, freqs, context, None,
            id_tokens=id_tokens, id_lens=id_lens
        )
        self.offloader.submit_move_blocks_forward(self.block_idx)

        # return make_contiguous(x, e, e0, seq_lens, grid_sizes, freqs, context)
        return make_contiguous(x, e, e0, seq_lens, grid_sizes, freqs,
                               context, id_tokens, id_lens, face_emb)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.head
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        # drop id-tokens tuple elements
        x, e, e0, seq_lens, grid_sizes, freqs, context, *_ = inputs

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x, dim=0)
