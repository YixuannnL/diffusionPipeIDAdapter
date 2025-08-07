from pathlib import Path
import re
from typing import Tuple, Dict, List

import json 
import peft
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import safetensors.torch
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms
import imageio

from utils.common import is_main_process, VIDEO_EXTENSIONS, round_to_nearest_multiple, round_down_to_multiple


def make_contiguous(*tensors):
    return tuple(x.contiguous() for x in tensors)


def extract_clips(video, face_video, target_frames, video_clip_mode):
    # video is (channels, num_frames, height, width)
    frames = video.shape[1]
    if frames < target_frames:
        # TODO: think about how to handle this case. Maybe the video should have already been thrown out?
        print(f'video with shape {video.shape} is being skipped because it has less than the target_frames')
        return []
    if video_clip_mode == 'single_beginning':
        return [video[:, :target_frames, ...]], [face_video[:, :target_frames, ...]] if face_video is not None else None
    elif video_clip_mode == 'single_middle':
        start = int((frames - target_frames) / 2)
        assert frames-start >= target_frames
        return [video[:, start:start+target_frames, ...]], [face_video[:, start:start+target_frames, ...]] if face_video is not None else None
    elif video_clip_mode == 'multiple_overlapping':
        # Extract multiple clips so we use the whole video for training.
        # The clips might overlap a little bit. We never cut anything off the end of the video.
        num_clips = ((frames - 1) // target_frames) + 1
        start_indices = torch.linspace(0, frames-target_frames, num_clips).int()
        return [video[:, i:i+target_frames, ...] for i in start_indices], [face_video[:, i:i+target_frames, ...] for i in start_indices] if face_video is not None else None    
    else:
        raise NotImplementedError(f'video_clip_mode={video_clip_mode} is not recognized')


def convert_crop_and_resize(pil_img, width_and_height):
    if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
        pil_img = pil_img.convert('RGBA')

    # add white background for transparent images
    if pil_img.mode == 'RGBA':
        canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
        canvas.alpha_composite(pil_img)
        pil_img = canvas.convert('RGB')
    else:
        pil_img = pil_img.convert('RGB')

    return ImageOps.fit(pil_img, width_and_height)

def letterbox_to(tgt_w: int, tgt_h: int, img: Image.Image) -> Image.Image:
    """保持宽高比缩放，再黑边 pad 到 (tgt_h, tgt_w)"""
    orig_w, orig_h = img.size
    scale = min(tgt_w / orig_w, tgt_h / orig_h)   # 等比缩小/放大因子
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)

    # 计算左右 / 上下 pad
    pad_w = tgt_w - new_w
    pad_h = tgt_h - new_h
    left   = pad_w // 2
    right  = pad_w - left
    top    = pad_h // 2
    bottom = pad_h - top

    padded = TF.pad(resized, padding=[left, top, right, bottom], fill=0)
    return padded


class PreprocessMediaFile:
    def __init__(self, config, support_video=False, framerate=None, round_height=1, round_width=1, round_frames=1):
        self.config = config
        self.video_clip_mode = config.get('video_clip_mode', 'single_beginning')
        print(f'using video_clip_mode={self.video_clip_mode}')
        self.pil_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.support_video = support_video
        self.framerate = framerate
        print(f'using framerate={self.framerate}')
        self.round_height = round_height
        self.round_width = round_width
        self.round_frames = round_frames
        if self.support_video:
            assert self.framerate

    def __call__(self, filepath, mask_filepath, bbox_filepath,size_bucket=None):
        
        # ---------- 0. 打开图像 / 视频 ---------- #
        is_video = (Path(filepath).suffix in VIDEO_EXTENSIONS)
        if is_video:
            assert self.support_video
            video = [
                Image.fromarray(f)
                for f in imageio.v3.imiter(filepath, fps=self.framerate)
            ]
            num_frames = len(video)
            height, width = video[0].height, video[0].width
        else:
            pil_img = Image.open(filepath)
            height, width = pil_img.height, pil_img.width
            num_frames = 1
            video = [pil_img]

        # ---------- 1. 目标分辨率 ---------- #
        if size_bucket is not None:
            size_bucket_width, size_bucket_height, size_bucket_frames = size_bucket
        else:
            size_bucket_width, size_bucket_height, size_bucket_frames = width, height, num_frames

        height_rounded = round_to_nearest_multiple(size_bucket_height, self.round_height)
        width_rounded = round_to_nearest_multiple(size_bucket_width, self.round_width)
        frames_rounded = round_down_to_multiple(size_bucket_frames - 1, self.round_frames) + 1
        resize_wh = (width_rounded, height_rounded)

        # ---------- 2. mask ---------- #
        if mask_filepath:
            mask_img = Image.open(mask_filepath).convert('RGB')
            img_hw = (height, width)
            mask_hw = (mask_img.height, mask_img.width)
            if mask_hw != img_hw:
                raise ValueError(
                    f'Mask shape {mask_hw} was not the same as image shape {img_hw}.\n'
                    f'Image path: {filepath}\n'
                    f'Mask path: {mask_filepath}'
                )
            mask_img = ImageOps.fit(mask_img, resize_wh)
            mask = torchvision.transforms.functional.to_tensor(mask_img)[0].to(torch.float16)  # use first channel
        else:
            mask = None

        # ---------- 3. bbox ---------- #
        frame_boxes: dict[int, list[tuple[float, float, float, float]]] | None = None
        if bbox_filepath and Path(bbox_filepath).exists():
            with open(bbox_filepath, 'r') as f:
                raw = json.load(f)
            frame_boxes = {
                int(k): [
                    (face["box"]["x1"], face["box"]["y1"],
                     face["box"]["x2"], face["box"]["y2"])
                    for face in v.get("face", [])
                ]
                for k, v in raw.items()
            }

        # ---------- 4. 遍历帧，生成视频张量 & face 张量 ---------- #
        video_tensor = torch.empty((num_frames, 3, height_rounded, width_rounded))
        face_frames: list[torch.Tensor] = []
        
        for idx, frame in enumerate(video):
            if not isinstance(frame, Image.Image):
                frame = torchvision.transforms.functional.to_pil_image(frame)
                
            # a) always resize整帧
            video_tensor[idx] = self.pil_to_tensor(convert_crop_and_resize(frame, resize_wh))
            
            # b) 尝试裁脸
            boxes = frame_boxes.get(idx, []) if frame_boxes else []  
            if len(boxes) == 1:
                x1, y1, x2, y2 = boxes[0]
                crop = frame.crop((x1, y1, x2, y2))
            else:
                crop = frame 
            if idx == 0:
                crop.save(f'process_img/crop_{idx}_{bbox_filepath}.png')
                frame.save(f'process_img/frame_{idx}_{bbox_filepath}.png')
            face_frames.append(self.pil_to_tensor(letterbox_to(*resize_wh, crop)))
                 
        # ---------- 5. 打包返回 ---------- #
        # (F,C,H,W) → (C,F,H,W)
        video_tensor = video_tensor.permute(1, 0, 2, 3).contiguous()
        face_tensor = torch.stack(face_frames) if face_frames else None
        if face_tensor is not None:
            face_tensor = face_tensor.permute(1, 0, 2, 3).contiguous()

        if not self.support_video:                       # 纯图像模型
            return [(video_tensor.squeeze(0), mask, None)]
        if not is_video:                                  # 单张图像走这里
            return [(video_tensor, mask, face_tensor)]
               
        # 需要切 clip 的视频
        vids, face_vids = extract_clips(video_tensor, face_tensor, frames_rounded, self.video_clip_mode)
        if face_vids is not None:
            return ([(v, mask) for v in vids],
                    [(fv, mask) for fv in face_vids])
        else:
            return [(v, mask) for v in vids], None



class BasePipeline:
    framerate = None

    def load_diffusion_model(self):
        pass

    def get_vae(self):
        raise NotImplementedError()

    def get_text_encoders(self):
        raise NotImplementedError()

    def configure_adapter(self, adapter_config):
        target_linear_modules = set()
        for name, module in self.transformer.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules:
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, nn.Linear):
                    target_linear_modules.add(full_submodule_name)
        target_linear_modules = list(target_linear_modules)

        adapter_type = adapter_config['type']
        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=adapter_config['rank'],
                lora_alpha=adapter_config['alpha'],
                lora_dropout=adapter_config['dropout'],
                bias='none',
                target_modules=target_linear_modules
            )
        # else:
        elif adapter_type != 'video_id':
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        self.peft_config = peft_config
        self.lora_model = peft.get_peft_model(self.transformer, peft_config)
        if is_main_process():
            self.lora_model.print_trainable_parameters()
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config['dtype'])

    def save_adapter(self, save_dir, peft_state_dict):
        raise NotImplementedError()

    def load_adapter_weights(self, adapter_path):
        if is_main_process():
            print(f'Loading adapter weights from path {adapter_path}')
        safetensors_files = list(Path(adapter_path).glob('*.safetensors'))
        if len(safetensors_files) == 0:
            raise RuntimeError(f'No safetensors file found in {adapter_path}')
        if len(safetensors_files) > 1:
            raise RuntimeError(f'Multiple safetensors files found in {adapter_path}')
        adapter_state_dict = safetensors.torch.load_file(safetensors_files[0])
        modified_state_dict = {}
        model_parameters = set(name for name, p in self.transformer.named_parameters())
        for k, v in adapter_state_dict.items():
            # Replace Diffusers or ComfyUI prefix
            k = re.sub(r'^(transformer|diffusion_model)\.', '', k)
            # Replace weight at end for LoRA format
            k = re.sub(r'\.weight$', '.default.weight', k)
            if k not in model_parameters:
                raise RuntimeError(f'modified_state_dict key {k} is not in the model parameters')
            modified_state_dict[k] = v
        self.transformer.load_state_dict(modified_state_dict, strict=False)

    def load_and_fuse_adapter(self, path):
        peft_config = peft.LoraConfig.from_pretrained(path)
        lora_model = peft.get_peft_model(self.transformer, peft_config)
        self.load_adapter_weights(path)
        lora_model.merge_and_unload()

    def save_model(self, save_dir, diffusers_sd):
        raise NotImplementedError()

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(self.config, support_video=False)

    def get_call_vae_fn(self, vae):
        raise NotImplementedError()

    def get_call_text_encoder_fn(self, text_encoder):
        raise NotImplementedError()

    def prepare_inputs(self, inputs, timestep_quantile=None):
        raise NotImplementedError()

    def to_layers(self):
        raise NotImplementedError()

    def model_specific_dataset_config_validation(self, dataset_config):
        pass

    # Get param groups that will be passed into the optimizer. Models can override this, e.g. SDXL
    # supports separate learning rates for unet and text encoders.
    def get_param_groups(self, parameters):
        return parameters

    # Default loss_fn. MSE between output and target, with mask support.
    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                loss = F.mse_loss(output, target, reduction='none')
                # empty tensor means no masking
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss.mean()
            return loss
        return loss_fn

    def enable_block_swap(self, blocks_to_swap):
        raise NotImplementedError('Block swapping is not implemented for this model')

    def prepare_block_swap_training(self):
        pass

    def prepare_block_swap_inference(self, disable_block_swap=False):
        pass
