"""Modified from https://github.com/kijai/ComfyUI-EasyAnimateWrapper/blob/main/nodes.py
"""
import gc
import json
import os

import comfy.model_management as mm
import cv2
import folder_paths
import numpy as np
import torch
from comfy.utils import ProgressBar, load_torch_file
from diffusers import (AutoencoderKL, CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer

from ..cogvideox.data.bucket_sampler import ASPECT_RATIO_512, get_closest_ratio
from ..cogvideox.models.cogvideox_fun_vae import AutoencoderKLCogVideoX
from ..cogvideox.models.cogvideox_fun_transformer3d import CogVideoXTransformer3DModel
from ..cogvideox.pipeline.pipeline_cogvideox_fun import CogVideoX_Fun_Pipeline
from ..cogvideox.pipeline.pipeline_cogvideox_fun_control import \
    CogVideoX_Fun_Pipeline_Control
from ..cogvideox.pipeline.pipeline_cogvideox_fun_inpaint import (
    CogVideoX_Fun_Pipeline_Inpaint)
from ..cogvideox.utils.lora_utils import merge_lora, unmerge_lora
from ..cogvideox.utils.utils import (get_image_to_video_latent,
                                     get_video_to_video_latent,
                                     save_videos_grid)

# Compatible with Alibaba EAS for quick launch
eas_cache_dir       = '/stable-diffusion-cache/models'
# The directory of the cogvideoxfun
script_directory    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

def numpy2pil(image):
    return Image.fromarray(np.clip(255. * image, 0, 255).astype(np.uint8))

def to_pil(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        return tensor2pil(image)
    if isinstance(image, np.ndarray):
        return numpy2pil(image)
    raise ValueError(f"Cannot convert {type(image)} to PIL.Image")

class LoadCogVideoX_Fun_Model:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [ 
                        'CogVideoX-Fun-2b-InP',
                        'CogVideoX-Fun-5b-InP',
                        'CogVideoX-Fun-V1.1-2b-InP',
                        'CogVideoX-Fun-V1.1-5b-InP',
                        'CogVideoX-Fun-V1.1-2b-Pose',
                        'CogVideoX-Fun-V1.1-5b-Pose',
                        'CogVideoX-Fun-V1.1-5b-Control',
                        'CogVideoX-Fun-V1.5-5b-InP',
                    ],
                    {
                        "default": 'CogVideoX-Fun-V1.1-2b-InP',
                    }
                ),
                "model_type": (
                    ["Inpaint", "Control"],
                    {
                        "default": "Inpaint",
                    }
                ),
                "low_gpu_memory_mode":(
                    [False, True],
                    {
                        "default": False,
                    }
                ),
                "precision": (
                    ['fp16', 'bf16'],
                    {
                        "default": 'fp16'
                    }
                ),
                
            },
        }

    RETURN_TYPES = ("CogVideoXFUNSMODEL",)
    RETURN_NAMES = ("cogvideoxfun_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, low_gpu_memory_mode, model, model_type, precision):
        # Init weight_dtype and device
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()
        weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        # Init processbar
        pbar = ProgressBar(3)

        # Detect model is existing or not 
        model_path = os.path.join(folder_paths.models_dir, "CogVideoX_Fun", model)
      
        if not os.path.exists(model_path):
            if os.path.exists(eas_cache_dir):
                model_path = os.path.join(eas_cache_dir, 'CogVideoX_Fun', model)
            else:
                model_path = os.path.join(folder_paths.models_dir, "CogVideoX-Fun", model)
                if not os.path.exists(model_path):
                    if os.path.exists(eas_cache_dir):
                        model_path = os.path.join(eas_cache_dir, 'CogVideoX_Fun', model)
                    else:
                        # Detect model is existing or not 
                        print(f"Please download cogvideoxfun model to: {model_path}")

        vae = AutoencoderKLCogVideoX.from_pretrained(
            model_path, 
            subfolder="vae", 
        ).to(weight_dtype)
        # Update pbar
        pbar.update(1)

        # Load Sampler
        print("Load Sampler.")
        scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder='scheduler')
        # Update pbar
        pbar.update(1)
        
        # Get Transformer
        transformer = CogVideoXTransformer3DModel.from_pretrained_2d(
            model_path, 
            subfolder="transformer",
            low_cpu_mem_usage=True, 
        ).to(weight_dtype)
        # Update pbar
        pbar.update(1) 

        # Get pipeline
        if model_type == "Inpaint":
            if transformer.config.in_channels != vae.config.latent_channels:
                pipeline = CogVideoX_Fun_Pipeline_Inpaint.from_pretrained(
                    model_path,
                    vae=vae, 
                    transformer=transformer,
                    scheduler=scheduler,
                    torch_dtype=weight_dtype
                )
            else:
                pipeline = CogVideoX_Fun_Pipeline.from_pretrained(
                    model_path,
                    vae=vae, 
                    transformer=transformer,
                    scheduler=scheduler,
                    torch_dtype=weight_dtype
                )
        else:
            pipeline = CogVideoX_Fun_Pipeline_Control.from_pretrained(
                    model_path,
                    vae=vae, 
                    transformer=transformer,
                    scheduler=scheduler,
                    torch_dtype=weight_dtype
                )
        if low_gpu_memory_mode:
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline.enable_model_cpu_offload()

        cogvideoxfun_model = {
            'pipeline': pipeline, 
            'dtype': weight_dtype,
            'model_path': model_path,
            'model_type': model_type,
            'loras': [],
            'strength_model': [],
        }
        return (cogvideoxfun_model,)

class LoadCogVideoX_Fun_Lora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cogvideoxfun_model": ("CogVideoXFUNSMODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"), {"default": None,}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("CogVideoXFUNSMODEL",)
    RETURN_NAMES = ("cogvideoxfun_model",)
    FUNCTION = "load_lora"
    CATEGORY = "CogVideoXFUNWrapper"

    def load_lora(self, cogvideoxfun_model, lora_name, strength_model):
        if lora_name is not None:
            return (
                {
                    'pipeline': cogvideoxfun_model["pipeline"], 
                    'dtype': cogvideoxfun_model["dtype"],
                    'model_path': cogvideoxfun_model["model_path"],
                    'loras': cogvideoxfun_model.get("loras", []) + [folder_paths.get_full_path("loras", lora_name)],
                    'strength_model': cogvideoxfun_model.get("strength_model", []) + [strength_model],
                }, 
            )
        else:
            return (cogvideoxfun_model,)

class CogVideoX_FUN_TextBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "",}),
            }
        }
    
    RETURN_TYPES = ("STRING_PROMPT",)
    RETURN_NAMES =("prompt",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, prompt):
        return (prompt, )

class CogVideoX_Fun_I2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cogvideoxfun_model": (
                    "CogVideoXFUNSMODEL", 
                ),
                "prompt": (
                    "STRING_PROMPT",
                ),
                "negative_prompt": (
                    "STRING_PROMPT",
                ),
                "video_length": (
                    "INT", {"default": 49, "min": 5, "max": 85, "step": 4}
                ),
                "base_resolution": (
                    [ 
                        512,
                        768,
                        960,
                        1024,
                    ], {"default": 768}
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 50, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.01}
                ),
                "scheduler": (
                    [ 
                        "Euler",
                        "Euler A",
                        "DPM++",
                        "PNDM",
                        "DDIM",
                    ],
                    {
                        "default": 'DDIM'
                    }
                )
            },
            "optional":{
                "start_img": ("IMAGE",),
                "end_img": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, cogvideoxfun_model, prompt, negative_prompt, video_length, base_resolution, seed, steps, cfg, scheduler, start_img=None, end_img=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()

        start_img = [to_pil(_start_img) for _start_img in start_img] if start_img is not None else None
        end_img = [to_pil(_end_img) for _end_img in end_img] if end_img is not None else None
        # Count most suitable height and width
        aspect_ratio_sample_size    = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        original_width, original_height = start_img[0].size if type(start_img) is list else Image.open(start_img).size
        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height, width = [int(x / 16) * 16 for x in closest_size]
        
        # Get Pipeline
        pipeline = cogvideoxfun_model['pipeline']
        model_path = cogvideoxfun_model['model_path']

        # Load Sampler
        if scheduler == "DPM++":
            noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        elif scheduler == "Euler":
            noise_scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        elif scheduler == "Euler A":
            noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        elif scheduler == "PNDM":
            noise_scheduler = PNDMScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        elif scheduler == "DDIM":
            noise_scheduler = DDIMScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        pipeline.scheduler = noise_scheduler

        generator= torch.Generator(device).manual_seed(seed)

        with torch.no_grad():
            video_length = int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            input_video, input_video_mask, clip_image = get_image_to_video_latent(start_img, end_img, video_length=video_length, sample_size=(height, width))

            for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
                pipeline = merge_lora(pipeline, _lora_path, _lora_weight)

            sample = pipeline(
                prompt, 
                num_frames = video_length,
                negative_prompt = negative_prompt,
                height      = height,
                width       = width,
                generator   = generator,
                guidance_scale = cfg,
                num_inference_steps = steps,

                video        = input_video,
                mask_video   = input_video_mask,
                comfyui_progressbar = True,
            ).videos
            videos = rearrange(sample, "b c t h w -> (b t) h w c")

            for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
                pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight)
        return (videos,)   


class CogVideoX_Fun_T2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cogvideoxfun_model": (
                    "CogVideoXFUNSMODEL", 
                ),
                "prompt": (
                    "STRING_PROMPT", 
                ),
                "negative_prompt": (
                    "STRING_PROMPT", 
                ),
                "video_length": (
                    "INT", {"default": 49, "min": 5, "max": 85, "step": 4}
                ),
                "width": (
                    "INT", {"default": 1008, "min": 64, "max": 2048, "step": 16}
                ),
                "height": (
                    "INT", {"default": 576, "min": 64, "max": 2048, "step": 16}
                ),
                "is_image":(
                    [
                        False,
                        True
                    ], 
                    {
                        "default": False,
                    }
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 50, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.01}
                ),
                "scheduler": (
                    [ 
                        "Euler",
                        "Euler A",
                        "DPM++",
                        "PNDM",
                        "DDIM",
                    ],
                    {
                        "default": 'DDIM'
                    }
                ),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, cogvideoxfun_model, prompt, negative_prompt, video_length, width, height, is_image, seed, steps, cfg, scheduler):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()

        # Get Pipeline
        pipeline = cogvideoxfun_model['pipeline']
        model_path = cogvideoxfun_model['model_path']

        # Load Sampler
        if scheduler == "DPM++":
            noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        elif scheduler == "Euler":
            noise_scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        elif scheduler == "Euler A":
            noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        elif scheduler == "PNDM":
            noise_scheduler = PNDMScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        elif scheduler == "DDIM":
            noise_scheduler = DDIMScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        pipeline.scheduler = noise_scheduler

        generator= torch.Generator(device).manual_seed(seed)
        
        video_length = 1 if is_image else video_length
        with torch.no_grad():
            video_length = int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            input_video, input_video_mask, clip_image = get_image_to_video_latent(None, None, video_length=video_length, sample_size=(height, width))

            for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
                pipeline = merge_lora(pipeline, _lora_path, _lora_weight)

            sample = pipeline(
                prompt, 
                num_frames = video_length,
                negative_prompt = negative_prompt,
                height      = height,
                width       = width,
                generator   = generator,
                guidance_scale = cfg,
                num_inference_steps = steps,

                video        = input_video,
                mask_video   = input_video_mask,
                comfyui_progressbar = True,
            ).videos
            videos = rearrange(sample, "b c t h w -> (b t) h w c")

            for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
                pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight)
        return (videos,)   

class CogVideoX_Fun_V2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cogvideoxfun_model": (
                    "CogVideoXFUNSMODEL", 
                ),
                "prompt": (
                    "STRING_PROMPT",
                ),
                "negative_prompt": (
                    "STRING_PROMPT",
                ),
                "video_length": (
                    "INT", {"default": 49, "min": 5, "max": 85, "step": 4}
                ),
                "base_resolution": (
                    [ 
                        512,
                        768,
                        960,
                        1024,
                    ], {"default": 768}
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 50, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.01}
                ),
                "denoise_strength": (
                    "FLOAT", {"default": 0.70, "min": 0.05, "max": 1.00, "step": 0.01}
                ),
                "scheduler": (
                    [ 
                        "Euler",
                        "Euler A",
                        "DPM++",
                        "PNDM",
                        "DDIM",
                    ],
                    {
                        "default": 'DDIM'
                    }
                ),
            },
            "optional":{
                "validation_video": ("IMAGE",),
                "control_video": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, cogvideoxfun_model, prompt, negative_prompt, video_length, base_resolution, seed, steps, cfg, denoise_strength, scheduler, validation_video=None, control_video=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()
        
        # Get Pipeline
        pipeline = cogvideoxfun_model['pipeline']
        model_path = cogvideoxfun_model['model_path']
        model_type = cogvideoxfun_model['model_type']

        # Count most suitable height and width
        aspect_ratio_sample_size    = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        if model_type == "Inpaint":
            if type(validation_video) is str:
                original_width, original_height = Image.fromarray(cv2.VideoCapture(validation_video).read()[1]).size
            else:
                validation_video = np.array(validation_video.cpu().numpy() * 255, np.uint8)
                original_width, original_height = Image.fromarray(validation_video[0]).size
        else:
            if type(control_video) is str:
                original_width, original_height = Image.fromarray(cv2.VideoCapture(control_video).read()[1]).size
            else:
                control_video = np.array(control_video.cpu().numpy() * 255, np.uint8)
                original_width, original_height = Image.fromarray(control_video[0]).size
        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height, width = [int(x / 16) * 16 for x in closest_size]

        # Load Sampler
        if scheduler == "DPM++":
            noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        elif scheduler == "Euler":
            noise_scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        elif scheduler == "Euler A":
            noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        elif scheduler == "PNDM":
            noise_scheduler = PNDMScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        elif scheduler == "DDIM":
            noise_scheduler = DDIMScheduler.from_pretrained(model_path, subfolder= 'scheduler')
        pipeline.scheduler = noise_scheduler

        generator= torch.Generator(device).manual_seed(seed)
        
        with torch.no_grad():
            video_length = int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            if model_type == "Inpaint":
                input_video, input_video_mask, clip_image = get_video_to_video_latent(validation_video, video_length=video_length, sample_size=(height, width), fps=8)
            else:
                input_video, input_video_mask, clip_image = get_video_to_video_latent(control_video, video_length=video_length, sample_size=(height, width), fps=8)

            for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
                pipeline = merge_lora(pipeline, _lora_path, _lora_weight)
            
            if model_type == "Inpaint":
                sample = pipeline(
                    prompt, 
                    num_frames = video_length,
                    negative_prompt = negative_prompt,
                    height      = height,
                    width       = width,
                    generator   = generator,
                    guidance_scale = cfg,
                    num_inference_steps = steps,

                    video        = input_video,
                    mask_video   = input_video_mask,
                    strength = float(denoise_strength),
                    comfyui_progressbar = True,
                ).videos
            else:
                sample = pipeline(
                    prompt, 
                    num_frames = video_length,
                    negative_prompt = negative_prompt,
                    height      = height,
                    width       = width,
                    generator   = generator,
                    guidance_scale = cfg,
                    num_inference_steps = steps,

                    control_video = input_video,
                    comfyui_progressbar = True,
                ).videos
            videos = rearrange(sample, "b c t h w -> (b t) h w c")

            for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
                pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight)
        return (videos,)   

NODE_CLASS_MAPPINGS = {
    "CogVideoX_FUN_TextBox": CogVideoX_FUN_TextBox,
    "LoadCogVideoX_Fun_Model": LoadCogVideoX_Fun_Model,
    "LoadCogVideoX_Fun_Lora": LoadCogVideoX_Fun_Lora,
    "CogVideoX_Fun_I2VSampler": CogVideoX_Fun_I2VSampler,
    "CogVideoX_Fun_T2VSampler": CogVideoX_Fun_T2VSampler,
    "CogVideoX_Fun_V2VSampler": CogVideoX_Fun_V2VSampler,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CogVideoX_FUN_TextBox": "CogVideoX_FUN_TextBox",
    "LoadCogVideoX_Fun_Model": "Load CogVideoX-Fun Model",
    "LoadCogVideoX_Fun_Lora": "Load CogVideoX-Fun Lora",
    "CogVideoX_Fun_I2VSampler": "CogVideoX-Fun Sampler for Image to Video",
    "CogVideoX_Fun_T2VSampler": "CogVideoX-Fun Sampler for Text to Video",
    "CogVideoX_Fun_V2VSampler": "CogVideoX-Fun Sampler for Video to Video",
}