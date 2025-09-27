"""Modified from https://github.com/kijai/ComfyUI-EasyAnimateWrapper/blob/main/nodes.py
"""
import gc
import json
import os

import comfy.model_management as mm
import cv2
import folder_paths
import copy
import numpy as np
import torch
from comfy.utils import ProgressBar, load_torch_file
from diffusers import EulerDiscreteScheduler
from einops import rearrange
from PIL import Image

from ...videox_fun.data.bucket_sampler import (ASPECT_RATIO_512,
                                              get_closest_ratio)
from ...videox_fun.models import (AutoencoderKLCogVideoX,
                                 CogVideoXTransformer3DModel, T5EncoderModel,
                                 T5Tokenizer)
from ...videox_fun.pipeline import (CogVideoXFunPipeline,
                                   CogVideoXFunControlPipeline,
                                   CogVideoXFunInpaintPipeline)
from ...videox_fun.ui.controller import all_cheduler_dict
from ...videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from ...videox_fun.utils.utils import (get_image_to_video_latent,
                                      get_video_to_video_latent,
                                      save_videos_grid)
from ...videox_fun.utils.fp8_optimization import convert_weight_dtype_wrapper, undo_convert_weight_dtype_wrapper
from ..comfyui_utils import (eas_cache_dir, script_directory,
                             search_model_in_possible_folders, to_pil)

# Used in lora cache
transformer_cpu_cache   = {}
# lora path before
lora_path_before        = ""

class LoadCogVideoXFunModel:
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
                        "CogVideoX-Fun-V1.1-2b-Control",
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
                "GPU_memory_mode":(
                    ["model_full_load", "model_full_load_and_qfloat8","model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"],
                    {
                        "default": "model_cpu_offload",
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

    RETURN_TYPES = ("FunModels",)
    RETURN_NAMES = ("cogvideoxfun_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, GPU_memory_mode, model, model_type, precision):
        # Init weight_dtype and device
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()
        weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        # Init processbar
        pbar = ProgressBar(5)

        # Detect model is existing or not
        possible_folders = ["CogVideoX_Fun", "Fun_Models", "VideoX_Fun", "Wan-AI"] + \
                [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models/Diffusion_Transformer")] # Possible folder names to check
        # Initialize model_name as None
        model_name = search_model_in_possible_folders(possible_folders, model)

        vae = AutoencoderKLCogVideoX.from_pretrained(
            model_name, 
            subfolder="vae", 
        ).to(weight_dtype)
        # Update pbar
        pbar.update(1)

        # Load Sampler
        print("Load Sampler.")
        scheduler = EulerDiscreteScheduler.from_pretrained(model_name, subfolder='scheduler')
        # Update pbar
        pbar.update(1)
        
        # Get Transformer
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_name, 
            subfolder="transformer",
            torch_dtype=torch.float8_e4m3fn if GPU_memory_mode == "model_cpu_offload_and_qfloat8" else weight_dtype,
        ).to(weight_dtype)
        # Update pbar
        pbar.update(1) 

        # Get tokenizer and text_encoder
        tokenizer = T5Tokenizer.from_pretrained(
            model_name, subfolder="tokenizer"
        )
        pbar.update(1) 

        text_encoder = T5EncoderModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=weight_dtype
        )
        pbar.update(1) 

        # Get pipeline
        if model_type == "Inpaint":
            if transformer.config.in_channels != vae.config.latent_channels:
                pipeline = CogVideoXFunInpaintPipeline(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    transformer=transformer,
                    scheduler=scheduler,
                )
            else:
                pipeline = CogVideoXFunPipeline(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    transformer=transformer,
                    scheduler=scheduler,
                )
        else:
            pipeline = CogVideoXFunControlPipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
            )

        pipeline.remove_all_hooks()
        undo_convert_weight_dtype_wrapper(transformer)

        if GPU_memory_mode == "sequential_cpu_offload":
            pipeline.enable_sequential_cpu_offload()
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.enable_model_cpu_offload()
        elif GPU_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload()
        elif GPU_memory_mode == "model_full_load_and_qfloat8":
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.to(device=device)
        else:
            pipeline.to("cuda")

        cogvideoxfun_model = {
            'pipeline': pipeline, 
            'dtype': weight_dtype,
            'model_name': model_name,
            'model_type': model_type,
            'loras': [],
            'strength_model': [],
        }
        return (cogvideoxfun_model,)

class LoadCogVideoXFunLora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cogvideoxfun_model": ("FunModels",),
                "lora_name": (folder_paths.get_filename_list("loras"), {"default": None,}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_cache":([False, True],  {"default": False,}),
            }
        }
    RETURN_TYPES = ("FunModels",)
    RETURN_NAMES = ("cogvideoxfun_model",)
    FUNCTION = "load_lora"
    CATEGORY = "CogVideoXFUNWrapper"

    def load_lora(self, cogvideoxfun_model, lora_name, strength_model, lora_cache):
        new_funmodels = dict(cogvideoxfun_model)  

        if lora_name is not None:
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                raise FileNotFoundError(f"LoRA 文件未找到: {lora_name}")

            new_funmodels['lora_cache'] = lora_cache
            new_funmodels['loras'] = cogvideoxfun_model.get("loras", []) + [lora_path]
            new_funmodels['strength_model'] = cogvideoxfun_model.get("strength_model", []) + [strength_model]

        return (new_funmodels,)

class CogVideoXFunT2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cogvideoxfun_model": (
                    "FunModels", 
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
        global transformer_cpu_cache
        global lora_path_before
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()

        # Get Pipeline
        pipeline = cogvideoxfun_model['pipeline']
        model_name = cogvideoxfun_model['model_name']
        weight_dtype = cogvideoxfun_model['dtype']

        # Load Sampler
        pipeline.scheduler = all_cheduler_dict[scheduler].from_pretrained(model_name, subfolder='scheduler')

        generator= torch.Generator(device).manual_seed(seed)
        
        video_length = 1 if is_image else video_length
        with torch.no_grad():
            video_length = int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            input_video, input_video_mask, clip_image = get_image_to_video_latent(None, None, video_length=video_length, sample_size=(height, width))

            # Apply lora
            if cogvideoxfun_model.get("lora_cache", False):
                if len(cogvideoxfun_model.get("loras", [])) != 0:
                    # Save the original weights to cpu
                    if len(transformer_cpu_cache) == 0:
                        print('Save transformer state_dict to cpu memory')
                        transformer_state_dict = pipeline.transformer.state_dict()
                        for key in transformer_state_dict:
                            transformer_cpu_cache[key] = transformer_state_dict[key].clone().cpu()
                    
                    lora_path_now = str(cogvideoxfun_model.get("loras", []) + cogvideoxfun_model.get("strength_model", []))
                    if lora_path_now != lora_path_before:
                        print('Merge Lora with Cache')
                        lora_path_before = copy.deepcopy(lora_path_now)
                        pipeline.transformer.load_state_dict(transformer_cpu_cache)
                        for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
                            pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
            else:
                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if len(transformer_cpu_cache) != 0:
                    pipeline.transformer.load_state_dict(transformer_cpu_cache)
                    transformer_cpu_cache = {}
                    lora_path_before = ""
                    gc.collect()
                print('Merge Lora')
                for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
                    pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)

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

            if not cogvideoxfun_model.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
        return (videos,)   

class CogVideoXFunInpaintSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cogvideoxfun_model": (
                    "FunModels", 
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
        global transformer_cpu_cache
        global lora_path_before
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()

        start_img = [to_pil(_start_img) for _start_img in start_img] if start_img is not None else None
        end_img = [to_pil(_end_img) for _end_img in end_img] if end_img is not None else None
        # Count most suitable height and width
        aspect_ratio_sample_size = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        original_width, original_height = start_img[0].size if type(start_img) is list else Image.open(start_img).size
        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height, width = [int(x / 16) * 16 for x in closest_size]
        
        # Get Pipeline
        pipeline = cogvideoxfun_model['pipeline']
        model_name = cogvideoxfun_model['model_name']
        weight_dtype = cogvideoxfun_model['dtype']
        
        # Load Sampler
        pipeline.scheduler = all_cheduler_dict[scheduler].from_pretrained(model_name, subfolder='scheduler')

        generator= torch.Generator(device).manual_seed(seed)

        with torch.no_grad():
            video_length = int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            input_video, input_video_mask, clip_image = get_image_to_video_latent(start_img, end_img, video_length=video_length, sample_size=(height, width))

            # Apply lora
            if cogvideoxfun_model.get("lora_cache", False):
                if len(cogvideoxfun_model.get("loras", [])) != 0:
                    # Save the original weights to cpu
                    if len(transformer_cpu_cache) == 0:
                        print('Save transformer state_dict to cpu memory')
                        transformer_state_dict = pipeline.transformer.state_dict()
                        for key in transformer_state_dict:
                            transformer_cpu_cache[key] = transformer_state_dict[key].clone().cpu()
                    
                    lora_path_now = str(cogvideoxfun_model.get("loras", []) + cogvideoxfun_model.get("strength_model", []))
                    if lora_path_now != lora_path_before:
                        print('Merge Lora with Cache')
                        lora_path_before = copy.deepcopy(lora_path_now)
                        pipeline.transformer.load_state_dict(transformer_cpu_cache)
                        for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
                            pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
            else:
                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if len(transformer_cpu_cache) != 0:
                    print('Delete cpu state_dict')
                    pipeline.transformer.load_state_dict(transformer_cpu_cache)
                    transformer_cpu_cache = {}
                    lora_path_before = ""
                    gc.collect()
                print('Merge Lora')
                for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
                    pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)

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

            if not cogvideoxfun_model.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
        return (videos,)   


class CogVideoXFunV2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cogvideoxfun_model": (
                    "FunModels", 
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
                    "FLOAT", {"default": 1.00, "min": 0.05, "max": 1.00, "step": 0.01}
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
        global transformer_cpu_cache
        global lora_path_before

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()
        
        # Get Pipeline
        pipeline = cogvideoxfun_model['pipeline']
        model_name = cogvideoxfun_model['model_name']
        weight_dtype = cogvideoxfun_model['dtype']
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
            if control_video is not None and type(control_video) is str:
                original_width, original_height = Image.fromarray(cv2.VideoCapture(control_video).read()[1]).size
            elif control_video is not None:
                control_video = np.array(control_video.cpu().numpy() * 255, np.uint8)
                original_width, original_height = Image.fromarray(control_video[0]).size
            else:
                original_width, original_height = 384 / 512 * base_resolution, 672 / 512 * base_resolution

        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height, width = [int(x / 16) * 16 for x in closest_size]

        # Load Sampler
        pipeline.scheduler = all_cheduler_dict[scheduler].from_pretrained(model_name, subfolder='scheduler')

        generator= torch.Generator(device).manual_seed(seed)
        
        with torch.no_grad():
            video_length = int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            if model_type == "Inpaint":
                input_video, input_video_mask, ref_image, clip_image = get_video_to_video_latent(validation_video, video_length=video_length, sample_size=(height, width), fps=8)
            else:
                input_video, input_video_mask, ref_image, clip_image = get_video_to_video_latent(control_video, video_length=video_length, sample_size=(height, width), fps=8)

            # Apply lora
            if cogvideoxfun_model.get("lora_cache", False):
                if len(cogvideoxfun_model.get("loras", [])) != 0:
                    # Save the original weights to cpu
                    if len(transformer_cpu_cache) == 0:
                        print('Save transformer state_dict to cpu memory')
                        transformer_state_dict = pipeline.transformer.state_dict()
                        for key in transformer_state_dict:
                            transformer_cpu_cache[key] = transformer_state_dict[key].clone().cpu()
                    
                    lora_path_now = str(cogvideoxfun_model.get("loras", []) + cogvideoxfun_model.get("strength_model", []))
                    if lora_path_now != lora_path_before:
                        print('Merge Lora with Cache')
                        lora_path_before = copy.deepcopy(lora_path_now)
                        pipeline.transformer.load_state_dict(transformer_cpu_cache)
                        for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
                            pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
            else:
                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if len(transformer_cpu_cache) != 0:
                    pipeline.transformer.load_state_dict(transformer_cpu_cache)
                    transformer_cpu_cache = {}
                    lora_path_before = ""
                    gc.collect()
                print('Merge Lora')
                for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
                    pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
            
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

            if not cogvideoxfun_model.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(cogvideoxfun_model.get("loras", []), cogvideoxfun_model.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
        return (videos,)   
