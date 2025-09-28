"""Modified from https://github.com/kijai/ComfyUI-EasyAnimateWrapper/blob/main/nodes.py
"""
import copy
import gc
import json
import os

import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

import comfy.model_management as mm
import folder_paths
from comfy.utils import ProgressBar, load_torch_file

from ...videox_fun.data.bucket_sampler import (ASPECT_RATIO_512,
                                               get_closest_ratio)
from ...videox_fun.data.dataset_image_video import process_pose_params
from ...videox_fun.models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                                  WanT5EncoderModel, WanTransformer3DModel)
from ...videox_fun.models.cache_utils import get_teacache_coefficients
from ...videox_fun.pipeline import (WanFunControlPipeline,
                                    WanFunInpaintPipeline, WanFunPipeline)
from ...videox_fun.ui.controller import all_cheduler_dict
from ...videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8, convert_weight_dtype_wrapper, undo_convert_weight_dtype_wrapper,
    replace_parameters_by_name)
from ...videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from ...videox_fun.utils.utils import (filter_kwargs, get_image_latent,
                                       get_image_to_video_latent,
                                       get_video_to_video_latent,
                                       save_videos_grid)
from ..comfyui_utils import (eas_cache_dir, script_directory,
                             search_model_in_possible_folders, to_pil)
from ..wan2_1.nodes import get_wan_scheduler

# Used in lora cache
transformer_cpu_cache   = {}
# lora path before
lora_path_before        = ""

class LoadWanFunModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [ 
                        'Wan2.1-Fun-1.3B-InP',
                        'Wan2.1-Fun-14B-InP',
                        'Wan2.1-Fun-1.3B-Control',
                        'Wan2.1-Fun-14B-Control',
                        'Wan2.1-Fun-V1.1-1.3B-InP',
                        'Wan2.1-Fun-V1.1-14B-InP',
                        'Wan2.1-Fun-V1.1-1.3B-Control',
                        'Wan2.1-Fun-V1.1-14B-Control',
                        'Wan2.1-Fun-V1.1-1.3B-Control-Camera',
                        'Wan2.1-Fun-V1.1-14B-Control-Camera',
                    ],
                    {
                        "default": 'Wan2.1-Fun-1.3B-InP',
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
                "config": (
                    [
                        "wan2.1/wan_civitai.yaml",
                    ],
                    {
                        "default": "wan2.1/wan_civitai.yaml",
                    }
                ),
                "precision": (
                    ['fp16', 'bf16'],
                    {
                        "default": 'bf16'
                    }
                ),
            },
        }

    RETURN_TYPES = ("FunModels",)
    RETURN_NAMES = ("funmodels",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, GPU_memory_mode, model_type, model, precision, config):
        # Init weight_dtype and device
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()
        weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        # Init processbar
        pbar = ProgressBar(5)

        # Load config
        config_path = f"{script_directory}/config/{config}"
        config = OmegaConf.load(config_path)

        # Detect model is existing or not
        possible_folders = ["CogVideoX_Fun", "Fun_Models", "VideoX_Fun", "Wan-AI"] + \
                [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models/Diffusion_Transformer")] # Possible folder names to check
        # Initialize model_name as None
        model_name = search_model_in_possible_folders(possible_folders, model)

        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        ).to(weight_dtype)
        # Update pbar
        pbar.update(1)

        # Load Sampler
        print("Load Sampler.")
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )
        # Update pbar
        pbar.update(1)
        
        # Get Transformer
        transformer = WanTransformer3DModel.from_pretrained(
            os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        # Update pbar
        pbar.update(1) 

        # Get tokenizer and text_encoder
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )
        pbar.update(1) 

        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        pbar.update(1) 

        if transformer.config.in_channels != vae.config.latent_channels:
            # Get Clip Image Encoder
            clip_image_encoder = CLIPModel.from_pretrained(
                os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
            ).to(weight_dtype)
            clip_image_encoder = clip_image_encoder.eval()

        # Get pipeline
        if model_type == "Inpaint":
            if transformer.config.in_channels != vae.config.latent_channels:
                pipeline = WanFunInpaintPipeline(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    transformer=transformer,
                    scheduler=scheduler,
                    clip_image_encoder=clip_image_encoder
                )
            else:
                pipeline = WanFunPipeline(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    transformer=transformer,
                    scheduler=scheduler,
                )
        else:
            pipeline = WanFunControlPipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
                clip_image_encoder=clip_image_encoder
            )

        pipeline.remove_all_hooks()
        undo_convert_weight_dtype_wrapper(transformer)

        if GPU_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(transformer, ["modulation",], device=device)
            transformer.freqs = transformer.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload()
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",])
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.enable_model_cpu_offload()
        elif GPU_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload()
        elif GPU_memory_mode == "model_full_load_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",])
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.to(device=device)
        else:
            pipeline.to(device)

        funmodels = {
            'pipeline': pipeline, 
            'dtype': weight_dtype,
            'model_name': model_name,
            'model_type': model_type,
            'loras': [],
            'strength_model': [],
            'config': config,
        }
        return (funmodels,)

class LoadWanFunLora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "funmodels": ("FunModels",),
                "lora_name": (folder_paths.get_filename_list("loras"), {"default": None,}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_cache":([False, True],  {"default": False,}),
            }
        }
    RETURN_TYPES = ("FunModels",)
    RETURN_NAMES = ("funmodels",)
    FUNCTION = "load_lora"
    CATEGORY = "CogVideoXFUNWrapper"

    def load_lora(self, funmodels, lora_name, strength_model, lora_cache):
        new_funmodels = dict(funmodels)  

        if lora_name is not None:
            lora_path = folder_paths.get_full_path("loras", lora_name)

            new_funmodels['lora_cache'] = lora_cache
            new_funmodels['loras'] = funmodels.get("loras", []) + [lora_path]
            new_funmodels['strength_model'] = funmodels.get("strength_model", []) + [strength_model]

        return (new_funmodels,)

class WanFunT2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "funmodels": (
                    "FunModels", 
                ),
                "prompt": (
                    "STRING_PROMPT", 
                ),
                "negative_prompt": (
                    "STRING_PROMPT", 
                ),
                "video_length": (
                    "INT", {"default": 81, "min": 5, "max": 161, "step": 4}
                ),
                "width": (
                    "INT", {"default": 832, "min": 64, "max": 2048, "step": 16}
                ),
                "height": (
                    "INT", {"default": 480, "min": 64, "max": 2048, "step": 16}
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
                    ["Flow", "Flow_Unipc", "Flow_DPM++"],
                    {
                        "default": 'Flow'
                    }
                ),
                "shift": (
                    "INT", {"default": 5, "min": 1, "max": 100, "step": 1}
                ),
                "teacache_threshold": (
                    "FLOAT", {"default": 0.10, "min": 0.00, "max": 1.00, "step": 0.005}
                ),
                "enable_teacache":(
                    [False, True],  {"default": True,}
                ),
                "num_skip_start_steps": (
                    "INT", {"default": 5, "min": 0, "max": 50, "step": 1}
                ),
                "teacache_offload":(
                    [False, True],  {"default": True,}
                ),
                "cfg_skip_ratio":(
                    "FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}
                ),
            },
            "optional": {
                "riflex_k": ("RIFLEXT_ARGS",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, funmodels, prompt, negative_prompt, video_length, width, height, is_image, seed, steps, cfg, scheduler, shift, teacache_threshold, enable_teacache, num_skip_start_steps, teacache_offload, cfg_skip_ratio, riflex_k=0):
        global transformer_cpu_cache
        global lora_path_before
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()

        # Get Pipeline
        pipeline = funmodels['pipeline']
        model_name = funmodels['model_name']
        weight_dtype = funmodels['dtype']

        # Load Sampler
        pipeline.scheduler = get_wan_scheduler(scheduler, shift)
        coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
        if coefficients is not None:
            print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
            pipeline.transformer.enable_teacache(
                coefficients, steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
            )
        else:
            pipeline.transformer.disable_teacache()

        if cfg_skip_ratio is not None:
            print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
            pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, steps)

        generator= torch.Generator(device).manual_seed(seed)
        
        video_length = 1 if is_image else video_length
        with torch.no_grad():
            video_length = int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1

            if riflex_k > 0:
                latent_frames = (video_length - 1) // pipeline.vae.config.temporal_compression_ratio + 1
                pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)

            # Apply lora
            if funmodels.get("lora_cache", False):
                if len(funmodels.get("loras", [])) != 0:
                    # Save the original weights to cpu
                    if len(transformer_cpu_cache) == 0:
                        print('Save transformer state_dict to cpu memory')
                        transformer_state_dict = pipeline.transformer.state_dict()
                        for key in transformer_state_dict:
                            transformer_cpu_cache[key] = transformer_state_dict[key].clone().cpu()
                    
                    lora_path_now = str(funmodels.get("loras", []) + funmodels.get("strength_model", []))
                    if lora_path_now != lora_path_before:
                        print('Merge Lora with Cache')
                        lora_path_before = copy.deepcopy(lora_path_now)
                        pipeline.transformer.load_state_dict(transformer_cpu_cache)
                        for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                            pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)
            else:
                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if len(transformer_cpu_cache) != 0:
                    pipeline.transformer.load_state_dict(transformer_cpu_cache)
                    transformer_cpu_cache = {}
                    lora_path_before = ""
                    gc.collect()
                print('Merge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)

            if pipeline.transformer.config.in_channels != pipeline.vae.config.latent_channels:
                input_video, input_video_mask, _ = get_image_to_video_latent(None, None, video_length=video_length, sample_size=(height, width))

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
                    comfyui_progressbar = True,
                ).videos
            videos = rearrange(sample, "b c t h w -> (b t) h w c")

            if not funmodels.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)
        return (videos,)   


class WanFunInpaintSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "funmodels": (
                    "FunModels", 
                ),
                "prompt": (
                    "STRING_PROMPT",
                ),
                "negative_prompt": (
                    "STRING_PROMPT",
                ),
                "video_length": (
                    "INT", {"default": 81, "min": 5, "max": 161, "step": 4}
                ),
                "base_resolution": (
                    [ 
                        512,
                        640,
                        768,
                        896,
                        960,
                        1024,
                    ], {"default": 640}
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
                    ["Flow", "Flow_Unipc", "Flow_DPM++"],
                    {
                        "default": 'Flow'
                    }
                ),
                "shift": (
                    "INT", {"default": 5, "min": 1, "max": 100, "step": 1}
                ),
                "teacache_threshold": (
                    "FLOAT", {"default": 0.10, "min": 0.00, "max": 1.00, "step": 0.005}
                ),
                "enable_teacache":(
                    [False, True],  {"default": True,}
                ),
                "num_skip_start_steps": (
                    "INT", {"default": 5, "min": 0, "max": 50, "step": 1}
                ),
                "teacache_offload":(
                    [False, True],  {"default": True,}
                ),
                "cfg_skip_ratio":(
                    "FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}
                ),
            },
            "optional": {
                "start_img": ("IMAGE",),
                "end_img": ("IMAGE",),
                "riflex_k": ("RIFLEXT_ARGS",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, funmodels, prompt, negative_prompt, video_length, base_resolution, seed, steps, cfg, scheduler, shift, teacache_threshold, enable_teacache, num_skip_start_steps, teacache_offload, cfg_skip_ratio, start_img=None, end_img=None, riflex_k=0):
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
        pipeline = funmodels['pipeline']
        model_name = funmodels['model_name']
        weight_dtype = funmodels['dtype']

        # Load Sampler
        pipeline.scheduler = get_wan_scheduler(scheduler, shift)
        coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
        if coefficients is not None:
            print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
            pipeline.transformer.enable_teacache(
                coefficients, steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
            )
        else:
            pipeline.transformer.disable_teacache()

        if cfg_skip_ratio is not None:
            print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
            pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, steps)

        generator= torch.Generator(device).manual_seed(seed)

        with torch.no_grad():
            video_length = int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1

            if riflex_k > 0:
                latent_frames = (video_length - 1) // pipeline.vae.config.temporal_compression_ratio + 1
                pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)

            input_video, input_video_mask, clip_image = get_image_to_video_latent(start_img, end_img, video_length=video_length, sample_size=(height, width))

            # Apply lora
            if funmodels.get("lora_cache", False):
                if len(funmodels.get("loras", [])) != 0:
                    # Save the original weights to cpu
                    if len(transformer_cpu_cache) == 0:
                        print('Save transformer state_dict to cpu memory')
                        transformer_state_dict = pipeline.transformer.state_dict()
                        for key in transformer_state_dict:
                            transformer_cpu_cache[key] = transformer_state_dict[key].clone().cpu()
                    
                    lora_path_now = str(funmodels.get("loras", []) + funmodels.get("strength_model", []))
                    if lora_path_now != lora_path_before:
                        print('Merge Lora with Cache')
                        lora_path_before = copy.deepcopy(lora_path_now)
                        pipeline.transformer.load_state_dict(transformer_cpu_cache)
                        for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                            pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)
            else:
                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if len(transformer_cpu_cache) != 0:
                    print('Delete cpu state_dict')
                    pipeline.transformer.load_state_dict(transformer_cpu_cache)
                    transformer_cpu_cache = {}
                    lora_path_before = ""
                    gc.collect()
                print('Merge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)

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
                clip_image   = clip_image,
                comfyui_progressbar = True,
            ).videos
            videos = rearrange(sample, "b c t h w -> (b t) h w c")

            if not funmodels.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)
        return (videos,)   


class WanFunV2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "funmodels": (
                    "FunModels", 
                ),
                "prompt": (
                    "STRING_PROMPT", 
                ),
                "negative_prompt": (
                    "STRING_PROMPT", 
                ),
                "video_length": (
                    "INT", {"default": 81, "min": 1, "max": 161, "step": 4}
                ),
                "base_resolution": (
                    [ 
                        512,
                        640,
                        768,
                        896,
                        960,
                        1024,
                    ], {"default": 640}
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
                    ["Flow", "Flow_Unipc", "Flow_DPM++"],
                    {
                        "default": 'Flow'
                    }
                ),
                "shift": (
                    "INT", {"default": 5, "min": 1, "max": 100, "step": 1}
                ),
                "teacache_threshold": (
                    "FLOAT", {"default": 0.10, "min": 0.00, "max": 1.00, "step": 0.005}
                ),
                "enable_teacache":(
                    [False, True],  {"default": True,}
                ),
                "num_skip_start_steps": (
                    "INT", {"default": 5, "min": 0, "max": 50, "step": 1}
                ),
                "teacache_offload":(
                    [False, True],  {"default": True,}
                ),
                "cfg_skip_ratio":(
                    "FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}
                ),
            },
            "optional": {
                "validation_video": ("IMAGE",),
                "control_video": ("IMAGE",),
                "start_image": ("IMAGE",),
                "ref_image": ("IMAGE",),
                "camera_conditions": ("STRING", {"forceInput": True}),
                "riflex_k": ("RIFLEXT_ARGS",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, funmodels, prompt, negative_prompt, video_length, base_resolution, seed, steps, cfg, denoise_strength, scheduler, shift, teacache_threshold, enable_teacache, num_skip_start_steps, teacache_offload, cfg_skip_ratio, validation_video=None, control_video=None, start_image=None, ref_image=None, camera_conditions=None, riflex_k=0):
        global transformer_cpu_cache
        global lora_path_before

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()
        
        # Get Pipeline
        pipeline = funmodels['pipeline']
        model_name = funmodels['model_name']
        weight_dtype = funmodels['dtype']
        model_type = funmodels['model_type']

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

            if ref_image is not None:
                ref_image = [to_pil(_ref_image) for _ref_image in ref_image]
                original_width, original_height = ref_image[0].size if type(ref_image) is list else Image.open(ref_image).size
            
            if start_image is not None:
                start_image = [to_pil(_start_image) for _start_image in start_image]
                original_width, original_height = start_image[0].size if type(start_image) is list else Image.open(start_image).size

        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height, width = [int(x / 16) * 16 for x in closest_size]

        # Load Sampler
        pipeline.scheduler = get_wan_scheduler(scheduler, shift)
        coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
        if coefficients is not None:
            print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
            pipeline.transformer.enable_teacache(
                coefficients, steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
            )
        else:
            pipeline.transformer.disable_teacache()

        if cfg_skip_ratio is not None:
            print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
            pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, steps)

        generator= torch.Generator(device).manual_seed(seed)

        with torch.no_grad():
            video_length = int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1

            if riflex_k > 0:
                latent_frames = (video_length - 1) // pipeline.vae.config.temporal_compression_ratio + 1
                pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)

            if model_type == "Inpaint":
                input_video, input_video_mask, ref_image, clip_image = get_video_to_video_latent(validation_video, video_length=video_length, sample_size=(height, width), fps=16, ref_image=ref_image[0] if ref_image is not None else ref_image)
            else:
                if ref_image is not None:
                    clip_image = ref_image[0].convert("RGB")
                elif start_image is not None:
                    clip_image = start_image[0].convert("RGB")
                else:
                    clip_image = None
        
                if ref_image is not None:
                    ref_image = get_image_latent(ref_image[0] if ref_image is not None else ref_image, sample_size=(height, width))
                
                if start_image is not None:
                    start_image = get_image_latent(start_image[0] if start_image is not None else start_image, sample_size=(height, width))

                if camera_conditions is not None and len(camera_conditions) > 0: 
                    poses      = json.loads(camera_conditions)
                    cam_params = np.array([[float(x) for x in pose] for pose in poses])
                    cam_params = np.concatenate([np.zeros_like(cam_params[:, :1]), cam_params], 1)
                    control_camera_video = process_pose_params(cam_params, width=width, height=height)
                    control_camera_video = control_camera_video[:video_length].permute([3, 0, 1, 2]).unsqueeze(0)
                    input_video, input_video_mask = None, None
                else:
                    control_camera_video = None
                    input_video, input_video_mask, _, _ = get_video_to_video_latent(control_video, video_length=video_length, sample_size=(height, width), fps=16, ref_image=None)

            # Apply lora
            if funmodels.get("lora_cache", False):
                if len(funmodels.get("loras", [])) != 0:
                    # Save the original weights to cpu
                    if len(transformer_cpu_cache) == 0:
                        print('Save transformer state_dict to cpu memory')
                        transformer_state_dict = pipeline.transformer.state_dict()
                        for key in transformer_state_dict:
                            transformer_cpu_cache[key] = transformer_state_dict[key].clone().cpu()
                    
                    lora_path_now = str(funmodels.get("loras", []) + funmodels.get("strength_model", []))
                    if lora_path_now != lora_path_before:
                        print('Merge Lora with Cache')
                        lora_path_before = copy.deepcopy(lora_path_now)
                        pipeline.transformer.load_state_dict(transformer_cpu_cache)
                        for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                            pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)
            else:
                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if len(transformer_cpu_cache) != 0:
                    pipeline.transformer.load_state_dict(transformer_cpu_cache)
                    transformer_cpu_cache = {}
                    lora_path_before = ""
                    gc.collect()
                print('Merge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)

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
                    clip_image   = clip_image, 
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

                    ref_image = ref_image,
                    start_image = start_image,
                    clip_image = clip_image,
                    control_video = input_video,
                    control_camera_video = control_camera_video,
                    comfyui_progressbar = True,
                ).videos
            videos = rearrange(sample, "b c t h w -> (b t) h w c")

            if not funmodels.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)
        return (videos,)   

