"""Modified from https://github.com/kijai/ComfyUI-EasyAnimateWrapper/blob/main/nodes.py
"""
import copy
import gc
import inspect
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
from ...videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                                  AutoTokenizer, CLIPModel,
                                  Wan2_2Transformer3DModel, WanT5EncoderModel)
from ...videox_fun.models.cache_utils import get_teacache_coefficients
from ...videox_fun.pipeline import (Wan2_2FunControlPipeline,
                                    Wan2_2FunInpaintPipeline,
                                    Wan2_2FunPipeline, Wan2_2I2VPipeline,
                                    Wan2_2Pipeline, Wan2_2TI2VPipeline)
from ...videox_fun.ui.controller import all_cheduler_dict
from ...videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8, convert_weight_dtype_wrapper, undo_convert_weight_dtype_wrapper,
    replace_parameters_by_name)
from ...videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from ...videox_fun.utils.utils import (filter_kwargs,
                                       get_image_to_video_latent,
                                       get_video_to_video_latent,
                                       save_videos_grid, get_autocast_dtype)
from ..wan2_1.nodes import get_wan_scheduler
from ..comfyui_utils import (eas_cache_dir, script_directory,
                             search_model_in_possible_folders, to_pil)

# Used in lora cache
transformer_cpu_cache       = {}
transformer_high_cpu_cache  = {}
# lora path before
lora_path_before            = ""
lora_high_path_before       = ""

class LoadWan2_2TransformerModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"default": "Wan2_1-T2V-1_3B_bf16.safetensors,"},
                ),
                "precision": (["fp16", "bf16"],
                    {"default": "bf16"}
                ),
            },
        }
    RETURN_TYPES = ("TransformerModel", "STRING")
    RETURN_NAMES = ("transformer", "model_name")
    FUNCTION    = "loadmodel"
    CATEGORY    = "CogVideoXFUNWrapper"

    def loadmodel(self, model_name, precision):
        # Init weight_dtype and device
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()
        weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        transformer = None

        model_path = folder_paths.get_full_path("diffusion_models", model_name)
        transformer_state_dict = load_torch_file(model_path, safe_load=True)
        
        eps             = 1e-6
        text_len        = 512
        freq_dim        = 256
        dim             = transformer_state_dict["patch_embedding.weight"].shape[0]
        hidden_size     = dim
        in_dim          = transformer_state_dict["patch_embedding.weight"].shape[1]
        in_channels     = in_dim
        ffn_dim         = transformer_state_dict["blocks.0.ffn.0.bias"].shape[0]

        add_ref_conv            = True if "ref_conv.weight" in transformer_state_dict else False
        in_dim_ref_conv         = transformer_state_dict["ref_conv.weight"].shape[1] if "ref_conv.weight" in transformer_state_dict else None
        add_control_adapter     = True if "control_adapter.conv.weight" in transformer_state_dict else False
        in_dim_control_adapter  = transformer_state_dict["control_adapter.conv.weight"].shape[1] if "control_adapter.conv.weight" in transformer_state_dict else None

        if dim == 5120:
            num_heads = 40
            num_layers = 40
            out_dim = 16
            downscale_factor_control_adapter = 8
            if in_dim == out_dim * 2 + 4:
                model_name_in_pipeline = "wan2.2-i2v-a14b"
            elif in_dim == out_dim:
                model_name_in_pipeline = "wan2.2-t2v-a14b"
            else:
                model_name_in_pipeline = "wan2.2-fun-a14b"
                
        elif dim == 3072:
            num_heads = 24
            num_layers = 30
            out_dim = 48
            downscale_factor_control_adapter = 16
            if in_dim == out_dim:
                model_name_in_pipeline = "wan2.2-ti2v-5b"
            else:
                model_name_in_pipeline = "wan2.2-fun-5b"
        else: 
            num_heads = 12
            num_layers = 30
            out_dim = 16
            downscale_factor_control_adapter = 8
            model_name_in_pipeline = "wan2.2-fun"
        
        if in_dim != out_dim:
            model_type = "i2v"
        else:
            model_type = "t2v"

        kwargs = dict(
            dim = dim,
            in_dim = in_dim,
            eps = eps,
            ffn_dim = ffn_dim,
            freq_dim = freq_dim,
            model_type = model_type,
            num_heads = num_heads,
            num_layers = num_layers,
            out_dim = out_dim,
            text_len = text_len,
            in_channels = in_channels,
            hidden_size = hidden_size,
            add_control_adapter = add_control_adapter,
            add_ref_conv = add_ref_conv,
            in_dim_control_adapter = in_dim_control_adapter // downscale_factor_control_adapter // downscale_factor_control_adapter if in_dim_control_adapter is not None else in_dim_control_adapter,
            in_dim_ref_conv = in_dim_ref_conv,
            downscale_factor_control_adapter = downscale_factor_control_adapter,
        )

        sig = inspect.signature(Wan2_2Transformer3DModel)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        transformer = Wan2_2Transformer3DModel(**accepted)
        transformer.load_state_dict(transformer_state_dict)
        transformer = transformer.eval().to(device=offload_device, dtype=weight_dtype)
        return (transformer, model_name_in_pipeline)

class CombineWan2_2Pipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transformer": ("TransformerModel",),
                "vae": ("VAEModel",),
                "text_encoder": ("TextEncoderModel",),
                "tokenizer": ("Tokenizer",),
                "model_name": ("STRING",),
                "GPU_memory_mode":(
                    ["model_full_load", "model_full_load_and_qfloat8","model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"],
                    {
                        "default": "model_cpu_offload",
                    }
                ),
                "model_type": (
                    ["Inpaint", "Control"],
                    {
                        "default": "Inpaint",
                    }
                ),
            },
            "optional":{
                "clip_encoder": ("ClipEncoderModel",),
                "transformer_2": ("TransformerModel",),
            },
        }

    RETURN_TYPES = ("FunModels",)
    RETURN_NAMES = ("funmodels",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, model_name, GPU_memory_mode, model_type, transformer, vae, text_encoder, tokenizer, clip_encoder=None, transformer_2=None):
        # Get pipeline
        weight_dtype    = transformer.dtype if transformer.dtype not in [torch.float32, torch.float8_e4m3fn, torch.float8_e5m2] else get_autocast_dtype()
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()

        # Get pipeline
        if model_type == "Inpaint":
            if "5b" in model_name:
                pipeline = Wan2_2TI2VPipeline(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    transformer=transformer,
                    transformer_2=transformer_2,
                    scheduler=None,
                )
            else:
                if transformer.config.in_channels != vae.config.latent_channels:
                    pipeline = Wan2_2FunInpaintPipeline(
                        vae=vae,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        transformer=transformer,
                        transformer_2=transformer_2,
                        scheduler=None,
                    )
                else:
                    pipeline = Wan2_2FunPipeline(
                        vae=vae,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        transformer=transformer,
                        transformer_2=transformer_2,
                        scheduler=None,
                    )
        else:
            pipeline = Wan2_2FunControlPipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                transformer_2=transformer_2,
                scheduler=None,
            )

        pipeline.remove_all_hooks()
        undo_convert_weight_dtype_wrapper(transformer)
        pipeline.to(device=offload_device)
        transformer = transformer.to(weight_dtype)

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
            'strength_model': []
        }
        return (funmodels,)

class LoadWan2_2Model:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        'Wan2.2-T2V-A14B',
                        'Wan2.2-I2V-A14B',
                        'Wan2.2-TI2V-5B',
                    ],
                    {
                        "default": 'Wan2.2-T2V-A14B',
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
                        "wan2.2/wan_civitai_t2v.yaml",
                        "wan2.2/wan_civitai_i2v.yaml",
                        "wan2.2/wan_civitai_5b.yaml",
                    ],
                    {
                        "default": "wan2.2/wan_civitai_t2v.yaml",
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
    RETURN_NAMES = ("funmodels",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, GPU_memory_mode, model, precision, config):
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

        # Get Vae
        Chosen_AutoencoderKL = {
            "AutoencoderKLWan": AutoencoderKLWan,
            "AutoencoderKLWan3_8": AutoencoderKLWan3_8
        }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
        vae = Chosen_AutoencoderKL.from_pretrained(
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
        transformer = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
            transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
                os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
                transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
                low_cpu_mem_usage=True,
                torch_dtype=weight_dtype,
            )
        else:
            transformer_2 = None
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

        # Get pipeline
        model_type = "Inpaint"
        if model_type == "Inpaint":
            if "wan_civitai_5b" in config_path:
                pipeline = Wan2_2TI2VPipeline(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    transformer=transformer,
                    transformer_2=transformer_2,
                    scheduler=scheduler,
                )
            else:
                if transformer.config.in_channels != vae.config.latent_channels:
                    pipeline = Wan2_2I2VPipeline(
                        transformer=transformer,
                        transformer_2=transformer_2,
                        vae=vae,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        scheduler=scheduler,
                    )
                else:
                    pipeline = Wan2_2Pipeline(
                        transformer=transformer,
                        transformer_2=transformer_2,
                        vae=vae,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        scheduler=scheduler,
                    )
        else:
            raise ValueError(f"Model type {model_type} not supported")

        pipeline.remove_all_hooks()
        undo_convert_weight_dtype_wrapper(transformer)

        if GPU_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(transformer, ["modulation",], device=device)
            transformer.freqs = transformer.freqs.to(device=device)
            if transformer_2 is not None:
                replace_parameters_by_name(transformer_2, ["modulation",], device=device)
                transformer_2.freqs = transformer_2.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            if transformer_2 is not None:
                convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
                convert_weight_dtype_wrapper(transformer_2, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_full_load_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            if transformer_2 is not None:
                convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
                convert_weight_dtype_wrapper(transformer_2, weight_dtype)
            pipeline.to(device=device)
        else:
            pipeline.to(device=device)

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

class LoadWan2_2Lora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "funmodels": ("FunModels",),
                "lora_name": (folder_paths.get_filename_list("loras"), {"default": None,}),
                "lora_high_name": (folder_paths.get_filename_list("loras"), {"default": None,}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_cache":([False, True],  {"default": False,}),
            }
        }
    RETURN_TYPES = ("FunModels",)
    RETURN_NAMES = ("funmodels",)
    FUNCTION = "load_lora"
    CATEGORY = "CogVideoXFUNWrapper"

    def load_lora(self, funmodels, lora_name, lora_high_name, strength_model, lora_cache):
        new_funmodels = dict(funmodels)
        if lora_name is not None:
            loras = list(new_funmodels.get("loras", [])) + [folder_paths.get_full_path("loras", lora_name)]
            loras_high = list(new_funmodels.get("loras_high", [])) + [folder_paths.get_full_path("loras", lora_high_name)]
            strength_models = list(new_funmodels.get("strength_model", [])) + [strength_model]
            new_funmodels['loras'] = loras
            new_funmodels['loras_high'] = loras_high
            new_funmodels['strength_model'] = strength_models
            new_funmodels['lora_cache'] = lora_cache
        return (new_funmodels,)

class Wan2_2T2VSampler:
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
                "boundary": (
                    "FLOAT", {"default": 0.875, "min": 0.00, "max": 1.00, "step": 0.001}
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
            "optional":{
                "riflex_k": ("RIFLEXT_ARGS",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, funmodels, prompt, negative_prompt, video_length, width, height, is_image, seed, steps, cfg, scheduler, shift, boundary, teacache_threshold, enable_teacache, num_skip_start_steps, teacache_offload, cfg_skip_ratio, riflex_k=0):
        global transformer_cpu_cache
        global transformer_high_cpu_cache
        global lora_path_before
        global lora_high_path_before
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
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)
        else:
            pipeline.transformer.disable_teacache()
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.disable_teacache()

        if cfg_skip_ratio is not None:
            print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
            pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, steps)
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

        generator= torch.Generator(device).manual_seed(seed)

        video_length = 1 if is_image else video_length
        with torch.no_grad():
            video_length = int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1

            if riflex_k > 0:
                latent_frames = (video_length - 1) // pipeline.vae.config.temporal_compression_ratio + 1
                pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
                if pipeline.transformer_2 is not None:
                    pipeline.transformer_2.enable_riflex(k = riflex_k, L_test = latent_frames)

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
                   
                    if pipeline.transformer_2 is not None:
                        # Save the original weights to cpu
                        if len(transformer_high_cpu_cache) == 0:
                            print('Save transformer high state_dict to cpu memory')
                            transformer_high_state_dict = pipeline.transformer_2.state_dict()
                            for key in transformer_high_state_dict:
                                transformer_high_cpu_cache[key] = transformer_high_state_dict[key].clone().cpu()

                        lora_high_path_now = str(funmodels.get("loras_high", []) + funmodels.get("strength_model", []))
                        if lora_high_path_now != lora_high_path_before:
                            print('Merge Lora High with Cache')
                            lora_high_path_before = copy.deepcopy(lora_high_path_now)
                            pipeline.transformer_2.load_state_dict(transformer_cpu_cache)
                            for _lora_path, _lora_weight in zip(funmodels.get("loras_high", []), funmodels.get("strength_model", [])):
                                pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")
            else:
                print('Merge Lora')
                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if len(transformer_cpu_cache) != 0:
                    pipeline.transformer.load_state_dict(transformer_cpu_cache)
                    transformer_cpu_cache = {}
                    lora_path_before = ""
                    gc.collect()
                
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)

                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if pipeline.transformer_2 is not None:
                    if len(transformer_high_cpu_cache) != 0:
                        pipeline.transformer_2.load_state_dict(transformer_high_cpu_cache)
                        transformer_high_cpu_cache = {}
                        lora_high_path_before = ""
                        gc.collect()

                    for _lora_path, _lora_weight in zip(funmodels.get("loras_high", []), funmodels.get("strength_model", [])):
                        pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")

            sample = pipeline(
                prompt, 
                num_frames = video_length,
                negative_prompt = negative_prompt,
                height      = height,
                width       = width,
                generator   = generator,
                guidance_scale = cfg,
                num_inference_steps = steps,
                boundary     = boundary,
                comfyui_progressbar = True,
            ).videos
            videos = rearrange(sample, "b c t h w -> (b t) h w c")

            if not funmodels.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)
                if pipeline.transformer_2 is not None:
                    for _lora_path, _lora_weight in zip(funmodels.get("loras_high", []), funmodels.get("strength_model", [])):
                        pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")
        return (videos,)   


class Wan2_2I2VSampler:
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
                "boundary": (
                    "FLOAT", {"default": 0.90, "min": 0.00, "max": 1.00, "step": 0.001}
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
            "optional":{
                "start_img": ("IMAGE",),
                "riflex_k": ("RIFLEXT_ARGS",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, funmodels, prompt, negative_prompt, video_length, base_resolution, seed, steps, cfg, scheduler, shift, boundary, teacache_threshold, enable_teacache, num_skip_start_steps, teacache_offload, cfg_skip_ratio, start_img=None, end_img=None, riflex_k=0):
        global transformer_cpu_cache
        global transformer_high_cpu_cache
        global lora_path_before
        global lora_high_path_before
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()
        
        # Get Pipeline
        pipeline = funmodels['pipeline']
        model_name = funmodels['model_name']
        weight_dtype = funmodels['dtype']

        start_img = [to_pil(_start_img) for _start_img in start_img] if start_img is not None else None
        end_img = [to_pil(_end_img) for _end_img in end_img] if end_img is not None else None
        # Count most suitable height and width
        spatial_compression_ratio = pipeline.vae.config.spatial_compression_ratio if hasattr(pipeline.vae.config, "spatial_compression_ratio") else 8
        aspect_ratio_sample_size = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        original_width, original_height = start_img[0].size if type(start_img) is list else Image.open(start_img).size
        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height, width = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in closest_size]

        # Load Sampler
        pipeline.scheduler = get_wan_scheduler(scheduler, shift)
        coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
        if coefficients is not None:
            print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
            pipeline.transformer.enable_teacache(
                coefficients, steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
            )
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)
        else:
            pipeline.transformer.disable_teacache()
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.disable_teacache()

        if cfg_skip_ratio is not None:
            print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
            pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, steps)
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

        generator= torch.Generator(device).manual_seed(seed)

        with torch.no_grad():
            video_length = int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            input_video, input_video_mask, clip_image = get_image_to_video_latent(start_img, end_img, video_length=video_length, sample_size=(height, width))

            if riflex_k > 0:
                latent_frames = (video_length - 1) // pipeline.vae.config.temporal_compression_ratio + 1
                pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
                if pipeline.transformer_2 is not None:
                    pipeline.transformer_2.enable_riflex(k = riflex_k, L_test = latent_frames)

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
                   
                    if pipeline.transformer_2 is not None:
                        # Save the original weights to cpu
                        if len(transformer_high_cpu_cache) == 0:
                            print('Save transformer high state_dict to cpu memory')
                            transformer_high_state_dict = pipeline.transformer_2.state_dict()
                            for key in transformer_high_state_dict:
                                transformer_high_cpu_cache[key] = transformer_high_state_dict[key].clone().cpu()

                        lora_high_path_now = str(funmodels.get("loras_high", []) + funmodels.get("strength_model", []))
                        if lora_high_path_now != lora_high_path_before:
                            print('Merge Lora High with Cache')
                            lora_high_path_before = copy.deepcopy(lora_high_path_now)
                            pipeline.transformer_2.load_state_dict(transformer_cpu_cache)
                            for _lora_path, _lora_weight in zip(funmodels.get("loras_high", []), funmodels.get("strength_model", [])):
                                pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")
            else:
                print('Merge Lora')
                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if len(transformer_cpu_cache) != 0:
                    pipeline.transformer.load_state_dict(transformer_cpu_cache)
                    transformer_cpu_cache = {}
                    lora_path_before = ""
                    gc.collect()
                
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)

                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if pipeline.transformer_2 is not None:
                    if len(transformer_high_cpu_cache) != 0:
                        pipeline.transformer_2.load_state_dict(transformer_high_cpu_cache)
                        transformer_high_cpu_cache = {}
                        lora_high_path_before = ""
                        gc.collect()

                    for _lora_path, _lora_weight in zip(funmodels.get("loras_high", []), funmodels.get("strength_model", [])):
                        pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")

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
                boundary     = boundary,
                comfyui_progressbar = True,
            ).videos
            videos = rearrange(sample, "b c t h w -> (b t) h w c")

            if not funmodels.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)
                if pipeline.transformer_2 is not None:
                    for _lora_path, _lora_weight in zip(funmodels.get("loras_high", []), funmodels.get("strength_model", [])):
                        pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")
        return (videos,)   