"""Modified from https://github.com/kijai/ComfyUI-EasyAnimateWrapper/blob/main/nodes.py
"""
import copy
import gc
import inspect
import json
import os

import comfy.model_management as mm
import cv2
import folder_paths
import numpy as np
import torch
from comfy.utils import ProgressBar, load_torch_file
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

from ...videox_fun.data.bucket_sampler import (ASPECT_RATIO_512,
                                               get_closest_ratio)
from ...videox_fun.models import (AutoencoderKLQwenImage, Qwen2_5_VLConfig,
                                  Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor,
                                  Qwen2Tokenizer, QwenImageTransformer2DModel)
from ...videox_fun.models.cache_utils import get_teacache_coefficients
from ...videox_fun.pipeline import QwenImagePipeline, QwenImageEditPipeline
from ...videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from ...videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ...videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8, convert_weight_dtype_wrapper, undo_convert_weight_dtype_wrapper,
    replace_parameters_by_name)
from ...videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from ...videox_fun.utils.utils import filter_kwargs, get_image, get_autocast_dtype
from ..comfyui_utils import (eas_cache_dir, script_directory, to_pil,
                             search_model_in_possible_folders,
                             search_sub_dir_in_possible_folders)

# Used in lora cache
transformer_cpu_cache       = {}
# lora path before
lora_path_before            = ""

def get_qwen_scheduler(sampler_name, shift):
    Chosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    scheduler_kwargs = {
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": 0.9,
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": 0.02,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False
    }
    scheduler_kwargs['shift'] = shift
    scheduler = Chosen_Scheduler(
        **filter_kwargs(Chosen_Scheduler, scheduler_kwargs)
    )
    return scheduler

class LoadQwenImageTransformerModel:
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
        
        model_name_in_pipeline = "Qwen-Image"
        kwargs = {
            "attention_head_dim": 128,
            "axes_dims_rope": [
                16,
                56,
                56
            ],
            "guidance_embeds": False,
            "in_channels": 64,
            "joint_attention_dim": 3584,
            "num_attention_heads": 24,
            "num_layers": 60,
            "out_channels": 16,
            "patch_size": 2,
            "pooled_projection_dim": 768
        }

        sig = inspect.signature(QwenImageTransformer2DModel)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        transformer = QwenImageTransformer2DModel(**accepted)
        transformer.load_state_dict(transformer_state_dict)
        transformer = transformer.eval().to(device=offload_device, dtype=weight_dtype)
        return (transformer, model_name_in_pipeline)

class LoadQwenImageVAEModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("vae"),
                    {"default": "QwenImage2.1_VAE.pth"}
                ),
                "precision": (["fp16", "bf16"],
                    {"default": "bf16"}
                ),
            },
        }

    RETURN_TYPES = ("VAEModel",)
    RETURN_NAMES = ("vae", )
    FUNCTION    = "loadmodel"
    CATEGORY    = "CogVideoXFUNWrapper"

    def loadmodel(self, model_name, precision,):
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()
        
        weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
        model_path = folder_paths.get_full_path("vae", model_name)
        vae_state_dict = load_torch_file(model_path, safe_load=True)

        if "conv1.weight" in vae_state_dict:
            use_wan_compiled_vae = True
            if not any(k.startswith("model.") for k in vae_state_dict.keys()):
                vae_state_dict = {f"model.{k}": v for k, v in vae_state_dict.items()}
        else:
            use_wan_compiled_vae = False

        kwargs = {
            "attn_scales": [],
            "base_dim": 96,
            "dim_mult": [
                1,
                2,
                4,
                4
            ],
            "dropout": 0.0,
            "latents_mean": [
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921
            ],
            "latents_std": [
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.916
            ],
            "num_res_blocks": 2,
            "temperal_downsample": [
                False,
                True,
                True
            ],
            "z_dim": 16
        }

        sig = inspect.signature(AutoencoderKLQwenImage)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}

        if use_wan_compiled_vae:
            from ...videox_fun.models.wan_vae import AutoencoderKLWanCompileQwenImage
            vae = AutoencoderKLWanCompileQwenImage(**accepted)
        else:
            vae = AutoencoderKLQwenImage(**accepted)
        vae.load_state_dict(vae_state_dict)
        vae = vae.eval().to(device=offload_device, dtype=weight_dtype)
        return (vae,)

class LoadQwenImageTextEncoderModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("text_encoders"),
                    {"default": "models_t5_umt5-xxl-enc-bf16.pth"}
                ),
                "precision": (["fp16", "bf16"],
                    {"default": "bf16"}
                ),
            },
        }

    RETURN_TYPES = ("TextEncoderModel", "Tokenizer")
    RETURN_NAMES = ("text_encoder", "tokenizer")
    FUNCTION    = "loadmodel"
    CATEGORY    = "CogVideoXFUNWrapper"

    def loadmodel(self, model_name, precision,):
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()
        
        weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
        model_path = folder_paths.get_full_path("text_encoders", model_name)
        text_state_dict = load_torch_file(model_path, safe_load=True)

        if not any(k.startswith("model.") for k in text_state_dict.keys()):
            text_state_dict = {f"model.{k}": v for k, v in text_state_dict.items()}

        kwargs = {
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "hidden_act": "silu",
            "hidden_size": 3584,
            "image_token_id": 151655,
            "initializer_range": 0.02,
            "intermediate_size": 18944,
            "max_position_embeddings": 128000,
            "max_window_layers": 28,
            "model_type": "qwen2_5_vl",
            "num_attention_heads": 28,
            "num_hidden_layers": 28,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {
                "mrope_section": [
                16,
                24,
                24
                ],
                "rope_type": "default",
                "type": "default"
            },
            "rope_theta": 1000000.0,
            "sliding_window": 32768,
            "text_config": {
                "architectures": [
                "Qwen2_5_VLForConditionalGeneration"
                ],
                "attention_dropout": 0.0,
                "bos_token_id": 151643,
                "eos_token_id": 151645,
                "hidden_act": "silu",
                "hidden_size": 3584,
                "image_token_id": None,
                "initializer_range": 0.02,
                "intermediate_size": 18944,
                "layer_types": [
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention"
                ],
                "max_position_embeddings": 128000,
                "max_window_layers": 28,
                "model_type": "qwen2_5_vl_text",
                "num_attention_heads": 28,
                "num_hidden_layers": 28,
                "num_key_value_heads": 4,
                "rms_norm_eps": 1e-06,
                "rope_scaling": {
                "mrope_section": [
                    16,
                    24,
                    24
                ],
                "rope_type": "default",
                "type": "default"
                },
                "rope_theta": 1000000.0,
                "sliding_window": None,
                "torch_dtype": "float32",
                "use_cache": True,
                "use_sliding_window": False,
                "video_token_id": None,
                "vision_end_token_id": 151653,
                "vision_start_token_id": 151652,
                "vision_token_id": 151654,
                "vocab_size": 152064
            },
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.53.1",
            "use_cache": True,
            "use_sliding_window": False,
            "video_token_id": 151656,
            "vision_config": {
                "depth": 32,
                "fullatt_block_indexes": [
                7,
                15,
                23,
                31
                ],
                "hidden_act": "silu",
                "hidden_size": 1280,
                "in_channels": 3,
                "in_chans": 3,
                "initializer_range": 0.02,
                "intermediate_size": 3420,
                "model_type": "qwen2_5_vl",
                "num_heads": 16,
                "out_hidden_size": 3584,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "spatial_patch_size": 14,
                "temporal_patch_size": 2,
                "tokens_per_second": 2,
                "torch_dtype": "float32",
                "window_size": 112
            },
            "vision_end_token_id": 151653,
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "vocab_size": 152064
        }
        config = Qwen2_5_VLConfig(**kwargs)
        text_encoder = Qwen2_5_VLForConditionalGeneration._from_config(config)
        def transform_key(key):
            key = key.replace("model.", "model.language_model.")
            key = key.replace("visual.", "model.visual.")
            return key
        text_state_dict = {transform_key(k): v for k, v in text_state_dict.items()}

        text_encoder.load_state_dict(text_state_dict)
        text_encoder = text_encoder.eval().to(device=offload_device, dtype=weight_dtype)

        possible_folders = ["CogVideoX_Fun", "Fun_Models", "VideoX_Fun", "Wan-AI", "Qwen"] + \
            [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models/Diffusion_Transformer")] # Possible folder names to check
        try:
            tokenizer_path = search_sub_dir_in_possible_folders(possible_folders, sub_dir_name="qwen2_tokenizer")
        except:
            try:
                tokenizer_path = os.path.join(search_sub_dir_in_possible_folders(possible_folders, sub_dir_name="Qwen-Image"), "tokenizer")
            except:
                tokenizer_path = search_sub_dir_in_possible_folders(possible_folders, sub_dir_name="Qwen2.5-VL-7B-Instruct")

        tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
        return (text_encoder, tokenizer)


class LoadQwenImageProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
        }

    RETURN_TYPES = ("Processor",)
    RETURN_NAMES = ("processor",)
    FUNCTION    = "loadmodel"
    CATEGORY    = "CogVideoXFUNWrapper"

    def loadmodel(self, ):
        possible_folders = ["CogVideoX_Fun", "Fun_Models", "VideoX_Fun", "Wan-AI", "Qwen"] + \
            [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models/Diffusion_Transformer")] # Possible folder names to check
        try:
            processor_path = search_sub_dir_in_possible_folders(possible_folders, sub_dir_name="qwen2_processor")
        except:
            try:
                processor_path = os.path.join(search_sub_dir_in_possible_folders(possible_folders, sub_dir_name="Qwen-Image-Edit"), "processor")
            except:
                processor_path = search_sub_dir_in_possible_folders(possible_folders, sub_dir_name="Qwen2.5-VL-7B-Instruct")

        # Get processor
        processor = Qwen2VLProcessor.from_pretrained(
            processor_path,
        )
        return (processor, )


class CombineQwenImagePipeline:
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
            },
            "optional":{
                "processor": ("Processor",),
            },
        }

    RETURN_TYPES = ("FunModels",)
    RETURN_NAMES = ("funmodels",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, model_name, GPU_memory_mode, transformer, vae, text_encoder, tokenizer, processor=None, transformer_2=None):
        # Get pipeline
        weight_dtype    = transformer.dtype if transformer.dtype not in [torch.float32, torch.float8_e4m3fn, torch.float8_e5m2] else get_autocast_dtype()
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()

        # Get pipeline
        model_type = "Inpaint"
        if model_type == "Inpaint":
            if processor is not None:
                pipeline = QwenImageEditPipeline(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    transformer=transformer,
                    scheduler=None,
                    processor=processor,
                )
            else:
                pipeline = QwenImagePipeline(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    transformer=transformer,
                    scheduler=None,
                )
        else:
            raise ValueError("Not supported now.")

        pipeline.remove_all_hooks()
        undo_convert_weight_dtype_wrapper(transformer)
        pipeline.to(device=offload_device)
        transformer = transformer.to(weight_dtype)

        if GPU_memory_mode == "sequential_cpu_offload":
            pipeline.enable_sequential_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["img_in", "txt_in", "timestep"], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_full_load_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["img_in", "txt_in", "timestep"], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.to(device=device)
        else:
            pipeline.to(device=device)

        funmodels = {
            'pipeline': pipeline, 
            'dtype': weight_dtype,
            'model_name': model_name,
            'model_type': model_type,
            'loras': [],
            'strength_model': []
        }
        return (funmodels,)

class LoadQwenImageModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        'Qwen-Image',
                        'Qwen-Image-Edit',
                    ],
                    {
                        "default": 'Qwen-Image',
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
    RETURN_NAMES = ("funmodels",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, GPU_memory_mode, model, precision):
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

        # Get Vae
        vae = AutoencoderKLQwenImage.from_pretrained(
            model_name, 
            subfolder="vae"
        ).to(weight_dtype)
        # Update pbar
        pbar.update(1)

        # Load Sampler
        print("Load Sampler.")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name, 
            subfolder="scheduler"
        )
        # Update pbar
        pbar.update(1)
        
        # Get Transformer
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_name, 
            subfolder="transformer",
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        # Update pbar
        pbar.update(1) 

        # Get tokenizer and text_encoder
        tokenizer = Qwen2Tokenizer.from_pretrained(
            model_name, subfolder="tokenizer"
        )
        pbar.update(1) 

        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=weight_dtype
        )
        pbar.update(1) 

        if os.path.exists(os.path.join(model_name, "processor")):
            need_processor = True
            # Get processor
            processor = Qwen2VLProcessor.from_pretrained(
                model_name,
                subfolder="processor"
            )
        else:
            need_processor = False

        model_type = "Inpaint"
        if model_type == "Inpaint":
            if need_processor:
                pipeline = QwenImageEditPipeline(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    transformer=transformer,
                    scheduler=None,
                    processor=processor,
                )
            else:
                pipeline = QwenImagePipeline(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    transformer=transformer,
                    scheduler=None,
                )
        else:
            raise ValueError("Not supported now.")

        pipeline.remove_all_hooks()
        undo_convert_weight_dtype_wrapper(transformer)

        if GPU_memory_mode == "sequential_cpu_offload":
            pipeline.enable_sequential_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["img_in", "txt_in", "timestep"], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_full_load_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["img_in", "txt_in", "timestep"], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.to(device=device)
        else:
            pipeline.to(device=device)

        funmodels = {
            'pipeline': pipeline, 
            'dtype': weight_dtype,
            'model_name': model_name,
            'model_type': model_type,
            'loras': [],
            'strength_model': []
        }
        return (funmodels,)

class LoadQwenImageLora:
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

class QwenImageT2VSampler:
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
                "width": (
                    "INT", {"default": 1344, "min": 64, "max": 2048, "step": 16}
                ),
                "height": (
                    "INT", {"default": 768, "min": 64, "max": 2048, "step": 16}
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 50, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.01}
                ),
                "scheduler": (
                    ["Flow", "Flow_Unipc", "Flow_DPM++"],
                    {
                        "default": 'Flow'
                    }
                ),
                "shift": (
                    "INT", {"default": 1, "min": 1, "max": 100, "step": 1}
                ),
                "teacache_threshold": (
                    "FLOAT", {"default": 0.250, "min": 0.00, "max": 1.00, "step": 0.005}
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
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, funmodels, prompt, negative_prompt, width, height, seed, steps, cfg, scheduler, shift, teacache_threshold, enable_teacache, num_skip_start_steps, teacache_offload, cfg_skip_ratio):
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
        pipeline.scheduler = get_qwen_scheduler(scheduler, shift)

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
                print('Merge Lora')
                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if len(transformer_cpu_cache) != 0:
                    pipeline.transformer.load_state_dict(transformer_cpu_cache)
                    transformer_cpu_cache = {}
                    lora_path_before = ""
                    gc.collect()
                
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)

            sample = pipeline(
                prompt, 
                negative_prompt = negative_prompt,
                height      = height,
                width       = width,
                generator   = generator,
                true_cfg_scale = cfg,
                num_inference_steps = steps,
                comfyui_progressbar = True,
            ).images
            image = torch.Tensor(np.array(sample[0])).unsqueeze(0) / 255

            if not funmodels.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)
        return (image,)   

class QwenImageEditSampler:
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
                "width": (
                    "INT", {"default": 1344, "min": 64, "max": 2048, "step": 16}
                ),
                "height": (
                    "INT", {"default": 768, "min": 64, "max": 2048, "step": 16}
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 50, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.01}
                ),
                "scheduler": (
                    ["Flow", "Flow_Unipc", "Flow_DPM++"],
                    {
                        "default": 'Flow'
                    }
                ),
                "shift": (
                    "INT", {"default": 1, "min": 1, "max": 100, "step": 1}
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
                "image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, funmodels, prompt, negative_prompt, width, height, seed, steps, cfg, scheduler, shift, teacache_threshold, enable_teacache, num_skip_start_steps, teacache_offload, cfg_skip_ratio, image=None):
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
        pipeline.scheduler = get_qwen_scheduler(scheduler, shift)

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
                print('Merge Lora')
                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if len(transformer_cpu_cache) != 0:
                    pipeline.transformer.load_state_dict(transformer_cpu_cache)
                    transformer_cpu_cache = {}
                    lora_path_before = ""
                    gc.collect()
                
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)

            image = [to_pil(image) for image in image]
            image = get_image(image[0]) if image is not None else image

            sample = pipeline(
                image       = image,
                prompt      = prompt,
                negative_prompt = negative_prompt,
                height      = height,
                width       = width,
                generator   = generator,
                true_cfg_scale = cfg,
                num_inference_steps = steps,
                comfyui_progressbar = True,
            ).images
            image = torch.Tensor(np.array(sample[0])).unsqueeze(0) / 255

            if not funmodels.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)
        return (image,)   
