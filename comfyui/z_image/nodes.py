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
from safetensors.torch import load_file

from ...videox_fun.data.bucket_sampler import (ASPECT_RATIO_512,
                                               get_closest_ratio)
from ...videox_fun.models import (AutoencoderKL, AutoTokenizer,
                                  Qwen2VLProcessor, Qwen3Config,
                                  Qwen3ForCausalLM,
                                  ZImageControlTransformer2DModel,
                                  ZImageTransformer2DModel)
from ...videox_fun.models.cache_utils import get_teacache_coefficients
from ...videox_fun.pipeline import ZImageControlPipeline, ZImagePipeline
from ...videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from ...videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ...videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8, convert_weight_dtype_wrapper,
    replace_parameters_by_name, undo_convert_weight_dtype_wrapper)
from ...videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from ...videox_fun.utils.utils import (filter_kwargs, get_autocast_dtype,
                                       get_image, get_image_latent)
from ..comfyui_utils import (eas_cache_dir, script_directory,
                             search_model_in_possible_folders,
                             search_sub_dir_in_possible_folders, to_pil)

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
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "_diffusers_version": "0.36.0.dev0",
        "num_train_timesteps": 1000,
        "use_dynamic_shifting": False,
        "shift": 3.0
    }
    scheduler_kwargs['shift'] = shift
    scheduler = Chosen_Scheduler(
        **filter_kwargs(Chosen_Scheduler, scheduler_kwargs)
    )
    return scheduler

class LoadZImageTransformerModel:
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
        def convert_state_dict(old_state_dict):
            new_state_dict = {}
            for key, value in old_state_dict.items():
                # 1. Convert x_embedder to all_x_embedder.2-1
                if key.startswith('x_embedder.'):
                    new_key = key.replace('x_embedder.', 'all_x_embedder.2-1.')
                    new_state_dict[new_key] = value
                # 2. Convert final_layer to all_final_layer.2-1
                elif key.startswith('final_layer.'):
                    new_key = key.replace('final_layer.', 'all_final_layer.2-1.')
                    new_state_dict[new_key] = value
                    
                # 3. Handle attention layers
                elif '.attention.' in key:
                    # Convert q_norm to norm_q
                    if '.q_norm.' in key:
                        new_key = key.replace('.q_norm.', '.norm_q.')
                        new_state_dict[new_key] = value
                        
                    # Convert k_norm to norm_k
                    elif '.k_norm.' in key:
                        new_key = key.replace('.k_norm.', '.norm_k.')
                        new_state_dict[new_key] = value
                        
                    # Convert out to to_out.0
                    elif '.out.' in key:
                        new_key = key.replace('.out.', '.to_out.0.')
                        new_state_dict[new_key] = value
                        
                    # Split qkv.weight into to_q, to_k, to_v
                    elif '.qkv.weight' in key:
                        q, k, v = value.chunk(3, dim=0)
                        base_key = key.replace('.qkv.weight', '')
                        new_state_dict[base_key + '.to_q.weight'] = q
                        new_state_dict[base_key + '.to_k.weight'] = k
                        new_state_dict[base_key + '.to_v.weight'] = v
                        
                    # Split qkv.bias into to_q, to_k, to_v (if exists)
                    elif '.qkv.bias' in key:
                        q, k, v = value.chunk(3, dim=0)
                        base_key = key.replace('.qkv.bias', '')
                        new_state_dict[base_key + '.to_q.bias'] = q
                        new_state_dict[base_key + '.to_k.bias'] = k
                        new_state_dict[base_key + '.to_v.bias'] = v
                        
                    else:
                        new_state_dict[key] = value
                else:
                    new_state_dict[key] = value
            
            return new_state_dict
        if "x_embedder.weight" in transformer_state_dict.keys():
            transformer_state_dict = convert_state_dict(transformer_state_dict)

        model_name_in_pipeline = "Z-Image"
        kwargs = {
            "_class_name": "ZImageTransformer2DModel",
            "_diffusers_version": "0.36.0.dev0",
            "all_f_patch_size": [
                1
            ],
            "all_patch_size": [
                2
            ],
            "axes_dims": [
                32,
                48,
                48
            ],
            "axes_lens": [
                1536,
                512,
                512
            ],
            "cap_feat_dim": 2560,
            "dim": 3840,
            "in_channels": 16,
            "n_heads": 30,
            "n_kv_heads": 30,
            "n_layers": 30,
            "n_refiner_layers": 2,
            "norm_eps": 1e-05,
            "qk_norm": True,
            "rope_theta": 256.0,
            "t_scale": 1000.0
        }

        sig = inspect.signature(ZImageTransformer2DModel)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        transformer = ZImageTransformer2DModel(**accepted)
        transformer.load_state_dict(transformer_state_dict)
        transformer = transformer.eval().to(device=offload_device, dtype=weight_dtype)
        return (transformer, model_name_in_pipeline)

class LoadZImageVAEModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("vae"),
                    {"default": "ZImage2.1_VAE.pth"}
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
        def convert_state_dict(old_state_dict):
            """
            Convert old format VAE state_dict to new format (diffusers format)
            """
            import re
            
            new_state_dict = {}
            
            # Determine the number of down/up blocks (usually 4)
            num_blocks = 4
            
            for old_key, value in old_state_dict.items():
                new_key = old_key
                
                # Process encoder
                if old_key.startswith('encoder.'):
                    # down blocks
                    if '.down.' in old_key:
                        # encoder.down.X.block.Y -> encoder.down_blocks.X.resnets.Y
                        new_key = new_key.replace('.down.', '.down_blocks.')
                        new_key = new_key.replace('.block.', '.resnets.')
                        # downsample
                        new_key = new_key.replace('.downsample.', '.downsamplers.0.')
                        # nin_shortcut -> conv_shortcut
                        new_key = new_key.replace('.nin_shortcut.', '.conv_shortcut.')
                    
                    # mid block
                    elif '.mid.' in old_key:
                        if '.block_1.' in old_key:
                            new_key = new_key.replace('.mid.block_1.', '.mid_block.resnets.0.')
                        elif '.block_2.' in old_key:
                            new_key = new_key.replace('.mid.block_2.', '.mid_block.resnets.1.')
                        elif '.attn_1.' in old_key:
                            new_key = new_key.replace('.mid.attn_1.', '.mid_block.attentions.0.')
                            # Convert attention layer naming
                            new_key = new_key.replace('.q.', '.to_q.')
                            new_key = new_key.replace('.k.', '.to_k.')
                            new_key = new_key.replace('.v.', '.to_v.')
                            new_key = new_key.replace('.proj_out.', '.to_out.0.')
                            new_key = new_key.replace('.norm.', '.group_norm.')
                            
                            # If it's an attention weight and is 4D, convert to 2D
                            if ('to_q.weight' in new_key or 'to_k.weight' in new_key or 
                                'to_v.weight' in new_key or 'to_out.0.weight' in new_key):
                                if len(value.shape) == 4:  # Conv2d weight [out, in, 1, 1]
                                    value = value.squeeze(-1).squeeze(-1)  # -> [out, in]
                    
                    # norm_out
                    elif '.norm_out.' in old_key:
                        new_key = new_key.replace('.norm_out.', '.conv_norm_out.')
                
                # Process decoder
                elif old_key.startswith('decoder.'):
                    # up blocks - need to reverse indices
                    if '.up.' in old_key:
                        # Extract original index
                        match = re.search(r'\.up\.(\d+)\.', old_key)
                        if match:
                            old_idx = int(match.group(1))
                            # Reverse index: 0->3, 1->2, 2->1, 3->0
                            new_idx = num_blocks - 1 - old_idx
                            
                            # decoder.up.X.block.Y -> decoder.up_blocks.X.resnets.Y
                            new_key = re.sub(r'\.up\.(\d+)\.', f'.up_blocks.{new_idx}.', new_key)
                            new_key = new_key.replace('.block.', '.resnets.')
                            # upsample
                            new_key = new_key.replace('.upsample.', '.upsamplers.0.')
                            # nin_shortcut -> conv_shortcut
                            new_key = new_key.replace('.nin_shortcut.', '.conv_shortcut.')
                    
                    # mid block
                    elif '.mid.' in old_key:
                        if '.block_1.' in old_key:
                            new_key = new_key.replace('.mid.block_1.', '.mid_block.resnets.0.')
                        elif '.block_2.' in old_key:
                            new_key = new_key.replace('.mid.block_2.', '.mid_block.resnets.1.')
                        elif '.attn_1.' in old_key:
                            new_key = new_key.replace('.mid.attn_1.', '.mid_block.attentions.0.')
                            # Convert attention layer naming
                            new_key = new_key.replace('.q.', '.to_q.')
                            new_key = new_key.replace('.k.', '.to_k.')
                            new_key = new_key.replace('.v.', '.to_v.')
                            new_key = new_key.replace('.proj_out.', '.to_out.0.')
                            new_key = new_key.replace('.norm.', '.group_norm.')
                            
                            # If it's an attention weight and is 4D, convert to 2D
                            if ('to_q.weight' in new_key or 'to_k.weight' in new_key or 
                                'to_v.weight' in new_key or 'to_out.0.weight' in new_key):
                                if len(value.shape) == 4:  # Conv2d weight [out, in, 1, 1]
                                    value = value.squeeze(-1).squeeze(-1)  # -> [out, in]
                    
                    # norm_out
                    elif '.norm_out.' in old_key:
                        new_key = new_key.replace('.norm_out.', '.conv_norm_out.')
                
                new_state_dict[new_key] = value
            
            return new_state_dict
        if "encoder.down.0.block.0.conv1.weight" in vae_state_dict.keys():
            vae_state_dict = convert_state_dict(vae_state_dict)
        
        kwargs = {
            "_class_name": "AutoencoderKL",
            "_diffusers_version": "0.36.0.dev0",
            "_name_or_path": "flux-dev",
            "act_fn": "silu",
            "block_out_channels": [
                128,
                256,
                512,
                512
            ],
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D"
            ],
            "force_upcast": True,
            "in_channels": 3,
            "latent_channels": 16,
            "latents_mean": None,
            "latents_std": None,
            "layers_per_block": 2,
            "mid_block_add_attention": True,
            "norm_num_groups": 32,
            "out_channels": 3,
            "sample_size": 1024,
            "scaling_factor": 0.3611,
            "shift_factor": 0.1159,
            "up_block_types": [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D"
            ],
            "use_post_quant_conv": False,
            "use_quant_conv": False
        }

        sig = inspect.signature(AutoencoderKL)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}

        vae = AutoencoderKL(**accepted)
        vae.load_state_dict(vae_state_dict)
        vae = vae.eval().to(device=offload_device, dtype=weight_dtype)
        return (vae,)

class LoadZImageTextEncoderModel:
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

        kwargs = {
            "architectures": [
                "Qwen3ForCausalLM"
            ],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2560,
            "initializer_range": 0.02,
            "intermediate_size": 9728,
            "max_position_embeddings": 40960,
            "max_window_layers": 36,
            "model_type": "qwen3",
            "num_attention_heads": 32,
            "num_hidden_layers": 36,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 1000000,
            "sliding_window": None,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.51.0",
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 151936
        }
        config = Qwen3Config(**kwargs)
        text_encoder = Qwen3ForCausalLM._from_config(config)
        m, u = text_encoder.load_state_dict(text_state_dict, strict=False)
        print(f"### Text Encoder missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        text_encoder = text_encoder.eval().to(device=offload_device, dtype=weight_dtype)

        possible_folders = ["CogVideoX_Fun", "Fun_Models", "VideoX_Fun", "Wan-AI", "Qwen"] + \
            [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models/Diffusion_Transformer")] # Possible folder names to check
        try:
            tokenizer_path = search_sub_dir_in_possible_folders(possible_folders, sub_dir_name="qwen3_tokenizer")
        except:
            try:
                tokenizer_path = os.path.join(search_sub_dir_in_possible_folders(possible_folders, sub_dir_name="Z-Image-Turbo"), "tokenizer")
            except:
                tokenizer_path = search_sub_dir_in_possible_folders(possible_folders, sub_dir_name="Qwen3-4B")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return (text_encoder, tokenizer)


class CombineZImagePipeline:
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

        if hasattr(transformer, "control_layers_places"):
            model_type = "Control"
        else:
            model_type = "Inpaint"

        # Get pipeline
        if model_type == "Inpaint":
            pipeline = ZImagePipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=None,
            )
        else:
            pipeline = ZImageControlPipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=None,
            )

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
            'GPU_memory_mode': GPU_memory_mode,
            'dtype': weight_dtype,
            'model_name': model_name,
            'model_type': model_type,
            'loras': [],
            'strength_model': []
        }
        return (funmodels,)

class LoadZImageModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "Z-Image-Turbo"
                    ],
                    {
                        "default": 'Z-Image-Turbo',
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
        vae = AutoencoderKL.from_pretrained(
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
        transformer = ZImageTransformer2DModel.from_pretrained(
            model_name, 
            subfolder="transformer",
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        # Update pbar
        pbar.update(1) 

        # Get tokenizer and text_encoder
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, subfolder="tokenizer"
        )
        pbar.update(1) 

        text_encoder = Qwen3ForCausalLM.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=weight_dtype
        )
        pbar.update(1) 


        model_type = "Inpaint"
        if model_type == "Inpaint":
            pipeline = ZImagePipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
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

        pipeline = pipeline
        funmodels = {
            'pipeline': pipeline, 
            'GPU_memory_mode': GPU_memory_mode,
            'dtype': weight_dtype,
            'model_name': model_name,
            'model_type': model_type,
            'loras': [],
            'strength_model': []
        }
        return (funmodels,)

class LoadZImageLora:
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
            loras = list(new_funmodels.get("loras", [])) + [folder_paths.get_full_path("loras", lora_name)]
            strength_models = list(new_funmodels.get("strength_model", [])) + [strength_model]
            new_funmodels['loras'] = loras
            new_funmodels['strength_model'] = strength_models
            new_funmodels['lora_cache'] = lora_cache
        return (new_funmodels,)

class LoadZImageControlNetInPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config": (
                    [
                        "z_image/z_image_control_2.1.yaml",
                        "z_image/z_image_control_2.0.yaml",
                        "z_image/z_image_control_1.0.yaml",
                    ],
                    {
                        "default": "z_image/z_image_control_2.1.yaml",
                    }
                ),
                "model_name": (
                    folder_paths.get_filename_list("model_patches"),
                    {"default": "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors", },
                ),
                "sub_transformer_name":(
                    ["transformer", "transformer_2"],
                    {
                        "default": "transformer",
                    }
                ),
                "funmodels": ("FunModels",),
            },
        }
    
    RETURN_TYPES = ("FunModels",)
    RETURN_NAMES = ("funmodels",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, config, model_name, sub_transformer_name, funmodels):
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()
        # Get Transformer
        transformer = getattr(funmodels["pipeline"], sub_transformer_name)
        transformer = transformer.cpu()

        # Load config
        config_path = f"{script_directory}/config/{config}"
        config = OmegaConf.load(config_path)
        kwargs = {
            "_class_name": "ZImageTransformer2DModel",
            "_diffusers_version": "0.36.0.dev0",
            "all_f_patch_size": [
                1
            ],
            "all_patch_size": [
                2
            ],
            "axes_dims": [
                32,
                48,
                48
            ],
            "axes_lens": [
                1536,
                512,
                512
            ],
            "cap_feat_dim": 2560,
            "dim": 3840,
            "in_channels": 16,
            "n_heads": 30,
            "n_kv_heads": 30,
            "n_layers": 30,
            "n_refiner_layers": 2,
            "norm_eps": 1e-05,
            "qk_norm": True,
            "rope_theta": 256.0,
            "t_scale": 1000.0
        }
        kwargs.update(OmegaConf.to_container(config['transformer_additional_kwargs']))
        sig = inspect.signature(ZImageControlTransformer2DModel)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}

        control_transformer = ZImageControlTransformer2DModel(**accepted).to(transformer.dtype)
        m, u = control_transformer.load_state_dict(transformer.state_dict(), strict=False)
        print(f"### Load Control Transformer missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

        model_path = folder_paths.get_full_path("model_patches", model_name)
        if model_path.endswith(".safetensors"):
            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path)
        m, u = control_transformer.load_state_dict(state_dict, strict=False)
        print(f"### Load Control Model missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

        GPU_memory_mode = funmodels["GPU_memory_mode"]
        weight_dtype    = funmodels['dtype']
        pipeline        = ZImageControlPipeline(
            vae=funmodels["pipeline"].vae,
            tokenizer=funmodels["pipeline"].tokenizer,
            text_encoder=funmodels["pipeline"].text_encoder,
            transformer=control_transformer,
            scheduler=funmodels["pipeline"].scheduler,
        ) 
        del transformer
        del funmodels["pipeline"]
        mm.soft_empty_cache()
        gc.collect()

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
        funmodels["pipeline"] = pipeline
        funmodels["model_type"] = "Control"
        return (funmodels, )

class LoadZImageControlNetInModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config": (
                    [
                        "z_image/z_image_control_2.1.yaml",
                        "z_image/z_image_control_2.0.yaml",
                        "z_image/z_image_control_1.0.yaml",
                    ],
                    {
                        "default": "z_image/z_image_control_2.1.yaml",
                    }
                ),
                "model_name": (
                    folder_paths.get_filename_list("model_patches"),
                    {"default": "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors",},
                ),
                "transformer": ("TransformerModel",),
            },
        }
    
    RETURN_TYPES = ("TransformerModel",)
    RETURN_NAMES = ("transformer",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, config, model_name, transformer):
        transformer = transformer.cpu()

        # Load config
        config_path = f"{script_directory}/config/{config}"
        config = OmegaConf.load(config_path)
        kwargs = {
            "_class_name": "ZImageTransformer2DModel",
            "_diffusers_version": "0.36.0.dev0",
            "all_f_patch_size": [
                1
            ],
            "all_patch_size": [
                2
            ],
            "axes_dims": [
                32,
                48,
                48
            ],
            "axes_lens": [
                1536,
                512,
                512
            ],
            "cap_feat_dim": 2560,
            "dim": 3840,
            "in_channels": 16,
            "n_heads": 30,
            "n_kv_heads": 30,
            "n_layers": 30,
            "n_refiner_layers": 2,
            "norm_eps": 1e-05,
            "qk_norm": True,
            "rope_theta": 256.0,
            "t_scale": 1000.0
        }
        kwargs.update(OmegaConf.to_container(config['transformer_additional_kwargs']))
        sig = inspect.signature(ZImageControlTransformer2DModel)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        control_transformer = ZImageControlTransformer2DModel(**accepted).to(transformer.dtype)
        control_transformer.load_state_dict(transformer.state_dict(), strict=False)
        print(f"Load Control Transformer")

        model_path = folder_paths.get_full_path("model_patches", model_name)
        if model_path.endswith(".safetensors"):
            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path)
        m, u = control_transformer.load_state_dict(state_dict, strict=False)
        print(f"### patch model missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        return (control_transformer, )

class ZImageT2ISampler:
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
                    "INT", {"default": 1568, "min": 64, "max": 20480, "step": 16}
                ),
                "height": (
                    "INT", {"default": 1184, "min": 64, "max": 20480, "step": 16}
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 8, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01}
                ),
                "scheduler": (
                    ["Flow", "Flow_Unipc", "Flow_DPM++"],
                    {
                        "default": 'Flow'
                    }
                ),
                "shift": (
                    "INT", {"default": 3, "min": 1, "max": 100, "step": 1}
                ), 
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, funmodels, prompt, negative_prompt, width, height, seed, steps, cfg, scheduler, shift):
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
        pipeline.scheduler = get_qwen_scheduler(scheduler, shift)

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
                guidance_scale = cfg,
                num_inference_steps = steps,
                comfyui_progressbar = True,
            ).images
            image = torch.Tensor(np.array(sample[0])).unsqueeze(0) / 255

            if not funmodels.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)
        return (image,)   

class ZImageControlSampler:
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
                    "INT", {"default": 1568, "min": 64, "max": 20480, "step": 16}
                ),
                "height": (
                    "INT", {"default": 1184, "min": 64, "max": 20480, "step": 16}
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 8, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01}
                ),
                "scheduler": (
                    ["Flow", "Flow_Unipc", "Flow_DPM++"],
                    {
                        "default": 'Flow'
                    }
                ),
                "shift": (
                    "INT", {"default": 3, "min": 1, "max": 100, "step": 1}
                ), 
                "control_context_scale": (
                    "FLOAT", {"default": 0.80, "min": 0.0, "max": 2.0, "step": 0.01}
                ),
            },
            "optional":{
                "control_image": ("IMAGE",),
                "inpaint_image": ("IMAGE",),
                "mask_image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, funmodels, prompt, negative_prompt, width, height, seed, steps, cfg, scheduler, shift, control_context_scale, control_image=None, inpaint_image=None, mask_image=None):
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
        sample_size = [height, width]

        # Load Sampler
        pipeline.scheduler = get_qwen_scheduler(scheduler, shift)

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

            if inpaint_image is not None:
                inpaint_image = [to_pil(inpaint_image) for inpaint_image in inpaint_image][0]
                inpaint_image = get_image_latent(inpaint_image, sample_size=sample_size)[:, :, 0]
            else:
                inpaint_image = torch.zeros([1, 3, sample_size[0], sample_size[1]])

            if mask_image is not None:
                mask_image = [to_pil(mask_image) for mask_image in mask_image][0]
                mask_image = get_image_latent(mask_image, sample_size=sample_size)[:, :1, 0]
            else:
                mask_image = torch.ones([1, 1, sample_size[0], sample_size[1]]) * 255

            if control_image is not None:
                control_image = [to_pil(control_image) for control_image in control_image][0]
                control_image = get_image_latent(control_image, sample_size=sample_size)[:, :, 0]

            sample = pipeline(
                prompt, 
                negative_prompt = negative_prompt,
                height      = height,
                width       = width,
                generator   = generator,
                guidance_scale = cfg,
                num_inference_steps = steps,
                image               = inpaint_image,
                mask_image          = mask_image,
                control_image       = control_image,
                control_context_scale = control_context_scale,
                comfyui_progressbar = True,
            ).images
            image = torch.Tensor(np.array(sample[0])).unsqueeze(0) / 255

            if not funmodels.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)
        return (image,)   
