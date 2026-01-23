
import copy
import gc
import inspect
import json
import os
from collections import OrderedDict

import accelerate
import comfy.model_management as mm
import cv2
import folder_paths
import numpy as np
import torch
from comfy.utils import ProgressBar, load_torch_file
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import __version__ as diffusers_version
from einops import rearrange
from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers import Mistral3Config

if diffusers_version >= "0.33.0":
    from diffusers.models.model_loading_utils import load_model_dict_into_meta
else:
    from diffusers.models.modeling_utils import \
        load_model_dict_into_meta

from ...videox_fun.data.bucket_sampler import (ASPECT_RATIO_512,
                                               get_closest_ratio)
from ...videox_fun.models import (AutoencoderKLFlux2,
                                  Flux2ControlTransformer2DModel,
                                  Flux2Transformer2DModel,
                                  Mistral3ForConditionalGeneration,
                                  PixtralProcessor)
from ...videox_fun.pipeline import Flux2ControlPipeline, Flux2Pipeline
from ...videox_fun.utils import (register_auto_device_hook,
                                 safe_enable_group_offload,
                                 safe_remove_group_offloading)
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

transformer_cpu_cache = {}
lora_path_before = ""

def get_flux2_scheduler(sampler_name, shift=1.0):
    Chosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    
    scheduler_kwargs = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "_diffusers_version": "0.36.0.dev0",
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "invert_sigmas": False,
        "max_image_seq_len": 4096,
        "max_shift": 1.15,
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "shift_terminal": None,
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


def convert_flux2_to_diffusers(original_state_dict):
    weight_dtype = get_autocast_dtype()
    converted = OrderedDict()
    
    def apply_scales(weight, prefix, layer_name):
        weight_scale_key = f'{prefix}.{layer_name}.weight_scale'
        input_scale_key = f'{prefix}.{layer_name}.input_scale'

        result = weight.to(weight_dtype) if weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else weight
        
        if weight_scale_key in original_state_dict:
            weight_scale = original_state_dict[weight_scale_key]
            weight_scale = weight_scale.to(weight_dtype) if weight_scale.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else weight_scale
            result = result * weight_scale

        return result
    
    # Time and guidance embeddings
    key = 'time_in.in_layer.weight'
    if key in original_state_dict:
        converted['time_guidance_embed.timestep_embedder.linear_1.weight'] = original_state_dict[key]
    
    key = 'time_in.out_layer.weight'
    if key in original_state_dict:
        converted['time_guidance_embed.timestep_embedder.linear_2.weight'] = original_state_dict[key]
    
    key = 'guidance_in.in_layer.weight'
    if key in original_state_dict:
        converted['time_guidance_embed.guidance_embedder.linear_1.weight'] = original_state_dict[key]
    
    key = 'guidance_in.out_layer.weight'
    if key in original_state_dict:
        converted['time_guidance_embed.guidance_embedder.linear_2.weight'] = original_state_dict[key]
    
    # Input projections
    key = 'img_in.weight'
    if key in original_state_dict:
        converted['x_embedder.weight'] = original_state_dict[key]
    
    key = 'txt_in.weight'
    if key in original_state_dict:
        converted['context_embedder.weight'] = original_state_dict[key]
    
    # Modulations
    key = 'double_stream_modulation_img.lin.weight'
    if key in original_state_dict:
        converted['double_stream_modulation_img.linear.weight'] = original_state_dict[key]
    
    key = 'double_stream_modulation_txt.lin.weight'
    if key in original_state_dict:
        converted['double_stream_modulation_txt.linear.weight'] = original_state_dict[key]
    
    key = 'single_stream_modulation.lin.weight'
    if key in original_state_dict:
        converted['single_stream_modulation.linear.weight'] = original_state_dict[key]
    
    # Double blocks (transformer_blocks)
    for i in range(8):
        prefix_old = f'double_blocks.{i}'
        prefix_new = f'transformer_blocks.{i}'
        
        qkv_key = f'{prefix_old}.img_attn.qkv.weight'
        if qkv_key in original_state_dict:
            qkv_weight = original_state_dict[qkv_key]
            qkv_weight = qkv_weight.to(weight_dtype) if qkv_weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else qkv_weight
            total_dim = qkv_weight.shape[0]
            single_dim = total_dim // 3
            converted[f'{prefix_new}.attn.to_q.weight'] = qkv_weight[:single_dim]
            converted[f'{prefix_new}.attn.to_k.weight'] = qkv_weight[single_dim:2*single_dim]
            converted[f'{prefix_new}.attn.to_v.weight'] = qkv_weight[2*single_dim:]
        
        # Norms
        key = f'{prefix_old}.img_attn.norm.query_norm.scale'
        if key in original_state_dict:
            converted[f'{prefix_new}.attn.norm_q.weight'] = original_state_dict[key]
        
        key = f'{prefix_old}.img_attn.norm.key_norm.scale'
        if key in original_state_dict:
            converted[f'{prefix_new}.attn.norm_k.weight'] = original_state_dict[key]
        
        # Output projection
        key = f'{prefix_old}.img_attn.proj.weight'
        if key in original_state_dict:
            converted[f'{prefix_new}.attn.to_out.0.weight'] = original_state_dict[key]
        
        # Text attention QKV (added)
        qkv_added_key = f'{prefix_old}.txt_attn.qkv.weight'
        if qkv_added_key in original_state_dict:
            qkv_weight = original_state_dict[qkv_added_key]
            qkv_weight = qkv_weight.to(weight_dtype) if qkv_weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else qkv_weight
            total_dim = qkv_weight.shape[0]
            single_dim = total_dim // 3
            converted[f'{prefix_new}.attn.add_q_proj.weight'] = qkv_weight[:single_dim]
            converted[f'{prefix_new}.attn.add_k_proj.weight'] = qkv_weight[single_dim:2*single_dim]
            converted[f'{prefix_new}.attn.add_v_proj.weight'] = qkv_weight[2*single_dim:]
        
        # Text norms
        key = f'{prefix_old}.txt_attn.norm.query_norm.scale'
        if key in original_state_dict:
            converted[f'{prefix_new}.attn.norm_added_q.weight'] = original_state_dict[key]
        
        key = f'{prefix_old}.txt_attn.norm.key_norm.scale'
        if key in original_state_dict:
            converted[f'{prefix_new}.attn.norm_added_k.weight'] = original_state_dict[key]
        
        # Text output projection
        key = f'{prefix_old}.txt_attn.proj.weight'
        if key in original_state_dict:
            converted[f'{prefix_new}.attn.to_add_out.weight'] = original_state_dict[key]
        
        # Image MLP with scales
        key = f'{prefix_old}.img_mlp.0.weight'
        if key in original_state_dict:
            converted[f'{prefix_new}.ff.linear_in.weight'] = apply_scales(
                original_state_dict[key],
                prefix_old, 'img_mlp.0'
            )
            asd = f'{prefix_new}.ff.linear_in.weight'

        
        key = f'{prefix_old}.img_mlp.2.weight'
        if key in original_state_dict:
            converted[f'{prefix_new}.ff.linear_out.weight'] = apply_scales(
                original_state_dict[key],
                prefix_old, 'img_mlp.2'
            )
        
        # Text MLP with scales
        key = f'{prefix_old}.txt_mlp.0.weight'
        if key in original_state_dict:
            converted[f'{prefix_new}.ff_context.linear_in.weight'] = apply_scales(
                original_state_dict[key],
                prefix_old, 'txt_mlp.0'
            )
        
        key = f'{prefix_old}.txt_mlp.2.weight'
        if key in original_state_dict:
            converted[f'{prefix_new}.ff_context.linear_out.weight'] = apply_scales(
                original_state_dict[key],
                prefix_old, 'txt_mlp.2'
            )
    
    # Single blocks (single_transformer_blocks)
    for i in range(48):
        prefix_old = f'single_blocks.{i}'
        prefix_new = f'single_transformer_blocks.{i}'
        
        # QKV+MLP projection with scales
        key = f'{prefix_old}.linear1.weight'
        if key in original_state_dict:
            converted[f'{prefix_new}.attn.to_qkv_mlp_proj.weight'] = apply_scales(
                original_state_dict[key],
                prefix_old, 'linear1'
            )
        
        # Norms
        key = f'{prefix_old}.norm.query_norm.scale'
        if key in original_state_dict:
            converted[f'{prefix_new}.attn.norm_q.weight'] = original_state_dict[key]
        
        key = f'{prefix_old}.norm.key_norm.scale'
        if key in original_state_dict:
            converted[f'{prefix_new}.attn.norm_k.weight'] = original_state_dict[key]
        
        # Output projection with scales
        key = f'{prefix_old}.linear2.weight'
        if key in original_state_dict:
            converted[f'{prefix_new}.attn.to_out.weight'] = apply_scales(
                original_state_dict[key],
                prefix_old, 'linear2'
            )
    
    # Final layer
    key = 'final_layer.adaLN_modulation.1.weight'
    if key in original_state_dict:
        height = original_state_dict[key].size()[0]
        height = int(height // 2)
        converted['norm_out.linear.weight'] = torch.cat(
            [original_state_dict[key][height:, :], original_state_dict[key][:height, :]], dim=0
        )

    key = 'final_layer.linear.weight'
    if key in original_state_dict:
        converted['proj_out.weight'] = original_state_dict[key]

    return converted


class LoadFlux2TransformerModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"default": "flux2_dev_fp8_e4m3fn.safetensors"},
                ),
                "precision": (
                    ["fp16", "bf16"],
                    {"default": "bf16"}
                ),
            },
        }
    
    RETURN_TYPES = ("TransformerModel", "STRING")
    RETURN_NAMES = ("transformer", "model_name")
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, model_name, precision):
        # Init weight_dtype and device
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()
        weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]

        mm.unload_all_models()
        mm.cleanup_models_gc()
        mm.soft_empty_cache()
        transformer = None

        model_path = folder_paths.get_full_path("diffusion_models", model_name)
        transformer_state_dict = load_torch_file(model_path, safe_load=True)
        
        model_name_in_pipeline = "FLUX.2-dev"
        kwargs = {
            "_class_name": "Flux2Transformer2DModel",
            "_diffusers_version": "0.36.0.dev0",
            "attention_head_dim": 128,
            "axes_dims_rope": [
                32,
                32,
                32,
                32
            ],
            "eps": 1e-06,
            "in_channels": 128,
            "joint_attention_dim": 15360,
            "mlp_ratio": 3.0,
            "num_attention_heads": 48,
            "num_layers": 8,
            "num_single_layers": 48,
            "out_channels": None,
            "patch_size": 1,
            "rope_theta": 2000,
            "timestep_guidance_channels": 256
        }

        sig = inspect.signature(Flux2Transformer2DModel)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        with accelerate.init_empty_weights():
            transformer = Flux2Transformer2DModel(**accepted)

        if 'time_in.in_layer.weight' in transformer_state_dict.keys():
            transformer_state_dict = convert_flux2_to_diffusers(transformer_state_dict)

        filtered_state_dict = {}
        for key in transformer_state_dict:
            if key in transformer.state_dict() and transformer.state_dict()[key].size() == transformer_state_dict[key].size():
                filtered_state_dict[key] = transformer_state_dict[key]
        missing_keys = set(transformer.state_dict().keys()) - set(filtered_state_dict.keys())
        if missing_keys:
            raise ValueError(f"Missing keys: {sorted(missing_keys)}")

        if diffusers_version >= "0.33.0":
            # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
            # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
            load_model_dict_into_meta(
                transformer,
                transformer_state_dict,
                dtype=weight_dtype,
                model_name_or_path="",
            )
        else:
            transformer._convert_deprecated_attention_blocks(transformer_state_dict)
            unexpected_keys = load_model_dict_into_meta(
                transformer,
                transformer_state_dict,
                device=offload_device,
                dtype=weight_dtype,
                model_name_or_path="",
            )

        transformer = transformer.eval().to(weight_dtype)
        return (transformer, model_name_in_pipeline)

class LoadFlux2VAEModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("vae"),
                    {"default": "flux2_vae.safetensors"}
                ),
                "precision": (
                    ["fp16", "bf16"],
                    {"default": "bf16"}
                ),
            },
        }

    RETURN_TYPES = ("VAEModel",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, model_name, precision):
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()
        
        weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
        model_path = folder_paths.get_full_path("vae", model_name)
        vae_state_dict = load_torch_file(model_path, safe_load=True)

        kwargs = {
            "_class_name": "AutoencoderKLFlux2",
            "_diffusers_version": "0.36.0.dev0",
            "act_fn": "silu",
            "batch_norm_eps": 0.0001,
            "batch_norm_momentum": 0.1,
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
            "latent_channels": 32,
            "layers_per_block": 2,
            "mid_block_add_attention": True,
            "norm_num_groups": 32,
            "out_channels": 3,
            "patch_size": [
                2,
                2
            ],
            "sample_size": 1024,
            "up_block_types": [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D"
            ],
            "use_post_quant_conv": True,
            "use_quant_conv": True
        }

        sig = inspect.signature(AutoencoderKLFlux2)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}

        vae = AutoencoderKLFlux2(**accepted)
        vae.load_state_dict(vae_state_dict)
        vae = vae.eval().to(device=offload_device, dtype=weight_dtype)
        return (vae,)


class LoadFlux2TextEncoderModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("text_encoders"),
                    {"default": "mistral3_fp8_scaled.safetensors"}
                ),
                "precision": (
                    ["fp16", "bf16"],
                    {"default": "bf16"}
                ),
            },
        }

    RETURN_TYPES = ("TextEncoderModel", "Tokenizer")
    RETURN_NAMES = ("text_encoder", "tokenizer")
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, model_name, precision):
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()
        
        weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
        model_path = folder_paths.get_full_path("text_encoders", model_name)
        text_state_dict = load_torch_file(model_path, safe_load=True)

        config_kwargs = {
            "architectures": [
                "Mistral3ForConditionalGeneration"
            ],
            "dtype": "bfloat16",
            "image_token_index": 10,
            "model_type": "mistral3",
            "multimodal_projector_bias": False,
            "projector_hidden_act": "gelu",
            "spatial_merge_size": 2,
            "text_config": {
                "attention_dropout": 0.0,
                "dtype": "bfloat16",
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 5120,
                "initializer_range": 0.02,
                "intermediate_size": 32768,
                "max_position_embeddings": 131072,
                "model_type": "mistral",
                "num_attention_heads": 32,
                "num_hidden_layers": 40,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-05,
                "rope_theta": 1000000000.0,
                "sliding_window": None,
                "use_cache": True,
                "vocab_size": 131072
            },
            "transformers_version": "4.57.1",
            "vision_config": {
                "attention_dropout": 0.0,
                "dtype": "bfloat16",
                "head_dim": 64,
                "hidden_act": "silu",
                "hidden_size": 1024,
                "image_size": 1540,
                "initializer_range": 0.02,
                "intermediate_size": 4096,
                "model_type": "pixtral",
                "num_attention_heads": 16,
                "num_channels": 3,
                "num_hidden_layers": 24,
                "patch_size": 14,
                "rope_theta": 10000.0
            },
            "vision_feature_layer": -1
        }
        config = Mistral3Config(**config_kwargs)
        text_encoder = Mistral3ForConditionalGeneration._from_config(config)
        
        if "tekken_model" in text_state_dict.keys():
            def convert_mistral3_to_diffusers(state_dict):
                new_state_dict = {}
                
                for key, value in state_dict.items():
                    if key == "tekken_model":
                        continue
                    if key.startswith("vision_tower."):
                        new_key = "model." + key
                    
                    elif key.startswith("multi_modal_projector."):
                        new_key = "model." + key
                    
                    elif key.startswith("model.layers."):
                        new_key = "model.language_" + key
                    
                    elif key.startswith("model.embed_tokens."):
                        new_key = "model.language_" + key
                    
                    elif key == "model.norm.weight":
                        new_key = "model.language_model.norm.weight"
                    
                    elif key.startswith("lm_head."):
                        new_key = key
                    
                    else:
                        print(f"Warning: Unmapped key: {key}")
                        new_key = key
                    
                    new_state_dict[new_key] = value
                
                return new_state_dict
        else:
            def convert_mistral3_to_diffusers(state_dict):
                new_state_dict = {}
                
                for key, value in state_dict.items():
                    if key.startswith('vision_tower.'):
                        new_key = 'model.' + key
                        
                    elif key.startswith('multi_modal_projector.'):
                        new_key = 'model.' + key
                        
                    elif key.startswith('language_model.model.embed_tokens.'):
                        new_key = key.replace('language_model.model.', 'model.language_model.')
                        
                    elif key.startswith('language_model.model.layers.'):
                        new_key = key.replace('language_model.model.', 'model.language_model.')
                        
                    elif key.startswith('language_model.model.norm.'):
                        new_key = key.replace('language_model.model.', 'model.language_model.')
                        
                    elif key.startswith('language_model.lm_head.'):
                        new_key = key.replace('language_model.', '')
                        
                    else:
                        new_key = key
                        
                    new_state_dict[new_key] = value
                
                return new_state_dict

        text_state_dict = convert_mistral3_to_diffusers(text_state_dict)
    
        text_encoder.load_state_dict(text_state_dict, strict=False)
        text_encoder = text_encoder.eval().to(device=offload_device, dtype=weight_dtype)

        possible_folders = ["CogVideoX_Fun", "Fun_Models", "VideoX_Fun", "Wan-AI", "Qwen"] + \
            [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models/Diffusion_Transformer")] # Possible folder names to check
        try:
            tokenizer_path = search_sub_dir_in_possible_folders(possible_folders, sub_dir_name="flux2_tokenizer")
        except:
            try:
                tokenizer_path = os.path.join(search_sub_dir_in_possible_folders(possible_folders, sub_dir_name="FLUX.2-dev"), "tokenizer")
            except:
                tokenizer_path = search_sub_dir_in_possible_folders(possible_folders, sub_dir_name="Mistral-Nemo-Instruct-2407")

        tokenizer = PixtralProcessor.from_pretrained(tokenizer_path)
        return (text_encoder, tokenizer)


class CombineFlux2Pipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transformer": ("TransformerModel",),
                "vae": ("VAEModel",),
                "text_encoder": ("TextEncoderModel",),
                "tokenizer": ("Tokenizer",),
                "model_name": ("STRING",),
                "GPU_memory_mode": (
                    [
                        "model_full_load", 
                        "model_full_load_and_qfloat8", 
                        "model_cpu_offload", 
                        "model_cpu_offload_and_qfloat8", 
                        "model_group_offload", 
                        "sequential_cpu_offload"
                    ],
                    {"default": "model_cpu_offload"}
                ),
            },
        }

    RETURN_TYPES = ("FunModels",)
    RETURN_NAMES = ("funmodels",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, model_name, GPU_memory_mode, transformer, vae, text_encoder, tokenizer):
        # Get pipeline
        weight_dtype    = transformer.dtype if transformer.dtype not in [torch.float32, torch.float8_e4m3fn, torch.float8_e5m2] else get_autocast_dtype()
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()

        # Get pipeline
        if hasattr(transformer, "control_transformer_blocks"):
            model_type = "Control"
        else:
            model_type = "Inpaint"

        if model_type == "Inpaint":
            pipeline = Flux2Pipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=None,
            )
        else:
            pipeline = Flux2ControlPipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=None,
            )

        pipeline.remove_all_hooks()
        safe_remove_group_offloading(pipeline)
        undo_convert_weight_dtype_wrapper(transformer)
        transformer = transformer.to(weight_dtype)

        if GPU_memory_mode == "sequential_cpu_offload":
            pipeline.enable_sequential_cpu_offload(device=device)
        elif GPU_memory_mode == "model_group_offload":
            register_auto_device_hook(pipeline.transformer)
            safe_enable_group_offload(pipeline, onload_device=device, offload_device=offload_device, offload_type="leaf_level", use_stream=True)
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


class LoadFlux2Model:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        'FLUX.2-dev',
                    ],
                    {"default": 'FLUX.2-dev'}
                ),
                "GPU_memory_mode":(
                    [
                        "model_full_load", "model_full_load_and_qfloat8", "model_cpu_offload", 
                        "model_cpu_offload_and_qfloat8", "model_group_offload", "sequential_cpu_offload"],
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
        mm.cleanup_models_gc()
        mm.soft_empty_cache()

        pbar = ProgressBar(5)

        # Detect model is existing or not
        possible_folders = ["CogVideoX_Fun", "Fun_Models", "VideoX_Fun", "Wan-AI"] + \
                [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models/Diffusion_Transformer")] # Possible folder names to check
        # Initialize model_name as None
        model_name = search_model_in_possible_folders(possible_folders, model)

        print("Loading VAE...")
        vae = AutoencoderKLFlux2.from_pretrained(
            model_name, 
            subfolder="vae"
        ).to(weight_dtype)
        pbar.update(1)

        print("Loading Scheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name, 
            subfolder="scheduler"
        )
        pbar.update(1)
        
        print("Loading Transformer...")
        transformer = Flux2Transformer2DModel.from_pretrained(
            model_name, 
            subfolder="transformer",
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        pbar.update(1)

        print("Loading Tokenizer...")
        tokenizer = PixtralProcessor.from_pretrained(
            model_name, 
            subfolder="tokenizer"
        )
        pbar.update(1)

        print("Loading Text Encoder...")
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            model_name, 
            subfolder="text_encoder", 
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=True,
        )
        pbar.update(1)

        pipeline = Flux2Pipeline(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=None,
        )

        pipeline.remove_all_hooks()
        undo_convert_weight_dtype_wrapper(transformer)

        if GPU_memory_mode == "sequential_cpu_offload":
            pipeline.enable_sequential_cpu_offload(device=device)
        elif GPU_memory_mode == "model_group_offload":
            register_auto_device_hook(pipeline.transformer)
            safe_enable_group_offload(pipeline, onload_device=device, offload_device=offload_device, offload_type="leaf_level", use_stream=True)
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


class LoadFlux2Lora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "funmodels": ("FunModels",),
                "lora_name": (folder_paths.get_filename_list("loras"), {"default": None,}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_cache": ([False, True], {"default": False,}),
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


class LoadFlux2ControlNetInPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config": (
                    [
                        "flux2/flux2_control.yaml",
                    ],
                    {
                        "default": "flux2/flux2_control.yaml",
                    }
                ),
                "model_name": (
                    folder_paths.get_filename_list("model_patches"),
                    {"default": "FLUX.2-dev-Fun-Controlnet-Union.safetensors", },
                ),
                "funmodels": ("FunModels",),
            },
        }
    
    RETURN_TYPES = ("FunModels",)
    RETURN_NAMES = ("funmodels",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, config, model_name, funmodels):
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()
        GPU_memory_mode = funmodels["GPU_memory_mode"]
        weight_dtype    = funmodels['dtype']

        # Remove hooks
        funmodels["pipeline"].remove_all_hooks()
        safe_remove_group_offloading(funmodels["pipeline"])

        # Get Transformer
        transformer = funmodels["pipeline"].transformer
        transformer = transformer.cpu()

        # Get state_dict
        transformer_state_dict = transformer.state_dict()
        del transformer
        mm.soft_empty_cache()
        gc.collect()

        # Load config
        config_path = f"{script_directory}/config/{config}"
        config = OmegaConf.load(config_path)
        kwargs = {
            "_class_name": "Flux2Transformer2DModel",
            "_diffusers_version": "0.36.0.dev0",
            "attention_head_dim": 128,
            "axes_dims_rope": [
                32,
                32,
                32,
                32
            ],
            "eps": 1e-06,
            "in_channels": 128,
            "joint_attention_dim": 15360,
            "mlp_ratio": 3.0,
            "num_attention_heads": 48,
            "num_layers": 8,
            "num_single_layers": 48,
            "out_channels": None,
            "patch_size": 1,
            "rope_theta": 2000,
            "timestep_guidance_channels": 256
        }
        kwargs.update(OmegaConf.to_container(config['transformer_additional_kwargs']))

        # Get Model
        sig = inspect.signature(Flux2ControlTransformer2DModel)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        with accelerate.init_empty_weights():
            control_transformer = Flux2ControlTransformer2DModel.from_config(accepted).to(weight_dtype)
        print(f"Load Flux Control Transformer")

        # Load Control state_dict
        control_model_path = folder_paths.get_full_path("model_patches", model_name)
        if control_model_path.endswith(".safetensors"):
            control_state_dict = load_file(control_model_path)
        else:
            control_state_dict = torch.load(control_model_path)

        state_dict = {**transformer_state_dict, **control_state_dict}
        if diffusers_version >= "0.33.0":
            # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
            # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
            load_model_dict_into_meta(
                control_transformer,
                state_dict,
                dtype=weight_dtype,
                model_name_or_path="",
            )
        else:
            control_transformer._convert_deprecated_attention_blocks(state_dict)
            load_model_dict_into_meta(
                control_transformer,
                state_dict,
                device=offload_device,
                dtype=weight_dtype,
                model_name_or_path="",
            )

        # Create Pipeline
        pipeline = Flux2ControlPipeline(
            vae=funmodels["pipeline"].vae,
            tokenizer=funmodels["pipeline"].tokenizer,
            text_encoder=funmodels["pipeline"].text_encoder,
            transformer=control_transformer,
            scheduler=funmodels["pipeline"].scheduler,
        ) 
        del funmodels["pipeline"]
        mm.soft_empty_cache()
        gc.collect()

        # Apply GPU memory mode
        if GPU_memory_mode == "sequential_cpu_offload":
            pipeline.enable_sequential_cpu_offload(device=device)
        elif GPU_memory_mode == "model_group_offload":
            register_auto_device_hook(pipeline.transformer)
            safe_enable_group_offload(pipeline, onload_device=device, offload_device=offload_device, offload_type="leaf_level", use_stream=True)
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(control_transformer, exclude_module_name=["img_in", "txt_in", "timestep"], device=device)
            convert_weight_dtype_wrapper(control_transformer, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_full_load_and_qfloat8":
            convert_model_weight_to_float8(control_transformer, exclude_module_name=["img_in", "txt_in", "timestep"], device=device)
            convert_weight_dtype_wrapper(control_transformer, weight_dtype)
            pipeline.to(device=device)
        else:
            pipeline.to(device=device)
        funmodels["pipeline"] = pipeline
        funmodels["model_type"] = "Control"
        return (funmodels, )


class LoadFlux2ControlNetInModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config": (
                    [
                        "flux2/flux2_control.yaml",
                    ],
                    {
                        "default": "flux2/flux2_control.yaml",
                    }
                ),
                "model_name": (
                    folder_paths.get_filename_list("model_patches"),
                    {"default": "FLUX.2-dev-Fun-Controlnet-Union.safetensors", },
                ),
                "transformer": ("TransformerModel",),
            },
        }
    
    RETURN_TYPES = ("TransformerModel",)
    RETURN_NAMES = ("transformer",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, config, model_name, transformer):
        offload_device  = mm.unet_offload_device()
        dtype           = transformer.dtype
        
        # Get Transformer
        transformer = transformer.cpu()

        # Get state_dict
        transformer_state_dict = transformer.state_dict()
        del transformer
        mm.soft_empty_cache()
        gc.collect()

        # Load config
        config_path = f"{script_directory}/config/{config}"
        config = OmegaConf.load(config_path)
        kwargs = {
            "_class_name": "Flux2Transformer2DModel",
            "_diffusers_version": "0.36.0.dev0",
            "attention_head_dim": 128,
            "axes_dims_rope": [
                32,
                32,
                32,
                32
            ],
            "eps": 1e-06,
            "in_channels": 128,
            "joint_attention_dim": 15360,
            "mlp_ratio": 3.0,
            "num_attention_heads": 48,
            "num_layers": 8,
            "num_single_layers": 48,
            "out_channels": None,
            "patch_size": 1,
            "rope_theta": 2000,
            "timestep_guidance_channels": 256
        }
        kwargs.update(OmegaConf.to_container(config['transformer_additional_kwargs']))

        # Get Model
        sig = inspect.signature(Flux2ControlTransformer2DModel)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        with accelerate.init_empty_weights():
            control_transformer = Flux2ControlTransformer2DModel.from_config(accepted).to(dtype)
        print(f"Load Flux Control Transformer")

        # Load Control state_dict
        control_model_path = folder_paths.get_full_path("model_patches", model_name)
        if control_model_path.endswith(".safetensors"):
            control_state_dict = load_file(control_model_path)
        else:
            control_state_dict = torch.load(control_model_path)

        state_dict = {**transformer_state_dict, **control_state_dict}
        if diffusers_version >= "0.33.0":
            # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
            # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
            load_model_dict_into_meta(
                control_transformer,
                state_dict,
                dtype=dtype,
                model_name_or_path="",
            )
        else:
            control_transformer._convert_deprecated_attention_blocks(state_dict)
            load_model_dict_into_meta(
                control_transformer,
                state_dict,
                device=offload_device,
                dtype=dtype,
                model_name_or_path="",
            )
        return (control_transformer, )


class Flux2T2ISampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "funmodels": ("FunModels",),
                "prompt": ("STRING_PROMPT",),
                "width": ("INT", {"default": 1728, "min": 64, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 992, "min": 64, "max": 2048, "step": 16}),
                "seed": ("INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.01}),
                "scheduler": (
                    ["Flow", "Flow_Unipc", "Flow_DPM++"],
                    {"default": 'Flow'}
                ),
                "shift": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(
        self, 
        funmodels, 
        prompt, 
        width, 
        height, 
        seed, 
        steps, 
        cfg, 
        scheduler, 
        shift,
    ):
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
        pipeline.scheduler = get_flux2_scheduler(scheduler, shift)

        generator = torch.Generator(device).manual_seed(seed)

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
                prompt=prompt,
                height=height,
                width=width,
                generator=generator,
                guidance_scale=cfg,
                num_inference_steps=steps,
                comfyui_progressbar=True,
            ).images
            
            image = torch.Tensor(np.array(sample[0])).unsqueeze(0) / 255

            if not funmodels.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)
        return (image,)   


class Flux2ControlSampler:
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
                "width": (
                    "INT", {"default": 992, "min": 64, "max": 20480, "step": 16}
                ),
                "height": (
                    "INT", {"default": 1728, "min": 64, "max": 20480, "step": 16}
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 50, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.01}
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
                "control_context_scale": (
                    "FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.01}
                ),
            },
            "optional":{
                "control_image": ("IMAGE",),
                "inpaint_image": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, funmodels, prompt, width, height, seed, steps, cfg, scheduler, shift, control_context_scale, control_image=None, inpaint_image=None, mask_image=None, image=None):
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
        pipeline.scheduler = get_flux2_scheduler(scheduler, shift)

        generator = torch.Generator(device).manual_seed(seed)

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

            # Process images
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

            if image is not None:
                image = [to_pil(image) for image in image]

            # Generate
            sample = pipeline(
                prompt              = prompt, 
                height              = sample_size[0],
                width               = sample_size[1],
                generator           = generator,
                guidance_scale      = cfg,
                image               = image,
                inpaint_image       = inpaint_image,
                mask_image          = mask_image,
                control_image       = control_image,
                num_inference_steps = steps,
                control_context_scale   = control_context_scale,
                comfyui_progressbar     = True,
            ).images
            image = torch.Tensor(np.array(sample[0])).unsqueeze(0) / 255

            # Unmerge lora
            if not funmodels.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device=device, dtype=weight_dtype)
        return (image,)   
