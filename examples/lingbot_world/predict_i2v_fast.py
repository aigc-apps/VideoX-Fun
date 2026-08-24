import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoTokenizer,
                               WanT5EncoderModel,
                               WanTransformer3DModel_LingbotWorldFast)
from videox_fun.pipeline import WanFunLingbotWorldFastPipeline
from videox_fun.utils import (register_auto_device_hook,
                              safe_enable_group_offload)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.utils import filter_kwargs, save_videos_grid
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, model_group_offload, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_group_offload transfers internal layer groups between CPU/CUDA, 
# balancing memory efficiency and speed between full-module and leaf-level offloading methods.
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
GPU_memory_mode     = "model_cpu_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# Config and model path.
# The lingbot-world fast checkpoint only ships the transformer (16 shards at the
# repo root); its VAE / T5 / tokenizer are sourced from the base-cam repo, which
# keeps the raw Wan2.1 layout used by ``config/wan2.1/wan_civitai.yaml``.
config_path         = "config/wan2.1/wan_civitai.yaml"
transformer_name    = "models/Diffusion_Transformer/lingbot-world-fast"
model_name          = "models/Diffusion_Transformer/lingbot-world-base-cam"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++".
# The distilled fast model is trained on the FlowUniPC schedule (int64 timesteps,
# shift applied inside set_timesteps); the tuned ``timesteps_index`` only matches
# that grid, so "Flow_Unipc" is required to reproduce the reference quality.
sampler_name        = "Flow_Unipc"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics of the
# distilled few-step flow-matching schedule. The official lingbot-world fast
# model is calibrated with sample_shift=10.0 (wan_i2v_A14B.py); generate_fast.py
# passes cfg.sample_shift (=10.0), NOT the generate() signature default of 5.0.
# Using 5.0 here builds the wrong sigma grid ([999,957,899,702] instead of
# [999,978,947,825]), so every renoise step is off-distribution and the decoded
# frames look grainy/painterly. 10.0 reproduces the reference quality.
shift               = 10.0
# stochastic_sampling=True selects the native calibrated few-step schedule
# (fixed-index FlowUniPC grid the fast model was distilled on) — the correct
# path. False falls back to the generic scheduler dispatch (off-distribution
# for the fast weights).
stochastic_sampling = True

# Load pretrained transformer weights (optional override).
transformer_path    = None
vae_path            = None
# LoRA path (optional). The fast model is a single distilled transformer, so only
# one LoRA is used here (no high-noise counterpart like the MoE predict_i2v.py).
lora_path           = None

# Camera control type - 'cam' (6-dim plücker).
control_type        = "cam"

# Self-Forcing causal inference config.
# `num_frame_per_block`: 3 = chunk-wise: 
#                        1 = frame-wise: 
num_frame_per_block = 3
# Local attention window size (-1 for global attention).
local_attn_size     = -1
sink_size           = 0

# Other params
# The reference derives the output resolution from a pixel-area budget and the
# input image aspect ratio (image2video_fast.py), rather than forcing a fixed
# size. ``sample_size`` here is only used as the area budget (max_area =
# sample_size[0] * sample_size[1]); the actual height/width are computed per
# image below so the aspect ratio is preserved (no stretch).
sample_size         = [480, 832]
video_length        = 81
fps                 = 16

# Use torch.float16 if GPU does not support torch.bfloat16
# some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16

# Camera trajectory (poses.npy / intrinsics.npy) + reference image + prompt.
action_path             = "asset/lingbot_demo"
validation_image_start  = "asset/lingbot_demo/image.jpg"

# prompts
prompt              = "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls."
# negative_prompt / guidance_scale mirror predict_i2v.py. The distilled few-step
# model is trained WITHOUT classifier-free guidance, so guidance_scale is left at
# 1.0 (CFG disabled, native behavior). Set it > 1.0 to enable CFG with the
# negative prompt (off-distribution, may degrade quality).
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
guidance_scale      = 1.0
seed                = 43
num_inference_steps = 4
lora_weight         = 0.55
save_path           = "samples/lingbot-world-i2v-fast"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

# Load transformer with causal inference + camera control support.
transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
transformer_additional_kwargs['local_attn_size'] = local_attn_size
transformer_additional_kwargs['sink_size'] = sink_size
transformer_additional_kwargs['control_type'] = control_type
transformer_additional_kwargs['cross_attn_type'] = 'cross_attn'

transformer = WanTransformer3DModel_LingbotWorldFast.from_pretrained(
    transformer_name,
    transformer_additional_kwargs=transformer_additional_kwargs,
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
scheduler_kwargs = OmegaConf.to_container(config['scheduler_kwargs'])
# The lingbot-world fast reference builds FlowUniPCMultistepScheduler with shift=1
# and applies the real shift only inside set_timesteps. The shared config carries
# shift=5.0 (used by other pipelines), which would double-shift the sigma grid
# here, so pin the constructor shift to 1 for the UniPC path; the runtime shift
# is forwarded to set_timesteps by the pipeline instead.
if Chosen_Scheduler is FlowUniPCMultistepScheduler:
    scheduler_kwargs['shift'] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, scheduler_kwargs)
)

# Get Pipeline
pipeline = WanFunLingbotWorldFastPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_group_offload":
    register_auto_device_hook(pipeline.transformer)
    safe_enable_group_offload(pipeline, onload_device=device, offload_device="cpu", offload_type="leaf_level", use_stream=True)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1

    image = Image.open(validation_image_start).convert("RGB")

    # Aspect-preserving resolution from the area budget (matches the reference):
    # keep width*height close to max_area while snapping to the VAE stride *
    # patch-size grid, so the input aspect ratio is preserved instead of forcing
    # a fixed sample_size (which would stretch a 16:9 frame).
    max_area = sample_size[0] * sample_size[1]
    vae_stride = vae.config.spatial_compression_ratio
    patch = pipeline.transformer.config.patch_size[1]
    aspect_ratio = image.height / image.width
    lat_h = round(np.sqrt(max_area * aspect_ratio) // vae_stride // patch * patch)
    lat_w = round(np.sqrt(max_area / aspect_ratio) // vae_stride // patch * patch)
    height = lat_h * vae_stride
    width = lat_w * vae_stride

    sample = pipeline(
        prompt,
        image       = image,
        negative_prompt = negative_prompt,
        action_path = action_path,
        control_type = control_type,
        height      = height,
        width       = width,
        num_frames  = video_length,
        num_frame_per_block = num_frame_per_block,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
        stochastic_sampling = stochastic_sampling,
        shift       = shift,
        generator   = generator,
    ).videos

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    video_path = os.path.join(save_path, prefix + ".mp4")
    save_videos_grid(sample, video_path, fps=fps)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results()
