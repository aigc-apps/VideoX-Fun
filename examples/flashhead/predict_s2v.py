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
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                               FlashHeadTransformer3DModel, FlashHeadAudioEncoder)
from videox_fun.pipeline import FlashHeadPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils import (register_auto_device_hook,
                              safe_enable_group_offload)
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_latent, get_image, 
                                    get_video_to_video_latent,
                                    merge_video_audio, save_videos_grid)

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
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
GPU_memory_mode     = "model_full_load"
# Multi GPUs config
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
# Compile will give a speedup in fixed resolution and need a little GPU memory.
# The compile_dit is not compatible with sequential_cpu_offload.
compile_dit         = False

# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
# model path
# Please Download https://modelscope.cn/models/AI-ModelScope/wav2vec2-base-960h/summary
model_name          = "models/Diffusion_Transformer/SoulX-FlashHead-1_3B"
model_name_audio    = "models/Diffusion_Transformer/wav2vec2-base-960h"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow"
shift               = 5.0

# Load pretrained model if need
transformer_path    = None
vae_path            = None
lora_path           = None

# Other params
sample_size             = [512, 512]
segment_frame_length    = 33
fps                     = 25 

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
# The path of the reference image
ref_image               = "asset/9.png"
# The path of the audio 
audio_path              = "asset/talk.wav"

# Audio guidance scale (FlashHead does not use text encoder, only audio conditioning)
audio_guide_scale   = 1.0
seed                = 42
num_inference_steps = 4
lora_weight         = 0.55
save_path           = "samples/flashhead-videos"

# FlashHead specific parameters
max_frames_num          = 500
color_correction_strength = 1.0
use_apg                 = False
apg_momentum            = 0.5
apg_norm_threshold      = 1.0
audio_encode_mode       = "once"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

transformer = FlashHeadTransformer3DModel.from_pretrained(
    os.path.join(model_name, "Model_Pro", config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, "VAE_Wan/Wan2.1_VAE.pth"),
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

# Initialize FlashHead audio encoder for real-time audio encoding
# Uses Wav2Vec2Model (not Wav2Vec2ForCTC) matching original FlashHead implementation
audio_encoder = FlashHeadAudioEncoder(
    model_name_audio, "cpu"
)

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config['scheduler_kwargs']['shift'] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline (FlashHead does not use text encoder or clip image encoder)
pipeline = FlashHeadPipeline(
    transformer=transformer,
    vae=vae,
    scheduler=scheduler,
    audio_encoder=audio_encoder, 
)
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")

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
    # For FlashHead, (segment_frame_length - 1) must be divisible by 4
    segment_frame_length = (segment_frame_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio + 1 if segment_frame_length != 1 else 1
    latent_frames = (segment_frame_length - 1) // vae.config.temporal_compression_ratio + 1

    # Prepare ref_image latent for FlashHead (no clip_image needed)
    ref_image = get_image_latent(ref_image, sample_size=sample_size)
    
    sample = pipeline(
        segment_frame_length = segment_frame_length,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        audio_guide_scale = audio_guide_scale,
        num_inference_steps = num_inference_steps,

        ref_image = ref_image,
        audio_path = audio_path,
        audio_encode_mode = audio_encode_mode,
        shift = shift,
        fps = fps,
        max_frames_num = max_frames_num,
        color_correction_strength = color_correction_strength,
        use_apg = use_apg,
        apg_momentum = apg_momentum,
        apg_norm_threshold = apg_norm_threshold,
    ).videos

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    if sample.size()[2] == 1:
        video_path = os.path.join(save_path, prefix + ".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)
    
    merge_video_audio(video_path=video_path, audio_path=audio_path)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results()
