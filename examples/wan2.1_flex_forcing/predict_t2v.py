import os
import sys
import time

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
                               WanTransformer3DModel_FlexForcing)
from videox_fun.pipeline import WanFlexForcingPipeline
from videox_fun.pipeline.pipeline_wan_flex_forcing import \
    PAPER_CHUNK_CONFIGS
from videox_fun.utils import (register_auto_device_hook,
                              safe_enable_group_offload)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import filter_kwargs, save_videos_grid

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
GPU_memory_mode     = "model_full_load"
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
# [NOTE]: flex_attention block masks are rebuilt per partition, so compiling the
# blocks only pays off when the ladder (`denoise_mode`, `num_inference_steps`)
# is fixed.
compile_dit         = False

# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
# model path
model_name          = "models/Diffusion_Transformer/Wan2.1-T2V-1.3B"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
shift               = 5 
stochastic_sampling = True

# Load pretrained model if need
# Any Wan2.1 / CausVid / Self-Forcing checkpoint loads as-is: the Flex-Forcing
# backbone inherits every parameter name and only the new `flex_kproj.*` tensors
# are reported missing (they are identity-initialised, so step 0 is unchanged).
transformer_path    = None
vae_path            = None
lora_path           = None

# Other params
# The paper evaluates 5 s clips: 81 pixel frames = 21 latent frames at 832x432.
sample_size         = [432, 832]
video_length        = 81
fps                 = 16

# Flex-Forcing (arXiv 2607.03509) inference config
# --- Does the frame partition change with the noise level? -----------------
#   "fixed"   -> one partition held for every denoising step, i.e. the
#                block-major Self-Forcing schedule (use `num_frame_per_block`).
#   "pyramid" -> Flex-Forcing 3.2: level 0 plans the whole clip in one
#                bidirectional chunk, each further denoising step binary-splits
#                every chunk - coarse (planning) -> fine (refinement), one level
#                per step. The depth follows `num_inference_steps` automatically,
#                so there is no second number to keep in sync. Levels only ever
#                *add* boundaries, so a KV cache written at a coarse level stays
#                valid at a finer one.
# Needs `stochastic_sampling = True` (the buffered step between two levels *is*
# the schedule's own re-noising) and `local_attn_size = -1` below.
# An int instead pins a truncated pyramid of exactly that many levels; for 21
# latent frames (= 81 pixel frames) that ladder is
#   2 -> [[21], [11, 10]]    3 -> [[21], [11, 10], [6, 5, 5, 5]]
denoise_mode        = "pyramid"
# How far the splitting above goes: every chunk is binary-split until it is at
# or below this block size, so it decides how causal the finest level is.
#   1 -> leaves are single frames, fully causal
#   3 -> leaves stay 3-frame blocks (classic Self-Forcing granularity); the
#        ladder then converges early and later steps reuse its finest level.
min_num_frame_per_block = 1
# Advanced: the 3.1 partition itself can also be pinned on the pipeline call
# (`chunk_spec = "18-3" / "ar" / "uniform:3"`); the pyramid above does not need
# it, since it derives every level from the whole-clip level 0.
# 3.3's K-Projection (the noise-level aligned Pi_{t<-0} of the cached clean keys)
# is deliberately not configurable here: the model builds `diag_rank1` and applies
# it on every call, so there is nothing left to set.
# --- Causal backbone (inherited from Self-Forcing) -------------------------
# `num_frame_per_block` only takes effect once the pyramid is off; the rollout
# derives the block size from the partition itself otherwise. `context_noise`
# is the noise level the clean context is committed at.
num_frame_per_block     = 3
independent_first_frame = False
context_noise           = 0.0
# Local attention window size (-1 for global attention). Must stay -1 whenever
# the pyramid is used, because a rolling window evicts by `current_start` deltas
# that a splitting chunk moves backwards. For long videos the paper uses a
# 21-latent-frame window with a 3-frame sink: local_attn_size = 21, sink_size = 3.
local_attn_size         = -1
sink_size               = 0

# Use torch.float16 if GPU does not support torch.bfloat16
# Some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
prompt              = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
guidance_scale      = 1.0
seed                = 43
# The paper's 2-step DMD model denoises at [1000, 500]; 4 steps ([1000, 750,
# 500, 250]) suit the CCD checkpoint. `denoise_mode = "pyramid"` uses one ladder
# level per step here, so a deeper pyramid just wants more steps.
num_inference_steps = 4
lora_weight         = 0.55
save_path           = "samples/wan-videos-flex-forcing-t2v"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

# Load transformer with the Flex-Forcing backbone
transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
transformer_additional_kwargs['local_attn_size'] = local_attn_size
transformer_additional_kwargs['sink_size'] = sink_size

transformer = WanTransformer3DModel_FlexForcing.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
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
    state_dict = state_dict["generator_ema"] if "generator_ema" in state_dict else state_dict
    state_dict = state_dict["generator"] if "generator" in state_dict else state_dict
    if any("._fsdp_wrapped_module." in k for k in state_dict.keys()):
        state_dict = {k.replace("model._fsdp_wrapped_module.", "model.", 1) if k.startswith("model._fsdp_wrapped_module.") else k: v for k, v in state_dict.items()}
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict.items()}

    m, u = transformer.load_state_dict(state_dict, strict=False)
    # `flex_kproj.*` is expected to be missing when loading a Self-Forcing /
    # CausVid checkpoint that predates Flex-Forcing.
    other_missing = [k for k in m if "flex_kproj" not in k]
    print(f"missing keys: {len(m)} ({len(m) - len(other_missing)} of them flex_kproj), "
          f"unexpected keys: {len(u)}")

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

# Get Pipeline
pipeline = WanFlexForcingPipeline(
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

print(f"[Flex-Forcing] denoise_mode={denoise_mode}, "
      f"min_num_frame_per_block={min_num_frame_per_block}, "
      f"local_attn_size={local_attn_size}, sink_size={sink_size}")
print(f"[Flex-Forcing] partitions measured in the paper: "
      f"{[list(c) for c in PAPER_CHUNK_CONFIGS]}")

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

    torch.cuda.synchronize()
    start_time = time.time()
    sample = pipeline(
        prompt, 
        num_frames = video_length,
        negative_prompt = negative_prompt,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guidance_scale          = guidance_scale,
        num_inference_steps     = num_inference_steps,
        shift                   = shift,
        num_frame_per_block     = num_frame_per_block,
        independent_first_frame = independent_first_frame,
        context_noise           = context_noise,
        stochastic_sampling     = stochastic_sampling,
        denoise_mode            = denoise_mode,
        min_num_frame_per_block = min_num_frame_per_block,
    ).videos
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    print(f"[Timing] {video_length} frames ({latent_frames} latent) in {elapsed:.2f}s "
          f"({video_length / elapsed:.2f} frames/s)")
    if getattr(pipeline, "kv_cache_pos", None) is not None:
        kv_tokens = pipeline.kv_cache_pos[0]["k"].shape[1]
        kv_mib = sum(c["k"].numel() + c["v"].numel()
                     for c in pipeline.kv_cache_pos + pipeline.kv_cache_neg) \
            * pipeline.kv_cache_pos[0]["k"].element_size() / (1024 ** 2)
        print(f"[KV cache] {kv_tokens} tokens per layer per branch, total {kv_mib:.1f} MiB (pos+neg, all layers)")

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    if video_length == 1:
        video_path = os.path.join(save_path, prefix + ".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results()
