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
from videox_fun.utils import (register_auto_device_hook,
                              safe_enable_group_offload)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_video_to_video_latent,
                                    save_videos_grid)

# Flex-Forcing (arXiv 2607.03509) 4.2: any-order / any-timestep editing.
#
# A strictly causal rollout can only ever condition on the past, so a finished
# clip cannot be revisited. Flex-Forcing can: the whole clip is committed to the
# KV cache as clean context, one span is re-noised to a *low-level refinement*
# timestep and denoised with the attention window widened to the entire clip, so
# the edited span conditions on clean tokens from both its past and its future
# while the high-level planning timesteps (and every other frame) stay untouched.
#
# Run predict_t2v.py first, or point `input_video_path` at any clip.

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, model_group_offload, sequential_cpu_offload].
GPU_memory_mode     = "model_full_load"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = True
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
# model path
model_name          = "models/Diffusion_Transformer/Wan2.1-T2V-1.3B"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow"
shift               = 5 
stochastic_sampling = True

# Load pretrained model if need
transformer_path    = None
vae_path            = None
lora_path           = None

# Other params
# The paper evaluates 5 s clips: 81 pixel frames = 21 latent frames at 832x432.
sample_size         = [432, 832]
video_length        = 81
fps                 = 16

# --- 4.2 editing config ----------------------------------------------------
# Clip to edit. `None` generates one with this same pipeline first (the
# generate -> edit round trip the paper demonstrates); otherwise an official
# demo clip under datasets/X-Fun-Videos-Demo/ is loaded.
input_video_path    = None
# Half-open range of **latent** frames to regenerate, e.g. (8, 15) for the
# middle third of a 21-latent-frame clip. `None` edits the whole clip. A middle
# span is the interesting case: it needs clean context from the future, which a
# causal rollout does not have.
edit_span           = (8, 15)
# How many *trailing* steps of the schedule to run. Keep it small - editing at a
# planning timestep would restructure the clip instead of refining it. This is
# the "restrict editing to low-level refinement timesteps" half of 4.2.
edit_steps          = 1
# Granularity of the clean-context commit. `None` commits the whole clip in one
# bidirectional pass; an int or an explicit partition commits chunk by chunk in
# temporal order, which bounds the peak memory of long clips.
commit_chunk_spec   = 7
# Prompt driving the edit (the original prompt is a reasonable default).
edit_prompt         = None
# 3.3's K-Projection has no runtime knob at all: the model builds `diag_rank1`
# and applies it to the cached clean keys on every pass, edit included.

# --- Causal backbone (inherited from Self-Forcing) -------------------------
# Only used for the optional generation pass above.
chunk_spec              = None
denoise_mode            = "pyramid"
min_num_frame_per_block = 1
num_frame_per_block     = 3
# Any-order editing requires the full KV cache: the edited span must see clean
# tokens on both sides, which a rolling window may already have evicted.
local_attn_size         = -1
sink_size               = 0
independent_first_frame = False
context_noise           = 0.0

# Use torch.float16 if GPU does not support torch.bfloat16
weight_dtype        = torch.bfloat16
prompt              = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
guidance_scale      = 1.0
seed                = 43
num_inference_steps = 2
lora_weight         = 0.55
save_path           = "samples/wan-videos-flex-forcing-edit"

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

print(f"[Flex-Forcing 4.2] edit_span={edit_span} latent frames, edit_steps={edit_steps} "
      f"of {num_inference_steps}, commit_chunk_spec={commit_chunk_spec}")

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

    # 1. The clip to edit: either load it or generate it with the same
    #    pipeline. Both are [B, C, F, H, W] in [0, 1], which is what
    #    `edit_video` takes.
    if input_video_path is not None:
        video, _, _, _ = get_video_to_video_latent(
            input_video_path, video_length, sample_size, fps=fps)
        source = video.to(device=device, dtype=weight_dtype)
    else:
        torch.cuda.synchronize()
        start_time = time.time()
        source = pipeline(
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
            chunk_spec              = chunk_spec,
            denoise_mode            = denoise_mode,
            min_num_frame_per_block = min_num_frame_per_block,
        ).videos
        torch.cuda.synchronize()
        print(f"[Timing] generated {video_length} frames ({latent_frames} latent) "
              f"in {time.time() - start_time:.2f}s")

    # 2. Edit one span at the refinement timesteps only, conditioning on the
    #    clean context of the whole clip - past and future alike.
    torch.cuda.synchronize()
    start_time = time.time()
    sample = pipeline.edit_video(
        prompt          = edit_prompt if edit_prompt is not None else prompt,
        video           = source,
        edit_span       = edit_span,
        negative_prompt = negative_prompt,
        guidance_scale  = guidance_scale,
        num_inference_steps = num_inference_steps,
        edit_steps      = edit_steps,
        shift           = shift,
        context_noise   = context_noise,
        chunk_spec      = commit_chunk_spec,
        generator       = generator,
    ).videos
    torch.cuda.synchronize()
    print(f"[Timing] edited span {edit_span} in {time.time() - start_time:.2f}s")

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    if video_length == 1:
        image = sample[0, :, 0].transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        Image.fromarray(image).save(os.path.join(save_path, prefix + ".png"))
    else:
        # Keep the source next to the edit so the untouched frames and the
        # refined span can be compared directly.
        save_videos_grid(sample, os.path.join(save_path, prefix + "-edited.mp4"), fps=fps)
        if input_video_path is None:
            save_videos_grid(source, os.path.join(save_path, prefix + "-source.mp4"), fps=fps)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results()
