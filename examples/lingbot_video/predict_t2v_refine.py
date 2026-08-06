import gc
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLQwenImage,
                               LingBotVideoTransformer3DModel,
                               Qwen3VLForConditionalGeneration)
from videox_fun.pipeline import LingBotVideoPipeline
from videox_fun.pipeline.pipeline_lingbot_video import (DEFAULT_NEGATIVE_PROMPT,
                                                       prepare_refiner_latent)
from videox_fun.utils import (register_auto_device_hook,
                              safe_enable_group_offload)
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper)
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.utils import save_videos_grid

from videox_fun.models.lingbot_video_rewriter import ensure_json_caption

# Two-stage LingBot-Video t2v: the base DiT samples at a low resolution, then the
# "refiner" DiT re-noises the upsampled latent to sigma = refiner_t_thresh and
# denoises it at the target resolution.
#
# The two DiTs are loaded and freed one at a time, so a single GPU only ever holds
# one 30B transformer (the MoE base and refiner are ~60GB each in bfloat16).

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, model_group_offload].
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
GPU_memory_mode     = "model_cpu_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
# Sequence parallelism shards the video tokens across ranks and keeps the text tokens
# replicated, so the video token count (T/pF * H/16 * W/16) must be divisible by
# ulysses_degree * ring_degree, at both the base and the refiner resolution.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False

# Config and model path
# model path
# The refiner ships only with the MoE 30B-A3B model, as its "refiner" subfolder.
model_name          = "models/Diffusion_Transformer/lingbot-video-moe-30b-a3b"
refiner_model_name  = model_name
# Subfolders of the base and refiner DiT inside the model root.
transformer_subpath = "transformer"
refiner_subpath     = "refiner"
# Rewriter weights: the base VLM and the rewriter LoRA used to rewrite the
# plain prompt into the structured JSON caption the DiT expects.
rewriter_base_model = "models/Diffusion_Transformer/Qwen3.6-27B"
rewriter_lora_path  = "models/Diffusion_Transformer/lingbot-video-rewriter-lora"

# Only "Flow_Unipc" is supported: LingBot-Video ships and was trained with FlowUniPCMultistepScheduler.
sampler_name        = "Flow_Unipc"
# Flow shift. 3.0 is the officially recommended value for both dense and MoE models.
shift               = 3.0

# Load pretrained model if need
transformer_path    = None
refiner_path        = None
vae_path            = None

# Base stage params. video_length must be 1 or 4n+1; 121 frames is 5s at 24 fps.
sample_size         = [480, 832]
video_length        = 81
fps                 = 24

# Use torch.float16 if GPU does not support torch.bfloat16
# some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
# prompts
# Write a plain natural-language prompt: it is ALWAYS rewritten into the
# structured JSON caption the DiT expects by the official prompt rewriter
# (EXPAND -> MAP, Qwen3.6-27B base + rewriter LoRA). Direct JSON/hand-written
# input is not a supported path; the rewrite result is cached under save_path.
prompt = (
    "A young musician sits on a weathered wooden stool in a sunlit rehearsal room, "
    "steadily strumming an acoustic guitar. Warm golden-hour light streams through "
    "tall windows, dust motes drifting slowly in the air. The camera slowly orbits "
    "from a side profile to a frontal view at eye level, keeping the musician "
    "centered in the frame."
)
negative_prompt     = DEFAULT_NEGATIVE_PROMPT
guidance_scale      = 3.0
seed                = 43
num_inference_steps = 40
save_path           = "samples/lingbot-video-t2v-refine"

# Refiner stage params (official defaults). The refiner attends over the full
# high-resolution latent, so 1088x1920 is heavy on a single GPU: prefer fewer
# frames there, or shard the DiT across GPUs.
refiner_sample_size     = [1088, 1920]
refiner_steps           = 8
refiner_guidance_scale  = 3.0
refiner_shift           = 3.0
# Re-noise level: the refiner only walks the schedule from this sigma down to 0.
refiner_t_thresh        = 0.85
# Extra low-noise steps appended after the truncated schedule.
refiner_sigma_tail_steps = 2

# Rewrite the prompt before loading any generation model (the rewriter's 27B
# base VLM is freed right after, so it never coexists with the DiT on GPU).
prompt = ensure_json_caption(
    prompt, mode="t2v", duration=round(video_length / fps, 2),
    cache_file=os.path.join(save_path, "caption_cache.json"),
    base=rewriter_base_model, adapter=rewriter_lora_path,
)

device = set_multi_gpus_devices(ulysses_degree, ring_degree)


def load_transformer(root, subpath, checkpoint_path):
    transformer = LingBotVideoTransformer3DModel.from_pretrained(
        os.path.join(root, subpath),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    # Re-apply the fp32-sensitive-module cast (norm / router / modulation stay fp32).
    transformer = transformer.to(weight_dtype)

    if checkpoint_path is not None:
        print(f"From checkpoint: {checkpoint_path}")
        if checkpoint_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    return transformer

def next_index():
    if not os.path.exists(save_path):
        return 1
    return len([path for path in os.listdir(save_path) if path.endswith("_base.mp4")]) + 1

def save_results(sample, index, prefix, save_fps):
    # Both stages of a run share an index: 00000001_base.mp4 / 00000001_refined.mp4.
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    video_path = os.path.join(save_path, f"{str(index).zfill(8)}_{prefix}.mp4")
    save_videos_grid(sample, video_path, fps=save_fps)
    return video_path

# Get Vae (diffusers-format QwenImage VAE, Wan-style 16ch causal VAE), shared by both stages
vae = AutoencoderKLQwenImage.from_pretrained(
    model_name,
    subfolder="vae",
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

# Get Processor (Qwen3-VL tokenizer + image processor)
processor = AutoProcessor.from_pretrained(
    os.path.join(model_name, "processor"),
)

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow_Unipc": FlowUniPCMultistepScheduler,
}[sampler_name]
scheduler = Chosen_Scheduler.from_pretrained(
    model_name, 
    subfolder="scheduler"
)

# Stage 0: encode the prompts once. Both stages condition on the same text, so the
# text encoder is released before either 30B DiT is loaded.
text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    os.path.join(model_name, "text_encoder"),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
encode_pipeline = LingBotVideoPipeline(
    transformer=None,
    vae=vae,
    text_encoder=text_encoder,
    processor=processor,
    scheduler=scheduler,
)
encode_pipeline.text_encoder.to(device)
with torch.no_grad():
    prompt_embeds, prompt_mask = encode_pipeline.encode_prompt(prompt, device=device)
    negative_prompt_embeds, negative_prompt_mask = encode_pipeline.encode_prompt(negative_prompt, device=device)
del encode_pipeline, text_encoder
gc.collect()
torch.cuda.empty_cache()

# Stage 1: base sampling at sample_size
transformer = load_transformer(model_name, transformer_subpath, transformer_path)
pipeline = LingBotVideoPipeline(
    transformer=transformer,
    vae=vae,
    text_encoder=None,
    processor=processor,
    scheduler=scheduler,
)
if GPU_memory_mode == "model_group_offload":
    register_auto_device_hook(pipeline.transformer)
    safe_enable_group_offload(pipeline, onload_device=device, offload_device="cpu", offload_type="leaf_level", use_stream=True)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["time_embedder", "time_modulation", "text_embedder", "norm", "router", "scale_shift_table", "proj_out"], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["time_embedder", "time_modulation", "text_embedder", "norm", "router", "scale_shift_table", "proj_out"], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")

generator = torch.Generator(device=device).manual_seed(seed)
with torch.no_grad():
    video_length = int((video_length - 1) // pipeline.vae_scale_factor_temporal * pipeline.vae_scale_factor_temporal) + 1 if video_length != 1 else 1

    base_sample = pipeline(
        prompt, 
        num_frames = video_length,
        prompt_embeds = prompt_embeds,
        prompt_mask = prompt_mask,
        negative_prompt_embeds = negative_prompt_embeds,
        negative_prompt_mask = negative_prompt_mask,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guidance_scale = guidance_scale,
        shift       = shift,
        num_inference_steps = num_inference_steps,
    ).videos

save_index = next_index()
if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        base_path = save_results(base_sample, save_index, "base", fps)
else:
    base_path = save_results(base_sample, save_index, "base", fps)

del pipeline, transformer
gc.collect()
torch.cuda.empty_cache()

# Stage 2: refinement at refiner_sample_size. Unlike the reference runner, the base
# frames are refined in memory instead of being re-read from the saved mp4, which
# skips a lossy encode/decode round trip.
refiner = load_transformer(refiner_model_name, refiner_subpath, refiner_path)
refiner_pipeline = LingBotVideoPipeline(
    transformer=refiner,
    vae=vae,
    text_encoder=None,
    processor=processor,
    scheduler=scheduler,
)
if GPU_memory_mode == "model_group_offload":
    register_auto_device_hook(refiner_pipeline.transformer)
    safe_enable_group_offload(refiner_pipeline, onload_device=device, offload_device="cpu", offload_type="leaf_level", use_stream=True)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(refiner, exclude_module_name=["time_embedder", "time_modulation", "text_embedder", "norm", "router", "scale_shift_table", "proj_out"], device=device)
    convert_weight_dtype_wrapper(refiner, weight_dtype)
    refiner_pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    refiner_pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(refiner, exclude_module_name=["time_embedder", "time_modulation", "text_embedder", "norm", "router", "scale_shift_table", "proj_out"], device=device)
    convert_weight_dtype_wrapper(refiner, weight_dtype)
    refiner_pipeline.to(device=device)
else:
    refiner_pipeline.to(device=device)

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    refiner.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        refiner_pipeline.transformer = shard_fn(refiner_pipeline.transformer)
        print("Add FSDP DIT")

refiner_generator = torch.Generator(device=device).manual_seed(seed)
with torch.no_grad():
    # video: [B, C, T, H, W] in [0, 1]
    bsz, channels, frames, _height, _width = base_sample.shape
    flat = base_sample.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channels, _height, _width)
    resized = F.interpolate(flat, size=(refiner_sample_size[0], refiner_sample_size[1]), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    lowres_video = resized.reshape(bsz, frames, channels, refiner_sample_size[0], refiner_sample_size[1]).permute(0, 2, 1, 3, 4).contiguous()
    x_up = refiner_pipeline.encode_video_latent(lowres_video, generator=refiner_generator)
    noise = torch.randn(x_up.shape, device=x_up.device, dtype=x_up.dtype, generator=refiner_generator)
    initial_latent = prepare_refiner_latent(x_up, noise, refiner_t_thresh)
    del lowres_video, x_up, noise

    # The refiner's unconditional branch uses zeroed conditions rather than the
    # negative prompt (null_cond_clone_zero in the reference implementation).
    refiner_sample = refiner_pipeline(
        prompt, 
        num_frames = video_length,
        prompt_embeds = prompt_embeds,
        prompt_mask = prompt_mask,
        negative_prompt_embeds = torch.zeros_like(prompt_embeds),
        negative_prompt_mask = prompt_mask.clone(),
        height      = refiner_sample_size[0],
        width       = refiner_sample_size[1],
        latents     = initial_latent,
        generator   = refiner_generator,
        guidance_scale = refiner_guidance_scale,
        shift       = refiner_shift,
        num_inference_steps = refiner_steps,
        t_thresh    = refiner_t_thresh,
        refiner_sigma_tail_steps = refiner_sigma_tail_steps,
    ).videos

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        refined_path = save_results(refiner_sample, save_index, "refined", fps)
        print(f"base: {base_path}\nrefined: {refined_path}")
else:
    refined_path = save_results(refiner_sample, save_index, "refined", fps)
    print(f"base: {base_path}\nrefined: {refined_path}")
