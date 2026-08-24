import os
import sys

import numpy as np
import torch
from PIL import Image
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
from videox_fun.pipeline.pipeline_lingbot_video import DEFAULT_NEGATIVE_PROMPT
from videox_fun.utils import (register_auto_device_hook,
                              safe_enable_group_offload)
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper)
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import save_videos_grid

from videox_fun.models.lingbot_video_rewriter import ensure_json_caption

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
# ulysses_degree * ring_degree.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False

# Config and model path
# model path
model_name          = "models/Diffusion_Transformer/lingbot-video-dense-1.3b/"
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
vae_path            = None
lora_path           = None

# Other params
sample_size         = [480, 832]
# video_length 1 generates a still image (t2i); videos must be 4n+1 frames.
video_length        = 81
fps                 = 24

# Use torch.float16 if GPU does not support torch.bfloat16
# some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
# prompts
# Write a plain natural-language prompt: it is ALWAYS rewritten into the
# structured JSON caption the DiT expects by the official prompt rewriter
# (EXPAND -> MAP, Qwen3.6-27B base + rewriter LoRA). Direct JSON/hand-written
# input is not a supported path; the rewrite result is cached under save_path
# so re-runs with the same prompt skip the rewrite.
prompt = (
    "A young musician sits on a weathered wooden stool in a sunlit rehearsal room, "
    "steadily strumming an acoustic guitar. Warm golden-hour light streams through "
    "tall windows, dust motes drifting slowly in the air; the wooden floor shows "
    "visible wear and the walls carry acoustic panels. The camera slowly orbits "
    "from a side profile to a frontal view at eye level, keeping the musician "
    "centered in the frame."
)
negative_prompt     = DEFAULT_NEGATIVE_PROMPT
guidance_scale      = 3.0
seed                = 43
num_inference_steps = 40
lora_weight         = 0.55
save_path           = "samples/lingbot-video-t2v"

# Rewrite the prompt before loading any generation model (the rewriter's 27B
# base VLM is freed right after, so it never coexists with the DiT on GPU).
prompt = ensure_json_caption(
    prompt, mode="t2v", duration=round(video_length / fps, 2),
    cache_file=os.path.join(save_path, "caption_cache.json"),
    base=rewriter_base_model, adapter=rewriter_lora_path,
)

device = set_multi_gpus_devices(ulysses_degree, ring_degree)


transformer = LingBotVideoTransformer3DModel.from_pretrained(
    os.path.join(model_name, "transformer"),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
# Re-apply the fp32-sensitive-module cast (norm / router / modulation stay fp32).
transformer = transformer.to(weight_dtype)

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

# Get Vae (diffusers-format QwenImage VAE, Wan-style 16ch causal VAE)
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

# Get Text encoder (Qwen3-VL)
text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    os.path.join(model_name, "text_encoder"),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow_Unipc": FlowUniPCMultistepScheduler,
}[sampler_name]
scheduler = Chosen_Scheduler.from_pretrained(
    model_name, 
    subfolder="scheduler"
)

# Get Pipeline
pipeline = LingBotVideoPipeline(
    transformer=transformer,
    vae=vae,
    text_encoder=text_encoder,
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

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

with torch.no_grad():
    video_length = int((video_length - 1) // pipeline.vae_scale_factor_temporal * pipeline.vae_scale_factor_temporal) + 1 if video_length != 1 else 1

    sample = pipeline(
        prompt, 
        num_frames = video_length,
        negative_prompt = negative_prompt,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guidance_scale = guidance_scale,
        shift       = shift,
        num_inference_steps = num_inference_steps,
    ).videos

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # count outputs only: caption_cache.json must not shift the index
    index = len([path for path in os.listdir(save_path) if path.endswith((".mp4", ".png"))]) + 1
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
