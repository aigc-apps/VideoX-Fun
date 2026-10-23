import os
import re
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
                               FastVideoWanTransformer3DModel, WanT5EncoderModel)
from videox_fun.pipeline import WanPipeline
from videox_fun.utils import (FlowDPMSolverMultistepScheduler,
                              FlowUniPCMultistepScheduler,
                              apply_gpu_memory_mode, filter_kwargs, merge_lora,
                              save_videos_grid, unmerge_lora)

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
# NOTE: the VSA sparse kernel has no sequence-parallel path, so a FastWan student runs single-GPU
# only; keep ulysses_degree = ring_degree = 1 (enable_multi_gpus_inference() raises otherwise).
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
# Base model (Wan-AI/Wan2.1-T2V-1.3B): provides config.json + the dense backbone + VAE + T5.
model_name          = "models/Diffusion_Transformer/Wan2.1-T2V-1.3B"

# ---- FastVideo VSA sparse-student config --------------------------------------------------
# Official released checkpoint: FastVideo/FastWan2.1-T2V-1.3B-Diffusers (VSA sparse attention +
# DMD few-step distillation). student_topk is the keep-ratio of key blocks per query and MUST
# equal the value the checkpoint was trained at: topk=0.20 -> 80% sparsity (--VSA-sparsity 0.8).
student_topk        = 0.20

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++".
# FastWan ships UniPCMultistepScheduler(flow_shift=3.0); "Flow_Unipc" + shift=3.0 mirrors it and
# keeps every denoising step at a high noise level (DMD-style), unlike "Flow" whose last step ~0.
sampler_name        = "Flow_Unipc"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 3.0 is the official
# flow_shift and the recommended value for 480p.
shift               = 3.0

# Official released checkpoint (diffusers layout: transformer/diffusion_pytorch_model.safetensors,
# diffusers key naming + the VSA to_gate_compress gate). Download:
#   HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
#     FastVideo/FastWan2.1-T2V-1.3B-Diffusers --local-dir <dir>
transformer_path    = "models/Diffusion_Transformer/FastWan2.1-T2V-1.3B-Diffusers/transformer/diffusion_pytorch_model.safetensors"
vae_path            = None
lora_path           = None

# Other params
sample_size         = [480, 848]
video_length        = 81
fps                 = 16

# Use torch.float16 if GPU does not support torch.bfloat16
# Some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
prompt              = "A cinematic aerial shot of a snow-covered mountain village at dawn, warm light spilling from wooden windows, smoke curling from the chimneys while a river of mist drifts through the valley below."
negative_prompt     = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
guidance_scale      = 1.0   # the DMD student is CFG-free
seed                = 1024
num_inference_steps = 3
lora_weight         = 0.55
save_path           = "samples/fastvideo-wan2.1-official"


def remap_fastwan_diffusers_to_vxf(state_dict):
    """Official FastWan diffusers transformer -> VideoX-Fun civitai WanTransformer3DModel layout.

    FastWan ships diffusers naming (attn1/attn2, ffn.net, condition_embedder, proj_out,
    scale_shift_table) plus the VSA gate ``to_gate_compress``; VideoX-Fun keeps the native Wan
    civitai naming (self_attn/cross_attn, ffn.0/2, time_/text_embedding, head.head, modulation)
    and nests the gate under ``self_attn.attn_op``. Only the cross-attn norm is affine, so the
    single diffusers ``norm2`` maps to civitai ``norm3``; ``patch_embedding`` is already 5D and
    passes through. Turns the 885 released keys into an exact 1:1 match (825 backbone + 30x2 gate).
    """
    TOP = {
        "condition_embedder.time_embedder.linear_1": "time_embedding.0",
        "condition_embedder.time_embedder.linear_2": "time_embedding.2",
        "condition_embedder.time_proj":              "time_projection.1",
        "condition_embedder.text_embedder.linear_1": "text_embedding.0",
        "condition_embedder.text_embedder.linear_2": "text_embedding.2",
        "proj_out":                                  "head.head",
        "scale_shift_table":                         "head.modulation",
    }
    out = {}
    for k, v in state_dict.items():
        m = re.match(r"blocks\.(\d+)\.(.+)$", k)
        if m:
            blk, rest = m.group(1), m.group(2)
            rest = rest.replace("attn1.", "self_attn.").replace("attn2.", "cross_attn.")
            rest = (rest.replace(".to_out.0", ".o").replace(".to_q", ".q")
                        .replace(".to_k", ".k").replace(".to_v", ".v"))
            rest = rest.replace("ffn.net.0.proj", "ffn.0").replace("ffn.net.2", "ffn.2")
            if rest.startswith("norm2."):
                rest = "norm3." + rest[len("norm2."):]
            if rest == "scale_shift_table":
                rest = "modulation"
            if rest.startswith("to_gate_compress"):
                rest = "self_attn.attn_op." + rest
            nk = "blocks." + blk + "." + rest
        else:
            nk = k
            for src, dst in TOP.items():
                if k == src or k.startswith(src + "."):
                    nk = dst + k[len(src):]
                    break
        out[nk] = v
    return out


device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

# Build the sparse student from the base Wan2.1 config, then REBUILD the VSA kernels at the
# released keep-ratio BEFORE loading weights: set_student_topk recreates attn_op (resetting
# to_gate_compress to its zero init), so the trained checkpoint values must be loaded after.
transformer = FastVideoWanTransformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
transformer.set_student_topk(student_topk)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if isinstance(state_dict, dict) and "state_dict" in state_dict else state_dict

    # Official diffusers release -> VideoX-Fun civitai key layout (route the VSA gate under attn_op).
    state_dict = remap_fastwan_diffusers_to_vxf(state_dict)

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    attn_op_missing = [k for k in m if "attn_op." in k]
    if attn_op_missing:
        # A VSA student MUST provide every attn_op param; missing ones mean the compressed branch
        # would run on freshly-init (zero) gate values. Fail loudly instead of silently.
        raise RuntimeError(
            f"{len(attn_op_missing)} VSA attn_op keys are missing from the checkpoint "
            f"(student_topk={student_topk}); e.g. {attn_op_missing[:4]}. Check the remap / topk.")
    if u:
        print(f"WARNING: {len(u)} unexpected keys ignored; e.g. {list(u)[:4]}")
    # For the official FastWan2.1-T2V-1.3B checkpoint this is expected to be 0 missing / 0 unexpected.

# Get Vae
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file
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
Chosen_Scheduler = {
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
pipeline = WanPipeline(
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

# Quantize (when the mode carries an "_and_<quant>" suffix) and then place the pipeline.
apply_gpu_memory_mode(pipeline, GPU_memory_mode, device, weight_dtype)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

    if enable_riflex:
        pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)

    sample = pipeline(
        prompt,
        num_frames = video_length,
        negative_prompt = negative_prompt,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,
        shift = shift,
    ).videos

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
        print(f"Saved image to: {video_path}")
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)
        print(f"Saved video to: {video_path}")

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results()
