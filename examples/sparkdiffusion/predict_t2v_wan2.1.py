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
                               SparkWanTransformer3DModel, WanT5EncoderModel)
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
GPU_memory_mode     = "sequential_cpu_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
# NOTE: the sparse (RoLa) kernel only has an ulysses head-parallel path (no ring attention), so a
# sparse student requires ring_degree=1; put every GPU into ulysses_degree instead.
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

# ---- SparkDiffusion sparse-student config -------------------------------------------------
# Official released checkpoint: SparkWan2.1-T2V-1.3B-480P-0.90Sparsity (RoLA sparse attention +
# CrossDistill few-step distillation). student_topk is the keep-ratio of key blocks per query and
# MUST equal the value the checkpoint was trained at: topk=0.10 -> 90% sparsity. It selects how
# many key blocks the sparse kernel keeps, so a mismatch silently mis-attends.
student_topk        = 0.10

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++".
# The few-step distilled student uses "Flow" (FlowMatchEulerDiscreteScheduler).
sampler_name        = "Flow"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics.
shift               = 5

# Official released checkpoint (single .pth, bf16, flat 'net.'-prefixed state_dict).
# Download:
#   HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
#     alibabagroup/SparkWan2.1-T2V-1.3B-480P-0.90Sparsity --local-dir <dir>
transformer_path    = "models/Personalized_Model/SparkWan2.1-T2V-1.3B-480P-0.90Sparsity/SparkWan2.1-T2V-1.3B-480P-0.90Sparsity.pth"
vae_path            = None
lora_path           = None

# Other params
sample_size         = [480, 832]
video_length        = 81
fps                 = 16

# Use torch.float16 if GPU does not support torch.bfloat16
# Some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
prompt              = "一只棕色的狗摇着头，坐在舒适房间里的浅色沙发上。在狗的后面，架子上有一幅镶框的画，周围是粉红色的花朵。房间里柔和温暖的灯光营造出舒适的氛围。"
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
guidance_scale      = 1.0   # the CrossDistill student is CFG-free
seed                = 43
num_inference_steps = 4
lora_weight         = 0.55
save_path           = "samples/sparkdiffusion-wan2.1-official"


def remap_spark_ckpt_to_vxf(state_dict):
    """Official SparkDiffusion .pth -> VideoX-Fun SparkWanTransformer3DModel key layout.

    The released checkpoint is a flat dict whose keys are all prefixed with ``net.`` and whose
    RoLa sparse-attention params sit DIRECTLY on ``self_attn`` (proj_q / proj_k / gate_proj /
    gate_bias). VideoX-Fun keeps the same native Wan backbone naming but nests those four params
    under a ``self_attn.attn_op`` submodule. Two edits turn the 945 released keys into an exact
    1:1 match for the model (825 backbone + 30 layers x 4 RoLa = 120):
      1) drop the leading ``net.``
      2) ``blocks.N.self_attn.{proj_q,proj_k,gate_proj,gate_bias}`` -> ``...self_attn.attn_op.{...}``
    """
    ROLA = {"proj_q.weight", "proj_k.weight", "gate_proj.weight", "gate_bias"}
    out = {}
    for k, v in state_dict.items():
        nk = k[len("net."):] if k.startswith("net.") else k
        m = re.match(r"(blocks\.\d+\.self_attn\.)(.+)$", nk)
        if m and m.group(2) in ROLA:
            nk = m.group(1) + "attn_op." + m.group(2)
        out[nk] = v
    return out


device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

# Build the sparse student from the base Wan2.1 config, then REBUILD the sparse kernels at the
# released keep-ratio BEFORE loading weights. Call order matters: set_student_topk recreates
# attn_op (resetting the RoLa params to their init), so the trained checkpoint values must be
# loaded afterwards to overwrite them.
transformer = SparkWanTransformer3DModel.from_pretrained(
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

    # Official release -> VideoX-Fun key layout (strip 'net.', route RoLa params under attn_op).
    state_dict = remap_spark_ckpt_to_vxf(state_dict)

    # The released checkpoint flattens the Conv3d patch_embedding to 2D (out, in*kT*kH*kW);
    # restore the 5D Conv3d weight exactly like the turbodiffusion loader does.
    if len(state_dict["patch_embedding.weight"].size()) == 2:
        out_channels = state_dict["patch_embedding.weight"].shape[0]
        in_channels = state_dict["patch_embedding.weight"].shape[1] // transformer.config.patch_size[1] // transformer.config.patch_size[-2]
        state_dict["patch_embedding.weight"] = state_dict["patch_embedding.weight"].reshape(
            out_channels, in_channels, 1, transformer.config.patch_size[1], transformer.config.patch_size[2]
        )

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    attn_op_missing = [k for k in m if "attn_op." in k]
    if attn_op_missing:
        # A sparse student MUST provide every attn_op param; missing ones mean the RoLa branch
        # would run on freshly-init values -> garbage output. Fail loudly instead of silently.
        raise RuntimeError(
            f"{len(attn_op_missing)} sparse attn_op keys are missing from the checkpoint "
            f"(student_topk={student_topk}); e.g. {attn_op_missing[:4]}. Check the remap / topk.")
    if u:
        print(f"WARNING: {len(u)} unexpected keys ignored; e.g. {list(u)[:4]}")
    # For the official 0.90-sparsity checkpoint this is expected to be 0 missing / 0 unexpected.

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
    # The sparse (RoLa) kernel only supports ulysses head-parallel (no ring-attention path); a
    # sparse student under ring_degree>1 would silently mis-attend.
    if ring_degree > 1 and transformer_path is not None:
        raise ValueError(
            "SparkDiffusion sparse multi-GPU requires ring_degree=1; put every GPU into "
            "ulysses_degree instead (num_heads must divide ulysses_degree).")
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

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results()
