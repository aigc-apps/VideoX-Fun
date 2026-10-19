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
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                               AutoTokenizer, SparkWan2_2Transformer3DModel,
                               Wan2_2Transformer3DModel, WanT5EncoderModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2Pipeline
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

# TeaCache config
# Kept off: TeaCache / cfg-skip skip whole denoising steps, which does not fit the few-step
# CFG-free sparse student.
enable_teacache     = False
# Recommended to be set between 0.05 and 0.30. A larger threshold can cache more steps, speeding up the inference process, 
# but it may cause slight differences between the generated content and the original content.
# # --------------------------------------------------------------------------------------------------- #
# | Model Name          | threshold | Model Name          | threshold |
# | Wan2.2-T2V-A14B     | 0.10~0.15 | Wan2.2-I2V-A14B     | 0.15~0.20 |
# # --------------------------------------------------------------------------------------------------- #
teacache_threshold  = 0.10
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload    = False

# Skip some cfg steps in inference
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# Config and model path
config_path         = "config/wan2.2/wan_civitai_t2v.yaml"
# Base model (Wan-AI/Wan2.2-T2V-A14B): provides config.json + dense low/high experts + VAE + T5.
model_name          = "models/Diffusion_Transformer/Wan2.2-T2V-A14B"

# ---- SparkDiffusion sparse-student config -------------------------------------------------
# Official released checkpoints: SparkWan2.2-T2V-14B-480P-0.95Sparsity-{Low,High}.pth (RoLA sparse
# attention + CrossDistill few-step distillation, dual-expert MoE). student_topk is the keep-ratio
# of key blocks per query and MUST equal the value the checkpoints were trained at: topk=0.05 ->
# 95% sparsity. A mismatch silently mis-attends.
student_topk        = 0.05

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++".
# The few-step distilled student uses "Flow" (FlowMatchEulerDiscreteScheduler).
sampler_name        = "Flow"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics.
shift               = 12

# Official released checkpoints (single .pth per expert, bf16, flat 'net.'-prefixed state_dict).
# transformer_path -> low-noise expert, transformer_high_path -> high-noise expert.
# Download:
#   HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
#     alibabagroup/SparkWan2.2-T2V-14B-480P-0.95Sparsity --local-dir <dir>
transformer_path        = "models/Personalized_Model/SparkWan2.2-T2V-14B-480P-0.95Sparsity/SparkWan2.2-T2V-14B-480P-0.95Sparsity-Low.pth"
transformer_high_path   = "models/Personalized_Model/SparkWan2.2-T2V-14B-480P-0.95Sparsity/SparkWan2.2-T2V-14B-480P-0.95Sparsity-High.pth"
vae_path                = None
lora_path               = None
lora_high_path          = None

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
lora_high_weight    = 0.55
save_path           = "samples/sparkdiffusion-wan2.2-official"


def remap_spark_ckpt_to_vxf(state_dict):
    """Official SparkDiffusion .pth -> VideoX-Fun SparkWan2_2Transformer3DModel key layout.

    The released checkpoint is a flat dict whose keys are all prefixed with ``net.`` and whose
    RoLa sparse-attention params sit DIRECTLY on ``self_attn`` (proj_q / proj_k / gate_proj /
    gate_bias). VideoX-Fun keeps the same native Wan backbone naming but nests those four params
    under a ``self_attn.attn_op`` submodule. Two edits make the keys match 1:1:
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


def load_spark_ckpt(path):
    """Load an official .pth and return it in VideoX-Fun key layout (remap + patch reshape)."""
    if path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(path)
    else:
        state_dict = torch.load(path, map_location="cpu")
    state_dict = state_dict["state_dict"] if isinstance(state_dict, dict) and "state_dict" in state_dict else state_dict
    return remap_spark_ckpt_to_vxf(state_dict)


device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)
boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)

# Build one transformer per MoE expert, then REBUILD the sparse kernels at the released keep-ratio
# BEFORE loading weights. Call order matters: set_student_topk recreates attn_op (resetting the
# RoLa params to their init), so the trained checkpoint values must be loaded afterwards.
low_subpath  = config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')
high_subpath = config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')

if transformer_path is not None:
    transformer = SparkWan2_2Transformer3DModel.from_pretrained(
        os.path.join(model_name, low_subpath),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    transformer.set_student_topk(student_topk)
else:
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(model_name, low_subpath),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
    if transformer_high_path is not None:
        transformer_2 = SparkWan2_2Transformer3DModel.from_pretrained(
            os.path.join(model_name, high_subpath),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        transformer_2.set_student_topk(student_topk)
    else:
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(model_name, high_subpath),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
else:
    transformer_2 = None

# Load the per-expert official checkpoints AFTER set_student_topk so the trained RoLa params
# overwrite the freshly-init ones; a missing attn_op.* key means the remap / topk is wrong.
if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    state_dict = load_spark_ckpt(transformer_path)
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
        raise RuntimeError(
            f"{len(attn_op_missing)} sparse attn_op keys missing on the low-noise expert "
            f"(student_topk={student_topk}); e.g. {attn_op_missing[:4]}. Check the remap / topk.")
    if u:
        print(f"WARNING: {len(u)} unexpected keys ignored; e.g. {list(u)[:4]}")

if transformer_2 is not None:
    if transformer_high_path is not None:
        print(f"From checkpoint: {transformer_high_path}")
        state_dict = load_spark_ckpt(transformer_high_path)
        if len(state_dict["patch_embedding.weight"].size()) == 2:
            out_channels = state_dict["patch_embedding.weight"].shape[0]
            in_channels = state_dict["patch_embedding.weight"].shape[1] // transformer_2.config.patch_size[1] // transformer_2.config.patch_size[-2]
            state_dict["patch_embedding.weight"] = state_dict["patch_embedding.weight"].reshape(
                out_channels, in_channels, 1, transformer_2.config.patch_size[1], transformer_2.config.patch_size[2]
            )
        m, u = transformer_2.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        attn_op_missing = [k for k in m if "attn_op." in k]
        if attn_op_missing:
            raise RuntimeError(
                f"{len(attn_op_missing)} sparse attn_op keys missing on the high-noise expert "
                f"(student_topk={student_topk}); e.g. {attn_op_missing[:4]}. Check the remap / topk.")
        if u:
            print(f"WARNING: {len(u)} unexpected keys ignored; e.g. {list(u)[:4]}")

# Get Vae
Chosen_AutoencoderKL = {
    "AutoencoderKLWan": AutoencoderKLWan,
    "AutoencoderKLWan3_8": AutoencoderKLWan3_8
}[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
vae = Chosen_AutoencoderKL.from_pretrained(
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
pipeline = Wan2_2Pipeline(
    transformer=transformer,
    transformer_2=transformer_2,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)
if ulysses_degree > 1 or ring_degree > 1:
    # The sparse (RoLa) kernel only supports ulysses head-parallel (no ring-attention path); a
    # sparse expert under ring_degree>1 would silently mis-attend.
    if ring_degree > 1 and (transformer_path is not None or transformer_high_path is not None):
        raise ValueError(
            "SparkDiffusion sparse multi-GPU requires ring_degree=1; put every GPU into "
            "ulysses_degree instead (num_heads must divide ulysses_degree).")
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if transformer_2 is not None:
        transformer_2.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        if transformer_2 is not None:
            pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    if transformer_2 is not None:
        for i in range(len(pipeline.transformer_2.blocks)):
            pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])
    print("Add Compile")

# Quantize (when the mode carries an "_and_<quant>" suffix) and then place the pipeline.
apply_gpu_memory_mode(pipeline, GPU_memory_mode, device, weight_dtype)

coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )
    if transformer_2 is not None:
        pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
    if transformer_2 is not None:
        pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
if lora_high_path is not None and transformer_2 is not None:
    pipeline = merge_lora(pipeline, lora_high_path, lora_high_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")

with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

    if enable_riflex:
        pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
        if transformer_2 is not None:
            pipeline.transformer_2.enable_riflex(k = riflex_k, L_test = latent_frames)

    sample = pipeline(
        prompt,
        num_frames = video_length,
        negative_prompt = negative_prompt,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,
        boundary = boundary,
        shift = shift,
    ).videos

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
if lora_high_path is not None and transformer_2 is not None:
    pipeline = unmerge_lora(pipeline, lora_high_path, lora_high_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")

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
