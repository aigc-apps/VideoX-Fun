import os
import sys

import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLQwenImage21,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor,
                               QwenImage21ControlTransformer2DModel)
from videox_fun.pipeline import QwenImage21ControlPipeline
from videox_fun.utils import (register_auto_device_hook,
                              safe_enable_group_offload)
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import get_image_latent

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
GPU_memory_mode     = "model_group_offload"
# Multi GPUs config
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = False
# Compile will give a speedup in fixed resolution and need a little GPU memory.
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# Config path (control_layers / control_in_dim live here and must match the trained adapter)
config_path         = "config/qwenimage21/qwenimage21_control.yaml"
# model path
model_name          = "models/Diffusion_Transformer/Qwen-Image-2.1"

# Choose the sampler. Qwen-Image 2.1 is a flow-matching model sampled with the Euler discrete scheduler.
sampler_name        = "Flow"

# Load pretrained model if need
transformer_path    = "models/Personalized_Model/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"
vae_path            = None
lora_path           = None

# Other params
sample_size         = [1728, 992]
# Cache the text and condition-image keys/values after the first denoising step. Valid because the
# transformer modulates those tokens from t = 0, making their activations step-independent.
use_kv_cache        = True

# Use torch.float16 if GPU does not support torch.bfloat16
# Some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
control_image       = "asset/pose.jpg"
control_context_scale   = 1.0

# Please use as detailed a prompt as possible to describe the object that needs to be generated.
prompt              = "画面中央是一位年轻女孩，她拥有一头令人印象深刻的亮紫色长发，发丝在海风中轻盈飘扬，营造出动感而唯美的效果。她的长发两侧各扎着黑色蝴蝶结发饰，增添了几分可爱与俏皮感。女孩身穿一袭纯白色无袖连衣裙，裙摆轻盈飘逸，与她清新的气质完美契合。她的妆容精致自然，淡粉色的唇妆和温柔的眼神流露出恬静优雅的气质。她单手叉腰，姿态自信从容，目光直视镜头，展现出既甜美又不失个性的魅力。背景是一片开阔的海景，湛蓝的海水在阳光照射下波光粼粼，闪烁着钻石般的光芒。天空呈现出清澈的蔚蓝色，点缀着几朵洁白的云朵，营造出晴朗明媚的夏日氛围。画面前景右下角可见粉紫色的小花丛和绿色植物，为整体构图增添了自然生机和色彩层次。整张照片色调明亮清新，紫色头发与白色裙装、蓝色海天形成鲜明而和谐的色彩对比。"
negative_prompt     = " "
guidance_scale      = 1.0
seed                = 43
num_inference_steps = 40
lora_weight         = 0.55
save_path           = "samples/qwenimage21-control-images"

assert ring_degree == 1, (
    "Qwen-Image 2.1 only supports Ulysses (head-parallel) sequence parallelism; ring_degree must be 1, "
    "because ring attention cannot express the block-causal mask or the prefix KV cache."
)
device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

# Transformer
transformer = QwenImage21ControlTransformer2DModel.from_pretrained(
    model_name,
    subfolder="transformer",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
).to(weight_dtype)

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
vae = AutoencoderKLQwenImage21.from_pretrained(
    model_name,
    subfolder="vae"
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

# Get processor and text_encoder. Qwen-Image 2.1 encodes the prompt (and any condition images) with a
# Qwen3-VL model, so a processor replaces the plain tokenizer used by the earlier Qwen-Image families.
processor = Qwen3VLProcessor.from_pretrained(
    model_name, subfolder="processor"
)
text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name, subfolder="text_encoder", torch_dtype=weight_dtype
)

# Get Scheduler
Chosen_Scheduler = {
    "Flow": FlowMatchEulerDiscreteScheduler,
}[sampler_name]
scheduler = Chosen_Scheduler.from_pretrained(
    model_name,
    subfolder="scheduler"
)

pipeline = QwenImage21ControlPipeline(
    vae=vae,
    text_encoder=text_encoder,
    processor=processor,
    transformer=transformer,
    scheduler=scheduler,
)

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype, module_to_wrapper=list(transformer.transformer_blocks) + list(transformer.control_blocks))
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype, module_to_wrapper=pipeline.text_encoder.model.language_model.layers)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.transformer_blocks)):
        pipeline.transformer.transformer_blocks[i] = torch.compile(pipeline.transformer.transformer_blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_group_offload":
    register_auto_device_hook(pipeline.transformer)
    safe_enable_group_offload(pipeline, onload_device=device, offload_device="cpu", offload_type="leaf_level", use_stream=True)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["img_in", "txt_in", "time_text_embed", "modulation"], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["img_in", "txt_in", "time_text_embed", "modulation"], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

# Load the control image through get_image_latent -- the same single-frame (1, 3, h, w) tensor that
# scripts/qwenimage21_fun/train_control.py validation builds, so the resize / normalization the model was trained
# with is reproduced here instead of relying on the pipeline's own PIL preprocessing.
if control_image is not None:
    control_image_input = get_image_latent(control_image, sample_size=sample_size)[:, :, 0]
else:
    control_image_input = None

with torch.no_grad():
    sample = pipeline(
        prompt,
        negative_prompt = negative_prompt,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        true_cfg_scale = guidance_scale,
        num_inference_steps = num_inference_steps,
        control_image       = control_image_input,
        control_context_scale = control_context_scale,
        use_kv_cache = use_kv_cache,
    ).images

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    # 2.1's VAE decodes to RGBA; JPEG cannot store an alpha channel, so every preview is saved as PNG.
    image_path = os.path.join(save_path, prefix + f"-{control_context_scale}.png")
    image = sample[0]
    image.save(image_path)
    print(f"Saved image to: {image_path}")

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results()
