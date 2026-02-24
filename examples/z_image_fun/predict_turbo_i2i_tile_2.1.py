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
from videox_fun.models import (AutoencoderKL, AutoTokenizer, Qwen3ForCausalLM,
                               ZImageControlTransformer2DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import ZImageControlPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image, get_image_latent,
                                    get_image_to_video_latent,
                                    get_video_to_video_latent,
                                    save_videos_grid)

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
fsdp_text_encoder   = False
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# Config and model path
config_path         = "config/z_image/z_image_control_2.1.yaml"
# model path
model_name          = "models/Diffusion_Transformer/Z-Image-Turbo"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow"

# Load pretrained model if need
transformer_path    = "models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.safetensors" 
vae_path            = None
lora_path           = None

# Other params
sample_size         = [1328, 1328]

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
control_image       = "asset/low_res.png"
# The inpaint_image and mask_image is useless in tile model, just set them to None.
inpaint_image       = None
mask_image          = None
control_context_scale = 0.85

# Please use as detailed a prompt as possible to describe the object that needs to be generated.
prompt              = "这是一张充满都市气息的户外人物肖像照片。画面中是一位年轻男性，他展现出时尚而自信的形象。人物拥有精心打理的短发发型，两侧修剪得较短，顶部保留一定长度，呈现出流行的Undercut造型。他佩戴着一副时尚的浅色墨镜或透明镜框眼镜，为整体造型增添了潮流感。脸上洋溢着温和友善的笑容，神情放松自然，给人以阳光开朗的印象。他身穿一件经典的牛仔外套，这件单品永不过时，展现出休闲又有型的穿衣风格。牛仔外套的蓝色调与整体氛围十分协调，领口处隐约可见内搭的衣物。照片的背景是典型的城市街景，可以看到模糊的建筑物、街道和行人，营造出繁华都市的氛围。背景经过了恰当的虚化处理，使人物主体更加突出。光线明亮而柔和，可能是白天的自然光，为照片带来清新通透的视觉效果。整张照片构图专业，景深控制得当，完美捕捉了一个现代都市年轻人充满活力和自信的瞬间，展现出积极向上的生活态度。"
negative_prompt     = " "
guidance_scale      = 0.00
seed                = 43
num_inference_steps = 8
lora_weight         = 0.55
save_path           = "samples/z-image-t2i-control"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

transformer = ZImageControlTransformer2DModel.from_pretrained(
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
vae = AutoencoderKL.from_pretrained(
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

# Get tokenizer and text_encoder
tokenizer = AutoTokenizer.from_pretrained(
    model_name, subfolder="tokenizer"
)
text_encoder = Qwen3ForCausalLM.from_pretrained(
    model_name, subfolder="text_encoder", torch_dtype=weight_dtype,
    low_cpu_mem_usage=True,
)

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
scheduler = Chosen_Scheduler.from_pretrained(
    model_name, 
    subfolder="scheduler"
)

pipeline = ZImageControlPipeline(
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    transformer=transformer,
    scheduler=scheduler,
)

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype, module_to_wrapper=list(transformer.layers))
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype, module_to_wrapper=list(text_encoder.model.layers))
        text_encoder = shard_fn(text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.transformer_blocks)):
        pipeline.transformer.transformer_blocks[i] = torch.compile(pipeline.transformer.transformer_blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["x_pad_token", "cap_pad_token"], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["x_pad_token", "cap_pad_token"], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

with torch.no_grad():
    if inpaint_image is not None:
        inpaint_image = get_image_latent(inpaint_image, sample_size=sample_size)[:, :, 0]
    else:
        inpaint_image = torch.zeros([1, 3, sample_size[0], sample_size[1]])

    if mask_image is not None:
        mask_image = get_image_latent(mask_image, sample_size=sample_size)[:, :1, 0]
    else:
        mask_image = torch.ones([1, 1, sample_size[0], sample_size[1]]) * 255

    if control_image is not None:
        control_image = get_image_latent(control_image, sample_size=sample_size)[:, :, 0]

    sample = pipeline(
        prompt      = prompt, 
        negative_prompt = negative_prompt,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guidance_scale = guidance_scale,
        image               = inpaint_image,
        mask_image          = mask_image,
        control_image       = control_image,
        num_inference_steps = num_inference_steps,
        control_context_scale = control_context_scale,
    ).images

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    video_path = os.path.join(save_path, prefix + ".png")
    image = sample[0]
    image.save(video_path)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results()