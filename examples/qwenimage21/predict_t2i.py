import os
import sys

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLQwenImage21,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor, QwenImage21Transformer2DModel)
from videox_fun.pipeline import QwenImage21Pipeline
from videox_fun.utils import apply_gpu_memory_mode, merge_lora, unmerge_lora

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
# Qwen-Image 2.1 uses a block-causal single-stream transformer with a prefix KV cache, which is not
# compatible with the sequence-parallel attention used by the other families. Please run it on a single
# GPU (ulysses_degree = 1 and ring_degree = 1).
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = False
# Compile will give a speedup in fixed resolution and need a little GPU memory.
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# model path
model_name          = "models/Diffusion_Transformer/Qwen-Image-2.1"

# Choose the sampler. Qwen-Image 2.1 is a flow-matching model sampled with the Euler discrete scheduler.
sampler_name        = "Flow"

# Load pretrained model if need
transformer_path    = None
vae_path            = None
lora_path           = None

# Other params
# sample_size is the output canvas in pixels as [height, width]; the pipeline rounds it down to a
# multiple of 32. Leave it as None to fall back to the pipeline's default square resolution.
sample_size         = [1024, 1024]
# Cache the text and condition-image keys/values after the first denoising step. Valid because the
# transformer modulates those tokens from t = 0, making their activations step-independent.
use_kv_cache        = True

# Use torch.float16 if GPU does not support torch.bfloat16
# Some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
# Please use as detailed a prompt as possible to describe the object that needs to be generated.
prompts             = ["a young girl with flowing long hair, wearing a white halter dress and smiling sweetly. The background features a blue seaside where seagulls fly freely."]
negative_prompt     = " "
guidance_scale      = 1.0
seed                = 43
num_inference_steps = 40
lora_weight         = 0.55
save_path           = "samples/qwenimage21-t2i"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)

transformer = QwenImage21Transformer2DModel.from_pretrained(
    model_name,
    subfolder="transformer",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
).to(weight_dtype)

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
vae = AutoencoderKLQwenImage21.from_pretrained(
    model_name,
    subfolder="vae"
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

pipeline = QwenImage21Pipeline(
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
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype, module_to_wrapper=list(transformer.transformer_blocks))
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype, module_to_wrapper=text_encoder.model.language_model.layers)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.transformer_blocks)):
        pipeline.transformer.transformer_blocks[i] = torch.compile(pipeline.transformer.transformer_blocks[i])
    print("Add Compile")

# Quantize (when the mode carries an "_and_<quant>" suffix) and then place the pipeline.
# The order lives inside the helper: quantization has to happen before the offload hooks
# are installed.
apply_gpu_memory_mode(pipeline, GPU_memory_mode, device, weight_dtype, exclude_module_name=['img_in', 'txt_in', 'time_text_embed', 'modulation'])

for prompt in prompts:
    generator = torch.Generator(device=device).manual_seed(seed)

    if lora_path is not None:
        pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

    with torch.no_grad():
        sample = pipeline(
            prompt,
            negative_prompt = negative_prompt,
            height      = sample_size[0] if sample_size is not None else None,
            width       = sample_size[1] if sample_size is not None else None,
            generator   = generator,
            true_cfg_scale = guidance_scale,
            num_inference_steps = num_inference_steps,
            use_kv_cache = use_kv_cache,
        ).images

    if lora_path is not None:
        pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

    def save_results():
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        index = len([path for path in os.listdir(save_path)]) + 1
        prefix = str(index).zfill(8)
        image_path = os.path.join(save_path, prefix + ".png")
        image = sample[0]
        image.save(image_path)
        print(f"Saved image to: {image_path}")

    if ulysses_degree * ring_degree > 1:
        import torch.distributed as dist
        if dist.get_rank() == 0:
            save_results()
    else:
        save_results()
