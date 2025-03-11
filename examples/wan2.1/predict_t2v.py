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

from cogvideox.models import (AutoencoderKLWan, WanT5EncoderModel, AutoTokenizer,
                              WanTransformer3DModel)
from cogvideox.pipeline import WanPipeline
from cogvideox.utils.fp8_optimization import (convert_model_weight_to_float8,
                                              convert_weight_dtype_wrapper)
from cogvideox.utils.lora_utils import merge_lora, unmerge_lora
from cogvideox.utils.utils import get_image_to_video_latent, save_videos_grid

# GPU memory mode, which can be choosen in [model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
GPU_memory_mode     = "model_cpu_offload"

# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
# model path
model_name          = "models/Diffusion_Transformer/Wan2.1-T2V-1.3B"

# Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" and "DDIM"
sampler_name        = "Flow"

# Load pretrained model if need
transformer_path    = None
vae_path            = None
lora_path           = None

# Other params
sample_size         = [480, 832]
video_length        = 81
fps                 = 16

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
prompt              = "A young woman with beautiful and clear eyes and blonde hair standing and white dress in a forest wearing a crown. She seems to be lost in thought, and the camera focuses on her face. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
guidance_scale      = 6.0
seed                = 43
num_inference_steps = 50
lora_weight         = 0.55
save_path           = "samples/wan-videos-t2v"

config = OmegaConf.load(config_path)

transformer = WanTransformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
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

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

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
).to(weight_dtype)

# Get Scheduler
Choosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
}[sampler_name]
def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

scheduler = Choosen_Scheduler(
    **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = WanPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)
if GPU_memory_mode == "sequential_cpu_offload":
    pipeline.enable_sequential_cpu_offload()
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",])
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload()
else:
    pipeline.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight)

with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

    sample = pipeline(
        prompt, 
        num_frames = video_length,
        negative_prompt = negative_prompt,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,
    ).videos

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight)

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