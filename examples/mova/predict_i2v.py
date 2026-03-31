import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import (AutoencoderKLMOVAAudio, AutoencoderKLWan,
                               AutoTokenizer, MOVADualTowerConditionalBridge,
                               MOVAModel, UMT5EncoderModel,
                               WanAudioTransformer3DModel,
                               WanTransformer3DModel)
from videox_fun.pipeline import MOVAPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import save_videos_with_audio_grid

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
GPU_memory_mode     = "sequential_cpu_offload"
# Compile will give a speedup in fixed resolution and need a little GPU memory.
# The compile_dit is not compatible with sequential_cpu_offload.
compile_dit         = False

# model path
model_name          = "models/Diffusion_Transformer/MOVA-360p"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow"
boundary_ratio      = 0.9

# Load pretrained model if need
transformer_path  = None
transformer_high_path = None
audio_dit_path      = None
bridge_path         = None
vae_path            = None
audio_vae_path      = None
lora_path           = None

# Other params
sample_size         = [352, 640]
video_length        = 81
fps                 = 24

# Use torch.float16 if GPU does not support torch.bfloat16
# Some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16

# Input image for I2V
validation_image    = "asset/single_person.jpg"

# prompts
prompt              = "A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks. At one point, he says, \"I would also say that this election in Germany wasn't surprising.\""
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指"
guidance_scale      = 5.0
seed                = 43
num_inference_steps = 50
lora_weight         = 0.55
save_path           = "samples/mova-videos-i2v"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# The from_pretrained method automatically converts WanModel config to WanTransformer3DModel config
print("Loading Video DiT (High Noise) with WanTransformer3DModel...")
transformer = WanTransformer3DModel.from_pretrained(
    model_name,
    subfolder="video_dit",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

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

# Video DiT 2 (Low Noise) - Using WanTransformer3DModel
print("Loading Video DiT 2 (Low Noise) with WanTransformer3DModel...")
transformer_2 = WanTransformer3DModel.from_pretrained(
    model_name,
    subfolder="video_dit_2",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_high_path is not None:
    print(f"From checkpoint: {transformer_high_path}")
    if transformer_high_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(transformer_high_path)
    else:
        state_dict = torch.load(transformer_high_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = transformer_2.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Audio DiT - Using WanAudioTransformer3DModel
print("Loading Audio DiT with WanAudioTransformer3DModel...")
audio_dit = WanAudioTransformer3DModel.from_pretrained(
    model_name,
    subfolder="audio_dit",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if audio_dit_path is not None:
    print(f"From checkpoint: {audio_dit_path}")
    if audio_dit_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(audio_dit_path)
    else:
        state_dict = torch.load(audio_dit_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = audio_dit.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Dual Tower Bridge
print("Loading Dual Tower Bridge...")
dual_tower_bridge = MOVADualTowerConditionalBridge.from_pretrained(
    model_name,
    subfolder="dual_tower_bridge",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if bridge_path is not None:
    print(f"From checkpoint: {bridge_path}")
    if bridge_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(bridge_path)
    else:
        state_dict = torch.load(bridge_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = dual_tower_bridge.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Video VAE
print("Loading Video VAE...")
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, "video_vae/diffusion_pytorch_model.safetensors")
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

audio_vae = AutoencoderKLMOVAAudio.from_pretrained(
    model_name,
    subfolder="audio_vae",
    torch_dtype=torch.float32,
)

if audio_vae_path is not None:
    print(f"From checkpoint: {audio_vae_path}")
    if audio_vae_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(audio_vae_path)
    else:
        state_dict = torch.load(audio_vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = audio_vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    subfolder="tokenizer",
)

# Get Text Encoder
print("Loading Text Encoder...")
text_encoder = UMT5EncoderModel.from_pretrained(
    model_name,
    subfolder="text_encoder",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Scheduler
print("Loading Scheduler...")
Chosen_Scheduler = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
scheduler = Chosen_Scheduler.from_pretrained(
    model_name,
    subfolder="scheduler"
)

# Build MOVAModel
print("Building MOVAModel...")
mova_model = MOVAModel(
    transformer=transformer,
    transformer_2=transformer_2,
    audio_dit=audio_dit,
    dual_tower_bridge=dual_tower_bridge,
)

# Build Pipeline
print("Building MOVAPipeline Pipeline...")
pipeline = MOVAPipeline(
    vae=vae,
    audio_vae=audio_vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
    mova_model=mova_model,
    audio_vae_type="dac",
)

if compile_dit:
    # Compile MOVAModel blocks
    for i in range(len(pipeline.mova_model.transformer.blocks)):
        pipeline.mova_model.transformer.blocks[i] = torch.compile(pipeline.mova_model.transformer.blocks[i])
    for i in range(len(pipeline.mova_model.transformer_2.blocks)):
        pipeline.mova_model.transformer_2.blocks[i] = torch.compile(pipeline.mova_model.transformer_2.blocks[i])
    for i in range(len(pipeline.mova_model.audio_dit.blocks)):
        pipeline.mova_model.audio_dit.blocks[i] = torch.compile(pipeline.mova_model.audio_dit.blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(pipeline.mova_model.transformer, ["modulation",], device=device)
    replace_parameters_by_name(pipeline.mova_model.transformer_2, ["modulation",], device=device)
    pipeline.mova_model.transformer.freqs = pipeline.mova_model.transformer.freqs.to(device=device)
    pipeline.mova_model.transformer_2.freqs = pipeline.mova_model.transformer_2.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(pipeline.mova_model.transformer, exclude_module_name=["modulation",], device=device)
    convert_model_weight_to_float8(pipeline.mova_model.transformer_2, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(pipeline.mova_model.transformer, weight_dtype)
    convert_weight_dtype_wrapper(pipeline.mova_model.transformer_2, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_group_offload":
    register_auto_device_hook(pipeline.mova_model.transformer)
    safe_enable_group_offload(pipeline, onload_device=device, offload_device="cpu", offload_type="leaf_level", use_stream=True)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(pipeline.mova_model.transformer, exclude_module_name=["modulation",], device=device)
    convert_model_weight_to_float8(pipeline.mova_model.transformer_2, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(pipeline.mova_model.transformer, weight_dtype)
    convert_weight_dtype_wrapper(pipeline.mova_model.transformer_2, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

# Run inference
print("Running inference...")
with torch.no_grad():
    image = Image.open(validation_image).convert("RGB")
    output = pipeline(
        prompt=prompt,
        image=image,
        negative_prompt=negative_prompt,
        height=sample_size[0],
        width=sample_size[1],
        num_frames=video_length,
        frame_rate=fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        boundary=boundary_ratio,
    )

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

sample = output.videos
audio = output.audio

# Get audio sample rate from pipeline
audio_sample_rate = pipeline.audio_sample_rate

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
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        sr = getattr(pipeline.audio_vae.config, "output_sampling_rate", audio_sample_rate)
        print("audio_sample_rate", audio_sample_rate, audio.size())
        save_videos_with_audio_grid(sample, audio, video_path, fps=fps, audio_sample_rate=sr)

save_results()