import os
import shutil
import sys

import av
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import (AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video,
                               LTX2TextConnectors, LTX2VideoTransformer3DModel,
                               LTX2Vocoder)
from videox_fun.pipeline import LTX2Pipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.utils import merge_video_audio, save_videos_grid

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
model_name          = "models/Diffusion_Transformer/LTX-2"
# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow"

# Load pretrained model if need
transformer_path    = None
vae_path            = None
lora_path           = None

# Other params
sample_size         = [512, 768]
video_length        = 121
fps                 = 24

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
prompt              = "A brown dog barks on a sofa, sitting on a light-colored couch in a cozy room. Behind the dog, there is a framed painting on a shelf, surrounded by pink flowers. "
negative_prompt     = "worst quality, inconsistent motion, blurry, jittery, distorted, static, low quality, artifacts"
guidance_scale      = 6.0
seed                = 43
num_inference_steps = 50
lora_weight         = 0.55
save_path           = "samples/ltx2-videos-t2v"

# Audio sample rate will be read from vocoder config
audio_sample_rate   = 24000

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Transformer
transformer = LTX2VideoTransformer3DModel.from_pretrained(
    model_name,
    subfolder="transformer",
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

# Video VAE
vae = AutoencoderKLLTX2Video.from_pretrained(
    model_name,
    subfolder="vae",
    torch_dtype=weight_dtype,
)

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

# Audio VAE
audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
    model_name,
    subfolder="audio_vae",
    torch_dtype=weight_dtype,
)

# Get Tokenizer
tokenizer = GemmaTokenizerFast.from_pretrained(
    model_name,
    subfolder="tokenizer",
)

# Get Text encoder
text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
    model_name,
    subfolder="text_encoder",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Connectors
connectors = LTX2TextConnectors.from_pretrained(
    model_name,
    subfolder="connectors",
    torch_dtype=weight_dtype,
)

# Vocoder
vocoder = LTX2Vocoder.from_pretrained(
    model_name,
    subfolder="vocoder",
    torch_dtype=weight_dtype,
)

# Get Scheduler
Chosen_Scheduler = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
scheduler = Chosen_Scheduler.from_pretrained(
    model_name, 
    subfolder="scheduler"
)

pipeline = LTX2Pipeline(
    scheduler=scheduler,
    vae=vae,
    audio_vae=audio_vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    connectors=connectors,
    transformer=transformer,
    vocoder=vocoder,
)

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
    convert_model_weight_to_float8(transformer, exclude_module_name=["img_in", "txt_in", "timestep"], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["img_in", "txt_in", "timestep"], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

with torch.no_grad():
    output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=sample_size[0],
        width=sample_size[1],
        num_frames=video_length,
        frame_rate=fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pt",
    )

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

sample = output.videos
audio = output.audio

def _prepare_audio_stream(container, audio_sample_rate):
    from fractions import Fraction
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream


def _write_audio(container, audio_stream, samples, audio_sample_rate):
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.shape[1] != 2 and samples.shape[0] == 2:
        samples = samples.T
    if samples.shape[1] != 2:
        # mono -> duplicate to stereo
        samples = samples.expand(-1, 2)

    if samples.dtype != torch.int16:
        samples = torch.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).to(torch.int16)

    frame_in = av.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio_sample_rate

    cc = audio_stream.codec_context
    target_format = cc.format or "fltp"
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    audio_next_pts = 0
    for rframe in resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        audio_next_pts += rframe.samples
        rframe.sample_rate = frame_in.sample_rate
        container.mux(audio_stream.encode(rframe))

    for packet in audio_stream.encode():
        container.mux(packet)


def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)

    if video_length == 1:
        out_path = os.path.join(save_path, prefix + ".png")
        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        Image.fromarray(image).save(out_path)
        print(f"Saved image to: {out_path}")
        return

    import torchvision
    from einops import rearrange

    frames_t = rearrange(sample, "b c t h w -> t b c h w")
    frame_list = []
    for x in frames_t:
        x = torchvision.utils.make_grid(x, nrow=6)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        x = (x * 255).numpy().astype(np.uint8)
        frame_list.append(x)

    height, width = frame_list[0].shape[:2]
    audio_tensor = audio[0].float().cpu()
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(-1)
    if audio_tensor.shape[1] != 2 and audio_tensor.shape[0] == 2:
        audio_tensor = audio_tensor.T
    sr = getattr(pipeline.vocoder.config, "output_sampling_rate", audio_sample_rate)

    out_path = os.path.join(save_path, prefix + "_with_audio.mp4")
    container = av.open(out_path, mode="w")
    v_stream = container.add_stream("libx264", rate=int(fps))
    v_stream.width = width
    v_stream.height = height
    v_stream.pix_fmt = "yuv420p"

    a_stream = _prepare_audio_stream(container, sr)

    for frame_np in frame_list:
        frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
        for pkt in v_stream.encode(frame):
            container.mux(pkt)
    for pkt in v_stream.encode():
        container.mux(pkt)

    _write_audio(container, a_stream, audio_tensor, sr)

    container.close()
    print(f"Saved merged video+audio to: {out_path}")

save_results()