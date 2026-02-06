import math
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from audio_separator.separator import Separator
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLLongCatVideo, AutoTokenizer,
                               LongCatVideoAvatarTransformer3DModel,
                               UMT5EncoderModel, Wav2Vec2FeatureExtractor,
                               Wav2Vec2ModelWrapper)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import LongCatVideoAvatarPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                    merge_video_audio, save_videos_grid)

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
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# model path
model_name          = "models/Diffusion_Transformer/LongCat-Video"
model_name_avatar   = "models/Diffusion_Transformer/LongCat-Video-Avatar"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow"

# Load pretrained model if need
transformer_path    = None
vae_path            = None
lora_path           = None

# Other params
sample_size         = [832, 480]
video_length        = 81
fps                 = 16

# Start Image
validation_image_start  = "asset/8.png"

# Audio params
audio_stride        = 2 
audio_path          = "asset/talk.wav"

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
# Prompt
prompt              = "A western man stands on stage under dramatic lighting, holding a microphone close to their mouth. Wearing a vibrant red jacket with gold embroidery, the singer is speaking while smoke swirls around them, creating a dynamic and atmospheric scene."
negative_prompt     = "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
guidance_scale      = 4.0
seed                = 43
num_inference_steps = 50
lora_weight         = 0.55
save_path           = "samples/longcat-avatar-videos-t2v"

device = set_multi_gpus_devices(1, 1)

transformer = LongCatVideoAvatarTransformer3DModel.from_pretrained(
    os.path.join(model_name_avatar, "avatar_single"),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype, cp_split_hw=[1, 1]
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
vae = AutoencoderKLLongCatVideo.from_pretrained(
    os.path.join(model_name, "vae"),
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
    os.path.join(model_name, "tokenizer"),
)

# Get Text encoder
text_encoder = UMT5EncoderModel.from_pretrained(
    os.path.join(model_name, "text_encoder"),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

# Get Audio encoder (for avatar mode)
audio_encoder = Wav2Vec2ModelWrapper(
    os.path.join(model_name_avatar, 'chinese-wav2vec2-base')
)
audio_encoder.feature_extractor._freeze_parameters()

wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    os.path.join(model_name_avatar, 'chinese-wav2vec2-base'), 
    local_files_only=True
)

# Get Vocal separator
vocal_separator_path = os.path.join(model_name_avatar, 'vocal_separator/Kim_Vocal_2.onnx')
audio_output_dir_temp = Path("./audio_temp_file")
audio_output_dir_temp.mkdir(parents=True, exist_ok=True)

vocal_separator = Separator(
    output_dir=audio_output_dir_temp / "vocals",
    output_single_stem="vocals",
    model_file_dir=os.path.dirname(vocal_separator_path),
)
vocal_separator.load_model(os.path.basename(vocal_separator_path))

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

# Get Pipeline
pipeline = LongCatVideoAvatarPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    audio_encoder=audio_encoder,
    wav2vec_feature_extractor=wav2vec_feature_extractor,
)

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

# Process audio if provided
audio_emb = None
if audio_path is not None:
    # Extract vocal from audio
    outputs = vocal_separator.separate(audio_path)
    if len(outputs) > 0:
        temp_vocal_path = audio_output_dir_temp / "vocals" / outputs[0]
        temp_vocal_path = temp_vocal_path.resolve().as_posix()
        
        # Load and pad audio to target length
        generate_duration = video_length / fps
        speech_array, sr = librosa.load(temp_vocal_path, sr=16000)
        source_duration = len(speech_array) / sr
        added_sample_nums = math.ceil((generate_duration - source_duration) * sr)
        if added_sample_nums > 0:
            speech_array = np.append(speech_array, [0.] * added_sample_nums)
        
        # Get audio embedding
        with torch.no_grad():
            full_audio_emb = pipeline.get_audio_embedding(
                speech_array, 
                fps=fps * audio_stride, 
                device=device, 
                sample_rate=sr
            )
            
            # Prepare audio embedding with sliding window
            indices = torch.arange(2 * 2 + 1) - 2  # [-2, -1, 0, 1, 2]
            audio_start_idx = 0
            audio_end_idx = audio_start_idx + audio_stride * video_length
            
            center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
            center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
            audio_emb = full_audio_emb[center_indices][None, ...].to(device)
        
        # Clean up temp file
        if os.path.exists(temp_vocal_path):
            os.remove(temp_vocal_path)

with torch.no_grad():
    video_length = int((video_length - 1) // vae.scale_factor_temporal * vae.scale_factor_temporal) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.scale_factor_temporal + 1

    if validation_image_start is not None:
        input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, None, video_length=video_length, sample_size=sample_size)
    else:
        input_video, input_video_mask, clip_image = None, None, None
    
    sample = pipeline(
        prompt = prompt,
        num_frames = video_length,
        negative_prompt = negative_prompt,
        height = sample_size[0],
        width = sample_size[1],
        generator = generator,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,
        audio_emb = audio_emb,
        video       = input_video,
        mask_video  = input_video_mask,
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
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)
    
    merge_video_audio(video_path=video_path, audio_path=audio_path)

save_results()
