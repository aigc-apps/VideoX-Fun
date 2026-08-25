import os
import sys

import numpy as np
import torch
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLMiniMaxH3,
                               AutoencoderKLMiniMaxH3Audio,
                               MiniMaxH3ControlTransformer3DModel,
                               Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor)
from videox_fun.pipeline import MiniMaxH3ControlPipeline
from videox_fun.utils import (MiniMaxH3Scheduler, register_auto_device_hook,
                              safe_enable_group_offload)
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (get_video_to_video_latent,
                                    save_videos_with_audio_grid)


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
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
# Multi-GPU runs through the xfuser sequence-parallel path and must be launched with torchrun, e.g.
# `torchrun --nproc_per_node=2 examples/minimax_h3_fun/predict_v2v_control.py` for ulysses_degree=2, ring_degree=1.
# It is incompatible with the *cpu_offload* memory modes (accelerate offload hooks own a single device);
# use model_full_load / model_full_load_and_qfloat8 there, with fsdp_dit to save memory.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus. The Qwen3-VL conditioner is ~62 GB, so with fsdp_dit alone every
# rank still replicates it; fsdp_text_encoder shards it too. Note it must wrap the inner `text_encoder.model`
# (Qwen3VLModel): encode_prompt calls that submodule directly, so a wrap on the top-level module would never fire.
fsdp_dit            = False
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with sequential_cpu_offload.
compile_dit         = False

# model path
model_name          = "models/Diffusion_Transformer/MiniMax-H3"
# Control branch layout, must match the yaml `train_control.py` ran with: `control_blocks_places` selects the
# layers the control blocks attach to and `control_in_dim` the channels the control rows carry (49 for an
# `--enable_inpaint` checkpoint, whose `control_proj_in` is widened with the mask channels). Leaving it None
# builds the default 24-channel branch, which cannot load an inpaint checkpoint.
config_path         = "config/minimax_h3/minimax_h3_control.yaml"

# Load pretrained model if need. The control branch is not part of the released MiniMax-H3 weights, so a base
# `model_name` starts the side branch as an identity (`after_proj` is zero) and the c ontrol video has no effect;
# point `transformer_path` at a control checkpoint trained by `scripts/minimax_h3_fun/train_control.py`.
transformer_path    = "models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union/MiniMax-H3-Fun-Controlnet-Union.safetensors"
vae_path            = None
lora_path           = None

# Other params
# MiniMax-H3 generates at a fixed 24 fps, only accepts multiples of 32 as height / width, and the generation
# follows the control video's actual length — snapped down to the largest 17 * n + 5 the video VAE can decode so
# a short control video is never padded (the duration has to stay under 15 seconds), capped by video_length.
# Control inference fits the control video onto this canvas with the training's resize + crop geometry, so
# sample_size must be set (it cannot be None).
sample_size         = [1280, 704]
video_length        = 243
fps                 = 24
# Scale applied to every control skip before it is added to the main branch. 0.0 switches the control branch off,
# values below 1.0 weaken the guidance of the control video.
control_context_scale = 1.00

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
control_video       = "asset/pose.mp4"
# Inpaint inputs, only read by checkpoints trained with `--enable_inpaint` (control_in_dim widened, e.g. 49):
# `inpaint_video` is the source video behind the mask and `inpaint_video_mask` marks the regions to regenerate
# (white = repaint, black = keep). With an inpaint checkpoint but no inpaint inputs given, the pipeline zero-pads
# the mask channels and the run degrades to pure generation; a mask-less checkpoint rejects them outright.
inpaint_video       = None
inpaint_video_mask  = None
prompt              = "视频中，一位年轻女性站在阳光洒满的沙滩上，背景是无垠碧蓝的大海与澄澈如洗的天空，构成一幅充满夏日度假氛围的画面。她身穿一件深海军蓝吊带泳衣，线条简约贴身，凸显健康匀称的身材曲线；外搭一条纯白色背带短裙，裙摆轻盈飘逸，随风微微扬起，增添了几分俏皮与少女感。她的长发柔顺披肩，发梢微卷，在阳光下泛着自然光泽，耳畔垂挂着一对小巧精致的珍珠吊坠耳环，为整体造型注入一丝温柔优雅的气息。她面带甜美笑容，嘴角上扬，露出整齐洁白的牙齿，眼神清澈明亮，直视镜头时流露出真诚与自信，仿佛在与观众分享此刻的快乐。起初，她双臂向两侧张开，手掌舒展，像是在拥抱整个大海与天空；随后手臂缓缓收回并向前挥动，动作节奏轻快而富有韵律，如同在跳舞或做简单的热身操，展现出轻松自在、无忧无虑的状态。她的腿部微微分开站立，姿态稳健又不失灵动，裙摆随着动作轻轻摇曳，与海风形成自然互动。远处海浪轻拍沙滩，发出柔和的“哗哗”声，虽无声但可想象其韵律，与她的动作相得益彰，营造出宁静而愉悦的听觉联想。"
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
seed                = 43
# Number of denoising steps, i.e. of model evaluations: num_inference_steps = 40 runs 40 of them.
num_inference_steps = 40
# The released checkpoint is guidance-distilled: leave guidance_scale at 1 to run one forward pass per step
# with no CFG — the distill checkpoints of train_control_distill.py already bake the teacher's CFG target into
# the weights, so any value above 1 applies guidance twice and degrades the output. A value above 1 enables
# classifier-free guidance with a negative_prompt, running two passes.
guidance_scale      = 1.0
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
# The exponential sigma shifts of the two schedules. None keeps the ones of the checkpoint (12.0 video, 3.0 audio).
flow_shift          = None
audio_flow_shift    = None
lora_weight         = 0.55
save_path           = "samples/minimax-h3-videos-v2v-control"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)

# The yaml pins the control branch layout exactly as in training (scripts/minimax_h3_fun/train_control.py), where
# `transformer_additional_kwargs` is spread into `from_pretrained` the same way.
transformer_load_kwargs = {}
if config_path is not None:
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    transformer_load_kwargs.update(
        OmegaConf.to_container(config["transformer_additional_kwargs"], resolve=True)
    )

# `model_name` may point either at a converted diffusers layout or at an *original* MiniMax-H3 partition (e.g.
# `MiniMax-H3/FL2VA`); the original shards are converted on the fly while loading, no intermediate copy on disk.
# Transformer. `from_pretrained` fills the control branch the released checkpoint does not carry: every control
# block is initialised from the main block it is attached to and `control_proj_in` from `proj_in`, with
# before_proj / after_proj zeroed, so a freshly loaded model is numerically identical to the base MiniMax-H3 model.
transformer = MiniMaxH3ControlTransformer3DModel.from_pretrained(
    model_name,
    subfolder="transformer",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
    **transformer_load_kwargs,
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

# Video VAE. The released weights are float32 and the decode runs under float16 autocast, so the VAE is not
# downcast even when the rest of the pipeline is bfloat16.
vae = AutoencoderKLMiniMaxH3.from_pretrained(
    model_name,
    subfolder="vae",
    low_cpu_mem_usage=True,
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

# Audio VAE, waveform in / waveform out: MiniMax-H3 has no separate vocoder.
audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
    model_name,
    subfolder="audio_vae",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

# Get Tokenizer and Processor
tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(model_name, "tokenizer"))
processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_name, "processor"))

# Get Text encoder. MiniMax-H3 reads the unnormalized hidden state after the 50th decoder layer of Qwen3-VL.
text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    os.path.join(model_name, "text_encoder"),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Schedulers. MiniMax-H3 steps the video and the audio latents down two schedules inside one transformer call.
scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="scheduler")
audio_scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="audio_scheduler")

pipeline = MiniMaxH3ControlPipeline(
    vae=vae,
    audio_vae=audio_vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    processor=processor,
    transformer=transformer,
    scheduler=scheduler,
    audio_scheduler=audio_scheduler,
)

# The float32 modules of the mixed-precision checkpoint stay untouched by the float8 quantization. The `proj_in`
# entry also covers the control patch projection `control_proj_in`, which shares the video patch projection's dtype.
fp8_exclude_module_name = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out",
]
use_qfloat8 = "qfloat8" in GPU_memory_mode
if use_qfloat8:
    convert_model_weight_to_float8(transformer, exclude_module_name=fp8_exclude_module_name, device=device)

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        fp32_modules = [m for m in transformer.modules()
                        if any(p.dtype == torch.float32 for p in m.parameters(recurse=False))]
        shard_fn = partial(shard_model, device_id=device, param_dtype=None, cast_dtype=False,
                          module_to_wrapper=list(transformer.transformer_blocks) + list(transformer.control_blocks),
                          ignored_modules=fp32_modules)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype,
                          module_to_wrapper=list(text_encoder.model.language_model.layers))
        pipeline.text_encoder.model = shard_fn(pipeline.text_encoder.model)
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
    convert_weight_dtype_wrapper(pipeline.transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_weight_dtype_wrapper(pipeline.transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

def snap_num_frames(actual_num_frames, max_num_frames):
    """
    Pick the generation length from the control video instead of padding a short one: the largest `17 * n + 5`
    the video VAE can decode that does not exceed the frames actually read (capped by `max_num_frames`), snapping
    down so no tail frame is ever repeated. A control video below 5 frames is raised to 5, the smallest count
    the video VAE can encode.
    """
    num_frames = min(actual_num_frames, max_num_frames)
    num_frames = (num_frames - 5) // 17 * 17 + 5
    return max(num_frames, 5)

with torch.no_grad():
    control_video, _, _, _ = get_video_to_video_latent(control_video, video_length=video_length, sample_size=sample_size, fps=fps, ref_image=None, keep_aspect_ratio=True)

    # Generate at the control video's actual length, never padding; only control videos below the 5 frames the
    # video VAE can encode are raised to 5.
    num_frames = snap_num_frames(control_video.shape[2], video_length)
    if num_frames != video_length:
        print(f"[{os.environ.get('RANK', '0')}] control video holds {control_video.shape[2]} frames, generating "
              f"{num_frames} instead of {video_length}", flush=True)

    mask_video = None
    if inpaint_video is not None:
        if inpaint_video_mask is None:
            raise ValueError("inpaint_video_mask is required when inpaint_video is provided")
        inpaint_video, _, _, _ = get_video_to_video_latent(inpaint_video, video_length=video_length, sample_size=sample_size, fps=fps, ref_image=None, keep_aspect_ratio=True)
        inpaint_video_mask, _, _, _ = get_video_to_video_latent(inpaint_video_mask, video_length=video_length, sample_size=sample_size, fps=fps, ref_image=None, keep_aspect_ratio=True)
        # Binarize the grayscale mask onto one channel: 1 marks the regions to regenerate, mirroring the training
        # `get_random_mask` convention the visibility map `1 - mask` is built from.
        mask_video = (inpaint_video_mask[:, :1] > 0.5).to(inpaint_video_mask.dtype)

    output = pipeline(
        prompt=prompt,
        control_video=control_video,
        control_context_scale=control_context_scale,
        mask_video=mask_video,
        inpaint_video=inpaint_video,
        height=None if sample_size is None else sample_size[0],
        width=None if sample_size is None else sample_size[1],
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        flow_shift=flow_shift,
        audio_flow_shift=audio_flow_shift,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        generator=generator,
        output_type="pt",
    )
print(f"[{os.environ.get('RANK', '0')}] generation done, decoding", flush=True)

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

sample = output.videos
audio = output.audio
audio_sample_rate = output.sampling_rate

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    video_path = os.path.join(save_path, prefix + ".mp4")
    save_videos_with_audio_grid(sample, audio, video_path, fps=fps, audio_sample_rate=audio_sample_rate)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
    # Keep every rank alive until the saving rank finishes; an early exit of one rank makes the elastic launcher
    # terminate the others.
    dist.barrier()
else:
    save_results()
