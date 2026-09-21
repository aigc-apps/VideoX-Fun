import os
import sys

import torch

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (AutoencoderKLMiniMaxH3,
                               AutoencoderKLMiniMaxH3Audio,
                               MiniMaxH3Transformer3DModel, Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor)
from videox_fun.pipeline import MiniMaxH3StreamingPipeline
from videox_fun.utils import (MiniMaxH3Scheduler, register_auto_device_hook,
                              safe_enable_group_offload)
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import save_videos_with_audio_grid

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
# The transformer alone is 61.7 GB in bfloat16 and the Qwen3-VL conditioner another 62.1 GB; with the persistent
# streaming K/V cache on top, both model_cpu_offload and model_cpu_offload_and_qfloat8 exceed a single 80 GB card
# (measured OOM), so model_group_offload is the verified default for one-card streaming.
GPU_memory_mode     = "model_group_offload"
# Multi GPUs config. Streaming inference runs the whole attention sequence on one GPU (the persistent
# K/V cache must stay whole-sequence on one device), so keep ulysses_degree = ring_degree = 1 and the
# FSDP switches off. To generate many prompts with one prompt per GPU, use
# `examples/taomate_h3/_predict_t2av_streaming_list.py` instead.
ulysses_degree      = 1
ring_degree         = 1
fsdp_dit            = False
fsdp_text_encoder   = False
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with sequential_cpu_offload.
compile_dit         = False

# model path
model_name          = "models/Diffusion_Transformer/MiniMax-H3"

# Load pretrained model if need
# A full finetune goes in `transformer_path` (a training checkpoint's `transformer` folder or a single
# safetensors file). A LoRA goes in `lora_path`, which accepts either a kohya safetensors checkpoint (e.g. the
# output of `scripts/taomate_h3/train_distill_lora.py`) or the *official* TaoMate-H3 adapter directory:
# `merge_lora` tells the two apart (a directory vs a file) and converts the official adapter to the kohya
# layout on the fly.
transformer_path    = None
vae_path            = None
# The official TaoMate-H3 adapter (rank 128 / alpha 128, the step-3000 generator EMA) from
# `TaoLiveAIGC/TaoMate-H3`. The distilled 3-step schedule is this adapter's own behaviour, so it is on by
# default — the bare base weights do not reproduce the official runtime. Download it first with:
#   hf download TaoLiveAIGC/TaoMate-H3 --include "config.json" "adapter_config.json" "adapter_model.safetensors" \
#     --local-dir models/Diffusion_Transformer/TaoMate-H3-adapter
lora_path           = "models/Diffusion_Transformer/TaoMate-H3-adapter"

# Other params
# The canvas: the short edge must be 480, 768 or 1088 and both edges 32-aligned (480x864 is the
# resolution the official TaoMate-H3 demo ships). The teacher artifact is bound to this geometry.
sample_size         = [864, 480]
# How many 5-second stream requests to generate: the first one runs the direct 124-frame plan, every
# following one the canonical 119-frame continuation spliced behind its predecessor. One prompt covers
# the whole timeline.
request_count       = 2
fps                 = 24

# Use torch.float16 if GPU does not support torch.bfloat16
# Some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
# The prompt the default Base10 teacher artifact (`samples/taomate_h3_teacher/00000000`) was generated for —
# `Self-Forcing/prompts/self_forcing_all_prompts.json` entry 0. A list gives one prompt per stream request
# (the official `--prompt-json` shape); the artifact must then carry exactly those prompts.
prompt              = (
    "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city "
    "signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red "
    "dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a "
    "bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a "
    "mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the "
    "lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to "
    "one side, highlighting her graceful strides."
)
# The authored seed. Request i draws its video noise from seed + i * 1000003; the audio noise seeds come
# from the teacher artifact, so the artifact must have been generated for this very seed.
seed                = 43
# The offline Base10 audio-teacher artifact directory produced by `examples/taomate_h3/predict_audio.py`
# for exactly this prompt, seed and resolution (set the same value in its `audio_teacher_dir` knob; the
# producer runs audio-only base-weight forwards, no video decode). Generate it before the first run.
audio_teacher_dir   = "samples/taomate_h3_teacher/00000000"
# Merge weight of `lora_path`. The official TaoMate-H3 adapter ships alpha == rank, so 1.0 reproduces the
# official runtime; lower it (e.g. 0.55) when blending a kohya finetune checkpoint instead.
lora_weight         = 1.0
save_path           = "samples/taomate-h3-videos-t2av-streaming"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)

# `model_name` may point either at a converted diffusers layout or at an *original* MiniMax-H3 partition (e.g.
# `MiniMax-H3/FL2VA`); the original shards are converted on the fly while loading, no intermediate copy on disk.
# Transformer
transformer = MiniMaxH3Transformer3DModel.from_pretrained(
    model_name,
    subfolder="transformer",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if os.path.isdir(transformer_path):
        # A training checkpoint's `transformer` folder carries its own config.json, so the loader restores the
        # mixed-precision contract of the checkpoint (`_keep_in_fp32_modules`) by itself.
        transformer = MiniMaxH3Transformer3DModel.from_pretrained(
            transformer_path,
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        # `strict=False` accepts a file whose keys belong to another model — a LoRA checkpoint, say — by loading
        # nothing at all and silently generating with the base weights, so an unexpected key is a hard error.
        assert len(u) == 0, (
            f"{transformer_path} holds {len(u)} key(s) the transformer does not have, e.g. {u[:3]}. A LoRA "
            "checkpoint belongs in `lora_path`, not `transformer_path`."
        )

# Video VAE. The released weights are float32 and the decode runs under float16 autocast, so the VAE is not
# downcast even when the rest of the pipeline is bfloat16 (this is also how the training scripts load it).
vae = AutoencoderKLMiniMaxH3.from_pretrained(
    model_name,
    subfolder="vae",
    low_cpu_mem_usage=True,
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

# Audio VAE, waveform in / waveform out: MiniMax-H3 has no separate vocoder. Float32 as released, like the video VAE.
audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
    model_name,
    subfolder="audio_vae",
    low_cpu_mem_usage=True,
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

# Get Schedulers. The streaming pipeline overrides the step counts internally (the distilled 3-step video
# schedule and the teacher-anchored audio schedule), so the checkpoint schedules only seed the class.
scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="scheduler")
audio_scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="audio_scheduler")

pipeline = MiniMaxH3StreamingPipeline(
    vae=vae,
    audio_vae=audio_vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    processor=processor,
    transformer=transformer,
    scheduler=scheduler,
    audio_scheduler=audio_scheduler,
)

# The float32 modules of the mixed-precision checkpoint stay untouched by the float8 quantization.
fp8_exclude_module_name = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out",
]
use_qfloat8 = "qfloat8" in GPU_memory_mode
if use_qfloat8:
    convert_model_weight_to_float8(transformer, exclude_module_name=fp8_exclude_module_name, device=device)

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

# Merge LoRA through the standard entry point: `merge_lora` detects an official TaoMate-H3 adapter
# directory and converts it to the kohya layout on the fly, and takes a kohya safetensors checkpoint
# as before; no CFG pass is run either way.
if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
else:
    print(
        "WARNING: no LoRA is loaded. The 3-step distilled schedule expects the official TaoMate-H3 adapter "
        "(`lora_path`, e.g. models/Diffusion_Transformer/TaoMate-H3-adapter); the bare base weights do not reproduce the "
        "official runtime and the result will be visibly off.",
        flush=True,
    )

# One continuous video + soundtrack 5 seconds at a time: each request reuses the cleaned audio/video K/V
# of everything before it (full-sequence attention, no re-encoding), the video denoises in the distilled
# 3-step schedule, and the audio is anchored by the *offline* Base10 teacher artifact.
with torch.no_grad():
    output = pipeline(
        prompt=prompt,
        audio_teacher_dir=audio_teacher_dir,
        height=None if sample_size is None else sample_size[0],
        width=None if sample_size is None else sample_size[1],
        request_count=request_count,
        seed=seed,
        output_type="pt",
    )
print(f"[{os.environ.get('RANK', '0')}] generation done, decoding", flush=True)

# Restore the merged weights after generation: both the official adapter directory and the kohya
# checkpoint go through the same `unmerge_lora` path.
if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

for receipt in output.request_receipts:
    plan = receipt["plan"]
    print(
        f"request {receipt['request_index']}: continuation={receipt['canonical_continuation']}, "
        f"phases={len(plan['phases'])}, published video latents={receipt['published_video_latents']}, "
        f"audio latents/channel={receipt['published_audio_latents_per_channel']}, "
        f"retained history={receipt['retained_history_tokens']} tokens"
    )

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
