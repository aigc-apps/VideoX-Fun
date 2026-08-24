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
                               MiniMaxH3Transformer3DModel, Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor)
from videox_fun.pipeline import MiniMaxH3Pipeline
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
GPU_memory_mode     = "model_group_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus. The Qwen3-VL conditioner is ~62 GB, so with fsdp_dit alone every
# rank still replicates it; fsdp_text_encoder shards it too. Note it must wrap the inner `text_encoder.model`
# (Qwen3VLModel): encode_prompt calls that submodule directly, so a wrap on the top-level module would never fire.
fsdp_dit            = False
fsdp_text_encoder   = False
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with sequential_cpu_offload.
compile_dit         = False

# model path
model_name          = "models/Diffusion_Transformer/MiniMax-H3"

# Load pretrained model if need
# A full finetune goes in `transformer_path`, either as the `transformer` folder a training checkpoint writes
# (`output_dir_minimax_h3/checkpoint-N/transformer`, config.json included) or as a single safetensors file. A LoRA
# goes in `lora_path`: handed to `transformer_path` it would match no key at all and load nothing.
transformer_path    = None
vae_path            = None
lora_path           = None

# Other params
# MiniMax-H3 generates at a fixed 24 fps, only accepts multiples of 32 as height / width, and snaps video_length up
# to the next 17 * n + 5 the video VAE can decode (the duration has to stay between 5 and 15 seconds).
# Leave sample_size as None to use MiniMax-H3's own 16:9 canvas (768x1344).
sample_size         = [704, 1280]
video_length        = 124
fps                 = 24

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
prompt              = "A red fox trotting through a snowy pine forest, snow crunching underfoot"
seed                = 43
# Number of denoising steps, i.e. of model evaluations: num_inference_steps = 40 runs 40 of them.
num_inference_steps = 40
# The released checkpoint is guidance-distilled: leave guidance_scale at 1 to run one forward pass per step
# with no CFG. A value above 1 enables classifier-free guidance with a negative_prompt, running two passes.
guidance_scale      = 1
# The exponential sigma shifts of the two schedules. None keeps the ones of the checkpoint (12.0 video, 3.0 audio).
flow_shift          = None
audio_flow_shift    = None
lora_weight         = 0.55
save_path           = "samples/minimax-h3-videos-t2v"

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

# Get Schedulers. MiniMax-H3 steps the video and the audio latents down two schedules inside one transformer call.
scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="scheduler")
audio_scheduler = MiniMaxH3Scheduler.from_pretrained(model_name, subfolder="audio_scheduler")

pipeline = MiniMaxH3Pipeline(
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
    # Quantize before any FSDP wrapping so the fp8 tensors become the FSDP storage dtype; the per-forward
    # dequant wrapper is applied later only when the DiT is not FSDP-sharded (it would rewrite `param.data`
    # behind FSDP's flat storage and corrupt the compute).
    convert_model_weight_to_float8(transformer, exclude_module_name=fp8_exclude_module_name, device=device)

dit_is_fsdp = False
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        # The mixed-precision checkpoint pins the patch embedders / timestep MLP / output heads to float32;
        # FSDP keeps them replicated via ignored_states so the flat buffers stay uniform-dtype.
        #
        # Root cause of the temporal flicker, verified by per-step / per-block instrumentation: with
        # `MixedPrecision(param_dtype=...)` the root FSDP unit applies `cast_root_forward_inputs` (default
        # True), so the whole root forward runs in `param_dtype`. That casts the root forward inputs — the
        # sinusoidal timestep embedding, the packed latents, the context — to bfloat16 and forces the fp32-
        # pinned heads (proj_in / time_embedder / audio_proj_in) to compute on coarsely rounded inputs in
        # bfloat16 instead of their native fp32; the deviation compounds over the sampling steps and flips
        # trajectories that sit on the numerical-stability edge into coherent flicker at fixed latent-time
        # positions, seed-independently.
        # Sharding with `param_dtype=None` + `cast_dtype=False` casts nothing (no MixedPrecision compute
        # dtype, no root input cast), keeps the native fp32 hidden path and matches the non-FSDP numerics.
        #
        # The qfloat8 path cannot use the no-cast scheme: fp8 storage has no dequant wrapper under FSDP, so
        # `MixedPrecision(param_dtype)` is the only dequant route there and the forward collapses to
        # bfloat16 anyway — measured to flicker even worse than the bf16-cast path. With `fsdp_dit=True`
        # prefer a non-qfloat8 memory mode; sharding already drops the per-rank DiT/TE weights to
        # ~(62+62)/n_gpu GB, fp8 saves little on top of it.
        fp32_modules = [m for m in transformer.modules()
                        if any(p.dtype == torch.float32 for p in m.parameters(recurse=False))]
        if use_qfloat8:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype,
                              module_to_wrapper=list(transformer.transformer_blocks),
                              ignored_modules=[m for m in fp32_modules])
        else:
            shard_fn = partial(shard_model, device_id=device, param_dtype=None, cast_dtype=False,
                              module_to_wrapper=list(transformer.transformer_blocks),
                              ignored_modules=fp32_modules)
        pipeline.transformer = shard_fn(pipeline.transformer)
        dit_is_fsdp = True
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
    if not dit_is_fsdp:
        convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    if not dit_is_fsdp:
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
        height=None if sample_size is None else sample_size[0],
        width=None if sample_size is None else sample_size[1],
        num_frames=video_length,
        num_inference_steps=num_inference_steps,
        flow_shift=flow_shift,
        audio_flow_shift=audio_flow_shift,
        guidance_scale=guidance_scale,
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
