import os
import sys

import torch

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLMiniMaxH3,
                               AutoencoderKLMiniMaxH3Audio,
                               FastVideoMiniMaxH3Transformer3DModel,
                               Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor)
from videox_fun.pipeline import MiniMaxH3Pipeline
from videox_fun.utils import (MiniMaxH3Scheduler, apply_gpu_memory_mode,
                              convert_model_weight_to_float8, merge_lora,
                              save_videos_with_audio_grid, unmerge_lora)

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
# The MiniMax-H3 stack (~19B transformer + the ~62GB Qwen3-VL conditioner) does not fit a single
# GPU at full load, so keep an offloading mode here.
GPU_memory_mode     = "model_group_offload"
# Multi GPUs config
# NOTE: the VSA sparse kernel has no sequence-parallel path, so a FastVideo-H3 student runs
# single-GPU only; keep ulysses_degree = ring_degree = 1 (enable_multi_gpus_inference() raises
# otherwise, rather than silently falling back to the dense multi-GPU processor).
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus. Unused at ulysses_degree = ring_degree = 1.
fsdp_dit            = False
fsdp_text_encoder   = False
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with sequential_cpu_offload.
compile_dit         = False

# model path
model_name          = "models/Diffusion_Transformer/MiniMax-H3"

# ---- FastVideo VSA sparse-student config --------------------------------------------------
# Official FastH3 8-step DMD student (FastVideo/FastVideo-FastH3-8-Step-V2): a MiniMax-H3 backbone
# distilled with data-free DMD2 + VSA-H3 at 80% sparsity, carrying a trained `to_gate_compress` gate
# per block. `model_name` still supplies the pipeline components (text encoder / VAEs / tokenizer /
# schedulers) from the dense base; the transformer comes from `transformer_path` below.
# student_topk is the keep-ratio of key blocks per query (0.20 -> 80% sparsity, the official
# vsa_sparsity=0.8); set it to 1.0 to fall back to dense attention.
student_topk        = 0.20

# Transformer weights. Accepts either the official FastH3 diffusers transformer folder (`.../transformer`,
# sharded; its gate keys `attn.to_gate_compress.*` are remapped to this repo's `attn.attn_op.to_gate_compress.*`
# on load), or a single-file native-layout state_dict from this repo's own training (loaded AFTER
# set_student_topk so the trained gate overwrites the zero init). Set to None to run the sparse student on
# the dense base weights (zero-init gate).
transformer_path    = "models/Diffusion_Transformer/FastVideo-FastH3-8-Step-V2/transformer"
vae_path            = None
lora_path           = None
pdd_lora_path       = None

# Other params
# MiniMax-H3 generates at a fixed 24 fps, only accepts multiples of 32 as height / width, and snaps video_length up
# to the next 17 * n + 5 the video VAE can decode (the duration has to stay between 5 and 15 seconds).
# Leave sample_size as None to use MiniMax-H3's own 16:9 canvas (768x1344).
sample_size         = [704, 1280]
video_length        = 124
fps                 = 24

# Use torch.float16 if GPU does not support torch.bfloat16
# Some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
prompt              = "A red fox trotting through a snowy pine forest, snow crunching underfoot"
seed                = 43
# Number of denoising steps, i.e. of model evaluations. The official FastH3-8-Step-V2 runs eight
# transformer forwards (fastvideo_inference.json: transformer_forwards=8).
num_inference_steps = 8
# The released checkpoint is guidance-distilled: leave guidance_scale at 1 to run one forward pass per step
# with no CFG. A value above 1 enables classifier-free guidance with a negative_prompt, running two passes.
guidance_scale      = 1
# The exponential sigma shifts of the two schedules. The official 8-step DMD grid
# (dmd_denoising_steps [999,874,...,125]) is *uniform*; here a uniform sigma grid is produced by
# shift=1.0 (set_shift warps the grid, so the checkpoint's video shift 10 / audio 3 do not map onto
# flow_shift directly). flow_shift=1.0 reproduces the trained video schedule; audio keeps shift 3.
flow_shift          = 1.0
audio_flow_shift    = 3.0
lora_weight         = 0.55
save_path           = "samples/fasth3-8step-videos-t2v"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)

# Transformer: build the VSA sparse student, then REBUILD the VSA kernels at the requested keep-ratio.
# set_student_topk recreates attn_op (resetting to_gate_compress to its zero init), so it must run
# BEFORE any trained gate is loaded. `transformer_path` may be the official FastH3 diffusers transformer
# folder (sharded) or a single-file native-layout state_dict; when it is a folder the backbone comes from
# it directly, otherwise from the dense `model_name` base (whose transformer lives in `transformer/`).
is_transformer_dir = transformer_path is not None and os.path.isdir(transformer_path)
transformer = FastVideoMiniMaxH3Transformer3DModel.from_pretrained(
    transformer_path if is_transformer_dir else model_name,
    subfolder="" if is_transformer_dir else "transformer",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
transformer.set_student_topk(student_topk)

# The official FastH3 checkpoint is sharded and names its gate `attn.to_gate_compress.*`; this repo
# nests it under `attn.attn_op.to_gate_compress.*`. from_pretrained above loaded the (distilled)
# backbone and dropped the gate keys as unexpected, so load just the gate here, remapped, and assert
# every block got one (a silently zero gate would leave the compressed branch untrained -> artifacts).
if is_transformer_dir and student_topk < 1.0:
    import json
    from safetensors import safe_open
    tdir = transformer_path
    weight_map = json.load(open(os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    shard_to_gate_keys = {}
    for key, shard in weight_map.items():
        if "to_gate_compress" in key:
            shard_to_gate_keys.setdefault(shard, []).append(key)
    gate_sd = {}
    for shard, keys in shard_to_gate_keys.items():
        with safe_open(os.path.join(tdir, shard), framework="pt") as f:
            for key in keys:
                gate_sd[key.replace("attn.to_gate_compress", "attn.attn_op.to_gate_compress")] = f.get_tensor(key)
    m, u = transformer.load_state_dict(gate_sd, strict=False)
    gate_loaded = sum(1 for k in gate_sd if k not in m)
    print(f"FastH3 gate: {gate_loaded}/{len(gate_sd)} loaded, unexpected {len(u)}", flush=True)
    assert len(u) == 0 and gate_loaded == len(gate_sd), (
        f"FastH3 gate remap incomplete: loaded {gate_loaded}/{len(gate_sd)}, unexpected {len(u)}.")

# Single-file native-layout state_dict (gate already under attn_op): overwrite the base backbone wholesale.
if transformer_path is not None and not is_transformer_dir:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if isinstance(state_dict, dict) and "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    attn_op_missing = [k for k in m if "attn_op." in k]
    if attn_op_missing and student_topk < 1.0:
        # A trained VSA student should provide every attn_op param; missing ones mean the compressed
        # branch would run on freshly-init (zero) gate values. Fail loudly instead of silently.
        raise RuntimeError(
            f"{len(attn_op_missing)} VSA attn_op keys are missing from the checkpoint "
            f"(student_topk={student_topk}); e.g. {attn_op_missing[:4]}. Check the checkpoint / topk.")
    # `strict=False` accepts a file whose keys belong to another model — a LoRA checkpoint, say — by
    # loading nothing at all, so an unexpected key is a hard error.
    assert len(u) == 0, (
        f"{transformer_path} holds {len(u)} key(s) the transformer does not have, e.g. {list(u)[:3]}. A LoRA "
        "checkpoint belongs in `lora_path`, not `transformer_path`."
    )

pdd_config = None
if pdd_lora_path is not None:
    if lora_path is not None:
        raise ValueError("`lora_path` and `pdd_lora_path` cannot be used together.")
    from videox_fun.models.minimax_h3_pdd import (load_pdd_lora,
                                                  pdd_num_inference_steps,
                                                  pdd_step_callback)
    pdd_config = load_pdd_lora(transformer, pdd_lora_path)
    num_inference_steps = pdd_num_inference_steps(pdd_config, num_inference_steps, teacher_default=40)

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
        from safetensors.torch import load_file
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
    convert_model_weight_to_float8(transformer, exclude_module_name=fp8_exclude_module_name, device=device)

# The pure *cpu_offload modes (qfloat8 variants excluded, unsupported under FSDP). Under sequence parallelism the
# sharded modules offload through FSDP's own CPUOffload, enabled here at wrap time; the pipeline then routes the
# whole-model offload call to the non-FSDP VAE / audio VAE automatically, so no per-mode branch is needed below.
wants_offload = GPU_memory_mode in ("model_cpu_offload", "model_group_offload", "sequential_cpu_offload")

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    # NOTE: raises for the VSA student (no sequence-parallel path); keep ulysses_degree=ring_degree=1.
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        fp32_modules = [m for m in transformer.modules()
                        if any(p.dtype == torch.float32 for p in m.parameters(recurse=False))]
        shard_fn = partial(shard_model, device_id=device, param_dtype=None, cast_dtype=False,
                          module_to_wrapper=list(transformer.transformer_blocks),
                          ignored_modules=fp32_modules, offload_to_cpu=wants_offload)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype,
                          module_to_wrapper=list(text_encoder.model.language_model.layers),
                          offload_to_cpu=wants_offload)
        pipeline.text_encoder.model = shard_fn(pipeline.text_encoder.model)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.transformer_blocks)):
        pipeline.transformer.transformer_blocks[i] = torch.compile(pipeline.transformer.transformer_blocks[i])
    print("Add Compile")

# Quantize (when the mode carries an "_and_<quant>" suffix) and then place the pipeline.
# The order lives inside the helper: quantization has to happen before the offload hooks
# are installed.
# The FP8 conversion above has already run (before the FSDP sharding, on purpose); only the dequant
# wrapper and the memory placement are left, which is what the preconverted tag installs.
apply_gpu_memory_mode(pipeline, GPU_memory_mode, device, weight_dtype,
                      quant_tag="qfloat8_preconverted" if GPU_memory_mode.endswith("_and_qfloat8") else None,
                      exclude_module_name=[])

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

pdd_callback = None if pdd_config is None else pdd_step_callback(
    transformer, scheduler, audio_scheduler, pdd_config, num_inference_steps
)

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
        callback_on_step_end=pdd_callback,
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
