import os
import sys

import torch

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLMiniMaxH3Image,
                               MiniMaxH3ImageTransformer2DModel,
                               Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor)
from videox_fun.models.minimax_h3_image_transformer2d import \
    resolve_image_transformer_component
from videox_fun.pipeline import MiniMaxH3ImagePipeline
from videox_fun.utils import (MiniMaxH3Scheduler, apply_gpu_memory_mode,
                              convert_model_weight_to_float8, get_image,
                              merge_lora, unmerge_lora,
                              validate_image_lora_head)

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, model_group_offload, sequential_cpu_offload, text_encoder_offload, text_encoder_offload_and_qfloat8].
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
# 
# text_encoder_offload keeps only the text encoder in CPU RAM, streaming it to the GPU module by module for
# its single encode pass, so the transformer stays resident (add `_and_qfloat8` for float8 storage). 
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
fsdp_dit            = True
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with sequential_cpu_offload.
compile_dit         = False

# Model path. Prefer a native `transformer/` when present; otherwise use `transformer_diffusers/`.
model_name          = "models/Diffusion_Transformer/MiniMax-H3-Image"
# `single` uses the original shared projection. Set `pdd` explicitly for the released six-step heads.
output_head_type    = "single"
# [NOTE]: Noise schedule shift parameter. None keeps the released default (12).
# PDD uses its fixed six-step schedule and requires flow_shift = None.
flow_shift          = 12

# Load pretrained model if need
# A full finetune goes in `transformer_path`, either as the `transformer` folder a training checkpoint writes
# (`output_dir_minimax_h3_image/checkpoint-N/transformer`, config.json included) or as a single safetensors file.
# A LoRA goes in `lora_path`: handed to `transformer_path` it would match no key at all and load nothing.
transformer_path    = None
vae_path            = None
lora_path           = None

# Other params
# MiniMax-H3-Image rounds both sides of its canvas to a multiple of 32 and never generates below
# `min_generation_area` pixels: a smaller request is generated on a canvas scaled up to that area at the same
# aspect ratio and resized back with LANCZOS, so `sample_size` is the size of the returned image.
sample_size         = [1024, 1024]
min_generation_area = 2048 * 2048

# Use torch.float16 if GPU does not support torch.bfloat16
# Some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype        = torch.bfloat16
# The reference images, read in the order they are listed here: that order is what labels them `<Picture 1>`,
# `<Picture 2>`, ... in the prompt and what lays them out on the rotary clock. At most 16 of them, each normalized
# onto a short edge of 2048 whatever `sample_size` asks for.
images              = ["asset/ref_1.png"]
prompt              = "Turn the camera into a watermelon, keep the composition and the lighting."
seed                = 43
# PDD keeps its fixed six-step schedule: use 6 or None with output_head_type="pdd".
num_inference_steps = 40
lora_weight         = 0.55
save_path           = "samples/minimax-h3-image-i2i-edit"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)

# Guidance is distilled into the released backbone in both output-head modes.
transformer_dir = resolve_image_transformer_component(model_name)
print(f"Loading Image transformer from: {transformer_dir}", flush=True)
transformer = MiniMaxH3ImageTransformer2DModel.from_pretrained(
    transformer_dir,
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
    output_head_type=output_head_type,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if os.path.isdir(transformer_path):
        # A training checkpoint's `transformer` folder carries its own config.json, so the loader restores the
        # mixed-precision contract of the checkpoint (`_keep_in_fp32_modules`) by itself.
        transformer = MiniMaxH3ImageTransformer2DModel.from_pretrained(
            transformer_path,
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
            output_head_type=output_head_type,
        )
    else:
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
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

if lora_path is not None:
    if os.path.isfile(lora_path) and lora_path.endswith(".safetensors"):
        if not validate_image_lora_head(lora_path, output_head_type,
                                        getattr(transformer.config, "single_head_sha256", None)):
            print("Warning: legacy LoRA has no output-head metadata.")

# Image VAE. The released weights are float32 and the decode runs under float16 autocast, so the VAE is not
# downcast even when the rest of the pipeline is bfloat16 (this is also how the training scripts load it).
vae = AutoencoderKLMiniMaxH3Image.from_pretrained(
    model_name,
    subfolder="vae_diffusers",
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

# Get Scheduler. The grid itself is fixed by the checkpoint; only its update form comes from the released config.
scheduler = MiniMaxH3Scheduler.from_pretrained(os.path.join(model_name, "scheduler"))

pipeline = MiniMaxH3ImagePipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    processor=processor,
    transformer=transformer,
    scheduler=scheduler,
)

# The float32 modules of the mixed-precision checkpoint stay untouched by the float8 quantization.
fp8_exclude_module_name = [
    "proj_in", "context_embedder", "time_embedder", "time_proj", "rope",
    "token_refiner", "norm_out", "proj_out", "pdd_proj_out",
]
use_qfloat8 = "qfloat8" in GPU_memory_mode
if use_qfloat8:
    convert_model_weight_to_float8(transformer, exclude_module_name=fp8_exclude_module_name, device=device)

# The pure *cpu_offload modes (qfloat8 variants excluded, unsupported under FSDP). Under sequence parallelism the
# sharded modules offload through FSDP's own CPUOffload, enabled here at wrap time; the pipeline then routes the
# whole-model offload call to the non-FSDP VAE automatically, so no per-mode branch is needed below.
wants_offload = GPU_memory_mode in ("model_cpu_offload", "model_group_offload", "sequential_cpu_offload")
# text_encoder_offload keeps the transformer resident and offloads only the conditioner, so under FSDP the text
# encoder is wrapped with CPUOffload while the DiT stays on the card; match the base to cover `_and_qfloat8`.
te_wants_offload = wants_offload or GPU_memory_mode.startswith("text_encoder_offload")

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
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
                          offload_to_cpu=te_wants_offload)
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

images = [get_image(image) for image in images]

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=weight_dtype,
                                     enabled=weight_dtype in (torch.float16, torch.bfloat16)):
    sample = pipeline(
        prompt=prompt,
        images=images,
        height=sample_size[0],
        width=sample_size[1],
        min_generation_area=min_generation_area,
        generator=generator,
        num_inference_steps=num_inference_steps,
        flow_shift=flow_shift,
    ).images

print(f"[{os.environ.get('RANK', '0')}] generation done", flush=True)

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

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
    # Keep every rank alive until the saving rank finishes; an early exit of one rank makes the elastic launcher
    # terminate the others.
    dist.barrier()
else:
    save_results()
