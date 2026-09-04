# VideoX-Fun Integration Skeletons

Starting templates. **Always open the mirrored family's real file and adapt it** — these skeletons show shape and required reuse points, not full implementations. Replace `<family>` / `<Family>` / `<task>`.

## Config — `config/<family>/<variant>.yaml` (optional)

> **Not always required.** Author a YAML only for civitai-format / custom single-file layouts. A standard diffusers-layout checkpoint (`model_index.json` + per-subfolder `config.json`) loads directly via `from_pretrained(model_name, subfolder=...)` with no YAML — set `config_path = None` and guard `if config_path is not None:` (see `examples/minimax_h3_fun/predict_v2v_control.py`).

```yaml
format: civitai
pipeline: <Family>
transformer_additional_kwargs:
  transformer_subpath: ./
  dict_mapping:
    in_dim: in_channels
    dim: hidden_size

vae_kwargs:
  vae_subpath: <Family>_VAE.pth
  temporal_compression_ratio: 4
  spatial_compression_ratio: 8

text_encoder_kwargs:
  text_encoder_subpath: <text_encoder>.pth
  tokenizer_subpath: <tokenizer_id>
  text_length: 512

scheduler_kwargs:
  scheduler_subpath: null
  num_train_timesteps: 1000
  shift: 5.0

# Only for i2v / models with a CLIP image encoder:
image_encoder_kwargs:
  image_encoder_subpath: <image_encoder>.pth
```

## Inference — `examples/<family>/predict_<task>.py`

```python
import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

# --- sys.path bootstrap (required, before importing videox_fun) ---
current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKL<Family>, <Family>TextEncoder,
                               <Family>Transformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import <Family>Pipeline
from videox_fun.utils import register_auto_device_hook, safe_enable_group_offload
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                    save_videos_grid)

# --- user config block (keep the conventional order + comments) ---
GPU_memory_mode     = "sequential_cpu_offload"
ulysses_degree      = 1
ring_degree         = 1
fsdp_dit            = False
fsdp_text_encoder   = True
compile_dit         = False
enable_teacache     = True
teacache_threshold  = 0.10
num_skip_start_steps = 5
teacache_offload    = False
cfg_skip_ratio      = 0
enable_riflex       = False
riflex_k            = 6
config_path         = "config/<family>/<variant>.yaml"
model_name          = "models/Diffusion_Transformer/<Family>-Model"
sampler_name        = "Flow"
shift               = 3
transformer_path    = None
vae_path            = None
lora_path           = None
sample_size         = [480, 832]
video_length        = 81
fps                 = 16
weight_dtype        = torch.bfloat16
prompt              = "..."
negative_prompt     = "..."
guidance_scale      = 6.0
seed                = 43
num_inference_steps = 50
lora_weight         = 0.55
save_path           = "samples/<family>-<task>"

# --- device + config (config_path may be None for a diffusers-layout checkpoint) ---
device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)   # or guard: if config_path is not None: ... (then load components via subfolder=...)

# --- components (when a YAML is used, paths/kwargs come from config; otherwise pass subfolder=... directly) ---
transformer = <Family>Transformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True, torch_dtype=weight_dtype,
)
# optional transformer_path / vae_path override -> load_state_dict(strict=False) + print missing/unexpected
vae = AutoencoderKL<Family>.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')))
text_encoder = <Family>TextEncoder.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True, torch_dtype=weight_dtype).eval()

# --- scheduler selection dict ---
Chosen_Scheduler = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
scheduler = Chosen_Scheduler(**filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs'])))

# --- pipeline ---
pipeline = <Family>Pipeline(vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
                            transformer=transformer, scheduler=scheduler)

# --- multi-gpu / fsdp / compile ---
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        pipeline.transformer = partial(shard_model, device_id=device, param_dtype=weight_dtype)(pipeline.transformer)
    if fsdp_text_encoder:
        pipeline.text_encoder = partial(shard_model, device_id=device, param_dtype=weight_dtype)(pipeline.text_encoder)
if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])

# --- GPU_memory_mode branching (keep this exact order) ---
if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_group_offload":
    register_auto_device_hook(pipeline.transformer)
    safe_enable_group_offload(pipeline, onload_device=device, offload_device="cpu", offload_type="leaf_level", use_stream=True)
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

# --- teacache / cfg_skip / riflex / lora ---
coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    pipeline.transformer.enable_teacache(coefficients, num_inference_steps, teacache_threshold,
                                         num_skip_start_steps=num_skip_start_steps, offload=teacache_offload)
if cfg_skip_ratio is not None:
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
generator = torch.Generator(device=device).manual_seed(seed)
if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

# --- inference ---
with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    if enable_riflex:
        pipeline.transformer.enable_riflex(k=riflex_k, L_test=(video_length - 1) // vae.config.temporal_compression_ratio + 1)
    sample = pipeline(prompt, num_frames=video_length, negative_prompt=negative_prompt,
                      height=sample_size[0], width=sample_size[1], generator=generator,
                      guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                      shift=shift).videos
if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

# --- save (rank 0 only when multi-gpu) ---
def save_results():
    os.makedirs(save_path, exist_ok=True)
    prefix = str(len(os.listdir(save_path)) + 1).zfill(8)
    if video_length == 1:
        image = (sample[0, :, 0].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
        Image.fromarray(image).save(os.path.join(save_path, prefix + ".png"))
    else:
        save_videos_grid(sample, os.path.join(save_path, prefix + ".mp4"), fps=fps)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results()
```

For i2v, gate the CLIP image encoder and pass `video`/`mask_video`:
```python
if transformer.config.in_channels != vae.config.latent_channels:
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder'))).to(weight_dtype).eval()
    input_video, input_video_mask, _ = get_image_to_video_latent(start_image, None, video_length=video_length, sample_size=sample_size)
    # pipeline = <Family>InpaintPipeline(..., clip_image_encoder=clip_image_encoder)
    # sample = pipeline(..., video=input_video, mask_video=input_video_mask).videos
```

## Pipeline class — `videox_fun/pipeline/pipeline_<family>.py`

```python
from dataclasses import dataclass
from typing import List, Optional, Union
import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor

from ..models import AutoencoderKL<Family>, <Family>Transformer3DModel
from ..utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.get_logger(__name__)
EXAMPLE_DOC_STRING = """Examples:\n```python\npass\n```"""

# reuse retrieve_timesteps verbatim from pipeline_wan.py

@dataclass
class <Family>PipelineOutput(BaseOutput):
    videos: torch.Tensor

class <Family>Pipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(self, tokenizer, text_encoder, vae, transformer, scheduler):
        super().__init__()
        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
                              transformer=transformer, scheduler=scheduler)
        # video_processor / vae_scale_factor / etc. as in pipeline_wan.py

    def encode_prompt(self, prompt, negative_prompt, device, num_videos_per_prompt=1, ...):
        ...  # mirror pipeline_wan.py

    def prepare_latents(self, batch_size, num_channels_latents, height, width, num_frames, dtype, device, generator, latents=None):
        ...

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(self, prompt, negative_prompt=None, height=480, width=832, num_frames=81,
                 num_inference_steps=50, guidance_scale=6.0, generator=None, shift=1.0,
                 callback_on_step_end=None, return_dict=True, **kwargs) -> Union[<Family>PipelineOutput, tuple]:
        # 1. encode_prompt  2. prepare_latents  3. retrieve_timesteps
        # 4. denoising loop with guidance  5. vae.decode  6. return <Family>PipelineOutput(videos=...)
        ...
```
Then register in `videox_fun/pipeline/__init__.py`:
```python
from .pipeline_<family> import <Family>Pipeline
```

## Model class — `videox_fun/models/<family>_transformer3d.py`

```python
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from .attention_utils import attention   # unified FA/SDPA backend — do not hand-roll SDPA

class <Family>Transformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, model_type='t2v', in_dim=16, dim=2048, ffn_dim=8192,
                 num_heads=16, num_layers=32, in_channels=16, hidden_size=2048, ...):
        super().__init__()
        ...

    def _set_gradient_checkpointing(self, *args, **kwargs):
        self.gradient_checkpointing = True

    def enable_multi_gpus_inference(self): ...   # route attn through dist/<family>_xfuser.py
    def enable_teacache(self, ...): ...
    def enable_cfg_skip(self, ...): ...

    def forward(self, x, timestep, context, ...): ...

    @classmethod
    def from_pretrained(cls, pretrained_model_path, subfolder=None,
                        transformer_additional_kwargs=None, low_cpu_mem_usage=False,
                        torch_dtype=torch.bfloat16):
        ...  # mirror wan_transformer3d.py: config.json -> dict_mapping -> init_empty_weights
             # -> load .bin/.safetensors -> shape-filter -> initialize missing keys -> load
```
Then register in `videox_fun/models/__init__.py`:
```python
from .<family>_transformer3d import <Family>Transformer3DModel
from .<family>_vae import AutoencoderKL<Family>
```

## Training — `scripts/<family>/train.py` (key reuse points)

```python
"""Modified from https://github.com/huggingface/diffusers/.../train_text_to_image.py"""
import argparse, gc, logging, math, os, sys
import accelerate, diffusers, torch, transformers
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf

# same sys.path bootstrap as predict scripts
from videox_fun.data import (ASPECT_RATIO_512, AspectRatioBatchImageVideoSampler,
                             ImageVideoDataset, ImageVideoSampler, RandomSampler,
                             get_closest_ratio, get_random_mask)
from videox_fun.models import AutoencoderKL<Family>, <Family>Transformer3DModel
from videox_fun.pipeline import <Family>Pipeline          # REUSED for validation
from videox_fun.utils.lora_utils import create_network     # for train_lora
from videox_fun.utils.utils import save_videos_grid, get_image_to_video_latent

def log_validation(vae, text_encoder, tokenizer, transformer3d, args, config,
                   accelerator, weight_dtype, global_step):
    # build <Family>Pipeline from accelerator.unwrap_model(transformer3d),
    # run validation_prompts, save_videos_grid to output_dir/sample/. Reuse the pipeline.
    ...

def parse_args():
    parser = argparse.ArgumentParser(...)
    # reuse the shared arg surface: --config_path, --pretrained_model_name_or_path,
    # --train_data_dir, --train_data_meta, --video_sample_n_frames, --train_batch_size,
    # --gradient_accumulation_steps, --learning_rate, --lr_scheduler, --checkpointing_steps,
    # --output_dir, --mixed_precision, --gradient_checkpointing, --enable_bucket,
    # --train_mode, --trainable_modules, --validation_prompts ... (add only what's needed)
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision, ...)
    config = OmegaConf.load(args.config_path)
    # load transformer/vae/text_encoder via config

    # --- Dataset: pick by task (see reference.md §8) ---
    #   T2V/I2V base + inpaint  -> ImageVideoDataset(enable_inpaint = args.train_mode != "normal")
    #   Control                -> ImageVideoControlDataset(enable_camera_info = ...)
    #   Image edit             -> ImageEditDataset
    #   Speech/audio (S2V)     -> VideoSpeechDataset / VideoSpeechControlDataset
    #   Animate                -> VideoAnimateDataset
    #   Distill text / GRPO / DPO -> TextDataset
    # Smoke-test on the matching official demo dataset (reference.md §8), e.g.
    #   datasets/X-Fun-Videos-Demo + metadata_add_width_height.json for T2V/I2V.
    train_dataset = ImageVideoDataset(
        args.train_data_meta, args.train_data_dir,
        video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride,
        video_sample_n_frames=args.video_sample_n_frames, video_repeat=args.video_repeat,
        image_sample_size=args.image_sample_size, enable_bucket=args.enable_bucket,
        enable_inpaint=True if args.train_mode != "normal" else False)

    # --- Sampler + DataLoader: branch on enable_bucket (see reference.md §8) ---
    batch_sampler_generator = torch.Generator().manual_seed(args.seed)
    if args.enable_bucket:
        aspect_ratio_sample_size = {k: [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[k]] for k in ASPECT_RATIO_512}
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset,
            batch_size=args.train_batch_size, train_folder=args.train_data_dir, drop_last=True,
            aspect_ratios=aspect_ratio_sample_size)
        def collate_fn(examples):
            new_examples = {"pixel_values": [], "text": []}
            if args.train_mode != "normal":
                new_examples.update({"mask_pixel_values": [], "mask": [], "clip_pixel_values": []})
            # get_closest_ratio -> Resize/CenterCrop/Normalize -> stack; masks via get_random_mask
            return new_examples
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index))
    else:
        batch_sampler = ImageVideoSampler(RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, args.train_batch_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=batch_sampler, num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index))

    # trainable-module filtering or create_network for LoRA
    # optimizer + get_scheduler; accelerator.prepare; checkpoint hooks
    # training loop: timestep sampling -> transformer forward -> loss -> backward
    # periodic log_validation(...); final save weights / LoRA
    ...

if __name__ == "__main__":
    main()
```

## Launcher — `scripts/<family>/train.sh`

```bash
export MODEL_NAME="models/Diffusion_Transformer/<Family>-Model"
# Test data = the official demo dataset matching the task (reference.md §8). Download once, e.g.:
#   modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
#   T2I -> X-Fun-Images-Demo | control -> X-Fun-{Videos,Images}-Controls-Demo
#   S2V -> X-Fun-Videos-Audios-Demo | image edit -> X-Fun-Images-Edit-Demo
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"  # = train_data_dir (data_root); media live under train/
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"  # = train_data_meta: [{"file_path","text","type","width","height"}] — see reference.md §8
# Metadata variants: VACE/subject-ref -> metadata_add_width_height_add_objects.json (X-Fun-Videos-Controls-Demo);
# audio-visual joint -> metadata_add_width_height_add_wav.json; lingbot_video -> metadata_lingbot_video_add_width_height.json
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/<family>/train.py \
  --config_path="config/<family>/<variant>.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --checkpointing_steps=50 \
  --output_dir="output_dir_<family>" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --enable_bucket \
  --low_vram \
  --train_mode="normal" \
  --trainable_modules "."
```

## Preprocessing (data gen) — `scripts/<family>/generate_<...>.py`

Offline generation of cached training data (latents / ODE-trajectory pairs / prompt embeddings). **Always multi-GPU** (`accelerate launch` + `Accelerator`) and **always safetensors** (`safetensors.torch.save_file` + an `outputs.json` index for `ImageVideoSafetensorsDataset`) — never LMDB, never `.pt`. Mirror `scripts/wan2.1_self_forcing/generate_ode_pairs.py`:

```python
# ...license header + sys.path bootstrap...
import argparse, json, math, os, torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import save_file
from tqdm import tqdm
from videox_fun.models import AutoencoderKLWan, WanT5EncoderModel, WanTransformer3DModel  # reuse repo models
from videox_fun.utils.utils import save_videos_grid                                         # reuse repo IO

def main():
    args = parse_args()   # --pretrained_model_name_or_path --config_path --caption_path --output_folder
                          # --num_inference_steps --guidance_scale --shift --mixed_precision ...
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device, world_size, rank = accelerator.device, accelerator.num_processes, accelerator.process_index
    torch.set_grad_enabled(False)                        # inference-only
    torch.backends.cuda.matmul.allow_tf32 = True

    config = OmegaConf.load(args.config_path)            # config-driven loading (Section 3)
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(accelerator.mixed_precision, torch.float32)
    text_encoder = WanT5EncoderModel.from_pretrained(..., additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']), torch_dtype=weight_dtype).to(device).eval()
    vae          = AutoencoderKLWan.from_pretrained(..., additional_kwargs=OmegaConf.to_container(config['vae_kwargs'])).to(device, dtype=weight_dtype).eval()
    transformer  = WanTransformer3DModel.from_pretrained(..., transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs'])).to(device, dtype=weight_dtype).eval()

    prompts = [l.rstrip() for l in open(args.caption_path, encoding="utf-8") if l.strip()]
    os.makedirs(args.output_folder, exist_ok=True)
    total_per_rank = math.ceil(len(prompts) / world_size)

    for index in tqdm(range(total_per_rank), disable=rank != 0, desc="Generating"):
        prompt_index = index * world_size + rank          # interleaved multi-GPU shard
        if prompt_index >= len(prompts):
            continue
        out_path = os.path.join(args.output_folder, f"{prompt_index:05d}.safetensors")
        if os.path.exists(out_path):                      # resume: skip already-done samples
            continue
        prompt = prompts[prompt_index]
        # ... encode prompt, sample noise, run the teacher ODE (CFG), collect latents ...
        save_file(                                        # safetensors ONLY (no lmdb / no .pt)
            {"latents": latents.cpu(), "prompt_embeds": text_embeds.cpu(), "prompt_attention_mask": mask.cpu()},
            out_path, metadata={"prompt": prompt},
        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:                       # rank-0 writes the JSON index
        entries = [{"file_path": os.path.join(args.output_folder, f"{i:05d}.safetensors")}
                   for i in range(len(prompts))
                   if os.path.exists(os.path.join(args.output_folder, f"{i:05d}.safetensors"))]
        json.dump(entries, open(os.path.join(args.output_folder, "outputs.json"), "w"), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
```

Launcher (`generate_<...>.sh`) — `accelerate launch` uses every visible GPU:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
accelerate launch --mixed_precision="bf16" scripts/<family>/generate_<...>.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --config_path="config/<family>/*.yaml" \
  --caption_path="datasets/prompts.txt" \
  --output_folder="datasets/<family>_ode_pairs" \
  --num_inference_steps=48 --guidance_scale=6.0 --shift=8.0
```

Training then reads the cache with `ImageVideoSafetensorsDataset(ann_path=".../outputs.json")` (single-file mode `{"file_path": ...}`, or per-tensor mode via `--save_per_tensor`). See reference.md §10.

> Dataset *curation* (scoring/filtering/captioning under `videox_fun/video_caption/`) is a different activity: also multi-GPU (accelerate `PartialState.split_between_processes`/`gather_object`, or vLLM tensor-parallel) but writes csv/jsonl metadata, not safetensors. See reference.md §10 “Related but different”.
