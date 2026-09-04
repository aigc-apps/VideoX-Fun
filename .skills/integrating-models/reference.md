# VideoX-Fun Integration Reference

Detailed conventions per layer. Read the mirrored family's real files alongside this — the existing code is always the source of truth.

## 1. Model definitions — `videox_fun/models/<family>_*.py`

### File naming
- Transformer / DiT: `<family>_transformer3d.py` (video) or `<family>_transformer2d.py` (image). Variants append a suffix: `_control`, `_s2v`, `_vace`, `_animate`, `_self_forcing`, `_avatar`.
- VAE: `<family>_vae.py` → class `AutoencoderKL<Family>`.
- Encoders: `<family>_text_encoder.py`, `<family>_audio_encoder.py`, `<family>_image_encoder.py`.

### Class shape (mirror `wan_transformer3d.py`)
```python
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin

class <Family>Transformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, model_type='t2v', patch_size=(1,2,2), in_dim=16, dim=2048,
                 ffn_dim=8192, num_heads=16, num_layers=32, in_channels=16,
                 hidden_size=2048, ...):
        super().__init__()
        ...
```
- Keep BOTH civitai names (`in_dim`, `dim`, `ffn_dim`) and diffusers aliases (`in_channels`, `hidden_size`) in `__init__` so either format maps cleanly.
- Implement `_set_gradient_checkpointing(self, *args, **kwargs)`.
- Attention must go through `videox_fun.models.attention_utils.attention` (backend-agnostic), not a hand-rolled `scaled_dot_product_attention`.
- Multi-GPU: expose `enable_multi_gpus_inference()` and route attention through the family's `dist/<family>_xfuser.py` processor.
- Speedups live on the model: `enable_teacache(...)`, `enable_cfg_skip(...)`, `enable_riflex(...)`.

### `from_pretrained` internals (do not simplify)
The custom classmethod must keep these behaviors (see `wan_transformer3d.py::from_pretrained`):
1. Accept `transformer_additional_kwargs`, `subfolder`, `low_cpu_mem_usage`, `torch_dtype`.
2. Read `config.json`; auto-convert foreign configs (e.g. diffsynth `has_image_input`) via a `_convert_from_*_config` helper.
3. Apply `dict_mapping`: pop it from kwargs, then for each `key: target` set `kwargs[target] = config[key]`.
4. Under `low_cpu_mem_usage`, build with `accelerate.init_empty_weights()`, load `.bin`/`.safetensors` (single file or glob all shards), and **filter by exact shape match** before loading.
5. Initialize missing keys deliberately: zero-init control/audio projections (`after_proj`, `before_proj`, `processor.k_proj/v_proj`, `audio_injector`, `cond_encoder`, ...), ones for norms, xavier for ≥2D weights, so new branches start as no-ops.

### Registry — `videox_fun/models/__init__.py`
Add an import line for every new public class, grouped with the family. Wrap optional-dependency imports in `try/except` with a helpful upgrade message (see the Qwen2.5-VL / Mistral3 blocks at the top).

## 2. Pipelines — `videox_fun/pipeline/pipeline_<family>*.py`

Mirror `pipeline_wan.py`. Required pieces:
- Module-level `retrieve_timesteps(scheduler, num_inference_steps, device, timesteps, sigmas, **kwargs)` (copied from diffusers) — reuse verbatim.
- `EXAMPLE_DOC_STRING` for the `@replace_example_docstring` decorator.
- Output dataclass:
  ```python
  @dataclass
  class <Family>PipelineOutput(BaseOutput):
      videos: torch.Tensor
  ```
- Pipeline class:
  ```python
  class <Family>Pipeline(DiffusionPipeline):
      _optional_component = [...]
      model_cpu_offload_seq = "text_encoder->transformer->vae"   # order matters for offload
      _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
      def __init__(self, tokenizer, text_encoder, vae, transformer, scheduler, ...): ...
      def encode_prompt(...): ...
      def prepare_latents(...): ...
      @torch.no_grad()
      @replace_example_docstring(EXAMPLE_DOC_STRING)
      def __call__(self, prompt, negative_prompt=..., height=..., width=...,
                   num_frames=..., num_inference_steps=..., guidance_scale=...,
                   generator=None, ..., return_dict=True) -> Union[<Family>PipelineOutput, Tuple]: ...
  ```
- Import schedulers from `..utils.fm_solvers` / `..utils.fm_solvers_unipc`, models from `..models`.
- Separate pipelines per task: base (`pipeline_<family>.py`), inpaint/i2v (`_inpaint`), control (`_control`), s2v, etc. Register all in `videox_fun/pipeline/__init__.py`, adding convenience aliases (e.g. `WanI2VPipeline = WanFunInpaintPipeline`) where existing code expects them.

## 3. Config — `config/<family>/<name>.yaml` (optional)

**The YAML is not mandatory.** Decide by checkpoint layout:
- **Required** for civitai-format / custom single-file layouts, where weights and key names are not diffusers-native. The YAML supplies `transformer_additional_kwargs` (incl. `dict_mapping` mapping civitai config keys → model `__init__` kwargs), component `*_subpath`s, and `vae/text_encoder/scheduler/image_encoder` kwargs.
- **Optional** for a standard diffusers-layout checkpoint (`model_index.json` + each subfolder carrying its own `config.json`). Load components directly: `<Family>Transformer3DModel.from_pretrained(model_name, subfolder="transformer", low_cpu_mem_usage=True, torch_dtype=...)`, `AutoencoderKL<Family>.from_pretrained(model_name, subfolder="vae")`, etc. Guard the config path exactly like `examples/minimax_h3_fun/predict_v2v_control.py`:
  ```python
  transformer_load_kwargs = {}
  if config_path is not None:
      from omegaconf import OmegaConf
      config = OmegaConf.load(config_path)
      transformer_load_kwargs.update(OmegaConf.to_container(config["transformer_additional_kwargs"], resolve=True))
  transformer = <Family>Transformer3DModel.from_pretrained(model_name, subfolder="transformer", **transformer_load_kwargs, ...)
  ```

When you do use a YAML, the canonical schema is below (see `config/wan2.1/wan_civitai.yaml`):
```yaml
format: civitai            # or diffusers — selects weight-key handling
pipeline: Wan              # family label consumed by API/ComfyUI loaders
transformer_additional_kwargs:
  transformer_subpath: ./  # subfolder under model_name holding the DiT
  dict_mapping:            # civitai config key -> model __init__ kwarg
    in_dim: in_channels
    dim: hidden_size
vae_kwargs:
  vae_subpath: Wan2.1_VAE.pth
  temporal_compression_ratio: 4
  spatial_compression_ratio: 8
text_encoder_kwargs:
  text_encoder_subpath: models_t5_umt5-xxl-enc-bf16.pth
  tokenizer_subpath: google/umt5-xxl
  text_length: 512
  ...
scheduler_kwargs:
  scheduler_subpath: null
  num_train_timesteps: 1000
  shift: 5.0
  ...
image_encoder_kwargs:      # only for i2v / models with a CLIP image encoder
  image_encoder_subpath: models_clip_...pth
```
Every `*_subpath` is joined onto `model_name` in scripts. Load with `OmegaConf.load` and pass `OmegaConf.to_container(config['<section>'])` into `from_pretrained`. Use `filter_kwargs(Cls, OmegaConf.to_container(config['scheduler_kwargs']))` to build schedulers.

## 4. Inference scripts — `examples/<family>/predict_<task>.py`

Anatomy, top to bottom (see `examples/wan2.1_fun/predict_t2v.py`):
1. **`sys.path` bootstrap** (before importing `videox_fun`):
   ```python
   current_file_path = os.path.abspath(__file__)
   project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
   for project_root in project_roots:
       sys.path.insert(0, project_root) if project_root not in sys.path else None
   ```
2. **User config block** as top-level variables with explanatory comments, in the conventional order: `GPU_memory_mode`, `ulysses_degree`/`ring_degree`, `fsdp_dit`/`fsdp_text_encoder`, `compile_dit`, TeaCache (`enable_teacache`, `teacache_threshold`, `num_skip_start_steps`, `teacache_offload`), `cfg_skip_ratio`, Riflex (`enable_riflex`, `riflex_k`), `config_path`, `model_name`, `sampler_name`, `shift`, `transformer_path`/`vae_path`/`lora_path`, `sample_size`, `video_length`, `fps`, `weight_dtype`, `prompt`/`negative_prompt`, `guidance_scale`, `seed`, `num_inference_steps`, `lora_weight`, `save_path`.
3. **Device + config**: `device = set_multi_gpus_devices(ulysses_degree, ring_degree)`; then either `config = OmegaConf.load(config_path)` (civitai/custom layout) **or** guard `if config_path is not None:` and load components directly from a diffusers-layout checkpoint (see §3).
4. **Component loading**: transformer (`from_pretrained(..., transformer_additional_kwargs=...)`), optional `transformer_path`/`vae_path` override with `load_state_dict(strict=False)` + missing/unexpected key print, vae, tokenizer, text_encoder, and clip image encoder gated by `transformer.config.in_channels != vae.config.latent_channels`.
5. **Scheduler selection dict**: `{"Flow": FlowMatchEulerDiscreteScheduler, "Flow_Unipc": FlowUniPCMultistepScheduler, "Flow_DPM++": FlowDPMSolverMultistepScheduler}[sampler_name]`; build with `filter_kwargs`.
6. **Pipeline construction**: choose base vs inpaint/i2v/control pipeline by the model's channel condition.
7. **Multi-GPU / FSDP / compile**: if `ulysses_degree>1 or ring_degree>1` call `transformer.enable_multi_gpus_inference()` and optionally `shard_model`; if `compile_dit`, `torch.compile` each `transformer.blocks[i]`.
8. **`GPU_memory_mode` branching** — keep this exact order:
   ```python
   if GPU_memory_mode == "sequential_cpu_offload":
       replace_parameters_by_name(transformer, ["modulation",], device=device)
       transformer.freqs = transformer.freqs.to(device=device)
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
   ```
9. **TeaCache / cfg_skip / Riflex** enablement, `generator = torch.Generator(device).manual_seed(seed)`, LoRA `merge_lora`.
10. **Inference** under `torch.no_grad()`; align `video_length` to `vae.config.temporal_compression_ratio`; pass `video`/`mask_video` for i2v via `get_image_to_video_latent`.
11. **`save_results()`**: `save_videos_grid(sample, path, fps=fps)` for video, PIL save for a single frame; only rank 0 saves when multi-GPU. LoRA `unmerge_lora` after.

Other entry points to mirror when needed: `app.py` (Gradio), `launch_api.py` (API server backed by `videox_fun/api`), `post_infer*.py` (batch/queue inference).

## 5. Training scripts — `scripts/<family>/train*.py`

Mirror `scripts/wan2.1_fun/train.py`. Structure:
1. Diffusers-derived license header + `"""Modified from ..."""` note.
2. Third-party imports, then the **same `sys.path` bootstrap**, then `from videox_fun.data/models/pipeline/utils import ...`.
3. Helper funcs: `filter_kwargs`, `resize_mask`, `linear_decay`, `generate_timestep_with_lognorm`.
4. **`log_validation(vae, text_encoder, tokenizer, clip_image_encoder, transformer3d, args, config, accelerator, weight_dtype, global_step)`** — builds the **inference Pipeline** from the live (unwrapped) transformer and runs it to produce sample videos under `output_dir/sample/`. Wrapped in try/except; handles DeepSpeed (`transformer3d.config` swap) and restores VAE/text-encoder placement (`low_vram`). **Reuse the pipeline; never write a separate sampler.**
5. **`parse_args()`** — reuse the shared argument surface: `--config_path`, `--pretrained_model_name_or_path`, `--train_data_dir`, `--train_data_meta`, `--image_sample_size`/`--video_sample_size`/`--token_sample_size`, `--video_sample_n_frames`, `--video_sample_stride`, `--train_batch_size`, `--gradient_accumulation_steps`, `--learning_rate`, `--lr_scheduler`, `--lr_warmup_steps`, `--checkpointing_steps`, `--output_dir`, `--mixed_precision`, `--gradient_checkpointing`, `--enable_bucket`, `--random_hw_adapt`, `--training_with_video_token_length`, `--uniform_sampling`, `--low_vram`, `--train_mode`, `--trainable_modules`, LoRA args (`--use_lora`, `--rank`, ...), `--validation_prompts`/`--validation_paths`. Add new args only when the family genuinely needs them.
6. **`main()`** — Accelerator setup, DeepSpeed/FSDP zero-stage handling (auto-sets `save_state`), model loading via config, dataset + bucket sampler from `videox_fun.data`, trainable-module filtering / LoRA network via `create_network`, optimizer + `get_scheduler`, `accelerator.prepare`, checkpoint save/load hooks, training loop with timestep sampling, loss, `log_validation` at intervals, and final weight/LoRA save.

### Resolution args — `--video_sample_size` (+ `--fix_sample_size`)
Canvas resolution is always driven by a **single square** `--video_sample_size` (`type=int`, height = width) — never by separate `--video_sample_height` / `--video_sample_width`. When a **fixed non-square shape** is required, add `--fix_sample_size` (`nargs=2, type=int, default=None`, `[height, width]`) that overrides the square size; mirror `scripts/wan2.2_fun/train_lora.py`, `scripts/z_image/train_distill.py`. Derive the effective `height` / `width` once in `parse_args()` and reuse them everywhere downstream:
```python
parser.add_argument("--video_sample_size", type=int, default=1280)
parser.add_argument("--fix_sample_size", nargs=2, type=int, default=None,
                    help="Fix Sample size [height, width] to override `--video_sample_size` with a fixed non-square shape.")
...
if args.fix_sample_size is not None:
    args.video_sample_height, args.video_sample_width = args.fix_sample_size
else:
    args.video_sample_height = args.video_sample_width = args.video_sample_size
```
In bucket datasets `--fix_sample_size` also forces `random_hw_adapt=False` / `training_with_video_token_length=False` and bumps `video_sample_size = max(max(fix_sample_size), video_sample_size)`; in data-free scripts (e.g. `scripts/minimax_h3/train_pdd_lora.py`) it simply pins the generation canvas. Always validate the size against the patch/VAE constraint (minimax_h3: `% 32`). The `.sh` launcher passes it space-separated (`nargs=2`): `--fix_sample_size 768 1344`.

### Launcher — `scripts/<family>/train*.sh`
`export MODEL_NAME/DATASET_NAME/DATASET_META_NAME`, then `accelerate launch --mixed_precision="bf16" scripts/<family>/train.py --config_path=... <full arg list>`. Include commented I2V/control variants and DeepSpeed/NCCL notes as the existing scripts do.

### Docs — `README_TRAIN.md` + `README_TRAIN_zh-CN.md`
Aligned bilingual pair: identical section order, identical commands and parameter tables; only the prose language differs. Follow the top-level section order used across existing training READMEs.

## 6. Shared infrastructure map (reuse, never reimplement)

| Need | Import from |
|------|-------------|
| Flow/DPM/UniPC schedulers | `diffusers`, `videox_fun.utils.fm_solvers`, `videox_fun.utils.fm_solvers_unipc` |
| LoRA create/merge/unmerge/convert | `videox_fun.utils.lora_utils` |
| FP8 quantization | `videox_fun.utils.fp8_optimization` |
| Group / leaf offload hooks | `videox_fun.utils.group_offload` |
| Multi-GPU device + FSDP shard + seq-parallel attn | `videox_fun.dist` |
| Save video/audio, image→video latents, kwarg filter, dimension calc | `videox_fun.utils.utils` |
| Datasets + bucket/aspect-ratio samplers + masks | `videox_fun.data` |
| TeaCache coefficients | `videox_fun.models.cache_utils` |

## 7. Naming quick reference

| Concept | Convention | Example |
|---------|-----------|---------|
| Model file | `<family>_transformer3d.py` | `wan_transformer3d.py` |
| Model class | `<Family>Transformer3DModel` | `WanTransformer3DModel` |
| VAE class | `AutoencoderKL<Family>` | `AutoencoderKLWan` |
| Pipeline file | `pipeline_<family>.py` | `pipeline_wan.py` |
| Pipeline class | `<Family>Pipeline` | `WanPipeline` / `WanFunInpaintPipeline` |
| Config | `config/<family>/<variant>.yaml` | `config/wan2.1/wan_civitai.yaml` |
| Inference | `examples/<family>/predict_<task>.py` | `predict_t2v.py`, `predict_i2v.py`, `predict_v2v_control.py` |
| Training | `scripts/<family>/train[_<variant>].py` | `train.py`, `train_lora.py`, `train_control.py`, `train_distill.py` |

## 8. Training data pipeline — dataset & sampler selection

Pick the dataset by **task / `train_mode`**, then the sampler by **`enable_bucket`** and dataset type. All datasets/samplers come from `videox_fun.data` — never write a new one.

### Annotation format — the `train_data_meta` file (`metadata.json` / `.csv`)
Every dataset class reads an annotation file (`args.train_data_meta`) that indexes the media under `args.train_data_dir` (`data_root`). `ImageVideoDataset` accepts **`.json`** (a top-level array of records) or **`.csv`** (`csv.DictReader`; the header row is the field names). Each record for ordinary image/video training:

| Field | Required | Meaning |
|-------|----------|---------|
| `file_path` | yes | Media path, resolved **relative to `train_data_dir`** via `os.path.join(data_root, file_path)`. If `data_root is None`, `file_path` is used as-is. |
| `text` | yes | Caption / prompt. Dropped to `""` with probability `text_drop_ratio` (default `0.1`) for classifier-free guidance. |
| `type` | no | `"video"` or `"image"`; **defaults to `"image"`** when the key is absent (`data_info.get('type', 'image')`). |

```json
[
  {"file_path": "train/00000000.mp4", "text": "A young woman gently turns her head to the right ...", "type": "video"},
  {"file_path": "train/00000001.jpg", "text": "a dog running on the beach", "type": "image"}
]
```
The directory layout matches the index — media in a `train/` subdir, the annotation file beside it. Ready-made examples ship in `datasets/X-Fun-Videos-Demo/` (`train/*.mp4` + `metadata.json`) and `datasets/X-Fun-Images-Demo/`. The equivalent `.csv`:
```csv
file_path,text,type
train/00000000.mp4,"A young woman gently turns her head to the right ...",video
train/00000001.jpg,"a dog running on the beach",image
```

**Variant datasets append extra fields to this same record shape**, each consumed by its own class (see the table below) — e.g. camera-pose adds `action_path` (`LingbotImageVideoDataset`), object/VACE/S2V variants add object fields (`object_file_path` / `objects`). The demo folders also ship several augmented metadata variants (next subsection). Always read the target class's `get_batch` for the exact fields it consumes.

### Ready-made demo datasets — the standard test data (never invent a test set)
Smoke tests, `log_validation` checks, and doc examples all run on the official demo datasets under `datasets/`, downloaded from ModelScope as `PAI/<name>`:
```bash
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```
Pick the demo by **task**, matching the dataset class in the table below:

| Demo dataset (`datasets/...`) | Contents | Extra metadata fields | Task it tests | Dataset class |
|-------------------------------|----------|----------------------|---------------|---------------|
| `X-Fun-Videos-Demo` | 16 videos (832×480) in `train/` | — | T2V / I2V base + inpaint, distill | `ImageVideoDataset` |
| `X-Fun-Videos-Controls-Demo` | 16 videos in `train/` + `canny/` + `object/<video_id>/` + `wav/` | `control_file_path`, `object_file_path` (list), `audio_path` | V2V control, VACE, S2V-with-control | `ImageVideoControlDataset`, `VideoSpeechControlDataset` |
| `X-Fun-Videos-Audios-Demo` | 17 video/audio pairs: `train/` (1280×720) + `wav/` (16 kHz mono) + `pose/` | `audio_path`, `control_file_path` | Speech-driven S2V / avatar / talking-head | `VideoSpeechDataset` |
| `X-Fun-Images-Demo` | 19 images in `train/` | — | T2I full fine-tune + LoRA (z_image / flux2 / qwenimage / lens / ernie) | `ImageVideoDataset` |
| `X-Fun-Images-Controls-Demo` | 19 images in `train/` + `canny/` | `control_file_path` | Image control / ControlNet / i2i inpaint | `ImageVideoControlDataset` |
| `X-Fun-Images-Edit-Demo` | 21 records: `source/souce-<id>/` (multi-source supported) → `train/` | `source_file_path` (**list**) | Image edit (Qwen-Image-Edit family) | `ImageEditDataset` |
| `X-Fun-Videos-Lingbot-Demo` | video + `intrinsics.npy` / `poses.npy` | camera pose / action | Camera-pose world model (`lingbot_world`) | `LingbotImageVideoDataset` |

**Which metadata file to point `--train_data_meta` at** (each demo ships several variants beside the media):

| Metadata file | Use when |
|---------------|----------|
| `metadata.json` | Base format only (`file_path` / `text` / `type`) — fine for a minimal check |
| `metadata_add_width_height.json` | **Default choice.** Adds `width` / `height` so bucketing doesn't decode media (matters on slow storage such as OSS). Used by non-VACE control / S2V training too |
| `metadata_add_width_height_add_objects.json` | VACE / subject-reference training (`object_file_path` list → `object/<video_id>/`; shuffled at train time) |
| `metadata_add_width_height_add_wav.json` | Audio-visual joint models (e.g. `minimax_h3_fun` control training): `audio_path` → `wav/`. Keep the `.sh` launcher and the README on the same file |
| `metadata_lingbot_video_add_width_height.json` | `lingbot_video` — `text` is already a structured JSON caption (lives in `X-Fun-Videos-Demo`) |
| `metadata_origin.json` | Pre-processing original kept for reference; not used for training |

Regenerate the width/height variant with the shipped helper when adding your own media:
`python scripts/process_json_add_width_and_height.py --input_file datasets/<Demo>/metadata.json --output_file datasets/<Demo>/metadata_add_width_height.json`.

`audio_path` optionality differs per class (`videox_fun/data/dataset_video.py`): `VideoSpeechDataset` reads `video_dict['audio_path']` directly, so it is **required**; `VideoSpeechControlDataset` uses `.get('audio_path')` and **falls back to the video file's own audio track** when the field is absent.

### Dataset by task (all take `train_data_meta, train_data_dir, ...`)
| Task / mode | Dataset class | Used by | Key kwargs |
|-------------|--------------|---------|-----------|
| T2V / I2V base (`normal` + inpaint) | `ImageVideoDataset` | `train.py`, `train_lora.py`, t2i `train.py` | `enable_inpaint = train_mode != "normal"`, `video_sample_size/stride/n_frames`, `image_sample_size`, `video_repeat` |
| Image T2I (qwenimage/flux/z_image) | `ImageVideoDataset` | `scripts/<img>/train.py` | `image_sample_size` |
| Control (canny/pose/depth/camera) | `ImageVideoControlDataset` | `train_control*.py`, `train_control_distill.py` | `enable_camera_info = train_mode == "control_camera_ref"` |
| Image Edit (source→target) | `ImageEditDataset` | `qwenimage/train_edit*.py` | `image_sample_size` |
| Speech/audio-driven (S2V, avatar, talking) | `VideoSpeechDataset` | `mova`, `ltx2`, `minimax_h3`, `fantasytalking`, `infinitetalk`, `flashhead`, `longcatvideo/train_avatar*` | audio + video fields |
| S2V **with control** | `VideoSpeechControlDataset` | `wan2.2/train_s2v*.py`, `minimax_h3_fun/train_control*` | audio + control |
| Motion/pose animate | `VideoAnimateDataset` | `wan2.2/train_animate*.py` | motion/pose driven |
| Distill text-only branch, GRPO, DPO | `TextDataset` | `train_distill*.py` (text branch), `z_image/train_grpo_lora.py`, `train_dpo_lora.py` | reads only the `text` field; `text_drop_ratio` |
| Precomputed latents (ODE pairs) | `ImageVideoSafetensorsDataset` | `wan2.1_self_forcing/train_ode.py` | `data_root` |
| Camera-pose conditioning | `LingbotImageVideoDataset` | `lingbot_world/train.py` | `intrinsics.npy` / `poses.npy` |
| Video-only (VAE/TAEHV distill) | `VideoDataset` | `taehv/train_taehv.py` | `sample_size/stride/n_frames`, `enable_inpaint=False` |

### Sampler by condition
| Condition | Sampler | Shape |
|-----------|---------|-------|
| `enable_bucket=True` (default; image+video) | `AspectRatioBatchImageVideoSampler` | `sampler=RandomSampler(ds, generator=g), dataset=train_dataset.dataset, batch_size, train_folder=args.train_data_dir, drop_last=True, aspect_ratios=aspect_ratio_sample_size` |
| `enable_bucket=False` | `ImageVideoSampler` | `ImageVideoSampler(RandomSampler(ds, generator=g), train_dataset, batch_size)` |
| `TextDataset` (distill text branch / GRPO / DPO) | `BatchSampler` (plain) | `BatchSampler(RandomSampler(ds, generator=g), batch_size, drop_last=True)`; GRPO adds `k_repeat=args.num_image_per_prompt` |
| video-only bucket (available, not used by current scripts) | `AspectRatioBatchSampler` | — |
| image-only bucket (available, not used by current scripts) | `AspectRatioBatchImageSampler` | — |

`aspect_ratio_sample_size` is built from `ASPECT_RATIO_512` scaled by `args.video_sample_size`; `get_closest_ratio` picks the bucket inside `collate_fn`.

### Universal DataLoader creation pattern
```python
batch_sampler_generator = torch.Generator().manual_seed(args.seed)
if args.enable_bucket:
    aspect_ratio_sample_size = {k: [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[k]] for k in ASPECT_RATIO_512}
    batch_sampler = AspectRatioBatchImageVideoSampler(
        sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset,
        batch_size=args.train_batch_size, train_folder=args.train_data_dir, drop_last=True,
        aspect_ratios=aspect_ratio_sample_size)
    def collate_fn(examples):
        new_examples = {"pixel_values": [], "text": []}
        if args.train_mode != "normal":            # inpaint/i2v adds mask fields
            new_examples.update({"mask_pixel_values": [], "mask": [], "clip_pixel_values": []})
        # bucket via get_closest_ratio -> transform (Resize/CenterCrop/Normalize) -> stack
        # masked branch uses get_random_mask(...)
        return new_examples
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn,
        persistent_workers=args.dataloader_num_workers != 0, num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn(args.seed + accelerator.process_index))
else:
    batch_sampler = ImageVideoSampler(RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, args.train_batch_size)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=batch_sampler,
        persistent_workers=args.dataloader_num_workers != 0, num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn(args.seed + accelerator.process_index))
```
`collate_fn` receives the `examples` **list** (not a `batch` dict); build every batch-level field (`text`, `pixel_values`, masks) explicitly from `examples` into `new_examples`. When `--enable_text_encoder_in_dataloader`, encode prompts inside `collate_fn` and emit `encoder_hidden_states` / `encoder_attention_mask`.

## 9. Inference task matrix — predict script → pipeline → inputs

Pick the pipeline by **task**; the `predict_<task>.py` name and its inputs follow the same convention across families.

| Task | `predict_<task>.py` | Pipeline (family example) | Extra `__call__` inputs | Input helper |
|------|--------------------|---------------------------|-------------------------|--------------|
| Text→Video | `predict_t2v.py` | `WanPipeline`, `Wan2_2Pipeline`, `CogVideoXFunPipeline`, `LongCatVideoPipeline`, `LTX2Pipeline` | `prompt` only | — |
| Image→Video | `predict_i2v.py` | `WanI2VPipeline`(=`WanFunInpaintPipeline`), `Wan2_2FunInpaintPipeline`, `Wan2_2I2VPipeline`, `HunyuanVideoI2VPipeline` | `video`, `mask_video` | `get_image_to_video_latent(start_image, end_image, video_length, sample_size)` |
| Text+Image→Video (5B) | `predict_ti2v.py` | `Wan2_2TI2VPipeline` | `prompt` (+ optional image) | `get_image_to_video_latent` |
| Video→Video Control | `predict_v2v_control.py` | `WanFunControlPipeline`, `Wan2_2FunControlPipeline` | `control_video` | `get_video_to_video_latent(control_video, ...)` |
| Control + reference | `predict_v2v_control_ref.py` | `WanFunControlPipeline` | `control_video` + `ref_image` | `get_video_to_video_latent` + `get_image_latent` |
| Control + camera | `predict_v2v_control_camera.py` | `WanFunControlPipeline` | `control_video` + camera pose | — |
| VACE (control/mask/i2v/s2v) | `predict_v2v_control.py`, `predict_v2v_mask.py`, `predict_s2v.py`, `predict_i2v.py` | `WanVacePipeline`, `Wan2_2VaceFunPipeline` | control/mask/ref | — |
| Speech→Video (audio) | `predict_s2v.py` | `Wan2_2S2VPipeline`, `MiniMaxH3Pipeline`, `InfiniteTalkPipeline`, `FantasyTalkingPipeline`, `FlashHeadPipeline`, `MOVAPipeline`, `LongCatVideoAvatarPipeline` | `audio` + reference image | — |
| Animate (motion/pose) | `predict_animate.py` | `Wan2_2AnimatePipeline` | motion/pose video + ref | — |
| Subject reference | `predict_s2v.py` (phantom) | `WanFunPhantomPipeline` | reference images | — |
| Text→Image | `predict_t2i.py` | `QwenImagePipeline`, `Flux2Pipeline`, `ZImagePipeline`, `LensPipeline`, `ErnieImagePipeline` | `prompt` | — |
| Image Control (t2i) | `predict_t2i_control.py` | `QwenImageControlPipeline`, `ZImageControlPipeline`, `Flux2ControlPipeline`, `QwenImageControlNetPipeline` | `control_image` | — |
| Inpaint (i2i) | `predict_i2i_inpaint.py` | `QwenImageControlPipeline`, `ZImageControlPipeline`, `Flux2ControlPipeline` | `image` + `mask` | — |
| Image Edit | `predict_t2i_edit.py`, `predict_t2i_edit_plus.py` | `QwenImageEditPipeline`, `QwenImageEditPlusPipeline` | source image + instruction | — |
| Layered edit | `predict_i2i_layered.py` | `QwenImageLayeredPipeline` | image | — |
| Camera-pose world | `predict_i2v.py` (lingbot_world) | `Wan2_2I2VPipeline`, `WanFunLingbotWorldFastPipeline` | image + camera pose | — |
| Latent upsample | `predict_i2v_upsample.py` | `LTX2LatentUpsamplePipeline`, `WanLatentUpsamplePipeline` | low-res latent/video | — |
| AR / streaming distill | `predict_t2v_stream.py` | `WanSelfForcingPipeline` | prompt (streamed) | — |

### Predict-script variant suffixes (same task, different backend/model)
| Suffix | Meaning |
|--------|---------|
| `_tae` | Fast decode via `AutoencoderTinyWan` (TAEHV) instead of the full VAE |
| `_2.2vae` | Uses the Wan2.2 VAE (`AutoencoderKLWan3_8`) |
| `_5b` | 5B-parameter model variant |
| `turbo` / distill | Distilled model, few-step inference (e.g. `predict_turbo_*.py`) |
| `_refine` | Two-stage refine pass |
| `_ref` / `_camera` | Adds reference-image / camera conditioning |

All variants keep the identical config block, `GPU_memory_mode` branching, and `save_results()` from Section 4 — only the loaded VAE/transformer and pipeline class change.

## 10. Preprocessing — offline training-data generation (multi-GPU + safetensors)

Here "preprocessing" means **generating/caching training data offline** with the teacher / VAE / text-encoder — latents, ODE-trajectory pairs, prompt/text embeddings — so training just reads cached tensors instead of re-encoding every step. Canonical example: `scripts/wan2.1_self_forcing/generate_ode_pairs.py` (+ `generate_ode_pairs.sh`); the loader-side contract is `ImageVideoSafetensorsDataset` in `videox_fun/data/dataset_image_video.py`. Two rules are non-negotiable.

### Rule 1 — multi-GPU is mandatory
Never a single-GPU / hardcoded `cuda:0` loop. Launch with `accelerate launch` and shard work across ranks by interleaving:
```python
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision=args.mixed_precision)
device, world_size, rank = accelerator.device, accelerator.num_processes, accelerator.process_index
torch.set_grad_enabled(False)                        # inference-only

total_per_rank = math.ceil(len(prompts) / world_size)
for index in tqdm(range(total_per_rank), disable=rank != 0):
    prompt_index = index * world_size + rank         # interleaved shard
    if prompt_index >= len(prompts):
        continue
    out_path = os.path.join(args.output_folder, f"{prompt_index:05d}.safetensors")
    if os.path.exists(out_path):                     # resume-friendly
        continue
    ...                                              # encode prompt / run teacher ODE / collect latents
accelerator.wait_for_everyone()
if accelerator.is_main_process:                      # write the JSON index once, on rank 0
    json.dump([{"file_path": p} for p in all_safetensor_paths],
              open(os.path.join(args.output_folder, "outputs.json"), "w"), ensure_ascii=False, indent=4)
```
Launcher (`.sh`): `accelerate launch --mixed_precision="bf16" scripts/<family>/generate_<...>.py --pretrained_model_name_or_path=... --config_path=config/<family>/*.yaml --output_folder=datasets/<...> ...`. Reuse `videox_fun.models` + config-driven `from_pretrained` (Section 3) and `videox_fun.utils.utils.save_videos_grid` for sample previews — do not write a new loader.

### Rule 2 — store as safetensors; do NOT use LMDB or `.pt`
Save every cached tensor with `safetensors.torch.save_file`, one `.safetensors` per sample (or per tensor), plus a JSON index of `{"file_path": ...}` entries:
```python
from safetensors.torch import save_file
save_file(
    {"latents": latents.cpu(), "prompt_embeds": text_embeds.cpu(), "prompt_attention_mask": mask.cpu()},
    out_path,                        # f"{prompt_index:05d}.safetensors"
    metadata={"prompt": prompt},
)
```
`ImageVideoSafetensorsDataset(ann_path, data_root=None)` reads that JSON and supports two layouts:
- **Single-file (default)**: `{"file_path": "scene.safetensors"}` — whole state dict in one archive.
- **Per-tensor (`--save_per_tensor`)**: `{"file_path": "scene_dir", "latents": ".../latents.safetensors", "prompt_embeds": ".../prompt_embeds.safetensors"}` — each key loaded and merged.

**Do not** cache preprocessed data in **LMDB** or as **`.pt`/`.pth` `torch.save` pickles**. safetensors is the repo-wide standard (also used for LoRA/weight saving), is pickle-free/safe, memory-maps fast, and is exactly what `ImageVideoSafetensorsDataset` loads. (Scope: this governs cached *data tensors*; accelerate optimizer/scheduler/scaler `.pt` states written during training checkpoints are a separate mechanism and unaffected.)

### Related but different — dataset curation
Scoring / filtering / captioning under `videox_fun/video_caption/` (`compute_*.py`, `internvl2_video_recaptioning.py`) is dataset *curation*, not latent caching. It is also multi-GPU (accelerate `PartialState.split_between_processes`/`gather_object`, or vLLM `tensor_parallel_size=device_count()`), but writes csv/jsonl **metadata** (not tensors), so Rule 2 does not apply there.
