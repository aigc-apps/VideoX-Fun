---
name: integrating-models
description: Guides adding, porting, or onboarding a diffusion model (transformer/VAE/encoder, inference pipeline, training script, config) into the VideoX-Fun repository by mirroring the closest existing model family and maximizing reuse of the repository's existing code and shared infrastructure. Use when integrating a new model/architecture, or when creating predict_*.py inference scripts, scripts/*/train*.py training scripts, pipeline_*.py, config/*.yaml, or model definitions under videox_fun/models/.
---

# Integrating Models into VideoX-Fun

## Core rule: maximize reuse of existing repo code — mirror, extend, never reinvent

**Prime directive: reuse this repository's existing code to the maximum.** Nearly every building block you need already exists in `videox_fun/` or in a sibling model family. Your job is to **find it, import it, and extend it** — not to write a parallel implementation. A new file should be mostly reused structure plus the genuinely model-specific delta; the less new code you write, the better.

**Reuse-first protocol — before writing ANY new function / class / util:**
1. **Search the repo first.** Grep `videox_fun/` and the closest family for an existing equivalent (weight loader, scheduler, sampler, offload, attention, LoRA, fp8, dataset, dist helper, save/metric util). If one exists → **import and reuse it**. If it is 80% right → **extend / parameterize it**, do not fork it.
2. **Only if nothing exists** may you add new code — and then put it in the shared layer (`videox_fun/utils`, `videox_fun/data`, `videox_fun/dist`) so the next model reuses it too, instead of burying it in a family folder.
3. **Never copy-paste** a util into a new file (that creates drift); import the single source of truth.

**Mirror the closest family.** Every model follows the **same layered template**. Integrating a model means finding the closest existing family and mirroring its structure, changing only what genuinely differs:
1. Pick the closest existing family by task type (t2v / i2v / v2v-control / s2v / t2i / edit / distill): `wan2.1`, `wan2.1_fun`, `wan2.2`, `qwenimage`, `flux2`, `minimax_h3`, `ltx2`, `longcatvideo`, `cogvideox_fun`, `z_image`, etc.
2. Read that family end-to-end across all layers:
   - `examples/<family>/predict_*.py` (inference entry)
   - `scripts/<family>/train*.py` + `*.sh` + `README_TRAIN*.md` (training)
   - `videox_fun/pipeline/pipeline_<family>*.py` (pipeline)
   - `videox_fun/models/<family>_*.py` (model definitions)
   - `config/<family>/*.yaml` (config)
3. Copy that structure and adapt. Keep names, argument sets, control flow, and reuse points identical in shape.

Writing a bespoke pipeline, weight loader, trainer, sampler, dataset, or offload scheme from scratch is a **failure mode**. If you are tempted to, **stop** and check the Reuse inventory below first.

## Repository layout (where each layer lives)

| Layer | Location | What it is |
|-------|----------|------------|
| Model definitions | `videox_fun/models/<family>_*.py` | Transformer / VAE / text-audio-image encoders. Diffusers `ModelMixin`+`ConfigMixin`, `@register_to_config`, custom `from_pretrained`. |
| Model registry | `videox_fun/models/__init__.py` | Imports every model class. **Must be updated** for a new model. |
| Inference pipelines | `videox_fun/pipeline/pipeline_<family>*.py` | `<Family>Pipeline(DiffusionPipeline)` with `__call__`. |
| Pipeline registry | `videox_fun/pipeline/__init__.py` | Imports every pipeline + aliases. **Must be updated.** |
| Configs (optional) | `config/<family>/*.yaml` | OmegaConf YAML for civitai/custom layouts; a standard diffusers-layout checkpoint can load without one. |
| Inference entry scripts | `examples/<family>/predict_*.py` | User-facing, config-block-at-top runnable scripts. |
| Inference services | `examples/<family>/{app.py,launch_api.py,post_infer*.py}` | Gradio UI / API server / batch inference. |
| Training scripts | `scripts/<family>/train*.py` | `train.py`, `train_lora.py`, `train_control.py`, `train_distill.py`, ... |
| Training launchers | `scripts/<family>/train*.sh` | `accelerate launch` / DeepSpeed command with full arg list. |
| Training docs | `scripts/<family>/README_TRAIN*.md` | Bilingual pairs: `README_TRAIN.md` + `README_TRAIN_zh-CN.md`. |
| Shared: schedulers/utils | `videox_fun/utils/` | `fm_solvers`, `fm_solvers_unipc`, `lora_utils`, `fp8_optimization`, `group_offload`, `utils.py`. |
| Shared: distributed | `videox_fun/dist/` | `fsdp.shard_model`, `fuser.set_multi_gpus_devices`, `<family>_xfuser` sequence-parallel attention. |
| Shared: data | `videox_fun/data/` | Datasets (`ImageVideoDataset`, `VideoDataset`, ...) + bucket/aspect-ratio samplers. |
| Demo / test datasets | `datasets/X-Fun-*-Demo/` | Ready-made smoke-test data, downloaded via `modelscope download --dataset PAI/<name>`; each ships several `metadata*.json` variants. **The only test data to use** (see reference.md §8). |
| Preprocessing (data gen) | `scripts/<family>/generate_*.py` / `train_preprocess.py` (+ `.sh`) | Offline multi-GPU generation of cached training data (latents / ODE pairs / embeddings) → per-sample `.safetensors` + `outputs.json`, loaded by `ImageVideoSafetensorsDataset`. |
| ComfyUI nodes | `comfyui/<family>/nodes.py` | Optional node integration mirroring the pipeline. |

## Integration workflow

Copy this checklist and track progress:

```
Integration Progress:
- [ ] Step 0: Choose the closest family to mirror; read it across all layers
- [ ] Step 1: Model definitions in videox_fun/models/ + register in models/__init__.py
- [ ] Step 2: Pipeline in videox_fun/pipeline/ + register in pipeline/__init__.py
- [ ] Step 3: Config YAML in config/<family>/
- [ ] Step 4: Inference script(s) in examples/<family>/predict_*.py
- [ ] Step 5: Training script(s) in scripts/<family>/train*.py + .sh
- [ ] Step 6: Training docs README_TRAIN.md + README_TRAIN_zh-CN.md
- [ ] Step 7: Reuse audit + verification (incl. smoke test on the matching demo dataset)
```

**Step 0 — Choose the mirror.** Match by task and architecture. A new control model mirrors an existing `*_fun`/`*_control` family; a new audio/talking model mirrors `minimax_h3`/`longcatvideo`/`infinitetalk`; a new image model mirrors `qwenimage`/`flux2`/`z_image`.

**Step 1 — Model.** Create `videox_fun/models/<family>_transformer3d.py` (or `2d`), `<family>_vae.py`, encoders as needed. Mirror the class shape: `class <Family>Transformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin)`, `_supports_gradient_checkpointing = True`, `@register_to_config __init__`, and a `from_pretrained` that supports `transformer_additional_kwargs`, `dict_mapping`, `low_cpu_mem_usage`, and missing-key init. Add imports to `videox_fun/models/__init__.py`.

**Step 2 — Pipeline.** Create `videox_fun/pipeline/pipeline_<family>.py`. Mirror `pipeline_wan.py`: module-level `retrieve_timesteps`, a `<Family>PipelineOutput(BaseOutput)` dataclass, `<Family>Pipeline(DiffusionPipeline)` with `model_cpu_offload_seq`, `_callback_tensor_inputs`, `__init__(vae, tokenizer, text_encoder, transformer, scheduler, ...)`, `encode_prompt`, and `__call__`. Add imports/aliases to `videox_fun/pipeline/__init__.py`.

**Step 3 — Config (optional).** A YAML under `config/<family>/` is **not always required**. It is needed mainly for **civitai-format / custom single-file layouts** — to supply `transformer_additional_kwargs`, `dict_mapping` (civitai key → `__init__` kwarg), component subpaths, and `vae_kwargs`/`text_encoder_kwargs`/`scheduler_kwargs`/`image_encoder_kwargs`. For a **standard diffusers-layout** checkpoint (`model_index.json` + per-subfolder `config.json`), load directly via `from_pretrained(model_name, subfolder=...)` with no YAML — mirror `examples/minimax_h3_fun/predict_v2v_control.py`, which guards `if config_path is not None:`. When you do add a YAML, load it via `OmegaConf.load(config_path)` and spread into `from_pretrained` instead of hardcoding those values.

**Step 4 — Inference script.** Create `examples/<family>/predict_<task>.py` following the exact template (config block at top → component loading → scheduler dict → pipeline construction → multi-GPU/FSDP/compile → `GPU_memory_mode` branching → TeaCache → LoRA merge → inference → `save_results`). See [examples.md](examples.md).

**Step 5 — Training script.** Create `scripts/<family>/train.py` (+ `train_lora.py` etc.). Mirror the shared structure: license header, `sys.path` bootstrap, imports from `videox_fun`, `log_validation()` that **reuses the inference Pipeline**, `parse_args()` (reuse the existing shared argument set), `main()`. Add a `train.sh` launcher. Reuse `videox_fun.data` datasets/samplers — do not write a new dataset.

**Step 6 — Docs.** Write `README_TRAIN.md` and `README_TRAIN_zh-CN.md` as an aligned bilingual pair (same structure, same commands/params, matching section order).

**Step 7 — Reuse audit + verification.** Confirm you reused shared infra (below), smoke-test the new train/predict path on the **matching official demo dataset** under `datasets/X-Fun-*-Demo/` (pick by task and metadata variant — see reference.md §8), then run the verification checklist. Never invent an ad-hoc test set and never leave `datasets/internal_datasets/` placeholders in shipped scripts/docs.

## Reuse inventory (use these, do not reimplement)

**Reuse-first catalog: import from here instead of reimplementing. If a helper you need is not listed, grep `videox_fun/` and the closest family before writing your own.**

- **Schedulers**: `FlowMatchEulerDiscreteScheduler`, `videox_fun.utils.fm_solvers.FlowDPMSolverMultistepScheduler`, `fm_solvers_unipc.FlowUniPCMultistepScheduler`. Selected via a `sampler_name` dict.
- **LoRA**: `videox_fun.utils.lora_utils` — `merge_lora`, `unmerge_lora`, `create_network`, `convert_peft_lora_to_kohya_lora`.
- **FP8 / quantization**: `videox_fun.utils.fp8_optimization` — `convert_model_weight_to_float8`, `convert_weight_dtype_wrapper`, `replace_parameters_by_name`.
- **Offloading**: `videox_fun.utils.group_offload` — `register_auto_device_hook`, `safe_enable_group_offload`; plus pipeline `enable_sequential_cpu_offload` / `enable_model_cpu_offload` / `.to(device)`.
- **Distributed**: `videox_fun.dist` — `set_multi_gpus_devices`, `shard_model` (FSDP), `<family>_xfuser` sequence-parallel attention processors, `enable_multi_gpus_inference()`.
- **IO / helpers**: `videox_fun.utils.utils` — `save_videos_grid`, `save_videos_with_audio_grid`, `get_image_to_video_latent`, `get_video_to_video_latent`, `get_image_latent`, `filter_kwargs`, `calculate_dimensions`.
- **Data**: `videox_fun.data` — `ImageVideoDataset`, `VideoDataset`, `ImageVideoControlDataset`, `VideoSpeechDataset`, bucket/aspect-ratio samplers, `get_closest_ratio`, `get_random_mask`.
- **Caching / speedups**: TeaCache (`models/cache_utils`, `get_teacache_coefficients`, `transformer.enable_teacache`), `enable_cfg_skip`, Riflex (`enable_riflex`), `torch.compile` on `transformer.blocks`.
- **Preprocessing (data gen, multi-GPU)**: mirror `scripts/wan2.1_self_forcing/generate_ode_pairs.py` — `accelerate launch` + `Accelerator` (interleaved rank sharding), config-driven `from_pretrained` for the teacher/VAE/text-encoder, `safetensors.torch.save_file` per sample + `outputs.json` index, consumed by `videox_fun.data.ImageVideoSafetensorsDataset`. Store as **safetensors only — never LMDB or `.pt`** (see reference.md §10).

## Non-negotiable conventions

- **Maximize reuse of existing repo code**: import existing `videox_fun/` helpers and mirror the closest family; never fork or copy-paste a util, and never write a parallel pipeline / loader / scheduler / sampler / offload. Genuinely-new shared code goes in `videox_fun/{utils,data,dist}` (so the next model reuses it), not buried in a family folder.
- **`sys.path` bootstrap**: every runnable script starts with the 3-level `project_roots` loop inserting into `sys.path` before importing `videox_fun`.
- **Config-driven loading (YAML optional)**: a `config/<family>/*.yaml` is required for civitai-format/custom layouts (it supplies `transformer_additional_kwargs`/`dict_mapping`/subpaths); it is **optional for standard diffusers-layout checkpoints**, which load directly via `from_pretrained(model_name, subfolder=...)`. When a YAML is used, don't hardcode the values it provides.
- **`GPU_memory_mode`**: support the standard six modes — `model_full_load`, `model_full_load_and_qfloat8`, `model_cpu_offload`, `model_cpu_offload_and_qfloat8`, `model_group_offload`, `sequential_cpu_offload` — with the exact branching order used in existing `predict_*.py`.
- **Naming**: files `<family>_transformer3d.py` / `<family>_vae.py` / `pipeline_<family>.py`; classes `<Family>Transformer3DModel` / `AutoencoderKL<Family>` / `<Family>Pipeline`.
- **Resolution args**: drive canvas size with a single square `--video_sample_size` (`type=int`, height = width); never `--video_sample_height` / `--video_sample_width`. For a fixed non-square shape add `--fix_sample_size` (`nargs=2, type=int`, `[height, width]`) that overrides the square size, and derive the effective height/width once in `parse_args()` (see reference.md §5).
- **Registries**: a model is not integrated until it is imported in BOTH `videox_fun/models/__init__.py` and `videox_fun/pipeline/__init__.py`.
- **Two weight formats**: support `civitai` and `diffusers` via config `format` + `dict_mapping` (maps civitai keys such as `in_dim`→`in_channels`, `dim`→`hidden_size`).
- **Bilingual docs**: training READMEs ship as EN + `_zh-CN` pairs with aligned structure and identical commands/params.
- **Test data = official demo datasets**: smoke tests, `log_validation` checks, launcher `.sh` defaults, and doc examples all point at `datasets/X-Fun-*-Demo/` (ModelScope `PAI/<name>`), with the metadata variant matching the task — `metadata_add_width_height.json` by default, `_add_objects.json` for VACE/subject-reference, `_add_wav.json` for audio-visual joint models, `metadata_lingbot_video_add_width_height.json` for `lingbot_video`. Selection matrix: reference.md §8.
- **Preprocessing = offline data generation, multi-GPU + safetensors**: cached training data (latents / ODE pairs / embeddings) is produced by `accelerate launch` scripts like `generate_ode_pairs.py` (interleaved rank sharding, resume by skipping existing files, `wait_for_everyone`, rank-0 JSON index) and saved with `safetensors.torch.save_file` + an `outputs.json` index for `ImageVideoSafetensorsDataset`. **Never single-GPU / `cuda:0`; never LMDB or `.pt`/`torch.save` pickles for preprocessed data.** See reference.md §10.

## Verification checklist

- [ ] New model classes imported in `videox_fun/models/__init__.py`
- [ ] New pipeline(s) imported in `videox_fun/pipeline/__init__.py`
- [ ] Config YAML present **only if** the checkpoint is civitai-format/custom-layout; a diffusers-layout model may load directly via `from_pretrained(model_name, subfolder=...)` with no YAML. When a YAML is used, it drives component loading (no hardcoded kwargs)
- [ ] `predict_*.py` mirrors an existing script: `sys.path` bootstrap, config block, scheduler dict, `GPU_memory_mode` branching, LoRA merge, `save_results`
- [ ] `train*.py` reuses `videox_fun.data` + shared args, and `log_validation()` reuses the inference Pipeline
- [ ] `train*.sh` launcher provided (`accelerate launch` / DeepSpeed)
- [ ] Shared infra reused (schedulers / lora_utils / fp8 / group_offload / dist / utils / data) — nothing reimplemented
- [ ] Any offline data-generation/preprocessing script runs multi-GPU (`accelerate launch` + `Accelerator`) and saves cached tensors as **safetensors + `outputs.json`** for `ImageVideoSafetensorsDataset` — never LMDB or `.pt`
- [ ] `README_TRAIN.md` + `README_TRAIN_zh-CN.md` aligned pair present
- [ ] Smoke test / doc examples use the matching `datasets/X-Fun-*-Demo` dataset and the correct `metadata*.json` variant — no `internal_datasets` placeholders (reference.md §8)
- [ ] Optional: ComfyUI node in `comfyui/<family>/nodes.py` mirrors the pipeline

## Additional resources

- Detailed file-by-file conventions, class/method shapes, and the model-loading internals: [reference.md](reference.md)
- **Dataset & sampler selection matrix** (which `videox_fun.data` dataset/loader each training task uses), **demo-dataset / metadata-variant selection matrix** (which `datasets/X-Fun-*-Demo` to smoke-test with), **inference task matrix** (which pipeline each `predict_<task>.py` uses), and **multi-GPU preprocessing patterns**: [reference.md](reference.md) §8–§10
- Concrete skeletons (config YAML, `predict_*.py`, pipeline class, training script + DataLoader): [examples.md](examples.md)
