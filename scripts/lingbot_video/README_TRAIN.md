# LingBot-Video TI2V Training Guide

This document provides the complete **ti2v (first-frame image + text to video) full-parameter training** workflow for LingBot-Video (single-stream joint DiT + Qwen3-VL text encoder), including environment setup, data preparation, FSDP distributed training and inference testing.

> **Note**: LingBot-Video has a completely different architecture from LingBot-World (Wan2.2 dual-Transformer + camera control):
> - **Single-stream joint DiT**: video tokens and text tokens are concatenated and processed with full self-attention (`LingBotVideoTransformer3DModel`);
> - **Qwen3-VL text encoder**: prompts go through a chat template, and the first-frame image is fed into the text sequence as visual tokens;
> - **Flow matching**: `x_t = (1-σ)·x0 + σ·noise`, target `noise - x0`, the transformer receives `timestep = σ·1000`;
> - **ti2v first-frame conditioning**: besides entering the Qwen3-VL text sequence, the first frame is also VAE-encoded separately into `cond_latent`, which is written back to the temporal prefix of the latent at every denoising step (inpainting semantics). The loss is computed only on non-conditioning frames.
>
> The training script is `scripts/lingbot_video/train.py`; `scripts/lingbot_video/train.sh` (dense 1.3B) and `scripts/lingbot_video/train_moe.sh` (MoE 30B-A3B) are the default FSDP FULL_SHARD launch configs.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick-test Dataset](#21-quick-test-dataset)
  - [2.2 metadata_lingbot_video_add_width_height.json Format](#22-metadata_lingbot_video_add_width_heightjson-format)
  - [2.3 Captions must be structured JSON captions](#23-captions-must-be-structured-json-captions)
- [3. Full-Parameter Training](#3-full-parameter-training)
  - [3.1 Download Pretrained Models](#31-download-pretrained-models)
  - [3.2 Quick Start (FSDP)](#32-quick-start-fsdp)
  - [3.3 Key Training Arguments](#33-key-training-arguments)
  - [3.4 Trainable Module Selection](#34-trainable-module-selection)
  - [3.5 Resume Training](#35-resume-training)
  - [3.6 How a Training Step Works](#36-how-a-training-step-works)
- [4. Inference Testing](#4-inference-testing)
- [5. FAQ](#5-faq)

---

## 1. Environment Setup

**Option 1: requirements.txt**

```bash
pip install -r requirements.txt
```

**Option 2: manual installation**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image scipy
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

> **Note**: the Qwen3-VL text encoder requires a recent `transformers` — `videox_fun/models/__init__.py` imports `Qwen3VLForConditionalGeneration` from transformers and falls back to `None` when it is missing, printing `Your transformers version is too old to load Qwen3VLForConditionalGeneration`. If you see that line (or `KeyError: 'qwen3_vl'` while reading the config), upgrade transformers.
>
> **Note**: the optional prompt rewriter (see [2.3](#23-captions-must-be-structured-json-captions)) needs an even newer stack (`transformers>=5.x` with the `qwen3_5` module) plus the official rewriter package under `repo/lingbot-video/rewriter`; run it from a dedicated venv so the training environment stays untouched.

---

## 2. Data Preparation

### 2.1 Quick-test Dataset

We provide a demo dataset that ships a handful of ready-to-train samples.

```bash
# Download the official demo dataset
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

It contains 16 videos at 832x480 under `train/` plus `metadata_lingbot_video_add_width_height.json`, whose `text` fields are already LingBot-Video structured JSON captions (see [2.3](#23-captions-must-be-structured-json-captions)), so it can be used for training as-is. LingBot-Video needs **no camera trajectory / action files**, and the ti2v condition frame is taken **automatically from the first frame of each video** (used both as the Qwen3-VL visual input and as the VAE cond_latent), so no separate image files are needed either.

### 2.2 metadata_lingbot_video_add_width_height.json Format

Standard VideoX-Fun format, one entry per video:

```json
[
    {
        "file_path": "train/00000000.mp4",
        "text": "{\"comprehensive_description\": {...}, \"prominent_elements\": [...], \"camera_info\": {...}}",
        "width": 832,
        "height": 480,
        "type": "video"
    },
    {
        "file_path": "train/00000001.mp4",
        "text": "{\"comprehensive_description\": {...}, \"prominent_elements\": [...], \"camera_info\": {...}}",
        "width": 832,
        "height": 480,
        "type": "video"
    }
]
```

- `file_path`: video path relative to `--train_data_dir` (set `--train_data_dir=""` to use absolute paths);
- `text`: **structured JSON caption, serialized as a single string**. The DiT was trained only on rewriter-style JSON captions; natural-language captions are out-of-distribution and degrade fine-tuning. `train.py` validates every entry with `is_valid_caption` at startup and logs a warning with the number of offending entries. See [2.3](#23-captions-must-be-structured-json-captions);
- `width` / `height`: original video size, used to pick the aspect-ratio bucket. **Recommended**; when missing, `AspectRatioBatchImageVideoSampler` probes the file with OpenCV while bucketing (extra IO per epoch). `scripts/process_json_add_width_and_height.py` fills the fields in for an existing metadata file;
- `type`: must be `"video"` (entries without it are treated as images and use `--image_sample_size`).

### 2.3 Captions must be structured JSON captions

The rewriter weights default to `models/Diffusion_Transformer/Qwen3.6-27B` and
`models/Diffusion_Transformer/lingbot-video-rewriter-lora`:

```bash
modelscope download --model Qwen/Qwen3.6-27B --local_dir models/Diffusion_Transformer/Qwen3.6-27B
modelscope download --model Robbyant/lingbot-video-rewriter-lora --local_dir models/Diffusion_Transformer/lingbot-video-rewriter-lora
```

A valid caption is a JSON object carrying the three top-level keys checked by
`is_valid_caption` (`videox_fun/models/lingbot_video_rewriter.py`, the single
source of truth for the schema):

```json
{
  "comprehensive_description": {
    "scene_content_description": "A drone flies over a mountain ridge at sunrise ...",
    "camera_movement_description": "The camera pushes forward slowly."
  },
  "prominent_elements": [
    {
      "name": "mountain ridge",
      "description": "a snow-dusted ridge line",
      "actions": [{"timestamp": "[0.0s - 3.3s]", "action": ""}],
      "location": "center",
      "relative_size": "large",
      "shape_and_color": "grey rock with white snow",
      "texture": "rough",
      "appearance_details": "",
      "relationship": "",
      "orientation": "",
      "pose": "",
      "expression": "",
      "clothing": "",
      "gender": "",
      "skin_tone_and_texture": ""
    }
  ],
  "camera_info": {
    "color": "Cool",
    "frame_size": "Wide",
    "shot_type_angle": "Aerial",
    "lens_size": "Wide",
    "composition": "Balanced",
    "lighting": "Soft light",
    "lighting_type": "Daylight"
  }
}
```

The allowed `camera_info` values are enumerated in `CAMERA_CHOICES`, and
`build_caption` / `element` / `cam` in the same module assemble a caption
programmatically.

**Batch conversion of a dataset** — rewrite the `text` field of the metadata
BEFORE training with the official prompt rewriter (needs the rewriter base VLM +
LoRA adapter):

```bash
export REWRITER_BASE_MODEL=models/Diffusion_Transformer/Qwen3.6-27B
export REWRITER_ADAPTER=models/Diffusion_Transformer/lingbot-video-rewriter-lora
python scripts/lingbot_video/prepare_captions.py \
    --metadata datasets/my_dataset/metadata.json \
    --data_root datasets/my_dataset \
    --output datasets/my_dataset/metadata_json.json \
    --mode ti2v --duration 3.3
# then in train.sh: DATASET_META_NAME="datasets/my_dataset/metadata_json.json"
```

- `--mode`: `t2v` / `ti2v` / `t2i`; with `ti2v` the video's first frame is read (decord, OpenCV fallback) and fed to the rewriter, so use the same mode you train with;
- `--duration`: clip duration in seconds handed to the rewriter — match your training clip (`video_sample_n_frames / fps`, e.g. 81 frames @ 24fps ≈ 3.3s);
- `--base` / `--adapter`: rewriter weights, alternatively `REWRITER_BASE_MODEL` / `REWRITER_ADAPTER`;
- entries that already hold a valid JSON caption are kept as-is unless `--overwrite` is passed; the output file is rewritten after every sample so long runs can be interrupted and resumed, and `--metadata` is never modified in place;
- entries the rewriter fails on keep their original text and are reported in the final summary — fix or re-run before training.

**Single prompt** (e.g. to build `--validation_prompts`) — `ensure_json_caption`
is the one entry point; already-valid captions pass through untouched, otherwise
the rewriter is loaded, used, freed, and the result cached:

```python
from PIL import Image
from videox_fun.models.lingbot_video_rewriter import ensure_json_caption

caption = ensure_json_caption(
    "A drone slowly flies over the mountains, clouds drift in the background.",
    mode="ti2v", duration=3.3,
    first_frame=Image.open("asset/1.png").convert("RGB"),   # ti2v only
    cache_file="samples/caption_cache.json",
)
```

---

## 3. Full-Parameter Training

### 3.1 Download Pretrained Models

LingBot-Video ships two variants (diffusers-format directories containing `transformer` / `vae` / `text_encoder` / `processor` / `scheduler` subfolders):

| Model | Notes |
| --- | --- |
| `lingbot-video-dense-1.3b` | Dense 1.3B, trainable on 1-2 GPUs, recommended for pipeline validation (`train.sh`) |
| `lingbot-video-moe-30b-a3b` | MoE 30B-A3B, multi-GPU FSDP FULL_SHARD training (`train_moe.sh`, ≥8×80GB recommended) |

Directory layout (`train.py` loads each subfolder by path, so all five must be present):

```
lingbot-video-dense-1.3b/
├── transformer/       # LingBotVideoTransformer3DModel (the only trained module)
├── vae/               # AutoencoderKLQwenImage (frozen)
├── text_encoder/      # Qwen3-VL (frozen)
├── processor/         # Qwen3-VL processor (AutoProcessor)
└── scheduler/         # FlowUniPCMultistepScheduler (only sigma_max / sigma_min are read)
```

The MoE model additionally ships a `refiner` DiT, which training does not touch (it is only used by `examples/lingbot_video/predict_t2v_refine.py`).

### 3.2 Quick Start (FSDP)

Edit the three environment variables at the top of `scripts/lingbot_video/train.sh` (dense 1.3B) or `scripts/lingbot_video/train_moe.sh` (MoE 30B) and run:

```bash
export MODEL_NAME="models/Diffusion_Transformer/lingbot-video-dense-1.3b"
export DATASET_NAME="datasets/X-Fun-Videos-Demo"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_lingbot_video_add_width_height.json"

sh scripts/lingbot_video/train.sh
```

The script launches **FSDP FULL_SHARD with `LingBotVideoBlock` as the wrap unit** (the MoE experts of the 30B model must be sharded together with their block, so the wrap class is fixed to `LingBotVideoBlock`). `train.sh` in full:

```bash
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=LingBotVideoBlock --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" \
    --fsdp_cpu_ram_efficient_loading False scripts/lingbot_video/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_lingbot_video_ti2v" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --train_shift=3.0 \
  --trainable_modules "." \
  --resume_from_checkpoint=latest
```

`train_moe.sh` is the same command with `MODEL_NAME=models/Diffusion_Transformer/lingbot-video-moe-30b-a3b` and `--output_dir="output_dir_lingbot_video_moe_ti2v"`; every other argument (including the 640 sample sizes) is identical.

> **Note**: `--use_fsdp` is not passed on the command line — `train.py` detects the accelerate FSDP plugin, derives the FSDP stage from the sharding strategy and switches to sharded checkpoint saving on its own. Neither script sets `--validation_prompts`, so periodic sampling is off by default (see [3.3](#33-key-training-arguments)).

> **VRAM tips**: with `--train_batch_size=1` + `--gradient_checkpointing`, dense 1.3B fits 81-frame 480p clips on 2×H20 (97GB); both scripts ship with 640 sample sizes, which needs correspondingly more — lower `--video_sample_size` / `--token_sample_size` if you OOM. MoE 30B is recommended on ≥8×80GB. The frozen bf16 Qwen3-VL text encoder is encoded online every step and is also FSDP-sharded (its `Qwen3VLTextDecoderLayer` / `Qwen3VLVisionBlock` layers are wrapped).

Monitor with tensorboard:

```bash
tensorboard --logdir=output_dir_lingbot_video_ti2v
```

### 3.3 Key Training Arguments

| Argument | Notes |
| --- | --- |
| `--pretrained_model_name_or_path` | LingBot-Video model root (must contain `transformer/ vae/ text_encoder/ processor/ scheduler/`) |
| `--train_data_dir` / `--train_data_meta` | Dataset root + metadata json; `file_path` entries are resolved against the root |
| `--train_batch_size=1` | **Keep it 1.** Qwen3-VL has variable-length text; batch>1 triggers the padding + attention_mask path, which works but is inefficient. Raise the effective batch with `--gradient_accumulation_steps` |
| `--video_sample_n_frames=81` | Sampled frames; the collate function rounds the batch length down to `4n+1` (VAE temporal ratio 4) |
| `--video_sample_stride=1` | Frame stride inside the sampled clip |
| `--video_repeat=1` | Repeats every video entry this many times in the dataset list (balances mixed image/video datasets) |
| `--video_sample_size=640` / `--image_sample_size=640` | Short-side bucket size for video / image entries (both launch scripts use 640) |
| `--token_sample_size=640` | Reference resolution of the token budget `video_sample_n_frames × token_sample_size²` used by `--training_with_video_token_length` |
| `--enable_bucket` + `--random_hw_adapt` | Aspect-ratio bucketing + random resolution per batch |
| `--training_with_video_token_length` | Keeps the total video token count constant while adapting H/W (fewer frames at higher resolution) |
| `--fix_sample_size H W` | Forces one fixed bucket size; it also disables `--random_hw_adapt` / `--training_with_video_token_length` |
| `--random_ratio_crop` | Random aspect-ratio crop instead of the closest-bucket resize |
| `--uniform_sampling` | Stratified uniform σ sampling: each rank group draws from its own slice of the `--train_sampling_steps` grid, so the ranks jointly cover the full noise range. Without it, σ indices come from the `--weighting_scheme` density (`--logit_mean` / `--logit_std` / `--mode_scale`) |
| `--train_sampling_steps=1000` | Number of discrete σ levels in the training sigma table |
| `--train_shift=3.0` | Sigma shift, consistent with the inference default shift=3.0: `σ' = s·σ/(1+(s-1)·σ)` |
| `--weighting_scheme` | Loss weighting (`sigma_sqrt` / `logit_normal` / `mode` / `cosmap` / `none`), default `none` (uniform) |
| `--gradient_checkpointing` | Gradient checkpointing, roughly halves activation memory at a slight speed cost |
| `--trainable_modules "."` | Full-parameter training; see [3.4](#34-trainable-module-selection) |
| `--trainable_modules_low_learning_rate` | Same substring matching, but those parameters get `learning_rate / 2` |
| `--transformer_path` / `--vae_path` | Load transformer / VAE weights from an external `.safetensors` or `.pt` before training starts |
| `--vae_mini_batch=1` | VAE chunked encoding batch size |
| `--low_vram` | Low-VRAM mode: VAE/Qwen3-VL are moved between GPU and CPU on demand |
| `--max_grad_norm=0.05` | Gradient clipping. In non-FSDP mode `--initial_grad_norm_ratio` / `--abnormal_norm_clip_start` additionally decay the limit and damp abnormal gradients; under FSDP the raw `max_grad_norm` is used |
| `--checkpointing_steps=50` / `--checkpoints_total_limit` | Checkpoint interval / maximum number of kept checkpoints |
| `--validation_prompts` + `--validation_paths` | ti2v sampling with `LingBotVideoI2VPipeline` every `--validation_steps` steps and every `--validation_epochs` epochs (prompts paired one-to-one with first-frame images, asserted at startup). Sampling is fixed to `guidance_scale=3.0`, 25 steps, `shift=--train_shift`, fps 24, resolution derived from `image_sample_size²` and the image aspect ratio; videos are written to `output_dir/sample/sample-{step}-rank{rank}-image-{i}.mp4`. The prompts must be JSON captions too |
| `--use_ema` | Non-FSDP only — FSDP FULL_SHARD raises `NotImplementedError` |
| `--report_model_info` | Logs per-parameter gradient norms to tensorboard (non-FSDP only) |
| `--resume_from_checkpoint=latest` | Resume from the latest checkpoint |

### 3.4 Trainable Module Selection

`--trainable_modules` filters parameter names by **substring match**; the default `["."]` trains everything:

```bash
# Full-parameter (default)
--trainable_modules "."

# Attention projections only
--trainable_modules "attn"

# FFN / MoE experts only
--trainable_modules "ffn" "experts"
```

Anything matched by `--trainable_modules_low_learning_rate` instead is trained at half the learning rate; a parameter matched by both lists is assigned to the full-rate group.

### 3.5 Resume Training

- Under FSDP (the default of both launch scripts) each `checkpoint-*` directory contains:
  - `diffusion_pytorch_model.safetensors` (full weights gathered on the main process, cast to bf16, directly usable by the inference scripts);
  - the accelerate sharded state (optimizer/scheduler, for resuming);
  - `sampler_pos_start.pkl` (sampler position + epoch, restored on resume to keep data order).
- Without FSDP the checkpoint holds a diffusers-style `transformer/` folder instead (plus `transformer_ema/` when `--use_ema` is set).
- Simply add `--resume_from_checkpoint=latest` to resume; on load, the sampler position is rewound by `dataloader_num_workers × num_processes × 2` samples to compensate for prefetched batches.
- A final `checkpoint-{global_step}` is also written when training finishes.

### 3.6 How a Training Step Works

Each training step (strictly aligned with the inference path):

1. **Data**: the DataLoader yields `(B, C, T, H, W) ∈ [-1, 1]` (resize + center-crop to the bucket size, then normalized with mean/std 0.5); the first frame `[:, :, 0]` is used as the condition image automatically. The first batch of the run is dumped to `output_dir/sanity_check/` for inspection. The dataset applies the default 10% text dropout (`text_drop_ratio=0.1`), i.e. some samples train with an empty prompt for classifier-free guidance;
2. **VAE encoding** (frozen, bf16; only the `latents_mean/std` normalization is computed in fp32):
   - full video → `latents` (normalized into DiT space via `latents_mean/std`);
   - first frame encoded separately → `cond_latent` (1 temporal frame);
3. **Qwen3-VL encoding** (frozen, bf16, no_grad): the prompt is wrapped with the chat template; the first frame is `smart_resize`d (aligned to `patch_size×merge_size`) and encoded as image tokens, producing `prompt_embeds` and `prompt_mask`;
4. **Flow-matching noise**:
   - σ is sampled from the sigma table (`linspace(sigma_max, sigma_min, N+1)[:-1]` with `--train_shift` applied, `N = --train_sampling_steps`; `sigma_max` / `sigma_min` come from the pretrained `scheduler/` config);
   - `x_t = (1-σ)·x0 + σ·noise`;
5. **ti2v inpainting**: `cond_latent` overwrites the temporal prefix of `x_t` (clean frame, no noise) and a `frame_mask` (condition frames = 0) is recorded;
6. **Forward**: `transformer(x_t, σ·1000, prompt_embeds, encoder_attention_mask=prompt_mask)` (the timestep scaling matches the pipeline's `_transformer_timestep`);
7. **Loss**: `Σ(MSE(noise_pred, noise - x0) · frame_mask · weighting) / (frame_mask.sum() · C · H · W)` — condition frames never contribute, and `weighting` comes from `--weighting_scheme` (all-ones for `none`). The `C·H·W` factor is a constant rescaling of the reported/optimized loss.

> Numerically sensitive modules (norm / router / modulation / scale_shift_table) stay in fp32 following the model's own rules while the rest runs in bf16, identical to inference. Under FSDP every parameter and buffer is force-cast to bf16 instead, because one flat FSDP shard requires a uniform dtype.

---

## 4. Inference Testing

The produced `checkpoint-*/diffusion_pytorch_model.safetensors` can be fed directly to `examples/lingbot_video/predict_i2v.py` via `transformer_path`:

```python
# examples/lingbot_video/predict_i2v.py
model_name       = "models/Diffusion_Transformer/lingbot-video-dense-1.3b"
transformer_path = "output_dir_lingbot_video_ti2v/checkpoint-500/diffusion_pytorch_model.safetensors"
shift            = 3.0     # keep it equal to the training --train_shift
```

Then:

```bash
python examples/lingbot_video/predict_i2v.py
```

The script loads the fine-tuned weights on top of the base model (`load_state_dict(..., strict=False)`) and runs ti2v sampling; `predict_t2v.py` / `predict_t2v_refine.py` work the same way (the latter additionally needs the refiner weights, which only ship with the MoE model as its `refiner` subfolder).

`predict_i2v.py` also turns its plain `prompt` into a JSON caption on its own: before any generation model is loaded it calls `ensure_json_caption(..., mode="ti2v", duration=round(video_length / fps, 2), first_frame=validation_image, base=rewriter_base_model, adapter=rewriter_lora_path)` and caches the result in `save_path/caption_cache.json`. `rewriter_base_model` / `rewriter_lora_path` already point at the weights from [2.3](#23-captions-must-be-structured-json-captions); passing an already-valid JSON caption skips the rewriter entirely.

---

## 5. FAQ

1. **OOM**: first try `--gradient_checkpointing` + `--train_batch_size=1`; then `--low_vram`; otherwise reduce `--video_sample_n_frames` / `--video_sample_size`.
2. **Loss not decreasing**: make sure `--train_shift` matches inference (3.0); check that captions match the video content; on tiny datasets you may raise `--learning_rate` (e.g. 5e-5) to see whether the model responds at all.
3. **MoE 30B training**: the code path supports it natively (FSDP FULL_SHARD wraps `LingBotVideoBlock`, experts travel with their block), but validate the data pipeline on dense 1.3B first.
4. **EMA**: `--use_ema` only works in non-FSDP (single-GPU/DDP) mode; FSDP FULL_SHARD is incompatible with EMAModel.
5. **Batch > 1**: it works, but Qwen3-VL right-pads the text and the DiT takes the attention_mask path, which is slower than batch=1.
6. **`... dataset captions are NOT structured JSON captions` warning at startup**: the metadata still holds natural-language text — convert it as described in [2.3](#23-captions-must-be-structured-json-captions). Training continues, but the DiT is fed out-of-distribution prompts.
7. **Validation crashes**: `log_validation` catches every exception, prints `Eval error on rank ...` and continues training, so a failed sampling step never aborts the run — check the printed message (typically a non-JSON validation prompt or a missing `--validation_paths` image).
