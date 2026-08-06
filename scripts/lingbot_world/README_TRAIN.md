# LingBot-World I2V Training Guide

This document provides a complete workflow for training the LingBot-World camera-controlled I2V model on top of Wan2.2, including environment setup, data preparation, distributed training, and inference testing.

> **Note**: LingBot-World is a camera-pose controlled Image-to-Video model built on the Wan2.2 I2V dual-Transformer backbone. Given a reference image, a text prompt **and a per-frame camera trajectory** (`poses.npy` + `intrinsics.npy`), it generates a video whose viewpoint follows the trajectory. The training script is a superset of `scripts/wan2.2/train.py`: adding `--enable_camera_control` switches the model to `WanTransformer3DModel_LingbotWorld`, the dataset to `LingbotImageVideoDataset`, and injects a per-sample camera condition into the forward pass. Everything else (dual-Transformer boundary type, FSDP, bucket sampling, EMA, ...) is inherited unchanged.

---

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Dataset Structure](#21-dataset-structure)
  - [2.2 metadata.json Format](#22-metadatajson-format)
  - [2.3 Camera Trajectory Files](#23-camera-trajectory-files)
  - [2.4 Intrinsics Convention](#24-intrinsics-convention)
  - [2.5 Relative vs Absolute Path Usage](#25-relative-vs-absolute-path-usage)
- [3. Full Parameter Training](#3-full-parameter-training)
  - [3.1 Download Pretrained Model](#31-download-pretrained-model)
  - [3.2 Quick Start (FSDP)](#32-quick-start-fsdp)
  - [3.3 Training Parameter Reference](#33-training-parameter-reference)
  - [3.4 Trainable Modules](#34-trainable-modules)
  - [3.5 Boundary Type (High / Low Noise)](#35-boundary-type-high--low-noise)
  - [3.6 What Happens Under the Hood](#36-what-happens-under-the-hood)
- [4. Inference Testing](#4-inference-testing)
  - [4.1 Load Trained Checkpoints](#41-load-trained-checkpoints)
  - [4.2 Image-to-Video (I2V) Inference](#42-image-to-video-i2v-inference)
- [5. Additional Resources](#5-additional-resources)

---

## 1. Environment Setup

**Option 1: Using requirements.txt**

```bash
pip install -r requirements.txt
```

**Option 2: Manual Installation**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image scipy
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

> **Note**: `scipy` is required by the LingBot camera utilities (`Slerp` / `interp1d` for pose interpolation).

---

## 2. Data Preparation

### 2.1 Dataset Structure

LingBot-World training expects each video sample to be paired with a **camera trajectory directory** that contains `poses.npy` and `intrinsics.npy`.

```
📦 datasets/
├── 📂 lingbot_world/
│   ├── 📂 videos/
│   │   ├── 📄 clip_000001.mp4
│   │   ├── 📄 clip_000002.mp4
│   │   └── 📄 ...
│   ├── 📂 actions/
│   │   ├── 📂 clip_000001/
│   │   │   ├── 📄 poses.npy
│   │   │   └── 📄 intrinsics.npy
│   │   ├── 📂 clip_000002/
│   │   │   ├── 📄 poses.npy
│   │   │   └── 📄 intrinsics.npy
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

The reference camera trajectory shipped in `asset/lingbot_demo/` (used by `examples/lingbot_world/predict_i2v.py`) is a working example of the file layout.

You can also download the ready-to-use demo dataset from ModelScope (7 videos at 832x480 with paired camera trajectories, suitable for smoke-testing the training loop):

```bash
modelscope download --dataset PAI/X-Fun-Videos-Lingbot-Demo --local_dir datasets/X-Fun-Videos-Lingbot-Demo
```

### 2.2 metadata.json Format

The annotation file adds one new field on top of the standard Wan2.2 schema — `action_path`.

```json
[
  {
    "file_path": "videos/clip_000001.mp4",
    "action_path": "actions/clip_000001",
    "text": "The video presents a soaring journey through a fantasy jungle...",
    "type": "video",
    "width": 832,
    "height": 480
  },
  {
    "file_path": "videos/clip_000002.mp4",
    "action_path": "actions/clip_000002",
    "text": "A first-person drone-style shot descending toward a stone castle...",
    "type": "video",
    "width": 832,
    "height": 480
  }
]
```

**Key Field Descriptions**:
- `file_path`: Video path (relative or absolute path).
- `action_path`: Directory containing `poses.npy` and `intrinsics.npy`. Relative paths are resolved against `--action_data_root` (defaults to `--train_data_dir`).
- `text`: Video description (English prompt).
- `type`: Data type, should be `"video"`. Image entries or video entries without `action_path` are silently skipped by the camera-injection path — the training loop degrades to the plain Wan2.2 forward on those samples.
- `width` / `height`: Video width and height (**recommended to provide**, used for bucket training).
  - You can use `scripts/process_json_add_width_and_height.py` to extract width and height from JSON files without these fields.

### 2.3 Camera Trajectory Files

Both files are plain NumPy arrays and follow the exact convention used by the inference script (`examples/lingbot_world/predict_i2v.py`):

- `poses.npy`: `float32`, shape `[F, 4, 4]`. Per-frame camera-to-world matrices in the **OpenCV coordinate system** (x right, y down, z forward). `F` must be at least `--video_sample_n_frames`, and `poses[i]` must correspond to the `i`-th frame of the video.
- `intrinsics.npy`: `float32`, shape `[F, 4]` or `[4]`. Each row is `(fx, fy, cx, cy)` in pixels **at the calibration resolution** (see [2.4](#24-intrinsics-convention)). Only the first row is used, matching the inference logic.

**Frame Sampling Alignment**:
- The base `ImageVideoDataset` selects a contiguous window of `min(video_sample_n_frames, video_length // video_sample_stride)` frames with a random start offset.
- `LingbotImageVideoDataset` slices `poses.npy` at the **same frame indices** used for the RGB clip (`c2ws[batch_index]`), so the trajectory always stays synchronized with the sampled video frames.
- The training helper `prepare_lingbot_dit_cond_dict_from_c2ws` then interpolates the sampled poses to `lat_f = (F - 1) // 4 + 1` latent frames and builds the plücker embedding token grid — identical to the inference path.

If `poses.npy` is shorter than `int(batch_index.max()) + 1`, the trajectory is dropped for that sample and the transformer runs without camera injection.

### 2.4 Intrinsics Convention

`intrinsics.npy` is calibrated at the **original** video resolution, not the training resolution. Each camera-controlled video entry **must** declare that calibration resolution per-sample in the annotation file via `intrinsics_org_height` / `intrinsics_org_width`:

```json
{
  "file_path": "videos/xxx.mp4",
  "action_path": "actions/xxx",
  "type": "video",
  "intrinsics_org_height": 480,
  "intrinsics_org_width": 832
}
```

At training time the intrinsics are automatically re-scaled from `(intrinsics_org_height, intrinsics_org_width)` to the current bucket size by `get_Ks_transformed` in `videox_fun/data/utils.py`. Set the fields to whatever resolution your `intrinsics.npy` was calibrated on — do not modify the intrinsics values themselves. This is per-sample, so datasets calibrated at different resolutions can be mixed freely.

> **Required.** Both fields must be present together on every camera-controlled sample. A sample that carries a camera trajectory but is missing them is **skipped during data loading** (the dataset re-samples another entry), so it never reaches the training step. (The old `--intrinsics_org_height/width` CLI flags have been removed.)

### 2.5 Relative vs Absolute Path Usage

**Relative Path**:

If your data uses relative paths, configure in the training script:

```bash
export DATASET_NAME="datasets/lingbot_world/"
export DATASET_META_NAME="datasets/lingbot_world/metadata.json"
```

`action_path` entries in `metadata.json` are resolved against `--action_data_root` (defaults to `$DATASET_NAME`).

**Absolute Path**:

If your data uses absolute paths, configure in the training script:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/lingbot_world/metadata.json"
```

You can also set `--action_data_root` explicitly if the trajectories live in a separate root.

> 💡 **Recommendation**: If the dataset is small and stored locally, use relative paths. If the dataset is stored on external storage (e.g., NAS, OSS) or shared across multiple machines, use absolute paths.

---

## 3. Full Parameter Training

### 3.1 Download Pretrained Model

Fine-tune from a released LingBot-World checkpoint (recommended, matches inference):

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Camera-pose LingBot-World base weights (same layout as Wan2.2-I2V-A14B).
modelscope download --model your-org/lingbot-world-base-cam --local_dir models/Diffusion_Transformer/lingbot-world-base-cam
```

You can also start from a plain Wan2.2-I2V-A14B checkpoint. In that case the LingBot-specific layers (`cam_injector_*`, `cam_scale_layer`, `cam_shift_layer`, `patch_embedding_wancamctrl`, `c2ws_hidden_states_layer{1,2}`) are randomly initialized and **must** be included in `--trainable_modules` (see [3.4](#34-trainable-modules)).

The config file remains `config/wan2.2/wan_civitai_i2v.yaml` because LingBot-World reuses the Wan2.2 I2V VAE / text encoder / dual-Transformer layout.

### 3.2 Quick Start (FSDP)

After downloading the dataset as in **2.1 Dataset Structure** (or simply downloading the demo dataset `PAI/X-Fun-Videos-Lingbot-Demo`) and the pretrained model as in **3.1 Download Pretrained Model**, you can directly copy and run the quick start command.

We recommend FSDP for LingBot-World training. Because LingBot replaces `WanAttentionBlock` with `LingbotWorldWanAttentionBlock` (which adds four `cam_*` linear layers), the FSDP transformer-wrap class is set to the LingBot block so its parameters stay in the same FSDP unit as attention/FFN. The shipped `scripts/lingbot_world/train.sh` uses a plain `accelerate launch` (no FSDP) on the **high-noise** branch at 640 resolution — add the FSDP flags below when the model does not fit.

**LingBot-World I2V Training Example** (low-noise branch):

```sh
export MODEL_NAME="models/Diffusion_Transformer/lingbot-world-base-cam"
export DATASET_NAME="datasets/X-Fun-Videos-Lingbot-Demo"
export DATASET_META_NAME="datasets/X-Fun-Videos-Lingbot-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=LingbotWorldWanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/lingbot_world/train.py \
  --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=832 \
  --video_sample_size=832 \
  --token_sample_size=832 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_lingbot_world_i2v_low" \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --boundary_type="low" \
  --train_mode="i2v" \
  --enable_camera_control \
  --trainable_modules "cam_injector_layer1" "cam_injector_layer2" "cam_scale_layer" "cam_shift_layer" "patch_embedding_wancamctrl" "c2ws_hidden_states_layer1" "c2ws_hidden_states_layer2" \
  --resume_from_checkpoint=latest
```

### 3.3 Training Parameter Reference

LingBot-World inherits every argument from `scripts/wan2.2/train.py`. Only the LingBot-specific flags are listed here — refer to [scripts/wan2.2/README_TRAIN.md](../wan2.2/README_TRAIN.md#33-training-parameter-reference) for the shared ones.

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--enable_camera_control` | Master switch. When set: use `WanTransformer3DModel_LingbotWorld`, use `LingbotImageVideoDataset`, feed `dit_cond_dict` to the forward pass. | - |
| `--action_data_root` | Root directory for resolving relative `action_path` entries. Defaults to `--train_data_dir` when unset. | `None` |
| `intrinsics_org_height` / `intrinsics_org_width` (annotation field) | Per-sample calibration resolution (px) of `intrinsics.npy`. Declared in the annotation file, **not** on the CLI. Required for every camera-controlled sample. | 480 / 832 |
| `--control_type` | LingBot-World control type. Only `cam` (6-dim plücker) is supported. | `cam` |

**Notes**:
- `train_batch_size=1` is the recommended (and validated) setting. Camera trajectories can have different lengths per sample so they are carried through the batch as a python list; the training loop currently consumes trajectory index 0. Use `--gradient_accumulation_steps` to raise the effective batch size.
- Without `--enable_camera_control`, `scripts/lingbot_world/train.py` behaves exactly like `scripts/wan2.2/train.py`.

### 3.4 Trainable Modules

Which parameters get gradients is controlled by `--trainable_modules` (substring match against parameter names). Two common recipes:

**(a) Only train the LingBot camera-injection layers** — recommended when fine-tuning a released LingBot-World checkpoint, or when bootstrapping the LingBot layers on top of a plain Wan2.2 I2V checkpoint:

```
--trainable_modules \
    "cam_injector_layer1" "cam_injector_layer2" \
    "cam_scale_layer" "cam_shift_layer" \
    "patch_embedding_wancamctrl" \
    "c2ws_hidden_states_layer1" "c2ws_hidden_states_layer2"
```

**(b) Full fine-tune** — every parameter is trainable:

```
--trainable_modules "."
```

Combine with FSDP for memory.

### 3.5 Boundary Type (High / Low Noise)

Wan2.2 uses a dual-Transformer setup: a **high-noise** transformer denoises the coarse structure and a **low-noise** transformer refines the details, split at `boundary` (default 0.900).

- `--boundary_type=low`: Train the low-noise transformer, sample timesteps from `[0, boundary * T]`.
- `--boundary_type=high` (what `train.sh` ships with): Train the high-noise transformer, sample timesteps from `[boundary * T, T]`.
- `--boundary_type=full`: Train a single transformer over the whole `[0, T]` range (rarely used for LingBot).

To cover the full pipeline, run the script twice — once with `--boundary_type=low` and once with `--boundary_type=high`. At inference time, pass both trained checkpoints via `transformer_path` (low) and `transformer_high_path` (high), or point `model_name` at a directory that holds both under the sub-paths defined in `config/wan2.2/wan_civitai_i2v.yaml`.

### 3.6 What Happens Under the Hood

The training script differs from the plain Wan2.2 I2V trainer in exactly four places:

1. **Model class swap** — `scripts/lingbot_world/train.py` selects `WanTransformer3DModel_LingbotWorld` when `--enable_camera_control` is set. The class inherits from `Wan2_2Transformer3DModel` and adds `patch_embedding_wancamctrl` / `c2ws_hidden_states_layer{1,2}` (global plücker → hidden-state projection) and per-block `cam_injector_layer{1,2}` / `cam_scale_layer` / `cam_shift_layer` inside `LingbotWorldWanAttentionBlock`.
2. **Dataset extension** — `LingbotImageVideoDataset` is a drop-in subclass of `ImageVideoDataset`. It reads `poses.npy` / `intrinsics.npy` from the sample's `action_path`, slices `poses.npy` at the same frame indices used for the RGB clip, and stores the result in `sample["action_c2ws"]` / `sample["action_intrinsics"]`.
3. **Camera condition preparation** — inside the training step, `prepare_lingbot_dit_cond_dict_from_c2ws(...)` is called per sample. This mirrors `prepare_lingbot_dit_cond_dict` from the inference path: interpolate the sampled poses to `lat_f = (frame_num - 1) // vae_temporal_ratio + 1` frames, compute framewise relative poses, scale intrinsics to the current bucket size, build the plücker embedding, and pack it into a `[1, C, lat_f, lat_h, lat_w]` tensor.
4. **Forward pass** — the produced `dit_cond_dict` is passed to `transformer3d(..., dit_cond_dict=dit_cond_dict)`. `WanTransformer3DModel_LingbotWorld.forward` embeds the plücker tensor into per-token camera hidden states and shares them with every `LingbotWorldWanAttentionBlock`, which applies the `(1 + cam_scale) * x + cam_shift` modulation between self-attention and cross-attention.

Everything else — flow-matching loss, gradient clipping, EMA, FSDP sharded state-dict saving — is inherited from `scripts/wan2.2/train.py` unchanged.

---

## 4. Inference Testing

### 4.1 Load Trained Checkpoints

Point [examples/lingbot_world/predict_i2v.py](../../examples/lingbot_world/predict_i2v.py) at your trained checkpoints. Two options:

**Option A** — save both branches under the same `model_name` directory following the sub-paths in `wan_civitai_i2v.yaml`:

```
model_name/
├── low_noise_model/            # trained by boundary_type=low
├── high_noise_model/           # trained by boundary_type=high
├── Wan2.1_VAE.pth
├── models_t5_umt5-xxl-enc-bf16.pth
└── ...
```

Then set `model_name = "/path/to/your/trained/lingbot-world"` in the inference script.

**Option B** — override individual transformer paths in the inference script:

```python
transformer_path      = "output_dir_lingbot_world_i2v_low/checkpoint-5000/diffusion_pytorch_model.safetensors"
transformer_high_path = "output_dir_lingbot_world_i2v_high/checkpoint-5000/diffusion_pytorch_model.safetensors"
```

### 4.2 Image-to-Video (I2V) Inference

Run the following command for single GPU inference:

```bash
python examples/lingbot_world/predict_i2v.py
```

Edit `examples/lingbot_world/predict_i2v.py` according to your needs. For initial inference, focus on the following parameters:

```python
# Choose based on GPU memory
GPU_memory_mode = "model_cpu_offload"
# Based on actual model path
model_name = "models/Diffusion_Transformer/lingbot-world-base-cam"
# Path to trained low-noise weights, e.g., "output_dir_lingbot_world_i2v_low/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = None
# Path to trained high-noise weights
transformer_high_path = None
# Starting image for image-to-video
validation_image_start = "asset/lingbot_demo/image.jpg"
# Directory containing poses.npy and intrinsics.npy (same layout as the training action_path)
action_path = "asset/lingbot_demo"
# Write based on the content you want to generate
prompt = "The video presents a soaring journey through a fantasy jungle..."
# ...
```

Refer to [scripts/wan2.2/README_TRAIN.md#4-inference-testing](../wan2.2/README_TRAIN.md#4-inference-testing) for GPU-memory-mode options and multi-GPU parallel inference — the LingBot inference script honors the same flags because it reuses the standard `Wan2_2I2VPipeline`.

---

## 5. Additional Resources

- **Base Wan2.2 training doc**: [scripts/wan2.2/README_TRAIN.md](../wan2.2/README_TRAIN.md)
- **Inference reference**: [examples/lingbot_world/predict_i2v.py](../../examples/lingbot_world/predict_i2v.py)
- **Camera / plücker utilities**: [videox_fun/data/utils.py](../../videox_fun/data/utils.py) — `prepare_lingbot_dit_cond_dict{,_from_c2ws}`
- **Model code**: [videox_fun/models/wan_transformer3d_lingbot_world.py](../../videox_fun/models/wan_transformer3d_lingbot_world.py)
- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
