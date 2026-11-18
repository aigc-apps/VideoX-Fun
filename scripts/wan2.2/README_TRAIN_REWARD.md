# Wan2.2 Reward LoRA Training Guide

This document provides a complete workflow for **Reward Backpropagation** fine-tuning of Wan2.2, i.e. optimizing a LoRA so that the generated videos better align with human preferences. It covers environment configuration, data preparation, the reward-specific training arguments (single / multi reward, differentiable VAE decoding, backprop strategy), training validation, and inference testing.

> **Note**: Wan2.2 is a video generation model supporting text-to-video (T2V), image-to-video (I2V), and text-image-to-video (TI2V). The A14B variants adopt a dual-Transformer architecture (high-noise / low-noise models), while the TI2V-5B variant uses a single-Transformer architecture. Reward LoRA training runs a differentiable rollout (a short denoising trajectory) and back-propagates the reward model's loss into the trainable expert, so it needs more VRAM than standard LoRA fine-tuning. This guide covers training your own reward LoRA for any of the Wan2.2 variants.

We explore the Reward Backpropagation technique <sup>[1](#ref1) [2](#ref2)</sup> to optimize the generated videos by [Wan2.2](https://github.com/aigc-apps/VideoX-Fun) for better alignment with human preferences. The training script is [train_reward_lora.py](https://github.com/aigc-apps/VideoX-Fun/blob/main/scripts/wan2.2/train_reward_lora.py); you can train your own reward LoRA and load it onto the corresponding base model as a plug-in.

> [!NOTE]
> Reward backpropagation tends to reward-hack: as the reward keeps rising the perceptual quality can plateau or degrade (adversarial artifacts in the background). Prefer a **video reward** (e.g. `VideoAlignReward`) or a **multi-reward** mix (e.g. `HPSv3Reward` + `VideoAlignReward`) over a pure image reward to mitigate the loss of temporal dynamism — see [3.4](#34-single-vs-multi-reward-training) and [5. Limitations](#5-limitations).

---

## Table of Contents
- [1. Environment Configuration](#1-environment-configuration)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 Quick Test Dataset](#21-quick-test-dataset)
  - [2.2 Reward Model Weights](#22-reward-model-weights)
  - [2.3 Dataset Structure and metadata.json](#23-dataset-structure-and-metadatajson)
- [3. Reward LoRA Training](#3-reward-lora-training)
  - [3.1 Download Pre-trained Base Model](#31-download-pre-trained-base-model)
  - [3.2 Quick Start (FSDP)](#32-quick-start-fsdp)
  - [3.3 Reward-specific Parameter Explanation](#33-reward-specific-parameter-explanation)
  - [3.4 Single vs Multi-Reward Training](#34-single-vs-multi-reward-training)
  - [3.5 Differentiable Decoding and `--use_1steps_vae`](#35-differentiable-decoding-and---use_1steps_vae)
  - [3.6 Training Validation](#36-training-validation)
- [4. Inference Testing](#4-inference-testing)
- [5. Limitations](#5-limitations)
- [6. More Resources](#6-more-resources)
- [References](#references)

---

## 1. Environment Configuration

**Method 1: Using requirements.txt**

```bash
pip install -r requirements.txt
```

**Method 2: Manual Dependency Installation**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

> Some video reward models (e.g. `VideoAlignReward`, `HPSv3Reward`) are built on a Qwen2-VL backbone and require `transformers==4.51.3`. Install the matching version when using those rewards.

**Method 3: Using Docker**

When using Docker, please ensure that the graphics card driver and CUDA environment are correctly installed on your machine, then execute the following commands:

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. Data Preparation

### 2.1 Quick Test Dataset

Reward LoRA training runs a differentiable rollout from text prompts, so a **prompt-only** dataset is sufficient for the standard T2V reward path (`train_mode="normal"`), which is the recommended setup. For the I2V / inpaint path, a video dataset provides the reference conditioning.

```bash
# Download official example dataset
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 2.2 Reward Model Weights

The backbone / checkpoint source differs per reward:

- **`HPSReward` (HPS v2.1) and `MPSReward`**: the score heads are tiny and are **downloaded automatically on first run** into the torch hub cache (`~/.cache/torch/hub`); note that **HPS additionally needs two special config files patched into `hpsv2`'s bundled open_clip** (see below), while MPS needs no extra setup;
- **`VideoAlignReward` (VideoReward) and `HPSv3Reward`**: built on Qwen2-VL backbones and fairly large, so **stage them locally in advance** and point `--reward_fn_kwargs` at the local paths.

The dependencies and sources are summarized below (see [3.3](#33-reward-specific-parameter-explanation) for the `--reward_fn` values):

| Reward | Backbone / dependency | Score head / checkpoint | Manual download |
|--|--|--|--|
| `HPSReward` (`version="v2.1"`) | open_clip `ViT-H-14`, requires `pip install hpsv2` + the `factory.py` patch and BPE vocab | `HPS_v2.1_compressed.pt` (auto-downloaded) | No, override with `model_path` |
| `MPSReward` | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` (config/tokenizer auto-downloaded); MPS code is vendored under `videox_fun/reward/MPS/` | `MPS_overall.pth` (auto-downloaded) | No, override with `model_path` |
| `VideoAlignReward` | `Qwen/Qwen2-VL-2B-Instruct` | `KlingTeam/VideoReward` (contains `model_config.json` + weights) | **Yes**, `model_path` is required |
| `HPSv3Reward` | `Qwen/Qwen2-VL-7B-Instruct` | `HPSv3.safetensors` from `MizzenAI/HPSv3` | Recommended; auto-downloaded from HF when omitted |

**HPS v2.1 (`HPSReward`)** — besides `pip install hpsv2`, you must drop two special config files into `hpsv2`'s bundled open_clip: `create_model_and_transforms("ViT-H-14", ...)` uses arguments added by the HPS fork, so it needs the patched `factory.py`; `get_tokenizer("ViT-H-14")` needs the BPE vocab `bpe_simple_vocab_16e6.txt.gz`. Loading fails if either is missing. The `ViT-H-14` backbone and the `HPS_v2.1_compressed.pt` head download automatically to the torch hub cache on first run:

```bash
pip install hpsv2
# Patch hpsv2's bundled open_clip factory.py and add the BPE vocab file
site_packages=$(python -c "import site; print(site.getsitepackages()[0])")
wget -O $site_packages/hpsv2/src/open_clip/factory.py https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/package/patches/hpsv2_src_open_clip_factory_patches.py
wget -O $site_packages/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz https://github.com/tgxs002/HPSv2/raw/refs/heads/master/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz
```

```bash
  --reward_fn="HPSReward" \
  --reward_fn_kwargs='{"version": "v2.1"}'
```

**MPS (`MPSReward`)** — the MPS code is vendored under `videox_fun/reward/MPS/` and needs no patching; the `MPS_overall.pth` head downloads automatically from OSS, and the config + tokenizer of `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` download automatically from HuggingFace (if HF is unreachable, point `processor_name_or_path` to a local mirror):

```bash
  --reward_fn="MPSReward" \
  --reward_fn_kwargs='{"max_reward": 1, "loss_scale": 1}'
```

**VideoReward (`VideoAlignReward`)** — needs the Qwen2-VL-2B backbone + a VideoReward head directory (containing `model_config.json` and weights). Download both in advance; `model_path` is required:

```bash
modelscope download --model Qwen/Qwen2-VL-2B-Instruct --local_dir models/Personalized_Model/Qwen2-VL-2B-Instruct
huggingface-cli download KlingTeam/VideoReward --local-dir models/Personalized_Model/VideoReward
```

```bash
  --reward_fn="VideoAlignReward" \
  --reward_fn_kwargs='{"model_path": "models/Personalized_Model/VideoReward/", "model_name_or_path": "models/Personalized_Model/Qwen2-VL-2B-Instruct", "reward_dim": "TA+MQ", "fps": 16, "use_norm": true, "differentiable": true, "max_reward": 2.0}'
```

**HPSv3 (`HPSv3Reward`)** — needs the Qwen2-VL-7B backbone + `HPSv3.safetensors`. When `checkpoint_path` is omitted it downloads automatically from HF `MizzenAI/HPSv3`; staging it locally is recommended:

```bash
modelscope download --model Qwen/Qwen2-VL-7B-Instruct --local_dir models/Personalized_Model/Qwen2-VL-7B-Instruct
modelscope download --model MizzenAI/HPSv3 --local_dir models/Personalized_Model/HPSv3
```

```bash
  --reward_fn="HPSv3Reward" \
  --reward_fn_kwargs='{"checkpoint_path": "models/Personalized_Model/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Personalized_Model/Qwen2-VL-7B-Instruct", "differentiable": true, "max_reward": 12}'
```

> With `--backprop`, `VideoAlignReward` / `HPSv3Reward` must set `"differentiable": true`, otherwise their `@torch.no_grad()` path cuts the gradient chain (see [3.3](#33-reward-specific-parameter-explanation)).

### 2.3 Dataset Structure and metadata.json

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

**Relative Path Format** (example):
```json
[
  {
    "file_path": "train/video001.mp4",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
    "type": "video",
    "width": 1024,
    "height": 1024
  }
]
```

**Key Field Descriptions**:
- `file_path`: Video path (relative or absolute path).
- `text`: Video description (English prompt).
- `type`: Data type, fixed as `"video"`.
- `width` / `height`: Video width and height (**highly recommended**, used for bucket training). You can use `scripts/process_json_add_width_and_height.py` to extract these fields.

**Relative vs Absolute Path**:
- Relative: `export DATASET_NAME="datasets/X-Fun-Videos-Demo/"` and `export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"`.
- Absolute: `export DATASET_NAME=""` and `export DATASET_META_NAME="/mnt/data/metadata_add_width_height.json"`.

---

## 3. Reward LoRA Training

### 3.1 Download Pre-trained Base Model

```bash
# Create model directory
mkdir -p models/Diffusion_Transformer

# Download Wan2.2 official weights
# T2V model (text-to-video, dual-Transformer)
modelscope download --model Wan-AI/Wan2.2-T2V-A14B --local_dir models/Diffusion_Transformer/Wan2.2-T2V-A14B
# or I2V model (image-to-video, dual-Transformer)
# modelscope download --model Wan-AI/Wan2.2-I2V-A14B --local_dir models/Diffusion_Transformer/Wan2.2-I2V-A14B
# or TI2V model (text-image-to-video, single-Transformer)
# modelscope download --model Wan-AI/Wan2.2-TI2V-5B --local_dir models/Diffusion_Transformer/Wan2.2-TI2V-5B
```

### 3.2 Quick Start (FSDP)

Reward LoRA training keeps the full denoising rollout and the reward model's forward graph in memory, so FSDP (`FULL_SHARD`) is recommended. With `--boundary_type="high"`, the *other* (frozen) expert is already sharded and resident on GPU and is reused during validation, so no second copy is re-loaded from disk.

**Wan2.2 T2V Reward LoRA Training Example (single HPS v2.1 reward, FSDP)**:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" \
  --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" \
  --fsdp_cpu_ram_efficient_loading False \
  scripts/wan2.2/train_reward_lora.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_reward_lora" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --enable_bucket \
  --boundary_type="high" \
  --rank=128 \
  --network_alpha=64 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --train_mode="normal" \
  --guidance_scale=5.0 \
  --num_inference_steps=40 \
  --num_decoded_latents=3 \
  --reward_fn="HPSReward" \
  --reward_fn_kwargs='{"version": "v2.1"}' \
  --backprop \
  --backprop_strategy="tail" \
  --backprop_num_steps=1 \
  --low_vram
```

> The `train_reward_lora.sh` script in this directory provides a ready-to-run template. Adjust `MODEL_NAME`, dataset paths, `--reward_fn`, and `--boundary_type` to your setup.

### 3.3 Reward-specific Parameter Explanation

Besides the standard LoRA arguments (see [README_TRAIN_LORA.md](./README_TRAIN_LORA.md)), the reward trainer adds the following:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--reward_fn` | Reward model class name(s). A **comma-separated list** enables multi-reward training; the differentiable losses are combined by a weighted sum. Available: `AestheticReward`, `HPSReward`, `PickScoreReward`, `MPSReward`, `HPSv3Reward`, `VideoAlignReward`. | `"HPSv3Reward,VideoAlignReward"` |
| `--reward_fn_kwargs` | Keyword args for the reward model(s). A **flat JSON dict** for a single reward; a **nested JSON dict keyed by reward name** for multiple rewards. | `'{"VideoAlignReward": {"max_reward": 2.0}, "HPSv3Reward": {"max_reward": 12}}'` |
| `--multi_reward_weights` | JSON dict of per-reward loss weights. Normalized to sum to 1; if omitted with multiple rewards, equal weights are used. Ignored for a single reward. | `'{"VideoAlignReward": 0.5, "HPSv3Reward": 0.5}'` |
| `--num_inference_steps` | Number of denoising steps in the differentiable rollout and validation. | 40 |
| `--num_decoded_latents` | Number of latent frames decoded back to pixels for the reward. The chunked causal VAE limits this to `<= 3` (gradient-checkpointing constraint) unless `--use_1steps_vae` is set. | 3 |
| `--num_reward_frames` | Randomly select this many decoded frames for reward evaluation. `None` = use all. Only applied to **image** rewards (see below). | 2 |
| `--use_1steps_vae` | Use the single-pass Wan2.2 VAE (`AutoencoderKLWan3_8_1Steps`) for differentiable decoding; numerically equivalent, decodes the whole latent in one forward and lifts the `<= 3` latent limit. | - |
| `--backprop` | Enable the reward-backpropagation training mode. | - |
| `--backprop_strategy` | Which rollout steps keep a gradient graph: `last`, `tail`, `uniform`, `random`. | `tail` |
| `--backprop_num_steps` | Number of backprop steps, used when `--backprop_strategy` is `tail`/`uniform`/`random`. | 1 |
| `--first_frame_mse_weight` | Weight for the first-frame latent MSE auxiliary loss in inpaint mode. | 1.0 |

**Image vs Video rewards (`is_image_reward`)**:
- **Image rewards** (`AestheticReward`, `HPSReward`, `PickScoreReward`, `MPSReward`, `HPSv3Reward`) score individual frames, so frame subsampling (`--num_reward_frames`) is safe.
- **Video rewards** (`VideoAlignReward`, `is_image_reward=False`) need the full temporal sequence to judge motion quality; collapsing frames would degrade the video to a near-static input and break the motion dimension. For these, all decoded frames are kept (count bounded by `--num_decoded_latents`).
- In inpaint mode (`train_mode="inpaint"`), the first frame is dropped before reward evaluation for every reward.

> **IMPORTANT**: when using `--backprop`, a video reward such as `VideoAlignReward` / `HPSv3Reward` must be constructed with `"differentiable": true` inside `--reward_fn_kwargs`; otherwise its `@torch.no_grad()` path cuts the gradient chain entirely.

### 3.4 Single vs Multi-Reward Training

A single reward uses `--reward_fn="HPSv3Reward"` with a flat `--reward_fn_kwargs`. To combine several objectives, pass a comma-separated list, a nested kwargs dict, and (optionally) weights. Per-reward loss and reward values are logged separately (e.g. `train/loss_HPSv3Reward`, `train/reward_VideoAlignReward`), while the optimized `loss = Σ wᵢ · lossᵢ`.

**Multi-reward example (HPSv3 + VideoAlign, T2V)** — the strongest alignment setup, since the video reward supplies the temporal (motion / consistency) signal that image rewards cannot:

```bash
  --num_decoded_latents=13 \
  --num_reward_frames=2 \
  --use_1steps_vae \
  --reward_fn="VideoAlignReward,HPSv3Reward" \
  --reward_fn_kwargs='{"VideoAlignReward": {"model_path": "models/Personalized_Model/VideoReward/", "model_name_or_path": "models/Personalized_Model/Qwen2-VL-2B-Instruct", "fps": 16, "reward_dim": "TA+MQ", "differentiable": true, "use_norm": true, "max_reward": 2.0, "loss_scale": 1.0}, "HPSv3Reward": {"checkpoint_path": "models/Personalized_Model/HPSv3/HPSv3.safetensors", "model_name_or_path": "models/Personalized_Model/Qwen2-VL-7B-Instruct", "differentiable": true, "max_reward": 12}}' \
  --multi_reward_weights='{"VideoAlignReward": 0.5, "HPSv3Reward": 0.5}' \
  --backprop \
  --backprop_strategy="tail" \
  --backprop_num_steps=1
```

### 3.5 Differentiable Decoding and `--use_1steps_vae`

The default chunked causal VAE (`AutoencoderKLWan3_8`) applies gradient checkpointing per chunk, which limits differentiable decoding to the **first 3 latent frames** (`--num_decoded_latents <= 3`). Setting `--num_decoded_latents > 3` without `--use_1steps_vae` raises a `ValueError`.

Enable `--use_1steps_vae` to switch to the single-pass variant, which decodes the whole latent in one forward, applies checkpointing cleanly per decoder block, and lifts the limit — useful when a video reward needs more temporal context. It is numerically equivalent and only affects the `AutoencoderKLWan3_8` (2.2 VAE) path.

### 3.6 Training Validation

You can configure validation parameters to periodically generate test videos during training so you can monitor alignment progress. Under FSDP, `log_validation` runs a pipeline forward that is a cross-rank collective, so **all processes must enter the validation call** (it is intentionally *not* wrapped in `is_main_process`); the trainable expert already carries the LoRA, and the frozen expert is reused rather than re-loaded.

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--validation_steps` | Run validation every N steps | 2000 |
| `--validation_epochs` | Run validation every N epochs | 5 |
| `--validation_prompts` | Prompts for validating video generation | None |
| `--validation_paths` | Reference image paths for I2V validation (inpaint mode only) | None |

**Validation Example**:

```bash
  --validation_prompts="A stylish woman strolls down a bustling Tokyo street at night, warm neon glow and animated city signage reflecting on the wet pavement; she wears a black leather jacket over a red dress, confident casual demeanor, medium shot, cinematic." \
  --validation_steps=50 \
  --validation_epochs=500
```

> Validation videos are saved to the `output_dir` directory. For T2V validation (without a reference image), keep `train_mode="normal"` and omit `--validation_paths`.

---

## 4. Inference Testing

The trained reward LoRA is a normal LoRA and is loaded the same way as [README_TRAIN_LORA.md](./README_TRAIN_LORA.md). For an A14B dual-Transformer model, set `lora_path` / `lora_weight` for the low-noise reward LoRA and `lora_high_path` / `lora_high_weight` for the high-noise reward LoRA in [examples/wan2.2/predict_t2v.py](https://github.com/aigc-apps/VideoX-Fun/blob/main/examples/wan2.2/predict_t2v.py) (or `predict_i2v.py`):

```python
# Path to the trained reward LoRA weights, e.g.
# "output_dir_wan2.2_reward_lora/checkpoint-xxx/diffusion_pytorch_model.safetensors"
lora_path = None
lora_high_path = None
lora_weight = 0.55
lora_high_weight = 0.55
```

Run single-GPU inference with:

```bash
python examples/wan2.2/predict_t2v.py
```

For memory-management modes and multi-GPU parallel inference, refer to the Inference Testing section of [README_TRAIN_LORA.md](./README_TRAIN_LORA.md).

---

## 5. Limitations

1. After training to a certain extent, the reward keeps increasing but the quality of the generated videos no longer improves. The model trickily learns shortcuts (adding artifacts in the background, i.e. adversarial patches) to raise the reward.
2. There is still a lack of suitable preference models for video generation. Directly using image preference models cannot evaluate preferences along the temporal dimension (such as dynamism and consistency), and tends to reduce the dynamism of generated videos. This can be partially mitigated by computing the reward using only the first decoded frame, but the impact persists — which is why a video reward (e.g. `VideoAlignReward`) with `--use_1steps_vae` and a larger `--num_decoded_latents`, or a multi-reward mix, is recommended for motion-sensitive alignment.

---

## 6. More Resources

- **Official GitHub**: https://github.com/aigc-apps/VideoX-Fun
- **Standard LoRA training**: [README_TRAIN_LORA.md](./README_TRAIN_LORA.md)
- **Chinese version**: [README_TRAIN_REWARD_zh-CN.md](./README_TRAIN_REWARD_zh-CN.md)

---

## References
<ol>
  <li id="ref1">Clark, Kevin, et al. "Directly fine-tuning diffusion models on differentiable rewards.". In ICLR 2024.</li>
  <li id="ref2">Prabhudesai, Mihir, et al. "Aligning text-to-image diffusion models with reward backpropagation." arXiv preprint arXiv:2310.03739 (2023).</li>
</ol>
