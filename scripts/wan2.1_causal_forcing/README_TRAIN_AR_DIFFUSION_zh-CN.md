# Wan2.1 Causal-Forcing 第一阶段：自回归扩散（AR Diffusion）训练指南

本文档介绍 **Causal-Forcing 第一阶段 — 自回归扩散（AR Diffusion）训练** 的完整流程。

> **什么是 Causal-Forcing？**
>
> Causal-Forcing 是一个三阶段流水线，将视频扩散模型压缩为 **少步因果自回归生成器**：
>
> 1. **第一阶段 — AR Diffusion**（`train_ar_diffusion.py`）：在干净视频 latent 上使用 **teacher forcing** 训练，让模型学会按块（或按帧）因果自回归地去噪，得到一个强大的 AR 骨架。
> 2. **第二阶段 — 因果一致性蒸馏**（`train_causal_consistency_distill.py`）：将多步 AR 模型蒸馏为每块一步的一致性模型。
> 3. **第三阶段 — 因果 DMD**（`train_causal_dmd.py`）：利用 14B 教师模型做分布匹配，进一步蒸馏为 **2 步** 生成器。
>
> 本文档仅覆盖 **第一阶段**。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、下载预训练模型](#二下载预训练模型)
- [三、准备训练数据](#三准备训练数据)
  - [3.1 快速测试数据集](#31-快速测试数据集)
  - [3.2 数据集结构](#32-数据集结构)
  - [3.3 metadata.json 格式](#33-metadatajson-格式)
- [四、训练](#四训练)
  - [4.1 快速开始](#41-快速开始)
  - [4.2 主要参数说明](#42-主要参数说明)
  - [4.3 Causal-Forcing 特有参数](#43-causal-forcing-特有参数)
  - [4.4 使用 FSDP 训练](#44-使用-fsdp-训练)
- [五、使用训练好的 Checkpoint](#五使用训练好的-checkpoint)
- [六、更多资源](#六更多资源)

---

## 一、环境配置

**方式 1：使用 requirements.txt**

```bash
pip install -r requirements.txt
```

**方式 2：手动安装依赖**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

**方式 3：使用 docker**

```bash
# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# 进入容器
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 二、下载预训练模型

第一阶段以 Wan2.1-T2V-1.3B 为基础模型，在其上进行因果 teacher forcing 训练。

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Wan2.1 T2V 基础模型
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B
```

---

## 三、准备训练数据

第一阶段直接在 **原始视频** 上训练，VAE 在线编码 — **不需要** ODE 轨迹对或预计算 latent。

### 3.1 快速测试数据集

我们提供了一个小型 demo 数据集用于快速测试：

```bash
# 下载官方示例数据集
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 3.2 数据集结构

```
📦 datasets/
├── 📂 X-Fun-Videos-Demo/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 3.3 metadata.json 格式

**相对路径**（推荐）：
```json
[
  {
    "file_path": "train/video001.mp4",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
    "type": "video"
  },
  {
    "file_path": "train/video002.mp4",
    "text": "A person walking through a forest, cinematic view",
    "type": "video"
  }
]
```

**绝对路径**：
```json
[
  {
    "file_path": "/mnt/data/videos/sunset.mp4",
    "text": "A beautiful sunset over the ocean",
    "type": "video"
  }
]
```

---

## 四、训练

### 4.1 快速开始

直接执行启动脚本 [train_ar_diffusion.sh](./train_ar_diffusion.sh)：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/wan2.1_causal_forcing/train_ar_diffusion.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --fix_sample_size 480 832 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=200 \
  --learning_rate=2e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_causal_forcing_ar_diffusion" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --num_frame_per_block=3 \
  --train_sampling_steps=1000 \
  --shift=5.0 \
  --use_timestep_weight \
  --trainable_modules "."
```

或者直接执行：

```bash
bash scripts/wan2.1_causal_forcing/train_ar_diffusion.sh
```

### 4.2 主要参数说明

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `--pretrained_model_name_or_path` | 初始化基础模型 | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B/` |
| `--config_path` | 模型配置 YAML | `config/wan2.1/wan_civitai.yaml` |
| `--train_data_dir` | 视频根目录（拼接到 `file_path` 之前） | `""` |
| `--train_data_meta` | `metadata.json` 路径 | `datasets/X-Fun-Videos-Demo/metadata.json` |
| `--train_batch_size` | 每卡 batch size | 1 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数 | 100 |
| `--checkpointing_steps` | 每 N 步保存 checkpoint | 200 |
| `--learning_rate` | 初始学习率 | 2e-06 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--output_dir` | 输出目录 | `output_dir_wan2.1_causal_forcing_ar_diffusion` |
| `--gradient_checkpointing` | 启用激活重计算 | - |
| `--mixed_precision` | `fp16` / `bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 3e-2 |
| `--adam_epsilon` | AdamW epsilon | 1e-10 |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--trainable_modules` | 可训练模块（`"."` 表示全量） | `"."` |

**视频采样参数**：

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `--image_sample_size` | 图像采样尺寸 | 640 |
| `--video_sample_size` | 视频采样尺寸 | 640 |
| `--token_sample_size` | Token 采样尺寸 | 640 |
| `--fix_sample_size` | 固定输出 `[高度, 宽度]` | `480 832` |
| `--video_sample_stride` | 帧采样步长 | 2 |
| `--video_sample_n_frames` | 视频帧数 | 81 |
| `--random_hw_adapt` | 启用随机分辨率自适应 | - |
| `--training_with_video_token_length` | 启用基于 token 长度的训练 | - |
| `--enable_bucket` | 启用宽高比 bucket 采样 | - |
| `--vae_mini_batch` | VAE 编码 mini-batch 大小（设为 1 可避免显存溢出） | 1 |

### 4.3 Causal-Forcing 特有参数

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `--num_frame_per_block` | 每个因果块的帧数。`3` = chunkwise，`1` = framewise | 3 |
| `--independent_first_frame` | 第一帧独立（`[1, N, N, ...]` 块模式，I2V 场景适用） | - |
| `--shift` | `FlowMatchEulerDiscreteScheduler` 的 shift 值（Causal-Forcing 默认 5.0） | 5.0 |
| `--train_sampling_steps` | flow matching 调度器总时间步数 | 1000 |
| `--use_timestep_weight` | 启用逐时间步高斯损失权重（以 T/2 为中心） | - |
| `--no_teacher_forcing` | 关闭 teacher forcing（改为 diffusion forcing） | - |
| `--noise_augmentation_max_timestep` | teacher forcing 时向干净上下文 token 添加轻微噪声（0 = 关闭） | 0 |

> **Teacher Forcing 说明**：默认情况下，第一阶段使用 teacher forcing — 模型在预测当前块时接收 **干净的 GT latent** 作为上下文（`clean_x`）。这可以稳定早期训练。如需使用 diffusion forcing，可通过 `--no_teacher_forcing` 关闭。

### 4.4 使用 FSDP 训练

上述脚本已配置 FSDP，使用 `CasualWanAttentionBlock` 自动包装。单卡训练无需 FSDP：

```bash
accelerate launch --mixed_precision="bf16" \
    scripts/wan2.1_causal_forcing/train_ar_diffusion.py \
    ...
```

---

## 五、使用训练好的 Checkpoint

第一阶段输出的 checkpoint 用于初始化 **第二阶段（因果一致性蒸馏）**。在 `train_causal_consistency_distill.py` 中通过 `--transformer_path` 和 `--teacher_transformer_path` 指定：

```bash
--transformer_path="output_dir_wan2.1_causal_forcing_ar_diffusion/checkpoint-{N}/diffusion_pytorch_model.safetensors" \
--teacher_transformer_path="output_dir_wan2.1_causal_forcing_ar_diffusion/checkpoint-{N}/diffusion_pytorch_model.safetensors"
```

完整的第二阶段流程参见 [train_causal_consistency_distill.sh](./train_causal_consistency_distill.sh)。

---

## 六、更多资源

- **Causal-Forcing 论文**：https://github.com/thu-ml/Causal-Forcing
- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
