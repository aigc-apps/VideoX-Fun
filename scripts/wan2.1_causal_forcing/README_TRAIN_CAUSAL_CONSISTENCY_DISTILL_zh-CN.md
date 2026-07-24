# Wan2.1 Causal-Forcing 第二阶段：因果一致性蒸馏（CCD）训练指南

本文档介绍 **Causal-Forcing 第二阶段 — 因果一致性蒸馏（CCD）** 的完整流程。

> **什么是因果一致性蒸馏？**
>
> CCD 是 Causal-Forcing 流水线的 **第二阶段**，将视频扩散模型压缩为 **少步因果自回归生成器**：
>
> 1. **第一阶段 — AR Diffusion**（`train_ar_diffusion.py`）：在干净视频 latent 上使用 teacher forcing 训练，得到强大的 AR 骨架。
> 2. **第二阶段 — 因果一致性蒸馏**（`train_causal_consistency_distill.py`）：利用 EMA teacher 和 CFG 引导，将多步 AR 模型蒸馏为 **每块一步** 的一致性模型。
> 3. **第三阶段 — 因果 DMD**（`train_causal_dmd.py`）：利用 14B 教师模型做分布匹配，进一步蒸馏为 **2 步** 生成器。
>
> 本文档仅覆盖 **第二阶段**。第一阶段参见 [README_TRAIN_AR_DIFFUSION_zh-CN.md](./README_TRAIN_AR_DIFFUSION_zh-CN.md)。

---

## 目录
- [一、前置条件](#一前置条件)
- [二、环境配置](#二环境配置)
- [三、下载预训练模型](#三下载预训练模型)
- [四、准备训练数据](#四准备训练数据)
  - [4.1 快速测试数据集](#41-快速测试数据集)
  - [4.2 数据集结构](#42-数据集结构)
  - [4.3 metadata.json 格式](#43-metadatajson-格式)
- [五、训练](#五训练)
  - [5.1 快速开始](#51-快速开始)
  - [5.2 主要参数说明](#52-主要参数说明)
  - [5.3 CCD 特有参数](#53-ccd-特有参数)
- [六、使用训练好的 Checkpoint](#六使用训练好的-checkpoint)
- [七、更多资源](#七更多资源)

---

## 一、前置条件

第二阶段需要 **第一阶段 AR Diffusion 的 checkpoint** 来初始化 generator 和 EMA teacher。

```bash
# 示例：第一阶段 AR Diffusion 训练输出的 checkpoint
export STAGE1_CKPT="output_dir_wan2.1_causal_forcing_ar_diffusion/checkpoint-8000/diffusion_pytorch_model.safetensors"
```

第一阶段的训练方法参见 [README_TRAIN_AR_DIFFUSION_zh-CN.md](./README_TRAIN_AR_DIFFUSION_zh-CN.md)。

---

## 二、环境配置

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

## 三、下载预训练模型

CCD 以 Wan2.1-T2V-1.3B 初始化 generator 和 teacher，然后加载第一阶段 checkpoint。

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Wan2.1 T2V 基础模型
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B
```

---

## 四、准备训练数据

CCD 直接在 **原始视频** 上训练，VAE 在线编码 — 数据格式与第一阶段相同。

### 4.1 快速测试数据集

我们提供了一个小型 demo 数据集用于快速测试：

```bash
# 下载官方示例数据集
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 4.2 数据集结构

```
📦 datasets/
├── 📂 X-Fun-Videos-Demo/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 4.3 metadata.json 格式

**相对路径**（推荐）：
```json
[
  {
    "file_path": "train/video001.mp4",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
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

## 五、训练

### 5.1 快速开始

直接执行启动脚本 [train_causal_consistency_distill.sh](./train_causal_consistency_distill.sh)：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
export STAGE1_CKPT="output_dir_wan2.1_causal_forcing_ar_diffusion/checkpoint-8000/diffusion_pytorch_model.safetensors"

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/wan2.1_causal_forcing/train_causal_consistency_distill.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --transformer_path=$STAGE1_CKPT \
  --teacher_transformer_path=$STAGE1_CKPT \
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
  --learning_rate=2.0e-06 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_causal_forcing_ccd" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=0.0 \
  --adam_beta1=0.0 \
  --adam_beta2=0.999 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=10.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --num_frame_per_block=3 \
  --shift=5.0 \
  --discrete_cd_N=48 \
  --guidance_scale=3.0 \
  --ema_weight=0.99 \
  --ema_start_step=200 \
  --trainable_modules "."
```

或者直接执行：

```bash
bash scripts/wan2.1_causal_forcing/train_causal_consistency_distill.sh
```

### 5.2 主要参数说明

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `--pretrained_model_name_or_path` | 初始化基础模型 | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B/` |
| `--config_path` | 模型配置 YAML | `config/wan2.1/wan_civitai.yaml` |
| `--train_data_dir` | 视频根目录（拼接到 `file_path` 之前） | `""` |
| `--train_data_meta` | `metadata.json` 路径 | `datasets/X-Fun-Videos-Demo/metadata.json` |
| `--transformer_path` | 第一阶段 checkpoint（generator 初始化） | `$STAGE1_CKPT` |
| `--teacher_transformer_path` | 第一阶段 checkpoint（teacher 初始化，默认同 generator） | `$STAGE1_CKPT` |
| `--train_batch_size` | 每卡 batch size | 1 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数 | 100 |
| `--checkpointing_steps` | 每 N 步保存 checkpoint | 200 |
| `--learning_rate` | 初始学习率 | 2e-06 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--output_dir` | 输出目录 | `output_dir_wan2.1_causal_forcing_ccd` |
| `--gradient_checkpointing` | 启用激活重计算 | - |
| `--mixed_precision` | `fp16` / `bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 0.0 |
| `--adam_beta1` | AdamW beta1（CCD 使用 0.0） | 0.0 |
| `--adam_beta2` | AdamW beta2 | 0.999 |
| `--adam_epsilon` | AdamW epsilon | 1e-10 |
| `--max_grad_norm` | 梯度裁剪阈值 | 10.0 |
| `--trainable_modules` | 可训练模块（`"."` 表示全量） | `"."` |
| `--low_vram` | 启用低显存模式（VAE/文本编码器分时加载） | - |

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

### 5.3 CCD 特有参数

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `--num_frame_per_block` | 每个因果块的帧数。`3` = chunkwise，`1` = framewise | 3 |
| `--independent_first_frame` | 第一帧独立（`[1, N, N, ...]` 块模式，I2V 场景适用） | - |
| `--shift` | `FlowMatchEulerDiscreteScheduler` 的 shift 值（Causal-Forcing 默认 5.0） | 5.0 |
| `--discrete_cd_N` | 一致性调度器的离散时间步数量 | 48 |
| `--guidance_scale` | EMA teacher 使用的 CFG 引导强度 | 3.0 |
| `--ema_weight` | EMA 衰减率（一致性目标 generator 副本，设为 <=0 禁用） | 0.99 |
| `--ema_start_step` | EMA 跟踪启动前的等待步数 | 200 |

> **CCD 工作原理**：对每个训练样本，CCD 执行以下步骤：
> 1. 加载干净视频 latent（通过 VAE 在线编码）。
> 2. 从离散一致性调度 `[0, N-2]` 中采样一个时间步索引。
> 3. 在时间步 `t` 和 `t_next` 分别给干净 latent 加噪。
> 4. **Generator** 在 `x_t` 上预测 `x0`。
> 5. **EMA teacher**（带 CFG 引导）在 `x_{t_next}` 上生成一致性目标。
> 6. 最小化 generator 预测与 teacher 目标之间的 L2 损失。

> **EMA Teacher**：EMA 副本通过 polyak 更新跟踪 generator（`ema = decay*ema + (1-decay)*gen`）。在 `--ema_start_step` 之前，EMA 直接镜像 generator。Teacher 使用 `--guidance_scale=3.0` 的 CFG 来生成更高质量的目标。

---

## 六、使用训练好的 Checkpoint

第二阶段 CCD 输出的 checkpoint 用于初始化 **第三阶段（因果 DMD）**。在 `train_causal_dmd.py` 中指定路径：

```bash
--ode_transformer_path="output_dir_wan2.1_causal_forcing_ccd/checkpoint-{N}/diffusion_pytorch_model.safetensors"
```

完整的第三阶段流程参见 `train_causal_dmd.sh`。

---

## 七、更多资源

- **Causal-Forcing 论文**：https://github.com/thu-ml/Causal-Forcing
- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
