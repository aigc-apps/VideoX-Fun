# Wan2.2 VACE Fun 训练指南

本文档提供 Wan2.2 VACE Fun（视频生成与编辑统一框架）训练的完整流程，包括环境配置、数据准备、多种分布式训练策略和推理测试。

> **说明**：Wan2.2 VACE Fun 是一个基于 Wan2.2 架构的视频生成与编辑统一模型，支持 I2V（图生视频）、S2V（主体参考视频生成）、V2V Control（可控视频生成）、V2V Mask（视频修复）等多种任务。Wan2.2 采用双 Transformer 架构（高噪声/低噪声模型）。本指南涵盖 Wan2.2 VACE Fun 的 VACE 模块训练流程，仅支持 A14B 模型变体。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径使用方案](#24-相对路径与绝对路径使用方案)
- [三、VACE 模块训练](#三vace-模块训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始（DeepSpeed-Zero-2）](#32-快速开始deepspeed-zero-2)
  - [3.3 VACE 专用参数解析](#33-vace-专用参数解析)
  - [3.4 训练验证](#34-训练验证)
  - [3.5 使用 FSDP 训练](#35-使用-fsdp-训练)
  - [3.6 其他后端](#36-其他后端)
  - [3.7 多机分布式训练](#37-多机分布式训练)
- [四、推理测试](#四推理测试)
  - [4.1 推理参数解析](#41-推理参数解析)
  - [4.2 VACE 视频生成推理](#42-vace-视频生成推理)
  - [4.3 多卡并行推理](#43-多卡并行推理)
- [五、更多资源](#五更多资源)

---

## 一、环境配置

**方式 1：使用requirements.txt**

```bash
pip install -r requirements.txt
```

**方式 2：手动安装依赖**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl deepspeed==0.17.0 numpy==1.26.4
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

**方式 3：使用docker**

使用docker的情况下，请保证机器中已经正确安装显卡驱动与CUDA环境，然后以此执行以下命令：

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 二、数据准备

### 2.1 快速测试数据集

我们提供了一个包含控制信号和主体参考图像的测试数据集，其中包含若干训练数据。

```bash
# 下载官方示例数据集（含控制信号 + 主体参考图像）
modelscope download --dataset PAI/X-Fun-Videos-Controls-Demo --local_dir ./datasets/X-Fun-Videos-Controls-Demo
```

下载后数据集包含以下 metadata 文件：
- `metadata.json`：基本格式（仅包含控制视频路径）
- `metadata_add_width_height.json`：含宽高信息（推荐用于 V2V Control 训练）
- `metadata_add_width_height_add_objects.json`：含宽高 + 主体参考图像（推荐用于 S2V 训练）

### 2.2 数据集结构

VACE 训练数据集除了原始视频外，还需要提供一一对应的控制信号视频（如 canny 边缘视频、姿态视频、深度视频等）。如需使用 S2V（主体参考视频生成）功能训练，还需提供主体参考图像。

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/                    # 原始训练视频
│   │   ├── 📄 00000000.mp4
│   │   ├── 📄 00000001.mp4
│   │   └── 📄 ...
│   ├── 📂 canny/                    # 控制信号视频（如 canny 边缘检测）
│   │   ├── 📄 00000000.mp4
│   │   ├── 📄 00000001.mp4
│   │   └── 📄 ...
│   ├── 📂 object/                   # 主体参考图像（可选，用于 S2V 训练）
│   │   ├── 📂 00000000/
│   │   │   └── 📄 0-0.jpg
│   │   ├── 📂 00000001/
│   │   │   └── 📄 1-0.jpg
│   │   └── 📂 ...
│   ├── 📄 metadata.json
│   └── 📄 metadata_add_width_height.json
```

> **说明**：
> - `train/` 目录存放原始视频
> - `canny/`（或 `pose/`、`depth/` 等）目录存放与原始视频一一对应的控制信号视频，文件名应与原始视频保持一致
> - `object/` 目录存放主体参考图像（可选），每个视频对应一个子目录，子目录中存放该视频的主体参考图像
> - 控制信号目录名可以自定义，只需在 `metadata.json` 的 `control_file_path` 中正确指向即可

### 2.3 metadata.json 格式

**基本格式**（仅包含控制视频）：
```json
[
  {
    "file_path": "train/00000000.mp4",
    "text": "A young woman gently turns her head to the right...",
    "type": "video",
    "control_file_path": "canny/00000000.mp4"
  },
  {
    "file_path": "train/00000001.mp4",
    "text": "A young woman parts her lips slightly...",
    "type": "video",
    "control_file_path": "canny/00000001.mp4"
  }
]
```

**包含宽高信息的格式**（推荐，对应 `metadata_add_width_height.json`）：
```json
[
  {
    "file_path": "train/00000000.mp4",
    "text": "A young woman gently turns her head to the right...",
    "type": "video",
    "control_file_path": "canny/00000000.mp4",
    "height": 480,
    "width": 832
  }
]
```

**包含主体参考图像的格式**（用于 S2V 训练，对应 `metadata_add_width_height_add_objects.json`）：
```json
[
  {
    "file_path": "train/00000000.mp4",
    "text": "A young woman gently turns her head to the right...",
    "type": "video",
    "control_file_path": "canny/00000000.mp4",
    "height": 480,
    "width": 832,
    "object_file_path": [
      "object/00000000/0-0.jpg"
    ]
  }
]
```

**绝对路径格式**：
```json
[
  {
    "file_path": "/mnt/data/train/00000000.mp4",
    "text": "A beautiful sunset over the ocean",
    "type": "video",
    "control_file_path": "/mnt/data/canny/00000000.mp4",
    "height": 480,
    "width": 832,
    "object_file_path": [
      "/mnt/data/object/00000000/0-0.jpg"
    ]
  }
]
```

**关键字段说明**：
- `file_path`：原始视频路径（相对或绝对路径）
- `text`：视频描述（英文提示词）
- `type`：数据类型，固定为 `"video"`
- `control_file_path`：控制信号视频路径（相对或绝对路径，**VACE 训练必需**）
- `object_file_path`：主体参考图像路径列表（可选，用于 S2V 主体参考训练）。每个元素为一张主体参考图像的路径，训练时会随机打乱顺序
- `width` / `height`：视频宽高（**最好提供**，用于分桶训练，如果不提供则自动在训练时读取，当数据存储在如oss这样的速度较慢的系统上时，可能会影响训练速度）。
  - 可以使用`scripts/process_json_add_width_and_height.py`文件对无width与height字段的json进行提取，支持处理图片与视频。
  - 使用方案为`python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Controls-Demo/metadata.json --output_file datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json`。

### 2.4 相对路径与绝对路径使用方案

**相对路径**：

如果数据的路径为相对路径，则在训练脚本中设置：

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
```

**绝对路径**：

如果数据的路径为绝对路径，则在训练脚本中设置：

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **建议**：如果数据集较小且存储在本地，推荐使用相对路径；如果数据集存储在外部存储（如 NAS、OSS）或多个机器共享存储，推荐使用绝对路径。

---

## 三、VACE 模块训练

### 3.1 下载预训练模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Wan2.2 VACE Fun 官方权重
modelscope download --model PAI/Wan2.2-VACE-Fun-A14B --local_dir models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B
```

### 3.2 快速开始（DeepSpeed-Zero-2）

如果按照 **2.1 快速测试数据集下载数据** 与 **3.1 下载预训练模型下载权重**后，直接复制快速开始的启动指令进行启动。

推荐使用 DeepSpeed-Zero-2 与 FSDP 方案进行训练。这里以 DeepSpeed-Zero-2 为例配置 shell 文件。

**Wan2.2 VACE Fun 训练示例（DeepSpeed-Zero-2）**：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.2_vace_fun/train.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_vace_fun" \
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
  --control_ref_image="random" \
  --boundary_type="low" \
  --trainable_modules "vace" \
  --low_vram
```

### 3.3 VACE 专用参数解析

**Wan2.2 双Transformer架构说明**：

Wan2.2采用了创新的双Transformer架构：
- **低噪声模型（Low Noise Model）**：负责处理低噪声阶段（接近最终输出）
- **高噪声模型（High Noise Model）**：负责处理高噪声阶段（初始生成阶段）
- **边界类型（boundary_type）**：
  - `low`：训练低噪声模型，高噪声模型使用预训练权重（推荐用于 VACE 模块训练）
  - `high`：训练高噪声模型，低噪声模型使用预训练权重
  - `full`：单模型训练（用于单Transformer模型）

**VACE 关键参数说明**：

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--config_path` | 配置文件路径 | `config/wan2.2/wan_civitai_t2v.yaml` |
| `--pretrained_model_name_or_path` | 预训练模型路径 | `models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B` |
| `--train_data_dir` | 训练数据目录 | `datasets/X-Fun-Videos-Controls-Demo/` |
| `--train_data_meta` | 训练数据元文件 | `datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json` |
| `--train_batch_size` | 每批次样本数 | 1 |
| `--image_sample_size` | 图像最大训练分辨率 | 640 |
| `--video_sample_size` | 视频最大训练分辨率 | 640 |
| `--token_sample_size` | Token 采样尺寸 | 640 |
| `--video_sample_stride` | 视频采样步幅 | 2 |
| `--video_sample_n_frames` | 视频采样帧数 | 81 |
| `--gradient_accumulation_steps` | 梯度累积步数（等效增大 batch） | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数 | 100 |
| `--checkpointing_steps` | 每 N 步保存 checkpoint | 50 |
| `--learning_rate` | 初始学习率（VACE 模块训练推荐值） | 2e-05 |
| `--lr_scheduler` | 学习率调度器：`linear`、`cosine`、`cosine_with_restarts`、`polynomial`、`constant`、`constant_with_warmup` | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--seed` | 随机种子（可复现训练） | 42 |
| `--output_dir` | 输出目录 | `output_dir_wan2.2_vace_fun` |
| `--gradient_checkpointing` | 激活重计算以节省显存 | - |
| `--mixed_precision` | 混合精度：`no`、`fp16`、`bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 3e-2 |
| `--adam_epsilon` | AdamW epsilon 值 | 1e-10 |
| `--vae_mini_batch` | VAE 编码时的迷你批次大小 | 1 |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--enable_bucket` | 启用分桶训练，不裁剪图片/视频，按分辨率分组训练 | - |
| `--random_hw_adapt` | 自动缩放图片/视频到 `[min_size, max_size]` 范围内的随机尺寸 | - |
| `--training_with_video_token_length` | 根据 token 长度训练，支持任意分辨率 | - |
| `--uniform_sampling` | 均匀采样 timestep（推荐启用） | - |
| `--low_vram` | 低显存模式，提高显存效率 | - |
| `--boundary_type` | Wan2.2双Transformer边界类型：`low`（训练低噪声模型）、`high`（训练高噪声模型）、`full`（训练单模型） | `low` |
| `--control_ref_image` | 参考图像来源：`first_frame`（首帧）、`random`（随机帧） | `random` |
| `--trainable_modules` | 可训练模块（`vace` 表示只训练 VACE 相关模块） | `"vace"` |
| `--trainable_modules_low_learning_rate` | 使用较低学习率训练的模块列表 | `[]` |
| `--resume_from_checkpoint` | 恢复训练路径，使用 `"latest"` 自动选择最新 checkpoint | None |
| `--validation_steps` | 每 N 步执行一次验证 | 2000 |
| `--validation_epochs` | 每 N 个epoch执行一次验证 | 5 |
| `--validation_prompts` | 验证视频生成的提示词 | `"一只棕色的狗摇着头..."` |
| `--validation_paths` | 验证使用的控制视频路径 | `"asset/pose.mp4"` |
| `--use_deepspeed` | 启用 DeepSpeed 分布式训练 | - |
| `--use_fsdp` | 启用 FSDP 分布式训练 | - |
| `--use_8bit_adam` | 使用 8-bit Adam 优化器节省显存 | - |
| `--use_came` | 使用 CAME 优化器 | - |
| `--multi_stream` | 使用 CUDA 多流提升性能 | - |
| `--snr_loss` | 使用 SNR 损失函数 | - |
| `--weighting_scheme` | Timestep 加权方案：`sigma_sqrt`、`logit_normal`、`mode`、`cosmap`、`none` | `none` |
| `--motion_sub_loss` | 启用运动子损失以提升时序一致性 | - |
| `--motion_sub_loss_ratio` | 运动子损失比例 | 0.25 |

**Sample Size 配置指南**：
- `video_sample_size` 表示视频的分辨率大小；当 `random_hw_adapt` 为 True 时，表示视频和图像分辨率的最小值。
- `image_sample_size` 表示图像的分辨率大小；当 `random_hw_adapt` 为 True 时，表示视频和图像分辨率的最大值。
- `token_sample_size` 表示当 `training_with_video_token_length` 为 True 时，最大 token 长度对应的分辨率。
- 由于配置可能产生混淆，**如果你不需要任意分辨率进行 finetuning**，建议将 `video_sample_size`、`image_sample_size` 和 `token_sample_size` 设置为相同的固定值，例如 **(320, 480, 512, 640, 960)**。
  - **全部设置为 320** 代表 **240P**。
  - **全部设置为 480** 代表 **320P**。
  - **全部设置为 640** 代表 **480P**。
  - **全部设置为 960** 代表 **720P**。

**Token Length 训练说明**：
- 当启用 `training_with_video_token_length` 时，模型根据 token 长度进行训练。
- 例如：512x512 分辨率、49 帧的视频，其 token 长度为 13,312，需要设置 `token_sample_size = 512`。
  - 在 512x512 分辨率下，视频帧数为 49 (~= 512 * 512 * 49 / 512 / 512)。
  - 在 768x768 分辨率下，视频帧数为 21 (~= 512 * 512 * 49 / 768 / 768)。
  - 在 1024x1024 分辨率下，视频帧数为 9 (~= 512 * 512 * 49 / 1024 / 1024)。
  - 这些分辨率与对应帧数的组合，使模型能够生成不同尺寸的视频。

### 3.4 训练验证

你可以配置验证参数，在训练过程中定期生成测试视频，以便监控训练进度和模型质量。

**验证参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--validation_steps` | 每 N 步执行一次验证 | 2000 |
| `--validation_epochs` | 每 N 个epoch执行一次验证 | 5 |
| `--validation_prompts` | 验证视频生成的提示词 | None |
| `--validation_paths` | 验证使用的控制视频路径 | None |

**验证示例**：

```bash
  --validation_paths "asset/pose.mp4" \
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="In this sunlit outdoor garden, a beautiful woman wears a knee-length white sleeveless dress, its hem swaying gently with her graceful movements like a dancing butterfly. Sunlight filters through the leaves, casting dappled shadows that highlight her soft features and clear eyes, enhancing her elegance. Every motion seems to speak of youth and vitality as she spins on the grass, her skirt fluttering around her, as if the entire garden rejoices in her dance. Colorful flowers all around—roses, chrysanthemums, lilies—sway in the breeze, releasing their fragrances and creating a relaxed and joyful atmosphere."
```

**注意事项**：
- 验证视频会保存到 `output_dir` 目录中
- 多提示词验证格式：`--validation_prompts "prompt1" "prompt2" "prompt3"`
- Wan2.2 VACE Fun 的验证会根据 `boundary_type` 自动选择使用单Transformer或双Transformer
- `validation_paths` 需要与 `validation_prompts` 一一对应，指向控制视频文件

### 3.5 使用 FSDP 训练

**如果使用多卡且使用DeepSpeed-Zero-2的情况下显存不足**，可以切换使用FSDP进行训练。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=VaceWanAttentionBlock,BaseWanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.2_vace_fun/train.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_vace_fun" \
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
  --control_ref_image="random" \
  --boundary_type="low" \
  --trainable_modules "vace" \
  --low_vram
```

> **说明**：在本仓库中，FSDP 比 DeepSpeed-Zero-3 更稳定且出错更少。当 DeepSpeed-Zero-2 在多GPU情况下遇到显存问题时，请使用 FSDP。

### 3.6 其他后端

#### 3.6.1 使用DeepSpeed-Zero-3进行训练

目前不太推荐使用 DeepSpeed Zero-3。在本仓库中，使用 FSDP 出错更少且更稳定。

DeepSpeed Zero-3 适合高分辨率的 14B Wan。训练后，您可以使用以下命令获取最终模型：
```bash
python scripts/zero_to_bf16.py output_dir/checkpoint-{our-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

训练 shell 命令如下：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/wan2.2_vace_fun/train.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_vace_fun" \
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
  --control_ref_image="random" \
  --boundary_type="low" \
  --trainable_modules "vace" \
  --low_vram
```

#### 3.6.2 不使用 DeepSpeed 与 FSDP 训练

**该方案并不被推荐，因为没有显存节约后端，容易造成显存不足**。这里仅提供训练Shell用于参考训练。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.2_vace_fun/train.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_vace_fun" \
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
  --control_ref_image="random" \
  --boundary_type="low" \
  --trainable_modules "vace" \
  --low_vram
```

> **说明**：这与 `train.sh` 脚本类似，但使用了正确的数据集路径。`train.sh` 脚本可以用作单GPU训练的起点。

### 3.7 多机分布式训练

**适合场景**：超大规模数据集、需要更快的训练速度

#### 3.7.1 环境配置

假设有 2 台机器，每台 8 张 GPU：

**机器 0（Master）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master 机器 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 机器总数
export NUM_PROCESS=16                # 总进程数 = 机器数 x 8
export RANK=0                        # 当前机器 rank（0 或 1）
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.2_vace_fun/train.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_wan2.2_vace_fun" \
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
  --control_ref_image="random" \
  --boundary_type="low" \
  --trainable_modules "vace" \
  --low_vram
```

**机器 1（Worker）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # 与 Master 相同
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # 注意这里是 1
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

# 使用与机器 0 相同的 accelerate launch 命令
```

#### 3.7.2 多机训练注意事项

- **网络要求**：
   - 推荐 RDMA/InfiniBand（高性能）
   - 无 RDMA 时添加环境变量：
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **数据同步**：所有机器必须能够访问相同的数据路径（NFS/共享存储）

---

## 四、推理测试

### 4.1 推理参数解析

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `GPU_memory_mode` | 显存管理模式，可选值见下表 | `model_group_offload` |
| `ulysses_degree` | Head 维度并行度，单卡时为 1 | 1 |
| `ring_degree` | Sequence 维度并行度，单卡时为 1 | 1 |
| `fsdp_dit` | 多卡推理时对 Transformer 使用 FSDP 节省显存 | `False` |
| `fsdp_text_encoder` | 多卡推理时对文本编码器使用 FSDP | `True` |
| `compile_dit` | 编译 Transformer 加速推理（固定分辨率下有效） | `False` |
| `model_name` | 模型路径 | `models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B` |
| `sampler_name` | 采样器类型：`Flow`、`Flow_Unipc`、`Flow_DPM++` | `Flow` |
| `transformer_path` | 加载训练好的低噪声 Transformer 权重路径 | `None` |
| `transformer_high_path` | 加载训练好的高噪声 Transformer 权重路径（仅双Transformer模型） | `None` |
| `vae_path` | 加载训练好的 VAE 权重路径 | `None` |
| `sample_size` | 生成视频分辨率 `[高度, 宽度]` | `[480, 832]` 或 `[832, 480]` |
| `video_length` | 生成视频帧数 | `81` |
| `fps` | 每秒帧数 | `16` |
| `weight_dtype` | 模型权重精度，不支持 bf16 的显卡使用 `torch.float16` | `torch.bfloat16` |
| `control_video` | 控制信号视频路径（如姿态视频，V2V Control 任务） | `"asset/pose.mp4"` |
| `start_image` | 起始帧图像路径（I2V 任务） | `"asset/1.png"` |
| `end_image` | 结束帧图像路径（可选） | `None` |
| `inpaint_video` | 待修复视频路径（V2V Mask 任务） | `"asset/inpaint_video.mp4"` |
| `inpaint_video_mask` | 修复掩码视频路径（V2V Mask 任务） | `"asset/inpaint_video_mask.mp4"` |
| `subject_ref_images` | 主体参考图像路径列表（S2V / V2V Control+Ref 任务） | `["asset/8.png", "asset/ref_1.png"]` |
| `vace_context_scale` | VACE 上下文缩放因子 | `1.00` |
| `prompt` | 正向提示词，描述生成内容 | `"一位年轻女子站在阳光明媚的海岸线上..."` |
| `negative_prompt` | 负向提示词，避免生成的内容 | `"色调艳丽，过曝，静态..."` |
| `guidance_scale` | 引导强度 | 5.0 |
| `seed` | 随机种子，用于复现结果 | 43 |
| `num_inference_steps` | 推理步数 | 50 |
| `save_path` | 生成视频保存路径 | `samples/vace-videos-fun` |

**显存管理模式说明**：

| 模式 | 说明 | 显存占用 |
|------|------|---------|
| `model_full_load` | 整个模型加载到 GPU | 最高 |
| `model_full_load_and_qfloat8` | 全量加载 + FP8 量化 | 高 |
| `model_cpu_offload` | 使用后将模型卸载到 CPU | 中等 |
| `model_cpu_offload_and_qfloat8` | CPU 卸载 + FP8 量化 | 中低 |
| `model_group_offload` | 层组在 CPU/CUDA 间切换 | 低 |
| `sequential_cpu_offload` | 逐层卸载（速度最慢） | 最低 |

### 4.2 VACE 视频生成推理

#### 4.2.1 推理脚本选择

Wan2.2 VACE Fun 提供多种推理脚本，请根据您的任务类型选择：

| 脚本 | 主要用途 | 关键输入 |
|------|---------|---------|
| `predict_i2v.py` | I2V（图生视频） | `start_image` |
| `predict_s2v.py` | S2V（主体参考视频生成） | `subject_ref_images` |
| `predict_v2v_control.py` | V2V Control（可控视频生成） | `control_video` |
| `predict_v2v_control_ref.py` | V2V Control + Ref（可控视频生成 + 参考图像） | `control_video` + `subject_ref_images` |
| `predict_v2v_mask.py` | V2V Mask（视频修复） | `inpaint_video` + `inpaint_video_mask` |

> **说明**：
> - A14B 模型使用双Transformer架构（低噪声+高噪声模型），需要配置 `transformer_path` 和 `transformer_high_path`
> - `predict_v2v_control_ref.py` 支持 Control + 参考图像，生成效果通常更好

#### 4.2.2 I2V 推理（图生视频）

单卡推理运行如下命令：

```bash
python examples/wan2.2_vace_fun/predict_i2v.py
```

根据需求修改 `examples/wan2.2_vace_fun/predict_i2v.py`，初次推理重点关注如下参数，如果对其他参数感兴趣，请查看上方的推理参数解析。

```python
# 根据显卡显存选择
GPU_memory_mode = "sequential_cpu_offload"
# 根据实际模型路径
model_name = "models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
# 训练好的低噪声权重路径，如 "output_dir_wan2.2_vace_fun/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = None
# 训练好的高噪声权重路径
transformer_high_path = None
# I2V 起始图像
start_image = "asset/1.png"
# 生成视频分辨率 [高度, 宽度]
sample_size = [480, 832]
# 根据生成内容编写
prompt = "一只棕色的狗舔了一下它的舌头，坐在舒适房间里的浅色沙发上..."
# ...
```

> **说明**：Wan2.2 VACE Fun 的 I2V 推理需要提供 `start_image` 起始图像，模型将根据图像生成视频。

#### 4.2.3 S2V 推理（主体参考视频生成）

```bash
python examples/wan2.2_vace_fun/predict_s2v.py
```

```python
# 根据显卡显存选择
GPU_memory_mode = "sequential_cpu_offload"
# 根据实际模型路径
model_name = "models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
# 训练好的低噪声权重路径
transformer_path = None
# 训练好的高噪声权重路径
transformer_high_path = None
# 主体参考图像列表
subject_ref_images = ["asset/8.png", "asset/ref_1.png"]
# 生成视频分辨率 [高度, 宽度]
sample_size = [480, 832]
# 根据生成内容编写
prompt = "海风作曲，浪花打拍。她握着一台亮黄色相机..."
# ...
```

> **说明**：S2V 推理通过 `subject_ref_images` 提供主体参考图像，模型将根据参考图像生成保持主体一致性的视频。`vace_context_scale` 控制主体参考的强度。

#### 4.2.4 V2V Control 推理（可控视频生成）

单卡推理运行如下命令：

```bash
python examples/wan2.2_vace_fun/predict_v2v_control.py
```

根据需求修改 `examples/wan2.2_vace_fun/predict_v2v_control.py`，重点关注如下参数：

```python
# 根据显卡显存选择
GPU_memory_mode = "sequential_cpu_offload"
# 根据实际模型路径
model_name = "models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
# 训练好的低噪声权重路径
transformer_path = None
# 训练好的高噪声权重路径
transformer_high_path = None
# 控制信号视频（如姿态视频）
control_video = "asset/pose.mp4"
# 生成视频分辨率 [高度, 宽度]
sample_size = [832, 480]
# 无参考图像
subject_ref_images = None
# 根据生成内容编写
prompt = "一位年轻女子站在阳光明媚的海岸线上，身穿清爽的白色衬衫与裙子..."
# ...
```

#### 4.2.5 V2V Control + Ref 推理（可控视频生成 + 参考图像）

```bash
python examples/wan2.2_vace_fun/predict_v2v_control_ref.py
```

```python
# 根据显卡显存选择
GPU_memory_mode = "sequential_cpu_offload"
# 根据实际模型路径
model_name = "models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
# 训练好的低噪声权重路径
transformer_path = None
# 训练好的高噪声权重路径
transformer_high_path = None
# 控制信号视频
control_video = "asset/pose.mp4"
# 参考图像路径
subject_ref_images = ["asset/8.png"]
# 生成视频分辨率 [高度, 宽度]
sample_size = [832, 480]
# 根据生成内容编写
prompt = "一位年轻女子站在阳光明媚的海岸线上，身穿清爽的白色衬衫与裙子..."
# ...
```

> **说明**：V2V Control + Ref 同时利用控制信号视频和参考图像，实现更精准的可控视频生成。

#### 4.2.6 V2V Mask 推理（视频修复）

```bash
python examples/wan2.2_vace_fun/predict_v2v_mask.py
```

```python
# 根据显卡显存选择
GPU_memory_mode = "sequential_cpu_offload"
# 根据实际模型路径
model_name = "models/Diffusion_Transformer/Wan2.2-VACE-Fun-A14B"
# 训练好的低噪声权重路径
transformer_path = None
# 训练好的高噪声权重路径
transformer_high_path = None
# 待修复视频
inpaint_video = "asset/inpaint_video.mp4"
# 修复掩码视频
inpaint_video_mask = "asset/inpaint_video_mask.mp4"
# 生成视频分辨率 [高度, 宽度]
sample_size = [480, 832]
# 无控制视频
control_video = None
# 无参考图像
subject_ref_images = None
# 根据生成内容编写
prompt = "一只棕色的兔子舔了一下它的舌头，坐在舒适房间里的浅色沙发上..."
# ...
```

> **说明**：V2V Mask 推理通过 `inpaint_video` 和 `inpaint_video_mask` 实现视频局部修复。掩码视频中白色区域表示需要修复的部分。

### 4.3 多卡并行推理

**适合场景**：高分辨率生成、加速推理

#### 安装并行推理依赖

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### 配置并行策略

编辑任意推理脚本（如 `examples/wan2.2_vace_fun/predict_v2v_control_ref.py`）：

```python
# 确保 ulysses_degree x ring_degree = 使用的 GPU 数
# 例如使用 2 张 GPU：
ulysses_degree = 2  # Head 维度并行
ring_degree = 1     # Sequence 维度并行
```

**配置原则**：
- `ulysses_degree` 必须能整除模型的 head 数
- `ring_degree` 是在 sequence 维度切分，会影响通信开销，在 head 能整除的情况下尽量不要用

**配置示例**：

| GPU 数量 | ulysses_degree | ring_degree | 说明 |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | 单 GPU |
| 4 | 4 | 1 | Head 并行 |
| 8 | 8 | 1 | Head 并行 |
| 8 | 4 | 2 | 混合并行 |

#### 运行多卡推理

```bash
torchrun --nproc-per-node=2 examples/wan2.2_vace_fun/predict_v2v_control_ref.py
```

---

## 五、更多资源

- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
