# FantasyTalking-S2V 全参数训练指南

本文档提供 FantasyTalking-S2V（音频驱动的数字人视频生成模型）全参数训练的完整工作流，包括环境配置、数据准备、分布式训练和推理测试。

> **注意**：FantasyTalking 是一个音频驱动的数字人视频生成模型，需要同时提供参考图像和音频文件来生成说话视频。训练数据需要包含视频、音频和参考图像。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径使用](#24-相对路径与绝对路径使用)
- [三、全参数训练](#三全参数训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始（DeepSpeed-Zero-2）](#32-快速开始deepspeed-zero-2)
  - [3.3 常用训练参数](#33-常用训练参数)
  - [3.4 训练验证](#34-训练验证)
  - [3.5 使用 FSDP 训练](#35-使用-fsdp-训练)
  - [3.6 不使用 DeepSpeed 或 FSDP 训练](#36-不使用-deepspeed-或-fsdp-训练)
  - [3.7 多机分布式训练](#37-多机分布式训练)
- [四、推理测试](#四推理测试)
  - [4.1 推理参数](#41-推理参数)
  - [4.2 单 GPU 推理](#42-单-gpu-推理)
  - [4.3 多 GPU 并行推理](#43-多-gpu-并行推理)
- [五、更多资源](#五更多资源)

---

## 一、环境配置

**方式一：使用 requirements.txt**

```bash
pip install -r requirements.txt
```

**方式二：手动安装依赖**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl deepspeed==0.17.0 numpy==1.26.4
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

**方式三：使用 Docker**

使用 Docker 时，请先确保本机已正确安装 GPU 驱动和 CUDA 环境，然后执行以下命令：

```bash
# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# 进入镜像
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 二、数据准备

### 2.1 快速测试数据集

我们提供了一个包含数个音视频训练样本的测试数据集。

```bash
# 下载官方示例数据集
modelscope download --dataset PAI/X-Fun-Videos-Audios-Demo --local_dir ./datasets/X-Fun-Videos-Audios-Demo
```

### 2.2 数据集结构

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   ├── 📂 wav/
│   │   ├── 📄 audio001.wav
│   │   ├── 📄 audio002.wav
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 2.3 metadata.json 格式

FantasyTalking 的 `metadata.json` 与 VideoX-Fun 的普通 JSON 格式略有不同，**需要额外添加 `audio_path` 字段**。

**相对路径格式**（示例）：

```json
[
  {
    "file_path": "train/00000001.mp4",
    "audio_path": "wav/00000001.wav",
    "text": "一个女孩在海边说话。",
    "type": "video",
    "width": 512,
    "height": 512
  },
  {
    "file_path": "train/00000002.mp4",
    "audio_path": "wav/00000002.wav",
    "text": "一个男人在房间里讲话。",
    "type": "video",
    "width": 512,
    "height": 512
  }
]
```

**绝对路径格式**：

```json
[
  {
    "file_path": "/path/to/your/dataset/train/00000001.mp4",
    "audio_path": "/path/to/your/dataset/wav/00000001.wav",
    "text": "一个女孩在海边说话。",
    "type": "video",
    "width": 512,
    "height": 512
  }
]
```

**字段说明**：
- `file_path`：视频文件的相对路径或绝对路径
- `audio_path`：音频文件的相对路径或绝对路径（**FantasyTalking 必需字段**）
  - 音频文件通常为 `.wav` 格式
  - 路径应与 `file_path` 对应，如 `train/00000001.mp4` 对应 `wav/00000001.wav`
- `text`：视频的文本描述（prompt，可选）
- `type`：数据类型，固定为 `"video"`
- `width` / `height`：视频尺寸（**推荐**提供，用于 bucket 训练；如果不提供，训练时会自动读取，当数据存储在 OSS 等慢速系统时可能拖慢训练速度）
  - 可以使用 `scripts/process_json_add_width_and_height.py` 为没有这些字段的 JSON 文件添加 width 和 height 字段，支持图片和视频
  - 使用方法：`python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Audios-Demo/metadata.json --output_file datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json`

### 2.4 相对路径与绝对路径使用

**相对路径**：

如果你的数据使用的是相对路径，训练脚本中请这样配置：

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
```

**绝对路径**：

如果你的数据使用的是绝对路径，训练脚本中请这样配置：

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/path/to/your/metadata.json"
```

> 💡 **建议**：如果数据集较小且存放在本地，请使用相对路径。如果数据集存放在外部存储（如 NAS、OSS）或多机共享，请使用绝对路径。

---

## 三、全参数训练

### 3.1 下载预训练模型

训练前需要下载以下预训练模型：

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer
mkdir -p models/Personalized_Model

# 下载 Wan2.1-I2V-14B-720P 模型
modelscope download --model Wan-AI/Wan2.1-I2V-14B-720P --local_dir models/Diffusion_Transformer/Wan2.1-I2V-14B-720P

# 下载音频编码器（wav2vec2）
modelscope download --model AI-ModelScope/wav2vec2-base-960h --local_dir models/Diffusion_Transformer/wav2vec2-base-960h

# 下载 FantasyTalking 预训练权重
modelscope download --model amap_cvlab/FantasyTalking --local_dir models/Personalized_Model/FantasyTalking/
```

### 3.2 快速开始（DeepSpeed-Zero-2）

如果你已按 **2.1 快速测试数据集** 下载了数据，按 **3.1 下载预训练模型** 下载了权重，你可以直接复制运行快速开始命令。

推荐使用 DeepSpeed-Zero-2 或 FSDP 进行训练。这里以 DeepSpeed-Zero-2 为例。

DeepSpeed-Zero-2 与 FSDP 的区别在于模型权重是否分片。**如果多卡使用 DeepSpeed-Zero-2 显存不够**，可切换为 FSDP。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
export MODEL_NAME_AUDIO="models/Diffusion_Transformer/wav2vec2-base-960h"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# 如果没有 RDMA 的多节点训练，取消注释以下两行
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/fantasytalking/train.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_audio_model_name_or_path=$MODEL_NAME_AUDIO \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_fantasytalking" \
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
  --low_vram \
  --transformer_path="models/Personalized_Model/FantasyTalking/fantasytalking_model.ckpt" \
  --trainable_modules "processor." "proj_model."
```

### 3.3 常用训练参数

以下是训练脚本中关键参数的详细说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `config_path` | 模型配置文件路径 | `config/wan2.1/wan_civitai.yaml` |
| `pretrained_model_name_or_path` | 预训练模型路径 | `models/Diffusion_Transformer/Wan2.1-I2V-14B-720P` |
| `pretrained_audio_model_name_or_path` | 音频编码器路径 | `None`（自动使用 $MODEL_NAME/audio_encoder） |
| `train_data_dir` | 训练数据集目录 | `datasets/internal_datasets/` |
| `train_data_meta` | 训练数据集元数据文件 | `datasets/internal_datasets/metadata.json` |
| `video_sample_size` | 视频采样尺寸（最大分辨率） | `512` |
| `token_sample_size` | Token 采样尺寸 | `512` |
| `video_sample_stride` | 视频采样步幅 | `1` |
| `video_sample_n_frames` | 视频采样帧数 | `81` |
| `train_batch_size` | 训练批次大小 | `1` |
| `gradient_accumulation_steps` | 梯度累积步数 | `1` |
| `dataloader_num_workers` | 数据加载工作线程数 | `8` |
| `num_train_epochs` | 训练轮数 | `100` |
| `checkpointing_steps` | 保存检查点的步数 | `50` |
| `learning_rate` | 学习率 | `2e-05` |
| `lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `lr_warmup_steps` | 学习率预热步数 | `100` |
| `seed` | 随机种子 | `42` |
| `output_dir` | 输出目录 | `output_dir_fantasytalking` |
| `gradient_checkpointing` | 启用梯度检查点节省显存 | `True` |
| `mixed_precision` | 混合精度训练：`bf16` 或 `fp16` | `bf16` |
| `adam_weight_decay` | Adam 权重衰减 | `3e-2` |
| `adam_epsilon` | Adam epsilon | `1e-10` |
| `vae_mini_batch` | VAE 小批次大小 | `1` |
| `max_grad_norm` | 最大梯度范数 | `0.05` |
| `transformer_path` | 预训练 Transformer 权重路径 | `models/FantasyTalking/fantasytalking_model.ckpt` |
| `trainable_modules` | 可训练模块列表 | `"processor." "proj_model."` |

**高级参数说明**：

以下参数在训练脚本中可能会让人困惑，这里详细解释：

- **`enable_bucket`**：启用 Bucket 训练。启用后，模型不会在中心裁剪视频，而是根据分辨率将视频分组到不同的 Bucket 中进行训练。这可以让模型更好地适应不同分辨率的视频。

- **`random_frame_crop`**：在视频帧上进行随机裁剪，用于模拟不同帧数的视频。这可以帮助模型更好地泛化到不同长度的视频。

- **`random_hw_adapt`**：启用自动高度和宽度缩放。启用后，训练视频的高度和宽度将设置为：
  - 最大值：`video_sample_size`
  - 最小值：`512`
  
  **示例**：启用 `random_hw_adapt`，设置 `video_sample_n_frames=81`，`video_sample_size=768` 时，训练输入的视频分辨率可以是 `512x512x81` 或 `768x768x81`。

- **`training_with_video_token_length`**：根据 Token 长度训练模型。启用后，训练视频的高度和宽度将设置为：
  - 最大值：`video_sample_size`
  - 最小值：`256`
  
  **示例**：启用 `training_with_video_token_length`，设置 `video_sample_n_frames=81`，`token_sample_size=512`，`video_sample_size=768` 时，训练输入的视频分辨率可以是 `256x256x81`、`512x512x81` 或 `768x768x37`。
  
  **Token 长度计算**：
  - 512x512 分辨率、81 帧的视频，Token 长度约为 21,952
  - 我们需要设置 `token_sample_size = 512`
    - 在 512x512 分辨率下，视频帧数为 81（≈ 512 * 512 * 81 / 512 / 512）
    - 在 768x768 分辨率下，视频帧数为 37（≈ 512 * 512 * 81 / 768 / 768）
    - 在 1024x1024 分辨率下，视频帧数为 16（≈ 512 * 512 * 81 / 1024 / 1024）
    - 这些分辨率与对应长度的组合，使模型能够生成不同尺寸的视频。

- **`resume_from_checkpoint`**：从之前的检查点恢复训练。可以使用路径或 `"latest"` 自动选择最后一个可用的检查点。

- **`low_vram`**：启用低显存模式，通过优化内存使用来减少显存占用。

- **`uniform_sampling`**：使用均匀采样策略进行 timestep 采样。

### 3.4 训练验证

你可以配置验证参数，在训练过程中定期生成测试视频，以便监控训练进度和模型质量。

**验证参数说明**：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--validation_steps` | 每 N 步执行一次验证 | 100 |
| `--validation_epochs` | 每 N 个 epoch 执行一次验证 | 500 |
| `--validation_image_paths` | 验证用的参考图像路径列表，可用空格分隔多个路径 | 多个空格分隔的图像路径 |
| `--validation_audio_paths` | 验证用的音频路径列表，可用空格分隔多个路径 | 多个空格分隔的音频路径 |
| `--validation_prompts` | 验证用的提示词列表，可用空格分隔多个提示词 | 多个空格分隔的提示词 |

**示例**：

```bash
  --validation_image_paths="asset/8.png" \
  --validation_audio_paths="asset/talk.wav" \
  --validation_prompts="一个女孩在海边说话。" \
  --validation_steps=100 \
  --validation_epochs=500
```

**注意事项**：
- `validation_image_paths`、`validation_audio_paths` 和 `validation_prompts` 的数量必须一致
- `validation_steps` 和 `validation_epochs` 同时设置时，满足任一条件即触发验证

### 3.5 使用 FSDP 训练

**如果多卡使用 DeepSpeed-Zero-2 显存不够**，可切换为 FSDP。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
export MODEL_NAME_AUDIO="models/Diffusion_Transformer/wav2vec2-base-960h"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=AudioAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/fantasytalking/train.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_audio_model_name_or_path=$MODEL_NAME_AUDIO \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_fantasytalking" \
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
  --low_vram \
  --transformer_path="models/Personalized_Model/FantasyTalking/fantasytalking_model.ckpt" \
  --trainable_modules "processor." "proj_model."
```

**FSDP 关键参数说明**：

| 参数 | 说明 |
|------|------|
| `--use_fsdp` | 启用 FSDP |
| `--fsdp_auto_wrap_policy` | 自动包装策略：`TRANSFORMER_BASED_WRAP` |
| `--fsdp_transformer_layer_cls_to_wrap` | 要包装的 Transformer 层类名：`AudioAttentionBlock` |
| `--fsdp_sharding_strategy` | 分片策略：`FULL_SHARD` |
| `--fsdp_state_dict_type` | 状态字典类型：`SHARDED_STATE_DICT` |
| `--fsdp_backward_prefetch` | 反向传播预取：`BACKWARD_PRE` |
| `--fsdp_cpu_ram_efficient_loading` | CPU 内存高效加载：`False` |

### 3.6 不使用 DeepSpeed 或 FSDP 训练

**不推荐此方式，因为缺少显存优化后端，容易显存溢出**。仅供参考。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
export MODEL_NAME_AUDIO="models/Diffusion_Transformer/wav2vec2-base-960h"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/fantasytalking/train.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_audio_model_name_or_path=$MODEL_NAME_AUDIO \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_fantasytalking" \
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
  --low_vram \
  --transformer_path="models/Personalized_Model/FantasyTalking/fantasytalking_model.ckpt" \
  --trainable_modules "processor." "proj_model."
```

### 3.7 多机分布式训练

**适用场景**：超大规模数据集，更快训练速度

#### 3.7.1 环境配置

假设 2 台机器，每台 8 张 GPU：

**机器 0（Master）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
export MODEL_NAME_AUDIO="models/Diffusion_Transformer/wav2vec2-base-960h"  # 如果为 None，将使用 $MODEL_NAME/audio_encoder
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export MASTER_ADDR="192.168.1.100"  # Master 机器 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 总机器数
export NUM_PROCESS=16                # 总进程数 = 机器数 × 8
export RANK=0                        # 当前机器 rank（0 或 1）
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/fantasytalking/train.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_audio_model_name_or_path=$MODEL_NAME_AUDIO \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_fantasytalking" \
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
  --low_vram \
  --transformer_path="models/Personalized_Model/FantasyTalking/fantasytalking_model.ckpt" \
  --trainable_modules "processor." "proj_model."
```

**机器 1（Worker）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
export MODEL_NAME_AUDIO="models/Diffusion_Transformer/wav2vec2-base-960h"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export MASTER_ADDR="192.168.1.100"  # 与 Master 相同
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # 注意这里是 1
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# 使用与机器 0 相同的 accelerate launch 命令
```

#### 3.7.2 多机训练注意事项

- **网络要求**：
  - 推荐使用 RDMA/InfiniBand（高性能）
  - 无 RDMA 时，需要添加环境变量：
    ```bash
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
    ```

- **数据同步**：所有机器必须能够访问相同的数据路径（NFS/共享存储）

---

## 四、推理测试

训练完成后，可以使用推理脚本测试生成的模型。

### 4.1 推理参数

推理脚本 `examples/fantasytalking/predict_s2v.py` 中的主要参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `GPU_memory_mode` | GPU 显存模式：`model_full_load`、`model_full_load_and_qfloat8`、`model_cpu_offload`、`model_cpu_offload_and_qfloat8`、`sequential_cpu_offload` | `sequential_cpu_offload` |
| `ulysses_degree` | 多 GPU 推理 Ulysses 并行度 | `1` |
| `ring_degree` | 多 GPU 推理 Ring 并行度 | `1` |
| `fsdp_dit` | 多 GPU 推理时对 Transformer 使用 FSDP 节省显存 | `False` |
| `compile_dit` | 编译 Transformer 加速推理（固定分辨率有效） | `False` |
| `config_path` | 模型配置文件路径 | `config/wan2.1/wan_civitai.yaml` |
| `model_name` | 模型路径 | `models/Diffusion_Transformer/Wan2.1-I2V-14B-720P` |
| `model_name_audio` | 音频编码器路径 | `models/Diffusion_Transformer/wav2vec2-base-960h` |
| `sampler_name` | 采样器类型：`Flow`、`Flow_Unipc`、`Flow_DPM++` | `Flow` |
| `shift` | 采样器 shift 参数 | 5.0 |
| `transformer_path` | 训练后的 Transformer 权重路径 | `models/Personalized_Model/FantasyTalking/fantasytalking_model.ckpt` |
| `vae_path` | 训练后的 VAE 权重路径 | `None` |
| `lora_path` | LoRA 权重路径 | `None` |
| `sample_size` | 生成视频分辨率 `[height, width]` | `[832, 480]` |
| `video_length` | 生成视频帧数 | `81` |
| `fps` | 每秒帧数 | `23` |
| `weight_dtype` | 模型权重精度，无 bf16 的显卡使用 `torch.float16` | `torch.bfloat16` |
| `validation_image_start` | 参考图像路径 | `"asset/8.png"` |
| `audio_path` | 输入音频路径 | `"asset/talk.wav"` |
| `prompt` | 生成提示词 | `"一个女孩在海边说话。"` |
| `negative_prompt` | 负向提示词 | 详见代码 |
| `guidance_scale` | 提示词引导强度 | `4.5` |
| `audio_guide_scale` | 音频引导强度 | `4.0` |
| `seed` | 随机种子，保证可重复性 | `43` |
| `num_inference_steps` | 推理步数 | `40` |
| `lora_weight` | LoRA 权重强度 | `0.55` |
| `save_path` | 生成视频保存路径 | `samples/fantasy-talking-videos-speech2v` |

**TeaCache 加速配置**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `enable_teacache` | 是否启用 TeaCache 加速 | `True` |
| `teacache_threshold` | TeaCache 阈值（推荐 0.05~0.30） | `0.10` |
| `num_skip_start_steps` | 跳过 TeaCache 的初始步数 | `5` |
| `teacache_offload` | 将 TeaCache 张量卸载到 CPU 节省显存 | `False` |

**GPU 显存模式说明**：

| 模式 | 说明 | 显存占用 |
|------|------|----------|
| `model_full_load` | 将整个模型加载到 GPU | 最高 |
| `model_full_load_and_qfloat8` | 全量加载 + FP8 量化 | 高 |
| `model_cpu_offload` | 使用后模型卸载到 CPU | 中 |
| `model_cpu_offload_and_qfloat8` | CPU 卸载 + FP8 量化 | 中低 |
| `sequential_cpu_offload` | 每层使用后卸载到 CPU（最慢） | 最低 |

### 4.2 单 GPU 推理

运行单卡推理：

```bash
python examples/fantasytalking/predict_s2v.py
```

根据需求编辑 `examples/fantasytalking/predict_s2v.py`。首次推理请重点修改以下参数，其他参数见上方推理参数说明。

```python
# 根据显卡显存选择
GPU_memory_mode = "model_full_load"
# 模型配置文件路径
config_path = "config/wan2.1/wan_civitai.yaml"
# 你的实际模型路径
model_name = "models/Diffusion_Transformer/Wan2.1-I2V-14B-720P"
# 音频编码器路径
model_name_audio = "models/Diffusion_Transformer/wav2vec2-base-960h"
# 训练后的权重路径，如 "output_dir_fantasytalking/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = "models/Personalized_Model/FantasyTalking/fantasytalking_model.ckpt"
# 参考图像路径
validation_image_start = "asset/8.png"
# 输入音频路径
audio_path = "asset/talk.wav"
# 生成提示词
prompt = "一个女孩在海边说话。"
# ...
```

### 4.3 多 GPU 并行推理

**适用场景**：高分辨率生成、加速推理

#### 安装并行推理依赖

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### 配置并行策略

编辑 `examples/fantasytalking/predict_s2v.py`：

```python
# 确保 ulysses_degree × ring_degree = 使用的 GPU 数
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
| 8 | 2 | 4 | 混合并行 |
| 8 | 8 | 1 | Head 并行 |

#### 运行多 GPU 推理

```bash
torchrun --nproc-per-node=2 examples/fantasytalking/predict_s2v.py
```

---

## 五、更多资源

- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun