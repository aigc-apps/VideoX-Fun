# Wan2.1 Fun Control 全参数训练指南

本文档提供 Wan2.1 Fun Control Diffusion Transformer 全参数训练的完整流程,包括环境配置、数据准备、分布式训练和推理测试。

> **说明**：Wan2.1 Fun Control 是一个支持可控视频生成（如姿态控制 Pose）的视频生成模型。本文档涵盖 Control 模型的全参数训练流程。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径使用方案](#24-相对路径与绝对路径使用方案)
- [三、全量参数训练](#三全量参数训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始（DeepSpeed-Zero-2）](#32-快速开始deepspeed-zero-2)
  - [3.3 训练常用参数解析](#33-训练常用参数解析)
  - [3.4 训练验证](#34-训练验证)
  - [3.5 使用 FSDP 训练](#35-使用-fsdp-训练)
  - [3.6 不使用 DeepSpeed 与 FSDP 训练](#36-不使用-deepspeed-与-fsdp-训练)
  - [3.7 多机分布式训练](#37-多机分布式训练)
- [四、推理测试](#四推理测试)
  - [4.1 推理参数解析](#41-推理参数解析)
  - [4.2 Control 视频推理](#42-control-视频推理)
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

我们提供了一个 Control 训练的测试数据集，其中包含若干训练视频及对应的控制视频（如姿态视频）。

```bash
# 下载官方示例数据集
modelscope download --dataset PAI/X-Fun-Videos-Controls-Demo --local_dir ./datasets/X-Fun-Videos-Controls-Demo
```

### 2.2 数据集结构

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   ├── 📂 control/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

> **注意**：`control/` 目录存放与 `train/` 目录中视频一一对应的控制信号视频（如姿态视频、边缘检测视频等）。

### 2.3 metadata.json 格式

**相对路径格式**（示例格式）：
```json
[
  {
    "file_path": "train/video001.mp4",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
    "type": "video",
    "control_file_path": "control/video001.mp4",
    "width": 1024,
    "height": 1024
  },
  {
    "file_path": "train/video002.mp4",
    "text": "A person walking through a forest, cinematic view",
    "type": "video",
    "control_file_path": "control/video002.mp4",
    "width": 1328,
    "height": 1328
  }
]
```

**绝对路径格式**：
```json
[
  {
    "file_path": "/mnt/data/videos/sunset.mp4",
    "text": "A beautiful sunset over the ocean",
    "type": "video",
    "control_file_path": "/mnt/data/control/sunset.mp4",
    "width": 1024,
    "height": 1024
  }
]
```

**关键字段说明**：
- `file_path`：视频路径（相对或绝对路径）
- `text`：视频描述（英文提示词）
- `type`：数据类型，固定为 `"video"`
- `control_file_path`：对应的控制信号视频路径（如姿态视频），路径格式与 `file_path` 保持一致
- `width` / `height`：视频宽高（**最好提供**，用于分桶训练，如果不提供则自动在训练时读取，当数据存储在如oss这样的速度较慢的系统上时，可能会影响训练速度）。
  - 可以使用`scripts/process_json_add_width_and_height.py`文件对无width与height字段的json进行提取，支持处理图片与视频。
  - 使用方案为`python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Controls-Demo/metadata.json --output_file datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json`。

### 2.4 相对路径与绝对路径使用方案

**相对路径**：

如果数据的路径为相对路径，则在训练脚本中设置：

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata_control.json"
```

**绝对路径**：

如果数据的路径为绝对路径，则在训练脚本中设置：

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata_control.json"
```

> 💡 **建议**：如果数据集较小且存储在本地，推荐使用相对路径；如果数据集存储在外部存储（如 NAS、OSS）或多个机器共享存储，推荐使用绝对路径。

---

## 三、全量参数训练

### 3.1 下载预训练模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Wan2.1 Fun Control 官方权重
modelscope download --model PAI/Wan2.1-Fun-V1.1-14B-Control --local_dir models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-Control

modelscope download --model PAI/Wan2.1-Fun-V1.1-1.3B-Control --local_dir models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control
```

### 3.2 快速开始（DeepSpeed-Zero-2）

如果按照 **2.1 快速测试数据集下载数据** 与 **3.1 下载预训练模型下载权重**后，直接复制快速开始的启动指令进行启动。

推荐使用DeepSpeed-Zero-2与FSDP方案进行训练。这里使用DeepSpeed-Zero-2为例配置shell文件。

本文中DeepSpeed-Zero-2与FSDP的差别在于是否对模型权重进行分片，**如果使用多卡且使用DeepSpeed-Zero-2的情况下显存不足**，可以切换使用FSDP进行训练。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-Control"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_fun/train_control.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
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
  --output_dir="output_dir_wan2.1_fun_control" \
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
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_full_ref_image_in_self_attention \
  --trainable_modules "."
```

### 3.3 训练常用参数解析

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--config_path` | 模型配置文件路径 | `config/wan2.1/wan_civitai.yaml` |
| `--pretrained_model_name_or_path` | 预训练模型路径 | `models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-Control` |
| `--train_data_dir` | 训练数据目录 | `datasets/internal_datasets/` |
| `--train_data_meta` | 训练数据元文件 | `datasets/internal_datasets/metadata_control.json` |
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
| `--learning_rate` | 初始学习率 | 2e-05 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--seed` | 随机种子 | 42 |
| `--output_dir` | 输出目录 | `output_dir_wan2.1_fun_control` |
| `--gradient_checkpointing` | 激活重计算 | - |
| `--mixed_precision` | 混合精度：`fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 3e-2 |
| `--adam_epsilon` | AdamW epsilon 值 | 1e-10 |
| `--vae_mini_batch` | VAE 编码时的迷你批次大小 | 1 |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--enable_bucket` | 启用分桶训练，不裁剪图片/视频，按分辨率分组训练 | - |
| `--random_hw_adapt` | 自动缩放图片/视频到 `[min_size, max_size]` 范围内的随机尺寸 | - |
| `--training_with_video_token_length` | 根据 token 长度训练，支持任意分辨率 | - |
| `--uniform_sampling` | 均匀采样 timestep（推荐启用） | - |
| `--low_vram` | 低显存模式 | - |
| `--train_mode` | 训练模式：`control`、`control_ref`、`control_camera_ref` | `control_ref` |
| `--control_ref_image` | 参考图像来源：`first_frame` 或 `random` | `random` |
| `--add_full_ref_image_in_self_attention` | 在自注意力中加入完整参考图像 | - |
| `--resume_from_checkpoint` | 恢复训练路径，使用 `"latest"` 自动选择最新 checkpoint | None |
| `--validation_steps` | 每 N 步执行一次验证 | 100 |
| `--validation_epochs` | 每 N 个epoch执行一次验证 | 500 |
| `--validation_prompts` | 验证视频生成的提示词 | `"A woman dancing..."` |
| `--validation_paths` | 验证时使用的控制视频路径 | `"asset/pose.mp4"` |
| `--trainable_modules` | 可训练模块（`"."` 表示所有模块） | `"."` |

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
- 例如：640x640 分辨率、81 帧的视频，其 token 长度约为 40,960，需要设置 `token_sample_size = 640`。
  - 在 640x640 分辨率下，视频帧数为 81。
  - 在 832x480 分辨率下，视频帧数约为 75 (~= 640 * 640 * 81 / 832 / 480)。
  - 这些分辨率与对应帧数的组合，使模型能够生成不同尺寸的视频。

**训练模式说明**：
- `train_mode="control"`：标准 Control 模式，使用控制视频引导生成。
- `train_mode="control_ref"`：Control + 参考图像模式，在控制视频的基础上增加参考图像信息。
- `train_mode="control_camera_ref"`：Control + 相机运动 + 参考图像模式。
- `control_ref_image`：在 `control_ref` 模式下，选择参考图像来源，`first_frame` 使用视频首帧，`random` 使用随机帧。

### 3.4 训练验证

你可以配置验证参数，在训练过程中定期生成测试视频，以便监控训练进度和模型质量。

**验证参数说明**：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--validation_steps` | 每 N 步执行一次验证 | 100 |
| `--validation_epochs` | 每 N 个epoch执行一次验证 | 500 |
| `--validation_prompts` | 验证视频生成的提示词,可用空格分隔多个提示词 | 多个空格分隔的提示词 |
| `--validation_paths` | 验证时使用的控制视频路径，与 `validation_prompts` 一一对应 | `"asset/pose.mp4"` |

**示例**：

```bash
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="In this sunlit outdoor garden, a beautiful woman wears a knee-length white sleeveless dress, its hem swaying gently with her graceful movements like a dancing butterfly. Sunlight filters through the leaves, casting dappled shadows that highlight her soft features and clear eyes, enhancing her elegance. Every motion seems to speak of youth and vitality as she spins on the grass, her skirt fluttering around her, as if the entire garden rejoices in her dance. Colorful flowers all around—roses, chrysanthemums, lilies—sway in the breeze, releasing their fragrances and creating a relaxed and joyful atmosphere." \
  --validation_paths "asset/pose.mp4" \
```

**注意事项**：
- 验证视频会保存到 `output_dir` 目录中
- 多提示词验证格式：`--validation_prompts "prompt1" "prompt2" "prompt3"`
- 多控制视频格式：`--validation_paths "path1.mp4" "path2.mp4" "path3.mp4"`
- `validation_prompts` 和 `validation_paths` 的数量需要一一对应

### 3.5 使用 FSDP 训练

**如果使用多卡且使用DeepSpeed-Zero-2的情况下显存不足**，可以切换使用FSDP进行训练。

```sh
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-Control"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.1_fun/train_control.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
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
  --output_dir="output_dir_wan2.1_fun_control" \
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
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_full_ref_image_in_self_attention \
  --trainable_modules "."
```

### 3.6 不使用 DeepSpeed 与 FSDP 训练

**该方案并不被推荐，因为没有显存节约后端，容易造成显存不足**。这里仅提供训练Shell用于参考训练。

```sh
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-Control"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.1_fun/train_control.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
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
  --output_dir="output_dir_wan2.1_fun_control" \
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
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_full_ref_image_in_self_attention \
  --trainable_modules "."
```

### 3.7 多机分布式训练

**适合场景**：超大规模数据集、需要更快的训练速度

#### 3.7.1 环境配置

假设有 2 台机器，每台 8 张 GPU：

**机器 0（Master）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-Control"
export DATASET_NAME="datasets/X-Fun-Videos-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Controls-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master 机器 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 机器总数
export NUM_PROCESS=16                # 总进程数 = 机器数 × 8
export RANK=0                        # 当前机器 rank（0 或 1）
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1_fun/train_control.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
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
  --output_dir="output_dir_wan2.1_fun_control" \
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
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --add_full_ref_image_in_self_attention \
  --trainable_modules "."
```

**机器 1（Worker）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-Control"
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
NCCL_DEBUG=INFO

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
| `config_path` | 模型配置文件路径 | `config/wan2.1/wan_civitai.yaml` |
| `GPU_memory_mode` | 显存管理模式，可选值见下表 | `sequential_cpu_offload` |
| `ulysses_degree` | Ulysses 并行度，用于多卡推理 | 1 |
| `ring_degree` | Ring 并行度，用于多卡推理 | 1 |
| `fsdp_dit` | 多卡推理时对 Transformer 使用 FSDP 节省显存 | `False` |
| `fsdp_text_encoder` | 多卡推理时对文本编码器使用 FSDP | `True` |
| `compile_dit` | 编译 Transformer 加速推理（固定分辨率下有效） | `False` |
| `model_name` | 模型路径 | `models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control` |
| `sampler_name` | 采样器类型：`Flow`、`Flow_Unipc`、`Flow_DPM++` | `Flow` |
| `transformer_path` | 加载训练好的 Transformer 权重路径 | `None` |
| `vae_path` | 加载训练好的 VAE 权重路径 | `None` |
| `lora_path` | LoRA 权重路径 | `None` |
| `sample_size` | 生成视频分辨率 `[高度, 宽度]` | `[832, 480]` |
| `video_length` | 生成视频帧数 | `49` |
| `fps` | 每秒帧数 | `16` |
| `weight_dtype` | 模型权重精度，不支持 bf16 的显卡使用 `torch.float16` | `torch.bfloat16` |
| `control_video` | 控制信号视频路径（如姿态视频） | `"asset/pose.mp4"` |
| `control_camera_txt` | 相机运动控制 txt 文件路径 | `None` |
| `ref_image` | 参考图像路径 | `"asset/6.png"` |
| `start_image` | 起始图像路径（与 ref_image 二选一） | `None` |
| `prompt` | 正向提示词，描述生成内容 | `"一位年轻女性..."` |
| `negative_prompt` | 负向提示词，避免生成的内容 | `"色调艳丽，过曝..."` |
| `guidance_scale` | 引导强度 | 6.0 |
| `seed` | 随机种子，用于复现结果 | 43 |
| `num_inference_steps` | 推理步数 | 50 |
| `lora_weight` | LoRA 权重强度 | 0.55 |
| `save_path` | 生成视频保存路径 | `samples/wan-videos-fun-control` |

**显存管理模式说明**：

| 模式 | 说明 | 显存占用 |
|------|------|---------|
| `model_full_load` | 整个模型加载到 GPU | 最高 |
| `model_full_load_and_qfloat8` | 全量加载 + FP8 量化 | 高 |
| `model_cpu_offload` | 使用后将模型卸载到 CPU | 中等 |
| `model_cpu_offload_and_qfloat8` | CPU 卸载 + FP8 量化 | 中低 |
| `model_group_offload` | 层组在 CPU/CUDA 间切换 | 低 |
| `sequential_cpu_offload` | 逐层卸载（速度最慢） | 最低 |

### 4.2 Control 视频推理

单卡推理运行如下命令：

```bash
python examples/wan2.1_fun/predict_v2v_control_ref.py
```

根据需求修改编辑 `examples/wan2.1_fun/predict_v2v_control_ref.py`，初次推理重点关注如下参数，如果对其他参数感兴趣，请查看上方的推理参数解析。

```python
# 模型配置文件路径
config_path = "config/wan2.1/wan_civitai.yaml"
# 根据显卡显存选择
GPU_memory_mode = "sequential_cpu_offload"
# 根据实际模型路径
model_name = "models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"  
# 训练好的权重路径，如 "output_dir_wan2.1_fun_control/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = None  
# 控制信号视频路径（如姿态视频）
control_video = "asset/pose.mp4"
# 参考图像路径
ref_image = "asset/6.png"
# 根据生成内容编写
prompt = "一位年轻女性穿着一件粉色的连衣裙..."  
# ...
```

### 4.3 多卡并行推理

**适合场景**：高分辨率生成、加速推理

#### 安装并行推理依赖

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### 配置并行策略

编辑 `examples/wan2.1_fun/predict_v2v_control_ref.py`：

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
| 8 | 8 | 1 | Head 并行 |
| 8 | 4 | 2 | 混合并行 |

#### 运行多卡推理

```bash
torchrun --nproc-per-node=2 examples/wan2.1_fun/predict_v2v_control_ref.py
```

---

## 五、更多资源

- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
