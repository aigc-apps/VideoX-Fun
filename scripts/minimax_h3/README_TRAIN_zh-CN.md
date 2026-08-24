# MiniMax-H3 全参数训练指南

本文档提供 MiniMax-H3 全参数训练的完整工作流，包括环境配置、数据准备、分布式训练和推理测试。

> **注意**：MiniMax-H3 是一个音视频生成视频模型，可以同时生成视频和对应的音频。训练数据需要同时包含视频和音频文件。

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
  - [3.2 快速开始（FSDP）](#32-快速开始fsdp)
  - [3.3 常用训练参数](#33-常用训练参数)
  - [3.4 训练验证](#34-训练验证)
  - [3.5 使用 DeepSpeed-Zero-2 训练](#35-使用-deepspeed-zero-2-训练)
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
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

**方式三：使用 Docker**

使用 Docker 时，请先确保本机已正确安装 GPU 驱动和 CUDA 环境，然后执行以下命令：

```
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

> ⚠️ **重要**：MiniMax-H3 是音视频联合生成模型。与普通视频训练不同，您**必须在 metadata.json 中提供 `audio_path` 字段**。训练时配套的音频波形会被重采样到音频 VAE 的采样率（32 kHz）。

**相对路径格式**（示例）：
```json
[
  {
    "file_path": "train/video001.mp4",
    "audio_path": "wav/audio001.wav",
    "text": "A brown dog barks on a sofa, sitting on a light-colored couch in a cozy room",
    "type": "video",
    "width": 768,
    "height": 512
  },
  {
    "file_path": "train/video002.mp4",
    "audio_path": "wav/audio002.wav",
    "text": "A group of young men in suits and sunglasses are walking down a city street",
    "type": "video",
    "width": 640,
    "height": 640
  }
]
```

**绝对路径格式**：
```json
[
  {
    "file_path": "/mnt/data/videos/dog.mp4",
    "audio_path": "/mnt/data/wavs/dog.wav",
    "text": "A brown dog barks on a sofa",
    "type": "video",
    "width": 768,
    "height": 512
  }
]
```

**关键字段说明**：
- `file_path`：视频文件路径（相对或绝对路径均可）
- `audio_path`：音频文件路径（**MiniMax-H3 特有且必填**，是与普通视频训练的主要区别）
  - 音频文件通常为 `.wav` 格式
  - 路径应与 `file_path` 对应，例如 `train/video001.mp4` 对应 `wav/audio001.wav`
- `text`：视频描述（英文 prompt）
- `type`：数据类型，固定为 `"video"`
- `width` / `height`：视频尺寸（**建议**提供以支持 bucket 训练；如不提供，训练时会自动读取，当数据存放在 OSS 等较慢的系统上时可能拖慢训练）
  - 可以使用 `scripts/process_json_add_width_and_height.py` 为缺少宽高字段的 JSON 文件补充 width 和 height 字段，同时支持图片和视频
  - 用法：`python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Audios-Demo/metadata.json --output_file datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json`

**数据集对比：MiniMax-H3 与普通视频训练**：

| 模型类型 | 必填字段 | 音频字段 |
|---------|---------|---------|
| 普通视频（WAN、CogVideoX 等） | `file_path`、`text`、`type` | ❌ 不需要 |
| **MiniMax-H3（音视频联合生成）** | `file_path`、`audio_path`、`text`、`type` | ✅ **必填** |

### 2.4 相对路径与绝对路径使用

**相对路径**：

如果您的数据使用相对路径，训练脚本配置如下：

```bash
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
```

**绝对路径**：

如果您的数据使用绝对路径，训练脚本配置如下：

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata_add_width_height.json"
```

> 💡 **建议**：如果数据集较小且存储在本地，使用相对路径。如果数据集存储在外部存储（如 NAS、OSS）上，或需要跨多台机器共享，使用绝对路径。

---

## 三、全参数训练

### 3.1 下载预训练模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 MiniMax-H3 官方权重
hf download MiniMax-AI/MiniMax-H3 --local-dir models/Diffusion_Transformer/MiniMax-H3
```

> 💡 加载器既支持上述转换后的 diffusers 布局，也支持*原始的* MiniMax-H3 分片布局（如 `MiniMax-H3/FL2VA`）；原始分片会在加载时在线转换，不会在磁盘上产生中间副本。

### 3.2 快速开始（FSDP）

如果您已按照 **2.1 快速测试数据集** 下载数据、按照 **3.1 下载预训练模型** 下载权重，可以直接复制并运行以下快速开始命令。

推荐使用 FSDP 训练 MiniMax-H3。其 transformer 在 bfloat16 下约 62 GB，Qwen3-VL 条件器还有约 62 GB，因此必须在多卡间分片模型权重——FSDP（`FULL_SHARD`）可以分片权重，而 DeepSpeed-Zero-2 不能。

```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# 无 RDMA 的多机环境需要设置 NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/minimax_h3/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --trainable_modules "." \
  --t2v_ratio=0.25 \
  --train_mode="fl2v"
```

### 3.3 常用训练参数

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--pretrained_model_name_or_path` | 预训练模型路径 | `models/Diffusion_Transformer/MiniMax-H3` |
| `--train_data_dir` | 训练数据目录 | `datasets/X-Fun-Videos-Audios-Demo/` |
| `--train_data_meta` | 训练数据元信息文件 | `datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json` |
| `--train_batch_size` | 每批样本数 | 1 |
| `--video_sample_size` | 训练的最大视频分辨率 | 960 |
| `--token_sample_size` | token 长度采样尺寸 | 960 |
| `--video_sample_stride` | 抽帧步长（MiniMax-H3 为 24 fps） | 1 |
| `--video_sample_n_frames` | 采样帧数，须满足视频 VAE 的 `17*n+5` 形式（时长保持在 5 到 15 秒之间） | 124 |
| `--video_repeat` | 每个 epoch 中每个视频重复的次数 | 1 |
| `--gradient_accumulation_steps` | 梯度累积步数（等效更大的 batch） | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 4 |
| `--num_train_epochs` | 训练轮数 | 100 |
| `--checkpointing_steps` | 每 N 步保存一次 checkpoint | 500 |
| `--learning_rate` | 初始学习率 | 1e-5 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--seed` | 随机种子 | 42 |
| `--output_dir` | 输出目录 | `output_dir_minimax_h3` |
| `--gradient_checkpointing` | 开启激活值检查点 | - |
| `--gradient_checkpointing_save_on_cpu` | 将 transformer blocks 为反向传播保存的激活值卸载到 CPU 内存 | - |
| `--mixed_precision` | 混合精度：`fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 3e-2 |
| `--adam_epsilon` | AdamW epsilon 值 | 1e-10 |
| `--vae_mini_batch` | VAE 编码的 mini-batch 大小 | 1 |
| `--max_grad_norm` | 梯度裁剪阈值 | 1.0 |
| `--random_hw_adapt` | 自动将视频缩放到 `[512, video_sample_size]` 范围内的随机尺寸 | - |
| `--training_with_video_token_length` | 按 token 长度训练，而非固定分辨率 | - |
| `--enable_bucket` | 开启 bucket 训练：按分辨率分组训练完整视频，不做中心裁剪 | - |
| `--uniform_sampling` | 均匀时间步采样 | - |
| `--low_vram` | VAE 与条件器常驻 CPU，仅在编码时移上 GPU | - |
| `--train_mode` | `t2v`（纯文本）或 `fl2v`（首帧 keyframe 条件，keyframe 取自训练样本自身） | `fl2v` |
| `--t2v_ratio` | 在 `--train_mode=fl2v` 下，按该比例的步数丢弃 keyframe 改训 t2v，使一次训练同时保留两种条件。取值须在 [0, 1] 内且仅适用于 fl2v；0 表示纯 fl2v | 0.25 |
| `--resume_from_checkpoint` | 从 checkpoint 路径恢复训练，使用 `"latest"` 自动选择最新 | 无 |
| `--trainable_modules` | 可训练模块（`"."` 表示全部模块） | `"."` |
| `--validation_steps` | 每 N 步执行一次验证 | 2000 |
| `--validation_epochs` | 每 N 个 epoch 执行一次验证 | 5 |
| `--validation_prompts` | 验证时使用的 prompt | `"A man in a blue blazer..."` |


### 3.4 训练验证

您可以配置验证参数，在训练过程中周期性生成测试视频，以便监控训练进度和模型质量。

**验证参数**：

| 参数 | 说明 | 推荐值 |
|-----------|-------------|-------------------|
| `--validation_steps` | 每 N 步执行一次验证 | 2000 |
| `--validation_epochs` | 每 N 个 epoch 执行一次验证 | 5 |
| `--validation_prompts` | 用于生成验证视频的 prompt，多个 prompt 以空格分隔 | 空格分隔的 prompt 字符串 |

**示例**：

```bash
  --validation_steps=2000 \
  --validation_epochs=5 \
  --validation_prompts="A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks."
```

**注意**：
- 验证视频将保存到 `output_dir/sample/` 目录
- 多 prompt 验证用法：`--validation_prompts "prompt1" "prompt2" "prompt3"`

### 3.5 使用 DeepSpeed-Zero-2 训练

> ⚠️ **警告**：DeepSpeed-Zero-2 只切分优化器状态和梯度，**不切分模型权重**。MiniMax-H3 的 transformer 约 62 GB，每张 GPU 仍需持有完整的权重副本，该配置通常会导致显存不足。MiniMax-H3 请优先使用 FSDP（**3.2**）；以下命令仅供参考。

```sh
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# 无 RDMA 的多机环境需要设置 NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/minimax_h3/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --trainable_modules "." \
  --t2v_ratio=0.25 \
  --train_mode="fl2v"
```

### 3.6 不使用 DeepSpeed 或 FSDP 训练

**不推荐该方式，因为没有节省显存的后端**。MiniMax-H3 的 transformer 约 62 GB、Qwen3-VL 条件器约 62 GB，且会在每张 GPU 上完整复制，几乎必然显存不足。以下命令仅供参考。

```sh
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
# 无 RDMA 的多机环境需要设置 NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/minimax_h3/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --trainable_modules "." \
  --t2v_ratio=0.25 \
  --train_mode="fl2v"
```

### 3.7 多机分布式训练

**适用场景**：超大规模数据集、更快的训练速度

#### 3.7.1 环境配置

假设 2 台机器，每台 8 张 GPU：

**机器 0（Master）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master 机器 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 机器总数
export NUM_PROCESS=16                # 总进程数 = 机器数 × 8
export RANK=0                        # 当前机器 rank（0 或 1）
# 无 RDMA 的多机环境需要设置 NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/minimax_h3/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_size=960 \
  --token_sample_size=960 \
  --video_sample_stride=1 \
  --video_sample_n_frames=124 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --trainable_modules "." \
  --t2v_ratio=0.25 \
  --train_mode="fl2v"
```

**机器 1（Worker）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATASET_NAME="datasets/X-Fun-Videos-Audios-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # 与 Master 相同
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # 注意这里是 1
# 无 RDMA 的多机环境需要设置 NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# 使用与机器 0 相同的 accelerate launch 命令
```

#### 3.7.2 多机训练注意事项

- **网络要求**：
   - 推荐 RDMA/InfiniBand（高性能）
   - 无 RDMA 时，添加环境变量：
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **数据同步**：所有机器必须能访问相同的数据路径（NFS/共享存储）

## 四、推理测试

### 4.1 推理参数

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `GPU_memory_mode` | GPU 显存模式，可选项见下表 | `model_cpu_offload` |
| `ulysses_degree` | 头维度并行度，单卡为 1 | 1 |
| `ring_degree` | 序列维度并行度，单卡为 1 | 1 |
| `fsdp_dit` | 多卡推理时对 Transformer 使用 FSDP 以节省显存 | `False` |
| `fsdp_text_encoder` | 多卡推理时对 Qwen3-VL 文本编码器使用 FSDP 以节省显存 | `False` |
| `compile_dit` | 编译 Transformer 以加速推理（固定分辨率下有效） | `False` |
| `model_name` | 模型路径 | `models/Diffusion_Transformer/MiniMax-H3` |
| `transformer_path` | 训练好的 Transformer 权重路径 | `None` |
| `vae_path` | 训练好的 VAE 权重路径 | `None` |
| `lora_path` | LoRA 权重路径 | `None` |
| `sample_size` | 生成视频分辨率 `[height, width]`；宽高必须是 32 的倍数。设为 `None` 时使用 MiniMax-H3 自带的 16:9 画布（768x1344） | `[704, 1280]` |
| `video_length` | 生成帧数，会向上取整到视频 VAE 可解码的下一个 `17*n+5`（时长保持在 5 到 15 秒之间） | 124 |
| `fps` | 每秒帧数（MiniMax-H3 固定以 24 fps 生成） | 24 |
| `weight_dtype` | 模型权重精度，不支持 bf16 的 GPU 请使用 `torch.float16` | `torch.bfloat16` |
| `prompt` | 描述生成内容的正向提示词 | `"A red fox trotting..."` |
| `seed` | 用于复现的随机种子 | 43 |
| `num_inference_steps` | 去噪步数，即模型前向次数；其构建的 sigma 网格在此之上额外含一个终点 0 | 40 |
| `guidance_scale` | 引导强度。发布的权重已做 guidance 蒸馏：保持 1 时每步只做一次前向、不走 CFG；大于 1 时启用 classifier-free guidance、走两次前向 | 1 |
| `flow_shift` | 视频调度的指数 sigma shift，`None` 时沿用权重自带值（12.0） | `None` |
| `audio_flow_shift` | 音频调度的指数 sigma shift，`None` 时沿用权重自带值（3.0） | `None` |
| `lora_weight` | LoRA 权重强度 | 0.55 |
| `save_path` | 生成视频保存路径 | `samples/minimax-h3-videos-t2v` |

**GPU 显存模式说明**：

| 模式 | 说明 | 显存占用 |
|------|------|---------|
| `model_full_load` | 整个模型加载到 GPU | 最高 |
| `model_full_load_and_qfloat8` | 全量加载 + FP8 量化 | 高 |
| `model_cpu_offload` | 模型用完后卸载到 CPU | 中 |
| `model_cpu_offload_and_qfloat8` | CPU 卸载 + FP8 量化 | 中低 |
| `model_group_offload` | 层级分组在 CPU/CUDA 间换入换出 | 低 |
| `sequential_cpu_offload` | 逐层卸载（最慢） | 最低 |

> 💡 transformer 在 bfloat16 下有 61.7 GB，Qwen3-VL 条件器还有 62.1 GB，因此单张 80 GB 卡需要使用 `model_cpu_offload` 或 `model_group_offload`。

### 4.2 单 GPU 推理

运行单卡推理：

```bash
python examples/minimax_h3/predict_t2v.py
```

按需编辑 `examples/minimax_h3/predict_t2v.py`。首次推理请重点关注以下参数，其他参数见上文推理参数章节。

```python
# 根据 GPU 显存选择
GPU_memory_mode = "model_cpu_offload"
# 您的实际模型路径
model_name = "models/Diffusion_Transformer/MiniMax-H3"  
# 训练好的权重路径，如 "output_dir_minimax_h3/checkpoint-xxx/transformer"
transformer_path = None  
# 按要生成的内容填写
prompt = "A red fox trotting through a snowy pine forest, snow crunching underfoot"  
# ...
```

### 4.3 多 GPU 并行推理

**适用场景**：高分辨率生成、推理加速

#### 安装并行推理依赖

```bash
pip install xfuser yunchang
```

#### 配置并行策略

编辑 `examples/minimax_h3/predict_t2v.py`：

```python
# 确保 ulysses_degree × ring_degree = GPU 数量
# 例如使用 2 张 GPU：
ulysses_degree = 2  # 头维度并行
ring_degree = 1     # 序列维度并行
```

**配置原则**：
- `ulysses_degree` 必须能整除模型的注意力头数
- `ring_degree` 在序列维度切分，会影响通信开销；头数能整除时尽量不用
- 多卡走 xfuser 序列并行路径，**与 `*cpu_offload*` 显存模式不兼容**（accelerate 的 offload hook 绑定单一设备）；多卡时请使用 `model_full_load` / `model_full_load_and_qfloat8`，并用 `fsdp_dit` / `fsdp_text_encoder` 节省显存

**配置示例**：

| GPU 数量 | ulysses_degree | ring_degree | 说明 |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | 单卡 |
| 4 | 4 | 1 | 头并行 |
| 8 | 2 | 4 | 混合并行 |
| 8 | 8 | 1 | 头并行 |

#### 运行多卡推理

```bash
torchrun --nproc_per_node=2 examples/minimax_h3/predict_t2v.py
```

## 五、更多资源

- **MiniMax-H3 官方 GitHub**：https://github.com/MiniMax-AI/MiniMax-H3
- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
