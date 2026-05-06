# Wan2.1 蒸馏 LoRA 训练指南

本文档提供了将 Wan2.1 进行蒸馏 + LoRA 微调的完整工作流,包括环境配置、数据准备、分布式训练和推理测试。

> **说明**:本训练方式结合了蒸馏(减少推理步数)和 LoRA(参数高效微调)两种技术,可以在显存占用较低的情况下,将推理步数从 25-50 步减少到 4-8 步,同时保持或提升视频生成质量。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径使用方案](#24-相对路径与绝对路径使用方案)
- [三、蒸馏 LoRA 训练](#三蒸馏-lora-训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始（DeepSpeed-Zero-2）](#32-快速开始deepspeed-zero-2)
  - [3.3 训练常用参数解析](#33-训练常用参数解析)
  - [3.4 训练验证](#34-训练验证)
  - [3.5 使用 FSDP 训练](#35-使用-fsdp-训练)
  - [3.6 其他后端](#36-其他后端)
  - [3.7 多机分布式训练](#37-多机分布式训练)
- [四、推理测试](#四推理测试)
  - [4.1 推理参数解析](#41-推理参数解析)
  - [4.2 文生视频（T2V）推理](#42-文生视频t2v推理)
  - [4.3 图生视频（I2V）推理](#43-图生视频i2v推理)
  - [4.4 多卡并行推理](#44-多卡并行推理)
- [五、更多资源](#五更多资源)

---

## 一、环境配置

**方式 1:使用 requirements.txt**

```bash
pip install -r requirements.txt
```

**方式 2:手动安装依赖**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl deepspeed==0.17.0 numpy==1.26.4
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

**方式 3:使用 docker**

使用 docker 的情况下,请保证机器中已经正确安装显卡驱动与 CUDA 环境,然后以此执行以下命令:

```
# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# 进入容器
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 二、数据准备

### 2.1 快速测试数据集

我们提供了一个测试的数据集,其中包含若干训练数据。

```bash
# 下载官方示例数据集
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 2.2 数据集结构

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 video001.mp4
│   │   ├── 📄 video002.mp4
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 2.3 metadata.json 格式

**相对路径格式**(示例格式):
```json
[
  {
    "file_path": "train/video001.mp4",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
    "type": "video",
    "width": 1024,
    "height": 1024
  },
  {
    "file_path": "train/video002.mp4",
    "text": "A person walking through a forest, cinematic view",
    "type": "video",
    "width": 1328,
    "height": 1328
  }
]
```

**绝对路径格式**:
```json
[
  {
    "file_path": "/mnt/data/videos/sunset.mp4",
    "text": "A beautiful sunset over the ocean",
    "type": "video",
    "width": 1024,
    "height": 1024
  }
]
```

**关键字段说明**:
- `file_path`:视频路径(相对或绝对路径)
- `text`:视频描述(英文提示词)
- `type`:数据类型,固定为 `"video"`
- `width` / `height`:视频宽高(**最好提供**,用于分桶训练,如果不提供则自动在训练时读取,当数据存储在如 oss 这样的速度较慢的系统上时,可能会影响训练速度)。
  - 可以使用`scripts/process_json_add_width_and_height.py`文件对无 width 与 height 字段的 json 进行提取,支持处理图片与视频。
  - 使用方案为`python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Videos-Demo/metadata.json --output_file datasets/X-Fun-Videos-Demo/metadata_add_width_height.json`。

### 2.4 相对路径与绝对路径使用方案

**相对路径**:

如果数据的路径为相对路径,则在训练脚本中设置:

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
```

**绝对路径**:

如果数据的路径为绝对路径,则在训练脚本中设置:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **建议**:如果数据集较小且存储在本地,推荐使用相对路径;如果数据集存储在外部存储(如 NAS、OSS)或多个机器共享存储,推荐使用绝对路径。

---

## 三、蒸馏 LoRA 训练

### 3.1 下载预训练模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Wan2.1 官方权重
# T2V 模型(文生视频)
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B
# 或 I2V 模型(图生视频)
# modelscope download --model Wan-AI/Wan2.1-I2V-14B-480P --local_dir models/Diffusion_Transformer/Wan2.1-I2V-14B-480P
```

### 3.2 快速开始（DeepSpeed-Zero-2）

如果按照 **2.1 快速测试数据集下载数据** 与 **3.1 下载预训练模型下载权重**后，直接复制快速开始的启动指令进行启动。

推荐使用DeepSpeed-Zero-2与FSDP方案进行训练。这里使用DeepSpeed-Zero-2为例配置shell文件。

本文中DeepSpeed-Zero-2与FSDP的差别在于是否对模型权重进行分片，**如果使用多卡且使用DeepSpeed-Zero-2的情况下显存不足**，可以切换使用FSDP进行训练。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1/train_distill_lora.py \
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
  --learning_rate=1e-05 \
  --learning_rate_critic=1e-06 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_distill_lora" \
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
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --train_mode="normal" \
  --low_vram
```

### 3.3 训练常用参数解析

**LoRA 特有参数**:

在蒸馏训练的基础上,LoRA 训练增加了以下特有参数:

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--use_peft_lora` | 是否使用 PEFT 模块添加 LoRA,使用该模块会更节省显存 | - |
| `--rank` | LoRA 更新矩阵的维度(秩) | 64 |
| `--network_alpha` | LoRA 更新矩阵的缩放系数 | 32 |
| `--target_name` | LoRA 应用的组件/模块,以逗号分隔 | `"q,k,v,ffn.0,ffn.2"` |
| `--lora_skip_name` | LoRA 跳过的模块(不训练) | None |

**LoRA 配置建议**:
- **rank=64, network_alpha=32**:适用于大多数场景,在质量和显存之间取得平衡
- **rank=128, network_alpha=64**:更高质量的微调,但需要更多显存
- **target_name="q,k,v,ffn.0,ffn.2"**:对注意力层和前馈网络进行微调,这是常用的配置
- **use_peft_lora**:强烈建议启用,可以显著降低显存占用

**关键参数说明**:

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--pretrained_model_name_or_path` | 预训练模型路径 | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B/` |
| `--train_data_dir` | 训练数据目录 | `datasets/internal_datasets/` |
| `--train_data_meta` | 训练数据元文件 | `datasets/internal_datasets/metadata.json` |
| `--train_batch_size` | 每批次样本数 | 1 |
| `--image_sample_size` | 图像最大训练分辨率 | 640 |
| `--video_sample_size` | 视频最大训练分辨率 | 640 |
| `--token_sample_size` | Token 采样尺寸 | 640 |
| `--video_sample_stride` | 视频采样步幅 | 2 |
| `--video_sample_n_frames` | 视频采样帧数 | 81 |
| `--gradient_accumulation_steps` | 梯度累积步数(等效增大 batch) | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数 | 100 |
| `--checkpointing_steps` | 每 N 步保存 checkpoint | 50 |
| `--learning_rate` | 初始学习率(生成器) | 1e-05 |
| `--learning_rate_critic` | 初始学习率(判别器) | 1e-06 |
| `--seed` | 随机种子 | 42 |
| `--output_dir` | 输出目录 | `output_dir_wan2.1_distill_lora` |
| `--gradient_checkpointing` | 激活重计算 | - |
| `--mixed_precision` | 混合精度:`fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 3e-2 |
| `--adam_epsilon` | AdamW epsilon 值 | 1e-10 |
| `--vae_mini_batch` | VAE 编码时的迷你批次大小 | 1 |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--enable_bucket` | 启用分桶训练,不裁剪图片/视频,按分辨率分组训练 | - |
| `--random_hw_adapt` | 自动缩放图片/视频到 `[min_size, max_size]` 范围内的随机尺寸 | - |
| `--training_with_video_token_length` | 根据 token 长度训练,支持任意分辨率 | - |
| `--uniform_sampling` | 均匀采样 timestep | - |
| `--low_vram` | 低显存模式 | - |
| `--train_mode` | 训练模式:`normal`(普通)或 `i2v`(图生视频) | `normal` |
| `--resume_from_checkpoint` | 恢复训练路径,使用 `"latest"` 自动选择最新 checkpoint | None |
| `--validation_steps` | 每 N 步执行一次验证 | 2000 |
| `--validation_epochs` | 每 N 个 epoch 执行一次验证 | 5 |
| `--validation_prompts` | 验证视频生成的提示词 | `"一只棕色的狗摇着头..."` |
| `--validation_paths` | 验证 I2V 的参考图像路径(仅 i2v 模式) | `"asset/1.png"` |

**蒸馏特有参数**:

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--denoising_step_indices_list` | 去噪步骤列表(蒸馏核心参数) | `1000 750 500 250` |
| `--real_guidance_scale` | 用于评分的真实 guidance scale | 6.0 |
| `--fake_guidance_scale` | 用于评分的虚拟 guidance scale | 0.0 |
| `--gen_update_interval` | 生成器更新间隔 | 5 |
| `--negative_prompt` | 用于蒸馏的负向提示词 | 中文负向提示词 |
| `--train_sampling_steps` | 训练采样步数 | 1000 |

**Sample Size 配置指南**:
- `video_sample_size` 表示视频的分辨率大小;当 `random_hw_adapt` 为 True 时,表示视频和图像分辨率的最小值。
- `image_sample_size` 表示图像的分辨率大小;当 `random_hw_adapt` 为 True 时,表示视频和图像分辨率的最大值。
- `token_sample_size` 表示当 `training_with_video_token_length` 为 True 时,最大 token 长度对应的分辨率。
- 由于配置可能产生混淆,**如果你不需要任意分辨率进行 finetuning**,建议将 `video_sample_size`、`image_sample_size` 和 `token_sample_size` 设置为相同的固定值,例如 **(320, 480, 512, 640, 960)**。
  - **全部设置为 320** 代表 **240P**。
  - **全部设置为 480** 代表 **320P**。
  - **全部设置为 640** 代表 **480P**。
  - **全部设置为 960** 代表 **720P**。

**Token Length 训练说明**:
- 当启用 `training_with_video_token_length` 时,模型根据 token 长度进行训练。
- 例如:512x512 分辨率、49 帧的视频,其 token 长度为 13,312,需要设置 `token_sample_size = 512`。
  - 在 512x512 分辨率下,视频帧数为 49 (~= 512 * 512 * 49 / 512 / 512)。
  - 在 768x768 分辨率下,视频帧数为 21 (~= 512 * 512 * 49 / 768 / 768)。
  - 在 1024x1024 分辨率下,视频帧数为 9 (~= 512 * 512 * 49 / 1024 / 1024)。
  - 这些分辨率与对应帧数的组合,使模型能够生成不同尺寸的视频。

### 3.4 训练验证

你可以配置验证参数,在训练过程中定期生成测试视频,以便监控训练进度和模型质量。

**验证参数说明**:

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--validation_steps` | 每 N 步执行一次验证 | 2000 |
| `--validation_epochs` | 每 N 个 epoch 执行一次验证 | 5 |
| `--validation_prompts` | 验证视频生成的提示词 | 英文提示词 |
| `--validation_paths` | 验证 I2V 的参考图像路径(仅 i2v/inpaint 模式) | `"asset/1.png"` |

**normal 模式示例**(T2V 验证):

```bash
  --validation_steps=2000 \
  --validation_epochs=5 \
  --validation_prompts="A brown dog shaking its head, sitting on a light-colored sofa in a cozy room. Behind the dog, there's a framed painting on a shelf, surrounded by pink flowers. The soft, warm lighting in the room creates a comfortable atmosphere."
```

**i2v/inpaint 模式示例**（I2V 验证）：

```bash
  --validation_paths "asset/1.png" \
  --validation_steps=2000 \
  --validation_epochs=5 \
  --validation_prompts="A brown dog shaking its head, sitting on a light-colored sofa in a cozy room. Behind the dog, there's a framed painting on a shelf, surrounded by pink flowers. The soft, warm lighting in the room creates a comfortable atmosphere."
```

**注意事项**:
- 验证视频会保存到 `output_dir` 目录中
- 多提示词验证格式:`--validation_prompts "prompt1" "prompt2" "prompt3"`
- `i2v` 或 `inpaint` 模式必须提供 `--validation_paths` 参数

### 3.5 使用 FSDP 训练

如果使用多卡且使用DeepSpeed-Zero-2的情况下显存不足，可以切换使用FSDP进行训练。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.1/train_distill_lora.py \
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
  --learning_rate=1e-05 \
  --learning_rate_critic=1e-06 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_distill_lora" \
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
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --train_mode="normal" \
  --low_vram
```

### 3.6 其他后端

#### 3.6.1 使用DeepSpeed-Zero-3进行训练

目前不太推荐使用 DeepSpeed Zero-3。在本仓库中，使用 FSDP 出错更少且更稳定。

已知 DeepSpeed Zero-3 与 PEFT 不兼容。

DeepSpeed Zero-3 适合高分辨率的 14B Wan。训练后，您可以使用以下命令获取最终模型：
```bash
python scripts/zero_to_bf16.py output_dir/checkpoint-{your-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

训练 shell 命令如下：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/wan2.1/train_distill_lora.py \
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
  --learning_rate=1e-05 \
  --learning_rate_critic=1e-06 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_distill_lora" \
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
  --train_mode="normal" \
  --low_vram
```

#### 3.6.2 不使用 DeepSpeed 与 FSDP 训练

**该方案并不被推荐，因为没有显存节约后端，容易造成显存不足**。这里仅提供训练Shell用于参考训练。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.1/train_distill_lora.py \
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
  --learning_rate=1e-05 \
  --learning_rate_critic=1e-06 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_distill_lora" \
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
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --train_mode="normal" \
  --low_vram
```

### 3.7 多机分布式训练

**适合场景**:超大规模数据集、需要更快的训练速度

#### 3.7.1 环境配置

假设有 2 台机器,每台 8 张 GPU:

**机器 0(Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
export MASTER_ADDR="192.168.1.100"  # Master 机器 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 机器总数
export NUM_PROCESS=16                # 总进程数 = 机器数 × 8
export RANK=0                        # 当前机器 rank(0 或 1)
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/wan2.1/train_distill_lora.py \
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
  --learning_rate=1e-05 \
  --learning_rate_critic=1e-06 \
  --seed=42 \
  --output_dir="output_dir_wan2.1_distill_lora" \
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
  --rank=64 \
  --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --train_mode="normal" \
  --low_vram
```

**机器 1(Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B/"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata.json"
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

- **网络要求**:
   - 推荐 RDMA/InfiniBand(高性能)
   - 无 RDMA 时添加环境变量:
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **数据同步**:所有机器必须能够访问相同的数据路径(NFS/共享存储)

---

## 四、推理测试

### 4.1 推理参数解析

**关键参数说明**:

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `GPU_memory_mode` | 显存管理模式,可选值见下表 | `model_group_offload` |
| `ulysses_degree` | Head 维度并行度,单卡时为 1 | 1 |
| `ring_degree` | Sequence 维度并行度,单卡时为 1 | 1 |
| `fsdp_dit` | 多卡推理时对 Transformer 使用 FSDP 节省显存 | `False` |
| `fsdp_text_encoder` | 多卡推理时对文本编码器使用 FSDP | `True` |
| `compile_dit` | 编译 Transformer 加速推理(固定分辨率下有效) | `False` |
| `model_name` | 模型路径 | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B` |
| `sampler_name` | 采样器类型:`Flow`、`Flow_Unipc`、`Flow_DPM++` | `Flow_Unipc` |
| `transformer_path` | 加载训练好的 Transformer 权重路径 | `None` 或基础模型权重 |
| `vae_path` | 加载训练好的 VAE 权重路径 | `None` |
| `lora_path` | LoRA 权重路径(蒸馏 LoRA 训练产出) | `output_dir_wan2.1_distill_lora/checkpoint-xxx/pytorch_lora_weights.safetensors` |
| `sample_size` | 生成视频分辨率 `[高度, 宽度]` | `[480, 832]` 或 `[832, 480]` |
| `video_length` | 生成视频帧数 | `81` |
| `fps` | 每秒帧数 | `16` |
| `weight_dtype` | 模型权重精度,不支持 bf16 的显卡使用 `torch.float16` | `torch.bfloat16` |
| `validation_image_start` | 图生视频的参考图像路径(I2V 模式) | `"asset/1.png"` |
| `prompt` | 正向提示词,描述生成内容 | `"一只棕色的狗摇着头..."` |
| `negative_prompt` | 负向提示词,避免生成的内容 | `"低分辨率,低画质..."` |
| `guidance_scale` | 引导强度(蒸馏模型通常使用 1.0) | 1.0 |
| `seed` | 随机种子,用于复现结果 | 43 |
| `num_inference_steps` | 推理步数(蒸馏模型通常为 4) | 4 |
| `lora_weight` | LoRA 权重强度 | 0.55 |
| `save_path` | 生成视频保存路径 | `samples/wan-videos-i2v` 或 `samples/wan-videos-t2v` |

**显存管理模式说明**:

| 模式 | 说明 | 显存占用 |
|------|------|---------|
| `model_full_load` | 整个模型加载到 GPU | 最高 |
| `model_full_load_and_qfloat8` | 全量加载 + FP8 量化 | 高 |
| `model_cpu_offload` | 使用后将模型卸载到 CPU | 中等 |
| `model_cpu_offload_and_qfloat8` | CPU 卸载 + FP8 量化 | 中低 |
| `model_group_offload` | 层组在 CPU/CUDA 间切换 | 低 |
| `sequential_cpu_offload` | 逐层卸载(速度最慢) | 最低 |

### 4.2 文生视频(T2V)推理

单卡推理运行如下命令:

```bash
python examples/wan2.1/predict_t2v.py
```

根据需求修改编辑 `examples/wan2.1/predict_t2v.py`,初次推理重点关注如下参数,如果对其他参数感兴趣,请查看上方的推理参数解析。

```python
# 根据显卡显存选择
GPU_memory_mode = "sequential_cpu_offload"
# 根据实际模型路径
model_name = "models/Diffusion_Transformer/Wan2.1-T2V-1.3B"  
# 基础模型权重路径(如果有训练过的全量权重)
transformer_path = None  
# LoRA 权重路径,如 "output_dir_wan2.1_distill_lora/checkpoint-xxx/pytorch_lora_weights.safetensors"
lora_path = None  
# 蒸馏模型通常使用 4 步
num_inference_steps = 4
# 蒸馏模型 guidance_scale 通常为 1.0
guidance_scale = 1.0
# LoRA 权重强度
lora_weight = 0.55
# 根据生成内容编写
prompt = "A brown dog shaking its head, sitting on a light-colored sofa in a cozy room. Behind the dog, there's a framed painting on a shelf, surrounded by pink flowers. The soft, warm lighting in the room creates a comfortable atmosphere."  
# ...
```

### 4.3 图生视频(I2V)推理

单卡推理运行如下命令:

```bash
python examples/wan2.1/predict_i2v.py
```

根据需求修改编辑 `examples/wan2.1/predict_i2v.py`,初次推理重点关注如下参数,如果对其他参数感兴趣,请查看上方的推理参数解析。

```python
# 根据显卡显存选择
GPU_memory_mode = "sequential_cpu_offload"
# 根据实际模型路径
model_name = "models/Diffusion_Transformer/Wan2.1-I2V-14B-480P"  
# 基础模型权重路径(如果有训练过的全量权重)
transformer_path = None  
# LoRA 权重路径,如 "output_dir_wan2.1_distill_lora/checkpoint-xxx/pytorch_lora_weights.safetensors"
lora_path = None  
# 蒸馏模型通常使用 4 步
num_inference_steps = 4
# 蒸馏模型 guidance_scale 通常为 1.0
guidance_scale = 1.0
# LoRA 权重强度
lora_weight = 0.55
# 图生视频的起始图像
validation_image_start = "asset/1.png"
# 根据生成内容编写
prompt = "A brown dog shaking its head, sitting on a light-colored sofa in a cozy room. Behind the dog, there's a framed painting on a shelf, surrounded by pink flowers. The soft, warm lighting in the room creates a comfortable atmosphere."  
# ...
```

### 4.4 多卡并行推理

**适合场景**:高分辨率生成、加速推理

#### 安装并行推理依赖

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### 配置并行策略

编辑 `examples/wan2.1/predict_t2v.py` 或 `examples/wan2.1/predict_i2v.py`:

```python
# 确保 ulysses_degree × ring_degree = 使用的 GPU 数
# 例如使用 2 张 GPU:
ulysses_degree = 2  # Head 维度并行
ring_degree = 1     # Sequence 维度并行
```

**配置原则**:
- `ulysses_degree` 必须能整除模型的 head 数
- `ring_degree` 是在 sequence 维度切分,会影响通信开销,在 head 能整除的情况下尽量不要用

**配置示例**:

| GPU 数量 | ulysses_degree | ring_degree | 说明 |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | 单 GPU |
| 4 | 4 | 1 | Head 并行 |
| 8 | 8 | 1 | Head 并行 |
| 8 | 4 | 2 | 混合并行 |

#### 运行多卡推理

```bash
torchrun --nproc-per-node=2 examples/wan2.1/predict_t2v.py
```

---

## 五、更多资源

- **官方 GitHub**:https://github.com/aigc-apps/VideoX-Fun
