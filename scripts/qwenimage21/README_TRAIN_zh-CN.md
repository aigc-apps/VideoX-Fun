# Qwen-Image 2.1 全参数训练指南

本文档提供 Qwen-Image 2.1 Diffusion Transformer 全参数训练的完整流程,包括环境配置、数据准备、分布式训练与推理测试。

---

## 目录
- [1. 环境配置](#1-环境配置)
- [2. 数据准备](#2-数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径的用法](#24-相对路径与绝对路径的用法)
- [3. 全参数训练](#3-全参数训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始(DeepSpeed-Zero-2)](#32-快速开始deepspeed-zero-2)
  - [3.3 常用训练参数](#33-常用训练参数)
  - [3.4 训练验证](#34-训练验证)
  - [3.5 使用 FSDP 训练](#35-使用-fsdp-训练)
  - [3.6 其他后端](#36-其他后端)
  - [3.7 多机分布式训练](#37-多机分布式训练)
- [4. 推理测试](#4-推理测试)
  - [4.1 推理参数解析](#41-推理参数解析)
  - [4.2 单卡推理](#42-单卡推理)
  - [4.3 多卡并行推理](#43-多卡并行推理)
- [5. 更多资源](#5-更多资源)

---

## 1. 环境配置

**方式一:使用 requirements.txt**

```bash
pip install -r requirements.txt
```

**方式二:手动安装依赖**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

**方式三:使用 Docker**

使用 Docker 时,请确保机器上已正确安装 GPU 驱动与 CUDA 环境,然后执行以下命令:

```
# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# 进入镜像
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 2. 数据准备

### 2.1 快速测试数据集

我们提供了一个包含若干训练样本的测试数据集。

```bash
# 下载官方 demo 数据集
modelscope download --dataset PAI/X-Fun-Images-Demo --local_dir ./datasets/X-Fun-Images-Demo
```

### 2.2 数据集结构

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 image001.jpg
│   │   ├── 📄 image002.png
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 2.3 metadata.json 格式

**相对路径格式**(示例):
```json
[
  {
    "file_path": "train/image001.jpg",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
    "width": 1024,
    "height": 1024
  },
  {
    "file_path": "train/image002.png",
    "text": "Portrait of a young woman, studio lighting, high quality",
    "width": 1024,
    "height": 1024
  }
]
```

**绝对路径格式**:
```json
[
  {
    "file_path": "/mnt/data/images/sunset.jpg",
    "text": "A beautiful sunset over the ocean",
    "width": 1024,
    "height": 1024
  }
]
```

**关键字段说明**:
- `file_path`:图像路径(相对或绝对)
- `text`:图像描述(英文 prompt)
- `width` / `height`:图像尺寸(**建议**提供以便 bucket 训练;若不提供,训练时会自动读取,当数据存放在 OSS 等慢速系统上时可能拖慢训练)
  - 可使用 `scripts/process_json_add_width_and_height.py` 为缺少这些字段的 JSON 文件补充 width/height,同时支持图像与视频
  - 用法:`python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Images-Demo/metadata.json --output_file datasets/X-Fun-Images-Demo/metadata_add_width_height.json`

> 💡 训练图像以 RGB 读入,并在 VAE 编码前自动合成到一层不透明 alpha 通道上,因此你**无需**提供 RGBA 数据。

### 2.4 相对路径与绝对路径的用法

**相对路径**:

如果数据使用相对路径,按如下方式配置训练脚本:

```bash
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
```

**绝对路径**:

如果数据使用绝对路径,按如下方式配置训练脚本:

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata_add_width_height.json"
```

> 💡 **建议**:数据集较小且存放在本地时使用相对路径;数据集存放在外部存储(如 NAS、OSS)或需跨多台机器共享时使用绝对路径。

---

## 3. 全参数训练

### 3.1 下载预训练模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Qwen-Image 2.1 官方权重
modelscope download --model Qwen/Qwen-Image-2.1 --local_dir models/Diffusion_Transformer/Qwen-Image-2.1
```

> 💡 如果 ModelScope id 与上面不同,请改为官方 Qwen-Image-2.1 发布页对应的 id。你也可以将 `--pretrained_model_name_or_path` 指向任意符合 diffusers 布局、且包含 `transformer/`、`vae/`、`text_encoder/`、`processor/`、`scheduler/` 子目录的本地目录。

### 3.2 快速开始(DeepSpeed-Zero-2)

如果你已按 **2.1 快速测试数据集** 下载数据、并按 **3.1 下载预训练模型** 下载权重,可直接复制运行下面的快速开始命令。

训练推荐使用 DeepSpeed-Zero-2 或 FSDP,这里以 DeepSpeed-Zero-2 为例。

DeepSpeed-Zero-2 与 FSDP 的区别在于是否对模型权重做分片。**如果多卡使用 DeepSpeed-Zero-2 时显存不足**,可切换为 FSDP。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 与 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境。
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/qwenimage21/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1024 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_qwenimage21" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "."
```

### 3.3 常用训练参数

**关键参数说明**:

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--pretrained_model_name_or_path` | 预训练模型路径 | `models/Diffusion_Transformer/Qwen-Image-2.1` |
| `--train_data_dir` | 训练数据目录 | `datasets/X-Fun-Images-Demo/` |
| `--train_data_meta` | 训练数据元信息文件 | `datasets/X-Fun-Images-Demo/metadata_add_width_height.json` |
| `--train_batch_size` | 每个 batch 的样本数 | 1 |
| `--image_sample_size` | 最大训练分辨率,自动分桶 | 1024 |
| `--gradient_accumulation_steps` | 梯度累积步数(等效更大 batch) | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数 | 100 |
| `--checkpointing_steps` | 每 N 步保存一次 checkpoint | 50 |
| `--learning_rate` | 初始学习率 | 2e-05 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率 warmup 步数 | 100 |
| `--seed` | 随机种子 | 42 |
| `--output_dir` | 输出目录 | `output_dir_qwenimage21` |
| `--gradient_checkpointing` | 启用激活重计算 | - |
| `--mixed_precision` | 混合精度:`fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 3e-2 |
| `--adam_epsilon` | AdamW epsilon 值 | 1e-10 |
| `--vae_mini_batch` | VAE 编码的 mini-batch 大小 | 1 |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--enable_bucket` | 启用 bucket 训练:按分辨率分组训练整图,不做中心裁剪 | - |
| `--random_hw_adapt` | 将图像自动缩放到 `[512, image_sample_size]` 区间内的随机尺寸 | - |
| `--resume_from_checkpoint` | 从 checkpoint 路径恢复训练,使用 `"latest"` 自动选择最新 | None |
| `--uniform_sampling` | 均匀时间步采样 | - |
| `--trainable_modules` | 可训练模块(`"."` 表示全部模块) | `"."` |
| `--tokenizer_max_length` | 送入 Qwen3-VL 文本编码器的最大 prompt token 长度 | 1024 |
| `--validation_steps` | 每 N 步执行一次验证 | 100 |
| `--validation_epochs` | 每 N 个 epoch 执行一次验证 | 100 |
| `--validation_prompts` | 验证时使用的 prompt | `"1girl, black_hair, ..."` |


### 3.4 训练验证

你可以配置验证参数,在训练过程中周期性生成测试图像,以便监控训练进度与模型质量。

**验证参数**:

```bash
accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/qwenimage21/train.py \
  # ... (其他训练参数)
  --validation_steps=100 \
  --validation_epochs=100 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body"
```

**参数说明**:

| 参数 | 说明 | 推荐值 |
|-----------|-------------|-------------------|
| `--validation_steps` | 每 N 步执行一次验证。若数据集较大、想节省验证时间,可设更大的值(如 100 或 500) | 100 |
| `--validation_epochs` | 每 N 个 epoch 执行一次验证 | 100 |
| `--validation_prompts` | 用于验证生图的 prompt。多个 prompt 用空格分隔的字符串表示 | 空格分隔的 prompt 字符串 |

**注意**:
- 验证图像会保存到 `output_dir` 目录
- 设置 `--validation_steps=1` 表示每步都验证,可能拖慢训练,请按需调整
- 多 prompt 验证用法:`--validation_prompts "prompt1" "prompt2" "prompt3"`


### 3.5 使用 FSDP 训练

**如果多卡使用 DeepSpeed-Zero-2 时显存不足**,可切换为 FSDP。注意 Qwen-Image 2.1 需要 wrap 的 transformer 层类名为 `QwenImage21TransformerBlock`。

```sh
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 与 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境。
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=QwenImage21TransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/qwenimage21/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1024 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_qwenimage21" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "."
```

### 3.6 其他后端

#### 3.6.1 使用 DeepSpeed-Zero-3 训练

目前不太推荐 DeepSpeed Zero-3。在本仓库中,使用 FSDP 报错更少、更稳定。

DeepSpeed Zero-3:

训练结束后,可用以下命令得到最终模型:

```sh
python scripts/zero_to_bf16.py output_dir/checkpoint-{our-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

训练 shell 命令:
```sh
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 与 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境。
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/qwenimage21/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1024 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_qwenimage21" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "."
```

#### 3.6.2 不使用 DeepSpeed 或 FSDP 训练

**不推荐该方式,因为缺少省显存的后端,很容易导致显存溢出(OOM)**。此处仅供参考。

```sh
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 与 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境。
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/qwenimage21/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1024 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_qwenimage21" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "."
```

### 3.7 多机分布式训练

**适用于**:超大规模数据集、更快的训练速度

#### 3.7.1 环境配置

假设有 2 台机器、每台 8 卡:

**机器 0(Master)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # 主机器 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 机器总数
export NUM_PROCESS=16                # 总进程数 = 机器数 × 8
export RANK=0                        # 当前机器 rank(0 或 1)
# NCCL_IB_DISABLE=1 与 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境。
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/qwenimage21/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1024 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_qwenimage21" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "."
```

**机器 1(Worker)**:
```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # 与 Master 相同
export MASTER_PORT=10086
export WORLD_SIZE=2
export NUM_PROCESS=16
export RANK=1  # 注意这里是 1
# NCCL_IB_DISABLE=1 与 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境。
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# 使用与机器 0 相同的 accelerate launch 命令
```

#### 3.7.2 多机训练注意事项

- **网络要求**:
   - 推荐 RDMA/InfiniBand(高性能)
   - 无 RDMA 时,添加环境变量:
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **数据同步**:所有机器必须能访问相同的数据路径(NFS/共享存储)

## 4. 推理测试

> ℹ️ **支持多卡(仅 Ulysses)**:Qwen-Image 2.1 支持 Ulysses(head 并行)序列并行——设 `ulysses_degree > 1` 即可把单张图的去噪切分到多卡(降低单卡延迟与激活显存,且与单卡数学等价)。`ring_degree` **必须保持为 1**:ring attention 会旋转 KV 分块,无法表达 2.1 的 block-causal 掩码与 prefix KV cache。`ulysses_degree` 必须能整除 `num_attention_heads`(32)。详见 [4.3 多卡并行推理](#43-多卡并行推理)。单卡显存不足时,也可使用下方的显存管理模式(offload / FP8)。

### 4.1 推理参数解析

**关键参数说明**(见 `examples/qwenimage21/predict_t2i.py`):

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `GPU_memory_mode` | 显存管理模式,可选项见下表 | `model_full_load` |
| `ulysses_degree` | Ulysses(head)并行度。需整除 `num_attention_heads`(32),即 1/2/4/8/16/32;`>1` 时把单图切分到多卡 | 1 |
| `ring_degree` | sequence(ring)并行度。**必须保持为 1**——ring 无法表达 block-causal 掩码与 prefix KV cache | 1 |
| `compile_dit` | 编译 Transformer 以加速推理(固定分辨率下有效) | `False` |
| `model_name` | 模型路径 | `models/Diffusion_Transformer/Qwen-Image-2.1` |
| `sampler_name` | 采样器类型。Qwen-Image 2.1 为 flow-matching,仅支持 `Flow` | `Flow` |
| `transformer_path` | 加载已训练 Transformer 权重的路径 | `None` |
| `vae_path` | 加载已训练 VAE 权重的路径 | `None` |
| `lora_path` | LoRA 权重路径 | `None` |
| `sample_size` | 生成图像分辨率 `[height, width]`,会向下取整到 32 的倍数;为 `None` 时回退到 pipeline 默认方图 | `[1024, 1024]` |
| `use_kv_cache` | 在第一个去噪步后缓存文本/条件的 key-value 以加速推理 | `True` |
| `weight_dtype` | 模型权重精度,不支持 bf16 的 GPU 请用 `torch.float16` | `torch.bfloat16` |
| `prompts` | 描述生成内容的正向 prompt | `["a young girl ..."]` |
| `negative_prompt` | 需要规避内容的负向 prompt | `" "` |
| `guidance_scale` | 引导强度(以 `true_cfg_scale` 传入 pipeline) | 1.0 |
| `seed` | 随机种子,用于复现结果 | 43 |
| `num_inference_steps` | 推理步数 | 40 |
| `lora_weight` | LoRA 权重强度 | 1 |
| `save_path` | 生成图像保存路径 | `samples/qwenimage21-t2i` |

**显存管理模式说明**:

| 模式 | 说明 | 显存占用 |
|------|------|---------|
| `model_full_load` | 将整个模型加载到 GPU | 最高 |
| `model_full_load_and_qfloat8` | 全量加载 + FP8 量化 | 高 |
| `model_cpu_offload` | 用完后将模型 offload 到 CPU | 中 |
| `model_cpu_offload_and_qfloat8` | CPU offload + FP8 量化 | 中低 |
| `model_group_offload` | 以层组为单位在 CPU/CUDA 间切换 | 低 |
| `sequential_cpu_offload` | 逐层 offload(最慢) | 最低 |

### 4.2 单卡推理

#### 快速开始

运行以下命令进行单卡推理:

```bash
python examples/qwenimage21/predict_t2i.py
```

按需编辑 `examples/qwenimage21/predict_t2i.py`。首次推理重点关注以下参数,其余参数参见上面的推理参数解析。

```python
# 根据 GPU 显存选择
GPU_memory_mode = "model_full_load"
# 根据实际模型路径填写
model_name = "models/Diffusion_Transformer/Qwen-Image-2.1"
# 已训练权重路径,例如 "output_dir_qwenimage21/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = None
# 根据生成内容填写
prompts = ["a young girl with flowing long hair, wearing a white halter dress"]
# ...
```

### 4.3 多卡并行推理

**适合场景**:高分辨率生成、加速单图推理。Qwen-Image 2.1 按注意力 **head** 切分到多卡(Ulysses 序列并行):经过 all-to-all 后每张卡持有"完整序列 + 部分 head",因此 block-causal 多趟 prefill 与 prefix KV cache 逻辑原样运行,输出与单卡**数学等价**。

#### 安装并行推理依赖

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### 配置并行策略

编辑 `examples/qwenimage21/predict_t2i.py`:

```python
# ulysses_degree × ring_degree = GPU 数量;Qwen-Image 2.1 的 ring_degree 必须保持为 1
# 例如使用 2 张 GPU:
ulysses_degree = 2  # Head(Ulysses)并行
ring_degree = 1     # 必须为 1
```

**配置原则**:
- `ulysses_degree` 必须能整除 `num_attention_heads`(32),即取 1/2/4/8/16/32 之一。
- `ring_degree` 必须保持为 **1**:ring attention 会旋转 KV 分块,无法表达 2.1 的 block-causal 掩码与 prefix KV cache(脚本内已加断言)。
- joint(文本 + 图像)序列会在内部 pad 到 `ulysses_degree` 的倍数,pad 位置的 key 会被置为 invalid,因此结果与单卡完全一致。
- Ulysses 在每张卡上都复制模型权重(切分的是激活而非参数)。显存吃紧时可同时设 `fsdp_dit = True` 对 Transformer 做权重分片。

**示例配置**:

| GPU 数量 | ulysses_degree | ring_degree | 说明 |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | 单卡 |
| 2 | 2 | 1 | Head 并行 |
| 4 | 4 | 1 | Head 并行 |
| 8 | 8 | 1 | Head 并行 |

#### 运行多卡推理

```bash
torchrun --nproc-per-node=2 examples/qwenimage21/predict_t2i.py
```

将 `--nproc-per-node` 设为与 `ulysses_degree` 相同。

## 5. 更多资源

- **官方 GitHub**: https://github.com/aigc-apps/VideoX-Fun
