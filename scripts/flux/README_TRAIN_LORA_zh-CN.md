# FLUX.1 LoRA 微调训练指南

本文档提供 FLUX.1 LoRA 微调训练的完整流程，包括环境配置、数据准备、多种分布式训练策略和推理测试。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径使用方案](#24-相对路径与绝对路径使用方案)
- [三、LoRA 训练](#三lora-训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始（DeepSpeed-Zero-2）](#32-快速开始deepspeed-zero-2)
  - [3.3 LoRA 专用参数解析](#33-lora-专用参数解析)
  - [3.4 使用 FSDP 训练](#34-使用-fsdp-训练)
  - [3.5 其他后端](#35-其他后端)
  - [3.6 多机分布式训练](#36-多机分布式训练)
- [四、推理测试](#四推理测试)
  - [4.1 推理参数解析](#41-推理参数解析)
  - [4.2 单卡推理](#42-单卡推理)
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

我们提供了一个测试的数据集，其中包含若干训练数据。

```bash
# 下载官方示例数据集
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

**相对路径格式**（示例格式）：
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

**绝对路径格式**：
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

**关键字段说明**：
- `file_path`：图片路径（相对或绝对路径）
- `text`：图片描述（英文提示词）
- `width` / `height`：图片宽高（**建议**提供以支持 bucket 训练；若不提供，训练时会自动读取，但在 OSS 等较慢系统中可能拖慢训练速度）
  - 可使用 `scripts/process_json_add_width_and_height.py` 为没有宽高字段的 JSON 文件添加，支持图片和视频
  - 用法：`python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Images-Demo/metadata.json --output_file datasets/X-Fun-Images-Demo/metadata_add_width_height.json`

### 2.4 相对路径与绝对路径使用方案

**使用相对路径**：

如果你的数据使用相对路径，训练脚本中这样配置：

```bash
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
```

**使用绝对路径**：

如果你的数据使用绝对路径，训练脚本中这样配置：

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 💡 **建议**：如果数据集较小且存储在本地，使用相对路径。如果数据集存储在外部存储（如 NAS、OSS）或多机共享，使用绝对路径。

---

## 三、LoRA 训练

### 3.1 下载预训练模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 FLUX.1 官方权重
modelscope download --model black-forest-labs/FLUX.1-dev --local_dir models/Diffusion_Transformer/FLUX.1-dev
```

### 3.2 快速开始（DeepSpeed-Zero-2）

如果你已经按照 **2.1 快速测试数据集** 下载了数据，按照 **3.1 下载预训练模型** 下载了权重，则可以直接复制运行快速开始的命令。

训练推荐使用 DeepSpeed-Zero-2 或 FSDP。这里以 DeepSpeed-Zero-2 为例。

DeepSpeed-Zero-2 和 FSDP 的区别在于是否对模型权重进行分片。**如果多卡使用 DeepSpeed-Zero-2 时显存不足**，可以切换为 FSDP。

```bash
export MODEL_NAME="models/Diffusion_Transformer/FLUX.1-dev"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/flux/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1024 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_flux_lora" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --rank=64 \
  --network_alpha=32 \
  --target_name="to_q,to_k,to_v,ff.0,ff.2,ff_context.0,ff_context.2" \
  --use_peft_lora \
  --uniform_sampling
```

### 3.3 LoRA 专用参数解析

**LoRA 关键参数说明**：

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--pretrained_model_name_or_path` | 预训练模型路径 | `models/Diffusion_Transformer/FLUX.1-dev` |
| `--train_data_dir` | 训练数据目录 | `datasets/internal_datasets/` |
| `--train_data_meta` | 训练数据元文件 | `datasets/internal_datasets/metadata.json` |
| `--train_batch_size` | 每张卡的批次大小 | 1 |
| `--image_sample_size` | 最大训练分辨率（自动分桶） | 1024 |
| `--gradient_accumulation_steps` | 梯度累积步数（等效增大 batch size） | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数 | 100 |
| `--checkpointing_steps` | 每 N 步保存检查点 | 50 |
| `--learning_rate` | 初始学习率（LoRA 推荐值） | 1e-04 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--seed` | 随机种子（可复现训练） | 42 |
| `--output_dir` | 输出目录 | `output_dir_flux_lora` |
| `--gradient_checkpointing` | 启用梯度检查点 | - |
| `--mixed_precision` | 混合精度：`fp16/bf16` | `bf16` |
| `--enable_bucket` | 启用桶训练（不中心裁剪，按分辨率分组后训练完整图像） | - |
| `--uniform_sampling` | 均匀时间步采样（推荐） | - |
| `--resume_from_checkpoint` | 恢复训练的路径，使用 `"latest"` 自动选择最新检查点 | None |
| `--rank` | LoRA 更新矩阵维度（rank 越高表达能力越强但显存占用越大） | 64 |
| `--network_alpha` | LoRA 更新矩阵缩放系数（通常为 rank 的一半或相同） | 32 |
| `--target_name` | 应用 LoRA 的组件/模块，用逗号分隔 | `to_q,to_k,to_v,ff.0,ff.2,ff_context.0,ff_context.2` |
| `--use_peft_lora` | 使用 PEFT 模块添加 LoRA（更节省显存） | - |

### 3.4 使用 FSDP 训练

**如果多卡使用 DeepSpeed-Zero-2 时显存不足**，可以切换为 FSDP。

> ✅ **推荐**：FSDP 在本仓库中经过充分测试，错误更少且更稳定。

```sh
export MODEL_NAME="models/Diffusion_Transformer/FLUX.1-dev"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap FluxSingleTransformerBlock,FluxTransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/flux/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1024 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_flux_lora" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --rank=64 \
  --network_alpha=32 \
  --target_name="to_q,to_k,to_v,ff.0,ff.2,ff_context.0,ff_context.2" \
  --use_peft_lora \
  --uniform_sampling
```

### 3.5 不使用 DeepSpeed 或 FSDP 训练

**不推荐此方法，因为没有显存优化的后端，可能导致显存不足**。仅供参考。

```sh
export MODEL_NAME="models/Diffusion_Transformer/FLUX.1-dev"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/flux/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1024 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_flux_lora" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --rank=64 \
  --network_alpha=32 \
  --target_name="to_q,to_k,to_v,ff.0,ff.2,ff_context.0,ff_context.2" \
  --use_peft_lora \
  --uniform_sampling
```

### 3.6 多机分布式训练

**适用场景**：大规模数据集，更快的训练速度

#### 3.6.1 环境配置

假设 2 台机器，每台 8 张 GPU：

**机器 0（Master）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/FLUX.1-dev"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master 机器 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 总机器数
export NUM_PROCESS=16                # 总进程数 = 机器数 × 8
export RANK=0                        # 当前机器 rank（0 或 1）
# NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1 用于无 RDMA 的多机环境
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/flux/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1024 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir_flux_lora" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --rank=64 \
  --network_alpha=32 \
  --target_name="to_q,to_k,to_v,ff.0,ff.2,ff_context.0,ff_context.2" \
  --use_peft_lora \
  --uniform_sampling
```

**机器 1（Worker）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/FLUX.1-dev"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
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

#### 3.6.2 多机训练注意事项

- **网络要求**：
   - 推荐使用 RDMA/InfiniBand（高性能）
   - 无 RDMA 时，添加环境变量：
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **数据同步**：所有机器必须能访问相同的数据路径（NFS/共享存储）

---

## 四、推理测试

### 4.1 推理参数解析

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `GPU_memory_mode` | GPU 显存管理模式，见下表 | `model_cpu_offload_and_qfloat8` |
| `ulysses_degree` | 头维度并行度，单卡为 1 | 1 |
| `ring_degree` | 序列维度并行度，单卡为 1 | 1 |
| `fsdp_dit` | 多卡推理时对 Transformer 使用 FSDP 以节省显存 | `False` |
| `fsdp_text_encoder` | 多卡推理时对文本编码器使用 FSDP | `False` |
| `compile_dit` | 编译 Transformer 以加速推理（固定分辨率时有效） | `False` |
| `model_name` | 模型路径 | `models/Diffusion_Transformer/FLUX.1-dev` |
| `sampler_name` | 采样器类型：`Flow`、`Flow_Unipc`、`Flow_DPM++` | `Flow` |
| `transformer_path` | 训练后的 Transformer 权重路径 | `None` |
| `vae_path` | 训练后的 VAE 权重路径 | `None` |
| `lora_path` | LoRA 权重路径 | `None` |
| `sample_size` | 生成图像分辨率 `[height, width]` | `[1344, 768]` |
| `weight_dtype` | 模型权重精度，不支持 bf16 的显卡使用 `torch.float16` | `torch.bfloat16` |
| `prompt` | 正向提示词，描述要生成的内容 | `"1girl, black_hair..."` |
| `negative_prompt` | 反向提示词，描述要避免的内容 | `"The video is not of a high quality..."` |
| `guidance_scale` | 引导强度 | 1.0 |
| `seed` | 随机种子，用于复现结果 | 43 |
| `num_inference_steps` | 推理步数 | 50 |
| `lora_weight` | LoRA 权重强度 | 0.55 |
| `save_path` | 生成图像保存路径 | `samples/flux-t2i` |

**GPU 显存管理模式**：

| 模式 | 说明 | 显存占用 |
|------|------|---------|
| `model_full_load` | 加载整个模型到 GPU | 最高 |
| `model_full_load_and_qfloat8` | 完整加载 + FP8 量化 | 高 |
| `model_cpu_offload` | 使用后卸载模型到 CPU | 中 |
| `model_cpu_offload_and_qfloat8` | CPU 卸载 + FP8 量化 | 中低 |
| `model_group_offload` | 层组在 CPU/CUDA 间切换 | 低 |
| `sequential_cpu_offload` | 逐层卸载（最慢） | 最低 |

### 4.2 单卡推理

使用以下命令运行单卡推理：

```bash
python examples/flux/predict_t2i.py
```

根据需要编辑 `examples/flux/predict_t2i.py`。首次推理时重点关注以下参数，其他参数见上方推理参数解析。

```python
# 根据 GPU 显存选择
GPU_memory_mode = "model_cpu_offload_and_qfloat8"
# 根据实际模型路径
model_name = "models/Diffusion_Transformer/FLUX.1-dev"  
# LoRA 权重路径，例如 "output_dir_flux_lora/checkpoint-xxx/lora_weights.safetensors"
lora_path = None
# LoRA 权重强度
lora_weight = 0.55
# 根据要生成的内容编写
prompt = "1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body"  
# ...
```

### 4.3 多卡并行推理

**适用场景**：高分辨率生成，更快的推理速度

#### 安装并行推理依赖

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### 配置并行策略

编辑 `examples/flux/predict_t2i.py`：

```python
# 确保 ulysses_degree × ring_degree = 使用的 GPU 数
# 例如使用 2 张 GPU：
ulysses_degree = 2  # 头维度并行
ring_degree = 1     # 序列维度并行
```

**配置原则**：
- `ulysses_degree` 必须能整除模型的头数。
- `ring_degree` 在序列维度切分，影响通信开销。头数能整除时尽量避免使用。

**示例配置**：

| GPU 数量 | ulysses_degree | ring_degree | 说明 |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | 单卡 |
| 4 | 4 | 1 | 头并行 |
| 8 | 8 | 1 | 头并行 |
| 8 | 4 | 2 | 混合并行 |

#### 运行多卡推理

```bash
torchrun --nproc-per-node=2 examples/flux/predict_t2i.py
```

## 五、更多资源

- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun