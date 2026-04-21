# Qwen-Image-Edit 全量参数训练指南

本文档提供 Qwen-Image-Edit Diffusion Transformer 全量参数训练的完整流程，包括环境配置、数据准备、分布式训练和推理测试。

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
  - [3.6 其他后端](#36-其他后端)
  - [3.7 多机分布式训练](#37-多机分布式训练)
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
modelscope download --dataset PAI/X-Fun-Images-Edit-Demo --local_dir ./datasets/X-Fun-Images-Edit-Demo
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

**Edit 模型数据格式**：

Edit 模型的 metadata.json 与普通版本不同，需要添加 `source_file_path` 字段。

- Qwen-Image-Edit：只需要一个文件在 source_file_path 中
- Qwen-Image-Edit-2509：需要一个或多个文件在 source_file_path 中

**相对路径格式**（Edit 模型）：
```json
[
    {
      "file_path": "train/00000001.jpg",
      "source_file_path": ["source/00000001.jpg"],
      "text": "一位年轻女子站在阳光明媚的海岸线上，身穿清爽的白色衬衫与裙子",
      "type": "image"
    },
    {
      "file_path": "train/00000002.jpg",
      "source_file_path": ["source/00000002.jpg"],
      "text": "一位紫发年轻女子站在海岸线上，背景是辽阔的大海",
      "type": "image"
    }
]
```

**关键字段说明**：
- `file_path`：目标图像路径（训练后要生成的图像）
- `source_file_path`：源图像路径数组（用于编辑的原始图像）
  - Edit 模型会基于 source_file_path 的图像，根据 text 描述进行编辑生成 file_path 的图像
  - Qwen-Image-Edit 只需要一个源文件，Qwen-Image-Edit-2509 支持多个源文件
- `text`：图像描述（提示词，描述期望生成的内容）
- `type`：数据类型（"image" 或 "video"）
- `width` / `height`：图片宽高（**最好提供**，用于分桶训练，如果不提供则自动在训练时读取，当数据存储在如oss这样的速度较慢的系统上时，可能会影响训练速度）。
  - 可以使用`scripts/process_json_add_width_and_height.py`文件对无width与height字段的json进行提取，支持处理图片与视频。
  - 使用方案为`python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Images-Demo/metadata.json --output_file datasets/X-Fun-Images-Demo/metadata_add_width_height.json`。

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

## 三、全量参数训练

### 3.1 下载预训练模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Qwen-Image-Edit 官方权重
modelscope download --model Qwen/Qwen-Image-Edit --local_dir models/Diffusion_Transformer/Qwen-Image-Edit
```

### 3.2 快速开始（DeepSpeed-Zero-2）

如果按照 **2.1 快速测试数据集下载数据** 与 **3.1 下载预训练模型下载权重**后，直接复制快速开始的启动指令进行启动。

**训练注意事项**：
- **不使用 DeepSpeed 的警告**：训练 Qwen-Image-Edit 时不使用 DeepSpeed 可能导致 GPU 显存不足，推荐使用 DeepSpeed-Zero-2 或 FSDP
- **DeepSpeed Zero-3 建议**：目前不强烈推荐使用 DeepSpeed Zero-3。在本仓库中，使用 FSDP 错误更少且更稳定
- 如果使用 DeepSpeed Zero-3，训练后需要使用以下命令获取最终模型：
  ```bash
  python scripts/zero_to_bf16.py output_dir/checkpoint-{步数} output_dir/checkpoint-{步数}-outputs --max_shard_size 80GB --safe_serialization
  ```

推荐使用DeepSpeed-Zero-2与FSDP方案进行训练。这里使用DeepSpeed-Zero-2为例配置shell文件。

本文中DeepSpeed-Zero-2与FSDP的差别在于是否对模型权重进行分片，**如果使用多卡且使用DeepSpeed-Zero-2的情况下显存不足**，可以切换使用FSDP进行训练。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-Edit"
export DATASET_NAME="datasets/X-Fun-Images-Edit-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Edit-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/qwenimage/train_edit.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1328 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_qwenimage_edit" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "." \
  --train_mode "qwen_image_edit"
```

### 3.3 训练常用参数解析

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--pretrained_model_name_or_path` | 预训练模型路径 | `models/Diffusion_Transformer/Qwen-Image-Edit` |
| `--train_data_dir` | 训练数据目录 | `datasets/internal_datasets/` |
| `--train_data_meta` | 训练数据元文件 | `datasets/internal_datasets/metadata.json` |
| `--train_batch_size` | 每批次样本数 | 1 |
| `--image_sample_size` | 最大训练分辨率，代码会自动分桶 | 1328 |
| `--gradient_accumulation_steps` | 梯度累积步数（等效增大 batch） | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数 | 100 |
| `--checkpointing_steps` | 每 N 步保存 checkpoint | 50 |
| `--learning_rate` | 初始学习率 | 2e-05 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--seed` | 随机种子 | 42 |
| `--output_dir` | 输出目录 | `output_dir` |
| `--gradient_checkpointing` | 激活重计算 | - |
| `--mixed_precision` | 混合精度：`fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 3e-2 |
| `--adam_epsilon` | AdamW epsilon 值 | 1e-10 |
| `--vae_mini_batch` | VAE 编码时的迷你批次大小 | 1 |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--enable_bucket` | 启用分桶训练，不裁剪图片，按分辨率分组训练整个图像 | - |
| `--random_hw_adapt` | 自动缩放图片到 `[512, image_sample_size]` 范围内的随机尺寸 | - |
| `--resume_from_checkpoint` | 恢复训练路径，使用 `"latest"` 自动选择最新 checkpoint | None |
| `--uniform_sampling` | 均匀采样 timestep | - |
| `--trainable_modules` | 可训练模块（`.` 表示所有模块） | `"."` |
| `--train_mode` | 训练模式：`qwen_image_edit` 用于 Qwen-Image-Edit，`qwen_image_edit_plus` 用于 Qwen-Image-Edit-2509 | `"qwen_image_edit"` |
| `--validation_steps` | 每 N 步执行一次验证 | 100 |
| `--validation_epochs` | 每 N 个epoch执行一次验证 | 500 |
| `--validation_prompts` | 验证时使用的提示词 | `"1girl, black_hair, ..."` |
| `--validation_image_paths` | 验证时使用的源图像路径（Edit 模型专用） | `"asset/8.jpg"` |

**random_hw_adapt 详细说明**：
- 当启用 `random_hw_adapt` 且 `image_sample_size=1024` 时，训练图像的分辨率范围是 `512x512` 到 `1024x1024`
- 与 `enable_bucket` 配合使用可以更灵活地处理不同分辨率的图像
- 例如：`random_hw_adapt=true`、`image_sample_size=1024`，则训练时图像会被随机缩放到 512 到 1024 之间的尺寸


### 3.4 训练验证

你可以配置验证参数，在训练过程中定期生成测试图像，以便监控训练进度和模型质量。

**验证参数配置**：

```bash
accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/qwenimage/train_edit.py \
  # ... (其他训练参数)
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body" \
  --validation_image_paths="asset/8.jpg"
```

**参数说明**：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--validation_steps` | 每 N 步执行一次验证。如果数据集较大，想节省验证时间，可以设置更大的值（如100或500） | 100 |
| `--validation_epochs` | 每 N 个epoch执行一次验证 | 500 |
| `--validation_prompts` | 验证图像生成的提示词。可以设置多个提示词，用空格分隔 | 多个空格分隔的提示词 |
| `--validation_image_paths` | 验证时使用的源图像路径（Edit 模型专用） | `asset/8.jpg` |

**注意事项**：
- 验证图像会保存到 `output_dir` 目录中
- 设置 `--validation_steps=1` 表示每一步都进行验证，可能会拖慢训练速度，可根据实际需求调整
- 多提示词验证格式：`--validation_prompts "prompt1" "prompt2" "prompt3"`
- Edit 模型验证时必须提供 `--validation_image_paths` 参数，指定用于编辑的源图像

### 3.5 使用 FSDP 训练

**如果使用多卡且使用DeepSpeed-Zero-2的情况下显存不足**，可以切换使用FSDP进行训练。

```sh
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-Edit"
export DATASET_NAME="datasets/X-Fun-Images-Edit-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Edit-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=QwenImageTransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/qwenimage/train_edit.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1328 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_qwenimage_edit" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "." \
  --train_mode "qwen_image_edit"
```

### 3.6 其他后端

#### 3.6.1 使用DeepSpeed-Zero-3进行训练

目前不太推荐使用 DeepSpeed Zero-3。在本仓库中，使用 FSDP 出错更少且更稳定。

DeepSpeed Zero-3：

训练完成后，您可以使用以下命令获取最终模型：

```sh
python scripts/zero_to_bf16.py output_dir/checkpoint-{our-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

执行命令为：
```sh
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-Edit"
export DATASET_NAME="datasets/X-Fun-Images-Edit-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Edit-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard scripts/qwenimage/train_edit.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1328 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "." \
  --train_mode "qwen_image_edit"
```

#### 3.6.2 不使用 DeepSpeed 与 FSDP 训练

**该方案并不被推荐，因为没有显存节约后端，容易造成显存不足**。这里仅提供训练Shell用于参考训练。

```sh
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-Edit"
export DATASET_NAME="datasets/X-Fun-Images-Edit-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Edit-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/qwenimage/train_edit.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1328 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_qwenimage_edit" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "." \
  --train_mode "qwen_image_edit"
```

### 3.7 多机分布式训练

**适合场景**：超大规模数据集、需要更快的训练速度

#### 3.7.1 环境配置

假设有 2 台机器，每台 8 张 GPU：

**机器 0（Master）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-Edit"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
export MASTER_ADDR="192.168.1.100"  # Master 机器 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 机器总数
export NUM_PROCESS=16                # 总进程数 = 机器数 × 8
export RANK=0                        # 当前机器 rank（0 或 1）
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/qwenimage/train_edit.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --image_sample_size=1328 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_qwenimage_edit" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "." \
  --train_mode "qwen_image_edit"
```

**机器 1（Worker）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-Edit"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
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

## 四、推理测试

### 4.1 推理参数解析

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `GPU_memory_mode` | 显存管理模式，可选值见下表 | `model_group_offload` |
| `ulysses_degree` | Head 维度并行度，单卡时为 1 | 1 |
| `ring_degree` | Sequence 维度并行度，单卡时为 1 | 1 |
| `fsdp_dit` | 多卡推理时对 Transformer 使用 FSDP 节省显存 | `False` |
| `fsdp_text_encoder` | 多卡推理时对文本编码器使用 FSDP | `False` |
| `compile_dit` | 编译 Transformer 加速推理（固定分辨率下有效） | `False` |
| `enable_teacache` | 启用 TeaCache 加速推理 | `True` |
| `teacache_threshold` | TeaCache 阈值，建议 0.05~0.30，越大越快但质量可能下降 | 0.25 |
| `num_skip_start_steps` | 推理开始跳过的步数，减少对生成质量的影响 | 5 |
| `teacache_offload` | 将 TeaCache 张量卸载到 CPU 节省显存 | `False` |
| `cfg_skip_ratio` | 跳过部分 CFG 步数加速推理，建议 0.00~0.25 | 0 |
| `model_name` | 模型路径 | `models/Diffusion_Transformer/Qwen-Image-Edit` |
| `sampler_name` | 采样器类型：`Flow`、`Flow_Unipc`、`Flow_DPM++` | `Flow` |
| `transformer_path` | 加载训练好的 Transformer 权重路径 | `None` |
| `vae_path` | 加载训练好的 VAE 权重路径 | `None` |
| `lora_path` | LoRA 权重路径 | `None` |
| `sample_size` | 生成图像分辨率 `[高度, 宽度]` | `[1344, 768]` |
| `weight_dtype` | 模型权重精度，不支持 bf16 的显卡使用 `torch.float16` | `torch.bfloat16` |
| `prompt` | 正向提示词，描述生成内容 | `"1girl, black_hair..."` |
| `negative_prompt` | 负向提示词，避免生成的内容 | `" "` |
| `guidance_scale` | 引导强度 | 4.0 |
| `seed` | 随机种子，用于复现结果 | 43 |
| `num_inference_steps` | 推理步数 | 50 |
| `lora_weight` | LoRA 权重强度 | 0.55 |
| `save_path` | 生成图像保存路径 | `samples/qwenimage-t2i` |

**显存管理模式说明**：

| 模式 | 说明 | 显存占用 |
|------|------|---------|
| `model_full_load` | 整个模型加载到 GPU | 最高 |
| `model_full_load_and_qfloat8` | 全量加载 + FP8 量化 | 高 |
| `model_cpu_offload` | 使用后将模型卸载到 CPU | 中等 |
| `model_cpu_offload_and_qfloat8` | CPU 卸载 + FP8 量化 | 中低 |
| `model_group_offload` | 层组在 CPU/CUDA 间切换 | 低 |
| `sequential_cpu_offload` | 逐层卸载（速度最慢） | 最低 |

### 4.2 单卡推理

#### 快速开始

单卡推理运行如下命令：

```bash
python examples/qwenimage/predict_t2i_edit.py
```

根据需求修改编辑 `examples/qwenimage/predict_t2i_edit.py`，初次推理重点关注如下参数，如果对其他参数感兴趣，请查看上方的推理参数解析。

```python
# 根据显卡显存选择
GPU_memory_mode = "model_group_offload"
# 根据实际模型路径
model_name = "models/Diffusion_Transformer/Qwen-Image-Edit"  
# 训练好的权重路径，如 "output_dir_qwenimage_edit/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = None  
# 根据生成内容编写
prompt = "一位年轻女子站在阳光明媚的海岸线上，身穿清爽的白色衬衫与裙子"  
# ...
```

### 4.3 多卡并行推理

**适合场景**：高分辨率生成、加速推理

#### 安装并行推理依赖

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### 配置并行策略

编辑 `examples/qwenimage/predict_t2i_edit.py`：

```python
# 确保 ulysses_degree × ring_degree = GPU 数量
# 例如使用 2 张 GPU：
ulysses_degree = 2  # Head 维度并行
ring_degree = 1     # Sequence 维度并行
```

**配置原则**：
- `ulysses_degree` 必须能整除模型的head数。
- `ring_degree` 会在sequence上切分，影响通信开销，在head数能切分的时候尽量不用。

**示例配置**：

| GPU 数量 | ulysses_degree | ring_degree | 说明 |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | 单卡 |
| 4 | 4 | 1 | Head 并行 |
| 8 | 8 | 1 | Head 并行 |
| 8 | 4 | 2 | 混合并行 |

#### 运行多卡推理

```bash
torchrun --nproc-per-node=2 examples/qwenimage/predict_t2i_edit.py
```

## 五、更多资源

- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun