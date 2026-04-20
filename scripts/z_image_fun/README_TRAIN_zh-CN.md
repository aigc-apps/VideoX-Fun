# Z-Image Turbo Control 全量参数训练指南

本文档提供 Z-Image Turbo Control 模型训练的完整流程,包括环境配置、数据准备、分布式训练和推理测试。

在 Z-Image 训练中,可以选择使用 DeepSpeed 或 FSDP 来节省大量显存。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径使用方案](#24-相对路径与绝对路径使用方案)
- [三、Control 训练](#三control-训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始(DeepSpeed-Zero-2)](#32-快速开始deepspeed-zero-2)
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

**方式 3:使用docker**

使用docker的情况下,请保证机器中已经正确安装显卡驱动与CUDA环境,然后以此执行以下命令:

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 二、数据准备

### 2.1 快速测试数据集

我们提供了一个测试的数据集，其中包含若干训练数据以及对应的控制文件。

```bash
# 下载官方示例数据集
modelscope download --dataset PAI/X-Fun-Images-Controls-Demo --local_dir ./datasets/X-Fun-Images-Controls-Demo
```

### 2.2 数据集结构

```
📦 datasets/
├── 📂 my_dataset/
│   ├── 📂 train/
│   │   ├── 📄 image001.jpg
│   │   ├── 📄 image002.png
│   │   └── 📄 ...
│   ├── 📂 control/
│   │   ├── 📄 image001.jpg
│   │   ├── 📄 image002.png
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 2.3 metadata.json 格式

Control 模式的 metadata.json 与普通 Z-Image 的 json 略有不同，需要额外添加 `control_file_path` 字段。

建议使用 [DWPose](https://github.com/IDEA-Research/DWPose) 等工具生成控制文件（如姿态估计图）。

**相对路径格式**（示例格式）：
```json
[
    {
      "file_path": "train/image001.jpg",
      "control_file_path": "control/image001.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "width": 1024,
      "height": 1024,
      "type": "image"
    },
    {
      "file_path": "train/image002.jpg",
      "control_file_path": "control/image002.jpg",
      "text": "A beautiful woman standing on the beach at sunset.",
      "width": 1328,
      "height": 1328,
      "type": "image"
    }
]
```

**绝对路径格式**：
```json
[
    {
      "file_path": "/mnt/data/images/image001.jpg",
      "control_file_path": "/mnt/data/controls/image001.jpg",
      "text": "A group of young men in suits and sunglasses.",
      "width": 1024,
      "height": 1024,
      "type": "image"
    }
]
```

**关键字段说明**：
- `file_path`：原图路径（相对或绝对路径）
- `control_file_path`：控制文件路径，如姿态图、边缘检测图等
- `text`：图片描述（英文提示词）
- `width` / `height`：图片宽高（**最好提供**，用于分桶训练，如果不提供则自动在训练时读取）
- `type`：数据类型，图像数据为 `"image"`

> 💡 **提示**：可以使用 `scripts/process_json_add_width_and_height.py` 对无 width 与 height 字段的 json 进行提取。

### 2.4 相对路径与绝对路径使用方案

**方案 1:使用相对路径(推荐)**

当数据路径不固定,或需要在不同机器上训练时,推荐使用相对路径。

在 `metadata.json` 中配置相对路径,然后在训练脚本中通过 `--train_data_dir` 指定数据集根目录:

```json
[
  {
    "file_path": "train/image001.jpg",
    "control_file_path": "control/image001.jpg",
    "text": "A group of young men in suits and sunglasses are walking down a city street.",
    "width": 1024,
    "height": 1024,
    "type": "image"
  }
]
```

训练时会自动在 `--train_data_dir` 下寻找相对路径对应的文件。

**方案 2:使用绝对路径**

如果数据集路径固定,可以直接在 `metadata.json` 中配置绝对路径:

```json
[
  {
    "file_path": "/mnt/data/images/image001.jpg",
    "control_file_path": "/mnt/data/controls/image001.jpg",
    "text": "A group of young men in suits and sunglasses.",
    "width": 1024,
    "height": 1024,
    "type": "image"
  }
]
```

使用绝对路径时,`--train_data_dir` 参数仅作为默认路径,实际会优先使用 json 中的绝对路径。

---

## 三、Control 训练

### 3.1 下载预训练模型

**ModelScope 下载**：

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer
mkdir -p models/Personalized_Model

# 下载 Z-Image 官方权重
modelscope download --model Tongyi-MAI/Z-Image --local_dir models/Diffusion_Transformer/Z-Image

# 下载 Z-Image-Turbo 快速推理版
modelscope download --model Tongyi-MAI/Z-Image-Turbo --local_dir models/Diffusion_Transformer/Z-Image-Turbo

# 下载 Z-Image Control 预训练权重
modelscope download --model PAI/Z-Image-Fun-Controlnet-Union-2.1 --local_dir models/Personalized_Model/Z-Image-Fun-Controlnet-Union-2.1

# 下载 Z-Image-Turbo Control 预训练权重
modelscope download --model PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1 --local_dir models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.1
```

**HuggingFace 下载**：

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer
mkdir -p models/Personalized_Model

# 下载 Z-Image 官方权重
hf download Tongyi-MAI/Z-Image --local-dir models/Diffusion_Transformer/Z-Image

# 下载 Z-Image-Turbo 快速推理版
hf download Tongyi-MAI/Z-Image-Turbo --local-dir models/Diffusion_Transformer/Z-Image-Turbo

# 下载 Z-Image Control 预训练权重
hf download alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1 --local-dir models/Personalized_Model/Z-Image-Fun-Controlnet-Union-2.1

# 下载 Z-Image-Turbo Control 预训练权重
hf download alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 --local-dir models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.1
```

### 3.2 快速开始（DeepSpeed-Zero-2）

推荐使用 DeepSpeed-Zero-2 或 FSDP 方案进行训练，可以节省大量显存。

如果按照 **2.1 快速测试数据集下载数据** 与 **3.1 下载预训练模型下载权重**后，直接复制以下启动指令进行启动。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Z-Image-Turbo"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/z_image_fun/train_control.py \
  --config_path="config/z_image/z_image_control_2.1.yaml" \
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
  --output_dir="output_dir_z_image_control" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --transformer_path="models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors" \
  --trainable_modules "control"
```

### 3.3 训练常用参数解析

**关键参数说明**:

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--pretrained_model_name_or_path` | 预训练模型路径 | `models/Diffusion_Transformer/Z-Image-Turbo` |
| `--train_data_dir` | 训练数据目录 | `datasets/internal_datasets/` |
| `--train_data_meta` | 训练数据元文件 | `datasets/internal_datasets/metadata.json` |
| `--train_batch_size` | 每批次样本数 | 1 |
| `--image_sample_size` | 最大训练分辨率,代码会自动分桶 | 1328 |
| `--gradient_accumulation_steps` | 梯度累积步数(等效增大 batch) | 1 |
| `--dataloader_num_workers` | DataLoader 子进程数 | 8 |
| `--num_train_epochs` | 训练 epoch 数 | 100 |
| `--checkpointing_steps` | 每 N 步保存 checkpoint | 50 |
| `--learning_rate` | 初始学习率 | 2e-05 |
| `--lr_scheduler` | 学习率调度器 | `constant_with_warmup` |
| `--lr_warmup_steps` | 学习率预热步数 | 100 |
| `--seed` | 随机种子 | 42 |
| `--output_dir` | 输出目录 | `output_dir_z_image_control` |
| `--gradient_checkpointing` | 激活重计算 | - |
| `--mixed_precision` | 混合精度:`fp16/bf16` | `bf16` |
| `--adam_weight_decay` | AdamW 权重衰减 | 3e-2 |
| `--adam_epsilon` | AdamW epsilon 值 | 1e-10 |
| `--vae_mini_batch` | VAE 编码时的迷你批次大小 | 1 |
| `--max_grad_norm` | 梯度裁剪阈值 | 0.05 |
| `--enable_bucket` | 启用分桶训练,不裁剪图片,按分辨率分组训练整个图像 | - |
| `--uniform_sampling` | 均匀采样 timestep | - |
| `--transformer_path` | 加载预训练的 Control 权重 | `models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors` |
| `--trainable_modules` | 可训练模块(`"control"` 表示仅训练 control 模块) | `"control"` |
| `--validation_steps` | 每 N 步执行一次验证 | 50 |
| `--validation_epochs` | 每 N 个epoch执行一次验证 | 500 |
| `--validation_prompts` | 验证时使用的提示词 | `"1girl, black_hair, ..."` |

### 3.4 训练验证

训练过程中可以设置验证参数来定期评估模型效果：

```bash
  --validation_paths "asset/pose.jpg" \
  --validation_steps=50 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, earrings, freckles, grey_background, jewelry, lips, long_hair, looking_at_viewer, nose, piercing, realistic, red_lips, solo, upper_body"
```

**验证参数说明**：
- `--validation_paths`：控制图像路径，用于验证时输入控制条件（支持多张图像）
- `--validation_steps`：每隔多少 step 进行一次验证（与 `--validation_epochs` 任一满足即触发）
- `--validation_epochs`：每隔多少 epoch 进行一次验证（与 `--validation_steps` 任一满足即触发）
- `--validation_prompts`：验证时使用的提示词（支持多个提示词，与 `--validation_paths` 一一对应）

验证结果会保存在 `{output_dir}/sample/` 目录下，文件名格式为 `sample-{global_step}-rank{process_index}-image-{index}.jpg`。

### 3.5 使用 FSDP 训练

如果 DeepSpeed-Zero-2 显存不足，可以切换使用 FSDP 进行训练：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Z-Image-Turbo"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap BaseZImageTransformerBlock,ZImageControlTransformerBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/z_image_fun/train_control.py \
  --config_path="config/z_image/z_image_control_2.1.yaml" \
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
  --output_dir="output_dir_z_image_control" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --transformer_path="models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors" \
  --trainable_modules "control"
```

### 3.6 其他后端

#### 3.6.1 不使用 DeepSpeed 与 FSDP 训练

不使用 DeepSpeed 或 FSDP 可能会导致显存不足，仅建议在显存充足的情况下使用：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Z-Image-Turbo"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/z_image_fun/train_control.py \
  --config_path="config/z_image/z_image_control_2.1.yaml" \
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
  --output_dir="output_dir_z_image_control" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --transformer_path="models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors" \
  --trainable_modules "control"
```

### 3.7 多机分布式训练

**适合场景**：超大规模数据集、需要更快的训练速度

#### 3.7.1 环境配置

当使用多机训练时，请设置以下环境变量：

```bash
export MASTER_ADDR="your master address"
export MASTER_PORT=10086
export WORLD_SIZE=1 # The number of machines
export NUM_PROCESS=8 # The number of processes, such as WORLD_SIZE * 8
export RANK=0 # The rank of this machine

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK scripts/z_image_fun/train_control.py \
  [其他训练参数...]
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

**关键参数说明**:

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `GPU_memory_mode` | 显存管理模式,可选值见下表 | `model_group_offload` |
| `ulysses_degree` | Head 维度并行度,单卡时为 1 | 1 |
| `ring_degree` | Sequence 维度并行度,单卡时为 1 | 1 |
| `fsdp_dit` | 多卡推理时对 Transformer 使用 FSDP 节省显存 | `False` |
| `fsdp_text_encoder` | 多卡推理时对文本编码器使用 FSDP | `False` |
| `compile_dit` | 编译 Transformer 加速推理(固定分辨率下有效) | `False` |
| `enable_teacache` | 启用 TeaCache 加速推理 | `True` |
| `teacache_threshold` | TeaCache 阈值,建议 0.05~0.30,越大越快但质量可能下降 | 0.30 |
| `num_skip_start_steps` | 推理开始跳过的步数,减少对生成质量的影响 | 5 |
| `teacache_offload` | 将 TeaCache 张量卸载到 CPU 节省显存 | `False` |
| `cfg_skip_ratio` | 跳过部分 CFG 步数加速推理,建议 0.00~0.25 | 0 |
| `config_path` | 配置文件路径 | `config/z_image/z_image_control.yaml` |
| `model_name` | 模型路径 | `models/Diffusion_Transformer/Z-Image-Turbo` |
| `sampler_name` | 采样器类型:`Flow`、`Flow_Unipc`、`Flow_DPM++` | `Flow` |
| `transformer_path` | 加载训练好的 Transformer 权重路径 | `models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union.safetensors` |
| `vae_path` | 加载训练好的 VAE 权重路径 | `None` |
| `lora_path` | LoRA 权重路径 | `None` |
| `sample_size` | 生成图像分辨率 `[高度, 宽度]` | `[1728, 992]` |
| `weight_dtype` | 模型权重精度,不支持 bf16 的显卡使用 `torch.float16` | `torch.bfloat16` |
| `control_image` | 控制图像路径(如姿态图) | `asset/pose.jpg` |
| `inpaint_image` | 修复输入图(可选) | `asset/8.png` |
| `mask_image` | 掩码图像(可选) | `asset/mask.png` |
| `control_context_scale` | 控制条件权重,推荐值 0.80 | 0.80 |
| `prompt` | 正向提示词,描述生成内容 | `"画面中央是一位年轻女孩..."` |
| `negative_prompt` | 负向提示词,避免生成的内容 | `" "` |
| `guidance_scale` | 引导强度 | 4.0 |
| `seed` | 随机种子,用于复现结果 | 43 |
| `num_inference_steps` | 推理步数 | 50 |
| `lora_weight` | LoRA 权重强度 | 0.55 |
| `save_path` | 生成图像保存路径 | `samples/z-image-t2i-control` |

**显存管理模式说明**:

| 模式 | 说明 | 显存占用 |
|------|------|---------|
| `model_full_load` | 整个模型加载到 GPU | 最高 |
| `model_full_load_and_qfloat8` | 全量加载 + FP8 量化 | 高 |
| `model_cpu_offload` | 使用后将模型卸载到 CPU | 中等 |
| `model_cpu_offload_and_qfloat8` | CPU 卸载 + FP8 量化 | 中低 |
| `model_group_offload` | 层组在 CPU/CUDA 间切换 | 低 |
| `sequential_cpu_offload` | 逐层卸载(速度最慢) | 最低 |

### 4.2 单卡推理

#### 快速开始

单卡推理运行如下命令:

```bash
python examples/z_image_fun/predict_turbo_t2i_control.py
```

根据需求修改编辑 `examples/z_image_fun/predict_turbo_t2i_control.py`,初次推理重点关注如下参数,如果对其他参数感兴趣,请查看上方的推理参数解析。

```python
# 根据显卡显存选择
GPU_memory_mode = "model_group_offload"
# 根据实际模型路径
model_name = "models/Diffusion_Transformer/Z-Image-Turbo"  
# 训练好的权重路径,如 "output_dir_z_image_control/checkpoint-xxx/diffusion_pytorch_model.safetensors"
transformer_path = "models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union.safetensors"  
# 控制图像路径
control_image = "asset/pose.jpg"
# 根据生成内容编写
prompt = "画面中央是一位年轻女孩..."  
# ...
```

生成结果会保存在 `samples/z-image-t2i-control` 目录下。

**图像修复推理**:

如需使用图像修复功能,可以运行:

```bash
python examples/z_image_fun/predict_turbo_i2i_inpaint_2.1.py
```

该脚本支持同时使用控制图像和修复蒙版进行图像生成。

### 4.3 多卡并行推理

**适合场景**:高分辨率生成、加速推理

#### 安装并行推理依赖

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### 配置并行策略

编辑 `examples/z_image_fun/predict_turbo_t2i_control.py`:

```python
# 确保 ulysses_degree × ring_degree = GPU 数量
# 例如使用 2 张 GPU:
ulysses_degree = 2  # Head 维度并行
ring_degree = 1     # Sequence 维度并行
```

**配置原则**:
- `ulysses_degree` 必须能整除模型的head数。
- `ring_degree` 会在sequence上切分,影响通信开销,在head数能切分的时候尽量不用。

**示例配置**:

| GPU 数量 | ulysses_degree | ring_degree | 说明 |
|---------|---------------|-------------|------|
| 1 | 1 | 1 | 单卡 |
| 4 | 4 | 1 | Head 并行 |
| 8 | 8 | 1 | Head 并行 |
| 8 | 4 | 2 | 混合并行 |

#### 运行多卡推理

```bash
torchrun --nproc-per-node=2 examples/z_image_fun/predict_turbo_t2i_control.py
```

## 五、更多资源

- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun