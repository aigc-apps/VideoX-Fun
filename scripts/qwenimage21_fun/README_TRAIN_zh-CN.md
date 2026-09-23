# Qwen-Image 2.1 Control(ControlNet-Union)训练指南

本文档提供在冻结的 **Qwen-Image 2.1** 基座 transformer 之上训练 **ControlNet-Union** 适配器的完整流程,包括环境配置、
数据准备、分布式训练、CFG 蒸馏与推理测试。

一整套零初始化的 `control_blocks` 并行链会产出逐层残差(`hints`),再加回冻结的基座 block,因此适配器初始时等价于恒等
跳连,并逐步学习控制信号。训练时只更新控制模块(`--trainable_modules "control"`)。

该适配器是 control + inpaint 的 **union**:条件张量 `control_context` 打包了
`[control_latents(64) | mask(1) | masked-image latents(64)] = 129` 个通道,因此单个适配器同时处理空间控制
(depth / canny / pose / …)与图像修补(inpainting)。

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
  - [3.8 CFG 蒸馏](#38-cfg-蒸馏)
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
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

**方式 3：使用docker**

使用docker的情况下,请保证机器中已经正确安装显卡驱动与CUDA环境,然后依次执行以下命令:

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

> **Qwen-Image 2.1 特有**:文本编码器是 **Qwen3-VL** 模型,因此环境需要一个包含 `qwen3_vl` 结构的 `transformers`
> 版本(比 `requirements.txt` 里的基线更新)。如果 `Qwen3VLForConditionalGeneration` / `Qwen3VLProcessor` 导入为
> `None`,说明你的 `transformers` 太旧。

---

## 二、数据准备

Control 训练使用 `ImageVideoControlDataset`。

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
│   ├── 📂 train/                 # 目标图(模型应生成的内容)
│   │   ├── 📄 image001.jpg
│   │   └── 📄 ...
│   ├── 📂 control/               # 配对的 control / 条件图(pose、canny、depth…)
│   │   ├── 📄 image001.png
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 2.3 metadata.json 格式

清单文件是标准图像 metadata JSON,外加一个 `control_file_path` 字段,把每张**目标图**与其 **control 图**配对。

```json
[
    {
      "file_path": "train/image001.jpg",
      "control_file_path": "control/image001.png",
      "text": "A young woman, studio lighting, high quality.",
      "width": 1024,
      "height": 1024,
      "type": "image"
    }
]
```

**关键字段说明**:
- `file_path`:**目标图**(相对或绝对路径)。
- `control_file_path`:**control / 条件图**(姿态图、边缘图、深度图、线稿…)。它以 RGB 载入,并与目标图使用**完全相同**的
  变换做 resize / crop,从而保证像素对齐。
- `text`:描述(caption)。
- `width` / `height`:推荐提供,用于 bucket 训练;可用 `scripts/process_json_add_width_and_height.py` 为缺失字段的
  JSON 补上。
- `type`:图像数据为 `"image"`。

> **你只需提供目标图 + control 图,不需要提供掩膜。** inpaint 掩膜是即时生成的:
> - collate 中用 `get_random_mask` 生成随机矩形遮挡。
> - 送进 union 分支的被遮罩图始终是 `target * (1 - mask)`。

> **RGBA 说明**:2.1 VAE 读取 RGBA。训练图以 RGB 载入,在编码前会自动合成到不透明 alpha 通道,因此你不需要提供 RGBA 数据。

### 2.4 相对路径与绝对路径使用方案

**相对路径**(小型本地数据集):
```bash
export DATASET_NAME="datasets/X-Fun-Images-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Controls-Demo/metadata_add_width_height.json"
```

**绝对路径**(NAS / OSS / 多机共享数据):
```bash
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata.json"
```

> 如果数据集存放在外部存储或被多台机器共享,推荐使用绝对路径。

---

## 三、Control 训练

### 3.1 下载预训练模型

将 `MODEL_NAME` 指向本地的 **Qwen-Image 2.1** checkpoint 目录。其 `transformer/` 子目录提供冻结的基座权重;控制模块在
载入时零初始化。

```bash
mkdir -p models/Diffusion_Transformer
# 将你的 Qwen-Image 2.1 权重放到这里,例如
# models/Diffusion_Transformer/Qwen-Image-2.1/{transformer,vae,text_encoder,...}
```

> **没有公开的 2.1 ControlNet-Union checkpoint。** 与 Qwen-Image 2512 不同,2.1 目前并没有发布的
> `...-Fun-Controlnet-Union.safetensors`,因此训练从零开始,control 分支**零初始化**。所以启动脚本不写 `--transformer_path`;
> 只有在你需要 resume / fine-tune 一个已训练好的 control checkpoint(或你自己用 `scripts/*/extract_control_weights.py` 得到的)
> 时才加上它。

### 3.2 快速开始（DeepSpeed-Zero-2）

推荐使用 DeepSpeed-Zero-2 或 FSDP 方案进行训练,可以节省大量显存。

如果按照 **2.1 快速测试数据集**下载数据与 **3.1 下载预训练模型**放置权重后,直接复制以下启动指令进行启动。

```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/qwenimage21_fun/train_control.py \
  --config_path="config/qwenimage21/qwenimage21_control.yaml" \
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
  --output_dir="output_dir_qwen_image_21_control" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "control"
```

### 3.3 训练常用参数解析

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--config_path` | **必填。** 用 `control_layers` / `control_in_dim` 构建 control transformer | `config/qwenimage21/qwenimage21_control.yaml` |
| `--pretrained_model_name_or_path` | 基座 Qwen-Image 2.1 模型(冻结权重) | `models/Diffusion_Transformer/Qwen-Image-2.1` |
| `--train_data_dir` / `--train_data_meta` | 数据根目录 / 清单 JSON | `""` / `/path/metadata.json` |
| `--trainable_modules` | `"control"` 只训练 `control_blocks.*` + `control_img_in.*`,基座冻结 | `"control"` |
| `--transformer_path` | **从零训练时省略**;仅在 resume / fine-tune 已训练的 control 分支时加上 | *(无)* |
| `--image_sample_size` | 最大训练分辨率,自动 bucket | `1024` |
| `--train_batch_size` / `--gradient_accumulation_steps` | 单卡 batch / 梯度累积 | `1` / `1` |
| `--learning_rate` | 初始学习率 | `2e-05` |
| `--lr_scheduler` / `--lr_warmup_steps` | 学习率调度 / 预热 | `constant_with_warmup` / `100` |
| `--checkpointing_steps` | 每 N 步保存 checkpoint | `50` |
| `--gradient_checkpointing` | 激活重计算 | flag |
| `--vae_mini_batch` | VAE 编码 mini-batch(control 要编码 3 路 latent) | `1` |
| `--max_grad_norm` | 梯度裁剪 | `0.05` |
| `--enable_bucket` | 按分辨率分组、不裁剪的 bucket 训练 | flag |
| `--uniform_sampling` | 均匀 timestep 采样 | flag |
| `--low_vram` | 空闲时卸载 VAE / 文本编码器以省显存 | flag(可选) |

> **显存**:每个 control step 要编码**三路** latent(目标图、control 图、被遮罩图),比普通训练更重。请保持
> `--vae_mini_batch=1`;显存紧张时再加 `--low_vram`。

### 3.4 训练验证

在训练时配置验证参数,定期渲染 control 预览:

```bash
  --validation_paths "asset/pose.jpg" \
  --validation_steps=50 \
  --validation_epochs=500 \
  --validation_prompts="1girl, black_hair, brown_eyes, ... solo, upper_body"
```

- `--validation_prompts` 与 `--validation_paths` 的数量必须**匹配**,每对的第 i 项一起使用。输出分辨率由每张 control 图的
  宽高比经 `calculate_dimensions(image_sample_size^2, w/h)` 推出。
- 验证在 `--validation_steps` 或 `--validation_epochs` 任一满足时触发。
- 预览写入 `{output_dir}/sample/`。由于 2.1 VAE 解码为 **RGBA**,预览保存为 **`.png`**(JPEG 无法存 alpha)。
- `log_validation` 包在 `try/except` 里:错误的 control 路径只会打印 `Eval error on rank N`,不会让训练崩溃。要确认验证
  是否真的产出了图,请查看 `output_dir/sample/` **并**在日志里 grep `Eval error`。

### 3.5 使用 FSDP 训练

如果 DeepSpeed-Zero-2 显存不足,可以切换使用 FSDP 进行训练。配套的启动脚本 `scripts/qwenimage21_fun/train_control.sh` 运行的就是
下面这条命令,修改其顶部路径(`MODEL_NAME`、`DATASET_META_NAME` 等)后可直接 `bash` 运行(wrap 类必须与 control 模型的 block 同名):

```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap=BaseQwenImage21TransformerBlock,QwenImage21ControlTransformerBlock \
  --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
  scripts/qwenimage21_fun/train_control.py \
  --config_path="config/qwenimage21/qwenimage21_control.yaml" \
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
  --output_dir="output_dir_qwen_image_21_control" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "control"
```

### 3.6 其他后端

#### 3.6.1 不使用 DeepSpeed 与 FSDP 训练

不使用 DeepSpeed 或 FSDP 可能会导致显存不足,仅建议在显存充足的情况下使用;普通 DDP 还需要在每张卡上完整复制 2.1 基座
transformer,通常并不推荐:

```bash
export MODEL_NAME="models/Diffusion_Transformer/Qwen-Image-2.1"
export DATASET_NAME="datasets/X-Fun-Images-Controls-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Controls-Demo/metadata_add_width_height.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/qwenimage21_fun/train_control.py \
  --config_path="config/qwenimage21/qwenimage21_control.yaml" \
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
  --output_dir="output_dir_qwen_image_21_control" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
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

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK scripts/qwenimage21_fun/train_control.py \
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

### 3.8 CFG 蒸馏

`train_control_distill.py` / `train_control_distill.sh` 是一个**可选的第二阶段**。它把 classifier-free guidance(CFG)
蒸馏进已训练好的 control 分支,使**推理阶段无需 guidance scale**。

算法(思路与 `scripts/minimax_h3_fun` / `scripts/flux2_fun` 的 control 蒸馏完全一致):
- **冻结的 teacher**(同一 control 模型的另一份拷贝,用**相同**的 `--transformer_path` 载入,即你第一阶段训练好的 control 分支)
  每步做两次前向,分别在 prompt 与**空**负 prompt 上,两者都**带 control 条件**。两个速度合成 CFG 目标:
  `target = uncond + (cond - uncond) * real_guidance_scale`。
- 可训练的 **student**(control 分支)只跑一次带条件的前向,向该目标回归(速度空间 MSE)。同样只训练
  `--trainable_modules "control"`。

在你已有一个训练好的 control checkpoint 后运行:

```bash
# 将 CONTROL_TRANSFORMER_PATH 指向第一阶段 checkpoint 的
# output_dir_qwen_image_21_control/<ts>/checkpoint-<step>/diffusion_pytorch_model.safetensors
bash scripts/qwenimage21_fun/train_control_distill.sh
```

蒸馏专有参数:

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--transformer_path` | **必填**:student 与 teacher 都载入的、已训练好的 control 分支 | `/root/diffusion_pytorch_model.safetensors` |
| `--real_guidance_scale` | 作用于 teacher 以合成目标的 CFG scale | `3.5` |
| `--learning_rate` | 蒸馏使用更低的学习率 | `2e-06` |
| `--output_dir` | 蒸馏适配器单独的输出目录 | `output_dir_qwen_image_21_control_distill` |

> teacher 是每卡上一份独立的、不分片的 bf16 拷贝(只有 student 被 FSDP 分片),因此很吃显存。可加 `--low_vram` 让 teacher
> 只在其两次前向时上卡。蒸馏后的 student 推理时用 `guidance_scale = 1.0`(CFG 已烘焙进权重)。

---

## 四、推理测试

### 4.1 推理参数解析

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `config_path` | 必须与训练好的适配器 config 一致 | `config/qwenimage21/qwenimage21_control.yaml` |
| `model_name` | 基座 Qwen-Image 2.1 路径 | `models/Diffusion_Transformer/Qwen-Image-2.1` |
| `transformer_path` | 训练好的 control 权重(`control_*` 键以 `strict=False` 载入),基线可为 `None` | `output_dir_qwen_image_21_control/.../diffusion_pytorch_model.safetensors` |
| `sampler_name` | flow-matching 采样器 | `Flow` |
| `sample_size` | 输出画布 `[height, width]` | `[1728, 992]` |
| `control_image` | control 条件图(`predict_t2i_control.py`) | `asset/pose.jpg` |
| `control_image_path` | 可选 control 条件图(`predict_i2i_inpaint.py`,默认 `None`) | `asset/pose.jpg` |
| `control_context_scale` | control 分支强度(适配器训练时消费的取值) | `1.0` |
| `image_path` / `mask_path` | inpaint 输入图 / 掩码(仅 `predict_i2i_inpaint.py`,见 4.2) | `asset/pose.jpg` / `asset/mask.png` |
| `guidance_scale` | CFG 强度。CFG 蒸馏后的 checkpoint 用 `1.0` | `1.0` |
| `weight_dtype` | 不支持 bf16 的卡(v100、2080Ti…)用 `torch.float16` | `torch.bfloat16` |
| `GPU_memory_mode` | 显存管理模式,可选值见下表 | `model_group_offload` |
| `ulysses_degree` / `ring_degree` | 多卡并行(见 4.3)。`ring_degree` 必须为 `1` | `1` / `1` |
| `num_inference_steps` / `seed` | 采样步数 / 随机种子 | `40` / `43` |
| `save_path` | 输出目录 | `samples/qwenimage21-control-images` |

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

```bash
python examples/qwenimage21_fun/predict_t2i_control.py
```

修改文件顶部常量以匹配你的环境。pipeline 会预处理并 VAE 编码 `control_image`(接受 PIL 图 / 路径),构建 129 通道的
`control_context`,并注入 control 残差:

```python
GPU_memory_mode     = "model_group_offload"
model_name          = "models/Diffusion_Transformer/Qwen-Image-2.1"
transformer_path    = "models/Personalized_Model/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"  # 或训练输出的 diffusion_pytorch_model.safetensors
control_image       = "asset/pose.jpg"
control_context_scale = 1.0
prompt              = "A young woman with long straight black hair ..."
sample_size         = [1728, 992]
num_inference_steps = 40
```

结果保存到 `samples/qwenimage21-control-images/*.png`。

> 当存在 `control_context` 时,KV cache 会**自动禁用**:control 残差依赖每步的基座 joint 流,前缀缓存会失效。

**图像修补推理**:

union 适配器也能做 inpainting。`predict_i2i_inpaint.py` 传入 `image_path` + `mask_path`(并把 `control_image_path` 留为 `None`),
于是只用 129 通道条件的 inpaint 半边:

```bash
python examples/qwenimage21_fun/predict_i2i_inpaint.py
```

`mask_path` 语义:**白色(`>= 0.5`)= 重绘**,黑色 = 保留。你可以同时提供 control 图与 inpaint 对,以使用完整 union。

### 4.3 多卡并行推理

**适合场景**:高分辨率生成、加速推理

Qwen-Image 2.1 **仅支持 Ulysses(head 并行)序列并行**。

#### 安装并行推理依赖

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### 配置并行策略

编辑 `examples/qwenimage21_fun/predict_t2i_control.py`:

```python
# 确保 ulysses_degree × ring_degree = GPU 数量
# 例如使用 4 张 GPU:
ulysses_degree = 4  # Head 维度并行
ring_degree = 1     # Sequence 维度并行,必须保持为 1
```

**配置原则**:
- `ulysses_degree` 必须整除 `num_attention_heads`(32):取 `1/2/4/8/16/32`。
- `ring_degree` **必须为 1** —— ring attention 会轮转 KV chunk,无法表达 2.1 的 block-causal mask 或其前缀 KV cache。

**示例配置**:

| GPU 数 | ulysses_degree | ring_degree |
|--------|----------------|-------------|
| 1 | 1 | 1 |
| 4 | 4 | 1 |
| 8 | 8 | 1 |

#### 运行多卡推理

```bash
# 设 ulysses_degree > 1,保持 ring_degree = 1,且 GPU 数 = ulysses_degree * ring_degree。
torchrun --nproc-per-node=4 examples/qwenimage21_fun/predict_t2i_control.py
```

---

## 五、更多资源

- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
- **Qwen-Image 官方仓库**：https://github.com/QwenLM/Qwen-Image
