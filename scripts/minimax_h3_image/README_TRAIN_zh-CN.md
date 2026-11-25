# MiniMax-H3-Image 训练指南

本文档给出 MiniMax-H3-Image 全量参数训练的完整流程：环境配置、数据准备、全量参数训练与推理验证。MiniMax-H3-Image 是 MiniMax-H3 家族的单流 packed-sequence 图像 transformer。LoRA 训练见 `scripts/minimax_h3_image/README_TRAIN_LORA_zh-CN.md`。

MiniMax-H3-Image 用同一个模型完成文生图与图像编辑：一次请求被打包成单条序列，行序为 `[文本 | 参考图 | 生成图]`，因此 `--train_mode=t2i` 与 `--train_mode=edit` 的差别只在于数据集 meta 是否带 `source_file_path`。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 数据集结构](#22-数据集结构)
  - [2.3 metadata.json 格式](#23-metadatajson-格式)
  - [2.4 相对路径与绝对路径的用法](#24-相对路径与绝对路径的用法)
- [三、全量参数训练](#三全量参数训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始（FSDP）](#32-快速开始fsdp)
  - [3.3 常用训练参数](#33-常用训练参数)
  - [3.4 训练验证](#34-训练验证)
  - [3.5 图像编辑训练](#35-图像编辑训练)
  - [3.6 其他后端](#36-其他后端)
  - [3.7 多机分布式训练](#37-多机分布式训练)
- [四、推理验证](#四推理验证)
  - [4.1 推理参数解析](#41-推理参数解析)
  - [4.2 单卡推理](#42-单卡推理)
  - [4.3 多卡并行推理](#43-多卡并行推理)
- [五、其他资源](#五其他资源)

---

## 一、环境配置

**方式一：使用 requirements.txt**

```bash
pip install -r requirements.txt
# Versions used to verify MiniMax-H3-Image (includes Qwen3-VL support).
pip install "transformers==4.57.3" "diffusers==0.32.2" "accelerate==1.7.0" "peft==0.17.0"
```

**方式二：手动安装依赖**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft==0.17.0" "accelerate==1.7.0" "gradio>=3.41.2" "diffusers==0.32.2" "transformers==4.57.3"
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

我们提供两个测试数据集，一个用于文生图，一个用于图像编辑。

```bash
# 下载官方示例数据集
modelscope download --dataset PAI/X-Fun-Images-Demo --local_dir ./datasets/X-Fun-Images-Demo
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
│   ├── 📂 source/                     # 参考图，仅 `--train_mode=edit` 需要
│   │   └── 📄 ...
│   └── 📄 metadata.json
```

### 2.3 metadata.json 格式

**相对路径格式**（示例）：
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
    "text": "Make the jacket red",
    "width": 1328,
    "height": 1328,
    "source_file_path": "source/image002.png"
  }
]
```

**绝对路径格式**：
```json
[
  {
    "file_path": "/absolute/path/to/images/sunset.jpg",
    "text": "A beautiful sunset over the ocean",
    "width": 1024,
    "height": 1024
  }
]
```

**关键字段说明**：
- `file_path`：目标图像路径（相对或绝对）
- `text`：编辑指令，或对目标图像的描述
- `source_file_path`：参考图路径。仅在 `--train_mode=edit` 下读取；一次请求需要多张参考图时可写成列表
- `width` / `height`：图像尺寸（**建议**提供以启用分桶训练；若未提供，训练时会自动读取，数据存放在 OSS 等较慢系统上时会拖慢训练）
  - 可用 `scripts/process_json_add_width_and_height.py` 为缺少宽高字段的 JSON 补齐
  - 用法：`python scripts/process_json_add_width_and_height.py --input_file datasets/X-Fun-Images-Demo/metadata.json --output_file datasets/X-Fun-Images-Demo/metadata_add_width_height.json`

参考图按自身分辨率编码：训练脚本会把它的短边缩放到 2048，并把两边都取整到 32 的倍数，不做裁剪。因此参考图的代价是序列长度，而不是画布大小，单条样本请控制参考图数量。

`text` 按原样使用，并被截断到发布模型的 4096 个 prompt token，截断时会有警告。该上限只作用于文本本身：参考图的视觉块排在序列最前面，不会挤掉文本的长度。

### 2.4 相对路径与绝对路径的用法

**相对路径**：

若数据使用相对路径，训练脚本按如下配置：

```bash
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
```

**绝对路径**：

若数据使用绝对路径，训练脚本按如下配置：

```bash
export DATASET_NAME=""
export DATASET_META_NAME="/absolute/path/to/metadata_add_width_height.json"
```

> 💡 **建议**：数据集较小且存放在本地时使用相对路径；数据集位于外部存储（如 NAS、OSS）或多机共享时使用绝对路径。

---

## 三、全量参数训练

### 3.1 下载预训练模型

训练和推理默认读取原始 `transformer/`，加载时逐张量转换，并用其中 `final_layer.video_out` 初始化共享的 `proj_out`；`transformer_diffusers/` 默认不读取，仅在缺少原始版本时回退使用，此时需通过 `<model_root>/original_output_head.safetensors` 提供原始单投影。图像 VAE 使用 `vae_diffusers/`；条件器就是 MiniMax-H3 使用的 Qwen3-VL，因此视频版快照可以按组件软链接复用。

```bash
# 下载 MiniMax-H3-Image 官方权重
modelscope download --model MiniMax/MiniMax-H3-Image --local_dir models/Diffusion_Transformer/MiniMax-H3-Image
```

### 3.2 快速开始（FSDP）

33B 骨干连同冻结的条件器无法放进单卡，因此推荐用 FSDP 切分。每个 transformer block（`MiniMaxH3TransformerBlock`）是一个切分单元，这也正是 `train_lora.py` 的 LoRA 模块能跟随所属层一起切分的前提。

```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3-Image"
export DATASET_NAME="datasets/X-Fun-Images-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Demo/metadata_add_width_height.json"
# 多机无 RDMA 时启用 NCCL_IB_DISABLE=1 与 NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3_image/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --train_batch_size=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_image" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --train_mode="t2i" \
  --trainable_modules "."
```

启动脚本的成品已放在 `scripts/minimax_h3_image/train.sh`。

**batch size 固定为 1**：packed 序列（文本 token + 参考图行 + 生成行）是按样本构建的，若要更大 batch 就得在打包内再做样本循环。需要等效大 batch 请使用 `--gradient_accumulation_steps`。

### 3.3 常用训练参数

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `--pretrained_model_name_or_path` | MiniMax-H3-Image 快照路径 | `models/Diffusion_Transformer/MiniMax-H3-Image` |
| `--train_data_dir` | 训练数据目录 | `datasets/X-Fun-Images-Demo/` |
| `--train_data_meta` | 训练数据元数据文件 | `datasets/X-Fun-Images-Demo/metadata_add_width_height.json` |
| `--train_mode` | `t2i` 只读 `file_path`；`edit` 额外读取 `source_file_path` | `t2i` |
| `--image_sample_size` | 训练画布边长，须为 32 的倍数；分桶据此缩放两边 | 1024 |
| `--fix_sample_size H W` | 固定画布而不分桶，取整到 32 的倍数 | - |
| `--min_generation_area` | 仅作用于验证样本的面积下限，默认为 `image_sample_size ** 2` | 4194304 |
| `--train_batch_size` | 保持为 1，见上文说明 | 1 |
| `--gradient_accumulation_steps` | 梯度累积步数（等效更大 batch） | 1 |
| `--sigma_shift` | 对采样的 sigma 施加指数偏移。1.0 与发布的 image 网格一致；更大值把训练重心移向噪声端 | 1.0 |
| `--uniform_sampling` | 均匀采样 sigma，而非 logit-normal 加权 | - |
| `--weighting_scheme` | loss 加权方式：`sigma_sqrt`、`logit_normal`、`mode`、`cosmap`、`none` | `none` |
| `--enable_bucket` | 按长宽比分组，图像既不裁剪也不填充 | - |
| `--random_hw_adapt` | 额外把每个 batch 随机缩放到画布的某个比例，属数据增强 | - |
| `--low_vram` | VAE 与条件器常驻 CPU，仅在编码时上卡 | - |
| `--gradient_checkpointing_save_on_cpu` | 重算所需的激活值存放在主存；更慢但省显存 | - |
| `--checkpointing_steps` | 每 N 步保存一次 checkpoint | 500 |
| `--resume_from_checkpoint` | 从指定 checkpoint 续训，`"latest"` 自动选择最近一个 | None |
| `--trainable_modules` | 可训练模块（`"."` 表示全部） | `"."` |
| `--use_ema` | 维护指数滑动平均，并用它做验证 | - |
| `--validation_prompts` / `--validation_steps` / `--validation_epochs` | 定期生成样图，见 **3.4** | - |

训练目标与采样器积分的 rectified-flow 目标一致：模型输入为 `x_t = t * x0 + (1 - t) * noise`，其中 `t = 1 - sigma`（`t = 1` 为干净图像，与 Wan 系列脚本相反），回归目标为 `x0 - noise`。条件行会做噪声增强且不参与监督。所有时间步共用原始单投影 `proj_out`。

### 3.4 训练验证

验证为可选功能，默认启动脚本未开启。追加验证参数即可在训练过程中定期生成测试图，便于观察训练进度与模型质量。

```bash
accelerate launch --mixed_precision="bf16" --use_fsdp ... scripts/minimax_h3_image/train.py \
  # ...（其余训练参数）
  --validation_steps=100 \
  --validation_epochs=500 \
  --validation_prompts="a red cube on a white table, studio lighting." "a portrait of an astronaut"
```

**说明**：
- 验证调用 `MiniMaxH3ImagePipeline`，使用单投影，默认 40 步、shift 12。每条 prompt 还需要一次 Qwen3-VL 条件器前向，验证期间进度条会暂停计时。
- 验证默认按 `--image_sample_size` 生成方图；要与竖图训练数据的画幅一致，可追加 `--fix_sample_size 1152 896`（高 宽），它会把训练画布与验证样图固定为同一非方形画幅。两个训练脚本还支持 `--validation_num_inference_steps` 与 `--validation_flow_shift` 覆盖验证调度。
- 验证样图写入 `output_dir/sample/sample-{step}-rank{rank}-image-{i}.png`，第一个训练 batch 写入 `output_dir/sanity_check/`，扫一眼即可确认数据通路。
- `--train_mode=edit` 时，用 `--validation_image_paths` 为每条 prompt 按同序配一张参考图。

### 3.5 图像编辑训练

编辑就是同一套训练脚本跑带 `source_file_path` 的 meta，画布通常更大：参考图以条件行的形式进入序列，且 10% 的样本会丢弃文本，以维持 guidance 蒸馏依赖的无条件分支。

```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3-Image"
export DATASET_NAME="datasets/X-Fun-Images-Edit-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Images-Edit-Demo/metadata_add_width_height.json"
# 多机无 RDMA 时启用 NCCL_IB_DISABLE=1 与 NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3_image/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --train_batch_size=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_minimax_h3_image" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --train_mode="edit" \
  --trainable_modules "."

# 可选验证（见 3.4）：为每条 --validation_prompts 按同序配一张参考图
# --validation_image_paths="datasets/X-Fun-Images-Edit-Demo/source/souce-20260421_140343_099985_000/10_00015405.png"
```

一条样本构建出的序列是 `文本 token + 视觉 token + 参考图 latent 行 + 生成行`，所以 edit 模式每多一张参考图大约多出 4096 个 token 的块。请保持 `--train_batch_size=1`，需要更大画布时提高 `--gradient_accumulation_steps`。

### 3.6 其他后端

#### 3.6.1 使用 DeepSpeed-Zero-2 训练

DeepSpeed-Zero-2 只在 rank 间切分优化器状态与梯度，每卡仍常驻完整骨干，因此需要单卡装下 33B 的显存。GPU 数量不多、画布较小时适用：

```bash
accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/minimax_h3_image/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  # ...（3.2 的参数，去掉 FSDP 相关旗标）
```

#### 3.6.2 使用 DeepSpeed-Zero-3 训练

Zero-3 会连权重一起切分。此时 checkpoint 是分片状态，训练脚本会自动改用 `--save_state`，取回完整权重用：

```bash
python scripts/zero_to_bf16.py output_dir/checkpoint-{your-num-steps} output_dir/checkpoint-{your-num-steps}-outputs --max_shard_size 80GB --safe_serialization
```

在 `accelerate launch` 中传 `--use_deepspeed --deepspeed_config_file config/zero_stage3_config.json`；训练脚本会从配置文件读取 stage 并据此切换保存 hook。

### 3.7 多机分布式训练

各机器代码与数据路径一致（NFS / 共享存储），每台一份 `accelerate` 配置，只有 rank 不同：

```bash
# 机器 0
accelerate launch --config_file multi_node_yaml/zero_stage2.yaml \
    --num_machines=2 --machine_rank=0 --main_process_ip=192.168.1.2 --main_process_port=12345 \
    scripts/minimax_h3_image/train.py ...

# 机器 1：同样的命令，改 --machine_rank=1
```

- **防火墙**：确保机器间端口互通，否则会报连接错误
- **NCCL 环境**：跨机无 RDMA 时，取消 `NCCL_IB_DISABLE=1` 与 `NCCL_P2P_DISABLE=1` 的注释
- **数据同步**：所有机器都能访问同一份数据路径（NFS/共享存储）

---

## 四、推理验证

### 4.1 推理参数解析

推理入口为 `examples/minimax_h3_image/predict_t2i.py` 与 `examples/minimax_h3_image/predict_i2i_edit.py`，通过修改文件顶部的变量来配置。

**关键参数说明**：

| 参数 | 说明 | 示例值 |
|------|------|-------|
| `GPU_memory_mode` | 显存管理模式，见下表 | `model_group_offload` |
| `ulysses_degree` / `ring_degree` | head / 序列维度并行度，单卡均为 1 | 1 |
| `fsdp_dit` | 多卡推理时用 FSDP 切分 transformer | `True` |
| `fsdp_text_encoder` | 同时切分 Qwen3-VL 条件器（否则每卡各存 ~62 GB） | `True` |
| `compile_dit` | 固定分辨率下编译 transformer 以加速 | `False` |
| `model_name` | 模型路径 | `models/Diffusion_Transformer/MiniMax-H3-Image` |
| `transformer_path` | 训练得到的 transformer 目录或文件 | `None` |
| `vae_path` | 训练得到的 VAE 权重 | `None` |
| `lora_path` | `train_lora.py` 产出的 LoRA 权重 | `None` |
| `lora_weight` | LoRA 强度 | 0.55 |
| `sample_size` | 返回图像的 `[height, width]` | `[1024, 1024]` |
| `min_generation_area` | 面积下限：更小的请求会在放大画布上生成再缩回 | `2048 * 2048` |
| `weight_dtype` | 不支持 bf16 的显卡（v100、2080 Ti）用 `torch.float16` | `torch.bfloat16` |
| `prompt` | 编辑指令或画面描述 | `"A red fox trotting..."` |
| `images` | 参考图，仅 `predict_i2i_edit.py` | `["/path/to/source.png"]` |
| `seed` | 随机种子，用于复现 | 43 |
| `save_path` | 生成图像保存路径 | `samples/minimax-h3-image-t2i` |

默认单投影可设置 `num_inference_steps` 和 `flow_shift`，默认分别为 40 和 12。已发布 backbone 已内化 guidance，不支持单独设置 `guidance_scale` 或 `negative_prompt`。原始 transformer 自带单投影；若只有 PDD 转换版，则需将 `original_output_head.safetensors` 放入模型根目录（见 **3.1**）。

**PDD（六步推理与微调）。** Parallel Decoding Distillation（PDD，[arXiv 2607.26004](https://arxiv.org/abs/2607.26004)）把 flow 模型变成并行解码器；发布的 Image 权重含六个 PDD 输出头，使用固定六步调度。推理时将 `output_head_type` 设为 `"pdd"` 即可选用，或 `MiniMaxH3ImagePipeline.from_pretrained(model_root, output_head_type="pdd")`。原始 transformer 模式下，PDD 头从模型根目录的 `pdd_fused_heads.safetensors` 与 `pdd_config.json` 读取；转换版直接读取 `pdd_proj_out`。PDD 使用固定六步调度，不接受自定义 `flow_shift`。 两个 Image 训练入口也可通过 `--output_head_type=pdd` 基于这些头进行普通 flow-matching 微调；LoRA 训练只更新适配器，输出头保持冻结。两个入口均不会运行仓库通用 PDD 的教师蒸馏。`train_lora.py` 还支持 `--pdd_discrete_timestep`，在发布的六个时间节点上训练。

**显存管理模式说明**：

| 模式 | 说明 | 显存占用 |
|------|------|---------|
| `model_full_load` | 整个模型常驻 GPU | 最高 |
| `model_full_load_and_qfloat8` | 全量常驻 + FP8 量化 | 高 |
| `model_cpu_offload` | 用后把模型卸载到 CPU | 中 |
| `model_cpu_offload_and_qfloat8` | CPU 卸载 + FP8 量化 | 中低 |
| `model_group_offload` | 按层组在 CPU/CUDA 间切换 | 低 |
| `sequential_cpu_offload` | 逐层顺序卸载（最慢） | 最低 |

### 4.2 单卡推理

#### 快速开始

请在仓库根目录执行。

```bash
python examples/minimax_h3_image/predict_t2i.py
python examples/minimax_h3_image/predict_i2i_edit.py
```

按需求修改文件顶部的变量。首次推理重点关注以下几项。

```python
# 根据显卡显存选择
GPU_memory_mode = "model_group_offload"
# 模型路径
model_name = "models/Diffusion_Transformer/MiniMax-H3-Image"
# 全量微调权重，如 "output_dir_minimax_h3_image/checkpoint-500/transformer"
transformer_path = None
# LoRA 权重，如 "output_dir_minimax_h3_image_lora/checkpoint-500.safetensors"
lora_path = None
# 根据生成内容填写
prompt = "A red fox trotting through a snowy pine forest, snow crunching underfoot"
```

#### LoRA 与输出头

两个脚本都默认 `output_head_type = "single"`；设置 `lora_path` 后，推理通过仓库已有的 `merge_lora` / `unmerge_lora` 合并和撤销 LoRA。支持本仓库导出的 kohya 与 PEFT 线性层格式。将 LoRA 合并进 BF16 底座可能舍入较小的更新，因此推理结果可能与单独计算 LoRA 的训练验证有所不同。不加载 LoRA 时保持 `lora_path = None`；全参训练得到的 transformer 目录用 `transformer_path`。图像编辑的参考图在 `predict_i2i_edit.py` 的 `images` 中按序列出。默认返回 1024×1024，生成画布最小面积为 2048×2048（`sample_size`、`min_generation_area`）。没有头元数据的 LoRA 在推理时只会触发告警，使用前请自行确认来源。更换投影不改变其余 backbone 的权重，40 步单头也不保证训练收敛或画质优于官方六步 PDD。

### 4.3 多卡并行推理

**适用场景**：高分辨率生成、加速推理

#### 安装并行推理依赖

```bash
pip install xfuser==0.4.2 yunchang==0.6.2
```

#### 配置并行策略

修改 `examples/minimax_h3_image/predict_t2i.py`：

```python
# 确保 ulysses_degree × ring_degree 等于使用的 GPU 数量
# 例如使用 2 卡：
ulysses_degree = 2  # head 维度并行
ring_degree = 1     # 序列维度并行
```

**配置原则**：
- `ulysses_degree` 需能整除模型的 56 个注意力头
- `ring_degree` 在序列维度切分，会影响通信开销；头数可分时优先不用它

#### 运行多卡推理

```bash
torchrun --nproc-per-node=2 examples/minimax_h3_image/predict_t2i.py
```

## 五、其他资源

- **MiniMax-H3-Image LoRA 训练**：`scripts/minimax_h3_image/README_TRAIN_LORA_zh-CN.md`
- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
- **模型主页**：https://modelscope.cn/models/MiniMax/MiniMax-H3-Image
