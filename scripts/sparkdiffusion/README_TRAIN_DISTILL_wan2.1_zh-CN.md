# SparkDiffusion 稀疏蒸馏训练指南（Wan2.1-T2V-1.3B）

本文档给出 Wan2.1-T2V-1.3B 上 **SparkDiffusion** 的完整流程：一条两阶段管线，先暖启一个**稀疏（RoLa）学生**，再把它**蒸馏**成少步、CFG-free 的生成器。内容涵盖环境配置、数据准备、两个启动脚本（Stage-1 暖启 + Stage-2 CrossDistill）、以及稀疏少步推理。Wan2.1-T2V-1.3B 是**单专家、覆盖全噪声区间**的模型，因此本文是 Wan2.2 MoE 指南（[README_TRAIN_DISTILL_wan2.2_zh-CN.md](./README_TRAIN_DISTILL_wan2.2_zh-CN.md)）的精简版：没有 MoE 分界、没有高/低噪专家拆分、也没有 init-rollout 接力。

> **什么是 SparkDiffusion？**
>
> SparkDiffusion（arXiv:2609.23153，《Mitigating the High-Sparsity Trap》）把**补偿式稀疏注意力**（RoLa）与 **CrossDistill** 少步配方结合起来。Wan2.1-T2V-1.3B 是覆盖整个噪声区间的单专家 diffusion transformer，因此学生在**整个 rf-time 区间 `[0, 1]` 上一次性**完成暖启与蒸馏——没有可供拆分的 MoE 分界。
>
> - **Stage 1 — 稀疏暖启**（[`train_finetune_sparse_wan2.1.py`](./train_finetune_sparse_wan2.1.py)）：把 RoLa 稀疏自注意力插入 dense 主干，用标准的 step-local flow-matching 目标短暂微调，使蒸馏从一个可用的粗先验出发，而不是从零初始化的稀疏分支出发。该阶段刻意保持**短**。
> - **Stage 2 — CrossDistill**（[`train_distill_wan2.1.py`](./train_distill_wan2.1.py)）：对着 **dense** teacher 把稀疏学生蒸馏成少步生成器。它是分布匹配蒸馏（DMD），含三个同驻网络——可训练的稀疏 `generator`、冻结的 dense `real_score` teacher、可训练的 dense `fake_score` critic——并额外加一项 PCM 式多样性损失。
>
> | 阶段 | 启动脚本 | 产物 |
> |-------|----------|--------|
> | **1 暖启** | [`train_finetune_sparse_wan2.1.sh`](./train_finetune_sparse_wan2.1.sh) | `output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola` |
> | **2 蒸馏** | [`train_distill_wan2.1.sh`](./train_distill_wan2.1.sh) | `output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola_distill` |
>
> 按顺序运行两个启动脚本：Stage-2 的 generator 由 Stage-1 暖启产物播种，而单个蒸馏 checkpoint 在推理时即为完整的少步学生。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
- [三、训练](#三训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 Stage 1 — 稀疏暖启](#32-stage-1--稀疏暖启)
  - [3.3 Stage 2 — CrossDistill](#33-stage-2--crossdistill)
  - [3.4 通用训练参数](#34-通用训练参数)
  - [3.5 FSDP 后端与显存说明](#35-fsdp-后端与显存说明)
  - [3.6 其他后端](#36-其他后端)
  - [3.7 多机分布式训练](#37-多机分布式训练)
- [四、推理 — 稀疏少步学生](#四推理--稀疏少步学生)
- [五、更多资源](#五更多资源)

---

## 一、环境配置

**方式一：使用 requirements.txt**

```bash
pip install -r requirements.txt
```

**方式二：手动安装**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install yunchang xfuser modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
pip install deepspeed==0.17.0 numpy==1.26.4
```

> 💡 RoLa 稀疏注意力依赖仓库内的 Triton block-sparse 内核（`videox_fun/models/attention_utils.py`），因此稀疏学生需要一个可用的 Triton 安装。

**方式三：使用 Docker**

```bash
# 拉取镜像
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# 进入镜像
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun
```

---

## 二、数据准备

SparkDiffusion 使用普通的「视频-字幕」数据（与标准 Wan2.1 训练格式一致）。

### 2.1 快速测试数据集

```bash
# 下载官方示例数据集
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

### 2.2 metadata.json 格式

```json
[
  {
    "file_path": "train/video001.mp4",
    "text": "A beautiful sunset over the ocean, golden hour lighting",
    "type": "video",
    "width": 1024,
    "height": 1024
  }
]
```

- `file_path`：视频路径（相对于 `DATASET_NAME`，或绝对路径）。
- `text`：字幕 / 提示词。
- `type`：固定为 `"video"`。
- `width` / `height`：建议提供以启用 bucket 训练；若缺失可用 `scripts/process_json_add_width_and_height.py` 补齐。

### 2.3 相对路径与绝对路径

```bash
# 相对路径
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"

# 绝对路径（数据在 NAS/OSS 上、跨机共享）
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/metadata_add_width_height.json"
```

---

## 三、训练

完整顺序为 **Stage-1 暖启 → Stage-2 CrossDistill**。请手动按此顺序运行两个启动脚本。

### 3.1 下载预训练模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 Wan2.1 T2V 官方权重
modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir models/Diffusion_Transformer/Wan2.1-T2V-1.3B
```

基座权重会播种 Stage-2 的全部三个网络（稀疏 `generator` 还会额外用 Stage-1 暖启产物初始化）。其目录结构为 `config/wan2.1/wan_civitai.yaml` 读取的 civitai 单文件形态：transformer 位于模型根目录（`transformer_subpath: ./`），与 `Wan2.1_VAE.pth`、UMT5 编码器 `models_t5_umt5-xxl-enc-bf16.pth` 以及 `google/umt5-xxl` 分词器并列。调度器 `shift` 为 `5.0`，蒸馏启动脚本用 `--teacher_timestep_shift=5.0` 与之对齐。

### 3.2 Stage 1 — 稀疏暖启

Stage 1 插入 RoLa 稀疏自注意力，并用 step-local flow-matching 目标短暂暖启。这是一个**短**阶段——短暂暖启即足以为蒸馏建立可用先验，因此启动脚本把它限制在 `--max_train_steps=400`。**不要**用该阶段自身的采样质量来评判它；清晰度是在 Stage 2 建立的。

用 [scripts/sparkdiffusion/train_finetune_sparse_wan2.1.sh](./train_finetune_sparse_wan2.1.sh) 运行：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
NCCL_DEBUG=INFO

accelerate launch --main_process_port=0 --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" \
    --fsdp_cpu_ram_efficient_loading False scripts/sparkdiffusion/train_finetune_sparse_wan2.1.py \
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
  --gradient_accumulation_steps=1 \
  --num_train_epochs=100 \
  --max_train_steps=400 \
  --checkpointing_steps=100 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --random_hw_adapt \
  --enable_bucket \
  --uniform_sampling \
  --trainable_modules "attn_op" \
  --student_topk=0.03 \
  --train_mode="normal" \
  --resume_from_checkpoint="latest"
```

或直接：

```bash
bash scripts/sparkdiffusion/train_finetune_sparse_wan2.1.sh
```

Stage-1 关键参数：

| 旗标 | 含义 |
|------|---------|
| `--student_topk` | 稀疏注意力中每个 query 保留的 key block 比例（稀疏度 = `1 - topk`）。`0.03` ≈ 97% 稀疏。**必须与** Stage 2 一致。 |
| `--trainable_modules` | `"attn_op"` = Stage A（只训练稀疏分支）；`"."` = Stage B（稀疏网全量微调，用 `--transformer_path` 载入 Stage A）。 |
| `--max_train_steps` | 暖启预算；保持简短。 |

完成后暖启产物为 `output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola/checkpoint-400/diffusion_pytorch_model.safetensors`。

### 3.3 Stage 2 — CrossDistill

Stage 2 对着 dense teacher 把稀疏学生蒸馏成少步、CFG-free 的生成器。`--sparse_student` 把 `generator` 变为 `SparkWanTransformer3DModel`，而 teacher 与 critic 保持 dense。`--generator_transformer_path` 载入 Stage-1 暖启 checkpoint。作为单一全区间专家，学生在整个 rf-time 区间 `[0, 1]` 上完成蒸馏——没有分界段、也没有 init-rollout 接力——同时保留 CrossDistill 多样性项（`--div_mode="pcm_random" --loss_scale_div=10.0`）。

用 [scripts/sparkdiffusion/train_distill_wan2.1.sh](./train_distill_wan2.1.sh) 运行：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
# Stage-1 暖启产物。可用环境变量覆盖；下方 --student_topk 必须与 Stage 1 一致。
export SPARSE_STUDENT_CKPT="output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola/checkpoint-400/diffusion_pytorch_model.safetensors"
NCCL_DEBUG=INFO

accelerate launch --main_process_port=0 --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_POST" \
    --fsdp_cpu_ram_efficient_loading False scripts/sparkdiffusion/train_distill_wan2.1.py \
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
  --gradient_accumulation_steps=1 \
  --num_train_epochs=100 \
  --checkpointing_steps=200 \
  --learning_rate=2e-06 \
  --learning_rate_critic=4e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola_distill" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --random_hw_adapt \
  --enable_bucket \
  --trainable_modules "." \
  --train_mode="normal" \
  --distill_loss="dmd" \
  --max_simulation_steps_fake=2 \
  --div_mode="pcm_random" \
  --loss_scale_div=10.0 \
  --teacher_timestep_shift=5.0 \
  --real_guidance_scale=5.0 \
  --sparse_student \
  --student_topk=0.03 \
  --generator_transformer_path=$SPARSE_STUDENT_CKPT \
  --resume_from_checkpoint="latest"
```

或直接：

```bash
bash scripts/sparkdiffusion/train_distill_wan2.1.sh
```

> ⚠️ **蒸馏阶段没有步数上限。** 它会一直运行直到你手动停止（kill 掉 `train_distill_wan2.1.py` 进程）；checkpoint 每 `--checkpointing_steps=200` 步写一次。

### 3.4 通用训练参数

| 参数 | 说明 | 示例值 |
|-----------|-------------|---------------|
| `--config_path` | Wan2.1 config（单专家，无 `boundary`） | `config/wan2.1/wan_civitai.yaml` |
| `--pretrained_model_name_or_path` | Wan2.1-T2V-1.3B 基座（transformer + VAE + UMT5） | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B` |
| `--train_data_dir` / `--train_data_meta` | 数据集根目录 / metadata JSON | 见 §2 |
| `--train_mode` | SparkDiffusion 仅支持 `normal`（T2V） | `normal` |
| `--student_topk` | 稀疏注意力保留比例（稀疏度 = 1 − topk）；Stage 1 与 Stage 2 必须一致 | `0.03` |
| `--sparse_student` | （Stage 2）把 generator 建成稀疏学生 | - |
| `--generator_transformer_path` | （Stage 2）载入 generator 的 Stage-1 暖启 checkpoint | `.../checkpoint-400/diffusion_pytorch_model.safetensors` |
| `--distill_loss` | 学生分布匹配损失（`dmd` / `tdm`，互斥） | `dmd` |
| `--max_simulation_steps_fake` | backward simulation 中学生的 Euler 步数；`>= 2` 使首段保持 `no_grad` | `2` |
| `--div_mode` / `--loss_scale_div` | PCM 式多样性项 | `pcm_random` / `10.0` |
| `--teacher_timestep_shift` | 冻结 dense teacher 的 timestep shift；与 Wan2.1 调度器 `shift` 对齐 | `5.0` |
| `--real_guidance_scale` | 冻结 dense teacher 的 CFG 尺度 | `5.0` |
| `--learning_rate` / `--learning_rate_critic` | 生成器 / critic 学习率 | `2e-06` / `4e-07` |
| `--checkpointing_steps` | 每 N 步保存 checkpoint | `200`（蒸馏）/ `100`（暖启） |
| `--max_grad_norm` | 梯度裁剪阈值 | `0.05` |
| `--resume_from_checkpoint` | 续训路径或 `"latest"` | `latest` |

> 💡 rf 时间网格参数（`--rf_t_min`、`--rf_t_max`、`--backward_timesteps`）对单专家模型默认覆盖整个区间 `[0, 1]`；请保持它们的 `-1` / 空默认值。

### 3.5 FSDP 后端与显存说明

Stage 2 需让**三个** transformer 同驻（稀疏 generator + dense teacher + dense critic）。每个 1.3B 远比 Wan2.2 的 14B 专家轻量，但优化器态仍占主导，因此蒸馏启动脚本保留 FSDP `FULL_SHARD` 与 `--gradient_checkpointing`：

- `--fsdp_sharding_strategy "FULL_SHARD"`——在多卡间分片 param/grad/optimizer。
- `--fsdp_backward_prefetch "BACKWARD_POST"`——降低 backward 期间的中位驻留。
- `--gradient_checkpointing`——在 backward 中重算激活。
- **OOM 的杠杆是分辨率、而非帧数。** 显存随 `seq_len` 增长；启动脚本以 `--image_sample_size=640 --video_sample_size=640 --token_sample_size=640` 配 `--random_hw_adapt --enable_bucket`、在 `--video_sample_n_frames=81` 下分桶。若 OOM，请降低 sample size（例如 `640 → 480`），而不是减少帧数。
- 帧数须满足 `(N − 1) % 4 == 0`。

> 💡 1.3B 下三个 transformer 通常无需 CPU offload 即可放下；若 OOM，在 `accelerate launch` 行加上 `--fsdp_offload_params true` 把 param/grad/optimizer offload 到 CPU（更慢）。`--low_vram` 只把冻结侧停到 CPU；它是对分片的补充，而非替代。

### 3.6 其他后端

Stage-2 启动脚本默认使用 FSDP `FULL_SHARD`。同一个 [`train_distill_wan2.1.py`](./train_distill_wan2.1.py) 也可在 DeepSpeed Zero-2 下运行，或不使用任何分片后端；在 1.3B 规模下这些方案比 Wan2.2 的 14B 专家更可行。请保持脚本参数与 §3.3 完全一致——只改动 `accelerate launch` 头部。

#### 3.6.1 使用 DeepSpeed-Zero-2 进行训练

DeepSpeed Zero-2 只分片优化器态与梯度、**不**分片参数。对三个同驻的 1.3B transformer 而言，它在默认分辨率下即可从容放下：

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/X-Fun-Videos-Demo/"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_add_width_height.json"
export SPARSE_STUDENT_CKPT="output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola/checkpoint-400/diffusion_pytorch_model.safetensors"
NCCL_DEBUG=INFO

accelerate launch --main_process_port=0 --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/sparkdiffusion/train_distill_wan2.1.py \
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
  --gradient_accumulation_steps=1 \
  --num_train_epochs=100 \
  --checkpointing_steps=200 \
  --learning_rate=2e-06 \
  --learning_rate_critic=4e-07 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola_distill" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --random_hw_adapt \
  --enable_bucket \
  --trainable_modules "." \
  --train_mode="normal" \
  --distill_loss="dmd" \
  --max_simulation_steps_fake=2 \
  --div_mode="pcm_random" \
  --loss_scale_div=10.0 \
  --teacher_timestep_shift=5.0 \
  --real_guidance_scale=5.0 \
  --sparse_student \
  --student_topk=0.03 \
  --generator_transformer_path=$SPARSE_STUDENT_CKPT \
  --resume_from_checkpoint="latest"
```

#### 3.6.2 不使用 DeepSpeed 与 FSDP 训练

完全去掉分片后端——三个同驻的 1.3B transformer 可放进单张高显存 GPU，不过多卡仍推荐 FSDP。把 `accelerate launch` 头部替换为朴素形式，脚本参数与 §3.6.1 保持一致：

```bash
accelerate launch --main_process_port=0 --mixed_precision="bf16" \
  scripts/sparkdiffusion/train_distill_wan2.1.py \
  <与 §3.6.1 相同的 --flag 参数块>
```

### 3.7 多机分布式训练

启动脚本为单机。多机时在 `accelerate launch` 行追加下列参数（所有机器须通过 NFS/共享存储共享同一份数据集 / 基座权重 / `output_dir`）：

```bash
--main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT \
--num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK
```

无 RDMA 时，另设 `NCCL_IB_DISABLE=1` 与 `NCCL_P2P_DISABLE=1`。

---

## 四、推理 — 稀疏少步学生

Stage-2 产物是一个少步、CFG-free 的稀疏学生。作为单专家模型，推理只载入**一个** checkpoint（没有高噪专家）。构建 wan2.1 稀疏类 `SparkWanTransformer3DModel`，设置训练时的 top-k，再载入蒸馏权重——顺序很关键，因为 `set_student_topk` 会（重）建接收训练好 RoLa 参数的 `attn_op` 内核：

```python
from videox_fun.models import SparkWanTransformer3DModel
transformer = SparkWanTransformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True, torch_dtype=weight_dtype,
)
transformer.set_student_topk(student_topk)          # 必须在 load_state_dict 之前
# transformer.load_state_dict(load_file(transformer_path), strict=False)
```

Wan2.1 T2V 入口 [examples/wan2.1/predict_t2v.py](../../examples/wan2.1/predict_t2v.py) 已在 `WanPipeline` 下运行少步 CFG-free 采样（`sampler_name="Flow"`、`num_inference_steps=4`、`guidance_scale=1.0`）；把 dense 的 `WanTransformer3DModel` 换成上面的稀疏构建即可。需设置：

| 参数 | 说明 | 示例值 |
|-----------|-------------|---------------|
| `GPU_memory_mode` | 显存模式 | `model_full_load` |
| `config_path` | Wan2.1 config | `config/wan2.1/wan_civitai.yaml` |
| `model_name` | Wan2.1-T2V-1.3B 基座（transformer / VAE / UMT5） | `models/Diffusion_Transformer/Wan2.1-T2V-1.3B` |
| `student_topk` | **必须等于**训练时的 `--student_topk` | `0.03` |
| `transformer_path` | 稀疏学生 checkpoint（单专家） | `output_dir_sparkdiffusion_wan2.1_1.3b_sparse_rola_distill/checkpoint-<step>/diffusion_pytorch_model.safetensors` |
| `sampler_name` | `Flow` 与少步 CFG-free 学生匹配 | `Flow` |
| `num_inference_steps` | 学生的少步步数 | `4` |
| `guidance_scale` | 蒸馏学生为 CFG-free | `1.0` |
| `sample_size` / `video_length` / `fps` | `[高, 宽]` / 帧数 / 帧率 | `[480, 832]` / `81` / `16` |

```bash
python examples/wan2.1/predict_t2v.py
```

> 💡 载入顺序为 `from_pretrained → set_student_topk → load_state_dict`；顺序颠倒会把训练好的 RoLa 权重重新清零。当 `attn_op.*` 键载入且 unexpected keys 为 **0** 时，即确认为稀疏 checkpoint。

> 📌 仓库自带的 [examples/sparkdiffusion/predict_t2v.py](../../examples/sparkdiffusion/predict_t2v.py) 面向 Wan2.2 双专家 MoE 学生（`SparkWan2_2Transformer3DModel` + `Wan2_2Pipeline`）；单专家的 Wan2.1 学生使用 `SparkWanTransformer3DModel` + `WanPipeline`，只载入一个 checkpoint。

> 📌 论文的 **Stage 3**（FP8/INT8 融合量化推理核）在本仓库中有意未接入。

---

## 五、更多资源

- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
- **论文**：SparkDiffusion，《Mitigating the High-Sparsity Trap》（arXiv:2609.23153）
- **稀疏学生模型**：`videox_fun/models/spark_transformer3d.py`（`SparkWanTransformer3DModel`）
- **RoLa 稀疏注意力内核**：`videox_fun/models/attention_utils.py`
- **Wan2.2 MoE 对应文档**：[README_TRAIN_DISTILL_wan2.2_zh-CN.md](./README_TRAIN_DISTILL_wan2.2_zh-CN.md)
