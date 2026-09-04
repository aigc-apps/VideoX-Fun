# MiniMax-H3 PDD LoRA 训练指南

本文档提供 MiniMax-H3 的 Parallel Decoding Distillation（PDD，[arXiv 2607.26004](https://arxiv.org/abs/2607.26004)）LoRA 训练完整工作流，包括环境配置、条件 cache 准备、分布式训练和推理测试。

> **注意**：MiniMax-H3 是一个音视频生成模型，可以同时生成视频和对应音频。PDD 训练是 **data-free** 的：从不读取目标视频。每个 rank 携带一条轨迹，用学生自己的预测向前滚动，并由同一骨干上的冻结教师监督。训练只需要缓存好的 Qwen3-VL 条件，因此约 62 GB 的文本编码器不会进入训练进程。

PDD 把预训练 flow 模型变成 *parallel decoder*。采样区间被离散成 `N` 个 interval，再按大小 `L` 分块；一次网络前向预测下一块中每个 interval 的平均速度，因此生成每步前进 `L` 个 interval（`NFE = N / L`）。默认配方是 `N = 32`、`L = 4`（8 NFE）。学生就是教师自己的 transformer，两个最终头（`proj_out` / `audio_proj_out`）各重复 `N` 次；关掉 LoRA 仍是教师，不需要第二份 33 B 骨干。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 Data-free 条件](#21-data-free-条件)
  - [2.2 Cache 结构](#22-cache-结构)
  - [2.3 生成 Cache](#23-生成-cache)
  - [2.4 标注 JSON 格式](#24-标注-json-格式)
  - [2.5 Ref2VA Request Cache](#25-ref2va-request-cache)
- [三、PDD LoRA 训练](#三pdd-lora-训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始（FSDP）](#32-快速开始fsdp)
    - [3.2.1 Ref2VA 训练](#321-ref2va-训练)
  - [3.3 PDD 训练参数](#33-pdd-训练参数)
  - [3.4 训练验证](#34-训练验证)
  - [3.5 Checkpoint 布局](#35-checkpoint-布局)
  - [3.6 使用 DeepSpeed-Zero-2 训练](#36-使用-deepspeed-zero-2-训练)
  - [3.7 不使用 DeepSpeed 或 FSDP 训练](#37-不使用-deepspeed-或-fsdp-训练)
  - [3.8 多机分布式训练](#38-多机分布式训练)
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

PDD 是 **data-free** 的：学生轨迹从噪声采样，因此完全不读取用于 loss 的目标视频/音频媒体。训练只读取 *条件*——`fl2va` 是 prompt，`ref2va` 是 prompt 加参考媒体。两种 `--train_mode` 都支持两条条件路径，由 `--enable_preprocess_training` 选择：

| `--train_mode` | 条件 | **Cache 路径**（`--enable_preprocess_training`） | **直接加载路径**（不加该参数） |
|----------------|------|--------------------------------------------------|-------------------------------|
| `fl2va` | prompt | 用 `generate_prompt_cache.py` 预编码，读取 `outputs.json` | 读取 `{"text": ...}` 标注（`TextDataset`），现场编码 |
| `ref2va` | prompt + 参考媒体 | 用 `generate_ref2va_request_cache.py` 预编码，读取 `outputs.json` | 读取 request 标注（`load_requests`），现场编码 prompt + 参考 latent |

- **Cache 路径**（适合长时间 / 反复训练）：Qwen3-VL embedding——`ref2va` 还有 VAE 编码的参考 latent——**只**预编码一次写入 safetensors，训练时约 62 GB 的条件器完全不加载。
- **直接加载路径**（适合快速起步或 request 集较小）：没有单独的预处理步骤；训练进程加载约 62 GB 条件器（`ref2va` 还有两个 VAE），逐条现场编码。请开启 `--low_vram`，使其仅在编码时搬上 GPU。

> 💡 两条路径都同时供训练轨迹和验证渲染使用——data-free PDD 没有 train/val 之分，因此 `--train_data_meta` 与 `--val_data_meta` 通常指向同一份标注（例如官方 `datasets/X-Fun-Videos-Audios-Demo`，标准测试数据，绝不用临时拼凑的 prompt 集）。

### 2.1 Data-free 条件

`--train_mode=fl2va`（默认配方，FL2VA / t2va packed 布局）需要 prompt 条件；`--train_mode=ref2va` 还需要参考媒体（图像 / 视频 / 音频），并默认加载 `transformer_ref`。走 **cache 路径** 时，用 `scripts/minimax_h3/generate_prompt_cache.sh`（`fl2va`）或 `scripts/minimax_h3/generate_ref2va_request_cache.sh`（`ref2va`）**只生成一次** cache；两者都在 `accelerate launch` 下多卡运行，此后 PDD 训练过程中不会加载约 62 GB 的 Qwen3-VL 条件器。走 **直接加载路径** 时没有预处理步骤：把 `--train_data_meta` 直接指向标注，训练进程现场编码（两条 `ref2va` 路径的完整流程见 **3.2.1**）。

### 2.2 Cache 结构

```
📦 datasets/
├── 📂 X-Fun-Videos-Audios-Demo/          # 官方 demo 数据集（`text` 字幕的来源）
│   └── 📄 metadata_add_width_height.json
└── 📂 minimax_h3_pdd_prompt_cache/       # 只生成一次；同时供训练与验证使用
    ├── 📄 outputs.json
    ├── 📄 00000.safetensors
    ├── 📄 00001.safetensors
    └── 📄 ...
```

`outputs.json` 是 `ImageVideoSafetensorsDataset` 读取的 `{"file_path": ".../00000.safetensors"}` 记录列表。`--train_data_dir` 是拼接到每个 `file_path` 之前的可选根目录；当 `outputs.json` 已使用仓库相对（或绝对）路径时（生成器即如此）可留空。每个 `fl2va` `.safetensors` 包含：

| 字段 | 说明 |
|------|------|
| `prompt_embeds` | MiniMax-H3 文本编码器层上的 Qwen3-VL hidden states（bfloat16） |
| `text_token_tags` | packed sequence 用的逐 token 标签（int64） |

### 2.3 生成 Cache

该步骤仅 **cache 路径** 需要；直接加载路径（**3.2.1 Route A**）直接读取标注，跳过此步。

```bash
# fl2va：多卡缓存官方 demo 数据集的 prompt 条件
bash scripts/minimax_h3/generate_prompt_cache.sh

# ref2va：缓存 request 条件（prompt embedding + 参考 latent）
bash scripts/minimax_h3/generate_ref2va_request_cache.sh
```

每个启动器只运行一次 `accelerate launch ... generate_*_cache.py`：每个 rank 走标注的交错分片，已存在的 `.safetensors` 会被跳过（resume），最后由 rank0 写出 `outputs.json`。请先编辑各 `.sh` 顶部的 `MODEL_NAME` / `DATASET_META` / `CACHE_ROOT` 变量。

> 💡 `--pretrained_model_name_or_path` 可以是转换后的 diffusers 布局，也可以是原始 MiniMax-H3 分区（例如 `MiniMax-H3/FL2VA`）。分词器从 `tokenizer/` 读取，processor 从 `processor/` 读取，条件器从 `text_encoder/` 读取。

### 2.4 标注 JSON 格式

`fl2va` 生成器读取官方 demo 数据集的标注 JSON——记录列表中 PDD 只用到 `text` 字幕（音视频数据集携带的 `file_path` / `audio_path` / `control_file_path` / `width` / `height` 字段在此被忽略）。也接受纯字符串列表，或 `{"prompt": ...}` / `{"examples": [...]}` jobs：

```json
[
  {"file_path": "train/00000001.mp4", "text": "A young woman gently turns her head to the right ...", "audio_path": "wav/00000001.wav", "type": "video"}
]
```

`ref2va` 路径——`generate_ref2va_request_cache.py`（cache）与直接加载训练皆然——则读取 request 记录的列表。既可以是显式的 `{"prompt": ..., "references": [...]}` request，其中每个 reference 是 `"image=..."` / `"video=..."` / `"audio=..."` 字符串（`predict_ref2va.py` 的 schema，按模型读取顺序排列）：

```json
[
  {"prompt": "The character turns and waves", "references": ["image=ref/face.png", "audio=ref/voice.wav"]}
]
```

也可以是——默认方式，遵循官方 demo 约定——与上面 `fl2va` 完全相同的音视频 demo 标注：`load_requests` 会从每条记录自身的 `file_path`（视频）+ `audio_path`（音频）推导出 `video=`/`audio=` reference（相对媒体路径按标注文件所在目录解析），并用其 `text` 作为 prompt。

只编码条件。分辨率和时长由训练/推理参数决定（`--video_sample_size` / `--fix_sample_size` / `--video_sample_n_frames`），不是 cache 字段。

### 2.5 Ref2VA Request Cache

`--train_mode=ref2va` 默认加载 `transformer_ref`。走 **cache 路径** 时读取 `generate_ref2va_request_cache.py` 写出的 request cache：除 `prompt_embeds` / `text_token_tags` 外，每条 request `.safetensors` 还带有 Ref2VA packed 布局所需的参考 latent，并扁平化为张量——`reference_kind_ids` / `reference_has_audio`（每个 reference 的类别与是否含音频）、`num_condition_latents` / `num_audio_condition_latents`，以及带下标的 `condition_latents_{i}` / `audio_condition_latents_{i}`。走 **直接加载路径** 时，同样的 latent 改为现场生成——`train_pdd_lora.py` 用 `load_requests` 读取 request 标注，用 Qwen3-VL 编码 prompt、用两个 VAE 编码参考——因此不需要 `.safetensors` cache（即 `--train_mode=ref2va` 且不加 `--enable_preprocess_training`）。

---

## 三、PDD LoRA 训练

### 3.1 下载预训练模型

```bash
# 创建模型目录
mkdir -p models/Diffusion_Transformer

# 下载 MiniMax-H3 官方权重
hf download MiniMax-AI/MiniMax-H3 --local-dir models/Diffusion_Transformer/MiniMax-H3
```

> 💡 加载器既接受上面转换后的 diffusers 布局，也接受 *原始* MiniMax-H3 分区（例如 `MiniMax-H3/FL2VA`）；原始分片在加载时即时转换，磁盘上不会留下中间副本。

### 3.2 快速开始（FSDP）

若已按 **2.3** 生成条件 cache、按 **3.1** 下载权重，可直接复制运行下面的命令。`scripts/minimax_h3/train_pdd_lora.sh` 是同一套 launch。

推荐使用 FSDP：虽然 PDD 不加载 Qwen3-VL，冻结的 transformer 在 bfloat16 下仍约 62 GB，必须跨 GPU 切分——FSDP（`FULL_SHARD`）会切分权重，DeepSpeed-Zero-2 不会。

必须使用 `--mixed_precision=no`。发布权重已将 `proj_out` / `audio_proj_out` 钉在 float32（`_keep_in_fp32_modules`）；由此构建的 parallel head 作为 float32 master 权重叠在 bfloat16 骨干上，训练过程不走 autocast。

**fl2va（读取 prompt cache——默认配方）：**

```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATA_DIR=""
export PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export VAL_PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
# 无 RDMA 的多机环境可设置 NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1。
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_preprocess_training \
  --train_data_dir=$DATA_DIR \
  --train_data_meta=$PROMPT_CACHE_META \
  --val_data_meta=$VAL_PROMPT_CACHE_META \
  --video_sample_n_frames=124 \
  --fix_sample_size 768 1344 \
  --train_batch_size=1 \
  --max_train_steps=3000 \
  --checkpointing_steps=200 \
  --learning_rate=1e-5 \
  --lora_learning_rate=1e-4 \
  --seed=43 \
  --output_dir="output_dir_minimax_h3_pdd_lora" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="no" \
  --adam_weight_decay=0.0 \
  --max_grad_norm=1.0 \
  --rank=64 \
  --network_alpha=64 \
  --low_vram \
  --target_name="to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear" \
  --train_mode="fl2va" \
  --pdd_num_steps=32 \
  --pdd_block_size=4 \
  --validation_steps=200 \
  --resume_from_checkpoint=latest
```

#### 3.2.1 Ref2VA 训练

`--train_mode=ref2va` 除 prompt 外还对参考媒体（图像 / 视频 / 音频）加以条件，并默认加载 `transformer_ref`。它可走任一条条件路径；下面两条命令都是在 FSDP 快速开始的基础上，只改动 mode / data 相关参数。

**Route A —— 直接加载（无 request cache）。** 把 `--train_data_meta` 直接指向 request 标注：既可以是显式的 `{"prompt", "references"}` schema，也可以是官方 `X-Fun-Videos-Audios-Demo` 标注（其自身的视频 + 音频会被当作 reference，见 **2.4**）。训练进程会加载 Qwen3-VL 条件器与两个 VAE，对每条 request 现场编码，因此请保留 `--low_vram`——它们仅在编码时搬上 GPU。

```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export REQUEST_META="datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_mode="ref2va" \
  --train_data_meta=$REQUEST_META \
  --val_data_meta=$REQUEST_META \
  --video_sample_n_frames=124 \
  --fix_sample_size 768 1344 \
  --train_batch_size=1 \
  --max_train_steps=3000 \
  --checkpointing_steps=200 \
  --learning_rate=1e-5 \
  --lora_learning_rate=1e-4 \
  --seed=43 \
  --output_dir="output_dir_minimax_h3_pdd_ref2va_lora" \
  --gradient_checkpointing \
  --gradient_checkpointing_save_on_cpu \
  --mixed_precision="no" \
  --adam_weight_decay=0.0 \
  --max_grad_norm=1.0 \
  --rank=64 \
  --network_alpha=64 \
  --low_vram \
  --target_name="to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear" \
  --pdd_num_steps=32 \
  --pdd_block_size=4 \
  --validation_steps=200 \
  --resume_from_checkpoint=latest
```

> ⚠️ Route A 会把约 62 GB 的 Qwen3-VL 条件器常驻在训练进程中（由 FSDP 与 transformer 一起切分），并在每次轨迹 reset 时用 VAE 编码该 request 的参考。若长时间或反复训练，请用 Route B 预编码一次，训练时就不再加载条件器。

**Route B —— request cache（适合长时间 / 反复训练）。** 先用 `generate_ref2va_request_cache.sh`（**2.3**）预编码一次，再加 `--enable_preprocess_training` 训练；约 62 GB 条件器不会进入训练进程：

```bash
# 1) 预编码 ref2va request 一次（多卡）：prompt embedding + 参考 latent → safetensors
bash scripts/minimax_h3/generate_ref2va_request_cache.sh

# 2) 从 cache 训练——即 Route A 改动以下参数
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export REQUEST_CACHE_META="datasets/minimax_h3_pdd_request_cache/outputs.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_mode="ref2va" \
  --enable_preprocess_training \
  --train_data_meta=$REQUEST_CACHE_META \
  --val_data_meta=$REQUEST_CACHE_META \
  ... # 其余参数与 Route A 相同
```

### 3.3 PDD 训练参数

**PDD / LoRA 参数**：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--pdd_num_steps` | 网格大小 `N` | 32 |
| `--pdd_block_size` | `L_min`：携带状态每次前进的 interval 数（`NFE = N / L`） | 4 |
| `--pdd_max_block_size` | `L_max`：抽 loss 目标时最宽的块。默认等于 `--pdd_block_size` | 4 |
| `--pdd_solver` | 估计教师平均速度的 Runge-Kutta 方法：`euler` 或 `midpoint` | `midpoint` |
| `--pdd_num_targets` | 一次学生前向在块内监督的下标 `k` 个数 | 2 |
| `--rank` | LoRA 更新矩阵的维度 | 64 |
| `--network_alpha` | LoRA 更新矩阵的缩放 | 64 |
| `--target_name` | 施加 LoRA 的模块（逗号分隔） | `to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear` |
| `--learning_rate` | parallel head 的学习率 | 1e-5 |
| `--lora_learning_rate` | 低秩更新的学习率 | 1e-4 |
| `--use_ema` | 对可训练参数做 EMA；验证和 `pdd_ema.safetensors` 都用 EMA | 默认关闭 |
| `--ema_decay` | EMA 衰减 | 0.99 |

**通用训练参数**：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--pretrained_model_name_or_path` | 预训练模型路径 | `models/Diffusion_Transformer/MiniMax-H3` |
| `--enable_preprocess_training` | 使用预处理好的 safetensors cache 训练，而非现场编码条件；`fl2va` 与 `ref2va` 都支持两条路径 | 开启 |
| `--train_data_dir` | 拼接到 `--train_data_meta` 每个 `file_path` 之前的可选根目录；当其存的是仓库相对/绝对路径时留空 | `""` |
| `--train_data_meta` | 加该参数时：cache 的 `outputs.json`。不加时：现场编码的标注（`fl2va`：`{"text": ...}`；`ref2va`：request 列表） | `datasets/minimax_h3_pdd_prompt_cache/outputs.json` |
| `--val_data_meta` | 与 `--train_data_meta` 一致：加该参数时用 cache 的 `outputs.json`，不加时用现场编码的标注（`fl2va`：`{"text": ...}`；`ref2va`：request 列表）。留空则跳过验证 | `datasets/minimax_h3_pdd_prompt_cache/outputs.json` |
| `--train_mode` | `fl2va`（t2va 布局）或 `ref2va`（`transformer_ref` + 参考媒体）；两者都可走 cache 或直接加载路径 | `fl2va` |
| `--transformer_subfolder` | Transformer 子目录。默认：`ref2va` 用 `transformer_ref`，否则 `transformer` | None |
| `--train_batch_size` | 必须为 1：每个 rank 只携带一条轨迹 | 1 |
| `--num_train_epochs` | 未指定 `--max_train_steps` 时的训练轮数。一轮对应条件集走一遍 | 100 |
| `--max_train_steps` | 总优化步数。若设置则覆盖 `--num_train_epochs` | 3000 |
| `--video_sample_n_frames` | 采样帧数，须符合视频 VAE 的 `17*n+5`（时长保持在 5 到 15 秒） | 124 |
| `--video_sample_size` | 正方形画布尺寸（高 = 宽）；必须是 32 的倍数 | 1280 |
| `--fix_sample_size` | 固定的 `[高, 宽]`，用于非正方形画布并覆盖 `--video_sample_size`；都必须是 32 的倍数 | 768 1344 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 1 |
| `--checkpointing_steps` | 每 N 步保存一次 checkpoint | 200 |
| `--seed` | 随机种子 | 43 |
| `--output_dir` | 输出目录 | `output_dir_minimax_h3_pdd_lora` |
| `--gradient_checkpointing` | 启用 activation checkpointing | - |
| `--gradient_checkpointing_save_on_cpu` | 将 transformer block 反向所需的激活卸载到 CPU | - |
| `--mixed_precision` | 使用 `no`。parallel head 保持 float32，骨干为 bfloat16 | `no` |
| `--adam_weight_decay` | AdamW weight decay | 0.0 |
| `--max_grad_norm` | 梯度裁剪阈值 | 1.0 |
| `--low_vram` | VAE 放在 CPU，仅在验证解码时搬到 GPU | - |
| `--resume_from_checkpoint` | 从 checkpoint 恢复，`"latest"` 自动选最新 | `latest` |
| `--validation_steps` | 每 N 步做一次验证 | 200 |
| `--validation_nfe` | 验证时学生的 NFE；必须整除 `--pdd_num_steps` | 8 |
| `--video_loss_weight` / `--audio_loss_weight` | 视频 + 音频联合 MSE 的权重 | 0.5 / 0.5 |

### 3.4 训练验证

验证 **不使用** `--validation_prompts`。它会按 rank 分片、以 `--validation_nfe` 生成 `--val_data_meta` 的每一条：加 `--enable_preprocess_training` 时是 cache 的 `outputs.json`，不加时是现场编码的标注——与训练路径一致，`fl2va`（prompt）与 `ref2va`（request）皆然。当 `--val_data_meta` 留空时跳过验证。

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--validation_steps` | 每 N 步验证一次 | 200 |
| `--validation_nfe` | 每条 clip 的学生前向次数（`N / NFE` 必须为整数） | 8 |

视频保存在 `output_dir/sample/`，文件名为 `sample-{step}-prompt{index}-{train_mode}-nfe{nfe}.mp4`（带音频）。

### 3.5 Checkpoint 布局

每个 `checkpoint-{step}/` 包含：

| 文件 | 作用 |
|------|------|
| `pdd.safetensors` | 现场（非 EMA）收集后的可训练张量（parallel head + LoRA），供 DDP resume |
| `pdd_ema.safetensors` | 开启 `--use_ema` 时的 EMA 导出；这是推理用文件 |
| `pdd_config.json` | rank / alpha / targets / 网格，供 `examples/minimax_h3/predict_t2v.py` 读取 |
| `optimizer.pt` / `scheduler.pt` / `scaler.pt` / `ema.pt` | DDP 训练器状态（Accelerate `--save_state` 写出的 `optimizer.bin` / `scheduler.bin` 在 resume 时同样接受） |

FSDP stage 3 / ZeRO-3 还会把 `accelerator.save_state` 写进同一目录（自动 `--save_state`），并仍然导出一份收集后的 `pdd.safetensors`（现场权重）；开启 EMA 时另写 `pdd_ema.safetensors`。更早的 checkpoint 把现场权重存在 `pdd_live.safetensors`；DDP resume 在该文件存在时仍会读取它。

### 3.6 使用 DeepSpeed-Zero-2 训练

> ⚠️ **警告**：DeepSpeed-Zero-2 只切分优化器状态和梯度，**不切分模型权重**。MiniMax-H3 transformer 约 62 GB，每张 GPU 仍会持有完整权重副本，通常会显存不足。请优先使用 FSDP（**3.2**）；下面命令仅供参考。

```sh
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATA_DIR=""
export PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export VAL_PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_preprocess_training \
  --train_data_dir=$DATA_DIR \
  --train_data_meta=$PROMPT_CACHE_META \
  --val_data_meta=$VAL_PROMPT_CACHE_META \
  ... # 与快速开始相同的 train_pdd_lora.py 参数
```

### 3.7 不使用 DeepSpeed 或 FSDP 训练

**不建议在 80 GB 卡上使用**：每张 GPU 仍会保留完整约 62 GB 的 transformer 副本。PDD 不加载 Qwen3-VL，因此 DDP 比 `scripts/minimax_h3/train_lora.py` 更轻，但默认仍应使用 FSDP（**3.2**）。从快速开始命令中去掉 `--use_fsdp` 和 FSDP wrap 参数即可；DDP resume 读取 `pdd.safetensors` 以及 `optimizer.pt` / `optimizer.bin`。

```sh
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATA_DIR=""
export PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export VAL_PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_preprocess_training \
  --train_data_dir=$DATA_DIR \
  --train_data_meta=$PROMPT_CACHE_META \
  --val_data_meta=$VAL_PROMPT_CACHE_META \
  ... # 与快速开始相同的 train_pdd_lora.py 参数
```

### 3.8 多机分布式训练

**适用场景**：更多 GPU、更快训练

#### 3.8.1 环境配置

假设 2 台机器，每台 8 卡：

**机器 0（Master）**：
```bash
export MODEL_NAME="models/Diffusion_Transformer/MiniMax-H3"
export DATA_DIR=""
export PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export VAL_PROMPT_CACHE_META="datasets/minimax_h3_pdd_prompt_cache/outputs.json"
export MASTER_ADDR="192.168.1.100"  # Master 机器 IP
export MASTER_PORT=10086
export WORLD_SIZE=2                  # 机器总数
export NUM_PROCESS=16                # 总进程数 = 机器数 × 8
export RANK=0                        # 当前机器 rank（0 或 1）
# 无 RDMA 的多机环境可设置 NCCL_IB_DISABLE=1 和 NCCL_P2P_DISABLE=1。
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="no" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock \
    --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT \
    --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False \
    scripts/minimax_h3/train_pdd_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_preprocess_training \
  --train_data_dir=$DATA_DIR \
  --train_data_meta=$PROMPT_CACHE_META \
  --val_data_meta=$VAL_PROMPT_CACHE_META \
  ... # 与快速开始相同的 train_pdd_lora.py 参数
```

**机器 1（Worker）**：
```bash
export RANK=1  # 注意这里是 1
# 其余环境变量与机器 0 相同

# 使用与机器 0 相同的 accelerate launch 命令
```

#### 3.8.2 多机训练注意事项

- **网络要求**：
   - 推荐 RDMA/InfiniBand（高性能）
   - 无 RDMA 时，添加环境变量：
     ```bash
     export NCCL_IB_DISABLE=1
     export NCCL_P2P_DISABLE=1
     ```

- **数据同步**：所有机器必须能访问相同的条件 cache 和模型路径（NFS/共享存储）

## 四、推理测试

PDD 推理从 `pdd_ema.safetensors` 挂上 parallel head 和 LoRA（没有 EMA 文件时回退到 `pdd.safetensors`），再按 `num_inference_steps` NFE 采样。使用 `examples/minimax_h3/predict_t2v.py`；不要同时设置 `lora_path`（`lora_path` 与 `pdd_lora_path` 不能一起用）。

默认配方（`N = 32`、`L = 4`）以 **8** 步推理。`num_inference_steps` 必须整除 `pdd_config.json` 中的 `pdd_num_steps`。若仍为教师默认的 40，脚本会自动改成 `N / L`。

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
| `pdd_lora_path` | PDD checkpoint 目录（优先加载 `pdd_ema.safetensors`，否则 `pdd.safetensors`，外加 `pdd_config.json`）或 `.safetensors` 文件 | `output_dir_minimax_h3_pdd_lora/checkpoint-3000` |
| `lora_path` | Turbo/PEFT LoRA；**不能**与 `pdd_lora_path` 同时使用 | `None` |
| `sample_size` | 生成视频分辨率 `[height, width]`；宽高必须是 32 的倍数。设为 `None` 时使用 MiniMax-H3 自带的 16:9 画布（768x1344） | `[768, 1344]` |
| `video_length` | 生成帧数，会向上取整到视频 VAE 可解码的下一个 `17*n+5`（时长保持在 5 到 15 秒） | 124 |
| `fps` | 每秒帧数（MiniMax-H3 固定以 24 fps 生成） | 24 |
| `weight_dtype` | 模型权重精度，不支持 bf16 的 GPU 请使用 `torch.float16` | `torch.bfloat16` |
| `prompt` | 描述生成内容的正向提示词 | `"A red fox trotting..."` |
| `seed` | 用于复现的随机种子 | 43 |
| `num_inference_steps` | 学生 NFE。默认 PDD 配方用 8（不是教师的 40） | 8 |
| `guidance_scale` | 引导强度。发布权重已做 guidance 蒸馏：保持 1 时每步只做一次前向、不走 CFG | 1 |
| `flow_shift` | 视频调度的指数 sigma shift，`None` 时沿用权重自带值（12.0） | `None` |
| `audio_flow_shift` | 音频调度的指数 sigma shift，`None` 时沿用权重自带值（3.0） | `None` |
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

> 💡 transformer 在 bfloat16 下有 61.7 GB，Qwen3-VL 条件器还有 62.1 GB，因此单张 80 GB 卡需要使用 `model_cpu_offload` 或 `model_group_offload`。推理会加载文本编码器；训练不会。

### 4.2 单 GPU 推理

运行单卡推理：

```bash
python examples/minimax_h3/predict_t2v.py
```

按需编辑 `examples/minimax_h3/predict_t2v.py`。PDD 推理请重点关注以下参数：

```python
# 根据 GPU 显存选择
GPU_memory_mode = "model_cpu_offload"
# 您的实际模型路径
model_name = "models/Diffusion_Transformer/MiniMax-H3"
# PDD checkpoint 目录或权重文件；目录优先加载 pdd_ema.safetensors。rank / alpha / targets / 网格从 pdd_config.json 读取
pdd_lora_path = "output_dir_minimax_h3_pdd_lora/checkpoint-3000"
# 设置 pdd_lora_path 时必须保持 None
lora_path = None
# 学生 NFE；默认 N=32 / L=4 配方为 8。留在 40 时脚本会改成 N / L
num_inference_steps = 8
# 按要生成的内容填写
prompt = "A red fox trotting through a snowy pine forest, snow crunching underfoot"
# ...
```

图生视频和 Ref2VA 在 `examples/minimax_h3/predict_i2v.py`、`examples/minimax_h3/predict_ref2va.py` 里使用同样的 `pdd_lora_path` 字段。Ref2VA 需要 `--train_mode=ref2va` 训出的 checkpoint。

### 4.3 多 GPU 并行推理

**适用场景**：高分辨率生成、推理加速

#### 安装并行推理依赖

```bash
pip install xfuser yunchang
```

#### 配置并行策略

编辑 `examples/minimax_h3/predict_t2v.py`：

```python
# 保证 ulysses_degree × ring_degree = GPU 数量
# 例如使用 2 张 GPU：
ulysses_degree = 2  # 头维度并行
ring_degree = 1     # 序列维度并行
```

**配置原则**：
- `ulysses_degree` 必须能整除模型的注意力头数
- `ring_degree` 在序列维切分，影响通信开销；头数能均分时尽量不用
- 多卡走 xfuser 序列并行路径，与 `*cpu_offload*` 显存模式 **不兼容**（accelerate offload hook 占用单一 device）；此时用 `model_full_load` / `model_full_load_and_qfloat8`，并用 `fsdp_dit` / `fsdp_text_encoder` 省显存
- 序列并行需要可用的 FlashAttention。没有时改为独立的单卡任务（`CUDA_VISIBLE_DEVICES=i`，`ulysses_degree = 1`，`ring_degree = 1`）

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

- **PDD 论文**：https://arxiv.org/abs/2607.26004
- **MiniMax-H3 官方 GitHub**：https://github.com/MiniMax-AI/MiniMax-H3
- **官方 GitHub**：https://github.com/aigc-apps/VideoX-Fun
- **基础 MiniMax-H3 LoRA 训练**：`scripts/minimax_h3/README_TRAIN_LORA.md`
- **MiniMax-H3 Fun 控制训练**：`scripts/minimax_h3_fun/README_TRAIN.md`
