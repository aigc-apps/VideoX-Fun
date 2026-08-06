# LingBot-Video TI2V 训练指南

本文档提供 LingBot-Video（单流 joint DiT + Qwen3-VL 文本编码器）的完整 **ti2v（首帧图 + 文本生视频）全参数训练**流程，包括环境配置、数据准备、FSDP 分布式训练和推理测试。

> **说明**：LingBot-Video 与 LingBot-World（Wan2.2 双 Transformer + 相机控制）架构完全不同：
> - **单流 joint DiT**：视频 token 与文本 token 拼接后做全自注意力（`LingBotVideoTransformer3DModel`）；
> - **Qwen3-VL 文本编码器**：prompt 经 chat 模板编码，首帧图像作为视觉 token 一并进入文本序列；
> - **Flow Matching**：`x_t = (1-σ)·x0 + σ·noise`，目标 `noise - x0`，transformer 接收 `timestep = σ·1000`；
> - **ti2v 首帧条件**：首帧除进入 Qwen3-VL 文本序列外，还单独经 VAE 编码得到 cond_latent，在每个去噪步覆写回 latent 的时间前缀（inpainting 语义），loss 只在非条件帧上计算。
>
> 训练脚本为 `scripts/lingbot_video/train.py`，默认启动配置为 `scripts/lingbot_video/train.sh`（dense 1.3B）与 `scripts/lingbot_video/train_moe.sh`（MoE 30B-A3B），均使用 FSDP FULL_SHARD。

---

## 目录
- [一、环境配置](#一环境配置)
- [二、数据准备](#二数据准备)
  - [2.1 快速测试数据集](#21-快速测试数据集)
  - [2.2 metadata_lingbot_video_add_width_height.json 格式](#22-metadata_lingbot_video_add_width_heightjson-格式)
  - [2.3 caption 必须是结构化 JSON caption](#23-caption-必须是结构化-json-caption)
- [三、全量参数训练](#三全量参数训练)
  - [3.1 下载预训练模型](#31-下载预训练模型)
  - [3.2 快速开始（FSDP）](#32-快速开始fsdp)
  - [3.3 训练常用参数解析](#33-训练常用参数解析)
  - [3.4 可训练模块选择](#34-可训练模块选择)
  - [3.5 断点续训](#35-断点续训)
  - [3.6 训练流程原理](#36-训练流程原理)
- [四、推理测试](#四推理测试)
- [五、常见问题](#五常见问题)

---

## 一、环境配置

**方式 1：使用 requirements.txt**

```bash
pip install -r requirements.txt
```

**方式 2：手动安装依赖**

```bash
pip install Pillow einops safetensors timm tomesd librosa "torch>=2.1.2" torchdiffeq torchsde decord datasets numpy scikit-image scipy
pip install omegaconf SentencePiece imageio[ffmpeg] imageio[pyav] tensorboard beautifulsoup4 ftfy func_timeout onnxruntime
pip install "peft>=0.17.0" "accelerate>=0.25.0" "gradio>=3.41.2" "diffusers>=0.30.1" "transformers>=4.46.2"
pip install modelscope openpyxl
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-python-headless
```

> **说明**：Qwen3-VL 文本编码器要求较新的 `transformers` —— `videox_fun/models/__init__.py` 从 transformers 导入 `Qwen3VLForConditionalGeneration`，导入失败时会退化为 `None` 并打印 `Your transformers version is too old to load Qwen3VLForConditionalGeneration`。看到该提示（或读取 config 时报 `KeyError: 'qwen3_vl'`）请升级 transformers。
>
> **说明**：可选的 prompt rewriter（见 [2.3](#23-caption-必须是结构化-json-caption)）要求更新的依赖栈（`transformers>=5.x`，含 `qwen3_5` 模块）以及位于 `repo/lingbot-video/rewriter` 的官方 rewriter 代码包，建议在独立 venv 中运行，避免污染训练环境。

---

## 二、数据准备

### 2.1 快速测试数据集

我们提供了一个测试的数据集，其中包含若干训练数据。

```bash
# 下载官方示例数据集
modelscope download --dataset PAI/X-Fun-Videos-Demo --local_dir ./datasets/X-Fun-Videos-Demo
```

该数据集包含 `train/` 下 16 个 832x480 视频，以及 `metadata_lingbot_video_add_width_height.json`，其 `text` 字段已经是 LingBot-Video 结构化 JSON caption（见 [2.3](#23-caption-必须是结构化-json-caption)），可直接用于训练。LingBot-Video 训练**不需要任何相机轨迹 / action 文件**，ti2v 的条件帧也会**自动从每条视频的第一帧取**（同时作为 Qwen3-VL 视觉输入与 VAE cond_latent），无需单独准备图片文件。

### 2.2 metadata_lingbot_video_add_width_height.json 格式

标准 VideoX-Fun 格式，每个视频一个条目：

```json
[
    {
        "file_path": "train/00000000.mp4",
        "text": "{\"comprehensive_description\": {...}, \"prominent_elements\": [...], \"camera_info\": {...}}",
        "width": 832,
        "height": 480,
        "type": "video"
    },
    {
        "file_path": "train/00000001.mp4",
        "text": "{\"comprehensive_description\": {...}, \"prominent_elements\": [...], \"camera_info\": {...}}",
        "width": 832,
        "height": 480,
        "type": "video"
    }
]
```

- `file_path`：相对 `--train_data_dir` 的视频路径（设 `--train_data_dir=""` 时可直接使用绝对路径）；
- `text`：**序列化为单个字符串的结构化 JSON caption**。DiT 仅用 rewriter 格式的 JSON caption 训练，自然语言 caption 属于分布外输入，会降低微调效果。`train.py` 启动时会用 `is_valid_caption` 逐条校验，并打印不合规条目的数量告警。见 [2.3](#23-caption-必须是结构化-json-caption)；
- `width` / `height`：原始视频宽高，用于选择宽高比分桶。**建议提供**；缺省时 `AspectRatioBatchImageVideoSampler` 会在分桶时用 OpenCV 实际读取视频（每轮额外 IO）。可用 `scripts/process_json_add_width_and_height.py` 为已有 metadata 补全这两个字段；
- `type`：必须为 `"video"`（缺省按图片处理，使用 `--image_sample_size`）。

### 2.3 caption 必须是结构化 JSON caption

rewriter 权重默认路径为 `models/Diffusion_Transformer/Qwen3.6-27B` 和
`models/Diffusion_Transformer/lingbot-video-rewriter-lora`：

```bash
modelscope download --model Qwen/Qwen3.6-27B --local_dir models/Diffusion_Transformer/Qwen3.6-27B
modelscope download --model Robbyant/lingbot-video-rewriter-lora --local_dir models/Diffusion_Transformer/lingbot-video-rewriter-lora
```

合法 caption 是一个 JSON 对象，必须含有 `is_valid_caption` 校验的三个顶层字段
（定义于 `videox_fun/models/lingbot_video_rewriter.py`，也是该 schema 的单一权威来源）：

```json
{
  "comprehensive_description": {
    "scene_content_description": "A drone flies over a mountain ridge at sunrise ...",
    "camera_movement_description": "The camera pushes forward slowly."
  },
  "prominent_elements": [
    {
      "name": "mountain ridge",
      "description": "a snow-dusted ridge line",
      "actions": [{"timestamp": "[0.0s - 3.3s]", "action": ""}],
      "location": "center",
      "relative_size": "large",
      "shape_and_color": "grey rock with white snow",
      "texture": "rough",
      "appearance_details": "",
      "relationship": "",
      "orientation": "",
      "pose": "",
      "expression": "",
      "clothing": "",
      "gender": "",
      "skin_tone_and_texture": ""
    }
  ],
  "camera_info": {
    "color": "Cool",
    "frame_size": "Wide",
    "shot_type_angle": "Aerial",
    "lens_size": "Wide",
    "composition": "Balanced",
    "lighting": "Soft light",
    "lighting_type": "Daylight"
  }
}
```

`camera_info` 的可选值枚举在 `CAMERA_CHOICES` 中；同一模块的 `build_caption` / `element` / `cam`
可以代码方式拼装 caption。

**数据集批量转换** —— 训练前先用官方 prompt rewriter 重写 metadata 的 `text` 字段
（需要 rewriter 基座 VLM + LoRA adapter 权重）：

```bash
export REWRITER_BASE_MODEL=models/Diffusion_Transformer/Qwen3.6-27B
export REWRITER_ADAPTER=models/Diffusion_Transformer/lingbot-video-rewriter-lora
python scripts/lingbot_video/prepare_captions.py \
    --metadata datasets/my_dataset/metadata.json \
    --data_root datasets/my_dataset \
    --output datasets/my_dataset/metadata_json.json \
    --mode ti2v --duration 3.3
# 然后在 train.sh 中：DATASET_META_NAME="datasets/my_dataset/metadata_json.json"
```

- `--mode`：`t2v` / `ti2v` / `t2i`；`ti2v` 下会读取视频首帧（decord，失败时回退 OpenCV）并送入 rewriter，建议与训练任务保持一致；
- `--duration`：传给 rewriter 的片段时长（秒），需与训练片段对齐（`video_sample_n_frames / fps`，如 81 帧 @ 24fps ≈ 3.3s）；
- `--base` / `--adapter`：rewriter 权重路径，也可用环境变量 `REWRITER_BASE_MODEL` / `REWRITER_ADAPTER`；
- 已是合法 JSON caption 的条目默认保留不动，除非传 `--overwrite`；每处理完一条就重写一次输出文件（可中断继续），且从不原地修改 `--metadata`；
- rewriter 失败的条目会保留原文本并在末尾汇总中报告，训练前需修正或重跑。

**单条 prompt**（例如准备 `--validation_prompts`）—— 统一入口是 `ensure_json_caption`：
已合法的 caption 直接透传，否则会加载 rewriter 改写、释放模型并缓存结果：

```python
from PIL import Image
from videox_fun.models.lingbot_video_rewriter import ensure_json_caption

caption = ensure_json_caption(
    "A drone slowly flies over the mountains, clouds drift in the background.",
    mode="ti2v", duration=3.3,
    first_frame=Image.open("asset/1.png").convert("RGB"),   # 仅 ti2v 需要
    cache_file="samples/caption_cache.json",
)
```

---

## 三、全量参数训练

### 3.1 下载预训练模型

LingBot-Video 提供两个变体（diffusers 格式目录，内含 `transformer` / `vae` / `text_encoder` / `processor` / `scheduler` 子目录）：

| 模型 | 说明 |
| --- | --- |
| `lingbot-video-dense-1.3b` | Dense 1.3B，单/双卡即可训练，推荐先用它验证流程（`train.sh`） |
| `lingbot-video-moe-30b-a3b` | MoE 30B-A3B，FSDP FULL_SHARD 多卡训练（`train_moe.sh`，建议 ≥8×80GB） |

模型目录结构（`train.py` 按路径逐个加载子目录，五个子目录必须齐备）：

```
lingbot-video-dense-1.3b/
├── transformer/       # LingBotVideoTransformer3DModel（唯一参与训练的模块）
├── vae/               # AutoencoderKLQwenImage（冻结）
├── text_encoder/      # Qwen3-VL（冻结）
├── processor/         # Qwen3-VL processor（AutoProcessor）
└── scheduler/         # FlowUniPCMultistepScheduler（仅读取 sigma_max / sigma_min）
```

MoE 模型额外附带一个 `refiner` DiT，训练不使用它（仅 `examples/lingbot_video/predict_t2v_refine.py` 使用）。

### 3.2 快速开始（FSDP）

修改 `scripts/lingbot_video/train.sh`（dense 1.3B）或 `scripts/lingbot_video/train_moe.sh`（MoE 30B）顶部三个环境变量后执行：

```bash
export MODEL_NAME="models/Diffusion_Transformer/lingbot-video-dense-1.3b"
export DATASET_NAME="datasets/X-Fun-Videos-Demo"
export DATASET_META_NAME="datasets/X-Fun-Videos-Demo/metadata_lingbot_video_add_width_height.json"

sh scripts/lingbot_video/train.sh
```

脚本默认以 **FSDP FULL_SHARD + LingBotVideoBlock 为 wrap 单元**启动（MoE 30B 的 expert 必须与所在 block 一起分片，故 wrap 类固定为 `LingBotVideoBlock`）。`train.sh` 完整内容：

```bash
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap=LingBotVideoBlock --fsdp_sharding_strategy "FULL_SHARD" \
    --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" \
    --fsdp_cpu_ram_efficient_loading False scripts/lingbot_video/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=1 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_lingbot_video_ti2v" \
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
  --train_shift=3.0 \
  --trainable_modules "." \
  --resume_from_checkpoint=latest
```

`train_moe.sh` 与上述命令的差异仅在于：`MODEL_NAME=models/Diffusion_Transformer/lingbot-video-moe-30b-a3b`、`--output_dir="output_dir_lingbot_video_moe_ti2v"`，其余参数（包括 640 的采样尺寸）完全相同。

> **说明**：命令行未传 `--use_fsdp` —— `train.py` 会自行检测 accelerate 的 FSDP plugin，根据 sharding strategy 推导 FSDP stage 并切到分片保存流程。两个脚本均未设置 `--validation_prompts`，因此默认不做周期验证采样（见 [3.3](#33-训练常用参数解析)）。

> **显存建议**：`--train_batch_size=1` + `--gradient_checkpointing` 下，dense 1.3B 在 2×H20（97GB）可训 81 帧 480p；两个脚本自带的采样尺寸是 640，显存开销相应更高 —— OOM 时可调低 `--video_sample_size` / `--token_sample_size`。MoE 30B 建议 ≥8×80GB。Qwen3-VL 文本编码器每步在线编码（冻结 bf16），本身也通过 FSDP 分片（包裹其 `Qwen3VLTextDecoderLayer` / `Qwen3VLVisionBlock` 层）。

训练中可用 tensorboard 观察：

```bash
tensorboard --logdir=output_dir_lingbot_video_ti2v
```

### 3.3 训练常用参数解析

| 参数 | 说明 |
| --- | --- |
| `--pretrained_model_name_or_path` | LingBot-Video 模型根目录（需含 `transformer/ vae/ text_encoder/ processor/ scheduler/`） |
| `--train_data_dir` / `--train_data_meta` | 数据集根目录 + metadata json；`file_path` 相对根目录解析 |
| `--train_batch_size=1` | **推荐恒为 1**。Qwen3-VL 变长文本，batch>1 会触发 padding + attention_mask 路径，可用但低效；提升等效 batch 请用 `--gradient_accumulation_steps` |
| `--video_sample_n_frames=81` | 采样帧数；collate 会将本批帧数向下对齐到 `4n+1`（VAE 时间压缩率 4） |
| `--video_sample_stride=1` | 采样片段内的帧间隔 |
| `--video_repeat=1` | 在数据集列表中把每个视频条目重复这么多次（用于平衡图片/视频混合数据集） |
| `--video_sample_size=640` / `--image_sample_size=640` | 视频/图像条目的短边分桶尺寸（两个启动脚本均为 640） |
| `--token_sample_size=640` | `--training_with_video_token_length` 所用 token 预算 `video_sample_n_frames × token_sample_size²` 的参考分辨率 |
| `--enable_bucket` + `--random_hw_adapt` | 宽高比分桶 + 逐批随机分辨率 |
| `--training_with_video_token_length` | 保持视频 token 总量恒定的同时自适应 H/W（分辨率高时帧数变少） |
| `--fix_sample_size H W` | 强制固定分桶尺寸，同时会禁用 `--random_hw_adapt` / `--training_with_video_token_length` |
| `--random_ratio_crop` | 用随机宽高比裁剪代替“最近分桶 resize” |
| `--uniform_sampling` | σ 分层均匀采样：每个 rank 组仅从 `--train_sampling_steps` 网格的自己那一段采样，整体上覆盖完整噪声区间；不加时 σ 下标由 `--weighting_scheme` 对应的密度采样（`--logit_mean` / `--logit_std` / `--mode_scale`） |
| `--train_sampling_steps=1000` | 训练 sigma 表的离散 σ 级数 |
| `--train_shift=3.0` | sigma shift，与推理默认 shift=3.0 保持一致：`σ' = s·σ/(1+(s-1)·σ)` |
| `--weighting_scheme` | loss 加权（`sigma_sqrt` / `logit_normal` / `mode` / `cosmap` / `none`），默认 `none`（等权） |
| `--gradient_checkpointing` | 梯度检查点，显存约省一半，速度略降 |
| `--trainable_modules "."` | 全参数训练；见 [3.4](#34-可训练模块选择) |
| `--trainable_modules_low_learning_rate` | 同样的子串匹配，但这部分参数使用 `learning_rate / 2` |
| `--transformer_path` / `--vae_path` | 训练开始前从外部 `.safetensors` / `.pt` 加载 transformer / VAE 权重 |
| `--vae_mini_batch=1` | VAE 分块编码批大小 |
| `--low_vram` | 低显存模式：VAE/Qwen3-VL 按需搬移 GPU/CPU |
| `--max_grad_norm=0.05` | 梯度裁剪。非 FSDP 模式下还会用 `--initial_grad_norm_ratio` / `--abnormal_norm_clip_start` 做阀值衰减与异常梯度抑制；FSDP 下直接按 `max_grad_norm` 裁剪 |
| `--checkpointing_steps=50` / `--checkpoints_total_limit` | 保存间隔 / 最多保留的 checkpoint 数量 |
| `--validation_prompts` + `--validation_paths` | 每 `--validation_steps` 步与每 `--validation_epochs` 个 epoch 用 `LingBotVideoI2VPipeline` 跑一次 ti2v 采样（prompt 与首帧图一一对应，启动时 assert 校验）。采样参数固定为 `guidance_scale=3.0`、25 步、`shift=--train_shift`、fps 24，分辨率由 `image_sample_size²` 与图片宽高比推得；视频保存到 `output_dir/sample/sample-{step}-rank{rank}-image-{i}.mp4`。验证 prompt 也必须是 JSON caption |
| `--use_ema` | 仅非 FSDP 可用 —— FSDP FULL_SHARD 下会抛 `NotImplementedError` |
| `--report_model_info` | 向 tensorboard 记录逐参数梯度范数（仅非 FSDP） |
| `--resume_from_checkpoint=latest` | 从最近 checkpoint 续训 |

### 3.4 可训练模块选择

`--trainable_modules` 按**子串匹配**过滤参数名，默认 `["."]` 即全参数：

```bash
# 全参数（默认）
--trainable_modules "."

# 只训注意力投影
--trainable_modules "attn"

# 只训 FFN / MoE expert
--trainable_modules "ffn" "experts"
```

被 `--trainable_modules_low_learning_rate` 匹配到的参数则以一半学习率训练；同时被两个列表匹配的参数归入全速率分组。

### 3.5 断点续训

- FSDP（两个启动脚本的默认方式）下每个 `checkpoint-*` 目录内保存：
  - `diffusion_pytorch_model.safetensors`（全量权重，主进程聚合并转为 bf16 导出，可直接喂推理脚本）；
  - accelerate sharded state（优化器/scheduler，用于续训）；
  - `sampler_pos_start.pkl`（采样器位置 + epoch，续训时恢复数据顺序）。
- 非 FSDP 模式下 checkpoint 保存的是 diffusers 风格的 `transformer/` 子目录（开启 `--use_ema` 时额外保存 `transformer_ema/`）。
- 加 `--resume_from_checkpoint=latest` 即可自动续训；加载时会将采样器位置回退 `dataloader_num_workers × num_processes × 2` 个样本，以补偿预取的批次。
- 训练结束时也会额外保存一份 `checkpoint-{global_step}`。

### 3.6 训练流程原理

每个训练步（与推理路径严格对齐）：

1. **数据**：DataLoader 取到 `(B, C, T, H, W) ∈ [-1, 1]`（先 resize + 中心裁剪到分桶尺寸，再按 mean/std 0.5 归一化），首帧 `[:, :, 0]` 自动作为条件图；本次训练的第一个 batch 会写到 `output_dir/sanity_check/` 供核对。数据集默认带 10% 文本 dropout（`text_drop_ratio=0.1`），即部分样本以空 prompt 训练以支持 classifier-free guidance；
2. **VAE 编码**（冻结、bf16；仅 `latents_mean/std` 归一化在 fp32 下计算）：
   - 整段视频 → `latents`（经 `latents_mean/std` 归一化到 DiT 空间）；
   - 首帧单独编码 → `cond_latent`（时间维 1 帧）；
3. **Qwen3-VL 编码**（冻结、bf16、no_grad）：prompt 套 chat 模板，首帧经 `smart_resize`（对齐到 `patch_size×merge_size`）作为图像 token 一并编码，得到 `prompt_embeds` 与 `prompt_mask`；
4. **Flow matching 加噪**：
   - σ 从 sigma 表采样（`linspace(sigma_max, sigma_min, N+1)[:-1]` 后施加 `--train_shift`，`N = --train_sampling_steps`；`sigma_max` / `sigma_min` 取自预训练 `scheduler/` 配置）；
   - `x_t = (1-σ)·x0 + σ·noise`；
5. **ti2v inpainting**：把 `cond_latent` 覆写到 `x_t` 的时间前缀（干净帧不加噪），并记录 `frame_mask`（条件帧=0）；
6. **前向**：`transformer(x_t, σ·1000, prompt_embeds, encoder_attention_mask=prompt_mask)`（timestep 缩放与 pipeline 的 `_transformer_timestep` 一致）；
7. **loss**：`Σ(MSE(noise_pred, noise - x0) · frame_mask · weighting) / (frame_mask.sum() · C · H · W)` —— 条件帧不参与，`weighting` 来自 `--weighting_scheme`（`none` 时全为 1）。分母中的 `C·H·W` 是对 loss 数值的一个常数缩放。

> norm / router / modulation / scale_shift_table 等数值敏感模块按模型自带规则保持 fp32，其余参数 bf16，与推理一致。FSDP 下则会将全部参数与 buffer 强制转为 bf16，因为同一个 FSDP 分片要求统一 dtype。

---

## 四、推理测试

训练产出的 `checkpoint-*/diffusion_pytorch_model.safetensors` 可直接作为 `examples/lingbot_video/predict_i2v.py` 的 `transformer_path`：

```python
# examples/lingbot_video/predict_i2v.py
model_name       = "models/Diffusion_Transformer/lingbot-video-dense-1.3b"
transformer_path = "output_dir_lingbot_video_ti2v/checkpoint-500/diffusion_pytorch_model.safetensors"
shift            = 3.0     # 与训练的 --train_shift 保持一致
```

然后：

```bash
python examples/lingbot_video/predict_i2v.py
```

脚本会把微调权重加载到 base 模型上（`load_state_dict(..., strict=False)`）做 ti2v 采样；`predict_t2v.py` / `predict_t2v_refine.py` 同理（后者需配套 refiner 权重，而 refiner 只随 MoE 模型以其 `refiner` 子目录提供）。

`predict_i2v.py` 自身也会先把普通 `prompt` 改写为 JSON caption：在加载任何生成模型之前调用 `ensure_json_caption(..., mode="ti2v", duration=round(video_length / fps, 2), first_frame=validation_image, base=rewriter_base_model, adapter=rewriter_lora_path)`，并将结果缓存到 `save_path/caption_cache.json`。脚本中的 `rewriter_base_model` / `rewriter_lora_path` 已经指向 [2.3](#23-caption-必须是结构化-json-caption) 的权重路径；直接传入合法 JSON caption 则会跳过 rewriter。

---

## 五、常见问题

1. **OOM**：优先 `--gradient_checkpointing` + `--train_batch_size=1`；再开 `--low_vram`；或降 `--video_sample_n_frames` / `--video_sample_size`。
2. **loss 不降**：确认 `--train_shift` 与推理一致（3.0）；确认 caption 与视频内容匹配；小数据集上可先调大 `--learning_rate`（如 5e-5）观察是否有响应。
3. **MoE 30B 训练**：代码路径天然支持（FSDP FULL_SHARD 按 `LingBotVideoBlock` 分片，expert 随 block 走），但建议先在 dense 1.3B 上验证数据管线。
4. **EMA**：`--use_ema` 仅在非 FSDP（单卡/DDP）模式可用；FSDP FULL_SHARD 与 EMAModel 不兼容。
5. **batch > 1**：可用，但 Qwen3-VL 会做右 padding，DiT 走 attention_mask 路径，效率低于 batch=1。
6. **启动时提示 `... dataset captions are NOT structured JSON captions`**：metadata 仍是自然语言文本，请按 [2.3](#23-caption-必须是结构化-json-caption) 转换。训练仍会继续，但 DiT 拿到的是分布外 prompt。
7. **验证报错**：`log_validation` 会捕获所有异常，打印 `Eval error on rank ...` 后继续训练，失败的采样不会中止任务 —— 根据打印信息排查（常见原因是验证 prompt 不是 JSON caption，或 `--validation_paths` 图片缺失）。
