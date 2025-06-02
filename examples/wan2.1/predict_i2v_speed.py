import os
import sys
import argparse
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

# 添加项目根目录到系统路径
current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                              WanT5EncoderModel, WanTransformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import WanI2VPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                   save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser()

    # 基础模型与推理参数
    parser.add_argument("--GPU_memory_mode", type=str, default="sequential_cpu_offload")
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--fsdp_dit", action="store_true")
    parser.add_argument("--fsdp_text_encoder", action="store_true")
    parser.add_argument("--compile_dit", action="store_true")

    # TeaCache 参数
    parser.add_argument("--enable_teacache", action="store_true")
    parser.add_argument("--teacache_threshold", type=float, default=0.10)
    parser.add_argument("--num_skip_start_steps", type=int, default=5)
    parser.add_argument("--teacache_offload", action="store_true")

    # CFG Skip 参数
    parser.add_argument("--cfg_skip_ratio", type=float, default=0.0)

    # Riflex 参数
    parser.add_argument("--enable_riflex", action="store_true")
    parser.add_argument("--riflex_k", type=int, default=6)

    # 模型路径和配置
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml")
    parser.add_argument("--model_name", type=str, default="models/Diffusion_Transformer/Wan2.1-I2V-14B-480P")
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--sample_size", nargs='+', type=int, default=[480, 832])
    parser.add_argument("--video_length", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--weight_dtype", type=str, default="bfloat16")

    # 输入图像相关
    parser.add_argument("--validation_image_start", type=str, default="asset/1.png")
    parser.add_argument("--validation_image_end", type=str, default=None)

    # 推理参数
    parser.add_argument("--prompt", type=str, default="一只棕色的狗摇着头，坐在舒适房间里的浅色沙发上。在狗的后面，架子上有一幅镶框的画，周围是粉红色的花朵。房间里柔和温暖的灯光营造出舒适的氛围。")
    parser.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--lora_weight", type=float, default=0.55)
    parser.add_argument("--save_path", type=str, default="samples/wan-videos-i2v")

    # 采样器设置
    parser.add_argument("--sampler_name", type=str, choices=["Flow", "Flow_Unipc", "Flow_DPM++"], default="Flow_Unipc")
    parser.add_argument("--shift", type=float, default=3.0)
    return parser.parse_args()

args = parse_args()

# 获取参数
GPU_memory_mode         = args.GPU_memory_mode
ulysses_degree          = args.ulysses_degree
ring_degree             = args.ring_degree
fsdp_dit                = args.fsdp_dit
fsdp_text_encoder       = args.fsdp_text_encoder
compile_dit             = args.compile_dit
enable_teacache         = args.enable_teacache
teacache_threshold      = args.teacache_threshold
num_skip_start_steps    = args.num_skip_start_steps
teacache_offload        = args.teacache_offload
cfg_skip_ratio          = args.cfg_skip_ratio
enable_riflex           = args.enable_riflex
riflex_k                = args.riflex_k
config_path             = args.config_path
model_name              = args.model_name
transformer_path        = args.transformer_path
vae_path                = args.vae_path
lora_path               = args.lora_path
sample_size             = args.sample_size
video_length            = args.video_length
fps                     = args.fps
weight_dtype            = torch.bfloat16 if args.weight_dtype == "bfloat16" else torch.float16
prompt                  = args.prompt
negative_prompt         = args.negative_prompt
guidance_scale          = args.guidance_scale
seed                    = args.seed
num_inference_steps     = args.num_inference_steps
lora_weight             = args.lora_weight
save_path               = args.save_path
sampler_name            = args.sampler_name
shift                   = args.shift
validation_image_start  = args.validation_image_start
validation_image_end    = args.validation_image_end

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

# 初始化模型组件
transformer = WanTransformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# 获取VAE
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# 获取分词器
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# 获取文本编码器
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# 获取CLIP图像编码器
clip_image_encoder = CLIPModel.from_pretrained(
    os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
).to(weight_dtype)
clip_image_encoder = clip_image_encoder.eval()

# 获取调度器
scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}
Choosen_Scheduler = scheduler_dict[sampler_name]

if sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
    config['scheduler_kwargs']['shift'] = 1
scheduler = Choosen_Scheduler(
    **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# 创建Pipeline
pipeline = WanI2VPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    clip_image_encoder=clip_image_encoder
)

# 分布式设置
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

# 编译优化
if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    print("Add Compile")

# 内存优化策略
if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

for i in range(2):
    # TeaCache配置
    coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
    if coefficients is not None:
        print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
        )

    # CFG跳过配置
    if cfg_skip_ratio is not None and cfg_skip_ratio > 0:
        print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
        pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)

    # 随机种子
    generator = torch.Generator(device=device).manual_seed(seed)

    # LoRA加载
    if lora_path is not None:
        pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device)

    # 生成视频
    with torch.no_grad():
        video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
        latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

        if enable_riflex:
            pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)

        # 输入图像处理
        input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, validation_image_end, video_length=video_length, sample_size=sample_size)

        # 执行推理
        sample = pipeline(
            prompt, 
            num_frames=video_length,
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            video=input_video,
            mask_video=input_video_mask,
            clip_image=clip_image,
            shift=shift,
        ).videos

    # LoRA卸载
    if lora_path is not None:
        pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)

    # 保存结果
    def save_results():
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        index = len([path for path in os.listdir(save_path)]) + 1
        prefix = str(index).zfill(8)
        if video_length == 1:
            video_path = os.path.join(save_path, prefix + ".png")
            image = sample[0, :, 0]
            image = image.transpose(0, 1).transpose(1, 2)
            image = (image * 255).numpy().astype(np.uint8)
            image = Image.fromarray(image)
            image.save(video_path)
        else:
            video_path = os.path.join(save_path, prefix + ".mp4")
            save_videos_grid(sample, video_path, fps=fps)

    # 分布式保存
    if ulysses_degree * ring_degree > 1:
        import torch.distributed as dist
        if dist.get_rank() == 0:
            save_results()
    else:
        save_results()
