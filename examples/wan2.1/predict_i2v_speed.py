import os
import sys
import argparse
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

# 添加项目根目录到 sys.path
current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# 导入模块
from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, WanT5EncoderModel, AutoTokenizer, WanTransformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import WanPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Video Generation with Wan2.1-Fun")
    
    # GPU Memory Optimization
    parser.add_argument("--GPU_memory_mode", type=str, default="sequential_cpu_offload",
                        choices=["model_full_load", "model_full_load_and_qfloat8", "model_cpu_offload", 
                                 "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"],
                        help="GPU memory optimization mode.")
    parser.add_argument("--ulysses_degree", type=int, default=1,
                        help="Ulysses parallelism degree.")
    parser.add_argument("--ring_degree", type=int, default=1,
                        help="Ring parallelism degree.")
    parser.add_argument("--fsdp_dit", action="store_true",
                        help="Use FSDP for transformer to save GPU memory.")
    parser.add_argument("--fsdp_text_encoder", action="store_true",
                        help="Use FSDP for text encoder to save GPU memory.")
    parser.add_argument("--compile_dit", action="store_true",
                        help="Compile transformer for fixed resolution speedup.")
    
    # TeaCache
    parser.add_argument("--enable_teacache", action="store_true",
                        help="Enable TeaCache optimization.")
    parser.add_argument("--teacache_threshold", type=float, default=0.10,
                        help="TeaCache threshold for step caching.")
    parser.add_argument("--num_skip_start_steps", type=int, default=5,
                        help="Number of steps to skip TeaCache at inference start.")
    parser.add_argument("--teacache_offload", action="store_true",
                        help="Offload TeaCache tensors to CPU.")
    
    # CFG Skip
    parser.add_argument("--cfg_skip_ratio", type=float, default=0.0,
                        help="CFG skip ratio for inference.")
    
    # Riflex
    parser.add_argument("--enable_riflex", action="store_true",
                        help="Enable Riflex frequency optimization.")
    parser.add_argument("--riflex_k", type=int, default=6,
                        help="Intrinsic frequency index for Riflex.")
    
    # Model Paths
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model config file.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Path to model directory.")
    parser.add_argument("--transformer_path", type=str, default=None,
                        help="Path to pre-trained transformer checkpoint.")
    parser.add_argument("--vae_path", type=str, default=None,
                        help="Path to pre-trained VAE checkpoint.")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA weights.")
    
    # Generation Parameters
    parser.add_argument("--sample_size", nargs=2, type=int, default=[480, 832],
                        help="Sample size [height, width].")
    parser.add_argument("--video_length", type=int, default=81,
                        help="Number of frames in the video.")
    parser.add_argument("--fps", type=int, default=16,
                        help="Frames per second for output video.")
    parser.add_argument("--weight_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"],
                        help="Weight data type (float16 or bfloat16).")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for video generation.")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Negative prompt for video generation.")
    parser.add_argument("--guidance_scale", type=float, default=6.0,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=43,
                        help="Random seed for reproducibility.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of inference steps.")
    parser.add_argument("--lora_weight", type=float, default=0.55,
                        help="LoRA weight scaling factor.")
    parser.add_argument("--save_path", type=str, default="samples/wan-videos-t2v",
                        help="Directory to save generated videos.")

    return parser.parse_args()

args = parse_args()

# 将 argparse 参数映射到原有变量
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

# 设备设置
device = set_multi_gpus_devices(ulysses_degree, ring_degree)

# 加载配置
config = OmegaConf.load(config_path)

# 加载 Transformer 模型
transformer = WanTransformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=not fsdp_dit,
    torch_dtype=weight_dtype
)

# 加载 Transformer Checkpoint（如有）
if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict.get("state_dict", state_dict)

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(m)}, Unexpected keys: {len(u)}")

# 加载 VAE
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

# 加载 VAE Checkpoint（如有）
if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict.get("state_dict", state_dict)

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(m)}, Unexpected keys: {len(u)}")

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer'))
)

# 加载 Text Encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype
)

# 加载 Scheduler
scheduler_class = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler
}[args.sampler_name]

if args.sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
    config['scheduler_kwargs']['shift'] = 1

scheduler = scheduler_class(
    **filter_kwargs(scheduler_class, OmegaConf.to_container(config['scheduler_kwargs']))
)

# 构建 Pipeline
pipeline = WanPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler
)

# 多卡并行设置
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

# TeaCache 配置
coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, 
        num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )

# CFG Skip 配置
if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)

# 随机种子
generator = torch.Generator(device=device).manual_seed(seed)

# 加载 LoRA（如有）
if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device)

# 执行推理
with torch.no_grad():
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1

    if enable_riflex:
        pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)

    sample = pipeline(
        prompt,
        num_frames=video_length,
        negative_prompt=negative_prompt,
        height=sample_size[0],
        width=sample_size[1],
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).videos

# 卸载 LoRA（如有）
if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)

# 保存结果
def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    if video_length == 1:
        video_path = os.path.join(save_path, f"{prefix}.png")
        image = sample[0, :, 0].permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        Image.fromarray(image).save(video_path)
    else:
        video_path = os.path.join(save_path, f"{prefix}.mp4")
        save_videos_grid(sample, video_path, fps=fps)

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results()
