# Based on https://github.com/guandeh17/Self-Forcing
import argparse
import gc
import json
import math
import os
import sys

import torch
from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from omegaconf import OmegaConf
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import (AutoencoderKLWan, WanT5EncoderModel,
                               WanTransformer3DModel)
from videox_fun.utils.utils import save_videos_grid


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def load_prompts(caption_path):
    with open(caption_path, encoding="utf-8") as f:
        return [line.rstrip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Generate ODE trajectory pairs for ODE regression training.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the model config YAML file (e.g. config/wan2.1/wan_civitai.yaml).",
    )
    parser.add_argument(
        "--caption_path",
        type=str,
        required=True,
        help="Path to a text file containing prompts, one per line. Download at https://huggingface.co/gdhe17/Self-Forcing/blob/main/vidprom_filtered_extended.txt",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="The output directory where per-prompt .safetensors ODE trajectory files will be saved.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="Classifier-free guidance scale for ODE denoising. Default: 6.0.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=48,
        help="Number of ODE denoising steps for the teacher model. Default: 48.",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=8.0,
        help="Shift value for FlowMatchEulerDiscreteScheduler. Default: 8.0.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=81,
        help="Number of pixel frames for the generated video. Default: 81.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height in pixels. Will be divided by VAE spatial ratio (8) for latent size. Default: 480.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width in pixels. Will be divided by VAE spatial ratio (8) for latent size. Default: 832.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        help="The negative prompt for classifier-free guidance.",
    )
    parser.add_argument(
        "--vae_mini_batch",
        type=int,
        default=1,
        help="Mini batch size for VAE decode. Default: 1.",
    )
    parser.add_argument(
        "--sample_every_n_prompts",
        type=int,
        default=0,
        help="Decode and save sample video every N prompts for visualization. 0 to disable. Default: 0.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Default: bf16.",
    )
    args = parser.parse_args()

    # Initialize accelerator for distributed generation
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    # Disable gradients globally since this is inference-only
    torch.set_grad_enabled(False)
    # Enable TF32 for faster computation on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = OmegaConf.load(args.config_path)

    # For mixed precision we cast all weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load tokenizer and text encoder
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path,
                     config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer'))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).to(device).eval()
    text_encoder.requires_grad_(False)

    # Load VAE
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(device, dtype=weight_dtype).eval()
    vae.requires_grad_(False)

    # Load bidirectional transformer (teacher)
    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path,
                     config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
    ).to(device, dtype=weight_dtype).eval()
    transformer.requires_grad_(False)

    # Load scheduler and configure shift
    scheduler_kwargs = OmegaConf.to_container(config['scheduler_kwargs'])
    scheduler_kwargs['shift'] = args.shift
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
    )
    noise_scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = noise_scheduler.timesteps

    # Compute latent shapes from VAE config
    # latent_h/w = pixel_h/w / spatial_compression_ratio
    # num_frames  = (pixel_frames - 1) / temporal_compression_ratio + 1
    latent_h = args.height // vae.spatial_compression_ratio
    latent_w = args.width // vae.spatial_compression_ratio
    latent_channels = vae.latent_channels
    num_frames = (args.video_sample_n_frames - 1) // vae.config.temporal_compression_ratio + 1

    # Compute seq_len for transformer
    patch_size = transformer.config.patch_size
    seq_len = math.ceil((latent_h * latent_w) / (patch_size[1] * patch_size[2]) * num_frames)

    # Load prompts and distribute across ranks
    prompts = load_prompts(args.caption_path)
    os.makedirs(args.output_folder, exist_ok=True)
    total_per_rank = int(math.ceil(len(prompts) / world_size))

    # Negative prompt embedding (unconditional)
    with torch.no_grad():
        neg_inputs = tokenizer(
            [args.negative_prompt], padding="max_length", max_length=512,
            truncation=True, add_special_tokens=True, return_tensors="pt"
        )
        neg_seq_lens = neg_inputs.attention_mask.gt(0).sum(dim=1).long()
        neg_embeds = text_encoder(neg_inputs.input_ids.to(device), attention_mask=neg_inputs.attention_mask.to(device))[0]
        neg_prompt_embeds = [neg_embeds[i, :neg_seq_lens[i]] for i in range(neg_embeds.shape[0])]

    # Main generation loop: each rank processes interleaved prompts
    for index in tqdm(range(total_per_rank), disable=rank != 0, desc="Generating ODE pairs"):
        prompt_index = index * world_size + rank
        if prompt_index >= len(prompts):
            continue

        prompt = prompts[prompt_index]
        output_path = os.path.join(args.output_folder, f"{prompt_index:05d}.safetensors")
        print(rank, output_path)
        if os.path.exists(output_path):
            continue

        # Encode prompt (keep padded [512, D] for saving to safetensors)
        text_inputs = tokenizer(
            [prompt], 
            padding="max_length", 
            max_length=512,
            truncation=True, 
            add_special_tokens=True, 
            return_tensors="pt"
        )
        prompt_attention_mask = text_inputs.attention_mask  # [1, 512]
        text_seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        text_embeds = text_encoder(text_inputs.input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]  # [1, 512, D]
        prompt_embeds = [text_embeds[i, :text_seq_lens[i]] for i in range(text_embeds.shape[0])]

        # Sample initial noise: [B, C, F, H, W]
        latents = torch.randn(
            [1, latent_channels, num_frames, latent_h, latent_w],
            dtype=weight_dtype, device=device
        )

        # Run full ODE denoising with CFG, collecting intermediate latents
        noisy_inputs = []

        # Reset scheduler state for each prompt to avoid stale `_step_index`
        # leaking across iterations and causing IndexError on `self.sigmas[sigma_idx + 1]`.
        noise_scheduler._step_index = None
        if hasattr(noise_scheduler, 'model_outputs'):
            noise_scheduler.model_outputs = []

        for progress_id, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0])  # [B]

            noisy_inputs.append(latents.clone())

            # Conditional prediction
            with torch.cuda.amp.autocast(dtype=weight_dtype):
                flow_pred_cond = transformer(
                    x=latents,
                    context=prompt_embeds,
                    t=timestep,
                    seq_len=seq_len,
                )

                # Unconditional prediction
                flow_pred_uncond = transformer(
                    x=latents,
                    context=neg_prompt_embeds,
                    t=timestep,
                    seq_len=seq_len,
                )

            # CFG
            flow_pred = flow_pred_uncond + args.guidance_scale * (flow_pred_cond - flow_pred_uncond)

            # Scheduler step
            latents = noise_scheduler.step(flow_pred, t, latents, return_dict=False)[0]

        # Append final clean latent
        noisy_inputs.append(latents.clone())

        # Stack all intermediate + final latents: [1, num_steps+1, C, F, H, W]
        noisy_inputs_tensor = torch.stack(noisy_inputs, dim=1)

        # Sparse sample 5 points along the ODE trajectory: [0, 12, 24, 36, -1]
        # This reduces storage while preserving the trajectory shape
        noisy_inputs_tensor = noisy_inputs_tensor[:, [0, 12, 24, 36, -1]]

        # Save as safetensors with latents, prompt_embeds, prompt_attention_mask
        save_file(
            {
                "latents": noisy_inputs_tensor.squeeze(0).cpu(),
                "prompt_embeds": text_embeds.squeeze(0).cpu(),
                "prompt_attention_mask": prompt_attention_mask.squeeze(0).cpu(),
            },
            output_path,
            metadata={"prompt": prompt},
        )

        # Decode and save sample video for visualization
        if args.sample_every_n_prompts > 0 and prompt_index % args.sample_every_n_prompts == 0:
            sample_dir = os.path.join(args.output_folder, "sample")
            os.makedirs(sample_dir, exist_ok=True)
            with torch.no_grad():
                # Decode the final clean latent (last sparse point)
                clean_latent = noisy_inputs_tensor[:, -1]  # [1, C, F, H, W]
                video = vae.decode(clean_latent.to(vae.dtype)).sample
                video = (video / 2 + 0.5).clamp(0, 1)
                save_videos_grid(
                    video.cpu().float(),
                    os.path.join(sample_dir, f"{prompt_index:05d}_clean.mp4"),
                )
            gc.collect()
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    # Write outputs.json annotation file for ImageVideoSafetensorsDataset
    # This JSON lists all generated safetensors files so they can be loaded by train_ode.py
    if accelerator.is_main_process:
        safe_json_writer = []
        for i in range(len(prompts)):
            safetensor_path = os.path.join(args.output_folder, f"{i:05d}.safetensors")
            if os.path.exists(safetensor_path):
                safe_json_writer.append({"file_path": safetensor_path})
        json_path = os.path.join(args.output_folder, "outputs.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(safe_json_writer, f, ensure_ascii=False, indent=4)
        print(f"Done. Generated {len(safe_json_writer)} ODE pairs, saved to {args.output_folder}")
        print(f"Annotation JSON: {json_path}")


if __name__ == "__main__":
    main()
