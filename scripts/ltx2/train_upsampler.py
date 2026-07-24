"""Training script for LTX2 Latent Upsampler.

Trains the LTX2LatentUpsamplerModel to spatially upsample VAE latents.
Training paradigm: pure supervised MSE regression on paired low/high-res latents.
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import gc
import logging
import math
import os
import pickle
import random
import shutil
import sys

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from packaging import version
from PIL import Image
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers.utils import ContextManagers

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.data import (ASPECT_RATIO_512, ASPECT_RATIO_RANDOM_CROP_512,
                             ASPECT_RATIO_RANDOM_CROP_PROB,
                             AspectRatioBatchImageVideoSampler,
                             ImageVideoDataset, ImageVideoSampler,
                             RandomSampler, VideoDataset,
                             get_closest_ratio, get_random_mask)
from videox_fun.models import AutoencoderKLLTX2Video, LTX2LatentUpsamplerModel
from videox_fun.utils.utils import save_videos_grid

# Will error if the minimal version of diffusers is not installed.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def get_random_downsample_ratio(sample_size, image_ratio=[],
                                all_choices=False, rng=None):
    def _create_special_list(length):
        if length == 1:
            return [1.0]
        if length >= 2:
            first_element = 0.75
            remaining_sum = 1.0 - first_element
            other_elements_value = remaining_sum / (length - 1)
            special_list = [first_element] + [other_elements_value] * (length - 1)
            return special_list

    if sample_size >= 1536:
        number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio
    elif sample_size >= 1024:
        number_list = [1, 1.25, 1.5, 2] + image_ratio
    elif sample_size >= 768:
        number_list = [1, 1.25, 1.5] + image_ratio
    elif sample_size >= 512:
        number_list = [1] + image_ratio
    else:
        number_list = [1]

    if all_choices:
        return number_list

    number_list_prob = np.array(_create_special_list(len(number_list)))
    if rng is None:
        return np.random.choice(number_list, p=number_list_prob)
    else:
        return rng.choice(number_list, p=number_list_prob)


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value


def log_validation(vae, latent_upsampler, args, accelerator, weight_dtype, global_step):
    """Validation: encode low-res -> upsample -> decode, save comparison videos."""
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype):
            logger.info("Running validation...")

            if args.validation_paths is None or len(args.validation_paths) == 0:
                logger.info("No validation_paths provided, skipping validation.")
                return

            try:
                from decord import VideoReader
            except ImportError:
                from videox_fun.data.utils import AVVideoReader as VideoReader

            for i, video_path in enumerate(args.validation_paths):
                if not os.path.exists(video_path):
                    logger.warning(f"Validation video not found: {video_path}")
                    continue

                # Load video frames
                vr = VideoReader(video_path)
                num_frames = min(len(vr), args.video_sample_n_frames)
                # Align to temporal compression ratio
                temporal_ratio = vae.config.temporal_compression_ratio
                num_frames = (num_frames - 1) // temporal_ratio * temporal_ratio + 1
                if num_frames <= 0:
                    num_frames = 1

                indices = list(range(num_frames))
                frames = vr.get_batch(indices).asnumpy()  # [F, H, W, C]

                # Preprocess to tensor [1, C, F, H, W]
                pixel_values = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
                pixel_values = pixel_values * 2.0 - 1.0  # normalize to [-1, 1]

                # Resize to target high-res size
                h_target = int(args.video_sample_size / 32) * 32
                w_target = h_target  # square for simplicity in validation
                pixel_values = F.interpolate(
                    pixel_values, size=(h_target, w_target), mode='bilinear', align_corners=False
                )
                pixel_values = pixel_values.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, F, H, W]
                pixel_values = pixel_values.to(device=accelerator.device, dtype=weight_dtype)

                # Encode high-res
                gt_latents = vae.encode(pixel_values)[0].mode()

                # Create low-res input (scale is fixed by upsampler architecture)
                scale = float(accelerator.unwrap_model(latent_upsampler).config.rational_spatial_scale)
                low_h, low_w = int(h_target / scale), int(w_target / scale)
                # Downsample spatially: flatten batch and frames, interpolate, unflatten
                b, c, f, h, w = pixel_values.shape
                pv_flat = pixel_values.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
                pv_low = F.interpolate(pv_flat, size=(low_h, low_w), mode='bilinear', align_corners=False)
                pixel_values_low = pv_low.reshape(b, f, c, low_h, low_w).permute(0, 2, 1, 3, 4)

                input_latents = vae.encode(pixel_values_low)[0].mode()

                # Upsample
                unwrapped_upsampler = accelerator.unwrap_model(latent_upsampler)
                predicted_latents = unwrapped_upsampler(input_latents)

                # Decode predictions
                # Use timestep conditioning if VAE supports it
                if getattr(vae.config, 'timestep_conditioning', False):
                    timestep = torch.zeros(1, device=accelerator.device, dtype=weight_dtype)
                    decoded_video = vae.decode(predicted_latents, timestep, return_dict=False)[0]
                else:
                    decoded_video = vae.decode(predicted_latents, return_dict=False)[0]

                # Also decode low-res for comparison (upsample back to same spatial size)
                if getattr(vae.config, 'timestep_conditioning', False):
                    decoded_low = vae.decode(input_latents, timestep, return_dict=False)[0]
                else:
                    decoded_low = vae.decode(input_latents, return_dict=False)[0]

                # Save videos
                os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)

                # Save upsampled result
                save_videos_grid(
                    decoded_video,
                    os.path.join(args.output_dir, f"sample/step{global_step}_val{i}_upsampled.mp4"),
                    rescale=True, fps=24
                )
                # Save low-res decoded for comparison
                save_videos_grid(
                    decoded_low,
                    os.path.join(args.output_dir, f"sample/step{global_step}_val{i}_lowres.mp4"),
                    rescale=True, fps=24
                )
                logger.info(f"Saved validation video {i} at step {global_step}")

    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Eval error on rank {accelerator.process_index} with info {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for LTX2 Latent Upsampler.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model (contains vae/ and latent_upsampler/ subfolders).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=("A json/csv containing the training data meta."),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_paths",
        type=str,
        default=None,
        nargs="+",
        help=("Video paths for validation (encode low-res -> upsample -> decode)."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_dir_ltx2_upsampler",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=1, help="Mini batch size for VAE encoding."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--use_came", action="store_true", help="Whether or not to use CAME optimizer."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_model_info", action="store_true", help="Whether or not to report more info about model (such as norm, grad)."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10 and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="ltx2-upsampler-train",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=1024,
        help="Target high-res video sample size.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=1024,
        help="Sample size of the image.",
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=1,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=121,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Num of repeat video.",
    )
    parser.add_argument(
        "--latent_upsampler_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other latent upsampler, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )
    parser.add_argument(
        '--trainable_modules',
        nargs='+',
        default=["."],
        help='Enter a list of trainable modules',
    )
    parser.add_argument(
        '--trainable_modules_low_learning_rate',
        nargs='+',
        default=[],
        help='Enter a list of trainable modules with lower learning rate',
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--use_fsdp", action="store_true", help="Whether or not to use fsdp."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--abnormal_norm_clip_start",
        type=int,
        default=1000,
        help=(
            'When do we start doing additional processing on abnormal gradients.'
        ),
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=5,
        help=(
            'The initial gradient is relative to the multiple of the max_grad_norm.'
        ),
    )
    parser.add_argument(
        "--multi_stream", action="store_true", help="Whether to use cuda multi-stream."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None", "0.15.0",
            message="Downloading 'non_ema' weights from revision branches of the Hub is deprecated.",
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    deepspeed_plugin = accelerator.state.deepspeed_plugin if hasattr(accelerator.state, "deepspeed_plugin") else None
    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        fsdp_stage = 0
        args.use_deepspeed = True
        if zero_stage == 3:
            args.save_state = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy in (ShardingStrategy.FULL_SHARD, None):
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        args.use_fsdp = True
        if fsdp_stage == 3:
            args.save_state = True
    else:
        zero_stage = 0
        fsdp_stage = 0

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(f"Init rng with seed {args.seed + accelerator.process_index if args.seed else 'None'}. Process_index is {accelerator.process_index}")

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # ==================== Load Models ====================
    # VAE (frozen)
    vae = AutoencoderKLLTX2Video.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
    )
    vae.eval()
    vae.requires_grad_(False)

    if args.vae_path is not None:
        print(f"Loading VAE from checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"VAE missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Latent Upsampler (trainable)
    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="latent_upsampler",
    )

    if args.latent_upsampler_path is not None:
        print(f"Loading latent upsampler from checkpoint: {args.latent_upsampler_path}")
        if args.latent_upsampler_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.latent_upsampler_path)
        else:
            state_dict = torch.load(args.latent_upsampler_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = latent_upsampler.load_state_dict(state_dict, strict=False)
        print(f"Upsampler missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Spatial scale is fixed by the upsampler architecture; read it directly from the model config.
    spatial_scale = float(latent_upsampler.config.rational_spatial_scale)
    if accelerator.is_main_process:
        logger.info(f"Using spatial_scale={spatial_scale} from latent_upsampler.config.rational_spatial_scale.")

    # Set trainable parameters
    latent_upsampler.requires_grad_(False)
    latent_upsampler.train()
    if accelerator.is_main_process:
        accelerator.print(f"Trainable modules '{args.trainable_modules}'.")
    for name, param in latent_upsampler.named_parameters():
        for trainable_module_name in args.trainable_modules + args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    # EMA
    if args.use_ema:
        from diffusers.training_utils import EMAModel
        if zero_stage == 3:
            raise NotImplementedError("DeepSpeed ZeRO-3 does not support EMA.")
        ema_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="latent_upsampler"
        ).to(weight_dtype)
        ema_upsampler = EMAModel(ema_upsampler.parameters(), model_cls=LTX2LatentUpsamplerModel, model_config=ema_upsampler.config)

    # ==================== Save/Load Hooks ====================
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        if fsdp_stage != 0 or zero_stage == 3:
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    from safetensors.torch import save_file
                    safetensor_save_path = os.path.join(output_dir, "diffusion_pytorch_model.safetensors")
                    accelerate_state_dict = {k: v.to(dtype=weight_dtype) for k, v in accelerate_state_dict.items()}
                    save_file(accelerate_state_dict, safetensor_save_path, metadata={"format": "pt"})
                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

            def load_model_hook(models, input_dir):
                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                        batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                    print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")
        else:
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    if args.use_ema:
                        ema_upsampler.save_pretrained(os.path.join(output_dir, "latent_upsampler_ema"))
                    models[0].save_pretrained(os.path.join(output_dir, "latent_upsampler"))
                    if not args.use_deepspeed:
                        weights.pop()
                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

            def load_model_hook(models, input_dir):
                if args.use_ema:
                    ema_path = os.path.join(input_dir, "latent_upsampler_ema")
                    load_model = LTX2LatentUpsamplerModel.from_pretrained(input_dir, subfolder="latent_upsampler_ema")
                    load_ema = EMAModel(load_model.parameters(), model_cls=LTX2LatentUpsamplerModel, model_config=load_model.config)
                    ema_upsampler.load_state_dict(load_ema.state_dict())
                    ema_upsampler.to(accelerator.device)
                    del load_model, load_ema

                for i in range(len(models)):
                    model = models.pop()
                    load_model = LTX2LatentUpsamplerModel.from_pretrained(input_dir, subfolder="latent_upsampler")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model

                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                        batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                    print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        latent_upsampler.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # ==================== Optimizer ====================
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        from came_pytorch import CAME
        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, latent_upsampler.parameters()))
    trainable_params_optim = [
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.learning_rate / 2},
    ]
    in_already = []
    for name, param in latent_upsampler.named_parameters():
        if not param.requires_grad:
            continue
        high_lr_flag = False
        if name in in_already:
            continue
        for trainable_module_name in args.trainable_modules:
            if trainable_module_name in name:
                in_already.append(name)
                high_lr_flag = True
                trainable_params_optim[0]['params'].append(param)
                if accelerator.is_main_process:
                    print(f"Set {name} to lr: {args.learning_rate}")
                break
        if high_lr_flag:
            continue
        for trainable_module_name in args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                in_already.append(name)
                trainable_params_optim[1]['params'].append(param)
                if accelerator.is_main_process:
                    print(f"Set {name} to lr: {args.learning_rate / 2}")
                break

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim, lr=args.learning_rate,
            betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim, lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
        )

    # ==================== Dataset ====================
    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio

    train_dataset = VideoDataset(
        args.train_data_meta, args.train_data_dir,
        sample_size=args.video_sample_size, sample_stride=args.video_sample_stride,
        sample_n_frames=args.video_sample_n_frames,
        enable_bucket=args.enable_bucket, enable_inpaint=False,
    )

    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            np.random.seed(_seed + worker_id)
            random.seed(_seed + worker_id)
        return _worker_init_fn

    if args.enable_bucket:
        aspect_ratio_sample_size = {key: [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator),
            dataset=train_dataset.dataset,
            batch_size=args.train_batch_size, train_folder=args.train_data_dir, drop_last=True,
            aspect_ratios=aspect_ratio_sample_size,
        )

        def collate_fn(examples):
            new_examples = {}
            new_examples["pixel_values"] = []
            new_examples["pixel_values_low"] = []

            pixel_value = examples[0]["pixel_values"]
            f, h, w, c = np.shape(pixel_value)

            if args.random_hw_adapt:
                random_downsample_ratio = get_random_downsample_ratio(args.video_sample_size, rng=rng)
            else:
                random_downsample_ratio = 1

            aspect_ratio_sample_size_local = {key: [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}

            closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size_local)
            closest_size = [int(x / 64) * 64 for x in closest_size]

            min_example_length = min([example["pixel_values"].shape[0] for example in examples])
            batch_video_length = int(min(args.video_sample_n_frames + sample_n_frames_bucket_interval, min_example_length))
            batch_video_length = (batch_video_length - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1
            if batch_video_length <= 0:
                batch_video_length = 1

            # Compute low-res target size (aligned to spatial_compression_ratio)
            scale = spatial_scale
            spatial_ratio = vae.config.spatial_compression_ratio
            closest_size_list = list(map(lambda x: int(x), closest_size))
            low_h = int(closest_size_list[0] / scale / spatial_ratio) * spatial_ratio
            low_w = int(closest_size_list[1] / scale / spatial_ratio) * spatial_ratio

            for example in examples:
                pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.0

                if closest_size_list[0] / h > closest_size_list[1] / w:
                    resize_size = closest_size_list[0], int(w * closest_size_list[0] / h)
                else:
                    resize_size = int(h * closest_size_list[1] / w), closest_size_list[1]

                # High-res transform
                transform_hr = transforms.Compose([
                    transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(closest_size_list),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ])
                pixel_values_hr = transform_hr(pixel_values)[:batch_video_length]
                new_examples["pixel_values"].append(pixel_values_hr)

                # Low-res: spatially downsample from high-res frames
                pixel_values_low = F.interpolate(
                    pixel_values_hr, size=(low_h, low_w), mode='bilinear', align_corners=False
                )
                new_examples["pixel_values_low"].append(pixel_values_low)

            new_examples["pixel_values"] = torch.stack(new_examples["pixel_values"])
            new_examples["pixel_values_low"] = torch.stack(new_examples["pixel_values_low"])
            return new_examples

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
        )
    else:
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = ImageVideoSampler(RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, args.train_batch_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
        )

    # ==================== LR Scheduler ====================
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # ==================== Prepare with Accelerator ====================
    latent_upsampler, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        latent_upsampler, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_upsampler.to(accelerator.device)

    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)

    # Recalculate
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        keys_to_pop = [k for k, v in tracker_config.items() if isinstance(v, list)]
        for k in keys_to_pop:
            tracker_config.pop(k)
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # ==================== Training ====================
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            pkl_path = os.path.join(os.path.join(args.output_dir, path), "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    _, first_epoch = pickle.load(file)
            else:
                first_epoch = global_step // num_update_steps_per_epoch
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps), initial=initial_global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    if args.multi_stream:
        vae_stream_1 = torch.cuda.Stream()
    else:
        vae_stream_1 = None

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            # Sanity check
            if epoch == first_epoch and step == 0:
                pixel_values_check = batch['pixel_values'].cpu()
                pixel_values_check = rearrange(pixel_values_check, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, pixel_value in enumerate(pixel_values_check):
                    save_videos_grid(pixel_value[None, ...], f"{args.output_dir}/sanity_check/sample_{idx}.mp4", rescale=True)

            with accelerator.accumulate(latent_upsampler):
                pixel_values = batch["pixel_values"].to(weight_dtype)  # [B, F, C, H, W]
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                pixel_values_low = batch["pixel_values_low"].to(weight_dtype)  # [B, F, C, H_low, W_low]
                pixel_values_low = rearrange(pixel_values_low, "b f c h w -> b c f h w")
                bsz = pixel_values.shape[0]

                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)

                with torch.no_grad():
                    # 1. VAE encode high-res -> GT latents (unnormalized)
                    bs = args.vae_mini_batch
                    gt_latents_list = []
                    for i in range(0, bsz, bs):
                        pv_bs = pixel_values[i:i + bs]
                        encoded = vae.encode(pv_bs)[0].mode()
                        gt_latents_list.append(encoded)
                    gt_latents = torch.cat(gt_latents_list, dim=0)

                    # 2. VAE encode low-res -> input latents
                    input_latents_list = []
                    for i in range(0, bsz, bs):
                        pv_bs = pixel_values_low[i:i + bs]
                        encoded = vae.encode(pv_bs)[0].mode()
                        input_latents_list.append(encoded)
                    input_latents = torch.cat(input_latents_list, dim=0)

                if args.low_vram:
                    vae.to('cpu')
                    torch.cuda.empty_cache()

                # 3. Forward through upsampler
                predicted_latents = latent_upsampler(input_latents)

                # 4. MSE Loss (in float32 for stability)
                loss = F.mse_loss(predicted_latents.float(), gt_latents.float())

                # Gather losses for logging
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_deepspeed and not args.use_fsdp:
                        trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                        if trainable_params_grads:
                            trainable_params_total_norm = torch.norm(
                                torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2
                            )
                            max_grad_norm = linear_decay(
                                args.max_grad_norm * args.initial_grad_norm_ratio,
                                args.max_grad_norm, args.abnormal_norm_clip_start, global_step
                            )
                            if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                                actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                            else:
                                actual_max_grad_norm = max_grad_norm
                        else:
                            actual_max_grad_norm = args.max_grad_norm
                    else:
                        actual_max_grad_norm = args.max_grad_norm

                    accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Post-step actions
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_upsampler.step(latent_upsampler.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # Checkpointing
                if global_step % args.checkpointing_steps == 0:
                    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(f"Removing {len(removing_checkpoints)} checkpoints")
                                for removing_checkpoint in removing_checkpoints:
                                    shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))

                        gc.collect()
                        torch.cuda.empty_cache()
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # Validation
                if args.validation_paths is not None and global_step % args.validation_steps == 0:
                    if args.use_ema:
                        ema_upsampler.store(latent_upsampler.parameters())
                        ema_upsampler.copy_to(latent_upsampler.parameters())
                    log_validation(vae, latent_upsampler, args, accelerator, weight_dtype, global_step)
                    if args.use_ema:
                        ema_upsampler.restore(latent_upsampler.parameters())

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        # Epoch-level validation
        if args.validation_paths is not None and epoch % args.validation_epochs == 0:
            if args.use_ema:
                ema_upsampler.store(latent_upsampler.parameters())
                ema_upsampler.copy_to(latent_upsampler.parameters())
            log_validation(vae, latent_upsampler, args, accelerator, weight_dtype, global_step)
            if args.use_ema:
                ema_upsampler.restore(latent_upsampler.parameters())

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        latent_upsampler_unwrapped = unwrap_model(latent_upsampler)
        if args.use_ema:
            ema_upsampler.copy_to(latent_upsampler_unwrapped.parameters())

    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
        gc.collect()
        torch.cuda.empty_cache()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
