"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
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
import contextlib
import gc
import json
import logging
import math
import os
import pickle
import shutil
import sys

import accelerate
import diffusers
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, FullStateDictConfig, ShardedOptimStateDictConfig,
    ShardedStateDictConfig)
from torch.utils.data import BatchSampler, Dataset, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import AutoTokenizer
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
                             RandomSampler, TextDataset, get_closest_ratio,
                             get_random_mask)
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                               CLIPModel, SparkWanTransformer3DModel,
                               WanT5EncoderModel, WanTransformer3DModel)
from videox_fun.pipeline import WanI2VPipeline, WanPipeline
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.tqdm_bar import PauseAwareTqdm
from videox_fun.utils.utils import (calculate_dimensions, get_image_latent,
                                    get_image_to_video_latent,
                                    save_videos_grid)

if is_wandb_available():
    import wandb


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

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
        return np.random.choice(number_list, p = number_list_prob)
    else:
        return rng.choice(number_list, p = number_list_prob)

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def log_validation(vae, text_encoder, tokenizer, transformer3d, args, config, accelerator, weight_dtype, global_step):
    try:
        is_deepspeed = type(transformer3d).__name__ == 'DeepSpeedEngine'
        if is_deepspeed:
            origin_config = transformer3d.config
            transformer3d.config = accelerator.unwrap_model(transformer3d).config
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
            # FSDP(CPUOffload)-wrapped modules own their placement: never .to() the sharded
            # text_encoder here or on exit; _te_sharded marks it and the generator below.
            _te_sharded = (
                getattr(accelerator.state, "fsdp_plugin", None) is not None
                or getattr(accelerator.state, "deepspeed_plugin", None) is not None
            )
            logger.info("Running validation... ")
            scheduler = FlowMatchEulerDiscreteScheduler(
                **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
            )
            transformer3d_1 = accelerator.unwrap_model(transformer3d) if type(transformer3d).__name__ == 'DistributedDataParallel' else transformer3d

            if args.train_mode != "normal":
                pipeline = WanI2VPipeline(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    transformer=transformer3d_1,
                    scheduler=scheduler,
                )
            else:
                pipeline = WanPipeline(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    transformer=transformer3d_1,
                    scheduler=scheduler,
                )
            # Stage only non-FSDP components on GPU; the FSDP-wrapped generator/text_encoder
            # self-stream in place. VAE gets an offload hook and pins the pipeline device.
            from videox_fun.dist import offload_components_cpu
            _fsdp_cls_name = 'FullyShardedDataParallel'
            for _comp in (pipeline.transformer,):
                if _comp is not None and type(_comp).__name__ != _fsdp_cls_name:
                    _comp.to(accelerator.device, dtype=weight_dtype)
            if not _te_sharded:
                text_encoder.to(accelerator.device, dtype=weight_dtype)
            # VAE is never FSDP-wrapped; give it an onload/offload hook and force pipeline device -> GPU.
            offload_components_cpu(pipeline, accelerator.device, ["vae"])

            if args.seed is None:
                generator = None
            else:
                rank_seed = args.seed + accelerator.process_index
                generator = torch.Generator(device=accelerator.device).manual_seed(rank_seed)
                logger.info(f"Rank {accelerator.process_index} using seed: {rank_seed}")

            for i in range(len(args.validation_prompts)):
                if args.train_mode != "normal":
                    start_image = Image.open(args.validation_paths[i])
                    width, height = start_image.width, start_image.height
                    width, height = calculate_dimensions(args.image_sample_size * args.image_sample_size,  width / height)

                    video_length = int((args.video_sample_n_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.video_sample_n_frames != 1 else 1
                    input_video, input_video_mask, _ = get_image_to_video_latent(args.validation_paths[i], None, video_length=video_length, sample_size=[height, width])
                    sample = pipeline(
                        args.validation_prompts[i],
                        num_frames = video_length,
                        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        height      = height,
                        width       = width,
                        generator   = generator,

                        video        = input_video,
                        mask_video   = input_video_mask,
                        num_inference_steps = 4,
                        guidance_scale      = 1.0,
                    ).videos

                    os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                    save_videos_grid(
                        sample, 
                        os.path.join(
                            args.output_dir, 
                            f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.mp4"
                        )
                    )
                else:
                    sample = pipeline(
                        args.validation_prompts[i],
                        num_frames = args.video_sample_n_frames,
                        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        height      = args.video_sample_size,
                        width       = args.video_sample_size,
                        generator   = generator,
                        num_inference_steps = 4,
                        guidance_scale      = 1.0,
                    ).videos
                    os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                    save_videos_grid(
                        sample, 
                        os.path.join(
                            args.output_dir, 
                            f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.mp4"
                        )
                    )

            # Detach the validation-only VAE offload hook before returning to training so that
            # manual_vae_move keeps sole control of the VAE's device placement.
            for _h in getattr(pipeline, "_cpu_offload_component_hooks", []):
                _h.remove()
            # Drop the pipeline and the local unwrapped handle so empty_cache below actually
            # reclaims the validation-only weights. The live generator survives via the
            # accelerator's own reference.
            del pipeline, transformer3d_1
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
            if not args.enable_text_encoder_in_dataloader and not _te_sharded:
                text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        if is_deepspeed:
            transformer3d.config = origin_config
    except Exception as e:
        if 'pipeline' in locals():
            for _h in getattr(pipeline, "_cpu_offload_component_hooks", []):
                _h.remove()
            del pipeline
        # Release the unwrapped validation handle before empty_cache so it is actually reclaimed here.
        if 'transformer3d_1' in locals():
            del transformer3d_1
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error on rank {accelerator.process_index} with info {e}")
        vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        _te_sharded_exc = (
            getattr(accelerator.state, "fsdp_plugin", None) is not None
            or getattr(accelerator.state, "deepspeed_plugin", None) is not None
        )
        if not args.enable_text_encoder_in_dataloader and not _te_sharded_exc:
            text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value

def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=(
            "A csv containing the training data. "
        ),
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
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_paths",
        type=str,
        default=None,
        nargs="+",
        help=("A set of control videos evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        help=("The negative prompt of cfg distill"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
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
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
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
        "--learning_rate_critic",
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
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
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
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=0.0, type=float, help="Max gradient norm. Default 0 disables gradient clipping to match official CrossDistill (grad_clip=False; Table 4 runs unclipped). Set >0 to enable.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
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
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
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
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
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
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    
    parser.add_argument(
        "--snr_loss", action="store_true", help="Whether or not to use snr_loss."
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader", action="store_true", help="Whether or not to use text encoder in dataloader."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_frame_crop", action="store_true", help="Whether enable random frame crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--training_with_video_token_length", action="store_true", help="The training stage of the model in training.",
    )
    parser.add_argument(
        "--auto_tile_batch_size", action="store_true", help="Whether to auto tile batch size.",
    )
    parser.add_argument(
        "--motion_sub_loss", action="store_true", help="Whether enable motion sub loss."
    )
    parser.add_argument(
        "--motion_sub_loss_ratio", type=float, default=0.25, help="The ratio of motion sub loss."
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--keep_all_node_same_token_length",
        action="store_true", 
        help="Reference of the length token.",
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=512,
        help="Sample size of the token.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="Sample size of the image.",
    )
    parser.add_argument(
        "--fix_sample_size", 
        nargs=2, type=int, default=None,
        help="Fix Sample size [height, width] when using bucket and collate_fn."
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=4,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=17,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Num of repeat video.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--generator_transformer_path",
        type=str,
        default=None,
        help=("Path to a (sparse stage-1) student checkpoint loaded into the generator. When "
              "--sparse_student is off it is also loaded into the dense critic (fake_score); "
              "with --sparse_student the critic stays dense (initialized from the teacher)."),
    )
    parser.add_argument(
        "--fake_score_transformer_path",
        type=str,
        default=None,
        help=("Path to a checkpoint loaded into the dense critic (fake_score) only."),
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
        help='Enter a list of trainable modules'
    )
    parser.add_argument(
        '--trainable_modules_low_learning_rate', 
        nargs='+', 
        default=[],
        help='Enter a list of trainable modules with lower learning rate'
    )
    parser.add_argument(
        '--tokenizer_max_length', 
        type=int,
        default=512,
        help='Max length of tokenizer'
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
        "--train_mode",
        type=str,
        default="normal",
        help=(
            'The training mode. Support `"normal"` (T2V, default), '
            '`"i2v"` (Image-to-Video), and `"ti2v"` (Text-Image-to-Video).'
        ),
    )
    parser.add_argument(
        "--gen_update_interval",
        type=int,
        default=5,
        help="The ratio to update transformer3d.",
    )
    parser.add_argument(
        "--fake_guidance_scale",
        type=float,
        default=0.0,
        help="The cfg scale for fake iscore.",
    )
    parser.add_argument(
        "--real_guidance_scale",
        type=float,
        default=5.0,
        help="The cfg scale for real score (official teacher_guidance default = 5.0).",
    )
    parser.add_argument(
        '--denoising_step_indices_list', 
        nargs='+', 
        default=[1000, 750, 500, 250],
        help="The denoising step list.",
    )
    parser.add_argument(
        "--randomize_step_indices",
        action="store_true",
        help="whether to use randomize timesteps indices in training.",
    )

    parser.add_argument(
        "--distill_loss",
        type=str,
        default="tdm",
        choices=["tdm", "dmd"],
        help="Student distribution-matching loss. TDM and DMD are mutually exclusive; exactly one of --loss_scale_tdm / --loss_scale_dmd must be > 0. 'tdm' = Trajectory Distribution Matching, 'dmd' = Distribution Matching Distillation.",
    )
    parser.add_argument(
        "--loss_scale_tdm", type=float, default=1.0, help="Weight of the TDM student loss (0 disables it)."
    )
    parser.add_argument(
        "--loss_scale_dmd", type=float, default=1.0, help="Weight of the DMD student loss (0 disables it)."
    )
    parser.add_argument(
        "--div_mode",
        type=str,
        default="pcm_random",
        choices=["pcm_random", "pcm_fixed", "ode_match"],
        help="Diversity supervision algorithm (shared by TDM/DMD, keeps the few-step student from mode collapse). 'ode_match' matches the student rf velocity at s=1 to a teacher K-step rollout; 'pcm_*' does PCM-style boundary alignment on phase 0.",
    )
    parser.add_argument(
        "--loss_scale_div", type=float, default=0.0, help="Weight of the diversity loss (0 disables it). Official default is 0; the CrossDistill recipe enables it (see train_distill_wan2.1.sh)."
    )
    parser.add_argument(
        "--div_K", type=int, default=8, help="Teacher rollout step count for ode_match diversity."
    )
    parser.add_argument(
        "--div_sigma_max", type=float, default=5000.0, help="sigma_max used to build the ode_match teacher rf grid."
    )
    parser.add_argument(
        "--dcm_total_steps", type=int, default=48, help="Discrete-time CM grid size used by the pcm-div modes."
    )
    parser.add_argument(
        "--dcm_skipping_interval_steps", type=int, default=1, help="Skipping interval on the CM grid used by the pcm-div modes."
    )
    parser.add_argument(
        "--max_simulation_steps_fake", type=int, default=4, help="Max number of student Euler steps during deterministic rf-Euler backward simulation (all times are rf s in [0,1])."
    )
    parser.add_argument(
        "--joint_backward_timesteps",
        nargs='+',
        type=float,
        default=[0.933781, 0.875, 0.608979],
        help="Canonical joint few-step rf grid (rCM, descending). Value = rf time tan(theta)/(1+tan(theta)) at theta=[1.5, mid, 1.0]. --backward_timesteps defaults to its interior knots (0 < k < 1) for the single-expert full range.",
    )
    parser.add_argument(
        "--backward_timesteps",
        nargs='+',
        type=float,
        default=[],
        help="rf mid-knots of the backward-simulation schedule (descending). Empty = auto: the interior knots of --joint_backward_timesteps (0 < k < 1).",
    )
    parser.add_argument(
        "--backward_simulation_start_t",
        type=float,
        default=-1.0,
        help="rf start time of backward simulation; -1 = auto (1.0 = pure noise, full range).",
    )
    parser.add_argument(
        "--backward_simulation_end_t", type=float, default=0.0, help="rf end time of backward simulation; 0 = x0."
    )
    parser.add_argument(
        "--teacher_timestep_shift",
        type=float,
        default=5.0,
        help="Official teacher sampling shift; defines the teacher timestep density (Wan2.1 t2v 1.3B = 5.0). Real (teacher) guidance is reused from --real_guidance_scale; fake-score guidance from --fake_guidance_scale.",
    )
    parser.add_argument(
        "--rf_t_min",
        type=float,
        default=-1.0,
        help="Lower rf-time bound for partition time sampling (D-time and div drawn from p_D then clamped to [rf_t_min, rf_t_max]). -1 = auto (full range: 0.0).",
    )
    parser.add_argument(
        "--rf_t_max",
        type=float,
        default=-1.0,
        help="Upper rf-time bound for partition time sampling. -1 = auto (full range: 1.0).",
    )
    parser.add_argument(
        "--p_D",
        type=str,
        default="lognormal",
        choices=["uniform_shift", "lognormal"],
        help="Distribution used to draw the critic/teacher comparison time s_D. Default 'lognormal' matches official p_D=LogNormal(p_mean=0.0, p_std=1.6).",
    )
    parser.add_argument(
        "--p_D_shift", type=float, default=5.0, help="shift for uniform_shift p_D (match teacher_timestep_shift)."
    )
    parser.add_argument(
        "--p_D_mean", type=float, default=0.0, help="mean of lognormal p_D (log-sigma space)."
    )
    parser.add_argument(
        "--p_D_std", type=float, default=1.6, help="std of lognormal p_D (log-sigma space)."
    )
    parser.add_argument(
        "--student_update_freq", type=int, default=5, help="Run the student update once every this many iterations (critic in between)."
    )
    parser.add_argument(
        "--critic_warmup", type=int, default=0, help="Number of initial iterations that always run the student (critic frozen)."
    )
    parser.add_argument(
        "--sparse_student",
        action="store_true",
        help="Distill a sparse (RoLa/SLA-style) student against the dense teacher: the generator uses SparkWanTransformer3DModel (sparse + linear self-attention), teacher and critic stay dense.",
    )
    parser.add_argument(
        "--student_topk",
        type=float,
        default=0.03,
        help="Ratio of key blocks kept per query in the sparse student self-attention (sparsity = 1 - topk). Must match the --student_topk used in train_finetune_sparse.py.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    # --- SparkDiffusion CrossDistill Argument Validation ---
    if args.train_mode != "normal":
        raise ValueError("SparkDiffusion distillation supports --train_mode normal (T2V) only.")
    if args.distill_loss == "tdm":
        args.loss_scale_dmd = 0.0
        if args.loss_scale_tdm <= 0:
            raise ValueError("--distill_loss tdm requires --loss_scale_tdm > 0.")
    elif args.distill_loss == "dmd":
        args.loss_scale_tdm = 0.0
        if args.loss_scale_dmd <= 0:
            raise ValueError("--distill_loss dmd requires --loss_scale_dmd > 0.")
    assert not (args.loss_scale_tdm > 0 and args.loss_scale_dmd > 0), \
        "loss_scale_tdm and loss_scale_dmd are mutually exclusive; set exactly one > 0."
    if args.loss_scale_div > 0:
        assert args.max_simulation_steps_fake >= 2, (
            "--loss_scale_div > 0 requires --max_simulation_steps_fake >= 2 so that the "
            "first (rf s=1) segment of backward simulation stays no_grad.")
    if args.student_update_freq <= 0:
        raise ValueError("--student_update_freq must be greater than zero.")
    if args.trainable_modules is None:
        args.trainable_modules = [""]

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    config = OmegaConf.load(args.config_path)

    # --- Wan2.1 is a single expert over the full rf-time range [0, 1]: no MoE boundary to
    # route on. Explicit --rf_t_min / --rf_t_max / --backward_simulation_start_t (>= 0)
    # still override the full-range defaults. ---
    if args.rf_t_min < 0:
        args.rf_t_min = 0.0
    if args.rf_t_max < 0:
        args.rf_t_max = 1.0
    if args.backward_simulation_start_t < 0:
        args.backward_simulation_start_t = 1.0
    if not args.backward_timesteps:
        args.backward_timesteps = [k for k in args.joint_backward_timesteps if 0.0 < k < 1.0]
    if not (0.0 <= args.rf_t_min < args.rf_t_max <= 1.0):
        raise ValueError("require 0 <= --rf_t_min < --rf_t_max <= 1.")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    accelerator_fake_score_transformer3d = Accelerator(
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
        print(f"Using DeepSpeed Zero stage: {zero_stage}")

        args.use_deepspeed = True
        if zero_stage == 3:
            print(f"Auto set save_state to True because zero_stage == 3")
            args.save_state = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None: # The fsdp_plugin.sharding_strategy is None in FSDP 2.
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        print(f"Using FSDP stage: {fsdp_stage}")

        args.use_fsdp = True
        if fsdp_stage == 3:
            print(f"Auto set save_state to True because fsdp_stage == 3")
            args.save_state = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("DeepSpeed is not enabled.")

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    args.denoising_step_indices_list = [int(i) for i in args.denoising_step_indices_list]
    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        text_encoder = text_encoder.eval()
        # Get Vae
        Chosen_AutoencoderKL = {
            "AutoencoderKLWan": AutoencoderKLWan,
            "AutoencoderKLWan3_8": AutoencoderKLWan3_8
        }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
        vae = Chosen_AutoencoderKL.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        vae.eval()
            
    # Get Transformer
    sub_path = config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')

    # Sparse student (SparkDiffusion stage-1 output): block-sparse + linear self-attention
    # (SparkWanTransformer3DModel); the dense teacher and the dense critic are untouched,
    # matching the reference pipeline where fake_score initializes from the teacher weights.
    def build_student_net(weights_dir):
        if args.sparse_student:
            net = SparkWanTransformer3DModel.from_pretrained(
                weights_dir,
                transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
                low_cpu_mem_usage=True,
            ).to(weight_dtype)
            # Materialize the zero-init sparse branch that the dense base cannot provide.
            net.set_student_topk(args.student_topk)
            return net
        return WanTransformer3DModel.from_pretrained(
            weights_dir,
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
        ).to(weight_dtype)

    generator_transformer3d = build_student_net(os.path.join(args.pretrained_model_name_or_path, sub_path))
    real_score_transformer3d = WanTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, sub_path),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
    ).to(weight_dtype)
    fake_score_transformer3d = WanTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, sub_path),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
    ).to(weight_dtype)

    # Wan2.1 is a single expert: no frozen high-noise relay is needed to roll the
    # backward-simulation starting latent (it always starts from pure noise at rf 1.0).
    init_rollout_transformer3d = None

    # Freeze vae and text_encoder and set generator_transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    generator_transformer3d.requires_grad_(False)
    real_score_transformer3d.requires_grad_(False)
    fake_score_transformer3d.requires_grad_(False)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = generator_transformer3d.load_state_dict(state_dict, strict=False)
        print(f"generator missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        m, u = fake_score_transformer3d.load_state_dict(state_dict, strict=False)
        print(f"fake_score missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        m, u = real_score_transformer3d.load_state_dict(state_dict, strict=False)
        print(f"real_score missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    if args.generator_transformer_path is not None:
        print(f"From generator/fake-score checkpoint: {args.generator_transformer_path}")
        if args.generator_transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.generator_transformer_path)
        else:
            state_dict = torch.load(args.generator_transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = generator_transformer3d.load_state_dict(state_dict, strict=False)
        print(f"generator missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        if args.sparse_student:
            # Sparse attn_op.* keys have no dense-critic counterpart; keep fake_score dense
            # (teacher-init), as in the reference pipeline where the critic skips student weights.
            print("sparse_student: keeping fake_score dense (skipping sparse checkpoint load)")
        else:
            m, u = fake_score_transformer3d.load_state_dict(state_dict, strict=False)
            print(f"fake_score missing keys: {len(m)}, unexpected keys: {len(u)}")
            assert len(u) == 0

    if args.fake_score_transformer_path is not None:
        print(f"From fake-score checkpoint: {args.fake_score_transformer_path}")
        if args.fake_score_transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.fake_score_transformer_path)
        else:
            state_dict = torch.load(args.fake_score_transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = fake_score_transformer3d.load_state_dict(state_dict, strict=False)
        print(f"fake_score missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
    
    # A good trainable modules is showed below now.
    # For 3D Patch: trainable_modules = ['ff.net', 'pos_embed', 'attn2', 'proj_out', 'timepositionalencoding', 'h_position', 'w_position']
    # For 2D Patch: trainable_modules = ['ff.net', 'attn2', 'timepositionalencoding', 'h_position', 'w_position']
    generator_transformer3d.train()
    fake_score_transformer3d.train()
    if accelerator.is_main_process:
        accelerator.print(
            f"Trainable modules '{args.trainable_modules}'."
        )
    for name, param in generator_transformer3d.named_parameters():
        for trainable_module_name in args.trainable_modules + args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                param.requires_grad = True
                break
    for name, param in fake_score_transformer3d.named_parameters():
        for trainable_module_name in args.trainable_modules + args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        if fsdp_stage != 0 or zero_stage == 3:
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    from safetensors.torch import save_file

                    safetensor_save_path = os.path.join(output_dir, f"diffusion_pytorch_model.safetensors")
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
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                    if not args.use_deepspeed:
                        weights.pop()

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

            def load_model_hook(models, input_dir):
                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = WanTransformer3DModel.from_pretrained(
                        input_dir, subfolder="transformer",
                        low_cpu_mem_usage=True,
                    )
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
        accelerator_fake_score_transformer3d.register_save_state_pre_hook(save_model_hook)
        accelerator_fake_score_transformer3d.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        generator_transformer3d.enable_gradient_checkpointing()
        fake_score_transformer3d.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except Exception:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, generator_transformer3d.parameters()))
    trainable_params_optim = [
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.learning_rate / 2},
    ]
    in_already = []
    for name, param in generator_transformer3d.named_parameters():
        high_lr_flag = False
        if name in in_already:
            continue
        for trainable_module_name in args.trainable_modules:
            if trainable_module_name in name:
                in_already.append(name)
                high_lr_flag = True
                trainable_params_optim[0]['params'].append(param)
                if accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate}")
                break
        if high_lr_flag:
            continue
        for trainable_module_name in args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                in_already.append(name)
                trainable_params_optim[1]['params'].append(param)
                if accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate / 2}")
                break

    fake_trainable_params = list(filter(lambda p: p.requires_grad, fake_score_transformer3d.parameters()))
    fake_trainable_params_optim = [
        {'params': [], 'lr': args.learning_rate_critic},
        {'params': [], 'lr': args.learning_rate_critic / 2},
    ]
    in_already = []
    for name, param in fake_score_transformer3d.named_parameters():
        high_lr_flag = False
        if name in in_already:
            continue
        for trainable_module_name in args.trainable_modules:
            if trainable_module_name in name:
                in_already.append(name)
                high_lr_flag = True
                fake_trainable_params_optim[0]['params'].append(param)
                if accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate_critic}")
                break
        if high_lr_flag:
            continue
        for trainable_module_name in args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                in_already.append(name)
                fake_trainable_params_optim[1]['params'].append(param)
                if accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate_critic / 2}")
                break

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            # weight_decay=args.adam_weight_decay,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
        critic_optimizer = optimizer_cls(
            fake_trainable_params_optim,
            lr=args.learning_rate_critic,
            # weight_decay=args.adam_weight_decay,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        critic_optimizer = optimizer_cls(
            fake_trainable_params_optim,
            lr=args.learning_rate_critic,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Get the training dataset
    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio
    spatial_compression_ratio = vae.config.spatial_compression_ratio
    
    if args.fix_sample_size is not None and args.enable_bucket:
        args.video_sample_size = max(max(args.fix_sample_size), args.video_sample_size)
        args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
        args.training_with_video_token_length = False
        args.random_hw_adapt = False

    # Get the dataset
    # DFD needs paired videos for the teacher-score input, everything else reuses the DMD data path.
    need_real_video = args.train_mode != "normal"
    if need_real_video:
        train_dataset = ImageVideoDataset(
            args.train_data_meta, args.train_data_dir,
            video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride, video_sample_n_frames=args.video_sample_n_frames, 
            video_repeat=args.video_repeat, 
            image_sample_size=args.image_sample_size,
            text_drop_ratio=0.0,
            enable_bucket=args.enable_bucket, enable_inpaint=True if args.train_mode != "normal" else False,
        )
    else:
        train_dataset = TextDataset(
            args.train_data_meta
        )

    def get_length_to_frame_num(token_length):
        if args.image_sample_size > args.video_sample_size:
            sample_sizes = list(range(args.video_sample_size, args.image_sample_size + 1, 128))

            if sample_sizes[-1] != args.image_sample_size:
                sample_sizes.append(args.image_sample_size)
        else:
            sample_sizes = [args.image_sample_size]
        
        length_to_frame_num = {
            sample_size: min(token_length / sample_size / sample_size, args.video_sample_n_frames) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1 for sample_size in sample_sizes
        }

        return length_to_frame_num

    if args.enable_bucket and need_real_video:
        aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset, 
            batch_size=args.train_batch_size, train_folder = args.train_data_dir, drop_last=True,
            aspect_ratios=aspect_ratio_sample_size,
        )

        def collate_fn(examples):
            # Get token length
            target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
            length_to_frame_num = get_length_to_frame_num(target_token_length)

            # Create new output
            new_examples                 = {}
            new_examples["target_token_length"] = target_token_length
            new_examples["pixel_values"] = []
            new_examples["text"]         = []
            # Used in Inpaint mode 
            if args.train_mode != "normal":
                new_examples["mask_pixel_values"] = []
                new_examples["mask"] = []
                new_examples["clip_pixel_values"] = []

            # Get downsample ratio in image and videos
            pixel_value     = examples[0]["pixel_values"]
            data_type       = examples[0]["data_type"]
            f, h, w, c      = np.shape(pixel_value)
            if data_type == 'image':
                random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(args.image_sample_size, image_ratio=[args.image_sample_size / args.video_sample_size], rng=rng)

                aspect_ratio_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}
                
                batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
            else:
                if args.random_hw_adapt:
                    if args.training_with_video_token_length:
                        local_min_size = np.min(np.array([np.mean(np.array([np.shape(example["pixel_values"])[1], np.shape(example["pixel_values"])[2]])) for example in examples]))
                        # The video will be resized to a lower resolution than its own.
                        choice_list = [length for length in list(length_to_frame_num.keys()) if length < local_min_size * 1.25]
                        if len(choice_list) == 0:
                            choice_list = list(length_to_frame_num.keys())
                        if rng is None:
                            local_video_sample_size = np.random.choice(choice_list)
                        else:
                            local_video_sample_size = rng.choice(choice_list)
                        batch_video_length = length_to_frame_num[local_video_sample_size]
                        random_downsample_ratio = args.video_sample_size / local_video_sample_size
                    else:
                        random_downsample_ratio = get_random_downsample_ratio(
                                args.video_sample_size, rng=rng)
                        batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
                else:
                    random_downsample_ratio = 1
                    batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval

                aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}

            if args.fix_sample_size is not None:
                fix_sample_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in args.fix_sample_size]
            elif args.random_ratio_crop:
                if rng is None:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                else:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        rng.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                random_sample_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in random_sample_size]
            else:
                closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
                closest_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in closest_size]

            min_example_length = min(
                [example["pixel_values"].shape[0] for example in examples]
            )
            batch_video_length = int(min(batch_video_length, min_example_length))
            
            # Magvae needs the number of frames to be 4n + 1.
            batch_video_length = (batch_video_length - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1

            if batch_video_length <= 0:
                batch_video_length = 1

            for example in examples:
                if args.fix_sample_size is not None:
                    # To 0~1
                    pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.

                    # Get adapt hw for resize
                    fix_sample_size = list(map(lambda x: int(x), fix_sample_size))
                    transform = transforms.Compose([
                        transforms.Resize(fix_sample_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(fix_sample_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
                elif args.random_ratio_crop:
                    # To 0~1
                    pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.

                    # Get adapt hw for resize
                    b, c, h, w = pixel_values.size()
                    th, tw = random_sample_size
                    if th / tw > h / w:
                        nh = int(th)
                        nw = int(w / h * nh)
                    else:
                        nw = int(tw)
                        nh = int(h / w * nw)
                    
                    transform = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
                else:
                    # To 0~1
                    pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.

                    # Get adapt hw for resize
                    closest_size = list(map(lambda x: int(x), closest_size))
                    if closest_size[0] / h > closest_size[1] / w:
                        resize_size = closest_size[0], int(w * closest_size[0] / h)
                    else:
                        resize_size = int(h * closest_size[1] / w), closest_size[1]
                    
                    transform = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(closest_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
                
                new_examples["pixel_values"].append(transform(pixel_values)[:batch_video_length])
                new_examples["text"].append(example["text"])

                if args.train_mode != "normal":
                    mask = get_random_mask(new_examples["pixel_values"][-1].size(), image_start_only=True)
                    mask_pixel_values = new_examples["pixel_values"][-1] * (1 - mask) 
                    # Wan 2.1 use 0 for masked pixels
                    # + torch.ones_like(new_examples["pixel_values"][-1]) * -1 * mask
                    new_examples["mask_pixel_values"].append(mask_pixel_values)
                    new_examples["mask"].append(mask)
                    
                    clip_pixel_values = new_examples["pixel_values"][-1][0].permute(1, 2, 0).contiguous()
                    clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
                    new_examples["clip_pixel_values"].append(clip_pixel_values)

            # Limit the number of frames to the same
            new_examples["pixel_values"] = torch.stack([example for example in new_examples["pixel_values"]])
            if args.train_mode != "normal":
                new_examples["mask_pixel_values"] = torch.stack([example for example in new_examples["mask_pixel_values"]])
                new_examples["mask"] = torch.stack([example for example in new_examples["mask"]])
                new_examples["clip_pixel_values"] = torch.stack([example for example in new_examples["clip_pixel_values"]])

            # Encode prompts when enable_text_encoder_in_dataloader=True
            if args.enable_text_encoder_in_dataloader:
                prompt_ids = tokenizer(
                    new_examples['text'], 
                    max_length=args.tokenizer_max_length, 
                    padding="max_length", 
                    add_special_tokens=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
                text_input_ids = prompt_ids.input_ids
                prompt_attention_mask = prompt_ids.attention_mask

                seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                prompt_embeds = text_encoder(text_input_ids.to("cpu"), attention_mask=prompt_attention_mask.to("cpu"))[0]
                prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

                new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
                new_examples['encoder_hidden_states'] = prompt_embeds
        
                neg_txt = [
                    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" for text in batch['text']
                ]
                neg_prompt_ids = tokenizer(
                    neg_txt, 
                    max_length=args.tokenizer_max_length, 
                    padding="max_length", 
                    add_special_tokens=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
                neg_text_input_ids = neg_prompt_ids.input_ids
                neg_prompt_attention_mask = neg_prompt_ids.attention_mask

                neg_seq_lens = neg_prompt_attention_mask.gt(0).sum(dim=1).long()
                neg_prompt_embeds = text_encoder(neg_text_input_ids.to("cpu"), attention_mask=neg_prompt_attention_mask.to("cpu"))[0]
                neg_prompt_embeds = [u[:v] for u, v in zip(neg_prompt_embeds, neg_seq_lens)]

                new_examples['neg_encoder_attention_mask'] = neg_prompt_ids.attention_mask
                new_examples['neg_encoder_hidden_states'] = neg_prompt_embeds

            return new_examples
        
        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
        )
    elif not need_real_video:
        def collate_fn(examples):
            new_examples = {}
            new_examples["text"] = []
            for example in examples:
                new_examples["text"].append(example["text"])

            # Encode prompts when enable_text_encoder_in_dataloader=True
            if args.enable_text_encoder_in_dataloader:
                prompt_ids = tokenizer(
                    new_examples['text'], 
                    max_length=args.tokenizer_max_length, 
                    padding="max_length", 
                    add_special_tokens=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
                text_input_ids = prompt_ids.input_ids
                prompt_attention_mask = prompt_ids.attention_mask

                seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                prompt_embeds = text_encoder(text_input_ids.to("cpu"), attention_mask=prompt_attention_mask.to("cpu"))[0]
                prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

                new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
                new_examples['encoder_hidden_states'] = prompt_embeds
        
                neg_txt = [
                    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" for text in batch['text']
                ]
                neg_prompt_ids = tokenizer(
                    neg_txt, 
                    max_length=args.tokenizer_max_length, 
                    padding="max_length", 
                    add_special_tokens=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
                neg_text_input_ids = neg_prompt_ids.input_ids
                neg_prompt_attention_mask = neg_prompt_ids.attention_mask

                neg_seq_lens = neg_prompt_attention_mask.gt(0).sum(dim=1).long()
                neg_prompt_embeds = text_encoder(neg_text_input_ids.to("cpu"), attention_mask=neg_prompt_attention_mask.to("cpu"))[0]
                neg_prompt_embeds = [u[:v] for u, v in zip(neg_prompt_embeds, neg_seq_lens)]

                new_examples['neg_encoder_attention_mask'] = neg_prompt_ids.attention_mask
                new_examples['neg_encoder_hidden_states'] = neg_prompt_embeds

            return new_examples
        
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = BatchSampler(RandomSampler(train_dataset, generator=batch_sampler_generator), batch_size=args.train_batch_size, drop_last=True)

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
        )
    else:
        # DataLoaders creation:
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = ImageVideoSampler(RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, args.train_batch_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler, 
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    fake_score_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=critic_optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    generator_transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        generator_transformer3d, optimizer, train_dataloader, lr_scheduler
    )
    fake_score_transformer3d, critic_optimizer, fake_score_lr_scheduler= accelerator_fake_score_transformer3d.prepare(
        fake_score_transformer3d, critic_optimizer, fake_score_lr_scheduler
    )
    if fsdp_stage != 0 or zero_stage != 0:
        from functools import partial

        from videox_fun.dist import set_multi_gpus_devices, shard_model
        # Under --low_vram use FSDP CPUOffload for CPU<->GPU streaming instead of manual
        # .to(device)/.to('cpu'), which fights the FSDP flat_param.
        shard_fn = partial(
            shard_model,
            device_id=accelerator.device,
            param_dtype=weight_dtype,
            offload_to_cpu=args.low_vram,
        )
        real_score_transformer3d = shard_fn(real_score_transformer3d)
        if init_rollout_transformer3d is not None:
            init_rollout_transformer3d = shard_fn(init_rollout_transformer3d)
        text_encoder = shard_fn(text_encoder)

    # Move text_encode and vae to gpu and cast to weight_dtype.
    # vae is never sharded and needs explicit placement; sharded modules are owned by FSDP,
    # so skip .to() on them to keep the flat_param attached.
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    manual_te_move = args.low_vram and fsdp_stage == 0 and zero_stage == 0
    manual_vae_move = args.low_vram
    if fsdp_stage == 0 and zero_stage == 0:
        real_score_transformer3d.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        if init_rollout_transformer3d is not None:
            init_rollout_transformer3d.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        if not args.enable_text_encoder_in_dataloader:
            text_encoder.to(accelerator.device if not args.low_vram else "cpu")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        keys_to_pop = [k for k, v in tracker_config.items() if isinstance(v, list)]
        for k in keys_to_pop:
            tracker_config.pop(k)
            print(f"Removed tracker_config['{k}']")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
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

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
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
            print(f"Load pkl from {pkl_path}. Get first_epoch = {first_epoch}.")

            accelerator.print(f"Resuming from checkpoint {path}")
            fake_score_path = os.path.join(path, "fake_score")
            accelerator.load_state(os.path.join(args.output_dir, path))
            accelerator_fake_score_transformer3d.load_state(os.path.join(args.output_dir, fake_score_path))
    else:
        initial_global_step = 0

    progress_bar = PauseAwareTqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.multi_stream and args.train_mode != "normal":
        # create extra cuda streams to speedup inpaint vae computation
        vae_stream_1 = torch.cuda.Stream()
        vae_stream_2 = torch.cuda.Stream()
    else:
        vae_stream_1 = None
        vae_stream_2 = None

    # --- SparkDiffusion CrossDistill Training Core ---
    # rf-native: time s in [0,1], latent x_s = (1-s)*x0 + s*eps, net fed t = s*T_SCALE predicts v = eps - x0.
    T_SCALE = 1000.0   # rectified_flow_t_scaling_factor

    def shift_rf_time(u, shift):
        return shift * u / (1.0 + (shift - 1.0) * u)

    def inv_shift_rf_time(rf, shift):
        return rf / (shift - (shift - 1.0) * rf) if shift > 0 else rf

    def draw_rf_time(shape, lower_bound=None, strict_lower=False):
        """Sample the comparison/critic rf time s_D ~ p_D, clamped to [rf_t_min, rf_t_max]."""
        dev, dtp = accelerator.device, torch.float64
        lo, hi = float(args.rf_t_min), float(args.rf_t_max)
        n = int(torch.Size(shape).numel()) if not isinstance(shape, int) else int(shape)
        if args.p_D == "uniform_shift":
            shift = args.p_D_shift
            u_lo, u_hi = inv_shift_rf_time(lo, shift), inv_shift_rf_time(hi, shift)
            u = torch.rand(n, device=dev, dtype=dtp) * (u_hi - u_lo) + u_lo
            rf = shift_rf_time(u, shift)
        else:  # lognormal in log-sigma space, mapped to rf via sigma/(1+sigma)
            import torch.distributions as D
            normal = D.Normal(args.p_D_mean, args.p_D_std)
            eps_val = torch.finfo(torch.float64).eps
            lg_lo = torch.tensor(lo / (1.0 - lo + eps_val)).log() if lo > 0 else torch.tensor(-40.0, dtype=dtp)
            lg_hi = torch.tensor(hi / (1.0 - hi + eps_val)).log() if hi < 1 else torch.tensor(40.0, dtype=dtp)
            cdf_lo, cdf_hi = normal.cdf(lg_lo.double()), normal.cdf(lg_hi.double())
            q = torch.rand(n, device=dev, dtype=dtp)
            q = (cdf_lo + q * (cdf_hi - cdf_lo)).clamp(min=eps_val, max=1.0 - eps_val)
            rf = normal.icdf(q).exp()
            rf = rf / (rf + 1.0)
        if not isinstance(shape, int):
            rf = rf.view(shape)
        if lower_bound is not None:
            lb = torch.as_tensor(lower_bound, device=dev, dtype=dtp)
            while lb.ndim < rf.ndim:
                lb = lb.unsqueeze(-1)
            rf = torch.maximum(rf, lb + (1e-4 if strict_lower else 0.0))
        return rf.clamp(min=lo, max=1.0).to(torch.float32)

    # Broadcast a sampled time/noise from rank 0 so all ranks train on identical randomness.
    def sync_broadcast(t):
        if dist.is_initialized():
            t = t.clone()
            dist.broadcast(t, src=0)
        return t

    # --- Network Forwards (net takes t in [0,1000], returns rf velocity v = eps - x0) ---
    def net_velocity(net, x_5d, s_B_1, embeds, seq_len):
        t = (s_B_1.reshape(-1) * T_SCALE).to(torch.float32)
        v = net(
            x=x_5d,
            context=embeds,
            t=t,
            seq_len=seq_len,
            y=None
        )
        return v.float()

    # Invert the rf interpolation x_s = (1-s)*x0 + s*eps  =>  x0 = x_s - s*v.
    def net_x0(net, x_5d, s_B_1, embeds, seq_len):
        v = net_velocity(net, x_5d, s_B_1, embeds, seq_len)
        s5 = s_B_1.reshape(-1, 1, 1, 1, 1).to(torch.float64)
        return x_5d.double() - s5 * v.double(), v

    # One rf-Euler step: x_{s_next} = x_{s_cur} + (s_next - s_cur)*v.
    def _euler(net, x, s_cur_B_1, s_next_B_1, embeds, seq_len):
        v = net_velocity(net, x, s_cur_B_1, embeds, seq_len)
        sc = s_cur_B_1.reshape(-1, 1, 1, 1, 1).to(x.dtype)
        sn = s_next_B_1.reshape(-1, 1, 1, 1, 1).to(x.dtype)
        return x + (sn - sc) * v.to(x.dtype)

    def backward_simulation(prompt_embeds, size, n_steps, seq_len, with_grad=False, tdm=False):
        """Roll the student down an rf-Euler schedule; only the last segment carries grad."""
        start_t = args.backward_simulation_start_t
        if start_t is None or start_t < 0:
            start_t = 1.0
        end_t = args.backward_simulation_end_t
        bt = list(args.backward_timesteps)
        traj = [start_t]
        if tdm:
            for i in range(n_steps):
                traj.append(bt[i] if i < len(bt) else end_t)
        else:
            for i in range(n_steps - 1):
                traj.append(bt[i] if i < len(bt) else end_t)
            traj.append(end_t)

        # Init latent = pure noise at rf 1.0 (single expert, full range; no init-rollout relay).
        x = torch.randn(size, device=accelerator.device, generator=torch_rng, dtype=weight_dtype)

        B = size[0]
        t_ts = [torch.full((B, 1), float(v), device=accelerator.device, dtype=torch.float32) for v in traj]
        n_seg = len(traj) - 1
        for step in range(n_seg):
            ctx_fn = torch.enable_grad() if (with_grad and step == n_seg - 1) else torch.no_grad()
            with ctx_fn:
                x = _euler(generator_transformer3d, x, t_ts[step], t_ts[step + 1], prompt_embeds, seq_len)
        return x, t_ts, traj

    # Teacher rf-time grid for ode_match div: K+1 descending sigmas from sigma_max down toward 0.
    def _build_div_teacher_schedule(K, sigma_max, shift):
        N_INF = 30
        assert K + 1 <= N_INF, f"div_K + 1 = {K+1} must be <= {N_INF}"
        sigma_max_rf = sigma_max / (sigma_max + 1.0)
        unshifted = inv_shift_rf_time(sigma_max_rf, shift)
        raw = torch.linspace(unshifted, 0.0, N_INF + 1, device=accelerator.device, dtype=torch.float32)[:-1]
        sig = shift_rf_time(raw, shift)
        return sig[:K + 1]

    def is_student_phase(it):
        if args.loss_scale_tdm <= 0 and args.loss_scale_dmd <= 0:
            return True
        if it < args.critic_warmup:
            return True
        return (it - args.critic_warmup) % args.student_update_freq == 0

    def num_sim_steps(eff):
        if args.loss_scale_div > 0:
            return eff % (args.max_simulation_steps_fake - 1) + 2
        return eff % args.max_simulation_steps_fake + 1

    # Patch-size for the student, used to compute seq_len once per batch.
    _patch = accelerator.unwrap_model(generator_transformer3d).config.patch_size[1:]
    neg_txt_default = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    for epoch in range(first_epoch, args.num_train_epochs):
        train_student = 0.0
        train_div = 0.0
        train_critic = 0.0
        train_gen_cnt = 0
        batch_sampler.sampler.generator = torch.Generator().manual_seed((args.seed if args.seed is not None else 0) + epoch)
        for step, batch in enumerate(train_dataloader):
            iteration = global_step
            student_phase = is_student_phase(iteration)

            with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                # --- Build the Target Latent Shape (T2V bucketing, same as the SFT path) ---
                text = batch['text']
                if args.fix_sample_size is not None:
                    local_sample_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in args.fix_sample_size]
                    num_frames = args.video_sample_n_frames
                else:
                    if args.random_hw_adapt and args.training_with_video_token_length:
                        target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
                        length_to_frame_num = get_length_to_frame_num(target_token_length)
                        local_length = np.random.choice(list(length_to_frame_num.keys())) if rng is None else rng.choice(list(length_to_frame_num.keys()))
                        num_frames = length_to_frame_num[local_length]
                        aspect_ratio_sample_size = {key: [x / 512 * local_length for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                        aspect_ratio_key = np.random.choice(list(aspect_ratio_sample_size.keys())) if rng is None else rng.choice(list(aspect_ratio_sample_size.keys()))
                        local_sample_size = aspect_ratio_sample_size[aspect_ratio_key]
                    else:
                        num_frames = args.video_sample_n_frames
                        aspect_ratio_sample_size = {key: [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                        aspect_ratio_key = np.random.choice(list(aspect_ratio_sample_size.keys())) if rng is None else rng.choice(list(aspect_ratio_sample_size.keys()))
                        local_sample_size = aspect_ratio_sample_size[aspect_ratio_key]
                    local_sample_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in local_sample_size]
                target_shape = (
                    len(text),
                    vae.latent_channels,
                    int((num_frames - 1) // vae.temporal_compression_ratio + 1),
                    int(local_sample_size[0] // vae.spatial_compression_ratio),
                    int(local_sample_size[1] // vae.spatial_compression_ratio),
                )
                bsz, channel, nfr, height, width = target_shape
                seq_len = math.ceil((width * height) / (_patch[0] * _patch[1]) * nfr)

                # --- Optional Real-Latent Anchor (only needed by the pcm diversity modes) ---
                anchor_x0 = None
                if args.loss_scale_div > 0 and args.div_mode in ("pcm_fixed", "pcm_random") \
                        and 'pixel_values' in batch and batch.get('pixel_values', None) is not None:
                    if manual_vae_move:
                        vae.to(accelerator.device)
                    pixel_values = rearrange(batch["pixel_values"].to(accelerator.device, dtype=weight_dtype, non_blocking=True), "b f c h w -> b c f h w")
                    with torch.no_grad():
                        bs = args.vae_mini_batch
                        enc = []
                        for i in range(0, pixel_values.shape[0], bs):
                            enc.append(vae.encode(pixel_values[i:i + bs])[0].sample())
                        anchor_x0 = torch.cat(enc, dim=0)
                    target_shape = anchor_x0.size()
                    bsz, channel, nfr, height, width = target_shape
                    seq_len = math.ceil((width * height) / (_patch[0] * _patch[1]) * nfr)
                    if manual_vae_move:
                        vae.to('cpu')
                        torch.cuda.empty_cache()

                # --- Text Embeddings (cond + uncond) ---
                if args.enable_text_encoder_in_dataloader:
                    prompt_embeds = batch['encoder_hidden_states'].to(device=accelerator.device)
                    neg_prompt_embeds = batch['neg_encoder_hidden_states'].to(device=accelerator.device)
                else:
                    if manual_te_move:
                        text_encoder.to(accelerator.device)
                    with torch.no_grad():
                        prompt_ids = tokenizer(
                            text,
                            padding="max_length",
                            max_length=args.tokenizer_max_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt"
                        )
                        seq_lens = prompt_ids.attention_mask.gt(0).sum(dim=1).long()
                        prompt_embeds = text_encoder(prompt_ids.input_ids.to(accelerator.device), attention_mask=prompt_ids.attention_mask.to(accelerator.device))[0]
                        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

                        neg_ids = tokenizer(
                            [neg_txt_default for _ in text],
                            padding="max_length",
                            max_length=args.tokenizer_max_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt"
                        )
                        neg_seq_lens = neg_ids.attention_mask.gt(0).sum(dim=1).long()
                        neg_prompt_embeds = text_encoder(neg_ids.input_ids.to(accelerator.device), attention_mask=neg_ids.attention_mask.to(accelerator.device))[0]
                        neg_prompt_embeds = [u[:v] for u, v in zip(neg_prompt_embeds, neg_seq_lens)]
                    if manual_te_move:
                        text_encoder.to('cpu')
                        torch.cuda.empty_cache()

            if student_phase:
                with accelerator.accumulate(generator_transformer3d), \
                        torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    eff_it = iteration if iteration < args.critic_warmup else args.critic_warmup + (iteration - args.critic_warmup) // args.student_update_freq
                    n_steps = num_sim_steps(eff_it)
                    use_tdm = (args.distill_loss == "tdm")

                    # --- Student Distribution-Matching Loss (TDM or DMD) + optional diversity term ---
                    G, t_ts, traj = backward_simulation(
                        prompt_embeds,
                        target_shape,
                        n_steps,
                        seq_len,
                        with_grad=True,
                        tdm=use_tdm
                    )
                    s_im1 = t_ts[-1] if use_tdm else torch.zeros_like(t_ts[-1])   # x0 (t_{i-1}=0) for DMD

                    # Comparison time s_D ~ p_D, drawn strictly above the last sim step.
                    D_time = sync_broadcast(draw_rf_time((target_shape[0], 1), lower_bound=s_im1, strict_lower=use_tdm)).to(accelerator.device)
                    eps = torch.randn(target_shape, device=accelerator.device, generator=torch_rng, dtype=weight_dtype)
                    sD = D_time.reshape(-1, 1, 1, 1, 1).to(torch.float64)
                    base = G.float()   # fp32 halves this latent's footprint

                    # Re-noise x0=G up to s_D (DMD: rf interp; TDM: bridge conditioned on t_{i-1}).
                    if use_tdm:
                        s1 = s_im1.reshape(-1, 1, 1, 1, 1).to(torch.float64)
                        om = (1.0 - s1).clamp(min=1e-6)
                        sigma_cond = (((sD - s1) * (sD + s1 - 2.0 * sD * s1)).clamp(min=0.0)).sqrt() / om
                        D_xt = (((1.0 - sD) / om) * base + sigma_cond * eps.double()).to(weight_dtype)
                    else:
                        D_xt = ((1.0 - sD) * base + sD * eps.double()).to(weight_dtype)

                    # Critic (fake_score) and CFG teacher both denoise D_xt to x0.
                    with torch.no_grad():
                        x0_fake = net_x0(
                            fake_score_transformer3d,
                            D_xt,
                            D_time,
                            prompt_embeds,
                            seq_len
                        )[0]
                        x0_teacher = net_x0(
                            real_score_transformer3d,
                            D_xt,
                            D_time,
                            prompt_embeds,
                            seq_len
                        )[0]
                        if args.real_guidance_scale and args.real_guidance_scale > 1.0:
                            x0_uncond = net_x0(
                                real_score_transformer3d,
                                D_xt,
                                D_time,
                                neg_prompt_embeds,
                                seq_len
                            )[0]
                            x0_teacher = x0_uncond + args.real_guidance_scale * (x0_teacher - x0_uncond)
                        if use_tdm:
                            # TDM matches in x_{t_{i-1}} space: reproject each x0 pred to s_im1 via its implied eps.
                            def to_t_im1(x_t, x0_pred):
                                x_t = x_t.double()
                                eps_pred = (x_t - (1.0 - sD) * x0_pred) / sD.clamp(min=1e-6)
                                return (1.0 - s1) * x0_pred + s1 * eps_pred
                            tgt_fake = to_t_im1(D_xt, x0_fake)
                            tgt_tea = to_t_im1(D_xt, x0_teacher)
                        else:
                            tgt_fake, tgt_tea = x0_fake, x0_teacher
                        # Normalized score difference: (fake - teacher) / per-sample |G - teacher|_1.
                        wf = (base - tgt_tea).abs().mean(dim=[1, 2, 3, 4], keepdim=True).clamp(min=1e-5)
                        grad = (tgt_fake - tgt_tea) / wf

                    # DMD surrogate loss
                    surrogate = (base - (base - grad).detach()) ** 2
                    surrogate[torch.isnan(surrogate).flatten(start_dim=1).any(dim=1)] = 0
                    loss_dm = surrogate.sum(dim=(1, 2, 3, 4))
                    scale = args.loss_scale_tdm if use_tdm else args.loss_scale_dmd
                    dm_term = scale * loss_dm

                    # Diversity term (ode_match / pcm); zeros_like (no graph) when div<=0 or n_steps<2.
                    div_term = torch.zeros_like(dm_term)
                    if args.loss_scale_div > 0 and n_steps >= 2:
                        if args.div_mode == "ode_match":
                            # ODE-match: student 1-step velocity vs teacher ODE rollout (no_grad, CFG).
                            s_grid = _build_div_teacher_schedule(args.div_K, args.div_sigma_max, args.teacher_timestep_shift)
                            K = s_grid.shape[0] - 1
                            eps = sync_broadcast(torch.randn(target_shape, device=accelerator.device, generator=torch_rng, dtype=weight_dtype))
                            x = eps
                            with torch.no_grad():
                                for k in range(K):
                                    sc = s_grid[k].view(1, 1).expand(target_shape[0], 1).contiguous()
                                    sn = s_grid[k + 1].view(1, 1).expand(target_shape[0], 1).contiguous()
                                    v = net_velocity(
                                        real_score_transformer3d,
                                        x,
                                        sc,
                                        prompt_embeds,
                                        seq_len
                                    ).to(weight_dtype)
                                    if args.real_guidance_scale and args.real_guidance_scale > 1.0:
                                        vu = net_velocity(
                                            real_score_transformer3d,
                                            x,
                                            sc,
                                            neg_prompt_embeds,
                                            seq_len
                                        ).to(weight_dtype)
                                        v = vu + args.real_guidance_scale * (v - vu)
                                    dt = (sn - sc).reshape(-1, 1, 1, 1, 1).to(x.dtype)
                                    x = x + dt * v
                            t_tilde = float(s_grid[-1].item())
                            d = 1.0 - t_tilde
                            assert d > 1e-6, f"1 - t_tilde too small: {d}"
                            v_tilde = (eps - x) / d
                            s1 = torch.ones(target_shape[0], 1, device=accelerator.device, dtype=torch.float32)
                            v_stu = net_velocity(
                                generator_transformer3d,
                                eps,
                                s1,
                                prompt_embeds,
                                seq_len
                            )
                            diff = v_stu.double() - v_tilde.double()
                            div_loss = (diff ** 2).sum(dim=(1, 2, 3, 4))
                            div_loss[torch.isnan(div_loss).flatten(start_dim=1).any(dim=1)] = 0
                            div_term = args.loss_scale_div * div_loss
                        elif args.div_mode in ("pcm_fixed", "pcm_random"):
                            # PCM consistency: grad student step must match the no_grad teacher rollout target.
                            B = target_shape[0]
                            K = args.dcm_skipping_interval_steps
                            dw = 1.0 / args.dcm_total_steps
                            shift = args.teacher_timestep_shift
                            eps = sync_broadcast(torch.randn(target_shape, device=accelerator.device, generator=torch_rng, dtype=weight_dtype))
                            rf_b = float(args.backward_timesteps[0]) if len(args.backward_timesteps) > 0 else float(args.backward_simulation_end_t)
                            s_b = inv_shift_rf_time(rf_b, shift)
                            w_right = 1.0 - s_b
                            # phase-0 offset w in [0, valid_width]; pcm_fixed pins w=0, pcm_random samples it.
                            valid_width = w_right - K * dw
                            if args.div_mode == "pcm_fixed":
                                w = torch.zeros(B, 1, device=accelerator.device, dtype=torch.float32)
                            else:
                                assert valid_width > 0, (
                                    f"pcm-div phase 0 valid width <= 0: w_right={w_right:.4f}, K*dw={K*dw:.4f}.")
                                w = sync_broadcast(torch.rand(B, 1, device=accelerator.device) * valid_width)
                            t_list = [shift_rf_time(1.0 - (w + k * dw), shift) for k in range(K + 1)]
                            t0, tK = t_list[0], t_list[-1]
                            t_bnd = torch.full((B, 1), rf_b, device=accelerator.device, dtype=torch.float32)
                            t0c = t0.reshape(-1, 1, 1, 1, 1).double()
                            anchor = anchor_x0.double() if anchor_x0 is not None else G.detach().double()
                            xt = ((1.0 - t0c) * anchor + t0c * eps.double()).to(weight_dtype)
                            # Run the no_grad teacher rollout + target student BEFORE the grad forward (FSDP safety).
                            with torch.no_grad():
                                xk = xt
                                for k in range(K):
                                    tk, tk1 = t_list[k], t_list[k + 1]
                                    dt = (tk - tk1).reshape(-1, 1, 1, 1, 1).to(xk.dtype)
                                    F_tea = net_velocity(
                                        real_score_transformer3d,
                                        xk,
                                        tk,
                                        prompt_embeds,
                                        seq_len
                                    ).to(xk.dtype)
                                    if args.real_guidance_scale and args.real_guidance_scale > 1.0:
                                        F_unc = net_velocity(
                                            real_score_transformer3d,
                                            xk,
                                            tk,
                                            neg_prompt_embeds,
                                            seq_len
                                        ).to(xk.dtype)
                                        F_tea = F_unc + args.real_guidance_scale * (F_tea - F_unc)
                                    xk = xk - dt * F_tea
                                x_target = _euler(
                                    generator_transformer3d,
                                    xk,
                                    tK,
                                    t_bnd,
                                    prompt_embeds,
                                    seq_len
                                ).detach()
                            x_pred = _euler(
                                generator_transformer3d,
                                xt,
                                t0,
                                t_bnd,
                                prompt_embeds,
                                seq_len
                            )
                            div_loss = (x_pred.double() - x_target.double()) ** 2
                            div_loss[torch.isnan(div_loss).flatten(start_dim=1).any(dim=1)] = 0
                            div_term = args.loss_scale_div * div_loss.sum(dim=(1, 2, 3, 4))

                    avg_dm = accelerator.gather(dm_term.detach().mean().repeat(args.train_batch_size)).mean()
                    avg_div = accelerator.gather(div_term.detach().mean().repeat(args.train_batch_size)).mean()
                    train_student += avg_dm.item() / args.gradient_accumulation_steps
                    train_div += avg_div.item() / args.gradient_accumulation_steps
                    train_gen_cnt += 1

                    # Single combined backward: a second accelerator.backward() on the same FSDP module
                    # trips the "flat_param on cpu" assert under CPUOffload, so dm+div sum into one call.
                    student_loss = dm_term.mean()
                    if div_term.requires_grad:   # div_term has no graph when div<=0 or n_steps<2
                        student_loss = student_loss + div_term.mean()
                    accelerator.backward(student_loss)
                    del dm_term, div_term, student_loss
                    if accelerator.sync_gradients and args.max_grad_norm and args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                sync_now = accelerator.sync_gradients
                active_lr = lr_scheduler
            else:
                with accelerator_fake_score_transformer3d.accumulate(fake_score_transformer3d), \
                        torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    eff_it = iteration if iteration < args.critic_warmup else args.critic_warmup + (iteration - args.critic_warmup) // args.student_update_freq
                    n_steps = num_sim_steps(iteration - eff_it - 1)

                    # --- Critic (fake_score) Denoising Loss on student x0, x0-space weighted ---
                    with torch.no_grad():
                        G, _, _ = backward_simulation(
                            prompt_embeds,
                            target_shape,
                            n_steps,
                            seq_len,
                            with_grad=False,
                            tdm=False
                        )
                    D_time = sync_broadcast(draw_rf_time((target_shape[0], 1))).to(accelerator.device)
                    eps = torch.randn(target_shape, device=accelerator.device, generator=torch_rng, dtype=weight_dtype)
                    sD = D_time.reshape(-1, 1, 1, 1, 1).to(torch.float64)
                    D_xt = ((1.0 - sD) * G.double() + sD * eps.double()).to(weight_dtype)
                    x0_fake = net_x0(
                        fake_score_transformer3d,
                        D_xt,
                        D_time,
                        prompt_embeds,
                        seq_len
                    )[0]

                    # x0-space weight (s_D^2+(1-s_D)^2)/s_D^2 keeps the MSE comparable across s_D.
                    weight = (sD ** 2 + (1.0 - sD) ** 2) / sD.clamp(min=1e-6) ** 2
                    crit = (((G.double() - x0_fake) ** 2 * weight).sum(dim=(1, 2, 3, 4))).mean()

                    avg_c = accelerator.gather(crit.detach().repeat(args.train_batch_size)).mean()
                    train_critic += avg_c.item() / args.gradient_accumulation_steps
                    accelerator_fake_score_transformer3d.backward(crit)
                    if accelerator_fake_score_transformer3d.sync_gradients and args.max_grad_norm and args.max_grad_norm > 0:
                        accelerator_fake_score_transformer3d.clip_grad_norm_(fake_trainable_params, args.max_grad_norm)
                    critic_optimizer.step()
                    fake_score_lr_scheduler.step()
                    critic_optimizer.zero_grad()
                sync_now = accelerator_fake_score_transformer3d.sync_gradients
                active_lr = fake_score_lr_scheduler

            if sync_now:
                progress_bar.update(1)
                global_step += 1
                tracker_logs = {
                    "train_student_loss": train_student / max(train_gen_cnt, 1),
                    "train_div_loss": train_div / max(train_gen_cnt, 1),
                    "train_critic_loss": train_critic,
                }
                accelerator.log(tracker_logs, step=global_step)
                train_student = 0.0
                train_div = 0.0
                train_critic = 0.0
                train_gen_cnt = 0

                if global_step % args.checkpointing_steps == 0:
                    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for removing_checkpoint in checkpoints[0:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        with progress_bar.paused():
                            fake_score_save_path = os.path.join(save_path, "fake_score")
                            accelerator.save_state(save_path)
                            accelerator_fake_score_transformer3d.save_state(fake_score_save_path)
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        logger.info(f"Saved state to {save_path}")

                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    with progress_bar.paused():
                        log_validation(
                            vae, text_encoder, tokenizer, generator_transformer3d,
                            args, config, accelerator, weight_dtype, global_step,
                        )

            logs = {"train_critic_loss": train_critic, "lr": active_lr.get_last_lr()[0], "phase": "student" if student_phase else "critic"}
            if student_phase:
                logs["train_student_loss"] = train_student
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            with progress_bar.paused():
                log_validation(
                    vae, text_encoder, tokenizer, generator_transformer3d,
                    args, config, accelerator, weight_dtype, global_step,
                )

    # Close the bar before the end-of-run checkpoint.
    progress_bar.close()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        generator_transformer3d = unwrap_model(generator_transformer3d)

    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        fake_score_save_path = os.path.join(save_path, "fake_score")
        accelerator.save_state(save_path)
        accelerator_fake_score_transformer3d.save_state(fake_score_save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
