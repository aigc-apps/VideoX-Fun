"""Modified from VideoX-Fun/scripts/wan2.2_fun/train_lora.py
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
import json
import logging
import math
import os
import pickle
import random
import shutil
import sys
from contextlib import contextmanager
from typing import List, Optional, Union

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from torch.utils.data import BatchSampler

import videox_fun.reward.reward_fn as reward_fn
from videox_fun.data import (ASPECT_RATIO_512, ASPECT_RATIO_RANDOM_CROP_512,
                             ASPECT_RATIO_RANDOM_CROP_PROB,
                             AspectRatioBatchImageVideoSampler,
                             ImageVideoDataset, ImageVideoSampler,
                             RandomSampler, TextDataset, get_closest_ratio,
                             get_random_mask)
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                               AutoencoderKLWan3_8_1Steps,
                               AutoencoderKLWan_1Steps,
                               Wan2_2Transformer3DModel, WanT5EncoderModel)
from videox_fun.pipeline import Wan2_2I2VPipeline, Wan2_2Pipeline
from videox_fun.utils.lora_utils import create_network, merge_lora
from videox_fun.utils.tqdm_bar import PauseAwareTqdm
from videox_fun.utils.utils import get_image_to_video_latent, save_videos_grid

if is_wandb_available():
    import wandb


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


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


def _get_t5_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]] = None,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]


def encode_prompt(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    negative_prompt: Optional[Union[str, List[str]]] = None,
    do_classifier_free_guidance: bool = True,
    num_videos_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
            Whether to use classifier free guidance or not.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        device: (`torch.device`, *optional*):
            torch device
        dtype: (`torch.dtype`, *optional*):
            torch dtype
    """
    # device = device or self._execution_device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_embeds = _get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

    if do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        negative_prompt_embeds = _get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

    return prompt_embeds, negative_prompt_embeds


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    import inspect

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


def log_validation(vae, text_encoder, tokenizer, transformer3d, network, config, args, accelerator, weight_dtype, global_step, other_transformer3d=None):
    try:
        logger.info("Running validation... ")
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )

        # Build dual-transformer setup based on boundary_type (aligned with train_lora.py).
        # The non-trainable expert is ALREADY resident on GPU during training (FSDP-sharded and
        # used by the denoise rollout), so reuse it via `other_transformer3d` instead of reloading
        # an unsharded full copy from disk -- the latter would spike VRAM by a whole ~14B expert on top of the resident reward models.
        if args.boundary_type == "full":
            transformer3d_1 = accelerator.unwrap_model(transformer3d) if type(transformer3d).__name__ == 'DistributedDataParallel' else transformer3d
            transformer3d_2 = None
        else:
            if other_transformer3d is None:
                raise ValueError("boundary_type != 'full' requires passing the resident other expert via `other_transformer3d`.")
            if args.boundary_type == "low":
                # transformer3d is the trained low-noise expert; other_transformer3d is the resident high-noise one.
                transformer3d_1 = accelerator.unwrap_model(transformer3d) if type(transformer3d).__name__ == 'DistributedDataParallel' else transformer3d
                transformer3d_2 = other_transformer3d
            else:
                # transformer3d is the trained high-noise expert; other_transformer3d is the resident low-noise one.
                transformer3d_1 = other_transformer3d
                transformer3d_2 = accelerator.unwrap_model(transformer3d) if type(transformer3d).__name__ == 'DistributedDataParallel' else transformer3d

        if args.train_mode != "normal":
            pipeline = Wan2_2I2VPipeline(
                vae=accelerator.unwrap_model(vae).to(weight_dtype),
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                transformer=transformer3d_1,
                transformer_2=transformer3d_2,
                scheduler=scheduler,
            )
        else:
            pipeline = Wan2_2Pipeline(
                vae=accelerator.unwrap_model(vae).to(weight_dtype),
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                transformer=transformer3d_1,
                transformer_2=transformer3d_2,
                scheduler=scheduler,
            )
        pipeline = pipeline.to(accelerator.device)

        if args.seed is None:
            generator = None
        else:
            rank_seed = args.seed + accelerator.process_index
            generator = torch.Generator(device=accelerator.device).manual_seed(rank_seed)
            logger.info(f"Rank {accelerator.process_index} using seed: {rank_seed}")

        for i in range(len(args.validation_prompts)):
            with torch.no_grad():
                if args.train_mode != "normal":
                    with torch.autocast("cuda", dtype=weight_dtype):
                        video_length = int((args.video_sample_n_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.video_sample_n_frames != 1 else 1
                        input_video, input_video_mask, _ = get_image_to_video_latent(None, None, video_length=video_length, sample_size=[args.video_sample_size, args.video_sample_size])
                        sample = pipeline(
                            args.validation_prompts[i],
                            num_frames = video_length,
                            negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                            height      = args.video_sample_size,
                            width       = args.video_sample_size,
                            generator   = generator,
                            video        = input_video,
                            mask_video   = input_video_mask,
                            num_inference_steps = args.num_inference_steps,
                            guidance_scale      = args.guidance_scale,
                            boundary            = config['transformer_additional_kwargs'].get('boundary', 0.900)
                        ).videos
                        os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                        save_videos_grid(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.mp4"))
                else:
                    with torch.autocast("cuda", dtype=weight_dtype):
                        sample = pipeline(
                            args.validation_prompts[i],
                            num_frames = args.video_sample_n_frames,
                            negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                            height      = args.video_sample_size,
                            width       = args.video_sample_size,
                            generator   = generator,
                            num_inference_steps = args.num_inference_steps,
                            guidance_scale      = args.guidance_scale,
                            boundary            = config['transformer_additional_kwargs'].get('boundary', 0.900)
                        ).videos
                        os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                        save_videos_grid(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.mp4"))

        del pipeline
        # NOTE: transformer3d_1/transformer3d_2 may reference the *shared* training experts
        # (via other_transformer3d), so they must NOT be deleted here.
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        transformer3d.to(accelerator.device, dtype=weight_dtype)
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error with info {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
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
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
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
        help="Whether or not to use gradient checkpointing (for DiT) to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--vae_gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing (for VAE) to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--use_1steps_vae",
        action="store_true",
        help="Use the single-pass (no feat_cache) variant of the configured Wan VAE (AutoencoderKLWan -> AutoencoderKLWan_1Steps, AutoencoderKLWan3_8 -> AutoencoderKLWan3_8_1Steps) instead of the chunked causal VAE. Numerically equivalent, but decodes the whole latent in one forward so gradient checkpointing is applied per decoder block cleanly and is not limited to the first 3 latent frames.",
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
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
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
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alpha",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
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
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )
    parser.add_argument("--save_state", action="store_true", help="Whether or not to save state.")

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
        "--boundary_type",
        type=str,
        default="low",
        help=(
            'The training boundary type for dual-Transformer architecture. '
            'Support `"low"` (train low-noise model), `"high"` (train high-noise model), '
            'and `"full"` (single model training like 5B).'
        ),
    )
    parser.add_argument(
        "--lora_skip_name",
        type=str,
        default=None,
        help=("The module is not trained in loras. "),
    )
    parser.add_argument(
        "--target_name",
        type=str,
        default=None,
        help=("The module is trained in loras. "),
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="inpaint",
        help=(
            'The training mode. Support `"normal"` (T2V), '
            '`"inpaint"` (I2V), and `"ti2v"` (Text-Image-to-Video, 5B only).'
        ),
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
        "--token_sample_size",
        type=int,
        default=512,
        help="Sample size of the token.",
    )
    parser.add_argument(
        "--fix_sample_size", 
        nargs=2, type=int, default=None,
        help="Fix Sample size [height, width] when using bucket and collate_fn."
    )
    parser.add_argument(
        '--tokenizer_max_length', 
        type=int,
        default=512,
        help='Max length of tokenizer'
    )

    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        required=True,
        help="The path to the training data annotation file (.json).",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Root directory for resolving relative file paths in the training data.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Sample size (height and width) of the video for training.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="Sample size of the image.",
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
        help="Number of frames to sample from the video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Number of times to repeat video entries in the dataset.",
    )
    parser.add_argument(
        '--eta', 
        type=float,
        default=0.0,
        help='eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, '
        'with 0.0 being fully deterministic and 1.0 being equivalent to the DDPM sampler.'
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float,
        default=6.0,
        help="The classifier-free diffusion guidance."
    )
    parser.add_argument(
        "--num_inference_steps", 
        type=int,
        default=50,
        help="The number of denoising steps in training and validation."
    )
    parser.add_argument(
        "--num_decoded_latents",
        type=int,
        default=3,
        help="The number of latents to be decoded."
    )
    parser.add_argument(
        "--num_sampled_frames",
        type=int,
        default=None,
        help="The number of sampled frames for the reward function."
    )
    parser.add_argument(
        "--num_reward_frames",
        type=int,
        default=None,
        help="Randomly select this many decoded frames for reward evaluation. None = use all."
    )
    parser.add_argument(
        "--reward_fn", 
        type=str,
        default="HPSReward",
        help='The reward function. Comma-separated names (e.g. "HPSv3Reward,VideoAlignReward") enable multi-reward training; the rewards\' differentiable losses are combined by a weighted sum.'
    )
    parser.add_argument(
        "--reward_fn_kwargs",
        type=str,
        default=None,
        help='The keyword arguments of the reward function. Flat JSON dict for a single reward; nested JSON dict keyed by reward name for multiple rewards, e.g. \'{"HPSv3Reward": {"max_reward": 10}, "VideoAlignReward": {}}\'. '
    )
    parser.add_argument(
        "--multi_reward_weights",
        type=str,
        default=None,
        help='JSON dict of per-reward loss weights for multi-reward, e.g. \'{"HPSv3Reward": 0.5, "VideoAlignReward": 0.5}\'. Normalized to sum to 1; if omitted with multiple rewards, equal weights are used. Ignored for a single reward.'
    )
    parser.add_argument(
        "--first_frame_mse_weight",
        type=float,
        default=0.0,
        help="Weight for first-frame latent MSE loss in inpaint mode. "
             "Anchors the denoised first frame to the condition image. 0 = disabled.",
    )
    parser.add_argument(
        "--backprop",
        action="store_true",
        default=False,
        help="Whether to use the reward backprop training mode.",
    )
    parser.add_argument(
        "--backprop_step_list",
        nargs="+",
        type=int,
        default=None,
        help="The preset step list for reward backprop. If provided, overrides `backprop_strategy`."
    )
    parser.add_argument(
        "--backprop_strategy",
        choices=["last", "tail", "uniform", "random"],
        default="last",
        help="The strategy for reward backprop."
    )
    parser.add_argument(
        "--stop_latent_model_input_gradient",
        action="store_true",
        default=False,
        help="Whether to stop the gradient of the latents during reward backprop.",
    )
    parser.add_argument(
        "--backprop_random_start_step",
        type=int,
        default=0,
        help="The random start step for reward backprop. Only used when `backprop_strategy` is random."
    )
    parser.add_argument(
        "--backprop_random_end_step",
        type=int,
        default=50,
        help="The random end step for reward backprop. Only used when `backprop_strategy` is random."
    )
    parser.add_argument(
        "--backprop_num_steps",
        type=int,
        default=5,
        help="The number of steps for backprop. Only used when `backprop_strategy` is tail/uniform/random."
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

    # Sanity check for backpropagation
    if args.backprop:
        if args.backprop_step_list is not None:
            logger.warning(
                f"The backprop_strategy {args.backprop_strategy} will be ignored "
                f"when using backprop_step_list {args.backprop_step_list}."
            )
            assert any(step <= args.num_inference_steps - 1 for step in args.backprop_step_list)
        else:
            if args.backprop_strategy in set(["tail", "uniform", "random"]):
                assert args.backprop_num_steps <= args.num_inference_steps - 1
            if args.backprop_strategy == "random":
                assert args.backprop_random_start_step <= args.backprop_random_end_step
                assert args.backprop_random_end_step <= args.num_inference_steps - 1

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None

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

    # Load scheduler, tokenizer and models.
    # TODO: support other noise schedulers (FlowUniPCMultistepScheduler and FlowDPMSolverMultistepScheduler)
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
        if args.use_1steps_vae:
            # Opt-in single-pass VAE: same weights / interface, whole-latent decode
            # with per-block gradient checkpointing (see --use_1steps_vae help).
            Chosen_AutoencoderKL = {
                AutoencoderKLWan: AutoencoderKLWan_1Steps,
                AutoencoderKLWan3_8: AutoencoderKLWan3_8_1Steps,
            }[Chosen_AutoencoderKL]
        vae = Chosen_AutoencoderKL.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        vae.eval()

        # Some reward models comes from `transformers`. We also need `unset_hf_deepspeed_config` to prevent the 
        # model from being partitioned during `zero.Init`.
        from transformers.integrations.deepspeed import \
            unset_hf_deepspeed_config
        unset_hf_deepspeed_config()

        # loss function(s): a single reward, or multiple comma-separated rewards combined by a
        # weighted sum of their differentiable losses. Multi-reward keeps every reward model on GPU
        # at once (the whole graph is backpropagated together), so it costs extra VRAM; enabling
        # --gradient_checkpointing mitigates this per reward model.
        reward_fn_names = [name.strip() for name in args.reward_fn.split(',') if name.strip()]
        is_multi_reward = len(reward_fn_names) > 1

        reward_fn_kwargs_all = {}
        if args.reward_fn_kwargs is not None:
            parsed_kwargs = json.loads(args.reward_fn_kwargs)
            if isinstance(parsed_kwargs, dict) and (is_multi_reward or any(n in parsed_kwargs for n in reward_fn_names)):
                reward_fn_kwargs_all = parsed_kwargs
            else:
                reward_fn_kwargs_all = {reward_fn_names[0]: parsed_kwargs}

        multi_reward_weights = {}
        if args.multi_reward_weights is not None:
            multi_reward_weights = json.loads(args.multi_reward_weights)
        if is_multi_reward:
            if not multi_reward_weights:
                multi_reward_weights = {n: 1.0 / len(reward_fn_names) for n in reward_fn_names}
            else:
                total = sum(multi_reward_weights.values())
                multi_reward_weights = {n: multi_reward_weights.get(n, 0.0) / total for n in reward_fn_names}

        # The main process builds each reward first so HF weight download happens once.
        if accelerator.is_main_process:
            for _name in reward_fn_names:
                getattr(reward_fn, _name)(device="cpu", dtype=weight_dtype, **reward_fn_kwargs_all.get(_name, {}))
        accelerator.wait_for_everyone()

        loss_fns = {}
        for _name in reward_fn_names:
            loss_fns[_name] = getattr(reward_fn, _name)(
                device=accelerator.device, dtype=weight_dtype, **reward_fn_kwargs_all.get(_name, {})
            )
            # Enable gradient checkpointing for the reward model to save VRAM during differentiable training.
            if args.gradient_checkpointing and hasattr(loss_fns[_name], 'inferencer'):
                if hasattr(loss_fns[_name].inferencer.model, 'gradient_checkpointing_enable'):
                    loss_fns[_name].inferencer.model.gradient_checkpointing_enable()
                    logger.info(f"Enabled gradient checkpointing for reward model {_name}.")
        if is_multi_reward:
            logger.info(f"Multi-reward enabled: {reward_fn_names} with weights {multi_reward_weights}")
        else:
            logger.info(f"Single reward function: {reward_fn_names[0]}")
    
    # Get Transformer
    # To unify the LoRA training for the 5/14B (including the high and low noise model),
    # the script contains numerous transformer3d variables.
    # `transformer3d` refers to the (sub)model currently being trained.
    # `low_transformer3d` is an instance of the 14B low noise model.
    # `high_transformer3d` is an instance of the 14B high noise model.
    # `local_transformer` refers to the (sub)model actually used during the denoising process.
    if args.boundary_type == "low" or args.boundary_type == "full":
        sub_path = config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')
    else:
        sub_path = config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')
    if args.boundary_type != "full":
        # 14B dual-expert: BOTH experts must always be loaded from their own subpath,
        # independent of which one is trainable. `sub_path` above selects the *trainable*
        # expert, so it must NOT be reused for `low_transformer3d`: when boundary_type=="high"
        # sub_path points at the high-noise model, and loading the low-noise expert from it
        # makes the denoising loop run the WRONG expert on the t<boundary refinement steps
        # (line ~2119), producing noisy / oversaturated samples even at step 0.
        low_sub_path = config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')
        high_sub_path = config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')
        low_transformer3d = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, low_sub_path),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True
        ).to(weight_dtype)
        high_transformer3d = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, high_sub_path),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True
        ).to(weight_dtype)
    else:
        # 5B
        transformer3d = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, sub_path),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True
        ).to(weight_dtype)


    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if args.boundary_type != "full":
        low_transformer3d.requires_grad_(False)
        high_transformer3d.requires_grad_(False)
        if args.boundary_type == "low":
            transformer3d = low_transformer3d
        else:
            transformer3d = high_transformer3d
    else:
        transformer3d.requires_grad_(False)

    # Lora will work with this...
    network = create_network(
        1.0,
        args.rank,
        args.network_alpha,
        text_encoder,
        transformer3d,
        neuron_dropout=None,
        target_name=args.target_name,
        skip_name=args.lora_skip_name,
    )
    network = network.to(weight_dtype)
    network.apply_to(text_encoder, transformer3d, args.train_text_encoder and not args.training_with_video_token_length, True)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
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

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        if fsdp_stage != 0 or zero_stage == 3:
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    from safetensors.torch import save_file
                    safetensor_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model.safetensors")
                    network_state_dict = {}
                    for key in accelerate_state_dict:
                        if "network" in key:
                            network_state_dict[key.replace("network.", "")] = accelerate_state_dict[key].to(weight_dtype)
                    save_file(network_state_dict, safetensor_save_path, metadata={"format": "pt"})

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
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    from safetensors.torch import save_file
                    safetensor_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model.safetensors")
                    network_state_dict = {}
                    for key in accelerate_state_dict:
                        if "network" in key:
                            network_state_dict[key.replace("network.", "")] = accelerate_state_dict[key].to(weight_dtype)
                    save_file(network_state_dict, safetensor_save_path, metadata={"format": "pt"})

                    if not args.use_deepspeed:
                        for _ in range(len(weights)):
                            weights.pop()

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

            def load_model_hook(models, input_dir):
                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                        batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                    print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
        # When training the high noise reward lora, the low noise transformer is still in the computation graph.
        if args.boundary_type == "high":
            low_transformer3d.enable_gradient_checkpointing()

    if args.vae_gradient_checkpointing:
        # Chunked causal VAE decodes autoregressively via feat_cache, so checkpointing
        # is only safe while the cache stays unused (num_decoded_latents <= 3); beyond
        # that the per-frame cache threading fights with activation recomputation.
        # The single-pass VAE (--use_1steps_vae) has no cache and checkpoints per
        # decoder block, so it is exempt from this limit.
        if not args.use_1steps_vae and args.num_decoded_latents > 3:
            raise ValueError(
                "vae_gradient_checkpointing with the chunked causal VAE only supports "
                "num_decoded_latents <= 3 (feat_cache kicks in beyond that). Pass "
                "--use_1steps_vae to lift this limit."
            )
        vae.enable_gradient_checkpointing()

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

    logging.info("Add network parameters")
    trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
    trainable_params_optim = network.prepare_optimizer_params(args.learning_rate / 2, args.learning_rate, args.learning_rate)

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
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

    # Get the training dataset
    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio
    spatial_compression_ratio = vae.config.spatial_compression_ratio
    
    if args.fix_sample_size is not None and args.enable_bucket:
        args.video_sample_size = max(max(args.fix_sample_size), args.video_sample_size)
        args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
        args.training_with_video_token_length = False
        args.random_hw_adapt = False

    # For pure T2V ("normal") reward backprop, only prompts are needed: rollouts are
    # generated from pure noise (torch.randn latent, shape derived from args) and scored
    # by the reward model, so no real media is loaded. This mirrors train_grpo_lora.py's
    # text_only branch. I2V/inpaint (train_mode != "normal") still uses ImageVideoDataset
    # for the conditioning frames/masks.
    # NOTE: in text-only mode --train_data_meta must be a JSON array of {"text": ...}.
    text_only = (args.train_mode == "normal")
    if text_only:
        train_dataset = TextDataset(
            args.train_data_meta,
            text_drop_ratio=0.0,
        )
    else:
        train_dataset = ImageVideoDataset(
            args.train_data_meta, args.train_data_dir,
            video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride, video_sample_n_frames=args.video_sample_n_frames, 
            video_repeat=args.video_repeat, 
            image_sample_size=args.image_sample_size,
            enable_bucket=args.enable_bucket, enable_inpaint=True if args.train_mode != "normal" else False,
        )

    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            print(f"worker_init_fn with {_seed + worker_id}")
            np.random.seed(_seed + worker_id)
            random.seed(_seed + worker_id)
        return _worker_init_fn

    if text_only:
        # Prompt-only pipeline (pure T2V): the batch carries just text. The latent shape
        # is derived from args inside the training loop and prompts are re-encoded there
        # via encode_prompt(train_prompt), so no bucket / aspect-ratio logic is needed.
        def collate_fn(examples):
            new_examples = {}
            new_examples["text"] = [example["text"] for example in examples]
            return new_examples

        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = BatchSampler(
            RandomSampler(train_dataset, generator=batch_sampler_generator),
            batch_size=args.train_batch_size, drop_last=True,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
        )
    elif args.enable_bucket:
        aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset, 
            batch_size=args.train_batch_size, train_folder = args.train_data_dir, drop_last=True,
            aspect_ratios=aspect_ratio_sample_size,
        )

        def collate_fn(examples):
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

            def get_random_downsample_ratio(sample_size, image_ratio=[],
                                            all_choices=False, rng=None):
                def _create_special_list(length):
                    if length == 1:
                        return [1.0]
                    if length >= 2:
                        first_element = 0.90
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
                random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(args.image_sample_size, image_ratio=[args.image_sample_size / args.video_sample_size])
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
                        local_video_sample_size = np.random.choice(choice_list)
                        batch_video_length = length_to_frame_num[local_video_sample_size]
                        random_downsample_ratio = args.video_sample_size / local_video_sample_size
                    else:
                        random_downsample_ratio = get_random_downsample_ratio(args.video_sample_size)
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
                encoder_hidden_states = text_encoder(
                    prompt_ids.input_ids
                )[0]
                new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
                new_examples['encoder_hidden_states'] = encoder_hidden_states

            return new_examples
        
        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
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
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
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

    # Prepare everything with our `accelerator`.
    transformer3d.network = network
    transformer3d = transformer3d.to(dtype=weight_dtype)
    transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, train_dataloader, lr_scheduler
    )
    # FSDP returns a new wrapper object; re-sync low/high_transformer3d references
    # so that the training loop's `local_transformer` uses the FSDP-wrapped model.
    if args.boundary_type != "full":
        if args.boundary_type == "low":
            low_transformer3d = transformer3d
        else:
            high_transformer3d = transformer3d

    if fsdp_stage != 0 or zero_stage != 0:
        from functools import partial

        from videox_fun.dist import set_multi_gpus_devices, shard_model
        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype)
        text_encoder = shard_fn(text_encoder)

        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype)
        if args.boundary_type == "high":
            low_transformer3d = shard_fn(low_transformer3d)
        else:
            high_transformer3d = shard_fn(high_transformer3d)

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    if args.boundary_type != "full":
        if args.low_vram:
            # In low_vram mode, only place the primary training transformer on GPU.
            # The secondary model stays on CPU and is loaded on-demand in the denoising loop.
            if args.boundary_type == "low":
                low_transformer3d.to(accelerator.device, dtype=weight_dtype)
            else:
                high_transformer3d.to(accelerator.device, dtype=weight_dtype)
        else:
            high_transformer3d.to(accelerator.device, dtype=weight_dtype)
            low_transformer3d.to(accelerator.device, dtype=weight_dtype)
    else:
        transformer3d.to(accelerator.device, dtype=weight_dtype)
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

            checkpoint_folder_path = os.path.join(args.output_dir, path)
            pkl_path = os.path.join(checkpoint_folder_path, "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    _, first_epoch = pickle.load(file)
            else:
                first_epoch = global_step // num_update_steps_per_epoch
            print(f"Load pkl from {pkl_path}. Get first_epoch = {first_epoch}.")

            if zero_stage != 3 and not args.use_fsdp:
                from safetensors.torch import load_file
                state_dict = load_file(os.path.join(checkpoint_folder_path, "lora_diffusion_pytorch_model.safetensors"), device=str(accelerator.device))
                m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

                optimizer_file_pt = os.path.join(checkpoint_folder_path, "optimizer.pt")
                optimizer_file_bin = os.path.join(checkpoint_folder_path, "optimizer.bin")
                optimizer_file_to_load = None

                if os.path.exists(optimizer_file_pt):
                    optimizer_file_to_load = optimizer_file_pt
                elif os.path.exists(optimizer_file_bin):
                    optimizer_file_to_load = optimizer_file_bin

                if optimizer_file_to_load:
                    try:
                        accelerator.print(f"Loading optimizer state from {optimizer_file_to_load}")
                        optimizer_state = torch.load(optimizer_file_to_load, map_location=accelerator.device)
                        optimizer.load_state_dict(optimizer_state)
                        accelerator.print("Optimizer state loaded successfully.")
                    except Exception as e:
                        accelerator.print(f"Failed to load optimizer state from {optimizer_file_to_load}: {e}")

                scheduler_file_pt = os.path.join(checkpoint_folder_path, "scheduler.pt")
                scheduler_file_bin = os.path.join(checkpoint_folder_path, "scheduler.bin")
                scheduler_file_to_load = None

                if os.path.exists(scheduler_file_pt):
                    scheduler_file_to_load = scheduler_file_pt
                elif os.path.exists(scheduler_file_bin):
                    scheduler_file_to_load = scheduler_file_bin

                if scheduler_file_to_load:
                    try:
                        accelerator.print(f"Loading scheduler state from {scheduler_file_to_load}")
                        scheduler_state = torch.load(scheduler_file_to_load, map_location=accelerator.device)
                        lr_scheduler.load_state_dict(scheduler_state)
                        accelerator.print("Scheduler state loaded successfully.")
                    except Exception as e:
                        accelerator.print(f"Failed to load scheduler state from {scheduler_file_to_load}: {e}")

                if hasattr(accelerator, 'scaler') and accelerator.scaler is not None:
                    scaler_file = os.path.join(checkpoint_folder_path, "scaler.pt")
                    if os.path.exists(scaler_file):
                        try:
                            accelerator.print(f"Loading GradScaler state from {scaler_file}")
                            scaler_state = torch.load(scaler_file, map_location=accelerator.device)
                            accelerator.scaler.load_state_dict(scaler_state)
                            accelerator.print("GradScaler state loaded successfully.")
                        except Exception as e:
                            accelerator.print(f"Failed to load GradScaler state: {e}")

            else:
                accelerator.load_state(checkpoint_folder_path)
                accelerator.print("accelerator.load_state() completed for zero_stage 3.")

    else:
        initial_global_step = 0

    # function for saving/removing
    def save_model(ckpt_file, unwrapped_nw):
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        if isinstance(unwrapped_nw, dict):
            from safetensors.torch import save_file
            save_file(unwrapped_nw, ckpt_file, metadata={"format": "pt"})
            return ckpt_file
        unwrapped_nw.save_weights(ckpt_file, weight_dtype, None)

    progress_bar = PauseAwareTqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    boundary = config["transformer_additional_kwargs"].get("boundary", 0.900)
    print(f"The boundary is {boundary} and the boundary_type is {args.boundary_type}.")

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        train_reward = 0.0
        train_first_frame_mse = 0.0
        # Per-reward monitoring (unweighted, keeps each reward's own scale) so a
        # multi-reward run can tell which reward is actually moving.
        train_reward_per_fn = {name: 0.0 for name in reward_fn_names}
        train_loss_per_fn = {name: 0.0 for name in reward_fn_names}
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)

        # Iterate over the dataset using DataLoader
        for step, batch in enumerate(train_dataloader):
            if (not text_only) and epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.mp4", rescale=True)
                if args.train_mode != "normal":
                    clip_pixel_values, mask_pixel_values, texts = batch['clip_pixel_values'].cpu(), batch['mask_pixel_values'].cpu(), batch['text']
                    mask_pixel_values = rearrange(mask_pixel_values, "b f c h w -> b c f h w")
                    for idx, (clip_pixel_value, pixel_value, text) in enumerate(zip(clip_pixel_values, mask_pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/mask_{gif_name[:10] if not text == '' else f'{global_step}-{idx}'}.mp4", rescale=True)

            # Extract text prompts from the batch
            train_prompt = batch["text"]
            logger.info(f"train_prompt: {train_prompt}")

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = args.guidance_scale > 1.0

            # Prepare timesteps
            # TODO: support other noise schedulers (FlowUniPCMultistepScheduler and FlowDPMSolverMultistepScheduler)
            noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device, mu=1)
            timesteps = noise_scheduler.timesteps

            # Prepare latent variables
            vae_scale_factor = spatial_compression_ratio
            tc_ratio = vae.config.temporal_compression_ratio

            # Get actual dimensions from data and align to VAE temporal constraint.
            if args.train_mode != "normal":
                mask_pixel_values = batch["mask_pixel_values"].to(weight_dtype)
                mask = batch["mask"].to(weight_dtype)
                raw_n_frames = mask.shape[1]  # [B, F, C, H, W]
                actual_n_frames = ((raw_n_frames - 1) // tc_ratio) * tc_ratio + 1
                if raw_n_frames > actual_n_frames:
                    mask_pixel_values = mask_pixel_values[:, :actual_n_frames]
                    mask = mask[:, :actual_n_frames]
                # Single image cannot be trained directly (reward model needs generated frames).
                # Pad to video_sample_n_frames so the model generates a full video from the image.
                if actual_n_frames == 1:
                    target_n_frames = args.video_sample_n_frames
                    # Fix first frame: set mask=0 (condition) and restore pixel values.
                    mask[:, 0:1] = 0
                    mask_pixel_values[:, 0:1] = batch["pixel_values"][:, 0:1].to(weight_dtype)
                    # Pad remaining frames: mask=1 (generate), pixels=0.
                    pad_mask = torch.ones(
                        mask.shape[0], target_n_frames - 1, mask.shape[2], mask.shape[3], mask.shape[4],
                        dtype=mask.dtype, device=mask.device
                    )
                    mask = torch.cat([mask, pad_mask], dim=1)
                    pad_pixels = torch.zeros(
                        mask_pixel_values.shape[0], target_n_frames - 1,
                        mask_pixel_values.shape[2], mask_pixel_values.shape[3], mask_pixel_values.shape[4],
                        dtype=mask_pixel_values.dtype, device=mask_pixel_values.device
                    )
                    mask_pixel_values = torch.cat([mask_pixel_values, pad_pixels], dim=1)
                    actual_n_frames = target_n_frames
                actual_h = mask.shape[3]
                actual_w = mask.shape[4]
            else:
                actual_n_frames = args.video_sample_n_frames
                actual_h = args.video_sample_size
                actual_w = args.video_sample_size

            # Make the inpaint latents to be zeros.
            if args.train_mode != "normal":
                t2v_flag = [(_mask == 1).all() for _mask in mask]
                new_t2v_flag = []
                for _mask in t2v_flag:
                    if _mask and np.random.rand() < 0.90:
                        new_t2v_flag.append(0)
                    else:
                        new_t2v_flag.append(1)
                t2v_flag = torch.from_numpy(np.array(new_t2v_flag)).to(accelerator.device, dtype=weight_dtype)

            latent_shape = [
                args.train_batch_size,
                vae.config.latent_channels,
                int((actual_n_frames - 1) // tc_ratio + 1) if actual_n_frames != 1 else 1,
                actual_h // vae_scale_factor,
                actual_w // vae_scale_factor,
            ]

            with accelerator.accumulate(transformer3d):
                # Keep the cross-step accumulation state in float32 (like train_grpo_lora.py's
                # sample_with_cfg, which clones latents to fp32). bf16 has ~3 decimal digits of
                # mantissa, and with the aggressive shift=12 schedule the first ~19/30 steps sit in
                # the very-high-noise region with large Euler strides, so storing x_t in bf16
                # compounds round-off into visible banding/noise even at step 0 (identity LoRA).
                # The model input is cast back to weight_dtype right before the forward below.
                latents = torch.randn(*latent_shape, device=accelerator.device, dtype=torch.float32, generator=torch_rng)
                bsz, channel, num_latent_frames, latent_h, latent_w = latents.size()

                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)

                with torch.no_grad():
                    # This way is quicker when batch grows up
                    def _batch_encode_vae(pixel_values):
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i : i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim = 0)

                    if args.train_mode != "normal":
                        mask = rearrange(mask, "b f c h w -> b c f h w")
                        mask = torch.concat(
                            [
                                torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2), 
                                mask[:, :, 1:]
                            ], dim=2
                        )
                        mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4])
                        mask = mask.transpose(1, 2)
                        mask_conditions = F.interpolate(mask[:, :1], size=latents.size()[-3:], mode='trilinear', align_corners=True).to(accelerator.device, weight_dtype)

                        # Encode inpaint latents.
                        masked_video_latents = _batch_encode_vae(mask_pixel_values).to(accelerator.device, weight_dtype)
                        if args.train_mode == "ti2v":
                            # ti2v: mask uses ti2v-style interpolation; masked_video_latents stays raw for post-step blending
                            mask_latents = mask_conditions
                        else:
                            # 14B (i2v): standard inpaint mask + masked_video_latents
                            mask = resize_mask(1 - mask, latents)
                            mask_latents = mask
                            inpaint_latents = torch.concat([mask, masked_video_latents], dim=1)
                            inpaint_latents = t2v_flag[:, None, None, None, None] * inpaint_latents

                if args.low_vram:
                    vae.to('cpu')
                    torch.cuda.empty_cache()

                # Reduce the vram by offload text encoders
                if args.low_vram:
                    torch.cuda.empty_cache()
                    text_encoder.to(accelerator.device)

                # Encode input prompt
                prompt_embeds, negative_prompt_embeds = encode_prompt(
                    tokenizer,
                    text_encoder,
                    train_prompt,
                    negative_prompt=["色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"] * len(train_prompt),
                    device=accelerator.device,
                    dtype=weight_dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )
                if do_classifier_free_guidance:
                    in_prompt_embeds = negative_prompt_embeds + prompt_embeds
                else:
                    in_prompt_embeds = prompt_embeds

                # Reduce the vram by offload text encoders
                if args.low_vram:
                    text_encoder.to("cpu")
                    torch.cuda.empty_cache()

                if args.train_mode != "normal":
                    if args.train_mode != "ti2v":
                        # non-ti2v: extract from concatenated inpaint_latents
                        mask_latents = inpaint_latents[:, :4]
                        masked_video_latents = inpaint_latents[:, 4:]
                    # else: 5B (ti2v) — mask_latents and masked_video_latents already set above
                else:
                    # T2V mode: all zeros (generate everything)
                    mask_latents = torch.tile(
                        torch.zeros_like(latents)[:, :1].to(accelerator.device, weight_dtype), [1, 4, 1, 1, 1]
                    )
                    masked_video_latents = torch.zeros_like(latents).to(accelerator.device, weight_dtype)
                    if args.train_mode == "ti2v":
                        mask_conditions = torch.ones_like(latents).to(accelerator.device, weight_dtype)[:, :1]

                if args.train_mode == "ti2v":
                    mask_conditions[:, :, 1:, :, :] = 1

                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                # Prepare extra step kwargs.
                extra_step_kwargs = prepare_extra_step_kwargs(noise_scheduler, generator, args.eta)

                target_shape = (vae.latent_channels, num_latent_frames, latent_w, latent_h)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) /
                    (accelerator.unwrap_model(transformer3d).config.patch_size[1] * accelerator.unwrap_model(transformer3d).config.patch_size[2]) *
                    target_shape[1]
                )

                num_inference_steps_for_backprop = args.num_inference_steps
                if args.boundary_type == "high":
                    # When training the high noise reward lora, the denoising steps in the high noise are eligible for backpropagation.
                    num_inference_steps_for_backprop = 0
                    for i, t in enumerate(noise_scheduler.timesteps):
                        if t >= boundary * noise_scheduler.config.num_train_timesteps:
                            num_inference_steps_for_backprop += 1
                
                # Denoising loop
                if args.backprop:
                    if args.backprop_step_list is None:
                        if args.backprop_strategy == "last":
                            backprop_step_list = [num_inference_steps_for_backprop - 1]
                        elif args.backprop_strategy == "tail":
                            backprop_step_list = list(range(num_inference_steps_for_backprop))[-args.backprop_num_steps:]
                        elif args.backprop_strategy == "uniform":
                            interval = num_inference_steps_for_backprop // args.backprop_num_steps
                            random_start = random.randint(0, interval)
                            backprop_step_list = [random_start + i * interval for i in range(args.backprop_num_steps)]
                        elif args.backprop_strategy == "random":
                            backprop_step_list = random.sample(
                                range(args.backprop_random_start_step, args.backprop_random_end_step + 1), args.backprop_num_steps
                            )
                        else:
                            raise ValueError(f"Invalid backprop strategy: {args.backprop_strategy}.")
                    else:
                        backprop_step_list = args.backprop_step_list
                
                # Sanity check for backpropagation (of 14B models)
                if args.boundary_type != "full":
                    high_step_list = []
                    low_step_list = []
                    for i, t in enumerate(timesteps):
                        if t >= boundary * noise_scheduler.config.num_train_timesteps:
                            high_step_list.append(i)
                        else:
                            low_step_list.append(i)
                    if args.boundary_type == "high":
                        assert all(step in high_step_list for step in backprop_step_list), \
                            f"{backprop_step_list} is not in {high_step_list}"
                    else:
                        assert all(step in low_step_list for step in backprop_step_list), \
                            f"{backprop_step_list} is not in {low_step_list}"
                
                # Real denoising rollout: show the bar only on the local main process, otherwise
                # every FSDP rank prints its own copy (the loop is a collective, so all ranks run it).
                for i, t in enumerate(tqdm(timesteps, desc="Denoise", disable=not accelerator.is_local_main_process)):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    if hasattr(noise_scheduler, "scale_model_input"):
                        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                    
                    # Construct y (mask + masked_video_latents) for I2V/inpaint conditioning.
                    # Pure T2V (normal) and ti2v pass y=None: the T2V transformer's
                    # patch_embedding expects latent channels only, so concatenating the
                    # zero mask/video condition (mask_latents:4 + masked_video_latents:48)
                    # would inflate the channel dim (48 -> 100) and crash the forward.
                    if args.train_mode in ("ti2v", "normal"):
                        # ti2v: mask_conditions handles conditioning; normal: no condition
                        y = None
                    else:
                        mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                        masked_video_latents_input = (
                            torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                        )
                        y = torch.cat([mask_input, masked_video_latents_input], dim=1).to(accelerator.device, weight_dtype)

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    if args.train_mode == "ti2v":
                        mc_step = mask_conditions.clone()
                        if do_classifier_free_guidance:
                            mc_step = torch.cat([mc_step] * 2)
                        mask_conditions_bs = mc_step.size()[0]
                        if not mask_conditions[:, :, 0, :, :].any():
                            mvl_input = torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                            latent_model_input = (1 - mc_step) * mvl_input + mc_step * latent_model_input
                            
                            temp_ts = (mc_step[:, 0, :, ::2, ::2] * t).flatten(1)
                            timestep = torch.cat([temp_ts, temp_ts.new_ones(mask_conditions_bs, seq_len - temp_ts.size(1)) * t], dim = 1)
                        else:
                            timestep = mc_step.new_ones(mask_conditions_bs, seq_len) * t
                    else:
                        timestep = t.expand(latent_model_input.shape[0])

                    if args.boundary_type != "full":
                        if t >= boundary * noise_scheduler.config.num_train_timesteps:
                            if args.low_vram and args.boundary_type == "low":
                                torch.cuda.empty_cache()
                                high_transformer3d.to(accelerator.device)
                            if args.low_vram and args.boundary_type == "high":
                                # Unload low_transformer3d to CPU; high stays on GPU for backprop.
                                low_transformer3d.to("cpu")
                                torch.cuda.empty_cache()
                            local_transformer = high_transformer3d
                        else:
                            if args.low_vram and args.boundary_type == "low":
                                # high_transformer3d's noise_pred is detached at high t,
                                # safe to offload before backward.
                                high_transformer3d.to("cpu")
                                torch.cuda.empty_cache()
                            if args.low_vram and args.boundary_type == "high":
                                torch.cuda.empty_cache()
                                low_transformer3d.to(accelerator.device)
                            local_transformer = low_transformer3d
                    else:
                        local_transformer = transformer3d

                    # Whether to enable DRTune: https://arxiv.org/abs/2405.00760
                    if args.stop_latent_model_input_gradient:
                        latent_model_input = latent_model_input.detach()

                    # predict noise model_output
                    with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                        noise_pred = local_transformer(
                            x=latent_model_input.to(weight_dtype),
                            context=in_prompt_embeds,
                            t=timestep,
                            seq_len=seq_len,
                            y=y
                        )

                    # Optimize the denoising results only for the specified steps.
                    if i in backprop_step_list:
                        noise_pred = noise_pred
                    else:
                        noise_pred = noise_pred.detach()

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond.float() + args.guidance_scale * (noise_pred_text.float() - noise_pred_uncond.float())

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    if args.train_mode == "ti2v":
                        blend_mask = mask_conditions  # [B, 1, T_lat, H_lat, W_lat], already in latent size
                        latents = (1 - blend_mask) * masked_video_latents + blend_mask * latents

                # decode latents (tensor)
                # Since the casual VAE decoding consumes a large amount of VRAM, and we need to keep the decoding 
                # operation within the computational graph. Thus, we only decode args.num_decoded_latents 
                # latent frames to calculate the reward.
                # Uniformly sample across the full temporal range for better motion quality evaluation.
                if args.low_vram:
                    vae.to(accelerator.device)
                total_latent_frames = latents.shape[2]
                local_num_decoded_latents = min(total_latent_frames, args.num_decoded_latents)
                sampled_latent_indices = torch.linspace(0, local_num_decoded_latents - 1, local_num_decoded_latents).long().tolist()

                sampled_latents = latents[:, :, sampled_latent_indices, :, :]
                sampled_frames = vae.decode(sampled_latents.to(vae.device, vae.dtype))[0]
                sampled_frames = sampled_frames.clamp(-1, 1)
                sampled_frames = (sampled_frames / 2 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]

                # Image rewards (HPSv3/HPS/MPS/Aesthetic/PickScore) score individual frames, so the
                # frame subsampling below is safe. Video rewards (VideoAlignReward, is_image_reward=False)
                # need the full temporal sequence to evaluate motion quality; collapsing frames here would
                # degrade the video to (near-)static input and break the MQ dimension. So derive a
                # per-reward frame view, gating all subsampling on `is_image_reward`, and keep the inpaint
                # first-frame drop for every reward. Matches train_grpo_lora.py's behavior.
                def _prepare_reward_frames(_is_image_reward):
                    _frames = sampled_frames
                    if _is_image_reward and args.num_sampled_frames is not None:
                        num_frames = _frames.size(2) - 1
                        _idx = torch.linspace(0, num_frames, steps=args.num_sampled_frames).long()
                        _frames = _frames[:, :, _idx, :, :]
                    # In inpaint/ti2v mode, skip the first frame (condition image) for all rewards
                    # to prevent the reward model from incentivizing first-frame drift.
                    if args.train_mode != "normal":
                        _frames = _frames[:, :, 1:]
                    # Randomly select frames for extra stochasticity - image rewards only; video
                    # rewards must keep all decoded frames (count bounded by --num_decoded_latents).
                    if _is_image_reward and args.num_reward_frames is not None and _frames.shape[2] >= args.num_reward_frames:
                        _ridx = torch.randperm(_frames.shape[2])[:args.num_reward_frames].sort().values
                        _frames = _frames[:, :, _ridx]
                    return _frames

                # Combine each reward's differentiable loss by a weighted sum.
                loss = None
                reward = None
                for _name in reward_fn_names:
                    _fn = loss_fns[_name]
                    _frames = _prepare_reward_frames(getattr(_fn, "is_image_reward", False))
                    _l, _r = _fn(_frames, train_prompt)
                    _w = multi_reward_weights.get(_name, 1.0 / len(reward_fn_names))
                    if loss is None:
                        loss, reward = _w * _l, _w * _r
                    else:
                        loss, reward = loss + _w * _l, reward + _w * _r
                    # Log each reward's raw (unweighted) loss/reward on its own scale.
                    _fn_avg_loss = accelerator.gather(_l.repeat(args.train_batch_size)).mean()
                    _fn_avg_reward = accelerator.gather(_r.repeat(args.train_batch_size)).mean()
                    train_loss_per_fn[_name] += _fn_avg_loss.item() / args.gradient_accumulation_steps
                    train_reward_per_fn[_name] += _fn_avg_reward.item() / args.gradient_accumulation_steps

                # First-frame latent MSE loss to anchor the condition image.
                first_frame_mse = torch.tensor(0.0, device=accelerator.device)
                if args.train_mode != "normal" and args.first_frame_mse_weight > 0:
                    first_frame_mse = F.mse_loss(
                        latents[:, :, 0:1].float(),
                        masked_video_latents[:, :, 0:1].detach().float()
                    )
                    loss = loss + args.first_frame_mse_weight * first_frame_mse

                # Gather the losses and rewards across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_reward = accelerator.gather(reward.repeat(args.train_batch_size)).mean()
                avg_first_frame_mse = accelerator.gather(first_frame_mse.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                train_reward += avg_reward.item() / args.gradient_accumulation_steps
                train_first_frame_mse += avg_first_frame_mse.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    # If use_deepspeed, `total_norm` cannot be logged by accelerator.
                    if not args.use_deepspeed:
                        accelerator.log({"total_norm": total_norm}, step=global_step)
                    else:
                        if hasattr(optimizer, "optimizer") and hasattr(optimizer.optimizer, "_global_grad_norm"):
                            accelerator.log({"total_norm":  optimizer.optimizer._global_grad_norm}, step=global_step)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Save train sample after gradient update to avoid VRAM pressure
                # between full-latent VAE decode and reward CUDA kernels.
                # Decode ALL latent frames to preserve causal VAE cache correctness;
                # temporal subsampling breaks the autoregressive cache.
                if global_step % args.checkpointing_steps == 0:
                    # Non-training window: the full-latent VAE decode + encode-to-disk can take tens
                    # of seconds, so keep it out of the progress-bar rate via paused().
                    with progress_bar.paused():
                        torch.cuda.empty_cache()
                        if args.low_vram:
                            vae.to(accelerator.device)
                        with torch.no_grad():
                            all_frames = vae.decode(latents.to(vae.device, vae.dtype))[0]
                        all_frames = all_frames.clamp(-1, 1)
                        all_frames = (all_frames / 2 + 0.5).clamp(0, 1)
                        saved_file = f"sample-{global_step}-{accelerator.process_index}.mp4"
                        save_videos_grid(
                            all_frames.to(torch.float32).detach().cpu(),
                            os.path.join(args.output_dir, "train_sample", saved_file),
                            fps=16
                        )
                        del all_frames
                        if args.low_vram:
                            vae.to("cpu")
                            torch.cuda.empty_cache()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                log_dict = {"train_loss": train_loss, "train_reward": train_reward, "train_first_frame_mse": train_first_frame_mse}
                for _name in reward_fn_names:
                    log_dict[f"train_reward_{_name}"] = train_reward_per_fn[_name]
                    log_dict[f"train_loss_{_name}"] = train_loss_per_fn[_name]
                accelerator.log(log_dict, step=global_step)
                train_loss = 0.0
                train_reward = 0.0
                train_first_frame_mse = 0.0
                train_reward_per_fn = {name: 0.0 for name in reward_fn_names}
                train_loss_per_fn = {name: 0.0 for name in reward_fn_names}

                if global_step % args.checkpointing_steps == 0:
                    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        # Keep the checkpoint save out of the progress-bar rate: a long save would
                        # otherwise land in the next step's interval and show up as a slow step.
                        with progress_bar.paused():
                            if not args.save_state:
                                safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                                save_model(safetensor_save_path, accelerator.unwrap_model(network))
                                logger.info(f"Saved safetensor to {safetensor_save_path}")
                            else:
                                accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                                accelerator.save_state(accelerator_save_path)
                                logger.info(f"Saved state to {accelerator_save_path}")

                # Run on ALL processes: under FSDP the pipeline forward triggers all-gather
                # collectives, so a rank0-only call would deadlock the other ranks.
                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    with progress_bar.paused():
                        log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            transformer3d,
                            network,
                            config,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            other_transformer3d=(low_transformer3d if args.boundary_type == "high" else (high_transformer3d if args.boundary_type == "low" else None)),
                        )

            logs = {"step_loss": loss.detach().item(), "step_reward": reward.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        # Run on ALL processes (FSDP collective forward), matching the step-level call.
        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            with progress_bar.paused():
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    transformer3d,
                    network,
                    config,
                    args,
                    accelerator,
                    weight_dtype,
                    global_step,
                    other_transformer3d=(low_transformer3d if args.boundary_type == "high" else (high_transformer3d if args.boundary_type == "low" else None)),
                )

    # Close the bar before the end-of-run checkpoint / summary window so its wall-clock is not
    # attributed to the last training step.
    progress_bar.close()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if not args.save_state:
            safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
            save_model(safetensor_save_path, accelerator.unwrap_model(network))
        else:
            accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(accelerator_save_path)
            logger.info(f"Saved state to {accelerator_save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
