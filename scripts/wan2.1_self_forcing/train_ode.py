# Based on https://github.com/guandeh17/Self-Forcing
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
import shutil
import sys
import time

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
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
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.data import ImageVideoSafetensorsDataset, RandomSampler
from videox_fun.models import (AutoencoderKLWan, WanT5EncoderModel,
                               WanTransformer3DModel_SelfForcing)
from videox_fun.pipeline import WanSelfForcingPipeline
from videox_fun.utils.utils import save_videos_grid

check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


# ============================================================================
# Utilities
# ============================================================================

def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def log_validation(transformer3d, args, config, accelerator, weight_dtype, global_step,
                   fsdp_stage=0, zero_stage=0):
    """Run validation. Loads tokenizer, text_encoder and vae lazily (only when called) and frees them after."""
    text_encoder = None
    vae = None
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
            logger.info("Running validation... ")

            # --- Lazy load tokenizer, text_encoder and vae just for this validation pass ---
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
            )

            logger.info("Loading text encoder for validation...")
            if zero_stage == 3:
                ctx = accelerate.state.AcceleratorState().deepspeed_plugin.zero3_init_context_manager(enable=False)
            else:
                from contextlib import nullcontext
                ctx = nullcontext()
            with ctx:
                text_encoder = WanT5EncoderModel.from_pretrained(
                    os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
                    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
                    low_cpu_mem_usage=True,
                    torch_dtype=weight_dtype,
                )
            text_encoder = text_encoder.eval()
            text_encoder.requires_grad_(False)

            logger.info("Loading VAE for validation...")
            vae = AutoencoderKLWan.from_pretrained(
                os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
                additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
            ).to(weight_dtype)
            vae.eval()
            vae.requires_grad_(False)

            # Apply FSDP sharding to text_encoder if FSDP/DeepSpeed is enabled
            if fsdp_stage != 0 or zero_stage != 0:
                from functools import partial

                from videox_fun.dist import shard_model
                shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype)
                text_encoder = shard_fn(text_encoder)

            scheduler_kwargs = OmegaConf.to_container(config['scheduler_kwargs'])
            scheduler_kwargs['shift'] = args.shift
            scheduler = FlowMatchEulerDiscreteScheduler(
                **filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
            )
            pipeline = WanSelfForcingPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=accelerator.unwrap_model(transformer3d) if type(transformer3d).__name__ == 'DistributedDataParallel' else transformer3d,
                scheduler=scheduler,
            )
            pipeline = pipeline.to(accelerator.device)

            if args.seed is None:
                generator = None
            else:
                rank_seed = args.seed + accelerator.process_index
                generator = torch.Generator(device=accelerator.device).manual_seed(rank_seed)

            for i in range(len(args.validation_prompts)):
                if args.fix_sample_size is not None:
                    height, width = args.fix_sample_size
                else:
                    height, width = args.video_sample_size, args.video_sample_size
                sample = pipeline(
                    args.validation_prompts[i],
                    num_frames=args.video_sample_n_frames,
                    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    height=height,
                    width=width,
                    generator=generator,
                    guidance_scale=1.0,
                    num_inference_steps=len(args.denoising_step_indices_list),
                    shift=args.shift,
                    num_frame_per_block=args.num_frame_per_block,
                    independent_first_frame=args.independent_first_frame,
                    context_noise=args.context_noise,
                    stochastic_sampling=True,
                ).videos
                os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                save_videos_grid(
                    sample,
                    os.path.join(args.output_dir, f"sample/sample-{global_step}-rank{accelerator.process_index}-{i}.mp4")
                )

            # --- Free text_encoder, vae and pipeline to release GPU memory ---
            del pipeline
            del text_encoder
            del vae
            text_encoder = None
            vae = None
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as e:
        if text_encoder is not None:
            del text_encoder
        if vae is not None:
            del vae
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Eval error on rank {accelerator.process_index} with info {e}")


def get_timestep_for_ode(
    min_timestep, max_timestep, batch_size, num_frames,
    num_frame_per_block, independent_first_frame, device,
    generator=None,
):
    """
    Generate random timestep indices per frame/block.
    Same timestep within each block (matches Self-Forcing ODE regression).
    Returns: [batch_size, num_frames] tensor of indices.
    """
    timestep = torch.randint(
        min_timestep, max_timestep,
        [batch_size, num_frames],
        device=device, dtype=torch.long,
        generator=generator,
    )
    if independent_first_frame:
        timestep_from_second = timestep[:, 1:]
        timestep_from_second = timestep_from_second.reshape(
            timestep_from_second.shape[0], -1, num_frame_per_block)
        timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
        timestep_from_second = timestep_from_second.reshape(
            timestep_from_second.shape[0], -1)
        timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
    else:
        timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
        timestep[:, :, 1:] = timestep[:, :, 0:1]
        timestep = timestep.reshape(timestep.shape[0], -1)
    return timestep


def initialize_kv_cache_for_training(batch_size, num_frames, frame_seq_length,
                                     num_layers, num_heads, head_dim, dtype, device):
    """Initialize KV cache for block-by-block training (mirrors train_distill)."""
    kv_cache_size = num_frames * frame_seq_length
    kv_cache = []
    for _ in range(num_layers):
        kv_cache.append({
            "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim],
                              dtype=dtype, device=device),
            "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim],
                              dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        })
    return kv_cache


def initialize_crossattn_cache_for_training(batch_size, text_len, num_layers,
                                             num_heads, head_dim, dtype, device):
    """Initialize cross-attention cache for block-by-block training."""
    crossattn_cache = []
    for _ in range(num_layers):
        crossattn_cache.append({
            "k": torch.zeros([batch_size, text_len, num_heads, head_dim],
                              dtype=dtype, device=device),
            "v": torch.zeros([batch_size, text_len, num_heads, head_dim],
                              dtype=dtype, device=device),
            "is_init": False,
        })
    return crossattn_cache


# ============================================================================
# Args
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ODE Regression Training for Self-Forcing")
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
        help="The config of the model in training.",
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
        required=True,
        help="Path to JSON annotation file listing ODE trajectory safetensors files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_ode_regression",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory. Will default to *output_dir/logs*.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
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
        default=2e-6,
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
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=3e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-10, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
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
            ' (default), `"wandb"` and `"comet_ml"`.'
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
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
        help="Max number of checkpoints to store.",
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
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help="A set of prompts evaluated every `--validation_steps` and logged to `--report_to`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=500,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=640,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=81,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--fix_sample_size",
        nargs=2, type=int, default=None,
        help="Fix Sample size [height, width] when using bucket and collate_fn.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="If you want to load the weight from other transformers, input its path.",
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Total number of scheduler timesteps for sampling.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank.")

    # Self-Forcing specific
    parser.add_argument(
        '--denoising_step_indices_list',
        nargs='+', type=int,
        default=[1000, 750, 500, 250],
        help="The denoising step list.",
    )
    parser.add_argument(
        "--num_frame_per_block",
        type=int,
        default=3,
        help="Number of frames per block for Self-Forcing causal training.",
    )
    parser.add_argument(
        "--independent_first_frame",
        action="store_true",
        help="Whether first frame is independent ([1, N, N, ...] pattern).",
    )
    parser.add_argument(
        "--context_noise",
        type=int,
        default=0,
        help="Context noise level for KV cache update (matches training config).",
    )
    parser.add_argument(
        '--trainable_modules',
        nargs='+',
        default=['.'],
        help='Enter a list of trainable modules.',
    )
    parser.add_argument(
        '--trainable_modules_low_learning_rate',
        nargs='+',
        default=[],
        help='Enter a list of trainable modules with lower learning rate.',
    )
    parser.add_argument(
        "--save_state", action="store_true", help="Whether to save full accelerator state on checkpoint."
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=8.0,
        help="Shift value for FlowMatchEulerDiscreteScheduler. Default: 8.0 (matches ODE data generation).",
    )
    parser.add_argument(
        "--use_kv_cache_training",
        action="store_true",
        help=(
            "If set, run block-by-block KV cache training that fully matches the "
            "pipeline_wan_self_forcing inference behavior. Otherwise fall back to "
            "the default one-shot causal-mask ODE regression (kept as baseline)."
        ),
    )
    parser.add_argument(
        "--prob_full_zero_start",
        type=float,
        default=0.0,
        help=(
            "Probability (per-sample) of forcing ALL frames in ALL blocks to use "
            "timestep index=0 (pure-noise start). Bridges the train-inference gap "
            "so the model also sees the real autoregressive rollout where every "
            "block starts from fresh noise. 0.0 disables (default)."
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main():
    args = parse_args()

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

    # Load scheduler.
    scheduler_kwargs = OmegaConf.to_container(config['scheduler_kwargs'])
    scheduler_kwargs['shift'] = args.shift
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
    )

    # NOTE: tokenizer, text_encoder and vae are NOT loaded here. They are lazily loaded inside
    # `log_validation` only when validation actually runs, then freed afterwards.

    # Get causal Transformer (generator)
    transformer3d = WanTransformer3DModel_SelfForcing.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
    ).to(weight_dtype)

    # Set transformer3d to non-trainable initially (trainable_modules will toggle below)
    transformer3d.requires_grad_(False)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        state_dict = state_dict["generator_ema"] if "generator_ema" in state_dict else state_dict
        state_dict = state_dict["generator"] if "generator" in state_dict else state_dict
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict.items()}
        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"Loaded transformer_path. missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    # Set trainable modules
    # A good trainable modules is showed below now.
    # For full finetune: trainable_modules = ['.']
    transformer3d.train()
    for name, param in transformer3d.named_parameters():
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
                        input_dir, subfolder="transformer"
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

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

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

    trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
    trainable_params_optim = [
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.learning_rate / 2},
    ]
    in_already = []
    for name, param in transformer3d.named_parameters():
        if not param.requires_grad:
            continue
        if name in in_already:
            continue
        high_lr = False
        for m in args.trainable_modules:
            if m in name:
                in_already.append(name)
                high_lr = True
                trainable_params_optim[0]['params'].append(param)
                break
        if high_lr:
            continue
        for m in args.trainable_modules_low_learning_rate:
            if m in name:
                in_already.append(name)
                trainable_params_optim[1]['params'].append(param)
                break

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
    train_dataset = ImageVideoSafetensorsDataset(
        args.train_data_meta, data_root=args.train_data_dir
    )
    # DataLoaders creation:
    from torch.utils.data import BatchSampler, Dataset, Sampler
    batch_sampler_generator = torch.Generator().manual_seed(args.seed)
    batch_sampler = BatchSampler(
        RandomSampler(train_dataset, generator=batch_sampler_generator), 
        batch_size=args.train_batch_size, drop_last=True
    )        

    def ode_safetensors_collate_fn(examples):
        """Collate safetensors-loaded ODE samples into a batch.

        Each sample is a dict with keys:
        - 'latents': [S, C, F, H, W]
        - 'prompt_embeds': [L, D]
        - 'prompt_attention_mask': [L]

        The default torch collate fails when, across samples, the same key has
        slightly different dtypes/lengths (e.g. attention_mask saved as bool/int
        vs long, or prompt_embeds with different seq lengths). This custom
        collate normalizes dtypes and pads variable-length text fields so
        `torch.stack` always succeeds.
        """
        out = {}

        # ---- latents: assume identical shape across samples (fixed by pipeline) ----
        latents = [ex["latents"] for ex in examples]
        target_latent_dtype = latents[0].dtype
        latents = [t.to(target_latent_dtype) for t in latents]
        out["latents"] = torch.stack(latents, dim=0)

        # ---- prompt_embeds: pad along seq dim, unify dtype ----
        embeds = [ex["prompt_embeds"] for ex in examples]
        embed_dtype = embeds[0].dtype
        max_len = max(e.shape[0] for e in embeds)
        padded_embeds = []
        for e in embeds:
            e = e.to(embed_dtype)
            if e.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - e.shape[0], *e.shape[1:], dtype=embed_dtype
                )
                e = torch.cat([e, pad], dim=0)
            padded_embeds.append(e)
        out["prompt_embeds"] = torch.stack(padded_embeds, dim=0)

        # ---- prompt_attention_mask: pad along seq dim, force long dtype ----
        masks = [ex["prompt_attention_mask"].long() for ex in examples]
        max_len = max(m.shape[0] for m in masks)
        padded_masks = []
        for m in masks:
            if m.shape[0] < max_len:
                pad = torch.zeros(max_len - m.shape[0], dtype=torch.long)
                m = torch.cat([m, pad], dim=0)
            padded_masks.append(m)
        out["prompt_attention_mask"] = torch.stack(padded_masks, dim=0)

        return out

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler, 
        persistent_workers=True if args.dataloader_num_workers != 0 else False,
        num_workers=args.dataloader_num_workers,
        collate_fn=ode_safetensors_collate_fn,
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
    transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, train_dataloader, lr_scheduler
    )

    # Compute denoising_step_list from scheduler
    noise_scheduler.set_timesteps(args.train_sampling_steps, device=accelerator.device)
    denoising_step_list = noise_scheduler.timesteps[
        args.train_sampling_steps - torch.tensor(args.denoising_step_indices_list)
    ]
    # Training denoising step list: append 0 (clean) for train-inference context alignment.
    # index=4 frames use clean latent as input but are excluded from loss via mask=(timestep!=0).
    # They serve as clean context for later blocks via causal attention.
    train_denoising_step_list = denoising_step_list
    if 0 not in denoising_step_list.tolist():
        train_denoising_step_list = torch.cat([
            denoising_step_list, torch.tensor([0], device=denoising_step_list.device)
        ])
    num_denoising_steps = len(train_denoising_step_list)
    if accelerator.is_main_process:
        print(f"Denoising step list (inference): {denoising_step_list.tolist()}")
        print(f"Denoising step list (training):  {train_denoising_step_list.tolist()}")
        print(f"num_denoising_steps (includes clean): {num_denoising_steps}")
        print(f"Dataset size: {len(train_dataset)}")

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
            accelerator.load_state(os.path.join(args.output_dir, path))
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # ====================================================================
    # Training loop
    # ====================================================================
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                # --- Read preprocessed prompt embeddings ---
                prompt_embeds_raw = batch["prompt_embeds"].to(device=accelerator.device, dtype=weight_dtype)
                encoder_attention_mask = batch["prompt_attention_mask"].to(device=accelerator.device)
                bsz = prompt_embeds_raw.shape[0]

                # Truncate to actual token length (same as preprocess training)
                seq_lens = encoder_attention_mask.gt(0).sum(dim=1).long()
                prompt_embeds = [prompt_embeds_raw[i, :seq_lens[i].item()] for i in range(bsz)]

                # --- Prepare ODE regression inputs ---
                # ode_latent: [B, num_denoising_steps, C, F, H, W]
                ode_latent = batch["latents"].to(device=accelerator.device, dtype=weight_dtype)

                # Target: clean endpoint (last timestep)
                target_latent = ode_latent[:, -1]  # [B, C, F, H, W]
                num_frames = target_latent.shape[2]
                C_dim, F_dim, H_dim, W_dim = (
                    ode_latent.shape[2], ode_latent.shape[3],
                    ode_latent.shape[4], ode_latent.shape[5],
                )

                patch_h, patch_w = accelerator.unwrap_model(transformer3d).config.patch_size[1:]
                frame_seqlen = (H_dim * W_dim) // (patch_h * patch_w)
                seq_len = frame_seqlen * num_frames

            with accelerator.accumulate(transformer3d):
                with torch.cuda.amp.autocast(dtype=weight_dtype):
                    if args.use_kv_cache_training:
                        # ============================================================
                        # Block-by-block KV cache training (autoregressive, single-step x0)
                        # Starting timestep is randomly sampled per block — same as the
                        # non-KV-cache (baseline) branch. Each block performs ONE forward
                        # to predict x0; KV cache is then refreshed with pred_block +
                        # context_noise to keep the autoregressive rollout intact.
                        # ============================================================
                        # 1) Block split (mirrors pipeline_wan_self_forcing)
                        if not args.independent_first_frame:
                            assert num_frames % args.num_frame_per_block == 0
                            num_blocks_split = num_frames // args.num_frame_per_block
                            all_num_frames = [args.num_frame_per_block] * num_blocks_split
                        else:
                            assert (num_frames - 1) % args.num_frame_per_block == 0
                            num_blocks_split = (num_frames - 1) // args.num_frame_per_block
                            all_num_frames = [1] + [args.num_frame_per_block] * num_blocks_split

                        # 2) Random timestep index per frame/block (same as baseline branch)
                        index = get_timestep_for_ode(
                            0, num_denoising_steps, bsz, num_frames,
                            args.num_frame_per_block, args.independent_first_frame,
                            accelerator.device,
                            generator=torch_rng,
                        )  # [B, F]
                        # Optional: force per-sample full-zero start to cover the
                        # real inference rollout (all blocks starting from pure noise).
                        if args.prob_full_zero_start > 0.0:
                            zero_mask = (
                                torch.rand(bsz, device=accelerator.device, generator=torch_rng)
                                < args.prob_full_zero_start
                            )
                            if zero_mask.any():
                                index[zero_mask] = 0
                        gather_index = index.reshape(bsz, 1, 1, num_frames, 1, 1).expand(
                            -1, -1, C_dim, -1, H_dim, W_dim
                        )
                        noisy_input_full = torch.gather(ode_latent, dim=1, index=gather_index).squeeze(1)
                        timestep_full = train_denoising_step_list[index]  # [B, F]

                        # 3) Initialize KV / cross-attention cache
                        cfg = accelerator.unwrap_model(transformer3d).config
                        num_layers_t = cfg.num_layers
                        num_heads_t = cfg.num_heads
                        head_dim_t = cfg.dim // num_heads_t
                        text_len = 512  # T5 sequence length
                        kv_cache = initialize_kv_cache_for_training(
                            batch_size=bsz,
                            num_frames=num_frames,
                            frame_seq_length=frame_seqlen,
                            num_layers=num_layers_t,
                            num_heads=num_heads_t,
                            head_dim=head_dim_t,
                            dtype=weight_dtype,
                            device=accelerator.device,
                        )
                        crossattn_cache = initialize_crossattn_cache_for_training(
                            batch_size=bsz,
                            text_len=text_len,
                            num_layers=num_layers_t,
                            num_heads=num_heads_t,
                            head_dim=head_dim_t,
                            dtype=weight_dtype,
                            device=accelerator.device,
                        )

                        # 4) Sigma / timestep lookup tables (per-frame sigma)
                        sigmas_full = noise_scheduler.sigmas.to(
                            device=accelerator.device, dtype=torch.float64
                        )
                        schedule_timesteps_full = noise_scheduler.timesteps.to(accelerator.device)

                        current_start_frame = 0
                        total_pred = torch.zeros_like(target_latent)
                        full_seq_len = frame_seqlen * num_frames

                        # 5) Block-by-block rollout — single-step x0 prediction per block
                        for block_idx, current_num_frames in enumerate(all_num_frames):
                            start_idx = current_start_frame
                            end_idx = current_start_frame + current_num_frames

                            noisy_input = noisy_input_full[:, :, start_idx:end_idx]
                            timestep_block = timestep_full[:, start_idx:end_idx].to(torch.int64)

                            flow_pred = transformer3d(
                                x=[noisy_input[i] for i in range(bsz)],
                                context=prompt_embeds,
                                t=timestep_block,
                                seq_len=full_seq_len,
                                kv_cache=kv_cache,
                                crossattn_cache=crossattn_cache,
                                current_start=current_start_frame * frame_seqlen,
                                cache_start=None,
                            )
                            if isinstance(flow_pred, list):
                                flow_pred = torch.stack(flow_pred, dim=0)

                            # Per-frame sigma -> x0 = xt - sigma * flow_pred
                            step_indices_block = torch.argmin(
                                (schedule_timesteps_full.unsqueeze(0)
                                 - timestep_block.reshape(-1).unsqueeze(1)).abs(), dim=1
                            )
                            sigma_block = sigmas_full[step_indices_block].to(weight_dtype)
                            # timestep=0 (clean context) must use sigma=0 exactly.
                            sigma_block[timestep_block.reshape(-1) == 0] = 0.0
                            sigma_block = sigma_block.reshape(bsz, 1, current_num_frames, 1, 1)
                            pred_block = noisy_input - sigma_block * flow_pred

                            total_pred[:, :, start_idx:end_idx] = pred_block

                            # 6) Update KV cache with student's pred_block + context_noise
                            #    (matches pipeline_wan_self_forcing L802-L839)
                            if block_idx < len(all_num_frames) - 1:
                                ctx_t = torch.full(
                                    [bsz, current_num_frames], args.context_noise,
                                    device=accelerator.device, dtype=torch.int64,
                                )
                                with torch.no_grad():
                                    transformer3d(
                                        x=[pred_block[i] for i in range(bsz)],
                                        context=prompt_embeds,
                                        t=ctx_t,
                                        seq_len=full_seq_len,
                                        kv_cache=kv_cache,
                                        crossattn_cache=crossattn_cache,
                                        current_start=current_start_frame * frame_seqlen,
                                        cache_start=None,
                                    )

                            current_start_frame += current_num_frames

                        # 7) ODE-endpoint MSE loss (mask out clean timestep=0 frames)
                        mask = (timestep_full != 0).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                        mask = mask.expand_as(target_latent).float()
                        if mask.sum() > 0:
                            loss = F.mse_loss(total_pred * mask, target_latent * mask, reduction="sum") / mask.sum()
                        else:
                            loss = F.mse_loss(total_pred, target_latent)
                    else:
                        # --- Baseline (one-shot causal-mask) preparation ---
                        # Random timestep index per frame/block
                        index = get_timestep_for_ode(
                            0, num_denoising_steps, bsz, num_frames,
                            args.num_frame_per_block, args.independent_first_frame,
                            accelerator.device,
                            generator=torch_rng,
                        )  # [B, F]
                        # Optional: force per-sample full-zero start to cover the
                        # real inference rollout (all blocks starting from pure noise).
                        if args.prob_full_zero_start > 0.0:
                            zero_mask = (
                                torch.rand(bsz, device=accelerator.device, generator=torch_rng)
                                < args.prob_full_zero_start
                            )
                            if zero_mask.any():
                                index[zero_mask] = 0

                        # Gather noisy input from ODE trajectory
                        gather_index = index.reshape(bsz, 1, 1, num_frames, 1, 1).expand(
                            -1, -1, C_dim, -1, H_dim, W_dim
                        )
                        noisy_input = torch.gather(ode_latent, dim=1, index=gather_index).squeeze(1)

                        # Compute actual timestep values: [B, F]
                        timestep = train_denoising_step_list[index]  # [B, F]

                        # Build causal block mask
                        accelerator.unwrap_model(transformer3d).create_block_mask_for_training(
                            num_frames=num_frames,
                            frame_seqlen=frame_seqlen,
                            num_frame_per_block=args.num_frame_per_block,
                            independent_first_frame=args.independent_first_frame,
                            device=accelerator.device
                        )

                        # Convert to list format for transformer
                        noisy_input_list = [noisy_input[i] for i in range(bsz)]

                        # ============================================================
                        # Baseline: one-shot causal-mask ODE regression
                        # ============================================================
                        flow_pred = transformer3d(
                            x=noisy_input_list,
                            context=prompt_embeds,
                            t=timestep,
                            seq_len=seq_len,
                        )

                        # Convert flow prediction to x0 prediction (per-frame).
                        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=torch.float64)
                        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                        step_indices = torch.argmin(
                            (schedule_timesteps.unsqueeze(0) - timestep.reshape(-1).unsqueeze(1)).abs(), dim=1
                        )
                        sigma = sigmas[step_indices].to(weight_dtype)
                        # Fix: timestep=0 (clean context frames) should have sigma=0 exactly.
                        sigma[timestep.reshape(-1) == 0] = 0.0
                        sigma = sigma.reshape(bsz, 1, num_frames, 1, 1)

                        pred_x0 = noisy_input - sigma * flow_pred

                        # MSE loss (mask t=0 frames)
                        mask = (timestep != 0).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                        mask = mask.expand_as(target_latent).float()

                        if mask.sum() > 0:
                            loss = F.mse_loss(pred_x0 * mask, target_latent * mask, reduction="sum") / mask.sum()
                        else:
                            loss = F.mse_loss(pred_x0, target_latent)

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging and checkpointing
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process or fsdp_stage == 3 or zero_stage == 3:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[:num_to_remove]

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
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    log_validation(
                        transformer3d,
                        args, config, accelerator, weight_dtype, global_step,
                        fsdp_stage=fsdp_stage, zero_stage=zero_stage,
                    )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            log_validation(
                transformer3d,
                args, config, accelerator, weight_dtype, global_step,
                fsdp_stage=fsdp_stage, zero_stage=zero_stage,
            )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process or fsdp_stage == 3 or zero_stage == 3:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
