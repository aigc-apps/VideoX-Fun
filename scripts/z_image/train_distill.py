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
import random
import shutil
import sys
from functools import partial
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Tuple,
                    Union)

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
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from qwen_vl_utils import process_vision_info

from videox_fun.data.bucket_sampler import (ASPECT_RATIO_512,
                                            ASPECT_RATIO_RANDOM_CROP_512,
                                            ASPECT_RATIO_RANDOM_CROP_PROB,
                                            AspectRatioBatchImageVideoSampler,
                                            RandomSampler, get_closest_ratio)
from videox_fun.data.dataset_image_video import TextDataset
from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKL, AutoProcessor, AutoTokenizer,
                               CLIPImageProcessor,
                               CLIPVisionModelWithProjection, Qwen2Tokenizer,
                               Qwen3ForCausalLM, QwenImageTransformer2DModel,
                               ZImageTransformer2DModel)
from videox_fun.pipeline import ZImagePipeline
from videox_fun.utils import (DiscreteSampling, RectifiedFlow_TrigFlowWrapper,
                              calculate_dimensions,
                              convert_peft_lora_to_kohya_lora, create_network,
                              get_image_latent, get_image_to_video_latent,
                              merge_lora, sample_trigflow_timesteps,
                              save_videos_grid, unmerge_lora)

if is_wandb_available():
    import wandb

def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

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

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def encode_prompt(
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    text_encoder = None, 
    tokenizer = None,
    max_sequence_length: int = 512,
) -> List[torch.FloatTensor]:
    if isinstance(prompt, str):
        prompt = [prompt]

    for i, prompt_item in enumerate(prompt):
        messages = [
            {"role": "user", "content": prompt_item},
        ]
        prompt_item = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt[i] = prompt_item

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    embeddings_list = []

    for i in range(len(prompt_embeds)):
        embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

    return embeddings_list

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def log_validation(vae, text_encoder, tokenizer, transformer3d, args, accelerator, weight_dtype, global_step):
    try:
        is_deepspeed = type(transformer3d).__name__ == 'DeepSpeedEngine'
        if is_deepspeed:
            origin_config = transformer3d.config
            transformer3d.config = accelerator.unwrap_model(transformer3d).config
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
            logger.info("Running validation... ")
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                args.pretrained_model_name_or_path, 
                subfolder="scheduler"
            )
            pipeline = ZImagePipeline(
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
                logger.info(f"Rank {accelerator.process_index} using seed: {rank_seed}")

            for i in range(len(args.validation_prompts)):
                sample = pipeline(
                    args.validation_prompts[i], 
                    negative_prompt = "bad detailed",
                    height      = args.image_sample_size,
                    width       = args.image_sample_size,
                    generator   = generator,
                    guidance_scale = 0,
                    num_inference_steps = len(args.denoising_step_indices_list),
                ).images
                os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                image = sample[0].save(
                    os.path.join(
                        args.output_dir, 
                        f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.jpg"
                    )
                )

            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
            if not args.enable_text_encoder_in_dataloader:
                text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        if is_deepspeed:
            transformer3d.config = origin_config
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error on rank {accelerator.process_index} with info {e}")
        vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        if not args.enable_text_encoder_in_dataloader:
            text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)


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
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=512,
        help="Sample size of the token.",
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
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
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--use_fsdp", action="store_true", help="Whether or not to use fsdp."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )

    parser.add_argument(
        "--use_trigflow",
        action="store_true",
        help="whether to use trigflow in training.",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=80.0,
        help="The max value of sigma in trigflow.",
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
        default=4,
        help="The cfg scale for real score.",
    )
    parser.add_argument(
        '--denoising_step_indices_list', 
        nargs='+', 
        default=[1000, 875, 750, 625, 500, 375, 250, 125],
        help="The denoising step list.",
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
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
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
    # `from_pretrained` So Qwen3ForCausalLM and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # Get Text encoder
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
        )
        text_encoder = text_encoder.eval()

        # Get Vae
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="vae"
        ).to(weight_dtype)
        vae.eval()

    # Get Transformer
    generator_transformer3d = ZImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="transformer",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    ).to(weight_dtype)
    real_score_transformer3d = ZImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="transformer",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    ).to(weight_dtype)
    fake_score_transformer3d = ZImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="transformer",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    ).to(weight_dtype)

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
        m, u = real_score_transformer3d.load_state_dict(state_dict, strict=False)
        m, u = fake_score_transformer3d.load_state_dict(state_dict, strict=False)
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
                    load_model = ZImageTransformer2DModel.from_pretrained(
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
        except:
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
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.learning_rate / 2},
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
                    print(f"Set {name} to lr : {args.learning_rate}")
                break
        if high_lr_flag:
            continue
        for trainable_module_name in args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                in_already.append(name)
                fake_trainable_params_optim[1]['params'].append(param)
                if accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate / 2}")
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
    if args.fix_sample_size is not None and args.enable_bucket:
        args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
        args.random_hw_adapt = False

    # Get the dataset
    train_dataset = TextDataset(
        args.train_data_meta
    )

    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            print(f"worker_init_fn with {_seed + worker_id}")
            np.random.seed(_seed + worker_id)
            random.seed(_seed + worker_id)
        return _worker_init_fn
    
    if args.enable_bucket:
        def collate_fn(examples):
            new_examples = {}
            new_examples["text"] = []
            for example in examples:
                new_examples["text"].append(example["text"])

            # Encode prompts when enable_text_encoder_in_dataloader=True
            if args.enable_text_encoder_in_dataloader:
                prompt_embeds = encode_prompt(
                    batch['text'], device="cpu",
                    text_encoder=text_encoder, 
                    tokenizer=tokenizer,
                )

                new_examples['prompt_embeds'] = prompt_embeds
        
                neg_prompt_embeds = encode_prompt(
                    [""], device="cpu",
                    text_encoder=text_encoder, 
                    tokenizer=tokenizer,
                )

                new_examples['neg_prompt_embeds'] = neg_prompt_embeds

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
    fake_score_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
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
        from videox_fun.dist import set_multi_gpus_devices, shard_model
        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype, module_to_wrapper=list(real_score_transformer3d.layers))
        real_score_transformer3d = shard_fn(real_score_transformer3d)

    if fsdp_stage != 0 or zero_stage != 0:
        from videox_fun.dist import set_multi_gpus_devices, shard_model
        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype, module_to_wrapper=list(text_encoder.model.layers))
        text_encoder = shard_fn(text_encoder)

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    real_score_transformer3d.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
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

    progress_bar = tqdm(
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

    # RectifiedFlow Mode
    denoising_step_list = noise_scheduler.timesteps[args.train_sampling_steps - torch.tensor(args.denoising_step_indices_list)]
    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    # TrigFlow Mode
    scaling = RectifiedFlow_TrigFlowWrapper(1, args.train_sampling_steps)
    sample_trigflow_timesteps_D = partial(
        sample_trigflow_timesteps,
        P_mean=0.0,
        P_std=1.6
    )

    def denoise(model, xt, timestep, prompt_embeds, noise_scheduler=None, trigflow_scaling=None, multiply_c_in=True):
        """
        Unified denoise function supporting both TrigFlow and Rectified Flow
        
        Args:
            model: Diffusion model
            xt: Noised input (B, C, T, H, W) or (B, C, H, W)
            timestep: Timesteps (B,) or (B, 1)
            prompt_embeds: Text condition embeddings
            noise_scheduler: Noise scheduler (required for Rectified Flow)
            trigflow_scaling: TrigFlow scaling function (required for TrigFlow)
            multiply_c_in: Whether to multiply c_in with input (TrigFlow only)
        
        Returns:
            x0_pred: Predicted clean data
            flow_pred: Predicted velocity/flow field
        """
        use_trigflow = getattr(args, 'use_trigflow', False)
        original_dtype = xt.dtype
        device = xt.device
        
        if use_trigflow:
            # TrigFlow path
            if trigflow_scaling is None:
                raise ValueError("trigflow_scaling is required when using trigflow")
            
            ndim = xt.ndim
            trigflow_t = timestep
            
            if trigflow_t.ndim == 1:
                trigflow_t = trigflow_t.view(-1, 1)
            
            if ndim == 4:
                trigflow_t_expanded = trigflow_t.view(-1, 1, 1, 1)
            elif ndim == 5:
                trigflow_t_expanded = trigflow_t.view(-1, 1, 1, 1, 1)
            else:
                raise ValueError(f"Expected 4D or 5D input, got {ndim}D tensor.")
            
            # Get TrigFlow preconditioning coefficients
            c_skip, c_out, c_in, c_noise = trigflow_scaling(trigflow_t_expanded)
            
            # Precondition input
            if multiply_c_in:
                model_input = (xt * c_in).to(xt.dtype)
            else:
                model_input = xt.to(xt.dtype)

            timestep_normalized = c_noise.squeeze(1).squeeze(1).squeeze(1).squeeze(1)
            
            # Model inference
            model_output = model(
                x=model_input,
                cap_feats=prompt_embeds,
                t=(1000 - timestep_normalized) / 1000,
            )[0]
            
            flow_pred = -model_output.double()
            
            # EDM-style x0 reconstruction
            x0_pred = c_skip * xt + c_out * flow_pred
            
        else:
            # Rectified Flow path
            if noise_scheduler is None:
                raise ValueError("scheduler is required for Rectified Flow")
            
            xt_double = xt.double()
            timestep = timestep.to(device).double()
            
            timesteps = noise_scheduler.timesteps.to(device).double()
            sigmas = noise_scheduler.sigmas.to(device).double()
            timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
            sigma_t = sigmas[timestep_id]
            
            ndim = xt.ndim
            if ndim == 4:
                sigma_t_expanded = sigma_t.view(-1, 1, 1, 1)
            elif ndim == 5:
                sigma_t_expanded = sigma_t.view(-1, 1, 1, 1, 1)
            else:
                raise ValueError(f"Expected 4D or 5D input, got {ndim}D tensor.")
            
            model_output = model(
                x=xt,
                cap_feats=prompt_embeds,
                t=(1000 - timestep) / 1000,
            )[0]
            
            flow_pred = -model_output.double()
            
            x0_pred = xt_double - sigma_t_expanded * flow_pred
        
        return x0_pred.to(original_dtype), flow_pred

    def add_noise(x0, noise, timesteps, noise_scheduler=None):
        """
        Unified add noise function supporting both TrigFlow and Rectified Flow
        
        Args:
            x0: Clean data
            noise: Gaussian noise
            timesteps: Timesteps
        
        Returns:
            xt: Noised data
        """
        use_trigflow = getattr(args, 'use_trigflow', False)
        
        if use_trigflow:
            # TrigFlow path: xt = cos(t) * x0 + sin(t) * noise
            trigflow_t = timesteps
            ndim = x0.ndim
            
            if trigflow_t.ndim == 1:
                trigflow_t = trigflow_t.view(-1, 1)
            
            if ndim == 4:
                trigflow_t_expanded = trigflow_t.view(-1, 1, 1, 1)
            elif ndim == 5:
                trigflow_t_expanded = trigflow_t.view(-1, 1, 1, 1, 1)
            else:
                raise ValueError(f"Expected 4D or 5D input, got {ndim}D tensor.")
            
            cos_t = torch.cos(trigflow_t_expanded)
            sin_t = torch.sin(trigflow_t_expanded)
            
            return cos_t * x0 + sin_t * noise
        
        else:
            # Rectified Flow path: xt = (1 - sigma) * x0 + sigma * noise
            if noise_scheduler is None:
                raise ValueError("noise_scheduler are required for Rectified Flow")
            
            def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                timesteps = timesteps.to(accelerator.device)

                step_indices = [
                    torch.argmin(torch.abs(schedule_timesteps - t)).item()
                    for t in timesteps
                ]
                step_indices = torch.tensor(step_indices, device=accelerator.device)
                sigma = sigmas[step_indices].flatten()

                while len(sigma.shape) < n_dim:
                    sigma = sigma.unsqueeze(-1)
                return sigma
            
            sigmas = get_sigmas(timesteps, n_dim=x0.ndim, dtype=x0.dtype)
            return (1.0 - sigmas) * x0 + sigmas * noise

    for epoch in range(first_epoch, args.num_train_epochs):
        train_dmd_loss = 0.0
        train_denoising_loss = 0.0
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to("cpu")

                with torch.no_grad():
                    text = batch['text']
                    if args.fix_sample_size is not None:
                        local_sample_size = [int(x / 16) * 16 for x in args.fix_sample_size]
                    else:
                        if args.random_hw_adapt:
                            aspect_ratio_sample_size = {key : [x / 512 * args.image_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                            if rng is None:
                                aspect_ratio_key = np.random.choice(list(aspect_ratio_sample_size.keys()))
                            else:
                                aspect_ratio_key = rng.choice(list(aspect_ratio_sample_size.keys()))
                            local_sample_size = aspect_ratio_sample_size[aspect_ratio_key]
                            local_sample_size = [int(x / 16) * 16 for x in local_sample_size]
                        else:
                            local_sample_size = [args.image_sample_size, args.image_sample_size]

                    vae_scale_factor = (
                        2 ** (len(vae.config.block_out_channels) - 1)
                    )
                    target_shape = (
                        len(text),
                        vae.latent_channels, 
                        1, 
                        int(local_sample_size[0] // vae_scale_factor),
                        int(local_sample_size[1] // vae_scale_factor), 
                    )

                # wait for latents = vae.encode(pixel_values) to complete
                if vae_stream_1 is not None:
                    torch.cuda.current_stream().wait_stream(vae_stream_1)

                if args.low_vram:
                    vae.to('cpu')
                    torch.cuda.empty_cache()
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to(accelerator.device)

                if args.enable_text_encoder_in_dataloader:
                    prompt_embeds = batch['prompt_embeds'].to(dtype=latents.dtype, device=accelerator.device)
                else:
                    with torch.no_grad():
                        prompt_embeds = encode_prompt(
                            batch['text'], 
                            device=accelerator.device,
                            text_encoder=text_encoder, 
                            tokenizer=tokenizer,
                        )
                        neg_prompt_embeds = encode_prompt(
                            ["亮度过高，过曝，严重的色彩失真，低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"], 
                            device=accelerator.device,
                            text_encoder=text_encoder, 
                            tokenizer=tokenizer,
                        )

                if args.low_vram and not args.enable_text_encoder_in_dataloader:
                    text_encoder.to('cpu')
                    torch.cuda.empty_cache()
                if args.low_vram:
                    real_score_transformer3d = real_score_transformer3d.to(accelerator.device)

            if getattr(args, 'use_trigflow', False):
                # Create discrete denoising steps
                t_max = torch.arctan(torch.tensor(args.sigma_max))
                denoising_step_list = torch.linspace(t_max.item(), 0.0, args.train_sampling_steps)
                denoising_step_list = denoising_step_list[args.train_sampling_steps - torch.tensor(args.denoising_step_indices_list)]
            else:
                image_seq_len = int(target_shape[-1] // 2 * target_shape[-2] // 2)
                mu = calculate_shift(
                    image_seq_len,
                    noise_scheduler.config.get("base_image_seq_len", 256),
                    noise_scheduler.config.get("max_image_seq_len", 4096),
                    noise_scheduler.config.get("base_shift", 0.5),
                    noise_scheduler.config.get("max_shift", 1.15),
                )
                noise_scheduler.sigma_min = 0.0
                noise_scheduler.set_timesteps(args.train_sampling_steps, device=accelerator.device, mu=mu) 
                denoising_step_list = noise_scheduler.timesteps[args.train_sampling_steps - torch.tensor(args.denoising_step_indices_list)]

            # ==================== Generator Update (DMD) ====================
            with accelerator.accumulate(generator_transformer3d):
                def generate_and_sync_list(num_denoising_steps, device):
                    indices = torch.randint(low=0, high=num_denoising_steps, size=(1,), generator=torch_rng, device=device)
                    if dist.is_initialized():
                        dist.broadcast(indices, src=0)
                    return indices.tolist()

                bsz, channel, num_frames, height, width = target_shape
                
                if step % args.gen_update_interval == 0:
                    generator_noise = torch.randn(target_shape, device=accelerator.device, generator=torch_rng, dtype=weight_dtype)
                    num_denoising_steps = len(denoising_step_list)
                    final_step_index = generate_and_sync_list(num_denoising_steps, device=generator_noise.device)[0]
                    
                    # Multi-step denoising (backward simulation)
                    for index in range(num_denoising_steps):
                        is_final_step = (index == final_step_index)
                        current_t = denoising_step_list[index].expand(bsz).to(accelerator.device)
                        
                        with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                            context_manager = torch.no_grad() if not is_final_step else contextlib.nullcontext()
                                
                            with context_manager:
                                generator_pred, _ = denoise(
                                    model=generator_transformer3d,
                                    xt=generator_noise,
                                    timestep=current_t,
                                    prompt_embeds=prompt_embeds,
                                    noise_scheduler=noise_scheduler,
                                    trigflow_scaling=scaling,
                                    multiply_c_in=False if index == 0 else True,
                                )
                            
                            if is_final_step:
                                break
                            
                            # Add noise for next step
                            if index < num_denoising_steps - 1:
                                next_t = denoising_step_list[index + 1].expand(bsz).to(accelerator.device)
                                generator_noise = add_noise(
                                    generator_pred,
                                    torch.randn(generator_pred.shape, dtype=generator_pred.dtype, device=generator_pred.device, generator=torch_rng),
                                    next_t,
                                    noise_scheduler=noise_scheduler
                                )
                        
                    if getattr(args, 'use_trigflow', False):
                        # Sample timesteps for discriminator (D distribution)
                        generator_timestep = sample_trigflow_timesteps_D(bsz, device=accelerator.device)
                    else:
                        indices = idx_sampling(bsz, generator=torch_rng, device=accelerator.device).long().cpu()
                        generator_timestep = noise_scheduler.timesteps[indices].to(device=accelerator.device)

                    # Add noise to generated samples
                    generator_denoised_input = add_noise(
                        generator_pred,
                        torch.randn(generator_pred.shape, dtype=generator_pred.dtype, device=generator_pred.device, generator=torch_rng),
                        generator_timestep,
                        noise_scheduler=noise_scheduler
                    ).detach().to(accelerator.device, dtype=weight_dtype)

                    # Compute fake score
                    with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device), torch.no_grad():
                        fake_score_main_cond, _ = denoise(
                            model=fake_score_transformer3d,
                            xt=generator_denoised_input,
                            timestep=generator_timestep,
                            prompt_embeds=prompt_embeds,
                            noise_scheduler=noise_scheduler,
                            trigflow_scaling=scaling
                        )

                        if args.fake_guidance_scale != 0.0:
                            fake_score_main_uncond, _ = denoise(
                                model=fake_score_transformer3d,
                                xt=generator_denoised_input,
                                timestep=generator_timestep,
                                prompt_embeds=neg_prompt_embeds,
                                noise_scheduler=noise_scheduler,
                                trigflow_scaling=scaling
                            )
                            fake_score_main = fake_score_main_uncond + (
                                fake_score_main_cond - fake_score_main_uncond
                            ) * args.fake_guidance_scale
                        else:
                            fake_score_main = fake_score_main_cond

                        # Compute real score (teacher)
                        real_score_main_cond, _ = denoise(
                            model=real_score_transformer3d,
                            xt=generator_denoised_input,
                            timestep=generator_timestep,
                            prompt_embeds=prompt_embeds,
                            noise_scheduler=noise_scheduler,
                            trigflow_scaling=scaling
                        )

                        if args.real_guidance_scale != 0.0:
                            real_score_main_uncond, _ = denoise(
                                model=real_score_transformer3d,
                                xt=generator_denoised_input,
                                timestep=generator_timestep,
                                prompt_embeds=neg_prompt_embeds,
                                noise_scheduler=noise_scheduler,
                                trigflow_scaling=scaling
                            )

                            real_score_main = real_score_main_uncond + (
                                real_score_main_cond - real_score_main_uncond
                            ) * args.real_guidance_scale
                        else:
                            real_score_main = real_score_main_cond

                    # DMD loss
                    fake_to_real_grad = fake_score_main - real_score_main
                    generator_to_real_norm = generator_pred - real_score_main
                    normalizer = torch.abs(generator_to_real_norm).mean(dim=[1, 2, 3, 4], keepdim=True).clip(min=1e-5)
                    fake_to_real_grad = fake_to_real_grad / normalizer
                    fake_to_real_grad = torch.nan_to_num(fake_to_real_grad)

                    dmd_loss = 0.5 * F.mse_loss(
                        generator_pred.double(),
                        (generator_pred.double() - fake_to_real_grad.double()).detach(),
                        reduction="mean"
                    )
                    avg_dmd_loss = accelerator.gather(dmd_loss.repeat(args.train_batch_size)).mean()
                    train_dmd_loss += avg_dmd_loss.item() / args.gradient_accumulation_steps

                    if args.low_vram:
                        real_score_transformer3d = real_score_transformer3d.to("cpu")
                        fake_score_transformer3d = fake_score_transformer3d.to("cpu")
                        torch.cuda.empty_cache()

                    accelerator.backward(dmd_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    if args.low_vram:
                        fake_score_transformer3d = fake_score_transformer3d.to(accelerator.device)
                        torch.cuda.empty_cache()

            with accelerator_fake_score_transformer3d.accumulate(fake_score_transformer3d):
                # --- Fake Critic Denoising Loss ---
                with torch.no_grad():
                    fake_score_critic_noise = torch.randn(target_shape, device=accelerator.device, generator=torch_rng, dtype=weight_dtype)
                    num_denoising_steps = len(denoising_step_list)
                    final_step_index = generate_and_sync_list(num_denoising_steps, device=fake_score_critic_noise.device)[0]

                    for index in range(num_denoising_steps):
                        is_final_step = (index == final_step_index)
                        current_t = denoising_step_list[index].expand(bsz).to(accelerator.device)
                    
                        with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                            fake_score_denoised_pred, _ = denoise(
                                model=generator_transformer3d,
                                xt=fake_score_critic_noise,
                                timestep=current_t,
                                prompt_embeds=prompt_embeds,
                                noise_scheduler=noise_scheduler,
                                trigflow_scaling=scaling,
                                multiply_c_in=False if index == 0 else True,
                            )
                            
                            if is_final_step:
                                break

                            if index < num_denoising_steps - 1:
                                next_t = denoising_step_list[index + 1].expand(bsz).to(accelerator.device)
                                fake_score_critic_noise = add_noise(
                                    fake_score_denoised_pred,
                                    torch.randn(fake_score_denoised_pred.shape, dtype=fake_score_denoised_pred.dtype, device=fake_score_denoised_pred.device, generator=torch_rng),
                                    next_t,
                                    noise_scheduler=noise_scheduler
                                )

                # Sample timesteps for critic
                if getattr(args, 'use_trigflow', False):
                    # Sample timesteps for discriminator (D distribution)
                    critic_timestep = sample_trigflow_timesteps_D(bsz, device=accelerator.device)
                else:
                    indices = idx_sampling(bsz, generator=torch_rng, device=accelerator.device).long().cpu()
                    critic_timestep = noise_scheduler.timesteps[indices].to(device=accelerator.device)
                critic_noise = torch.randn(fake_score_denoised_pred.shape, dtype=fake_score_denoised_pred.dtype, device=fake_score_denoised_pred.device, generator=torch_rng)

                fake_score_denoised_input = add_noise(
                    fake_score_denoised_pred,
                    critic_noise,
                    critic_timestep,
                    noise_scheduler=noise_scheduler
                )

                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    fake_score_pred, _ = denoise(
                        model=fake_score_transformer3d,
                        xt=fake_score_denoised_input,
                        timestep=critic_timestep,
                        prompt_embeds=prompt_embeds,
                        noise_scheduler=noise_scheduler,
                        trigflow_scaling=scaling
                    )

                def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
                    noise_pred = noise_pred.float()
                    target = target.float()
                    diff = noise_pred - target
                    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
                    mask = (diff.abs() <= threshold).float()
                    masked_loss = mse_loss * mask
                    if weighting is not None:
                        masked_loss = masked_loss * weighting
                    final_loss = masked_loss.mean()
                    return final_loss

                # Compute weighting based on sin(t) (following rCM)
                if getattr(args, 'use_trigflow', False):
                    ndim = fake_score_denoised_input.ndim
                    if critic_timestep.ndim == 1:
                        critic_timestep_view = critic_timestep.view(-1, 1)
                    else:
                        critic_timestep_view = critic_timestep
                    
                    if ndim == 4:
                        critic_t_expanded = critic_timestep_view.view(-1, 1, 1, 1)
                    elif ndim == 5:
                        critic_t_expanded = critic_timestep_view.view(-1, 1, 1, 1, 1)
                    
                    sin_t = torch.sin(critic_t_expanded)
                    weighting = 1.0 / (sin_t ** 2 + 1e-8)
                else:
                    weighting = None
                
                denoising_loss = custom_mse_loss(
                    fake_score_pred, 
                    fake_score_denoised_pred,
                    weighting=weighting
                )
        
                avg_denoising_loss = accelerator_fake_score_transformer3d.gather(denoising_loss.repeat(args.train_batch_size)).mean()
                train_denoising_loss += avg_denoising_loss.item() / args.gradient_accumulation_steps
            
                if args.low_vram:
                    generator_transformer3d = generator_transformer3d.to("cpu")
                    torch.cuda.empty_cache()

                accelerator_fake_score_transformer3d.backward(denoising_loss)
                if accelerator_fake_score_transformer3d.sync_gradients:
                    accelerator_fake_score_transformer3d.clip_grad_norm_(fake_trainable_params, args.max_grad_norm)
                critic_optimizer.step()
                fake_score_lr_scheduler.step()
                critic_optimizer.zero_grad()

                if args.low_vram:
                    fake_score_transformer3d = fake_score_transformer3d.to(accelerator.device)
                    generator_transformer3d = generator_transformer3d.to(accelerator.device)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_denoising_loss": train_denoising_loss, "train_dmd_loss": train_dmd_loss}, step=global_step)
                train_dmd_loss = 0.0
                train_denoising_loss = 0.0

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
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        fake_score_save_path = os.path.join(save_path, "fake_score")
                        accelerator.save_state(save_path)
                        accelerator_fake_score_transformer3d.save_state(fake_score_save_path)
                        logger.info(f"Saved state to {save_path}")

                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    log_validation(
                        vae,
                        text_encoder,
                        tokenizer,
                        generator_transformer3d,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )

            logs = {"denoising_loss": denoising_loss.detach().item(), "dmd_loss": dmd_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            log_validation(
                vae,
                text_encoder,
                tokenizer,
                generator_transformer3d,
                args,
                accelerator,
                weight_dtype,
                global_step,
            )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()