"""Causal-Forcing Stage 2 (Causal Consistency Distillation) training, modified from
scripts/wan2.1_self_forcing/train_distill.py and Causal-Forcing
(https://github.com/thu-ml/Causal-Forcing).
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
import shutil
import sys

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
from diffusers.utils import check_min_version, is_wandb_available
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

from videox_fun.data import (ASPECT_RATIO_512, ASPECT_RATIO_RANDOM_CROP_512,
                             ASPECT_RATIO_RANDOM_CROP_PROB,
                             AspectRatioBatchImageVideoSampler,
                             ImageVideoDataset, RandomSampler,
                             get_closest_ratio)
from videox_fun.models import (AutoencoderKLWan, WanT5EncoderModel,
                               WanTransformer3DModel_SelfForcing)
from videox_fun.pipeline import WanSelfForcingPipeline
from videox_fun.utils.utils import save_videos_grid

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
            logger.info("Running validation... ")
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
                logger.info(f"Rank {accelerator.process_index} using seed: {rank_seed}")

            for i in range(len(args.validation_prompts)):
                if args.fix_sample_size is not None:
                    height, width = args.fix_sample_size
                else:
                    height, width = args.video_sample_size, args.video_sample_size
                sample = pipeline(
                    args.validation_prompts[i],
                    num_frames = args.video_sample_n_frames,
                    negative_prompt = args.negative_prompt,
                    height      = height,
                    width       = width,
                    generator   = generator,
                    guidance_scale = args.validation_guidance_scale,
                    num_inference_steps = args.validation_num_inference_steps,
                    shift       = args.shift,
                    num_frame_per_block = args.num_frame_per_block,
                    independent_first_frame = args.independent_first_frame,
                    context_noise = 0,
                    stochastic_sampling = False,
                ).videos
                os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                save_videos_grid(
                    sample,
                    os.path.join(
                        args.output_dir,
                        f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.mp4"
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
    parser = argparse.ArgumentParser(description="Causal-Forcing Stage 2 (Option B): Causal Consistency Distillation Initialization for Wan2.1.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        help=("The negative prompt used for validation generation."),
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
        "--validation_guidance_scale",
        type=float,
        default=1,
        help="CFG scale used when sampling validation videos.",
    )
    parser.add_argument(
        "--validation_num_inference_steps",
        type=int,
        default=4,
        help=(
            "Number of denoising steps used for validation. AR diffusion validation runs a"
            " full multi-step rollout, so a larger value (e.g. 50) is recommended."
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
        "--training_with_video_token_length", action="store_true", help="The training stage of the model in training.",
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Total number of scheduler timesteps for sampling.",
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
        "--num_frame_per_block",
        type=int,
        default=3,
        help="Number of latent frames per causal block. 3 = chunk-wise, 1 = frame-wise."
    )
    parser.add_argument(
        "--independent_first_frame",
        action="store_true",
        help="Whether first frame is independent ([1, N, N, ...] pattern, useful for I2V)."
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=5.0,
        help="Shift value for FlowMatchEulerDiscreteScheduler. Causal-Forcing uses 5.0 by default."
    )
    parser.add_argument(
        "--discrete_cd_N",
        type=int,
        default=48,
        help="Number of discrete timesteps used for the consistency schedule (`discrete_cd_N` in Causal-Forcing). Default: 48."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale applied to the frozen teacher when generating the one-step ODE target. Default: 3.0."
    )
    parser.add_argument(
        "--ema_weight",
        type=float,
        default=0.99,
        help="EMA decay for the consistency-target generator copy. Set <=0 to disable EMA and use the live generator as the target."
    )
    parser.add_argument(
        "--ema_start_step",
        type=int,
        default=200,
        help="Number of optimizer steps to wait before EMA tracking starts. Before this point the EMA copy mirrors the live generator."
    )
    parser.add_argument(
        "--teacher_transformer_path",
        type=str,
        default=None,
        help="Optional path to a separate teacher (Stage 1 AR-diffusion) safetensors checkpoint. If unset, the teacher is initialised from the same weights as the generator."
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
        print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")
    else:
        rng = None
        torch_rng = None
        print(f"No seed provided; using global default RNG. Process_index is {accelerator.process_index}")

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
    scheduler_kwargs = OmegaConf.to_container(config['scheduler_kwargs'])
    scheduler_kwargs['shift'] = args.shift
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
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
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        vae.eval()

    # Get Transformer (causal generator)
    # IMPORTANT: keep the trainable transformer in fp32. accelerate's
    # mixed_precision="bf16" will autocast the forward to bf16 while keeping
    # the master weights and Adam moments in fp32. If params live in bf16,
    # every update (LR*grad ~ 1e-5 for LR=2e-6) falls below bf16 mantissa
    # precision (~1e-3 relative) and is rounded to zero — Stage 1 hit exactly
    # this and CCD has the same LR + same Adam betas so it would hit it again.
    # CF official keeps fp32 master weights via FSDP's default behavior
    # (MixedPrecision(param_dtype=bf16) with no compute_dtype set).
    transformer3d = WanTransformer3DModel_SelfForcing.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
    )
    # Stage 2 CCD trains the causal generator with the same block layout the
    # downstream Self-Forcing pipeline expects at sampling time.
    transformer3d.num_frame_per_block = args.num_frame_per_block
    transformer3d.independent_first_frame = args.independent_first_frame

    # Freeze vae and text_encoder; transformer3d is toggled per-module below.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        state_dict = state_dict["generator_ema"] if "generator_ema" in state_dict else state_dict
        state_dict = state_dict["generator"] if "generator" in state_dict else state_dict
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict.items()}

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    # EMA target: a non-trainable copy of the generator used as the
    # right-hand-side of the consistency objective. We use a real module
    # (instead of diffusers' EMAModel) so the CCD forward can call it directly,
    # which keeps the implementation parallel to Causal-Forcing's
    # `model/naive_consistency.py::NaiveConsistency`.
    use_ema = args.ema_weight is not None and args.ema_weight > 0
    if use_ema:
        # EMA must shadow the fp32 generator: the polyak update
        # `ema = decay*ema + (1-decay)*gen` mixes two values one mantissa apart,
        # which underflows in bf16 for decay=0.99 (the smaller-magnitude term
        # gets rounded away).
        ema_transformer3d = WanTransformer3DModel_SelfForcing.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        ema_transformer3d.num_frame_per_block = args.num_frame_per_block
        ema_transformer3d.independent_first_frame = args.independent_first_frame
        ema_transformer3d.requires_grad_(False)
        ema_transformer3d.eval()
        ema_transformer3d.load_state_dict(transformer3d.state_dict(), strict=True)
    else:
        ema_transformer3d = None

    # Teacher: frozen Stage 1 AR-diffusion model used to produce the one-step
    # ODE target. If `--teacher_transformer_path` is unset, it shares weights
    # with the (still-frozen) generator that was just loaded.
    # Teacher is frozen but kept fp32 for numerical parity with the generator
    # at init (we mirror the generator's state_dict below when no separate
    # teacher path is given). Cast to bf16 happens via autocast at call time.
    teacher_transformer3d = WanTransformer3DModel_SelfForcing.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
    )
    teacher_transformer3d.num_frame_per_block = args.num_frame_per_block
    teacher_transformer3d.independent_first_frame = args.independent_first_frame
    teacher_transformer3d.requires_grad_(False)
    teacher_transformer3d.eval()
    if args.teacher_transformer_path is not None:
        print(f"From checkpoint: {args.teacher_transformer_path}")
        if args.teacher_transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.teacher_transformer_path)
        else:
            state_dict = torch.load(args.teacher_transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        state_dict = state_dict["generator_ema"] if "generator_ema" in state_dict else state_dict
        state_dict = state_dict["generator"] if "generator" in state_dict else state_dict
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict.items()}

        m, u = teacher_transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
    elif args.transformer_path is not None:
        # Mirror the generator weights into the teacher.
        teacher_transformer3d.load_state_dict(transformer3d.state_dict(), strict=True)

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
    transformer3d.train()
    if accelerator.is_main_process:
        accelerator.print(
            f"Trainable modules '{args.trainable_modules}'."
        )
    for name, param in transformer3d.named_parameters():
        for trainable_module_name in args.trainable_modules + args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def _save_ema_pretrained(output_dir):
            if ema_transformer3d is not None and accelerator.is_main_process:
                ema_transformer3d.save_pretrained(os.path.join(output_dir, "ema_transformer"))

        def _load_ema_pretrained(input_dir):
            if ema_transformer3d is None:
                return
            ema_path = os.path.join(input_dir, "ema_transformer")
            if not os.path.exists(ema_path):
                return
            ema_loaded = WanTransformer3DModel_SelfForcing.from_pretrained(
                ema_path,
                transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
                low_cpu_mem_usage=True,
            )
            ema_transformer3d.load_state_dict(ema_loaded.state_dict(), strict=True)
            del ema_loaded
            print(f"Loaded EMA generator from {ema_path}.")

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
                _save_ema_pretrained(output_dir)

            def load_model_hook(models, input_dir):
                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                        batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                    print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")
                _load_ema_pretrained(input_dir)
        else:
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                    if not args.use_deepspeed:
                        weights.pop()

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)
                _save_ema_pretrained(output_dir)

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
                _load_ema_pretrained(input_dir)

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

    if args.fix_sample_size is not None and args.enable_bucket:
        args.video_sample_size = max(max(args.fix_sample_size), args.video_sample_size)
        args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
        args.training_with_video_token_length = False
        args.random_hw_adapt = False

    # Get the dataset (Stage 1 always trains on raw videos with teacher forcing).
    train_dataset = ImageVideoDataset(
        args.train_data_meta, args.train_data_dir,
        video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride, video_sample_n_frames=args.video_sample_n_frames,
        video_repeat=args.video_repeat,
        image_sample_size=args.image_sample_size,
        enable_bucket=args.enable_bucket, enable_inpaint=False,
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
            fix_sample_size = [int(x / 16) * 16 for x in args.fix_sample_size]
        elif args.random_ratio_crop:
            if rng is None:
                random_sample_size = aspect_ratio_random_crop_sample_size[
                    np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                ]
            else:
                random_sample_size = aspect_ratio_random_crop_sample_size[
                    rng.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                ]
            random_sample_size = [int(x / 16) * 16 for x in random_sample_size]
        else:
            closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
            closest_size = [int(x / 16) * 16 for x in closest_size]

        min_example_length = min(
            [example["pixel_values"].shape[0] for example in examples]
        )
        batch_video_length = int(min(batch_video_length, min_example_length))

        # Magvae needs the number of frames to be 4n + 1.
        batch_video_length = (batch_video_length - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1

        # Causal-Forcing needs the latent frame count to align with num_frame_per_block.
        k = (batch_video_length - 1) // sample_n_frames_bucket_interval
        if args.independent_first_frame:
            # latent_frames - 1 = k must be divisible by num_frame_per_block
            k = (k // args.num_frame_per_block) * args.num_frame_per_block
        else:
            # latent_frames = k + 1 must be divisible by num_frame_per_block
            k = ((k + 1) // args.num_frame_per_block) * args.num_frame_per_block - 1
        batch_video_length = k * sample_n_frames_bucket_interval + 1

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

        # Limit the number of frames to the same
        new_examples["pixel_values"] = torch.stack([example for example in new_examples["pixel_values"]])

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

        return new_examples

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
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

    # Prepare everything with our `accelerator`.
    transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, train_dataloader, lr_scheduler
    )
    if fsdp_stage != 0 or zero_stage != 0:
        from functools import partial

        from videox_fun.dist import set_multi_gpus_devices, shard_model
        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype)
        text_encoder = shard_fn(text_encoder)

        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype)
        teacher_transformer3d = shard_fn(teacher_transformer3d)

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    if not args.enable_text_encoder_in_dataloader:
        text_encoder.to(accelerator.device if not args.low_vram else "cpu")

    # Move the frozen teacher / EMA copies to the same device as the generator.
    # They are never wrapped by accelerator.prepare(), so we keep them in
    # inference mode on the local device only.
    # DO NOT pass dtype=weight_dtype here — we deliberately keep both modules
    # in fp32 to match the generator's master-weight precision:
    #   * Teacher's forward is autocast to bf16 anyway, so fp32 storage is
    #     just a parity/precision-safety choice (extra ~5GB on 1.3B is fine
    #     on a GB200).
    #   * EMA polyak update below (`v.mul_(decay).add_(..., alpha=1-decay)`)
    #     runs in v.dtype. With decay=0.99 the (1-decay)*delta term has
    #     magnitude ~LR*grad*0.01 ~ 1e-8, which underflows in bf16. EMA MUST
    #     be fp32 or the consistency target stops tracking the live generator.
    teacher_transformer3d.to(accelerator.device)
    if ema_transformer3d is not None:
        ema_transformer3d.to(accelerator.device)

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

    # Materialise the discrete consistency schedule with `args.discrete_cd_N`
    # timesteps. The CCD objective walks adjacent (t, t_next) pairs from this
    # grid — same convention as `FlowMatchScheduler.set_timesteps(num_inference_steps=discrete_cd_N)`
    # in Causal-Forcing's `model/naive_consistency.py`.
    noise_scheduler.set_timesteps(args.discrete_cd_N, device=accelerator.device)
    cd_timesteps = noise_scheduler.timesteps.clone()  # length N, sorted high -> low
    cd_sigmas = noise_scheduler.sigmas.clone()        # length N+1 (final 0 entry)
    # Lookup table used by `add_noise` / x0-conversion below.
    schedule_timesteps = cd_timesteps.to(accelerator.device)

    # Pre-encode the unconditional (negative) prompt once; reused for every
    # teacher CFG step. Under `--low_vram` the text encoder is parked on CPU,
    # so move it to the device for this one-off encode then put it back.
    with torch.no_grad():
        if args.low_vram:
            text_encoder.to(accelerator.device)
        neg_inputs = tokenizer(
            [args.negative_prompt], padding="max_length",
            max_length=args.tokenizer_max_length, truncation=True,
            add_special_tokens=True, return_tensors="pt",
        )
        neg_seq_len = neg_inputs.attention_mask.gt(0).sum(dim=1).long()
        neg_embed = text_encoder(neg_inputs.input_ids.to(accelerator.device), attention_mask=neg_inputs.attention_mask.to(accelerator.device))[0]
        negative_prompt_embed = neg_embed[0, :neg_seq_len[0]].detach()
        if args.low_vram:
            text_encoder.to("cpu")
            torch.cuda.empty_cache()

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.mp4", rescale=True)

            with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                # Convert prompts to text embeddings.
                if args.enable_text_encoder_in_dataloader:
                    prompt_embeds = batch['encoder_hidden_states'].to(device=accelerator.device)
                else:
                    with torch.no_grad():
                        prompt_ids = tokenizer(
                            batch['text'],
                            padding="max_length",
                            max_length=args.tokenizer_max_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt"
                        )
                        text_input_ids = prompt_ids.input_ids
                        prompt_attention_mask = prompt_ids.attention_mask

                        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                        # Under --low_vram the encoder is parked on CPU; move it to
                        # the device just for this forward pass and park it back.
                        if args.low_vram:
                            text_encoder.to(accelerator.device)
                        prompt_embeds = text_encoder(text_input_ids.to(accelerator.device), attention_mask=prompt_attention_mask.to(accelerator.device))[0]
                        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
                        if args.low_vram:
                            text_encoder.to('cpu')
                            torch.cuda.empty_cache()

                # Convert pixel videos to clean latents via the VAE.
                pixel_values = batch["pixel_values"].to(weight_dtype)
                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)

                with torch.no_grad():
                    def _batch_encode_vae(pixel_values):
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i : i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim=0)
                    clean_latents = _batch_encode_vae(pixel_values)

                if args.low_vram:
                    vae.to('cpu')
                    torch.cuda.empty_cache()

            with accelerator.accumulate(transformer3d):
                def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    sched_t = schedule_timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    step_indices = torch.argmin((sched_t.unsqueeze(0) - timesteps.reshape(-1).unsqueeze(1)).abs(), dim=1)
                    sigma = sigmas[step_indices]
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma

                def add_noise(latents, noise, timesteps):
                    """Per-frame flow-matching add_noise; supports timesteps of shape [B, F]."""
                    sigmas = get_sigmas(timesteps, n_dim=2, dtype=latents.dtype)
                    sigmas = sigmas.reshape(latents.shape[0], latents.shape[2], 1, 1, 1).permute(0, 2, 1, 3, 4)
                    return (1.0 - sigmas) * latents + sigmas * noise

                def _flow_to_x0(flow_pred, xt, timesteps):
                    """Recover x0 from flow-matching velocity: x0 = xt - sigma_t * flow_pred."""
                    sigmas = get_sigmas(timesteps, n_dim=2, dtype=flow_pred.dtype)
                    sigmas = sigmas.reshape(flow_pred.shape[0], flow_pred.shape[2], 1, 1, 1).permute(0, 2, 1, 3, 4)
                    return xt - sigmas * flow_pred

                bsz, channel, num_frames, height, width = clean_latents.shape

                # Sample one CCD timestep index per batch item from the
                # discrete schedule [0, N - 2] and share it across all causal
                # frames — mirrors `NaiveConsistency.generator_loss`.
                timestep_idx = torch.randint(
                    0, args.discrete_cd_N - 1, [bsz],
                    device=accelerator.device, generator=torch_rng,
                )
                t_scalar = cd_timesteps.to(accelerator.device)[timestep_idx].float()        # [B]
                t_next_scalar = cd_timesteps.to(accelerator.device)[timestep_idx + 1].float()  # [B]
                timestep = t_scalar.unsqueeze(1).expand(bsz, num_frames).contiguous()
                timestep_next = t_next_scalar.unsqueeze(1).expand(bsz, num_frames).contiguous()

                noise = torch.randn(clean_latents.shape, dtype=weight_dtype, device=accelerator.device, generator=torch_rng)
                noisy_latents = add_noise(clean_latents, noise, timestep)

                # Build the per-item lists once: every forward in this step is
                # teacher-forced on the clean latent, matching Stage 2 of
                # Causal-Forcing++.
                noisy_input_list = [noisy_latents[i] for i in range(bsz)]
                clean_x_list = [clean_latents[i] for i in range(bsz)]
                patch_h, patch_w = accelerator.unwrap_model(transformer3d).config.patch_size[1:]
                full_seq_len = num_frames * height * width // (patch_h * patch_w)

                # --- Teacher CFG forward + one ODE step to latent_t_next ---
                with torch.no_grad():
                    uncond_prompt_embeds = [negative_prompt_embed for _ in range(bsz)]
                    with torch.cuda.amp.autocast(dtype=weight_dtype):
                        flow_cond = teacher_transformer3d(
                            x=noisy_input_list, context=prompt_embeds,
                            t=timestep.to(torch.int64), seq_len=full_seq_len,
                            clean_x=clean_x_list,
                        )
                        flow_uncond = teacher_transformer3d(
                            x=noisy_input_list, context=uncond_prompt_embeds,
                            t=timestep.to(torch.int64), seq_len=full_seq_len,
                            clean_x=clean_x_list,
                        )
                    if isinstance(flow_cond, list):
                        flow_cond = torch.stack(flow_cond, dim=0)
                    if isinstance(flow_uncond, list):
                        flow_uncond = torch.stack(flow_uncond, dim=0)
                    flow_teacher = flow_uncond + args.guidance_scale * (flow_cond - flow_uncond)

                    # Per-frame Euler step on the flow-matching ODE.
                    # `dt = (t - t_next) / num_train_timestep`; divisor matches the
                    # `/1000` in `NaiveConsistency.generator_loss` (the CF
                    # FlowMatchScheduler uses `num_train_timesteps=1000`).
                    dt = ((timestep - timestep_next) / 1000.0).reshape(bsz, num_frames, 1, 1, 1).permute(0, 2, 1, 3, 4)
                    latent_t_next = noisy_latents - dt.to(noisy_latents.dtype) * flow_teacher.to(noisy_latents.dtype)

                # --- Generator at (latent_t, t)  ->  x0_pred_t (with grad) ---
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    flow_g = transformer3d(
                        x=noisy_input_list, context=prompt_embeds,
                        t=timestep.to(torch.int64), seq_len=full_seq_len,
                        clean_x=clean_x_list,
                    )
                    if isinstance(flow_g, list):
                        flow_g = torch.stack(flow_g, dim=0)
                    x0_pred_t = _flow_to_x0(flow_g, noisy_latents, timestep)

                # --- EMA-target at (latent_t_next, t_next) -> x0_pred_t_next (no grad) ---
                with torch.no_grad():
                    ema_model = ema_transformer3d if ema_transformer3d is not None else accelerator.unwrap_model(transformer3d)
                    noisy_next_list = [latent_t_next[i] for i in range(bsz)]
                    with torch.cuda.amp.autocast(dtype=weight_dtype):
                        flow_ema = ema_model(
                            x=noisy_next_list, context=prompt_embeds,
                            t=timestep_next.to(torch.int64), seq_len=full_seq_len,
                            clean_x=clean_x_list,
                        )
                    if isinstance(flow_ema, list):
                        flow_ema = torch.stack(flow_ema, dim=0)
                    x0_pred_t_next = _flow_to_x0(flow_ema, latent_t_next, timestep_next).detach()

                # CCD objective: MSE between the two x0 predictions.
                loss = F.mse_loss(x0_pred_t.float(), x0_pred_t_next.float())

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # EMA update for the consistency target. Before `--ema_start_step`
                # we keep the EMA copy in lock-step with the live generator, so
                # that early-training instability does not bias the right-hand
                # side of the CCD objective.
                #
                # Under FSDP the live generator only holds a *shard* of each
                # parameter per rank (the launch script uses FULL_SHARD +
                # SHARDED_STATE_DICT), so `unwrap_model(...).state_dict()` returns
                # sharded tensors whose keys/shapes do not line up with the full,
                # un-sharded `ema_transformer3d` copy that every rank runs forward
                # on. Accelerate's default FULL_STATE_DICT config is also
                # rank0-only + CPU-offloaded, which would leave rank>0 with empty
                # dicts and rank0 with device-mismatched CPU tensors. We therefore
                # force a full, un-sharded gather to *every* rank (rank0_only=False,
                # offload_to_cpu=False) so each rank can update its own
                # full-precision EMA copy locally and stay bit-for-bit in sync.
                if ema_transformer3d is not None and accelerator.sync_gradients:
                    if args.use_fsdp:
                        from torch.distributed.fsdp import \
                            FullyShardedDataParallel as FSDP
                        from torch.distributed.fsdp import (FullStateDictConfig,
                                                            StateDictType)
                        # Collective: all ranks enter this together because the
                        # enclosing guard only depends on synchronized state
                        # (`sync_gradients` / `use_fsdp`), never on rank-local data.
                        with FSDP.state_dict_type(
                            transformer3d,
                            StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
                        ):
                            live_state = transformer3d.state_dict()
                    else:
                        live_state = accelerator.unwrap_model(transformer3d).state_dict()
                    # FULL_STATE_DICT already strips FSDP wrapper prefixes, but stay
                    # defensive against any `_fsdp_wrapped_module.` leftovers so the
                    # keys match the plain EMA module's state_dict.
                    live_state = {
                        (k.replace("_fsdp_wrapped_module.", "") if "_fsdp_wrapped_module." in k else k): v
                        for k, v in live_state.items()
                    }
                    if global_step < args.ema_start_step:
                        ema_transformer3d.load_state_dict(live_state, strict=True)
                    else:
                        decay = args.ema_weight
                        ema_state = ema_transformer3d.state_dict()
                        with torch.no_grad():
                            for k, v in ema_state.items():
                                src = live_state[k].to(device=v.device, dtype=v.dtype)
                                if v.dtype.is_floating_point:
                                    v.mul_(decay).add_(src, alpha=1.0 - decay)
                                else:
                                    v.copy_(src)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

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
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    log_validation(
                        vae,
                        text_encoder,
                        tokenizer,
                        transformer3d,
                        args,
                        config,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            log_validation(
                vae,
                text_encoder,
                tokenizer,
                transformer3d,
                args,
                config,
                accelerator,
                weight_dtype,
                global_step,
            )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer3d = unwrap_model(transformer3d)

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
