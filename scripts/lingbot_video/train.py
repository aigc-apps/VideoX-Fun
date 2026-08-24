"""LingBot-Video ti2v full-parameter training script.

Flow-matching training of the LingBot-Video single-stream joint DiT, mirroring
the ti2v inference path of examples/lingbot_video/predict_i2v.py:

  - The first frame is used twice, exactly like at inference time:
      1. as visual input of the frozen Qwen3-VL text encoder
         (<|vision_start|>...<|image_pad|>...<|vision_end|> tokens);
      2. as a clean latent written into the temporal prefix of the noisy latent
         (inpainting-style clamping, see LingBotVideoPipeline._apply_inpainting).
  - Loss is computed on the non-conditioned latent frames only.
  - Sigma is sampled from the same shifted FlowUniPC schedule used at inference
     (sigma' = shift * sigma / (1 + (shift - 1) * sigma)), and the transformer
     receives timestep = sigma * 1000.

Modified from scripts/wan2.2/train.py / scripts/qwenimage/train.py.
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
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from packaging import version
from PIL import Image
from torch.utils.data import RandomSampler as TorchRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoProcessor
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
                             RandomSampler, get_closest_ratio)
from videox_fun.models import (AutoencoderKLQwenImage,
                               LingBotVideoTransformer3DModel,
                               Qwen3VLForConditionalGeneration)
from videox_fun.pipeline import LingBotVideoI2VPipeline
from videox_fun.pipeline.pipeline_lingbot_video_i2v import (SPATIAL_MERGE_SIZE,
                                                            smart_resize)
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.utils import calculate_dimensions, save_videos_grid

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
        return np.random.choice(number_list, p=number_list_prob)
    else:
        return rng.choice(number_list, p=number_list_prob)


def build_sigma_table(sigma_max, sigma_min, num_steps, shift):
    """The shifted sigma grid used by FlowUniPC at inference (see
    FlowUniPCMultistepScheduler.set_timesteps and compute_refiner_sigmas)."""
    sigmas = np.linspace(float(sigma_max), float(sigma_min), int(num_steps) + 1)[:-1]
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    return torch.from_numpy(sigmas.astype(np.float32))


def vae_latent_to_dit(vae, latents):
    """Normalize raw VAE latents with the per-channel mean/std, matching
    LingBotVideoPipeline._vae_latent_to_dit."""
    device = latents.device
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std_inv = (1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32)).view(1, -1, 1, 1, 1)
    return (latents.float() - mean) * std_inv


def frame_to_vlm_image(frame):
    """frame: (C, H, W) tensor in [-1, 1]. Returns the PIL image fed to
    Qwen3-VL, resized exactly like LingBotVideoI2VPipeline._vlm_image."""
    frame = (frame.detach().cpu().clamp(-1, 1) * 0.5 + 0.5)
    array = frame.permute(1, 2, 0).mul(255).byte().numpy()
    image = Image.fromarray(array, mode="RGB")
    return image


def resize_vlm_image(image, patch_factor):
    resized_height, resized_width = smart_resize(image.height, image.width, factor=patch_factor)
    return image.resize((resized_width, resized_height))


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def log_validation(vae, text_encoder, processor, transformer3d, args, accelerator, weight_dtype, global_step):
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
            logger.info("Running validation... ")
            scheduler = FlowUniPCMultistepScheduler.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="scheduler",
            )

            # Under FSDP the transformer3d passed here is the wrapped (sharded)
            # module; it must be used as-is because its forward is collective.
            pipeline = LingBotVideoI2VPipeline(
                transformer=transformer3d,
                vae=vae,
                text_encoder=text_encoder,
                processor=processor,
                scheduler=scheduler,
            )
            if not args.use_fsdp:
                pipeline = pipeline.to(accelerator.device)

            if args.seed is None:
                generator = None
            else:
                rank_seed = args.seed + accelerator.process_index
                generator = torch.Generator(device=accelerator.device).manual_seed(rank_seed)
                logger.info(f"Rank {accelerator.process_index} using seed: {rank_seed}")

            video_length = int((args.video_sample_n_frames - 1) // pipeline.vae_scale_factor_temporal * pipeline.vae_scale_factor_temporal) + 1 if args.video_sample_n_frames != 1 else 1

            for i in range(len(args.validation_prompts)):
                image = Image.open(args.validation_paths[i]).convert("RGB")
                width, height = calculate_dimensions(
                    args.image_sample_size * args.image_sample_size, image.width / image.height)
                width, height = int(width), int(height)

                sample = pipeline(
                    args.validation_prompts[i],
                    image       = image,
                    num_frames  = video_length,
                    height      = height,
                    width       = width,
                    generator   = generator,
                    guidance_scale = 3.0,
                    shift       = args.train_shift,
                    num_inference_steps = 25,
                ).videos

                os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                save_videos_grid(
                    sample,
                    os.path.join(
                        args.output_dir,
                        f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.mp4"
                    ),
                    fps=24,
                )

            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            vae.to(accelerator.device if not args.low_vram else "cpu")
            text_encoder.to(accelerator.device if not args.low_vram else "cpu")
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error on rank {accelerator.process_index} with info {e}")
        vae.to(accelerator.device if not args.low_vram else "cpu")
        text_encoder.to(accelerator.device if not args.low_vram else "cpu")


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value


def parse_args():
    parser = argparse.ArgumentParser(description="LingBot-Video ti2v training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to the LingBot-Video model root (contains transformer/ vae/ text_encoder/ processor/ scheduler/).",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help="A json containing the training data.",
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
        help="A set of prompts evaluated every `--validation_steps` and logged to `--report_to`. "
             "Must be structured JSON captions (rewriter schema), same as the training data.",
    )
    parser.add_argument(
        "--validation_paths",
        type=str,
        default=None,
        nargs="+",
        help="A set of condition (first-frame) images evaluated every `--validation_steps`, paired with `--validation_prompts`.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lingbot-video-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=1, help="mini batch size for vae."
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
        default=1e-5,
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
        default="lingbot-video-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
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
        "--training_with_video_token_length", action="store_true", help="Keep the video token length constant while adapting height and width.",
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Number of discrete sigma levels used for training-time sigma sampling.",
    )
    parser.add_argument(
        "--train_shift",
        type=float,
        default=3.0,
        help="Shift of the flow-matching sigma schedule, same as the inference `shift` (3.0 for LingBot-Video).",
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
        default=1,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=81,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Num of repeat video.",
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
        default=["."],
        help='Enter a list of trainable modules (substring match).'
    )
    parser.add_argument(
        '--trainable_modules_low_learning_rate',
        nargs='+',
        default=[],
        help='Enter a list of trainable modules with lower learning rate'
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
        help=('When do we start doing additional processing on abnormal gradients.'),
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=5,
        help=('The initial gradient is relative to the multiple of the max_grad_norm.'),
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

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.validation_prompts is not None and args.validation_paths is not None:
        assert len(args.validation_prompts) == len(args.validation_paths), \
            "validation_prompts and validation_paths must be paired one-to-one."

    return args



def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    if fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy in (ShardingStrategy.FULL_SHARD, None): # The fsdp_plugin.sharding_strategy is None in FSDP 2.
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
        print("FSDP is not enabled.")

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
    print(f"Init rng with seed {args.seed + accelerator.process_index if args.seed is not None else None}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder)
    # to half-precision as these weights are only used for inference, keeping weights in
    # full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load the sigma scheduler (only used to build the training sigma table).
    noise_scheduler = FlowUniPCMultistepScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    # Get Processor (Qwen3-VL tokenizer + image processor)
    processor = AutoProcessor.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "processor"),
    )
    vision_patch_size = getattr(getattr(processor, "image_processor", None), "config", None)
    vision_patch_size = getattr(vision_patch_size, "patch_size", 16) if vision_patch_size is not None else 16
    vlm_patch_factor = int(vision_patch_size) * SPATIAL_MERGE_SIZE

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns a list of context managers to disable deepspeed zero init
        """
        deepspeed_plugin = accelerate.state.AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []
        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # Get Text encoder (frozen Qwen3-VL)
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, "text_encoder"),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        text_encoder = text_encoder.eval()
        # Get Vae
        vae = AutoencoderKLQwenImage.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, "vae"),
        )
        vae.eval()

    # Get Transformer. `.to(weight_dtype)` keeps the fp32-sensitive modules
    # (norms / router / modulation / scale_shift_table) in fp32 automatically.
    transformer3d = LingBotVideoTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "transformer"),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).to(weight_dtype)

    # FSDP flattens parameters of one wrap unit into a single flat buffer and
    # requires a uniform dtype. The custom `.to()` above keeps fp32-sensitive
    # modules (norms / router / modulation / scale_shift_table) in fp32, so under
    # FSDP we force-cast everything to weight_dtype (matches bf16 inference).
    if fsdp_stage != 0:
        for _name, _param in transformer3d.named_parameters():
            _param.data = _param.data.to(weight_dtype)
        for _name, _buffer in transformer3d.named_buffers():
            _buffer.data = _buffer.data.to(weight_dtype)

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
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
            from safetensors.torch import load_file
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

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

    # Create EMA for the transformer3d.
    if args.use_ema:
        if fsdp_stage == 3:
            raise NotImplementedError("FSDP FULL_SHARD does not support EMA.")

        ema_transformer3d = LingBotVideoTransformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, "transformer"),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        ).to(weight_dtype)

        ema_transformer3d = EMAModel(ema_transformer3d.parameters(), model_cls=LingBotVideoTransformer3DModel, model_config=ema_transformer3d.config)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        if fsdp_stage != 0:
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
                    if args.use_ema:
                        ema_transformer3d.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                    models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                    weights.pop()

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

            def load_model_hook(models, input_dir):
                if args.use_ema:
                    ema_path = os.path.join(input_dir, "transformer_ema")
                    _, ema_kwargs = LingBotVideoTransformer3DModel.load_config(ema_path, return_unused_kwargs=True)
                    load_model = LingBotVideoTransformer3DModel.from_pretrained(
                        input_dir, subfolder="transformer_ema",
                    )
                    load_model = EMAModel(load_model.parameters(), model_cls=LingBotVideoTransformer3DModel, model_config=load_model.config)
                    load_model.load_state_dict(ema_kwargs)

                    ema_transformer3d.load_state_dict(load_model.state_dict())
                    ema_transformer3d.to(accelerator.device)
                    del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = LingBotVideoTransformer3DModel.from_pretrained(
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

    optimizer = optimizer_cls(
        trainable_params_optim,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the training dataset
    # AutoencoderKLQwenImage's config has no compression-ratio fields; the
    # pipeline constants are temporal=4 / spatial=8.
    sample_n_frames_bucket_interval = getattr(vae.config, "temporal_compression_ratio", 4)
    spatial_compression_ratio = getattr(vae.config, "spatial_compression_ratio", 8)

    if args.fix_sample_size is not None and args.enable_bucket:
        args.video_sample_size = max(max(args.fix_sample_size), args.video_sample_size)
        args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
        args.training_with_video_token_length = False
        args.random_hw_adapt = False

    # Get the dataset
    train_dataset = ImageVideoDataset(
        args.train_data_meta, args.train_data_dir,
        video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride, video_sample_n_frames=args.video_sample_n_frames,
        video_repeat=args.video_repeat,
        image_sample_size=args.image_sample_size,
        enable_bucket=args.enable_bucket, enable_inpaint=False,
    )

    # The DiT was trained on structured JSON captions only; plain natural
    # language captions are out-of-distribution and degrade fine-tuning.
    from videox_fun.models.lingbot_video_rewriter import is_valid_caption
    n_plain = sum(1 for d in train_dataset.dataset if not is_valid_caption(d.get("text", "")))
    if n_plain:
        logger.warning(
            f"{n_plain}/{len(train_dataset.dataset)} dataset captions are NOT structured JSON captions. "
            "LingBot-Video expects rewriter-style JSON captions; training on natural-language text "
            "feeds out-of-distribution prompts to the DiT. All captions must go through the rewriter "
            "(no hand-written captions) — convert the dataset with "
            "scripts/lingbot_video/prepare_captions.py (batch) or "
            "videox_fun.models.lingbot_video_rewriter.ensure_json_caption (single)."
        )

    if args.enable_bucket:
        aspect_ratio_sample_size = {key: [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset,
            batch_size=args.train_batch_size, train_folder=args.train_data_dir, drop_last=True,
            aspect_ratios=aspect_ratio_sample_size,
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

                aspect_ratio_sample_size = {key: [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key: [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}

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

                aspect_ratio_sample_size = {key: [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key: [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}

            if args.fix_sample_size is not None:
                fix_sample_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in args.fix_sample_size]
            elif args.random_ratio_crop:
                if rng is None:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()), p=ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                else:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        rng.choice(list(aspect_ratio_random_crop_sample_size.keys()), p=ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                random_sample_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in random_sample_size]
            else:
                closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
                closest_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in closest_size]

            min_example_length = min(
                [example["pixel_values"].shape[0] for example in examples]
            )
            batch_video_length = int(min(batch_video_length, min_example_length))

            # VAE needs the number of frames to be 4n + 1.
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

            # Limit the number of frames to the same
            new_examples["pixel_values"] = torch.stack([example for example in new_examples["pixel_values"]])

            return new_examples

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

    # Prepare everything with our `accelerator`.
    transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, train_dataloader, lr_scheduler
    )

    if fsdp_stage != 0:
        from functools import partial

        from videox_fun.dist import shard_model
        # Qwen3-VL has no `.blocks`; wrap the text decoder layers and vision
        # blocks explicitly via the transformer-class policy.
        shard_fn = partial(
            shard_model, device_id=accelerator.device, param_dtype=weight_dtype,
            transformer_layer_cls_to_wrap=["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"],
        )
        text_encoder = shard_fn(text_encoder)

    if args.use_ema:
        ema_transformer3d.to(accelerator.device)

    # Move text_encoder and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)

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

    # Training-time sigma table: the same shifted FlowUniPC grid used at inference.
    sigma_table = build_sigma_table(
        noise_scheduler.sigma_max, noise_scheduler.sigma_min,
        args.train_sampling_steps, args.train_shift,
    )
    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    # A throw-away pipeline used only to reuse the inference-time prompt encoding
    # (chat template + image tokens + system-prefix cropping).
    encode_pipeline = LingBotVideoI2VPipeline(
        transformer=None,
        vae=vae,
        text_encoder=text_encoder,
        processor=processor,
        scheduler=None,
    )

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

            with accelerator.accumulate(transformer3d):
                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")

                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)

                with torch.no_grad():
                    # This way is quicker when batch grows up
                    def _batch_encode_vae(pixel_values):
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i : i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs.to(vae.dtype)).latent_dist.sample()
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim=0)

                    latents = vae_latent_to_dit(vae, _batch_encode_vae(pixel_values))

                    # ti2v condition: the first frame encoded on its own, exactly
                    # like LingBotVideoI2VPipeline.encode_image_latent at inference.
                    cond_latent = vae_latent_to_dit(vae, _batch_encode_vae(pixel_values[:, :, :1]))

                if args.low_vram:
                    vae.to('cpu')
                    torch.cuda.empty_cache()

                # Text (+ first-frame image) encoding with the frozen Qwen3-VL.
                with torch.no_grad():
                    if args.low_vram:
                        text_encoder.to(accelerator.device)
                    vlm_images = [
                        resize_vlm_image(frame_to_vlm_image(frame), vlm_patch_factor)
                        for frame in pixel_values[:, :, 0]
                    ]
                    prompt_embeds, prompt_mask = encode_pipeline.encode_prompt(
                        batch['text'], images=vlm_images, device=accelerator.device)
                    prompt_embeds = prompt_embeds.to(device=latents.device)
                    prompt_mask = prompt_mask.to(device=latents.device)
                    if args.low_vram:
                        text_encoder.to('cpu')
                        torch.cuda.empty_cache()

                bsz, channel, num_frames, height, width = latents.size()
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

                # Sample sigmas from the shifted table.
                if not args.uniform_sampling:
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * args.train_sampling_steps).long()
                else:
                    indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                    indices = indices.long().cpu()
                sigmas = sigma_table[indices].to(device=latents.device)

                # Add noise according to flow matching.
                # zt = (1 - sigma) * x0 + sigma * noise
                sigmas = sigmas.view(-1, 1, 1, 1, 1).to(latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Clamp the condition frame(s) back to the clean latent, matching
                # LingBotVideoPipeline._apply_inpainting at inference. The loss is
                # masked on these frames accordingly.
                cond_t = cond_latent.shape[2]
                frame_mask = torch.ones_like(latents[:, :, :1].expand(-1, -1, num_frames, -1, -1))
                if num_frames > cond_t:
                    noisy_latents[:, :, :cond_t] = cond_latent.to(noisy_latents.dtype)
                    frame_mask[:, :, :cond_t] = 0.0
                else:
                    # Degenerate single-latent-frame sample: nothing to generate.
                    frame_mask[:, :, :] = 0.0

                # Predict the noise residual
                timesteps = (sigmas.reshape(-1) * 1000.0).float()
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    noise_pred = transformer3d(
                        noisy_latents,
                        timesteps,
                        prompt_embeds.to(weight_dtype),
                        encoder_attention_mask=prompt_mask,
                        return_dict=False,
                    )[0]

                target = noise - latents

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas.squeeze())
                weighting = weighting.view(-1, 1, 1, 1, 1).float()

                mse_loss = F.mse_loss(noise_pred.float(), target.float(), reduction='none')
                masked_loss = mse_loss * frame_mask * weighting
                denom = frame_mask.sum() * channel * height * width
                loss = masked_loss.sum() / denom.clamp(min=1.0)
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_fsdp:
                        trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                        trainable_params_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                        max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                        if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                            actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                        else:
                            actual_max_grad_norm = max_grad_norm
                    else:
                        actual_max_grad_norm = args.max_grad_norm

                    if not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        if trainable_params_total_norm > 1 and global_step > args.abnormal_norm_clip_start:
                            for name, param in transformer3d.named_parameters():
                                if param.requires_grad:
                                    writer.add_scalar(f'gradients/before_clip_norm/{name}', param.grad.norm(), global_step=global_step)

                    norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)
                    if not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        writer.add_scalar(f'gradients/norm_sum', norm_sum, global_step=global_step)
                        writer.add_scalar(f'gradients/actual_max_grad_norm', actual_max_grad_norm, global_step=global_step)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                if args.use_ema:
                    ema_transformer3d.step(transformer3d.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if args.use_fsdp or accelerator.is_main_process:
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
                    if args.use_ema:
                        # Store the transformer parameters temporarily and load the EMA parameters to perform inference.
                        ema_transformer3d.store(transformer3d.parameters())
                        ema_transformer3d.copy_to(transformer3d.parameters())
                    log_validation(
                        vae,
                        text_encoder,
                        processor,
                        transformer3d,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )
                    if args.use_ema:
                        # Switch back to the original transformer3d parameters.
                        ema_transformer3d.restore(transformer3d.parameters())

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            if args.use_ema:
                # Store the transformer parameters temporarily and load the EMA parameters to perform inference.
                ema_transformer3d.store(transformer3d.parameters())
                ema_transformer3d.copy_to(transformer3d.parameters())
            log_validation(
                vae,
                text_encoder,
                processor,
                transformer3d,
                args,
                accelerator,
                weight_dtype,
                global_step,
            )
            if args.use_ema:
                # Switch back to the original transformer3d parameters.
                ema_transformer3d.restore(transformer3d.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer3d = unwrap_model(transformer3d)
        if args.use_ema:
            ema_transformer3d.copy_to(transformer3d.parameters())

    if args.use_fsdp or accelerator.is_main_process:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
