# Modified from scripts/minimax_h3/train.py for MiniMax-H3-Image, aligned to the videox-fun unified training
# scaffold (parameter set, trainable modules, EMA, abnormal gradient clip, checkpointing).
#
# Full finetuning of the packed-sequence image transformer, covering `t2i` (text only) and `edit` (one or more
# reference images conditioning the generated image). It is the video trainer with the audio branch removed and the
# video branch flattened to one frame: the dataset, the collate and the bucket sampler come from the image families
# (`scripts/qwenimage/train.py`), while the rectified-flow recipe, the packed layout, the conditioner and the FSDP /
# offload machinery stay MiniMax-H3's and are imported from `videox_fun` instead of being restated here.
#
# MiniMax-H3's rectified-flow convention is the *opposite* of Wan's and is reproduced from
# `MiniMaxH3Scheduler.scale_noise` / `MiniMaxH3Scheduler.step`, the single source of truth:
#   * noising: `x_t = t * x0 + (1 - t) * noise` with `t = 1` clean, `t = 1 - sigma`,
#   * the transformer predicts a data-ward velocity, so the regression target is `x0 - noise`,
#   * the conditioning rows are noise-augmented to `t = 0.999` and are never supervised.
# Unlike video, the sigma is *not* exponentially shifted by default: the released image checkpoint samples on six
# fixed sigmas consumed verbatim (`shift_provided_sigmas = False` in its `scheduler_config.json`), so a plain draw
# is what matches it. `--sigma_shift` re-introduces the shift for runs that want mass moved to the noisy end.
#
# By default this uses the original shared `final_layer.video_out` (`proj_out`), like MiniMax-H3 video. Explicit
# `--output_head_type=pdd` selects the six released per-step heads. Both modes use ordinary flow-matching training;
# this script does not run a PDD distillation teacher or trajectory.
#
# Usage:
#   accelerate launch --mixed_precision=no scripts/minimax_h3_image/train.py \
#       --pretrained_model_name_or_path=models/Diffusion_Transformer/MiniMax-H3-Image \
#       --train_data_meta=datasets/X-Fun-Images-Demo/metadata_add_width_height.json \
#       --train_data_dir=datasets/X-Fun-Images-Demo/ \
#       --train_mode=t2i --image_sample_size=1024 --enable_bucket \
#       --gradient_checkpointing --low_vram --trainable_modules "."
#
#   # image editing: the meta carries a `source_file_path` per sample and `--train_mode=edit` reads it
#   accelerate launch --mixed_precision=no scripts/minimax_h3_image/train.py \
#       --pretrained_model_name_or_path=models/Diffusion_Transformer/MiniMax-H3-Image \
#       --train_data_meta=datasets/X-Fun-Images-Edit-Demo/metadata_add_width_height.json \
#       --train_data_dir=datasets/X-Fun-Images-Edit-Demo/ \
#       --train_mode=edit --image_sample_size=1024 --enable_bucket \
#       --gradient_checkpointing --low_vram --trainable_modules "."

import argparse
import gc
import logging
import math
import os
import pickle
import random
import sys
import warnings

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
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling)
from diffusers.utils.torch_utils import is_compiled_module
from packaging import version
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers.utils import ContextManagers

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.data import (ASPECT_RATIO_512,
                             AspectRatioBatchImageVideoSampler, ImageEditDataset,
                             ImageVideoDataset, RandomSampler,
                             get_closest_ratio)
from videox_fun.models import (AutoencoderKLMiniMaxH3Image,
                               MiniMaxH3ImageTransformer2DModel, Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration, Qwen3VLProcessor)
from videox_fun.models.minimax_h3_image_transformer2d import resolve_image_transformer_component
from videox_fun.pipeline import MiniMaxH3ImagePipeline
from videox_fun.pipeline.pipeline_minimax_h3 import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG, MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD,
    build_row_timesteps, normalize_reference_image, patchify_video_latents,
    ref2va_condition_rows)
from videox_fun.pipeline.pipeline_minimax_h3_image import (
    build_image_packed_sequence, head_timestep_grid)
from videox_fun.utils import (add_training_precision_args,
                              FSDPEMA, MiniMaxH3Scheduler, PauseAwareTqdm,
                              build_trainable_optimizer_groups,
                              configure_gradient_checkpointing_save_on_cpu,
                              configure_training_precision, create_optimizer,
                              get_image, initialize_training_seed,
                              parse_trainable_module_learning_rates,
                              pdd_head_index, shifted_sigma)

# Silences diffusers' `randn_tensor` notice about CPU generators producing CUDA tensors (the tensor is created
# on CPU and moved to GPU; harmless, only a marginal speed note).
warnings.filterwarnings("ignore", message="The passed generator was created on")


def component_folder(root, *names):
    r"""First of `names` that exists under `root`.

    The released partition ships its transformer and VAE as `transformer_diffusers` / `vae_diffusers`, and a
    converted snapshot as `transformer` / `vae`; this is the same rule
    `MiniMaxH3ImagePipeline.from_pretrained` applies, kept here because the trainer loads the components one by one
    (the transformer outside the DeepSpeed `zero.Init` context, the frozen ones inside it) rather than as a pipeline.
    """
    for name in names:
        folder = os.path.join(os.fspath(root), name)
        if os.path.isdir(folder):
            return folder
    raise FileNotFoundError(f"None of {names} was found under {root}.")


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value


logger = get_logger(__name__, log_level="INFO")


def log_validation(
    vae, text_encoder, tokenizer, processor, transformer,
    scheduler, args, accelerator, weight_dtype, global_step,
):
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=weight_dtype):
            logger.info("Running validation... ")
            pipeline = MiniMaxH3ImagePipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                processor=processor,
                # Under FSDP the transformer must keep its wrapper so `_pre_forward_unshard` materializes the
                # sharded FlatParameters during inference; unwrapping leaves weights as 1-D shard views.
                transformer=accelerator.unwrap_model(transformer) if type(transformer).__name__ == 'DistributedDataParallel' else transformer,
                scheduler=scheduler,
            )
            pipeline = pipeline.to(accelerator.device)

            if args.seed is None:
                generator = None
            else:
                rank_seed = args.seed + accelerator.process_index
                generator = torch.Generator(device=accelerator.device).manual_seed(rank_seed)
                logger.info(f"Rank {accelerator.process_index} using seed: {rank_seed}")

            # PDD uses its fixed six-step schedule; single-head validation uses its configured step count and
            # shift. Guidance is distilled in both cases. Edit references are encoded like training samples.
            min_generation_area = args.min_generation_area or args.image_sample_size ** 2
            for i in range(len(args.validation_prompts)):
                references = None
                if args.train_mode == "edit" and args.validation_image_paths:
                    references = [get_image(args.validation_image_paths[i])]
                output = pipeline(
                    args.validation_prompts[i],
                    images=references,
                    height=args.fix_sample_size[0] if args.fix_sample_size else args.image_sample_size,
                    width=args.fix_sample_size[1] if args.fix_sample_size else args.image_sample_size,
                    min_generation_area=min_generation_area,
                    generator=generator,
                    **({"num_inference_steps": args.validation_num_inference_steps,
                        "flow_shift": args.validation_flow_shift} if args.output_head_type == "single" else {}),
                )
                os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                output.images[0].save(
                    os.path.join(
                        args.output_dir,
                        f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.png",
                    )
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


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMax-H3-Image training (t2i / edit).")
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
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help="A csv/json containing the training data. Under --train_mode=edit each entry also carries a `source_file_path`.",
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
        "--output_head_type", choices=("single", "pdd"), default="single",
        help="Output projection mode: one shared projection by default; select pdd explicitly for per-step heads.",
    )
    parser.add_argument(
        "--validation_num_inference_steps", type=int, default=None,
        help="Single-head validation forwards (default 40); PDD keeps its fixed six-step schedule.",
    )
    parser.add_argument(
        "--validation_flow_shift", type=float, default=None,
        help="Single-head validation sigma shift (default 12).",
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_dir_minimax_h3_image",
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
        "--vae_mini_batch", type=int, default=1, help="mini batch size for vae."
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
        "--gradient_checkpointing_save_on_cpu",
        action="store_true",
        help="Save tensors from active gradient checkpoint calls on CPU to reduce accelerator memory.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--low_learning_rate", type=float, default=None,
        help="Learning rate for selected modules (default: half the main learning rate).",
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
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training."
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
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
        '--trainable_modules', 
        nargs='+', 
        help='Enter a list of trainable modules'
    )
    parser.add_argument(
        '--trainable_modules_low_learning_rate', 
        nargs='+', 
        default=[],
        help='Trainable module substrings for the low learning rate.'
    )
    parser.add_argument(
        "--abnormal_norm_clip_start",
        type=int,
        default=1000,
        help=(
            'When do we start doing additional processing on abnormal gradients. '
        ),
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=5,
        help=(
            'The initial gradient is relative to the multiple of the max_grad_norm. '
        ),
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
        "--sigma_shift",
        type=float,
        default=1.0,
        help=(
            "Exponential shift applied to the sampled sigma before it becomes `t = 1 - sigma`. 1.0 keeps the plain "
            "draw, which is what the released image grid is sampled on; the video value (`scheduler.shift`, 12.0) "
            "moves the training mass to the noisy end."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10 and an Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the"
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
    parser.add_argument(
        "--report_model_info", action="store_true", help="Whether or not to report more info about model (such as norm, grad)."
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
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--use_fsdp", action="store_true", help="Whether or not to use fsdp."
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # MiniMax-H3-Image specific
    parser.add_argument(
        "--train_mode",
        type=str,
        default="t2i",
        choices=["t2i", "edit"],
        help="t2i (text only) or edit (the `source_file_path` of each sample conditions the generated image).",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=1024,
        help="Square side of the training canvas; a multiple of 32, the model's VAE stride times its patch.",
    )
    parser.add_argument(
        "--fix_sample_size",
        type=int,
        nargs=2,
        default=None,
        help="Fix Sample size [height, width] when using bucket and collate_fn.",
    )
    parser.add_argument(
        "--min_generation_area",
        type=int,
        default=None,
        help=(
            "Area floor used by the validation samples only (the training canvas is the dataset size); defaults to "
            "--image_sample_size ** 2 so validation stays on the resolution the run trains on."
        ),
    )
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="Keep VAE and conditioner on CPU, move to GPU only while encoding.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="minimax_h3_image",
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
        help=("A set of prompts evaluated every `--validation_steps` / `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_image_paths",
        type=str,
        default=None,
        nargs="+",
        help=("Reference images for the validation samples of an `--train_mode=edit` run, one per prompt."),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )

    add_training_precision_args(parser)
    args = parser.parse_args()
    configure_training_precision(args)
    configure_gradient_checkpointing_save_on_cpu(args.gradient_checkpointing_save_on_cpu)
    parse_trainable_module_learning_rates(args, parser)
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    if not math.isfinite(args.sigma_shift) or args.sigma_shift <= 0:
        raise ValueError("--sigma_shift must be finite and positive.")
    if args.output_head_type == "pdd":
        if args.validation_flow_shift is not None or args.validation_num_inference_steps not in (None, 6):
            raise ValueError("PDD validation uses its fixed six-step grid and does not accept a flow shift.")
    else:
        if args.validation_num_inference_steps is not None and args.validation_num_inference_steps < 1:
            raise ValueError("--validation_num_inference_steps must be positive.")
        if args.validation_flow_shift is not None and (
            not math.isfinite(args.validation_flow_shift) or args.validation_flow_shift <= 0
        ):
            raise ValueError("--validation_flow_shift must be finite and positive.")

    if args.train_mode not in ("t2i", "edit"):
        raise ValueError(f"`train_mode` must be 't2i' or 'edit', got {args.train_mode!r}.")
    if not args.enable_bucket:
        raise ValueError(
            "MiniMax-H3-Image trains on per-sample canvases whose sides are multiples of 32 (the VAE's spatial "
            "stride times the transformer's patch), which the aspect-ratio buckets provide: pass --enable_bucket."
        )
    canvas_multiple = 32
    if args.fix_sample_size is not None:
        args.fix_sample_size = [int(x / canvas_multiple) * canvas_multiple for x in args.fix_sample_size]
        args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
        args.random_hw_adapt = False
    if args.image_sample_size % canvas_multiple:
        raise ValueError(
            f"`image_sample_size` {args.image_sample_size} must be a multiple of {canvas_multiple}: the canvas is "
            f"patched 2x2 into the transformer on top of the VAE's {canvas_multiple // 2}x spatial compression, and "
            "both the RoPE grid and the PDD output head key off that."
        )
    if args.train_batch_size != 1:
        raise ValueError("MiniMax-H3-Image packed-sequence training requires --train_batch_size=1.")

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
        print(f"Using DeepSpeed Zero stage: {zero_stage}")
        args.use_deepspeed = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        print(f"Using FSDP stage: {fsdp_stage}")
        args.use_fsdp = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("DeepSpeed/FSDP is not enabled.")

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # FSDP1 re-exposes old sharded gradient views after moving a zeroed model
    # CPU -> GPU. Reset only at the start of a new accumulation window.
    reset_fsdp_offload_grads = (
        args.low_vram and fsdp_plugin is not None and not getattr(accelerator, "is_fsdp2", False)
    )
    if reset_fsdp_offload_grads and args.gradient_accumulation_steps > 1:
        # Unsynchronized FSDP gradients are unsharded and cannot safely be
        # moved between devices. Synchronize each microbatch while retaining
        # Accelerator's normal optimizer-step accumulation schedule.
        accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = True
        accelerator.print("FSDP low_vram: synchronizing each accumulation microbatch before offload.")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
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
    initialize_training_seed(args, accelerator)
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast non-trainable weights to half-precision.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # ------------------------------------------------------------------ models
    # `pretrained_model_name_or_path` points at a MiniMax-H3-Image snapshot; the transformer and the VAE are read
    # from their released `*_diffusers` folders or from the plain `transformer` / `vae` names a converted snapshot
    # or a training checkpoint uses — `component_folder` is the same rule `MiniMaxH3ImagePipeline.from_pretrained`
    # applies.
    transformer = MiniMaxH3ImageTransformer2DModel.from_pretrained(
        resolve_image_transformer_component(args.pretrained_model_name_or_path),
        low_cpu_mem_usage=True, torch_dtype=weight_dtype, output_head_type=args.output_head_type,
    )
    single_head_sha256 = getattr(transformer.config, "single_head_sha256", None)
    if args.output_head_type == "single":
        logger.info(f"Using original single FP32 output head, sha256={single_head_sha256}")
    else:
        logger.info("Using explicitly selected PDD output heads")

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3. Excluding the
    # frozen models from `zero.Init` keeps them whole on every rank and lets the transformer be the only ZeRO-sharded
    # module — the same workaround `scripts/minimax_h3/train.py` documents.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # The image VAE stays float32 as released (its encode and decode run as float32 weights under float16
        # autocast), so it is loaded without `torch_dtype`; the mixed-precision loader mixin restores the pinned
        # fp32 modules anyway.
        vae = AutoencoderKLMiniMaxH3Image.from_pretrained(
            component_folder(args.pretrained_model_name_or_path, "vae_diffusers", "vae"), low_cpu_mem_usage=True,
        )
        tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(args.pretrained_model_name_or_path, "tokenizer"))
        processor = Qwen3VLProcessor.from_pretrained(os.path.join(args.pretrained_model_name_or_path, "processor"))
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, "text_encoder"), low_cpu_mem_usage=True, torch_dtype=weight_dtype,
        )
        text_encoder = text_encoder.eval()
    scheduler = MiniMaxH3Scheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Freeze the VAE and the conditioner; the transformer's trainable parameters are selected below through
    # `--trainable_modules`.
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
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

    transformer.train()
    if accelerator.is_main_process:
        accelerator.print(
            f"Trainable modules '{args.trainable_modules}'."
        )
    for name, param in transformer.named_parameters():
        for trainable_module_name in args.trainable_modules + args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    # Create EMA for the transformer.
    if args.use_ema:
        if zero_stage == 3:
            raise NotImplementedError("DeepSpeed Zero-3 does not support EMA.")

        ema_module = MiniMaxH3ImageTransformer2DModel.from_pretrained(
            resolve_image_transformer_component(args.pretrained_model_name_or_path),
            low_cpu_mem_usage=True, output_head_type=args.output_head_type,
        )
        if args.use_fsdp:
            # The EMA copy gets the same FSDP wrap as the live model so that
            # every local shard of the copy pairs 1:1 with the live shard.
            ema_transformer = FSDPEMA(ema_module, source=transformer, accelerator=accelerator, fsdp_plugin=fsdp_plugin)
        else:
            ema_module = ema_module.to(weight_dtype)
            ema_transformer = EMAModel(ema_module.parameters(), model_cls=MiniMaxH3ImageTransformer2DModel, model_config=ema_module.config)

    # ------------------------------------------------------------------ save / load hooks
    # `accelerate` 0.16.0+ supports custom saving hooks; the full transformer is serialized in the diffusers
    # layout (`transformer/`) so that `predict_t2i.py` loads it back with `--transformer_path` pointing at that
    # folder, with no hand-rolled key mapping.
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        if fsdp_stage != 0 or zero_stage == 3:
            def save_model_hook(models, weights, output_dir):
                if getattr(accelerator, "is_fsdp2", False):
                    # `accelerator.get_state_dict` gathers the FSDP2 full state dict on the GPU, which OOMs on a
                    # model this large at steady-state occupancy; offload the gather straight to CPU instead.
                    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
                    accelerate_state_dict = get_model_state_dict(
                        models[-1], options=StateDictOptions(full_state_dict=True, cpu_offload=True)
                    )
                else:
                    accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    from safetensors.torch import save_file

                    save_directory = os.path.join(output_dir, "transformer")
                    os.makedirs(save_directory, exist_ok=True)
                    safetensor_save_path = os.path.join(save_directory, f"diffusion_pytorch_model.safetensors")
                    # MiniMax-H3 ships a mixed-precision checkpoint, and the modules the model pins in float32
                    # (`_keep_in_fp32_modules`: the patch projection, the timestep MLP, the rope and the output
                    # heads) have to stay float32 here — casting them to `weight_dtype` would quantize exactly the
                    # weights whose precision the released model depends on.
                    fp32_patterns = MiniMaxH3ImageTransformer2DModel._keep_in_fp32_modules
                    accelerate_state_dict = {
                        k: v.to(dtype=torch.float32 if any(p in k for p in fp32_patterns) else weight_dtype)
                        for k, v in accelerate_state_dict.items()
                    }
                    save_file(accelerate_state_dict, safetensor_save_path, metadata={"format": "pt"})
                    accelerator.unwrap_model(models[-1]).save_config(save_directory)

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)
                if args.use_ema:
                    # Every rank joins the FULL_STATE_DICT all-gather inside.
                    ema_transformer.save_pretrained(os.path.join(output_dir, "transformer_ema"))

            def load_model_hook(models, input_dir):
                if args.use_ema:
                    ema_transformer.load_pretrained(os.path.join(input_dir, "transformer_ema"))
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
                        ema_transformer.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                    models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                    if not args.use_deepspeed:
                        weights.pop()

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

            def load_model_hook(models, input_dir):
                if args.use_ema:
                    ema_path = os.path.join(input_dir, "transformer_ema")
                    _, ema_kwargs = MiniMaxH3ImageTransformer2DModel.load_config(ema_path, return_unused_kwargs=True)
                    load_model = MiniMaxH3ImageTransformer2DModel.from_pretrained(
                        input_dir, subfolder="transformer_ema",
                        low_cpu_mem_usage=True, output_head_type=args.output_head_type,
                    )
                    load_model = EMAModel(load_model.parameters(), model_cls=MiniMaxH3ImageTransformer2DModel, model_config=load_model.config)
                    load_model.load_state_dict(ema_kwargs)

                    ema_transformer.load_state_dict(load_model.state_dict())
                    ema_transformer.to(accelerator.device)
                    del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = MiniMaxH3ImageTransformer2DModel.from_pretrained(
                        input_dir, subfolder="transformer",
                        low_cpu_mem_usage=True, output_head_type=args.output_head_type,
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
        transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    trainable_params_optim = build_trainable_optimizer_groups(
        transformer.named_parameters(), args, accelerator
    )
    num_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable: {len(trainable_params)} tensors, {num_trainable / 1e6:.2f} M parameters.")

    # ------------------------------------------------------------------ optimizer
    optimizer = create_optimizer(trainable_params_optim, args)

    # ------------------------------------------------------------------ data
    # `t2i` reads the `file_path` / `text` pairs of an image meta; `edit` additionally reads `source_file_path`,
    # the reference image(s) of the request, which `ImageEditDataset` hands over unresized — a reference is encoded
    # at its own resolution, so the collate must not touch it. Both datasets drop the text of 10% of the samples,
    # which is what keeps the unconditional branch alive in a guidance-distilled checkpoint.
    if args.train_mode == "edit":
        train_dataset = ImageEditDataset(
            args.train_data_meta, args.train_data_dir,
            image_sample_size=args.image_sample_size, enable_bucket=args.enable_bucket,
        )
    else:
        train_dataset = ImageVideoDataset(
            args.train_data_meta, args.train_data_dir,
            image_sample_size=args.image_sample_size, enable_bucket=args.enable_bucket,
        )

    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            print(f"worker_init_fn with {_seed + worker_id}")
            np.random.seed(_seed + worker_id)
            random.seed(_seed + worker_id)
        return _worker_init_fn

    aspect_ratio_sample_size = {key : [x / 512 * args.image_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
    batch_sampler_generator = torch.Generator().manual_seed(args.seed)
    batch_sampler = AspectRatioBatchImageVideoSampler(
        sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset,
        batch_size=args.train_batch_size, train_folder=args.train_data_dir, drop_last=True,
        aspect_ratios=aspect_ratio_sample_size,
    )

    def collate_fn(examples):
        def get_random_downsample_ratio(sample_size, image_ratio=[]):
            def _create_special_list(length):
                if length == 1:
                    return [1.0]
                first_element = 0.90
                remaining_sum = 1.0 - first_element
                return [first_element] + [remaining_sum / (length - 1)] * (length - 1)

            if sample_size >= 1536:
                number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio
            elif sample_size >= 1024:
                number_list = [1, 1.25, 1.5, 2] + image_ratio
            elif sample_size >= 768:
                number_list = [1, 1.25, 1.5] + image_ratio
            else:
                number_list = [1] + image_ratio

            number_list_prob = np.array(_create_special_list(len(number_list)))
            return np.random.choice(number_list, p=number_list_prob)

        # Create new output
        new_examples                 = {}
        new_examples["pixel_values"] = []
        new_examples["text"]         = []
        new_examples["source_pixel_values"] = []

        # Get downsample ratio in image
        pixel_value = examples[0]["pixel_values"]
        f, h, w, c  = np.shape(pixel_value)

        random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(args.image_sample_size)
        aspect_ratio_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}

        if args.fix_sample_size is not None:
            closest_size = args.fix_sample_size
        else:
            closest_size, _ = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
            # The canvas has to survive the VAE's 16x compression and still hold whole 2x2 patches.
            closest_size = [int(x / canvas_multiple) * canvas_multiple for x in closest_size]

        for example in examples:
            # To 0~1
            pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.

            # Get adapt hw for resize. No `Normalize` here: MiniMax-H3's encoder is fed [0, 1] pixels and applies
            # its own ImageNet mean / std, unlike the [0.5, 0.5] image families.
            closest_size = list(map(lambda x: int(x), closest_size))
            if closest_size[0] / h > closest_size[1] / w:
                resize_size = closest_size[0], int(w * closest_size[0] / h)
            else:
                resize_size = int(h * closest_size[1] / w), closest_size[1]

            transform = transforms.Compose([
                transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(closest_size),
            ])

            new_examples["pixel_values"].append(transform(pixel_values))
            new_examples["text"].append(example["text"])
            # The references keep their own resolution, so they pass through as the uint8 arrays the dataset read.
            new_examples["source_pixel_values"].append(example.get("source_pixel_values", []))

        # Limit the number of frames to the same (an image is one frame, so the batch stacks directly)
        new_examples["pixel_values"] = torch.stack([example for example in new_examples["pixel_values"]])
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
    # Preserve pretrained FP32 modules; Accelerator handles FSDP mixed dtypes.

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Shard the frozen text encoder *after* prepare (mirrors `scripts/minimax_h3/train.py`): the Qwen3-VL conditioner
    # is wrapped per decoder layer so the per-step unshard footprint stays small, and a post-prepare shard keeps the
    # text encoder out of the trainable FSDP unit.
    if fsdp_stage != 0 or zero_stage != 0:
        from videox_fun.dist import shard_model
        text_encoder.model = shard_model(
            text_encoder.model,
            device_id=accelerator.device,
            param_dtype=weight_dtype,
            module_to_wrapper=list(text_encoder.model.language_model.layers),
        )

    # Move the frozen models to the GPU (or CPU under `low_vram`) and cast to `weight_dtype`; an FSDP-sharded text
    # encoder is already on-device and dtype-pinned by `shard_model`, so it is left untouched.
    device = accelerator.device
    # The VAE stays float32 (mirrors the pipeline: float32 weights, float16 autocast only at the encode call site),
    # so it is moved without a dtype cast.
    vae.to(device if not args.low_vram else "cpu")
    transformer.to(device)
    text_encoder.to(device if not args.low_vram else "cpu", dtype=weight_dtype)

    # The master weights decide whether the run trains at all, so make the two facts that matter visible in the log:
    # the dtype must be float32 under mixed precision, and the per-rank parameter count must be the full count
    # divided by the world size (an unsharded count means the FlatParameters were materialized on every rank).
    if accelerator.is_main_process:
        master_dtypes = {parameter.dtype for parameter in transformer.parameters()}
        num_local_params = sum(parameter.numel() for parameter in transformer.parameters())
        logger.info(
            f"Master parameter dtype(s): {master_dtypes}, {num_local_params / 1e9:.2f} B parameters per rank "
            f"over {accelerator.num_processes} process(es)."
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config = {k: v for k, v in tracker_config.items() if not isinstance(v, list)}
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # ------------------------------------------------------------------ constants
    # Read the transformer config through the unwrap so it works under FSDP (the prepared `transformer` is a
    # sharded wrapper) as well as single-process.
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    latents_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1)
    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1)
    patch_size = tuple(unwrapped_transformer.config.patch_size)
    vae_stride = vae.spatial_compression_ratio
    head_grid = None if args.output_head_type == "single" else head_timestep_grid(int(unwrapped_transformer.config.num_pdd_heads))
    train_generator = torch.Generator(device="cpu")
    if args.seed is not None:
        train_generator.manual_seed(args.seed + accelerator.process_index)
    else:
        train_generator.seed()
    logger.info(f"Train generator seed: {train_generator.initial_seed()} (rank {accelerator.process_index})",
                main_process_only=False)

    # The presentation of a request — the text rows with the references' vision blocks in them, their modality tags
    # and the references' own conditioning latents — comes from the inference pipeline, so the seed-42 /
    # float16-rounding contract of the released model is stated once. The object holds the prepared transformer, so
    # it is a namespace for the two encode methods and never a second sampling loop.
    encoder = MiniMaxH3ImagePipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=transformer,
        scheduler=scheduler,
    )

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # ------------------------------------------------------------------ train loop
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Train mode = {args.train_mode}, canvas = {args.image_sample_size}, sigma shift = {args.sigma_shift}")

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

            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(checkpoint_folder_path)
    else:
        initial_global_step = 0

    progress_bar = PauseAwareTqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    needs_offload_grad_reset = reset_fsdp_offload_grads

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            # Sanity check: save the first batch so a glance at output_dir/sanity_check confirms the data pipe.
            if epoch == first_epoch and step == 0:
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(batch["pixel_values"].cpu(), batch["text"])):
                    pixel_value = pixel_value[0].permute(1, 2, 0).numpy()
                    gif_name = "-".join(text.replace("/", "").split()[:10]) if text else f"{global_step}-{idx}"
                    Image.fromarray((pixel_value * 255).clip(0, 255).astype(np.uint8)).save(
                        f"{args.output_dir}/sanity_check/{gif_name[:10]}.png"
                    )

            with accelerator.accumulate(transformer):
                # Batch-level training (bs=1): the packed-sequence layout (text tokens + reference rows + image rows)
                # is per-sample, so the batch *is* the sample and the encode / noise / forward / loss needs no sample
                # loop. This mirrors `scripts/minimax_h3/train.py` with the audio rows gone.
                pixel_values = batch["pixel_values"][0].to(device)
                text = batch["text"][0]

                # The 33 B transformer alone fills most of a card, so under `low_vram` it yields the GPU while the
                # VAE and the conditioner encode, and moves back for the forward / backward.
                if args.low_vram:
                    transformer.to("cpu")
                    torch.cuda.empty_cache()

                # ---- encode (per-sample, bs=1 so batch-level) ----
                _, _, height, width = pixel_values.shape
                latent_height = height // vae_stride
                latent_width = width // vae_stride

                # Put the references on the resolution the released recipe encodes them at (their own short edge),
                # ahead of everything else: both the conditioner's vision blocks and the VAE rows read them there.
                references = []
                if args.train_mode == "edit":
                    for source in batch["source_pixel_values"][0]:
                        source = Image.fromarray(np.asarray(source).astype(np.uint8)).convert("RGB")
                        references.append(normalize_reference_image(source))

                # MiniMax-H3's encoder input is `[0, 1]` pixels ImageNet-normalized; the VAE stays float32 and the
                # encode runs under float16 autocast, mirroring the decode recipe. The image VAE takes `(B, C, H, W)`
                # and returns a `(B, C_latent, h, w)` distribution — the frame axis the video VAE carries does not
                # exist yet, so it is added after the normalization.
                pixels = (pixel_values - pixel_mean) / pixel_std

                if args.low_vram:
                    vae.to(device)

                # Encode in `vae_mini_batch` mini batches, mirroring `_batch_encode_vae` in ltx2.3.
                def _batch_encode_vae(pixels):
                    bs = args.vae_mini_batch
                    new_pixel_values = []
                    for i in range(0, pixels.shape[0], bs):
                        pixels_bs = pixels[i : i + bs]
                        posterior = vae.encode(pixels_bs.float()).latent_dist
                        latents_bs = posterior.sample()
                        new_pixel_values.append((latents_bs.float() - latents_mean) / latents_std)
                    return torch.cat(new_pixel_values, dim=0)

                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    target_latents = _batch_encode_vae(pixels)[:, :, None]
                if args.low_vram:
                    target_latents = target_latents.cpu()

                # The conditioner reads `hidden_states[50]` of Qwen3-VL; an `edit` request carries one vision block
                # per reference ahead of the prompt. An FSDP-sharded text encoder tolerates symmetric `.to` moves, so
                # it is brought on-device right before the encode and back to CPU afterwards.
                if args.low_vram:
                    text_encoder.to(device)
                with torch.no_grad():
                    prompt_embeds, text_token_tags = encoder.encode_prompt(
                        text, references, device=device, dtype=weight_dtype
                    )
                    if references:
                        condition_latents, reference_grids = encoder.encode_reference_images(references, device=device)
                    else:
                        condition_latents, reference_grids = [], []
                if args.low_vram:
                    text_encoder.to("cpu")
                    prompt_embeds = prompt_embeds.cpu()

                # The references also reach the transformer as VAE rows. They are encoded by the pipeline above on
                # their own resolutions, so the VAE can leave now — under `low_vram` it was on GPU for both encodes.
                if args.low_vram:
                    vae.to("cpu")
                    torch.cuda.empty_cache()

                if args.low_vram:
                    transformer.to(device)
                    if needs_offload_grad_reset:
                        # Moving FSDP1 can restore stale flat-gradient views even
                        # after optimizer.zero_grad(); clear them on the new device.
                        transformer.zero_grad(set_to_none=True)
                        needs_offload_grad_reset = False
                    # Move encoded latents back to GPU for noising and forward.
                    target_latents = target_latents.to(device)
                    prompt_embeds = prompt_embeds.to(device)
                    condition_latents = [c.to(device) for c in condition_latents]

                # ---- noising ----
                # 1. Rows: target `x0`, the noise, the noised `x_t` and the regression target `x0 - noise`.
                x0_rows = patchify_video_latents(target_latents, patch_size)
                noise = torch.randn(
                    target_latents.shape, generator=train_generator, device="cpu", dtype=torch.float32
                ).to(device)
                noise_rows = patchify_video_latents(noise, patch_size)

                # 2. A time on the image schedule, `t = 1 - sigma`; `--sigma_shift` 1.0 keeps the released recipe,
                # which samples the checkpoint's six sigmas verbatim.
                if not args.uniform_sampling:
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=1,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    sigma = u.to(device).squeeze()
                else:
                    # Sample a random timestep for each image
                    sigma = torch.rand((), generator=train_generator, device="cpu", dtype=torch.float32).to(device)
                t = 1.0 - shifted_sigma(args.sigma_shift, sigma)
                xt_rows = t * x0_rows + (1.0 - t) * noise_rows
                target_rows = x0_rows - noise_rows

                # 3. Prepend the conditioning rows, noise-augmented and pinned at their augmentation level.
                num_condition_rows = 0
                condition_timestep = float(t)
                if condition_latents:
                    condition_rows = ref2va_condition_rows(
                        scheduler, condition_latents, patch_size,
                        generator=train_generator, device=device,
                    )
                    xt_rows = torch.cat([condition_rows, xt_rows])
                    num_condition_rows = condition_rows.shape[0]
                    condition_timestep = max(float(t), MINIMAX_H3_KEYFRAME_NOISE_AUG)

                # 4. The packed layout and its per-row timestep plan. There is no audio branch, so the audio clock
                # of `build_row_timesteps` is idle and the layout it returns holds an empty audio block.
                layout = build_image_packed_sequence(
                    text_token_tags, reference_grids, latent_height, latent_width, patch_size
                )
                if num_condition_rows != layout.num_condition_video_rows:
                    raise ValueError(
                        f"The layout reserved {layout.num_condition_video_rows} reference rows but the encoded "
                        f"references pack into {num_condition_rows}. The references the layout was built from and "
                        "the ones the conditioning was encoded from do not agree."
                    )
                unique_timesteps, timestep_indices = build_row_timesteps(
                    layout, float(t), 0.0, condition_timestep, 1.0
                )

                # 5. One forward over the packed sequence, through the head the sampler would use at this `t`. The
                # transformer aligns every input with the dtype of its projection itself (the patch projection is
                # float32 in the checkpoint), so no autocast.
                image_output = transformer(
                    hidden_states=xt_rows[None],
                    encoder_hidden_states=prompt_embeds,
                    timestep=unique_timesteps.to(device),
                    timestep_indices=timestep_indices.to(device),
                    token_tags=layout.token_tags.to(device),
                    position_ids=layout.position_ids.to(device),
                    image_indices=layout.video_indices.to(device),
                    text_indices=layout.text_indices.to(device),
                    pdd_step_index=0 if head_grid is None else pdd_head_index(head_grid, float(t)),
                    return_dict=False,
                )[0]

                # 6. MSE on the generated rows alone, in float32: the conditioning rows are re-imposed by
                # construction and never supervised.
                loss = F.mse_loss(
                    image_output[0, num_condition_rows:].float(), target_rows.float(), reduction="mean"
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_deepspeed and not args.use_fsdp:
                        trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                        trainable_params_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                        max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                        if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                            actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                        else:
                            actual_max_grad_norm = max_grad_norm
                    else:
                        actual_max_grad_norm = args.max_grad_norm

                    if not args.use_deepspeed and not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        if trainable_params_total_norm > 1 and global_step > args.abnormal_norm_clip_start:
                            for name, param in transformer.named_parameters():
                                if param.requires_grad:
                                    writer.add_scalar(f'gradients/before_clip_norm/{name}', param.grad.norm(), global_step=global_step)

                    if getattr(accelerator, "is_fsdp2", False):
                        # `accelerator.clip_grad_norm_` compares the passed parameters against `model.parameters()`
                        # with `==`, which dispatches `aten.eq` on the FSDP2 DTensor parameters and crashes; the
                        # FSDP2 branch of that check would call the same vanilla clip anyway.
                        norm_sum = torch.nn.utils.clip_grad_norm_(trainable_params, actual_max_grad_norm)
                    elif fsdp_plugin is not None:
                        # Only the root FSDP wrapper can reconstruct the global
                        # norm from its sharded trainable gradients.
                        accelerator.unscale_gradients()
                        norm_sum = transformer.clip_grad_norm_(actual_max_grad_norm)
                    else:
                        norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)
                    if not args.use_deepspeed and not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        writer.add_scalar(f'gradients/norm_sum', norm_sum, global_step=global_step)
                        writer.add_scalar(f'gradients/actual_max_grad_norm', actual_max_grad_norm, global_step=global_step)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if reset_fsdp_offload_grads and accelerator.sync_gradients:
                    needs_offload_grad_reset = True
                del target_latents, x0_rows, noise_rows, xt_rows, layout, image_output
                del prompt_embeds, text_token_tags, condition_latents, reference_grids, pixel_values, pixels

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_transformer.step(transformer.parameters())
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
                                    f"{len(checkpoints)} checkpoints already exist, scheduling {len(removing_checkpoints)} for cleanup after save"
                                )
                                logger.info(f"scheduling checkpoint cleanup: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    accelerator.defer_checkpoint_removal(removing_checkpoint)

                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # Keep the checkpoint out of the progress bar rate: a minute-long save would
                        # otherwise land in the next step's interval and be shown as a slow step. The
                        # save also stages the whole state in host RAM (safetensors materializes every
                        # tensor as bytes) and leaves the freed blocks in the allocator caches, so the
                        # cache flushes run inside the same window.
                        with progress_bar.paused():
                            accelerator.save_state(save_path)
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        logger.info(f"Saved state to {save_path}")

                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    with progress_bar.paused():
                        if args.use_ema:
                            # Store the transformer parameters temporarily and load the EMA parameters to perform inference.
                            ema_transformer.store(transformer.parameters())
                            ema_transformer.copy_to(transformer.parameters())
                        log_validation(
                            vae, text_encoder, tokenizer, processor, transformer,
                            scheduler, args, accelerator, weight_dtype, global_step,
                        )
                        if args.use_ema:
                            # Switch back to the original transformer parameters.
                            ema_transformer.restore(transformer.parameters())

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            with progress_bar.paused():
                if args.use_ema:
                    # Store the transformer parameters temporarily and load the EMA parameters to perform inference.
                    ema_transformer.store(transformer.parameters())
                    ema_transformer.copy_to(transformer.parameters())
                log_validation(
                    vae, text_encoder, tokenizer, processor, transformer,
                    scheduler, args, accelerator, weight_dtype, global_step,
                )
                if args.use_ema:
                    # Switch back to the original transformer parameters.
                    ema_transformer.restore(transformer.parameters())

        if global_step >= args.max_train_steps:
            break

    # Close the bar before the end-of-run checkpoint: tqdm keeps redrawing a live bar whenever
    # something else writes to the console. PauseAwareTqdm.close() rebases the closing line onto
    # the smoothed rate, so the worker warm-up and the first dataloader fetch do not dilute it.
    progress_bar.close()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if args.use_ema and args.use_fsdp:
        # Under FSDP every rank must write its own shards, and the shards only
        # exist while the model is still wrapped, so this runs before the
        # `unwrap_model` below.
        ema_transformer.copy_to(transformer.parameters())
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        if args.use_ema and not args.use_fsdp:
            ema_transformer.copy_to(transformer.parameters())

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
