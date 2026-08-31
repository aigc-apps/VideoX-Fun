# Modified from scripts/minimax_h3/train_lora.py for Parallel Decoding Distillation (PDD, arXiv 2607.26004).
# Scaffold (parameter set, resume, checkpointing) follows `scripts/minimax_h3/train_lora.py`.
#
# Data-free PDD LoRA of the packed-sequence transformer, covering `fl2va` (FL2VA / t2va layout) and `ref2va`.
# PDD trains a *parallel decoder*: the sampling interval is discretized into `N` intervals grouped into blocks of
# `L`, and one network evaluation predicts the mean velocity of every interval of the next block, so generation
# advances `L` intervals per evaluation (`NFE = N / L`). The student is the teacher's own backbone with the two
# final heads repeated `N` times (`videox_fun/models/minimax_h3_pdd.py`); the loss is a plain MSE onto a
# Runge-Kutta estimate of the teacher's mean velocity — no VSD, no adversarial term, no JVP.
#
# Training is *data-free* (Algorithm 3 of the paper): no target video is ever read. Each rank carries one
# trajectory, rolls it forward with the student's own predictions, and resets to fresh noise and a fresh prompt
# when it reaches the end of the grid. Only cached Qwen3-VL conditioning is needed, which keeps the 62 GB
# conditioner out of the run. `--train_mode=ref2va` additionally consumes cached reference latents.
# FSDP / DeepSpeed follow `scripts/minimax_h3/train_lora.py`: the plugin is read off Accelerator, ZeRO-3
# skips `zero.Init` on the frozen VAEs, FSDP stage 3 / ZeRO-3 resume through `accelerator.save_state`, and the
# student / teacher forwards always go through the prepared wrapper so a sharded 33 B backbone all-gathers.
#
# MiniMax-H3's rectified-flow convention is the *opposite* of Wan's and is reproduced here from
# `MiniMaxH3Scheduler.scale_noise` / `MiniMaxH3Scheduler.step`, the single source of truth:
#   * noising: `x_t = t * x0 + (1 - t) * noise` with `t = 1` clean, `t = 1 - sigma`,
#   * the sigma grid is exponentially shifted, `sigma' = s * sigma / (1 + (s - 1) * sigma)`, `s = 12.0` for video and `3.0` for audio,
#   * the transformer predicts a data-ward velocity, so the regression target is `x0 - noise`.
#
# The checkpoint is guidance-distilled: one forward per step, no unconditional branch.
#
# Usage:
#   accelerate launch --mixed_precision no scripts/minimax_h3_fun/train_pdd_lora.py \
#       --pretrained_model_name_or_path=models/Diffusion_Transformer/MiniMax-H3 \
#       --train_mode=fl2va --prompt_cache=datasets/minimax_h3_pdd_prompt_cache \
#       --output_dir=output_dir_minimax_h3_pdd_lora --gradient_checkpointing --resume_from_checkpoint=latest

import argparse
import gc
import json
import logging
import math
import os
import shutil
import sys
import time
import warnings
from types import SimpleNamespace

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
from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import is_compiled_module
from packaging import version
from tqdm.auto import tqdm
from transformers.utils import ContextManagers

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import (AutoencoderKLMiniMaxH3,
                               AutoencoderKLMiniMaxH3Audio,
                               MiniMaxH3Transformer3DModel)
from videox_fun.models.minimax_h3_pdd import (LoRALinear,
                                              MiniMaxH3ParallelHead, add_lora,
                                              attach_parallel_decoder,
                                              pdd_sampling_plan,
                                              pdd_state_dict,
                                              pdd_teacher_mean_velocity,
                                              pdd_time_grid,
                                              pdd_training_plan,
                                              set_parallel_plan, teacher_mode)
from videox_fun.pipeline import MiniMaxH3Pipeline
from videox_fun.pipeline.pipeline_minimax_h3 import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG, align_num_frames, audio_latent_num_frames,
    build_packed_sequence, build_ref2va_packed_sequence, build_row_timesteps,
    patchify_video_latents, video_latent_num_frames)
from videox_fun.utils import MiniMaxH3Scheduler
from videox_fun.utils.utils import save_videos_with_audio_grid

# Silences diffusers' `randn_tensor` notice about CPU generators producing CUDA tensors (the tensor is created
# on CPU and moved to GPU; harmless, only a marginal speed note).
warnings.filterwarnings("ignore", message="The passed generator was created on")


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    return initial_value + step_size * current_step


def load_cache_split(folder, kind):
    r"""Load a `train/` or `val/` split of cached `.pt` requests, in index order."""
    names = sorted(name for name in os.listdir(folder) if name.endswith(".pt"))
    if not names:
        raise FileNotFoundError(
            f"No cached {kind} entries under {folder}. Encode the Qwen3-VL conditioning first "
            f"(`prompt_embeds` / `text_token_tags`, and for ref2va the reference latents)."
        )
    return [torch.load(os.path.join(folder, name), weights_only=False) for name in names]


class FL2VATrajectory:
    r"""
    One rank's carried trajectory of the data-free PDD algorithm on the FL2VA / t2va layout.

    The state is a partially denoised sample plus the grid index it sits at. A step reads it, rolls it forward by
    `L_min` intervals with the student's own prediction, and the trajectory is thrown away and re-drawn from noise
    (with a new prompt) once it reaches the end of the grid.
    """

    def __init__(self, geometry, patch_size, latent_channels, audio_channels, prompts, rng, device):
        self.geometry = geometry
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.audio_channels = audio_channels
        self.prompts = prompts
        self.rng = rng
        self.device = device
        self.index = None

    def reset(self):
        num_latent_frames, latent_height, latent_width, num_audio_latents = self.geometry
        cached = self.prompts[int(self.rng.integers(len(self.prompts)))]
        self.prompt_embeds = cached["prompt_embeds"].to(self.device)
        text_token_tags = cached["text_token_tags"]
        if not torch.is_tensor(text_token_tags):
            text_token_tags = torch.tensor(text_token_tags, dtype=torch.long)
        self.layout = build_packed_sequence(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            self.patch_size,
        )
        self.indices = {
            name: getattr(self.layout, name).to(self.device)
            for name in ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices")
        }
        rows_per_frame = (latent_height // self.patch_size[1]) * (latent_width // self.patch_size[2])
        video_patch_dim = self.latent_channels * math.prod(self.patch_size)
        self.video = torch.randn(
            num_latent_frames * rows_per_frame, video_patch_dim, device=self.device, dtype=torch.float32
        )
        self.audio = torch.randn(
            num_audio_latents * 2, self.audio_channels, device=self.device, dtype=torch.float32
        )
        self.index = 0

    def forward_kwargs(self, video_time, audio_time):
        unique_timesteps, timestep_indices = build_row_timesteps(
            self.layout, float(video_time), float(audio_time), float(video_time), 1.0
        )
        return dict(
            encoder_hidden_states=self.prompt_embeds,
            timestep=unique_timesteps.to(self.device),
            timestep_indices=timestep_indices.to(self.device),
            return_dict=False,
            **self.indices,
        )

    def generated(self, video, audio):
        return video, audio

    def with_generated(self, video_tail, audio_tail):
        return video_tail, audio_tail


class Ref2VATrajectory:
    r"""
    One rank's carried trajectory of the data-free PDD algorithm, on a `ref2va` layout.

    The state is the two packed streams — each of them the request's fixed conditioning rows followed by the
    partially denoised generated rows — plus the grid index they sit at. The conditioning rows are drawn once per
    trajectory and never move; only the generated tail is rolled forward and supervised.
    """

    def __init__(self, geometry, patch_size, latent_channels, audio_channels, requests, scheduler, rng, device):
        self.geometry = geometry
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.audio_channels = audio_channels
        self.requests = requests
        self.scheduler = scheduler
        self.rng = rng
        self.device = device
        self.index = None

    def reset(self):
        num_latent_frames, latent_height, latent_width, num_audio_latents = self.geometry
        request = self.requests[int(self.rng.integers(len(self.requests)))]
        self.prompt_embeds = request["prompt_embeds"].to(self.device)
        references = [SimpleNamespace(kind=kind, has_audio=has_audio) for kind, has_audio in request["reference_kinds"]]
        condition_latents = request["condition_latents"]
        audio_condition_latents = request["audio_condition_latents"]
        text_token_tags = request["text_token_tags"]
        if not torch.is_tensor(text_token_tags):
            text_token_tags = torch.tensor(text_token_tags, dtype=torch.long)

        self.layout = build_ref2va_packed_sequence(
            text_token_tags,
            references,
            condition_latents,
            audio_condition_latents,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            self.patch_size,
        )
        self.indices = {
            name: getattr(self.layout, name).to(self.device)
            for name in ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices")
        }
        self.num_condition_video_rows = self.layout.num_condition_video_rows
        self.num_condition_audio_rows = self.layout.num_condition_audio_rows

        condition_rows = [
            patchify_video_latents(
                self.scheduler.scale_noise(
                    condition.to(self.device),
                    MINIMAX_H3_KEYFRAME_NOISE_AUG,
                    torch.randn(condition.shape, device=self.device, dtype=torch.float32),
                ),
                self.patch_size,
            )
            for condition in condition_latents
        ]

        rows_per_frame = (latent_height // self.patch_size[1]) * (latent_width // self.patch_size[2])
        video_patch_dim = self.latent_channels * math.prod(self.patch_size)
        video = torch.randn(
            num_latent_frames * rows_per_frame, video_patch_dim, device=self.device, dtype=torch.float32
        )
        audio = torch.randn(num_audio_latents * 2, self.audio_channels, device=self.device, dtype=torch.float32)
        self.video = torch.cat(condition_rows + [video]) if condition_rows else video
        self.audio = (
            torch.cat([rows.to(self.device) for rows in audio_condition_latents] + [audio])
            if audio_condition_latents
            else audio
        )
        self.index = 0

    def forward_kwargs(self, video_time, audio_time):
        unique_timesteps, timestep_indices = build_row_timesteps(
            self.layout,
            float(video_time),
            float(audio_time),
            max(float(video_time), MINIMAX_H3_KEYFRAME_NOISE_AUG),
            1.0,
        )
        return dict(
            encoder_hidden_states=self.prompt_embeds,
            timestep=unique_timesteps.to(self.device),
            timestep_indices=timestep_indices.to(self.device),
            return_dict=False,
            **self.indices,
        )

    def generated(self, video, audio):
        return video[self.num_condition_video_rows :], audio[self.num_condition_audio_rows :]

    def with_generated(self, video_tail, audio_tail):
        video = torch.cat([self.video[: self.num_condition_video_rows], video_tail])
        audio = torch.cat([self.audio[: self.num_condition_audio_rows], audio_tail])
        return video, audio


logger = get_logger(__name__, log_level="INFO")


def log_validation(
    vae, audio_vae, transformer, scheduler, audio_scheduler, args, accelerator, val_cache, grids, global_step,
):
    r"""
    Generate every held-out cache entry with the student at `--validation_nfe`, sharded across ranks, and save
    next to the run. Entries are already cached Qwen3-VL conditioning; this does not load JSON inference jobs.

    PDD generation is an ordinary Euler loop over the *block boundaries* of the grid: those boundaries are exactly
    the schedule `MiniMaxH3Scheduler` builds for `NFE` steps, so the released pipeline drives the student unchanged
    and the only PDD-specific work is arming the heads before each step.
    """
    assigned = [
        (index, entry)
        for index, entry in enumerate(val_cache)
        if index % accelerator.num_processes == accelerator.process_index
    ]
    if not assigned:
        return

    try:
        with torch.no_grad():
            logger.info("Running validation... ")
            _, _, video_steps, audio_steps = grids
            num_steps = video_steps.shape[0]
            student = accelerator.unwrap_model(transformer)
            sharded = (
                getattr(accelerator.state, "fsdp_plugin", None) is not None
                or getattr(accelerator.state, "deepspeed_plugin", None) is not None
            )

            pipeline = MiniMaxH3Pipeline(
                vae=vae,
                audio_vae=audio_vae,
                text_encoder=None,
                tokenizer=None,
                processor=None,
                transformer=transformer,
                scheduler=scheduler,
                audio_scheduler=audio_scheduler,
            )
            # Avoid `.to()` on an FSDP / DeepSpeed wrapper: it rematerializes FlatParameters on every rank.
            if not sharded:
                pipeline = pipeline.to(accelerator.device)
            block_size = num_steps // args.validation_nfe

            def arm(step_index):
                start = step_index * block_size
                set_parallel_plan(
                    student,
                    pdd_sampling_plan(video_steps, start, block_size).float(),
                    pdd_sampling_plan(audio_steps, start, block_size).float(),
                )

            def callback(pipe, step_index, timestep, callback_kwargs):
                if step_index + 1 < args.validation_nfe:
                    arm(step_index + 1)
                return {}

            vae.to(accelerator.device)
            audio_vae.to(accelerator.device)
            os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)

            for index, entry in assigned:
                arm(0)
                if args.seed is None:
                    generator = None
                else:
                    prompt_seed = args.seed + index
                    generator = torch.Generator(device=accelerator.device).manual_seed(prompt_seed)
                    logger.info(f"Rank {accelerator.process_index} prompt {index} using seed: {prompt_seed}")

                call_kwargs = dict(
                    prompt=None,
                    prompt_embeds=entry["prompt_embeds"],
                    text_token_tags=entry["text_token_tags"],
                    height=args.video_sample_height,
                    width=args.video_sample_width,
                    num_frames=args.video_sample_n_frames,
                    num_inference_steps=args.validation_nfe,
                    generator=generator,
                    output_type="pt",
                    callback_on_step_end=callback,
                )
                if args.train_mode == "ref2va":
                    call_kwargs.update(
                        normalized_references=[
                            SimpleNamespace(kind=kind, has_audio=has_audio) for kind, has_audio in entry["reference_kinds"]
                        ],
                        condition_latents=entry["condition_latents"],
                        audio_condition_latents=entry["audio_condition_latents"],
                    )

                output = pipeline(**call_kwargs)
                save_videos_with_audio_grid(
                    output.videos,
                    output.audio,
                    os.path.join(
                        args.output_dir,
                        f"sample/sample-{global_step}-prompt{index}-{args.train_mode}-nfe{args.validation_nfe}.mp4",
                    ),
                    fps=24,
                    audio_sample_rate=output.sampling_rate,
                )
                del output

            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            vae.to(accelerator.device if not args.low_vram else "cpu")
            audio_vae.to(accelerator.device if not args.low_vram else "cpu")
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error on rank {accelerator.process_index} with info {e}")
        vae.to(accelerator.device if not args.low_vram else "cpu")
        audio_vae.to(accelerator.device if not args.low_vram else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel Decoding Distillation LoRA of MiniMax-H3 (FL2VA / Ref2VA, video + audio)."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_dir_minimax_h3_pdd_lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=43, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=3000,
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
        help="Offload the activations saved for backward of the transformer blocks to CPU memory.",
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
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
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
        default=50,
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
    parser.add_argument("--save_state", action="store_true", help="Whether or not to save state.")
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--use_fsdp", action="store_true", help="Whether or not to use fsdp."
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alpha",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--target_name",
        type=str,
        default="to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear",
        help=("The module is trained in loras."),
    )
    # MiniMax-H3 specific
    parser.add_argument(
        "--train_mode",
        type=str,
        default="fl2va",
        choices=["fl2va", "ref2va"],
        help="fl2va: FL2VA / t2va packed layout and `--prompt_cache`. ref2va: Ref2VA layout and `--request_cache`.",
    )
    parser.add_argument(
        "--video_loss_weight",
        type=float,
        default=0.5,
        help="Weight of the video flow-matching loss in the joint video + audio loss.",
    )
    parser.add_argument(
        "--audio_loss_weight",
        type=float,
        default=0.5,
        help="Weight of the audio flow-matching loss in the joint video + audio loss.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=124,
        help="Number of frames (form 17*n+5).",
    )
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="Keep VAE and conditioner on CPU, move to GPU only while encoding.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="minimax_h3_pdd_lora",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help="Run validation every X steps.",
    )
    # PDD
    parser.add_argument(
        "--prompt_cache",
        type=str,
        default=None,
        help="Folder with `train/` and `val/` of cached Qwen3-VL conditioning for `--train_mode=fl2va`.",
    )
    parser.add_argument(
        "--request_cache",
        type=str,
        default=None,
        help="Folder with `train/` and `val/` of cached `ref2va` requests for `--train_mode=ref2va`.",
    )
    parser.add_argument(
        "--transformer_subfolder",
        type=str,
        default=None,
        help="Transformer subfolder. Default: `transformer_ref` for `--train_mode=ref2va`, else `transformer`.",
    )
    parser.add_argument(
        "--pdd_num_steps",
        type=int,
        default=32,
        help="The grid size `N`. The paper uses 128 for video models with the midpoint solver, 256 with Euler.",
    )
    parser.add_argument(
        "--pdd_block_size",
        type=int,
        default=4,
        help="`L_min`: the block the carried state advances by, so the student is trained for `N / L_min` NFE.",
    )
    parser.add_argument(
        "--pdd_max_block_size",
        type=int,
        default=None,
        help="`L_max`: the widest block a loss target is drawn from. Defaults to `--pdd_block_size`.",
    )
    parser.add_argument(
        "--pdd_solver",
        type=str,
        default="midpoint",
        choices=["euler", "midpoint"],
        help="Runge-Kutta method the teacher's mean velocity is estimated with.",
    )
    parser.add_argument(
        "--pdd_num_targets",
        type=int,
        default=2,
        help="How many intra-block indices `k` one student evaluation is supervised at.",
    )
    parser.add_argument("--lora_learning_rate", type=float, default=1e-4, help="Learning rate of the low-rank updates.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Keep an exponential moving average of the trainable set, and validate and checkpoint from it.",
    )
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--abnormal_norm_clip_start", type=int, default=1000)
    parser.add_argument("--initial_grad_norm_ratio", type=int, default=5)
    parser.add_argument("--video_sample_height", type=int, default=704)
    parser.add_argument("--video_sample_width", type=int, default=1280)
    parser.add_argument("--validation_nfe", type=int, default=8)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.pdd_max_block_size is None:
        args.pdd_max_block_size = args.pdd_block_size
    return args


PDD_WEIGHTS_NAME = "pdd.safetensors"
PDD_LIVE_WEIGHTS_NAME = "pdd_live.safetensors"


def filter_pdd_state_dict(transformer, state_dict):
    r"""Apply [`pdd_state_dict`]'s trainable-key filter to a (possibly FSDP-gathered) `state_dict`."""
    trainable = {
        name
        for name, module in transformer.named_modules()
        if isinstance(module, (MiniMaxH3ParallelHead, LoRALinear))
    }
    return {
        name: value.detach().cpu()
        for name, value in state_dict.items()
        if any(name.startswith(f"{prefix}.") for prefix in trainable) and ".base." not in name
    }


def save_pdd_weights(path, state_dict):
    from safetensors.torch import save_file
    save_file(
        {name: tensor.detach().contiguous().cpu() for name, tensor in state_dict.items()},
        path,
        metadata={"format": "pt"},
    )


def dump_pdd_config(args, save_path):
    r"""Write `pdd_config.json` with both this script's LoRA flags and the inference aliases `predict_t2v_args.py` reads."""
    config = dict(vars(args))
    config["lora_rank"] = args.rank
    config["lora_alpha"] = args.network_alpha
    config["lora_targets"] = args.target_name
    with open(os.path.join(save_path, "pdd_config.json"), "w") as handle:
        json.dump(config, handle, indent=1)


def save_resume_state(save_path, student, optimizer, lr_scheduler, ema, accelerator):
    r"""Optimizer / scheduler / live weights / EMA shadow — the pieces `pdd.safetensors` (the EMA export) does not hold."""
    os.makedirs(save_path, exist_ok=True)
    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
    save_pdd_weights(os.path.join(save_path, PDD_LIVE_WEIGHTS_NAME), pdd_state_dict(student))
    if ema is not None:
        torch.save(ema.state_dict(), os.path.join(save_path, "ema.pt"))
    if getattr(accelerator, "scaler", None) is not None:
        torch.save(accelerator.scaler.state_dict(), os.path.join(save_path, "scaler.pt"))


def load_resume_state(save_path, student, optimizer, lr_scheduler, ema, trainable_params, accelerator):
    r"""Load the trainer state written by [`save_resume_state`], same `.pt` / `.bin` lookup as `train_lora.py`."""
    from safetensors.torch import load_file

    weights_path = os.path.join(save_path, PDD_LIVE_WEIGHTS_NAME)
    state_dict = load_file(weights_path, device="cpu")
    m, u = student.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    assert len(u) == 0
    print(f"Loaded {len(state_dict)} PDD tensors from {weights_path}")

    device = accelerator.device
    optimizer_file_pt = os.path.join(save_path, "optimizer.pt")
    optimizer_file_bin = os.path.join(save_path, "optimizer.bin")
    optimizer_file_to_load = None
    if os.path.exists(optimizer_file_pt):
        optimizer_file_to_load = optimizer_file_pt
    elif os.path.exists(optimizer_file_bin):
        optimizer_file_to_load = optimizer_file_bin
    if optimizer_file_to_load:
        try:
            accelerator.print(f"Loading optimizer state from {optimizer_file_to_load}")
            optimizer.load_state_dict(torch.load(optimizer_file_to_load, map_location=device))
            accelerator.print("Optimizer state loaded successfully.")
        except Exception as e:
            accelerator.print(f"Failed to load optimizer state from {optimizer_file_to_load}: {e}")

    scheduler_file_pt = os.path.join(save_path, "scheduler.pt")
    scheduler_file_bin = os.path.join(save_path, "scheduler.bin")
    scheduler_file_to_load = None
    if os.path.exists(scheduler_file_pt):
        scheduler_file_to_load = scheduler_file_pt
    elif os.path.exists(scheduler_file_bin):
        scheduler_file_to_load = scheduler_file_bin
    if scheduler_file_to_load:
        try:
            accelerator.print(f"Loading scheduler state from {scheduler_file_to_load}")
            lr_scheduler.load_state_dict(torch.load(scheduler_file_to_load, map_location=device))
            accelerator.print("Scheduler state loaded successfully.")
        except Exception as e:
            accelerator.print(f"Failed to load scheduler state from {scheduler_file_to_load}: {e}")

    if getattr(accelerator, "scaler", None) is not None:
        scaler_file = os.path.join(save_path, "scaler.pt")
        if os.path.exists(scaler_file):
            try:
                accelerator.print(f"Loading GradScaler state from {scaler_file}")
                accelerator.scaler.load_state_dict(torch.load(scaler_file, map_location=device))
                accelerator.print("GradScaler state loaded successfully.")
            except Exception as e:
                accelerator.print(f"Failed to load GradScaler state: {e}")

    if ema is None:
        return
    ema_path = os.path.join(save_path, "ema.pt")
    if os.path.exists(ema_path):
        try:
            print(f"Loading EMA state from {ema_path}")
            ema.load_state_dict(torch.load(ema_path, map_location="cpu"))
            print("EMA state loaded successfully.")
            return
        except Exception as e:
            print(f"Failed to load EMA state from {ema_path}: {e}")
    ema.shadow_params = [parameter.detach().clone() for parameter in trainable_params]
    print(f"No ema.pt under {save_path}; EMA is re-seeded from the loaded weights.")


def main():
    args = parse_args()

    if args.train_mode not in ("fl2va", "ref2va"):
        raise ValueError(f"`train_mode` must be 'fl2va' or 'ref2va', got {args.train_mode!r}.")
    aligned_frames = align_num_frames(int(args.video_sample_n_frames))
    if aligned_frames != int(args.video_sample_n_frames):
        raise ValueError(
            f"`video_sample_n_frames` has to be of the form 17 * n + 5 the video VAE encodes, got "
            f"{args.video_sample_n_frames} (nearest is {aligned_frames})."
        )
    if args.video_sample_height % 32 or args.video_sample_width % 32:
        raise ValueError(
            f"`video_sample_height` / `video_sample_width` ({args.video_sample_height}x{args.video_sample_width}) "
            "must be multiples of 32: the canvas is patched 2x2 into the transformer and its RoPE grid keys off that."
        )
    if args.pdd_num_steps % args.pdd_block_size:
        raise ValueError(
            f"The grid size {args.pdd_num_steps} must be a multiple of the block size {args.pdd_block_size}: the "
            "block starts of the data-free algorithm are the multiples of `L_min` and the last one has to be the end "
            "of the grid."
        )
    if args.pdd_num_steps % args.validation_nfe:
        raise ValueError(
            f"`--validation_nfe` {args.validation_nfe} must divide the grid size {args.pdd_num_steps}: generation "
            "advances `N / NFE` intervals per evaluation."
        )
    if args.train_mode == "fl2va":
        if not args.prompt_cache:
            raise ValueError("`--train_mode=fl2va` requires `--prompt_cache` with `train/` and `val/` splits.")
    elif args.train_mode == "ref2va":
        if not args.request_cache:
            raise ValueError("`--train_mode=ref2va` requires `--request_cache` with `train/` and `val/` splits.")
    if args.train_batch_size != 1:
        raise ValueError("Data-free PDD carries one trajectory per rank and requires --train_batch_size=1.")

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
        if zero_stage == 3:
            print("Auto set save_state to True because zero_stage == 3")
            args.save_state = True
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
        if fsdp_stage == 3:
            print("Auto set save_state to True because fsdp_stage == 3")
            args.save_state = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("DeepSpeed/FSDP is not enabled.")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    # Per-rank seeding: the ranks of one global batch have to roll out *different* noise, otherwise the trajectories
    # of a step differ only in their prompt.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")
    else:
        rng = np.random.default_rng()
        print(f"Init rng without fixed seed. Process_index is {accelerator.process_index}")

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
    # PDD: the released checkpoint already pins `proj_out` / `audio_proj_out` in float32
    # (`_keep_in_fp32_modules`), so the parallel heads built from them are float32 master weights over a bfloat16
    # backbone. The model casts every input to its projection's dtype itself, so the run needs no autocast.
    weight_dtype = torch.bfloat16

    # ------------------------------------------------------------------ models
    # `pretrained_model_name_or_path` may point at a converted diffusers layout or at an *original* MiniMax-H3
    # partition; every component's `from_pretrained` auto-detects the layout and stream-converts the original
    # shards on the fly, so the caller never branches on the format itself.
    transformer_subfolder = args.transformer_subfolder or (
        "transformer_ref" if args.train_mode == "ref2va" else "transformer"
    )
    print(f"Loading transformer from subfolder `{transformer_subfolder}` (train_mode={args.train_mode}).")
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder=transformer_subfolder, low_cpu_mem_usage=True, torch_dtype=weight_dtype,
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
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained`. So the two VAEs will not enjoy the parameter sharding across multiple gpus
    # and only the transformer will get ZeRO sharded. PDD never loads the 62 GB conditioner.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # The two VAEs stay float32 as released (the encode/decode recipe is float16 autocast over float32
        # weights), so they are loaded without `torch_dtype`; the mixed-precision loader mixin restores the
        # pinned fp32 modules anyway. PDD validation is the only consumer.
        vae = AutoencoderKLMiniMaxH3.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", low_cpu_mem_usage=True,
        )
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="audio_vae", low_cpu_mem_usage=True,
        )
    scheduler = MiniMaxH3Scheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    audio_scheduler = MiniMaxH3Scheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="audio_scheduler")

    # Freeze everything; the LoRA modules and parallel heads created below are the only trainable parameters.
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    audio_vae.requires_grad_(False)

    # ------------------------------------------------------------------ LoRA
    num_adapters = add_lora(transformer, args.target_name.split(","), args.rank, args.network_alpha)
    attach_parallel_decoder(transformer, args.pdd_num_steps)
    transformer.train()
    # FSDP flattens one dtype per wrap unit. DDP keeps float32 LoRA master weights; under FSDP the adapters match
    # the Linear they wrap (bf16) so each `MiniMaxH3TransformerBlock` is uniform. The parallel heads stay float32
    # and are wrapped as their own units. Frozen `_keep_in_fp32_modules` embeddings are ignored so they are not
    # mixed into the bf16 root flatten (`--mixed_precision=no` does not install an FSDP MixedPrecision policy).
    if fsdp_plugin is not None:
        for module in transformer.modules():
            if isinstance(module, LoRALinear):
                dtype = module.base.weight.dtype
                module.lora_down.data = module.lora_down.data.to(dtype)
                module.lora_up.data = module.lora_up.data.to(dtype)
        wrap_names = list(fsdp_plugin.transformer_cls_names_to_wrap or [])
        if "MiniMaxH3ParallelHead" not in wrap_names:
            wrap_names.append("MiniMaxH3ParallelHead")
            fsdp_plugin.transformer_cls_names_to_wrap = wrap_names
        ignored = []
        for name in ("proj_in", "audio_proj_in", "time_embedder", "rope"):
            module = getattr(transformer, name, None)
            if isinstance(module, torch.nn.Module):
                # `sync_module_states=True` rejects CPU params on ignored modules; FSDP's `device_id` only
                # moves the flattened units.
                module.to(accelerator.device)
                ignored.append(module)
        fsdp_plugin.ignored_modules = ignored
        logger.info(
            "FSDP: LoRA adapters cast to the backbone dtype; wrap %s; ignored_modules=%s.",
            wrap_names,
            [module.__class__.__name__ for module in ignored],
        )

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

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # ------------------------------------------------------------------ save / load hooks
    # `accelerate` 0.16.0+ supports custom saving hooks. Under FSDP / ZeRO-3 the hook writes
    # `pdd_live.safetensors` from the gathered trainable tensors so DDP `--save_state` resume can reload
    # live weights; popping `weights` on the DDP path keeps `save_state` from serializing the frozen backbone.
    # The EMA / inference export `pdd.safetensors` is written after `ema.copy_to`.
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        if fsdp_stage != 0 or zero_stage == 3:
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process and accelerate_state_dict is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    save_pdd_weights(
                        os.path.join(output_dir, PDD_LIVE_WEIGHTS_NAME),
                        filter_pdd_state_dict(unwrap_model(models[-1]), accelerate_state_dict),
                    )
                    dump_pdd_config(args, output_dir)

            def load_model_hook(models, input_dir):
                return

        else:
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process and accelerate_state_dict is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    save_pdd_weights(
                        os.path.join(output_dir, PDD_LIVE_WEIGHTS_NAME),
                        filter_pdd_state_dict(unwrap_model(models[-1]), accelerate_state_dict),
                    )
                    dump_pdd_config(args, output_dir)
                    if not args.use_deepspeed:
                        for _ in range(len(weights)):
                            weights.pop()

            def load_model_hook(models, input_dir):
                return

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        lr_scale = args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        args.learning_rate = args.learning_rate * lr_scale
        args.lora_learning_rate = args.lora_learning_rate * lr_scale

    head_params, lora_params = [], []
    for name, parameter in transformer.named_parameters():
        if not parameter.requires_grad:
            continue
        (head_params if "proj_out" in name else lora_params).append(parameter)
    trainable_params = head_params + lora_params
    logger.info(
        f"LoRA created: {num_adapters} adapters, {sum(p.numel() for p in lora_params) / 1e6:.2f} M parameters; "
        f"{len(head_params)} parallel head tensors, {sum(p.numel() for p in head_params) / 1e6:.2f} M parameters."
    )

    # ------------------------------------------------------------------ optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        [
            {"params": head_params, "lr": args.learning_rate},
            {"params": lora_params, "lr": args.lora_learning_rate},
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ------------------------------------------------------------------ data
    # Data-free PDD never reads a target video: each rank carries one trajectory and only cached Qwen3-VL
    # conditioning (plus encoded reference latents under `ref2va`) is needed.
    cache_root = args.request_cache if args.train_mode == "ref2va" else args.prompt_cache
    train_cache = load_cache_split(os.path.join(cache_root, "train"), args.train_mode)
    val_cache = load_cache_split(os.path.join(cache_root, "val"), args.train_mode)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer.gradient_checkpointing_save_on_cpu = args.gradient_checkpointing_save_on_cpu
    transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)

    device = accelerator.device
    # The two VAEs stay float32 (mirrors the pipeline: float32 weights, float16 autocast only at the
    # encode/decode call site), so they are moved without a dtype cast. Under `--low_vram` they live on
    # CPU and only come on-device inside `log_validation`, the same residency as `train_lora.py`.
    # Under FSDP a `.to()` on the wrapped transformer rematerializes FlatParameters on every rank, so the
    # sharded model is left where `prepare` put it.
    vae.to(device if not args.low_vram else "cpu")
    audio_vae.to(device if not args.low_vram else "cpu")
    if fsdp_stage == 0 and zero_stage == 0:
        transformer.to(device)

    trainable_params = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    ema = (
        EMAModel(trainable_params, decay=args.ema_decay, use_ema_warmup=False, foreach=True)
        if args.use_ema
        else None
    )

    if accelerator.is_main_process:
        master_dtypes = {parameter.dtype for parameter in transformer.parameters()}
        num_local_params = sum(parameter.numel() for parameter in transformer.parameters())
        logger.info(
            f"Master parameter dtype(s): {master_dtypes}, {num_local_params / 1e9:.2f} B parameters per rank "
            f"over {accelerator.num_processes} process(es)."
        )

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config = {k: v for k, v in tracker_config.items() if not isinstance(v, list)}
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # ------------------------------------------------------------------ constants
    # Read the transformer config through the unwrap so it works under FSDP (the prepared `transformer` is a
    # sharded wrapper) as well as single-process.
    student = unwrap_model(transformer)
    patch_size = tuple(student.config.patch_size)
    latent_channels = student.config.in_channels
    audio_channels = student.config.audio_in_channels
    geometry = (
        video_latent_num_frames(args.video_sample_n_frames),
        args.video_sample_height // vae.spatial_compression_ratio,
        args.video_sample_width // vae.spatial_compression_ratio,
        audio_latent_num_frames(args.video_sample_n_frames),
    )
    video_grid = pdd_time_grid(scheduler.shift, args.pdd_num_steps)
    audio_grid = pdd_time_grid(audio_scheduler.shift, args.pdd_num_steps)
    grids = (video_grid, audio_grid, video_grid.diff(), audio_grid.diff())
    logger.info(
        f"Grid N={args.pdd_num_steps}, L_min={args.pdd_block_size}, L_max={args.pdd_max_block_size}: block starts at "
        f"t = {[round(float(video_grid[i]), 4) for i in range(0, args.pdd_num_steps + 1, args.pdd_block_size)]}"
    )
    if args.train_mode == "ref2va":
        trajectory = Ref2VATrajectory(
            geometry, patch_size, latent_channels, audio_channels, train_cache, scheduler, rng, device,
        )
    else:
        trajectory = FL2VATrajectory(
            geometry, patch_size, latent_channels, audio_channels, train_cache, rng, device,
        )
    target_seed = (args.seed if args.seed is not None else 0) + 1000 + accelerator.process_index
    target_rng = np.random.default_rng(np.random.PCG64(target_seed))

    # ------------------------------------------------------------------ train loop
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_cache)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Video / audio loss weights = {args.video_loss_weight} / {args.audio_loss_weight}")

    global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir) if os.path.isdir(args.output_dir) else []
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

            if args.resume_from_checkpoint != "latest" and os.path.isdir(args.resume_from_checkpoint):
                checkpoint_folder_path = args.resume_from_checkpoint
            else:
                checkpoint_folder_path = os.path.join(args.output_dir, path)
            if zero_stage != 3 and not args.use_fsdp:
                load_resume_state(
                    checkpoint_folder_path, student, optimizer, lr_scheduler, ema, trainable_params, accelerator
                )
            else:
                accelerator.load_state(checkpoint_folder_path)
                accelerator.print("accelerator.load_state() completed for FSDP / ZeRO stage 3.")
                if ema is not None:
                    ema.shadow_params = [parameter.detach().clone() for parameter in trainable_params]
                    print(f"EMA is re-seeded from the loaded FSDP / ZeRO weights under {checkpoint_folder_path}.")
            print(f"Resumed training from {checkpoint_folder_path} at step {global_step}.")
    else:
        initial_global_step = 0

    if ema is not None:
        ema.to(device)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    train_loss = 0.0
    train_video_loss = 0.0
    train_audio_loss = 0.0
    step_started = time.time()

    while global_step < args.max_train_steps:
        with accelerator.accumulate(transformer):
            if trajectory.index is None or trajectory.index >= args.pdd_num_steps:
                trajectory.reset()
            start = trajectory.index

            # Sample the intra-block indices the loss is evaluated at, `k ~ U{n, ..., min(n + L_max, N) - 1}`, without
            # replacement so that several targets always supervise several distinct heads.
            reach = min(start + args.pdd_max_block_size, args.pdd_num_steps)
            targets = sorted(
                target_rng.choice(
                    np.arange(start, reach), size=min(args.pdd_num_targets, reach - start), replace=False
                ).tolist()
            )

            # One student evaluation yields, per target, the displacement to `X_k` and the velocity `u_k` the loss
            # regresses, plus the `L_min` advance of the carried state (the paper's layer fusion, §3.1).
            set_parallel_plan(
                student,
                pdd_training_plan(grids[2], start, targets, args.pdd_block_size).float(),
                pdd_training_plan(grids[3], start, targets, args.pdd_block_size).float(),
            )
            video_output, audio_output = transformer(
                hidden_states=trajectory.video[None],
                audio_hidden_states=trajectory.audio[None],
                **trajectory.forward_kwargs(video_grid[start], audio_grid[start]),
            )
            # The heads run over every row and the modality rows are selected afterwards, so a ref2va output still
            # carries the conditioning rows in front. Only the generated tail is rolled forward and supervised.
            video_output = video_output[0].unflatten(-1, (-1, latent_channels * math.prod(patch_size)))
            audio_output = audio_output[0].unflatten(-1, (-1, audio_channels))
            video_output, audio_output = trajectory.generated(video_output, audio_output)
            state_video_tail, state_audio_tail = trajectory.generated(trajectory.video, trajectory.audio)

            video_loss = video_output.new_zeros(())
            audio_loss = audio_output.new_zeros(())
            for position, target in enumerate(targets):
                # The teacher's mean velocity is estimated on the student's own intra-block state (on-policy), and
                # the state is a constant of the loss (eq. 11's stop-gradient). Conditioning rows are put back in
                # front of the generated tail on ref2va; fl2va has no conditioning rows.
                state_video, state_audio = trajectory.with_generated(
                    state_video_tail + video_output[:, 2 * position].detach(),
                    state_audio_tail + audio_output[:, 2 * position].detach(),
                )
                with teacher_mode(student), torch.no_grad():
                    target_video, target_audio = pdd_teacher_mean_velocity(
                        transformer, trajectory.forward_kwargs, state_video, state_audio, target, grids, args.pdd_solver
                    )
                target_video, target_audio = trajectory.generated(target_video, target_audio)
                video_loss = video_loss + F.mse_loss(video_output[:, 2 * position + 1].float(), target_video)
                audio_loss = audio_loss + F.mse_loss(audio_output[:, 2 * position + 1].float(), target_audio)
            video_loss = video_loss / len(targets)
            audio_loss = audio_loss / len(targets)
            loss = args.video_loss_weight * video_loss + args.audio_loss_weight * audio_loss

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.detach()[None]).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps
            train_video_loss += (
                accelerator.gather(video_loss.detach()[None]).mean().item() / args.gradient_accumulation_steps
            )
            train_audio_loss += (
                accelerator.gather(audio_loss.detach()[None]).mean().item() / args.gradient_accumulation_steps
            )

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                max_grad_norm = linear_decay(
                    args.max_grad_norm * args.initial_grad_norm_ratio,
                    args.max_grad_norm,
                    args.abnormal_norm_clip_start,
                    global_step,
                )
                accelerator.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            trajectory.video, trajectory.audio = trajectory.with_generated(
                state_video_tail + video_output[:, -1].detach(),
                state_audio_tail + audio_output[:, -1].detach(),
            )
            trajectory.index = start + args.pdd_block_size
            del video_output, audio_output

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            if ema is not None:
                ema.step(trainable_params)
            progress_bar.update(1)
            global_step += 1
            accelerator.log(
                {
                    "train_loss": train_loss,
                    "video_loss": train_video_loss,
                    "audio_loss": train_audio_loss,
                    "grid_index": start,
                    "lr_heads": lr_scheduler.get_last_lr()[0],
                    "lr_lora": lr_scheduler.get_last_lr()[-1],
                    "step_seconds": time.time() - step_started,
                },
                step=global_step,
            )
            train_loss = 0.0
            train_video_loss = 0.0
            train_audio_loss = 0.0
            step_started = time.time()

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
                    if args.use_deepspeed or args.use_fsdp or args.save_state:
                        accelerator.save_state(save_path)
                    else:
                        save_resume_state(save_path, student, optimizer, lr_scheduler, ema, accelerator)
                        dump_pdd_config(args, save_path)

                if ema is not None:
                    ema.store(trainable_params)
                    ema.copy_to(trainable_params)
                if args.use_deepspeed or args.use_fsdp:
                    state_dict = accelerator.get_state_dict(transformer, unwrap=True)
                    if accelerator.is_main_process and state_dict is not None:
                        save_pdd_weights(
                            os.path.join(args.output_dir, f"checkpoint-{global_step}", PDD_WEIGHTS_NAME),
                            filter_pdd_state_dict(unwrap_model(transformer), state_dict),
                        )
                        dump_pdd_config(args, os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                        logger.info(f"Saved state to {os.path.join(args.output_dir, f'checkpoint-{global_step}')}")
                elif accelerator.is_main_process:
                    save_pdd_weights(
                        os.path.join(args.output_dir, f"checkpoint-{global_step}", PDD_WEIGHTS_NAME),
                        pdd_state_dict(unwrap_model(transformer)),
                    )
                    logger.info(f"Saved state to {os.path.join(args.output_dir, f'checkpoint-{global_step}')}")
                if ema is not None:
                    ema.restore(trainable_params)
                accelerator.wait_for_everyone()

            if global_step % args.validation_steps == 0:
                if ema is not None:
                    ema.store(trainable_params)
                    ema.copy_to(trainable_params)
                accelerator.wait_for_everyone()
                log_validation(
                    vae, audio_vae, transformer, scheduler, audio_scheduler, args, accelerator,
                    val_cache, grids, global_step,
                )
                accelerator.wait_for_everyone()
                if ema is not None:
                    ema.restore(trainable_params)
                step_started = time.time()

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        if args.use_deepspeed or args.use_fsdp or args.save_state:
            accelerator.save_state(save_path)
        else:
            save_resume_state(save_path, student, optimizer, lr_scheduler, ema, accelerator)
            dump_pdd_config(args, save_path)
        if ema is not None:
            ema.copy_to(trainable_params)
        if args.use_deepspeed or args.use_fsdp:
            state_dict = accelerator.get_state_dict(transformer, unwrap=True)
            if accelerator.is_main_process and state_dict is not None:
                save_pdd_weights(
                    os.path.join(save_path, PDD_WEIGHTS_NAME),
                    filter_pdd_state_dict(unwrap_model(transformer), state_dict),
                )
                dump_pdd_config(args, save_path)
                logger.info(f"Saved state to {save_path}")
        elif accelerator.is_main_process:
            save_pdd_weights(os.path.join(save_path, PDD_WEIGHTS_NAME), pdd_state_dict(unwrap_model(transformer)))
            logger.info(f"Saved state to {save_path}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
