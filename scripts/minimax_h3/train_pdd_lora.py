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
# when it reaches the end of the grid. The conditioning is either pre-encoded (`--enable_preprocess_training`,
# which keeps the 62 GB conditioner out of the run) or encoded on the fly from an annotation JSON; `ref2va`
# supports both, its on-the-fly route additionally VAE-encoding each request's reference latents.
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
#   # fl2va off a pre-encoded prompt cache (`--enable_preprocess_training`):
#   accelerate launch --mixed_precision no scripts/minimax_h3/train_pdd_lora.py \
#       --pretrained_model_name_or_path=models/Diffusion_Transformer/MiniMax-H3 \
#       --train_mode=fl2va --enable_preprocess_training \
#       --train_data_meta=datasets/minimax_h3_pdd_prompt_cache/outputs.json \
#       --output_dir=output_dir_minimax_h3_pdd_lora --gradient_checkpointing --resume_from_checkpoint=latest
#
#   # ref2va loaded directly from a request annotation (no request cache; encodes on the fly):
#   accelerate launch --mixed_precision no scripts/minimax_h3/train_pdd_lora.py \
#       --pretrained_model_name_or_path=models/Diffusion_Transformer/MiniMax-H3 \
#       --train_mode=ref2va \
#       --train_data_meta=datasets/X-Fun-Videos-Audios-Demo/metadata_add_width_height.json \
#       --output_dir=output_dir_minimax_h3_pdd_ref2va_lora --gradient_checkpointing --resume_from_checkpoint=latest

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
from torch.utils.data import DataLoader, Dataset
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

from videox_fun.data import (BatchSampler, ImageVideoSafetensorsDataset,
                             RandomSampler, TextDataset)
from videox_fun.models import (AutoencoderKLMiniMaxH3,
                               AutoencoderKLMiniMaxH3Audio,
                               MiniMaxH3Transformer3DModel, Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor)
from videox_fun.models.minimax_h3_pdd import (attach_parallel_decoder,
                                              pdd_teacher_mean_velocity,
                                              set_parallel_plan)
from videox_fun.utils.lora_utils_pdd import (PDD_EMA_WEIGHTS_NAME,
                                             PDD_LEGACY_LIVE_WEIGHTS_NAME,
                                             PDD_WEIGHTS_NAME, PDDLoRALinear,
                                             PDDParallelHead, add_pdd_lora,
                                             pdd_sampling_plan,
                                             pdd_state_dict,
                                             pdd_teacher_mode,
                                             pdd_time_grid,
                                             pdd_training_plan)
from videox_fun.pipeline import MiniMaxH3Pipeline
from videox_fun.pipeline.pipeline_minimax_h3 import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG, align_num_frames, audio_latent_num_frames,
    build_packed_sequence, build_ref2va_packed_sequence, build_row_timesteps,
    check_ref2va_references, normalize_ref2va_references,
    patchify_video_latents, video_latent_num_frames)
from videox_fun.utils import MiniMaxH3Scheduler
from videox_fun.utils.utils import save_videos_with_audio_grid

# The on-the-fly route (without `--enable_preprocess_training`) encodes conditioning with the canonical MiniMax-H3
# recipes rather than re-deriving them; these modules sit in the same directory, already on `sys.path`.
# `encode_prompt` builds the `fl2va` / `ref2va` presentation, `encode_reference_latents_for_training` the `ref2va`
# reference latents, and `load_requests` / `parse_reference` read a `ref2va` request annotation exactly as
# `generate_ref2va_request_cache.py` does.
from train_lora import encode_prompt, encode_reference_latents_for_training
from generate_ref2va_request_cache import load_requests, parse_reference

# Silences diffusers' `randn_tensor` notice about CPU generators producing CUDA tensors (the tensor is created
# on CPU and moved to GPU; harmless, only a marginal speed note).
warnings.filterwarnings("ignore", message="The passed generator was created on")


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    return initial_value + step_size * current_step


# The `ref2va` request cache flattens its ragged reference structure into tensors (safetensors holds tensors only):
# the per-reference kind / has-audio pair becomes two int vectors and the per-reference latents become indexed
# tensors under a count. This maps the kind id back to the string `Ref2VATrajectory` and `log_validation` expect.
_REFERENCE_KINDS = ("image", "video", "audio")


def reconstruct_cache_entry(state_dict, train_mode):
    r"""Rebuild one normalized conditioning entry from a flat safetensors `state_dict`.

    `ImageVideoSafetensorsDataset.__getitem__` returns the raw tensor dict written by `generate_prompt_cache.py`
    (`fl2va`) or `generate_ref2va_request_cache.py` (`ref2va`); this restores the `{prompt_embeds, text_token_tags}`
    (plus, for `ref2va`, `{reference_kinds, condition_latents, audio_condition_latents}`) shape the trajectories and
    validation consume, so both are agnostic to how the cache was serialized.
    """
    entry = {
        "prompt_embeds": state_dict["prompt_embeds"],
        "text_token_tags": state_dict["text_token_tags"],
    }
    if train_mode == "ref2va":
        kind_ids = state_dict["reference_kind_ids"].tolist()
        has_audio = state_dict["reference_has_audio"].tolist()
        entry["reference_kinds"] = [
            (_REFERENCE_KINDS[int(kind)], bool(int(flag))) for kind, flag in zip(kind_ids, has_audio)
        ]
        num_condition = int(state_dict["num_condition_latents"])
        entry["condition_latents"] = [state_dict[f"condition_latents_{i}"] for i in range(num_condition)]
        num_audio_condition = int(state_dict["num_audio_condition_latents"])
        entry["audio_condition_latents"] = [
            state_dict[f"audio_condition_latents_{i}"] for i in range(num_audio_condition)
        ]
    return entry


class _RequestDataset(Dataset):
    r"""Map-style wrapper over the in-memory `ref2va` request list `load_requests` returns.

    The on-the-fly `ref2va` route has no pre-encoded safetensors to read, so the requests (each a
    `{"prompt": str, "references": [str, ...]}` record) are carried in memory and flow through the same
    accelerate-sharded DataLoader as the other conditioning routes. The reference media are parsed and encoded in the
    conditioning iterator (main process), never in the collate / DataLoader workers.
    """

    def __init__(self, requests):
        self.requests = list(requests)

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, index):
        return self.requests[index]


class FL2VATrajectory:
    r"""
    One rank's carried trajectory of the data-free PDD algorithm on the FL2VA / t2va layout.

    The state is a partially denoised sample plus the grid index it sits at. A step reads it, rolls it forward by
    `L_min` intervals with the student's own prediction, and the trajectory is thrown away and re-drawn from noise
    (with a new prompt) once it reaches the end of the grid.
    """

    def __init__(self, geometry, patch_size, latent_channels, audio_channels, condition_iter, device):
        self.geometry = geometry
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.audio_channels = audio_channels
        self.condition_iter = condition_iter
        self.device = device
        self.index = None

    def reset(self):
        num_latent_frames, latent_height, latent_width, num_audio_latents = self.geometry
        # Draw the next conditioning entry from the (accelerate-sharded, cycling) dataloader iterator instead of
        # picking at random from a whole in-memory cache: each rank now sees a distinct slice of the data.
        cached = next(self.condition_iter)
        self.prompt_embeds = cached["prompt_embeds"].to(self.device)
        text_token_tags = cached["text_token_tags"]
        if not torch.is_tensor(text_token_tags):
            text_token_tags = torch.tensor(text_token_tags, dtype=torch.long)
        else:
            # `accelerator.prepare` moves the cached conditioning onto the GPU, but `build_packed_sequence` assembles
            # the layout on CPU (its outputs are moved to `self.device` just below), so keep the tags on CPU to match.
            text_token_tags = text_token_tags.cpu()
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

    def __init__(self, geometry, patch_size, latent_channels, audio_channels, condition_iter, scheduler, device):
        self.geometry = geometry
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.audio_channels = audio_channels
        self.condition_iter = condition_iter
        self.scheduler = scheduler
        self.device = device
        self.index = None

    def reset(self):
        num_latent_frames, latent_height, latent_width, num_audio_latents = self.geometry
        # Draw the next cached request from the (accelerate-sharded, cycling) dataloader iterator.
        request = next(self.condition_iter)
        self.prompt_embeds = request["prompt_embeds"].to(self.device)
        references = [SimpleNamespace(kind=kind, has_audio=has_audio) for kind, has_audio in request["reference_kinds"]]
        condition_latents = request["condition_latents"]
        audio_condition_latents = request["audio_condition_latents"]
        text_token_tags = request["text_token_tags"]
        if not torch.is_tensor(text_token_tags):
            text_token_tags = torch.tensor(text_token_tags, dtype=torch.long)
        else:
            # `accelerator.prepare` moves the cached conditioning onto the GPU, but `build_ref2va_packed_sequence`
            # assembles the layout on CPU (its outputs are moved to `self.device` just below), so keep the tags there.
            text_token_tags = text_token_tags.cpu()

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
    Render every validation cache entry with the student at `--validation_nfe` and save the mp4 (video + audio)
    next to the run. Each entry is already-cached Qwen3-VL conditioning, so no text encoder is loaded here.

    PDD generation is an ordinary Euler loop over the *block boundaries* of the grid: those boundaries are exactly
    the schedule `MiniMaxH3Scheduler` builds for `NFE` steps, so the released pipeline drives the student unchanged
    and the only PDD-specific work is arming the heads before each step.
    """
    sharded = (
        getattr(accelerator.state, "fsdp_plugin", None) is not None
        or getattr(accelerator.state, "deepspeed_plugin", None) is not None
    )
    if sharded:
        # Under FSDP / DeepSpeed every forward all-gathers the sharded params, so it is a *collective* that all ranks
        # must enter the same number of times. Splitting the entries by `index % num_processes` gives ranks different
        # counts whenever `len(val_cache)` is not a multiple of `num_processes` (2 entries over 4 ranks leaves ranks 2
        # and 3 with nothing to render); a rank that returns early never joins the all-gather the others block on, and
        # NCCL times out. So every rank walks the same `ceil(len / ranks)` rounds, cycling the entry list, and writes
        # a file only for the entries it owns (`index < len`) — the extra cycles are collective ballast that keep the
        # ranks in lockstep, not duplicate renders.
        num_rounds = math.ceil(len(val_cache) / accelerator.num_processes)
        assigned = []
        for round_index in range(num_rounds):
            index = round_index * accelerator.num_processes + accelerator.process_index
            assigned.append((index, val_cache[index % len(val_cache)], index < len(val_cache)))
    else:
        assigned = [
            (index, entry, True)
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

            for index, entry, owner in assigned:
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
                if owner:
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
        help="fl2va: FL2VA / t2va packed layout. ref2va: Ref2VA layout, which also consumes cached reference latents.",
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
        "--enable_preprocess_training",
        action="store_true",
        help=(
            "Train on the pre-processed (cached) conditioning instead of encoding it on the fly. When set, read the "
            "pre-encoded safetensors via `ImageVideoSafetensorsDataset(--train_data_meta=outputs.json, "
            "data_root=--train_data_dir)` — the ~62 GB conditioner stays out of training. When unset, load a Qwen3-VL "
            "conditioner in the run and encode the conditioning on the fly: `fl2va` reads prompts via "
            "`TextDataset(--train_data_meta)`, and `ref2va` reads a request annotation via `load_requests` and also "
            "VAE-encodes each request's reference latents."
        ),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Optional root prepended to each `file_path` of `--train_data_meta` (used with "
             "`--enable_preprocess_training`); leave empty when `outputs.json` already stores absolute paths.",
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help="Annotation JSON of the training conditioning. With `--enable_preprocess_training`: the `outputs.json` "
             "written by `generate_prompt_cache.py` / `generate_ref2va_request_cache.py`. Without it: `fl2va` reads a "
             "list of `{\"text\": ...}` records (`TextDataset`), and `ref2va` reads a list of requests "
             "(`{\"prompt\": ..., \"references\": [...]}`, or the demo record `load_requests` derives them from).",
    )
    parser.add_argument(
        "--val_data_meta",
        type=str,
        default=None,
        help="Optional annotation JSON of the held-out conditioning used by validation, mirroring `--train_data_meta`: "
             "with `--enable_preprocess_training` the `outputs.json` written by `generate_prompt_cache.py` / "
             "`generate_ref2va_request_cache.py`; without it the on-the-fly annotation (`fl2va`: `{\"text\": ...}` "
             "records; `ref2va`: the request list). Validation is skipped when it is left empty.",
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
        help="Keep an exponential moving average of the trainable set. Validation and `pdd_ema.safetensors` use it.",
    )
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--abnormal_norm_clip_start", type=int, default=1000)
    parser.add_argument("--initial_grad_norm_ratio", type=int, default=5)
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=1280,
        help="Square canvas size (height = width) for validation and latent geometry; must be a multiple of 32.",
    )
    parser.add_argument(
        "--fix_sample_size",
        nargs=2, type=int, default=None,
        help="Fix Sample size [height, width] to override `--video_sample_size` with a fixed non-square shape.",
    )
    parser.add_argument("--validation_nfe", type=int, default=8)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.pdd_max_block_size is None:
        args.pdd_max_block_size = args.pdd_block_size
    if args.fix_sample_size is not None:
        args.video_sample_height, args.video_sample_width = args.fix_sample_size
    else:
        args.video_sample_height = args.video_sample_width = args.video_sample_size
    return args


def gather_full_state_dict(model, accelerator):
    r"""Consolidated `state_dict` of a (possibly sharded) model, gathered to rank-0 CPU.

    Under `--fsdp_state_dict_type=SHARDED_STATE_DICT`, `accelerator.get_state_dict(..., unwrap=True)` hands back
    *sharded* tensors, so `pdd_state_dict`'s `.detach().cpu()` only materializes the params of the root wrap
    unit (the token refiner) and silently drops every separately-wrapped child unit — the `MiniMaxH3TransformerBlock`
    LoRA and both `PDDParallelHead`s — leaving a 13 MB stub instead of the full ~1.4 GB trainable set. Temporarily
    switching the FSDP root to `FULL_STATE_DICT` (offloaded to CPU, rank-0 only) reconstructs every original-named
    param across all wrap units. DeepSpeed ZeRO-3 and DDP already consolidate in `accelerator.get_state_dict`, so
    they keep using it. Collective: call on every rank; only rank 0 receives a non-empty dict.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    # `accelerator.get_state_dict(..., unwrap=True)` unwraps the FSDP root and reads `.state_dict()` off the *sharded*
    # original-param views, so it silently returns only the root wrap unit's tensors (the token refiner) and drops
    # every separately-wrapped child unit. Detect the FSDP wrapper directly — not via `fsdp_plugin`, which is not
    # reliably plumbed this far — and gather a FULL_STATE_DICT on the wrapped root instead.
    if isinstance(model, FSDP):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()
        if accelerator.is_main_process:
            logger.info(
                "gather_full_state_dict[FSDP FULL_STATE_DICT]: %d tensors (blocks.*=%d, proj_out*=%d).",
                len(state_dict),
                sum("blocks." in key for key in state_dict),
                sum("proj_out" in key for key in state_dict),
            )
        return state_dict
    state_dict = accelerator.get_state_dict(model, unwrap=True)
    if accelerator.is_main_process:
        logger.info(
            "gather_full_state_dict[get_state_dict fallback]: %d tensors (type(model)=%s).",
            len(state_dict) if state_dict else 0,
            type(model).__name__,
        )
    return state_dict


def save_pdd_weights(path, state_dict):
    from safetensors.torch import save_file
    save_file(
        {name: tensor.detach().contiguous().cpu() for name, tensor in state_dict.items()},
        path,
        metadata={"format": "pt"},
    )


def dump_pdd_config(args, save_path):
    r"""Write `pdd_config.json` with both this script's LoRA flags and the inference aliases `predict_t2v.py` reads."""
    config = dict(vars(args))
    config["lora_rank"] = args.rank
    config["lora_alpha"] = args.network_alpha
    config["lora_targets"] = args.target_name
    with open(os.path.join(save_path, "pdd_config.json"), "w") as handle:
        json.dump(config, handle, indent=1)


def save_resume_state(save_path, student, optimizer, lr_scheduler, ema, accelerator):
    r"""Optimizer / scheduler / live `pdd.safetensors` / EMA shadow — the pieces `pdd_ema.safetensors` does not hold."""
    os.makedirs(save_path, exist_ok=True)
    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
    save_pdd_weights(os.path.join(save_path, PDD_WEIGHTS_NAME), pdd_state_dict(student))
    if ema is not None:
        torch.save(ema.state_dict(), os.path.join(save_path, "ema.pt"))
    if getattr(accelerator, "scaler", None) is not None:
        torch.save(accelerator.scaler.state_dict(), os.path.join(save_path, "scaler.pt"))


def load_resume_state(save_path, student, optimizer, lr_scheduler, ema, trainable_params, accelerator):
    r"""Load the trainer state written by [`save_resume_state`]. Prefers legacy `pdd_live.safetensors` when present."""
    from safetensors.torch import load_file

    legacy_live = os.path.join(save_path, PDD_LEGACY_LIVE_WEIGHTS_NAME)
    weights_path = legacy_live if os.path.isfile(legacy_live) else os.path.join(save_path, PDD_WEIGHTS_NAME)
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
            f"`video_sample_size` / `fix_sample_size` ({args.video_sample_height}x{args.video_sample_width}) "
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
    if not args.train_data_meta:
        raise ValueError(
            "`--train_data_meta` is required: the `outputs.json` of the cached conditioning (with "
            "`--enable_preprocess_training`) or the on-the-fly annotation JSON without it (`fl2va`: `{\"text\": ...}` "
            "records; `ref2va`: the request list `load_requests` reads)."
        )
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
    # of a step differ only in their prompt. The conditioning order is drawn by the dataloader's own seeded sampler.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
        print(f"Init seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")
    else:
        print(f"Init without fixed seed. Process_index is {accelerator.process_index}")

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
    # and only the transformer will get ZeRO sharded. The 62 GB conditioner is loaded only when conditioning is
    # encoded on the fly (without `--enable_preprocess_training`); the cached route keeps it out of the run entirely.
    uses_text_encoder = not args.enable_preprocess_training
    tokenizer = processor = text_encoder = None
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
        if uses_text_encoder:
            # On-the-fly conditioning (`fl2va` prompts, `ref2va` requests): the same Qwen3-VL components
            # `train_lora.py` loads, so `train_lora.encode_prompt` runs unchanged. Mirrors `scripts/minimax_h3/train_lora.py`.
            tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(args.pretrained_model_name_or_path, "tokenizer"))
            processor = Qwen3VLProcessor.from_pretrained(os.path.join(args.pretrained_model_name_or_path, "processor"))
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                os.path.join(args.pretrained_model_name_or_path, "text_encoder"), low_cpu_mem_usage=True, torch_dtype=weight_dtype,
            )
            text_encoder = text_encoder.eval()
    scheduler = MiniMaxH3Scheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    audio_scheduler = MiniMaxH3Scheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="audio_scheduler")

    # Freeze everything; the LoRA modules and parallel heads created below are the only trainable parameters.
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    audio_vae.requires_grad_(False)
    if uses_text_encoder:
        text_encoder.requires_grad_(False)

    # ------------------------------------------------------------------ LoRA
    num_adapters = add_pdd_lora(transformer, args.target_name.split(","), args.rank, args.network_alpha)
    attach_parallel_decoder(transformer, args.pdd_num_steps)
    transformer.train()
    # FSDP flattens one dtype per wrap unit. DDP keeps float32 LoRA master weights; under FSDP the adapters match
    # the Linear they wrap (bf16) so each `MiniMaxH3TransformerBlock` is uniform. The parallel heads stay float32
    # and are wrapped as their own units. Frozen `_keep_in_fp32_modules` embeddings are ignored so they are not
    # mixed into the bf16 root flatten (`--mixed_precision=no` does not install an FSDP MixedPrecision policy).
    if fsdp_plugin is not None:
        for module in transformer.modules():
            if isinstance(module, PDDLoRALinear):
                dtype = module.base.weight.dtype
                module.lora_down.data = module.lora_down.data.to(dtype)
                module.lora_up.data = module.lora_up.data.to(dtype)
        wrap_names = list(fsdp_plugin.transformer_cls_names_to_wrap or [])
        if "PDDParallelHead" not in wrap_names:
            wrap_names.append("PDDParallelHead")
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
    # live `pdd.safetensors` from the gathered trainable tensors so DDP `--save_state` resume can reload
    # the current step; popping `weights` on the DDP path keeps `save_state` from serializing the frozen
    # backbone. The EMA inference export `pdd_ema.safetensors` is written after `ema.copy_to`.
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        if fsdp_stage != 0 or zero_stage == 3:
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = gather_full_state_dict(models[-1], accelerator)
                if accelerator.is_main_process and accelerate_state_dict is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    save_pdd_weights(
                        os.path.join(output_dir, PDD_WEIGHTS_NAME),
                        pdd_state_dict(unwrap_model(models[-1]), accelerate_state_dict),
                    )
                    dump_pdd_config(args, output_dir)

            def load_model_hook(models, input_dir):
                return

        else:
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = gather_full_state_dict(models[-1], accelerator)
                if accelerator.is_main_process and accelerate_state_dict is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    save_pdd_weights(
                        os.path.join(output_dir, PDD_WEIGHTS_NAME),
                        pdd_state_dict(unwrap_model(models[-1]), accelerate_state_dict),
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
    # Data-free PDD never reads a target video: each rank carries one trajectory and needs only the conditioning —
    # either pre-encoded safetensors (`--enable_preprocess_training`) or an annotation encoded on the fly (without it).
    # All three routes go through a DataLoader so `accelerator.prepare` shards the entries across ranks, replacing the
    # old whole-cache random pick.
    if args.enable_preprocess_training:
        train_dataset = ImageVideoSafetensorsDataset(args.train_data_meta, data_root=args.train_data_dir)

        def collate_fn(examples):
            return reconstruct_cache_entry(examples[0], args.train_mode)
    elif args.train_mode == "ref2va":
        # On-the-fly `ref2va`: read the request annotation with `load_requests` (the same reader the cache generator
        # uses) and encode each request's prompt + reference latents in the conditioning iterator below.
        train_dataset = _RequestDataset(load_requests(args.train_data_meta))

        def collate_fn(examples):
            # `--train_batch_size=1`: one request per trajectory reset, so hand over the single record; the Qwen3-VL /
            # VAE encode runs in the conditioning iterator below (main process), not in the collate / workers.
            return {"prompt": examples[0]["prompt"], "references": list(examples[0]["references"])}
    else:
        train_dataset = TextDataset(args.train_data_meta)

        def collate_fn(examples):
            # `--train_batch_size=1`: one conditioning entry per trajectory reset, so hand over the single record; the
            # Qwen3-VL encode runs in the conditioning iterator below (main process), not in the collate / workers.
            return {"text": examples[0]["text"]}

    batch_sampler_generator = torch.Generator().manual_seed(args.seed)
    batch_sampler = BatchSampler(
        RandomSampler(train_dataset, generator=batch_sampler_generator),
        batch_size=args.train_batch_size,
        drop_last=True,
    )
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    # The held-out validation conditioning mirrors the training route (a cache *or* on-the-fly prompts) instead of
    # forcing a pre-processed cache, so it is built further down — once the conditioner is sharded / on-device — right
    # before the trajectory setup. Validation is skipped when `--val_data_meta` is empty.

    # Scheduler and math around the number of training steps. One epoch is one pass through the conditioning set
    # (batch size is 1). `--max_train_steps` overrides `--num_train_epochs`, matching `scripts/minimax_h3/train_lora.py`.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer.gradient_checkpointing_save_on_cpu = args.gradient_checkpointing_save_on_cpu
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Shard the frozen text encoder *after* prepare (mirrors `train_lora.py`): the Qwen3-VL conditioner (~62 GB) is
    # wrapped per decoder layer so the per-step unshard footprint stays small, and a post-prepare shard keeps it out of
    # the trainable FSDP unit. Only the on-the-fly text route loads it (`uses_text_encoder`).
    sharded_text_encoder = uses_text_encoder and (fsdp_stage != 0 or zero_stage != 0)
    if sharded_text_encoder:
        from videox_fun.dist import shard_model
        text_encoder.model = shard_model(
            text_encoder.model,
            device_id=accelerator.device,
            param_dtype=weight_dtype,
            module_to_wrapper=list(text_encoder.model.language_model.layers),
        )

    device = accelerator.device
    # The two VAEs stay float32 (mirrors the pipeline: float32 weights, float16 autocast only at the
    # encode/decode call site), so they are moved without a dtype cast.
    vae.to(device if not args.low_vram else "cpu")
    audio_vae.to(device if not args.low_vram else "cpu")
    if fsdp_stage == 0 and zero_stage == 0:
        transformer.to(device)
    # An FSDP/ZeRO-sharded text encoder is already on-device and dtype-pinned by `shard_model`; otherwise move it to
    # the GPU, or keep it on CPU under `--low_vram` (the conditioning iterator moves it up only for each encode).
    if uses_text_encoder and not sharded_text_encoder:
        text_encoder.to(device if not args.low_vram else "cpu", dtype=weight_dtype)

    trainable_params = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    ema = (
        EMAModel(trainable_params, decay=args.ema_decay, use_ema_warmup=False, foreach=True)
        if args.use_ema
        else None
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed. One
    # epoch is one pass through the conditioning set (batch size 1), keeping `--num_train_epochs` / `--max_train_steps`
    # consistent with `train_lora.py`.
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

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

    # On-the-fly `ref2va` conditioning, shared by the training iterator and validation: parse the request's
    # references, then encode the prompt (Qwen3-VL) and the reference latents (the two VAEs) with the same recipes
    # `generate_ref2va_request_cache.py` uses. Under `--low_vram` the two encodes are serialized — conditioner up then
    # down, VAEs up then down — so the ~62 GB text encoder and a 124-frame reference video never share the GPU.
    ref2va_num_frames = align_num_frames(int(args.video_sample_n_frames))
    ref2va_audio_sr = getattr(audio_vae.config, "sampling_rate", 32000)

    def encode_request_on_the_fly(request):
        references = [parse_reference(entry) for entry in request["references"]]
        references = check_ref2va_references(references)
        references = normalize_ref2va_references(references, ref2va_num_frames, ref2va_audio_sr)

        if args.low_vram and not sharded_text_encoder:
            text_encoder.to(device)
        with torch.no_grad():
            prompt_embeds, text_token_tags = encode_prompt(
                text_encoder, tokenizer, processor, request["prompt"],
                references=references, device=device, dtype=weight_dtype,
            )
        if args.low_vram and not sharded_text_encoder:
            text_encoder.to("cpu")
            torch.cuda.empty_cache()

        if args.low_vram:
            vae.to(device)
            audio_vae.to(device)
        with torch.no_grad():
            condition_latents, audio_condition_latents = encode_reference_latents_for_training(
                vae, audio_vae, references, patch_size, device, audio_latent_channels=audio_channels,
            )
        if args.low_vram:
            vae.to("cpu")
            audio_vae.to("cpu")
            torch.cuda.empty_cache()

        return {
            "prompt_embeds": prompt_embeds,
            "text_token_tags": text_token_tags,
            "reference_kinds": [(reference.kind, bool(reference.has_audio)) for reference in references],
            "condition_latents": condition_latents,
            "audio_condition_latents": audio_condition_latents,
        }

    # Validation conditioning, built now that the conditioner is sharded / on-device so it can mirror the training
    # route: with `--enable_preprocess_training` a `generate_*_cache.py` cache (`outputs.json` + safetensors); without
    # it, the on-the-fly annotation encoded here (`fl2va` prompts via `TextDataset`, `ref2va` requests via
    # `encode_request_on_the_fly`). Every rank builds the full list (exactly like the cache route); `log_validation`
    # shards it at render time.
    val_cache = []
    if args.val_data_meta:
        if args.enable_preprocess_training:
            val_dataset = ImageVideoSafetensorsDataset(args.val_data_meta, data_root=args.train_data_dir)
            val_cache = [reconstruct_cache_entry(val_dataset[i], args.train_mode) for i in range(len(val_dataset))]
        elif args.train_mode == "ref2va":
            val_dataset = _RequestDataset(load_requests(args.val_data_meta))
            val_cache = [encode_request_on_the_fly(val_dataset[i]) for i in range(len(val_dataset))]
        else:
            val_dataset = TextDataset(args.val_data_meta)
            if args.low_vram and not sharded_text_encoder:
                text_encoder.to(device)
            with torch.no_grad():
                for i in range(len(val_dataset)):
                    prompt_embeds, text_token_tags = encode_prompt(
                        text_encoder, tokenizer, processor, val_dataset[i]["text"], device=device, dtype=weight_dtype,
                    )
                    val_cache.append({"prompt_embeds": prompt_embeds, "text_token_tags": text_token_tags})
            if args.low_vram and not sharded_text_encoder:
                text_encoder.to("cpu")
                torch.cuda.empty_cache()

    # The conditioning iterator turns the (accelerate-sharded, cycling) dataloader into the normalized entries the
    # trajectories pull on `reset()`. The pre-processed route already yields `{prompt_embeds, text_token_tags}` (plus
    # the ref2va reference tensors); the on-the-fly route encodes each entry here in the main process — `fl2va` prompts
    # via `encode_prompt`, `ref2va` requests via `encode_request_on_the_fly` — moving the conditioner / VAEs up only
    # for the encode under `--low_vram`.
    def conditioning_iterator():
        while True:
            for batch in train_dataloader:
                if not uses_text_encoder:
                    yield batch
                elif args.train_mode == "ref2va":
                    yield encode_request_on_the_fly(batch)
                else:
                    if args.low_vram and not sharded_text_encoder:
                        text_encoder.to(device)
                    with torch.no_grad():
                        prompt_embeds, text_token_tags = encode_prompt(
                            text_encoder, tokenizer, processor, batch["text"], device=device, dtype=weight_dtype,
                        )
                    if args.low_vram and not sharded_text_encoder:
                        text_encoder.to("cpu")
                        torch.cuda.empty_cache()
                    yield {"prompt_embeds": prompt_embeds, "text_token_tags": text_token_tags}

    condition_iter = conditioning_iterator()

    if args.train_mode == "ref2va":
        trajectory = Ref2VATrajectory(
            geometry, patch_size, latent_channels, audio_channels, condition_iter, scheduler, device,
        )
    else:
        trajectory = FL2VATrajectory(
            geometry, patch_size, latent_channels, audio_channels, condition_iter, device,
        )
    target_seed = (args.seed if args.seed is not None else 0) + 1000 + accelerator.process_index
    target_rng = np.random.default_rng(np.random.PCG64(target_seed))

    # ------------------------------------------------------------------ train loop
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
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
                with pdd_teacher_mode(student), torch.no_grad():
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
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if args.use_deepspeed or args.use_fsdp:
                        state_dict = gather_full_state_dict(transformer, accelerator)
                        if accelerator.is_main_process and state_dict is not None:
                            save_pdd_weights(
                                os.path.join(checkpoint_dir, PDD_EMA_WEIGHTS_NAME),
                                pdd_state_dict(unwrap_model(transformer), state_dict),
                            )
                            dump_pdd_config(args, checkpoint_dir)
                    elif accelerator.is_main_process:
                        save_pdd_weights(
                            os.path.join(checkpoint_dir, PDD_EMA_WEIGHTS_NAME),
                            pdd_state_dict(unwrap_model(transformer)),
                        )
                    ema.restore(trainable_params)
                if accelerator.is_main_process:
                    logger.info(f"Saved state to {os.path.join(args.output_dir, f'checkpoint-{global_step}')}")
                accelerator.wait_for_everyone()

            if global_step % args.validation_steps == 0 and val_cache:
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
                state_dict = gather_full_state_dict(transformer, accelerator)
                if accelerator.is_main_process and state_dict is not None:
                    save_pdd_weights(
                        os.path.join(save_path, PDD_EMA_WEIGHTS_NAME),
                        pdd_state_dict(unwrap_model(transformer), state_dict),
                    )
                    dump_pdd_config(args, save_path)
            elif accelerator.is_main_process:
                save_pdd_weights(os.path.join(save_path, PDD_EMA_WEIGHTS_NAME), pdd_state_dict(unwrap_model(transformer)))
        if accelerator.is_main_process:
            logger.info(f"Saved state to {save_path}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
