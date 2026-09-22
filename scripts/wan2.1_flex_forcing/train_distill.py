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
                             ImageVideoDataset, ImageVideoSampler,
                             RandomSampler, TextDataset, get_closest_ratio,
                             get_random_mask)
from videox_fun.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel,
                               WanTransformer3DModel,
                               WanTransformer3DModel_FlexForcing,
                               WanTransformer3DModel_SelfForcing)
from videox_fun.pipeline import (WanI2VPipeline, WanPipeline,
                                 WanFlexForcingPipeline,
                                 WanSelfForcingPipeline)
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.flex_chunking import (UNIFORM_BLOCK_PROB,
                                            broadcast_chunk_sizes,
                                            build_pyramid_partitions,
                                            chunk_boundaries,
                                            sample_flexible_chunks,
                                            uniform_chunks,
                                            validate_nested_partitions)
from videox_fun.utils.tqdm_bar import PauseAwareTqdm
from videox_fun.utils.utils import (calculate_dimensions, get_image_latent,
                                    get_image_to_video_latent,
                                    save_videos_grid)

if is_wandb_available():
    import wandb


def initialize_kv_cache_for_training(batch_size, num_frames, frame_seq_length, num_layers, num_heads, head_dim, dtype, device):
    """Initialize KV cache for block-by-block training"""
    kv_cache_size = num_frames * frame_seq_length
    kv_cache = []
    
    for _ in range(num_layers):
        kv_cache.append({
            "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
        })
    
    return kv_cache


def initialize_crossattn_cache_for_training(batch_size, text_len, num_layers, num_heads, head_dim, dtype, device):
    """Initialize cross-attention cache for block-by-block training"""
    crossattn_cache = []
    
    for _ in range(num_layers):
        crossattn_cache.append({
            "k": torch.zeros([batch_size, text_len, num_heads, head_dim], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, text_len, num_heads, head_dim], dtype=dtype, device=device),
            "is_init": False
        })
    
    return crossattn_cache


def slice_last_n_latent_frames(tensor, n):
    """Slice last n frames from [B, C, F, H, W] tensor."""
    if tensor.shape[2] <= n:
        return tensor
    return tensor[:, :, -n:]


def reencode_boundary_latent(vae, pred_latents, weight_dtype, score_num_frames=21):
    """
    Re-encode the boundary frame to get a clean latent for the score window.
    Follows Self-Forcing reference: decode all frames before the score window, take last pixel frame, re-encode.
    Input: pred_latents [B, C, F, H, W] (all generated latent frames)
    Output: boundary_latent [B, C, 1, H, W]
    """
    with torch.no_grad():
        # Decode all frames except the last (score_num_frames - 1) to pixels
        tail_len = score_num_frames - 1
        latent_to_decode = pred_latents[:, :, :-tail_len]
        # VAE expects [B, C, F, H, W], decode returns [B, C, F, H, W] pixels
        pixels = vae.decode(latent_to_decode.to(vae.dtype)).sample  # [B, C, F, H, W]
        # Take the last frame
        frame = pixels[:, :, -1:, :, :]  # [B, C, 1, H, W]
        # Re-encode the last frame to get clean boundary latent
        boundary_latent = vae.encode(frame)[0].sample().to(weight_dtype)  # [B, C, 1, H, W]
    return boundary_latent


def slice_for_score(pred, vae, weight_dtype, score_num_frames=21, independent_first_frame=False):
    """
    Slice the last `score_num_frames` latent frames for score computation.
    If pred has more than score_num_frames, re-encode boundary frame for clean context.
    Returns: (pred_for_score, score_num_frames, need_gradient_mask)
    """
    num_frames = pred.shape[2]
    if num_frames <= score_num_frames:
        return pred, num_frames, False

    # Re-encode boundary for cleaner score input
    try:
        boundary_latent = reencode_boundary_latent(vae, pred, weight_dtype, score_num_frames=score_num_frames)
        pred_for_score = torch.cat([boundary_latent, pred[:, :, -(score_num_frames - 1):]], dim=2)
    except Exception:
        # Fallback: simple slice without boundary re-encoding
        pred_for_score = pred[:, :, -score_num_frames:]

    return pred_for_score, score_num_frames, True


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

def log_validation(vae, text_encoder, tokenizer, clip_image_encoder, transformer3d, args, config, accelerator, weight_dtype, global_step):
    try:
        is_deepspeed = type(transformer3d).__name__ == 'DeepSpeedEngine'
        if is_deepspeed:
            origin_config = transformer3d.config
            transformer3d.config = accelerator.unwrap_model(transformer3d).config
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
            logger.info("Running validation... ")
            scheduler = FlowMatchEulerDiscreteScheduler(
                **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
            )
        
            if args.train_mode != "normal":
                raise NotImplementedError(f"Validation for train_mode '{args.train_mode}' is not yet supported with WanSelfForcingPipeline. Only T2V (train_mode='normal') is currently supported.")
            else:
                pipeline_cls = WanFlexForcingPipeline if args.flex_forcing else WanSelfForcingPipeline
                pipeline = pipeline_cls(
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
                if args.train_mode != "normal":
                    raise NotImplementedError(f"Validation for train_mode '{args.train_mode}' is not yet supported with WanSelfForcingPipeline. Only T2V (train_mode='normal') is currently supported.")
                else:
                    if args.fix_sample_size is not None:
                        height, width = args.fix_sample_size
                    else:
                        height, width = args.video_sample_size, args.video_sample_size
                    # Validation reuses the inference pipeline, so it has to build
                    # the layout inference builds - otherwise the samples say
                    # nothing about the model being trained.
                    flex_kwargs = {}
                    if args.flex_forcing:
                        if args.flex_pyramid_levels > 1:
                            # 3.2: exactly what predict_t2v.py passes. `chunk_spec=None`
                            # leaves level 0 as the whole clip (the planning step) and
                            # "pyramid" takes the ladder depth from num_inference_steps,
                            # so nothing has to be kept in sync by hand.
                            flex_kwargs = dict(
                                chunk_spec=None,
                                denoise_mode="pyramid",
                                min_num_frame_per_block=args.flex_min_num_frame_per_block,
                            )
                        else:
                            # No ladder in training, and with no pyramid the
                            # whole-clip band is empty - `[21]` is never drawn at
                            # `flex_chunk_max` below the frame count, so "bidir"
                            # would validate out of distribution too. The uniform
                            # block layout is the one band that is both fixed
                            # across checkpoints and actually trained:
                            # UNIFORM_BLOCK_PROB of iterations use it, against
                            # ~1.3% for the most common random partition.
                            flex_kwargs = dict(
                                chunk_spec=args.num_frame_per_block,
                                denoise_mode="fixed",
                                min_num_frame_per_block=args.flex_min_num_frame_per_block,
                            )
                    sample = pipeline(
                        args.validation_prompts[i],
                        num_frames = args.video_sample_n_frames,
                        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        height      = height,
                        width       = width,
                        generator   = generator,
                        guidance_scale = 1.0,
                        num_inference_steps = len(args.denoising_step_indices_list),
                        num_frame_per_block = args.num_frame_per_block,
                        independent_first_frame = args.independent_first_frame,
                        context_noise = args.context_noise,
                        **flex_kwargs,
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


# Fraction of pyramid iterations whose level 0 is the whole clip as a single
# chunk, i.e. fully bidirectional. 3.2's coarse planning step is exactly that at
# inference (`denoise_mode="pyramid"` with no `chunk_spec` builds `[[F], ...]`),
# so it has to appear in training; the remaining iterations keep the random
# partition of 3.1 so the causal end and every layout in between stay covered.
# Deliberately not a CLI flag - it is a property of the paper's schedule, not a
# knob the launcher should have to keep in sync.
COARSE_GLOBAL_PROB = 0.5


def sample_flex_partitions(args, num_frames, num_denoising_steps, torch_rng,
                           device, verbose=False):
    """Draw the Flex-Forcing partition ladder for one training iteration.

    arXiv 2607.03509 3.1 asks for a *random* partition per rollout (chunk sizes
    2..10) so a single model covers the whole causal-to-bidirectional spectrum,
    and 3.2 then nests that partition into a coarse-to-fine ladder with one level
    per denoising step. Returns ``None`` when Flex-Forcing is off, which leaves
    the inherited uniform ``num_frame_per_block`` masks untouched.

    The level-0 layout is a three-way mixture decided by a single uniform draw,
    so the constants below are the actual iteration shares: ``COARSE_GLOBAL_PROB``
    of iterations use one chunk over the whole clip (what inference's coarse
    planning step uses), ``UNIFORM_BLOCK_PROB`` pin the launcher's own uniform
    ``num_frame_per_block``, and the rest draw a random 2..10 partition. All of
    them are then refined into the same ladder, so a single set of weights covers
    `denoise_mode="pyramid"` whether or not the caller pins `chunk_spec`. With no
    pyramid the coarse band is empty and the split is 10% uniform / 90% random.
    """
    if not args.flex_forcing:
        return None
    u = torch.rand((), generator=torch_rng, device=device).item()
    coarse_prob = COARSE_GLOBAL_PROB if args.flex_pyramid_levels > 1 else 0.0
    if u < coarse_prob:
        # Coarse end of 3.1: reuse the ladder builder's own level-0 rule so the
        # `independent_first_frame` handling cannot drift from inference's.
        base = build_pyramid_partitions(
            num_frames, num_levels=1, base_chunks=None,
            independent_first_frame=args.independent_first_frame)[0]
        if verbose:
            print(f"level 0 = whole clip ({base}): 3.2 coarse planning layout")
    elif u < coarse_prob + UNIFORM_BLOCK_PROB:
        # `uniform_chunks`, not `normalize_chunk_spec`: the latter takes no
        # `independent_first_frame` argument (it encodes that as a leading 1), so
        # it would silently disagree with the two bands around it. This is the
        # band `log_validation` renders when there is no pyramid.
        base = uniform_chunks(
            num_frames, args.num_frame_per_block,
            independent_first_frame=args.independent_first_frame)
        if verbose:
            print(f"level 0 = uniform {args.num_frame_per_block}-frame blocks "
                  f"({base}): the launcher's own layout")
    else:
        base = sample_flexible_chunks(
            num_frames, min_chunk=args.flex_chunk_min, max_chunk=args.flex_chunk_max,
            generator=torch_rng, device=device,
            independent_first_frame=args.independent_first_frame)
    # Every rank has to train the same layout: the FlexAttention mask, and the
    # `num_frame_per_block` derived from it, must agree across the SP/FSDP group.
    base = broadcast_chunk_sizes(base, device=device)
    if args.flex_pyramid_levels <= 1:
        return [base]
    ladder = build_pyramid_partitions(
        num_frames, num_levels=args.flex_pyramid_levels,
        min_num_frame_per_block=args.flex_min_num_frame_per_block, base_chunks=base,
        independent_first_frame=args.independent_first_frame)
    if len(ladder) > num_denoising_steps:
        # Same short-circuit as at inference, where the rollout stops refining at
        # the last step: deeper levels would never be reached, so drop them
        # instead of reporting a pyramid that was not actually trained.
        if verbose:
            print(f"--flex_pyramid_levels={args.flex_pyramid_levels} builds "
                  f"{len(ladder)} levels but only {num_denoising_steps} denoising "
                  f"steps are trained; keeping the first {num_denoising_steps}.")
        ladder = ladder[:num_denoising_steps]
    return ladder


def install_flex_partition(transformer, partitions, step_index):
    """Install the partition that is active at ``step_index`` of the ladder.

    ``partitions is None`` (Flex-Forcing off) is a no-op, so the call sites need
    no branch of their own. The index is clamped: a ladder shorter than the
    denoising schedule keeps its finest level for the remaining steps.
    """
    if partitions is None:
        return None
    sizes = partitions[min(step_index, len(partitions) - 1)]
    transformer.set_flex_chunk_sizes(sizes)
    return sizes


def make_flex_mask_builder(transformer, args, num_frames, frame_seqlen, device,
                           teacher_forcing):
    """Return a closure that (re)builds the FlexAttention block mask.

    Both mask variants read the partition currently installed on ``transformer``,
    so wrapping them once here lets the 3.2 pyramid swap in a finer level at every
    denoising step without duplicating the teacher-forcing branch inside the step
    loop. With Flex-Forcing off the closure forwards to the inherited uniform
    builders unchanged.
    """
    def build():
        if teacher_forcing:
            transformer.create_teacher_forcing_mask(
                device=device,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                num_frame_per_block=args.num_frame_per_block,
            )
        else:
            transformer.create_block_mask_for_training(
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                num_frame_per_block=args.num_frame_per_block,
                independent_first_frame=args.independent_first_frame,
                device=device,
            )
    return build


def flex_self_context_enabled(args, clean_x, final_step_index):
    """Whether 3.3's K-Projection gets a self-generated clean context this round.

    This is a block-mask-path concern only, and this flag is inert under
    ``--use_kv_cache_training``. The KV-cache rollout passes no ``flex_state``,
    so ``_flex_project`` short-circuits and :math:`\\Pi` stays at its identity
    initialisation there - which matches KV inference, which never supplies a
    timestep either, so the two sides agree rather than one of them drifting.
    The block-mask path is the one that exercises :math:`\\Pi`, and it builds no
    KV cache, so with prompt-only data - ``clean_latents is None``, which the
    prompt-only branch sets unconditionally - no clean context exists anywhere
    in the sequence, :math:`\\Pi_{t\\leftarrow 0}` has nothing to project and
    never enters the autograd graph: its parameters sit in an optimizer group
    and still stay at the identity initialisation for the entire run.

    The denoising loop already re-rolls the model under ``no_grad`` and converts
    every non-final step to an x0 prediction, so the previous step's prediction
    *is* a level-0 context the model produced itself. That is what the KV-cache
    path commits ("feed denoised_pred directly"), it needs no real video, and
    unlike ground-truth teacher forcing it does not reintroduce the exposure
    bias self-forcing exists to remove.

    Only from step 1 on: at ``final_step_index == 0`` nothing has been denoised
    yet, and a context fabricated from a single t=1000 prediction would hand
    :math:`\\Pi` noise rather than a clean key.

    Generator and critic must agree on this, or the two sides of the DMD loss
    get rolled out under different attention and see different distributions.
    """
    on = bool(getattr(args, "flex_self_generated_context", False)
              and clean_x is None and final_step_index > 0)
    # Announced once rather than per micro-batch: a 600-step run doubles its
    # sequence on roughly half the iterations, and the log has to say so without
    # drowning the loss lines.
    if on and not flex_self_context_enabled.announced:
        flex_self_context_enabled.announced = True
        print("Flex-Forcing 3.3: no ground-truth clean half on this data, so the "
              "K-Projection reads the model's own previous-step x0 prediction as "
              "its clean context. Doubles the attention sequence, and is active "
              "only on iterations whose final denoising step is not the first.")
    return on


flex_self_context_enabled.announced = False


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
        "--ode_transformer_path",
        type=str,
        default=None,
        help=("If you want to load the ode-trained weight into generator transformer3d, input its path."),
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
            'The format of training data. Support `"normal"`'
            ' (default), `"i2v"`.'
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
        default=4.5,
        help="The cfg scale for real score.",
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
        "--index_jitter_ratio",
        type=float,
        default=0.3,
        help="Symmetric jitter budget (fraction of the neighboring gap) applied to the "
             "denoising step indices when --randomize_step_indices is enabled.",
    )
    parser.add_argument(
        "--flow_euler_rollout",
        action="store_true",
        help="Simulate the normal flow-matching inference rollout in the generator's multi-step "
             "self-rollout (LightX2V-style): keep the model prediction in flow/velocity space and "
             "advance to the next noise level with a deterministic Euler ODE step "
             "(x_next = x_t + (sigma_next - sigma_t) * v), instead of converting the prediction "
             "to x0 and re-noising with fresh noise. The final step still converts to x0 since "
             "the DMD objective is defined on x0. The critic re-rolls the generator, so it follows "
             "the same switch; with --flex_self_generated_context the clean context 3.3 reads is "
             "still an x0, converted separately from the flow-space rollout state.",
    )
    parser.add_argument(
        "--num_frame_per_block",
        type=int,
        default=3,
        help="Number of frames per block for Self-Forcing causal training"
    )
    parser.add_argument(
        "--flex_forcing",
        action="store_true",
        help="Enable Flex-Forcing (arXiv 2607.03509): replace the scalar "
             "--num_frame_per_block with a partition of the frame axis that is "
             "re-drawn every iteration. Instantiates "
             "WanTransformer3DModel_FlexForcing and validates with "
             "WanFlexForcingPipeline."
    )
    parser.add_argument(
        "--flex_chunk_min",
        type=int,
        default=2,
        help="Smallest chunk drawn when sampling a partition (paper 3.1 uses 2). "
             "1 is allowed but adds no coverage: refining a 2-frame chunk already "
             "yields 1-frame leaves."
    )
    parser.add_argument(
        "--flex_chunk_max",
        type=int,
        default=10,
        help="Largest chunk drawn when sampling a partition (paper 3.1 uses 10; "
             "setting it equal to --flex_chunk_min pins one fixed layout). Keep it "
             "below the latent frame count: the whole-clip layout is already "
             "covered by the pyramid mixture, so raising max to the frame count "
             "only spends draws on a layout you already have and thins out 3.1 - "
             "at 21 latent frames max=21 leaves 3649 distinct partitions versus "
             "4882 at max=10."
    )
    parser.add_argument(
        "--flex_pyramid_levels",
        type=int,
        default=1,
        help="1 = a single partition per iteration (3.1/3.3). >1 = nest it into a "
             "coarse-to-fine ladder, one level per denoising step (3.2). Both "
             "training paths roll the ladder out: block-mask swaps the partition in "
             "per step, --use_kv_cache_training descends it span by span the way "
             "inference does. Capped at the number of denoising steps."
    )
    parser.add_argument(
        "--flex_min_num_frame_per_block",
        type=int,
        default=1,
        help="Block size the 3.2 pyramid stops refining at: every chunk is "
             "binary-split until it is at or below this. 1 = fully causal leaves."
    )
    parser.add_argument(
        "--flex_self_generated_context",
        action="store_true",
        help="Feed the 3.3 K-Projection the model's own previous-step x0 prediction "
             "as the clean context, instead of ground-truth video. Needed when the "
             "block-mask path trains on prompts alone: it builds no KV cache, so "
             "without this there is no clean context at all and the K-Projection "
             "never enters the autograd graph. Doubles the attention sequence. "
             "Inert under --use_kv_cache_training: that path passes no flex_state, "
             "so the K-Projection is never invoked on either side and stays at its "
             "identity init. Also ignored when real --use_teacher_forcing data is "
             "present, and with --flex_forcing off."
    )
    parser.add_argument(
        "--independent_first_frame",
        action="store_true",
        help="Whether first frame is independent ([1, N, N, ...] pattern)"
    )
    parser.add_argument(
        "--use_kv_cache_training",
        action="store_true",
        help="Use KV cache block-by-block training (matches original Self-Forcing)"
    )
    parser.add_argument(
        "--score_num_frames",
        type=int,
        default=21,
        help="Number of latent frames for score computation window (default: 21, matching base model). "
             "fake_score/real_score always receive this many frames."
    )
    parser.add_argument(
        "--min_length_prob_bias",
        type=float,
        default=0.0,
        help="Probability bias for sampling the minimum length (score_num_frames). "
             "0.0 = uniform sampling (default), 0.5 = 50%% prob for min length, "
             "remaining prob distributed equally among longer lengths. "
             "Use this to increase 21-frame training ratio."
    )
    parser.add_argument(
        "--context_noise",
        type=int,
        default=0,
        help="Context noise level for KV cache update (matches training config)"
    )
    parser.add_argument(
        "--use_teacher_forcing",
        action="store_true",
        help="Enable teacher forcing training (pass clean_x to transformer)"
    )
    parser.add_argument(
        "--teacher_forcing_prob",
        type=float,
        default=1.0,
        help="Probability of applying teacher forcing per step (1.0 = always)"
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

    if args.flex_forcing:
        if args.flex_chunk_min < 1 or args.flex_chunk_max < args.flex_chunk_min:
            raise ValueError(
                f"--flex_chunk_min/--flex_chunk_max must satisfy 1 <= min <= max, "
                f"got {args.flex_chunk_min}/{args.flex_chunk_max}.")

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
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        vae.eval()
        # Get Clip Image Encoder
        if args.train_mode != "normal":
            clip_image_encoder = CLIPModel.from_pretrained(
                os.path.join(args.pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
            )
            clip_image_encoder = clip_image_encoder.eval()
        else:
            clip_image_encoder = None
            
    # Get Transformer
    # Flex-Forcing (arXiv 2607.03509) swaps in a generator that accepts a
    # per-iteration partition of the frame axis and carries the K-Projection of
    # 3.3. The two score models stay plain `WanTransformer3DModel`: they are the
    # bidirectional teacher / critic, exactly as in Self-Forcing.
    generator_transformer_cls = (
        WanTransformer3DModel_FlexForcing if args.flex_forcing
        else WanTransformer3DModel_SelfForcing)
    generator_transformer_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
    # 3.3 is not a launcher knob: which projection the generator gets comes from
    # the model config (`transformer_additional_kwargs.flex_kproj_mode`, whose
    # default is the paper's 'diag_rank1'). To ablate it, set that key to 'none'
    # in the config instead of adding a flag back.
    generator_transformer3d = generator_transformer_cls.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=generator_transformer_kwargs,
        low_cpu_mem_usage=True,
    ).to(weight_dtype)
    real_score_transformer3d = WanTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
    ).to(weight_dtype)
    fake_score_transformer3d = WanTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
    ).to(weight_dtype)

    # Freeze vae and text_encoder and set generator_transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    generator_transformer3d.requires_grad_(False)
    real_score_transformer3d.requires_grad_(False)
    fake_score_transformer3d.requires_grad_(False)
    if args.train_mode != "normal":
        clip_image_encoder.requires_grad_(False)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        state_dict = state_dict["generator_ema"] if "generator_ema" in state_dict else state_dict
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict.items()}

        m, u = generator_transformer3d.load_state_dict(state_dict, strict=False)
        m, u = real_score_transformer3d.load_state_dict(state_dict, strict=False)
        m, u = fake_score_transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    if args.ode_transformer_path is not None:
        print(f"From ode checkpoint: {args.ode_transformer_path}")
        if args.ode_transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.ode_transformer_path)
        else:
            state_dict = torch.load(args.ode_transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        state_dict = state_dict["generator_ema"] if "generator_ema" in state_dict else state_dict
        state_dict = state_dict["generator"] if "generator" in state_dict else state_dict
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict.items()}

        m, u = generator_transformer3d.load_state_dict(state_dict, strict=False)
        print(f"ode_transformer_path loaded into generator_transformer3d. missing keys: {len(m)}, unexpected keys: {len(u)}")
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
    if args.flex_forcing and any("flex_kproj" in n
                                 for n, _ in generator_transformer3d.named_parameters()):
        # Whatever projection the config asked for has to end up trainable: the
        # launchers' default --trainable_modules ['.'] already matches it, but a
        # narrowed list would silently freeze Pi at its identity initialisation.
        for name, param in generator_transformer3d.named_parameters():
            if "flex_kproj" in name:
                param.requires_grad = True
        if accelerator.is_main_process:
            accelerator.print(
                "Flex-Forcing K-Projection "
                f"('{getattr(generator_transformer3d, 'flex_kproj_mode', 'none')}') "
                "trainable."
            )
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

                    # The generator is a Flex-Forcing transformer while the critic is
                    # a plain one, yet both accelerators register this same hook.
                    # Loading through the wrong class would drop the K-Projection
                    # tensors (3.3) on a strict `load_state_dict` and overwrite
                    # `flex_kproj_mode` in the config, so pick the class of the model
                    # actually being loaded. Check the Flex subclass first: it
                    # inherits from the plain one, so `isinstance` matches both ways.
                    unwrapped = model.module if hasattr(model, "module") else model
                    model_cls = (
                        WanTransformer3DModel_FlexForcing
                        if isinstance(unwrapped, WanTransformer3DModel_FlexForcing)
                        else WanTransformer3DModel
                    )

                    # load diffusers style into model
                    load_model = model_cls.from_pretrained(
                        input_dir, subfolder="transformer"
                    )
                    unwrapped.register_to_config(**load_model.config)

                    unwrapped.load_state_dict(load_model.state_dict())
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
        # 3.3's K-Projection is grouped like every other layer: the launcher's
        # default --trainable_modules ['.'] matches `flex_kproj.*`, so Pi trains
        # at --learning_rate together with the generator, which is the rate the
        # paper uses for it.
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
    
    if args.fix_sample_size is not None and args.enable_bucket:
        args.video_sample_size = max(max(args.fix_sample_size), args.video_sample_size)
        args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
        args.training_with_video_token_length = False
        args.random_hw_adapt = False

    # Get the dataset
    if args.train_mode != "normal" or args.use_teacher_forcing:
        train_dataset = ImageVideoDataset(
            args.train_data_meta, args.train_data_dir,
            video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride, video_sample_n_frames=args.video_sample_n_frames, 
            video_repeat=args.video_repeat, 
            image_sample_size=args.image_sample_size,
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

    if (args.enable_bucket and args.train_mode != "normal") or args.use_teacher_forcing:
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

            # KV cache training requires latent frames divisible by num_frame_per_block
            if args.use_kv_cache_training:
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
                    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" for text in new_examples['text']
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
    elif args.train_mode == "normal":
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
                    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" for text in new_examples['text']
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
        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype)
        real_score_transformer3d = shard_fn(real_score_transformer3d)
    if fsdp_stage != 0 or zero_stage != 0:
        from functools import partial

        from videox_fun.dist import set_multi_gpus_devices, shard_model
        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype)
        text_encoder = shard_fn(text_encoder)

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    real_score_transformer3d.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    if not args.enable_text_encoder_in_dataloader:
        text_encoder.to(accelerator.device if not args.low_vram else "cpu")
    if args.train_mode != "normal":
        clip_image_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)

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

    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    def randomize_denoising_step_indices(
        denoising_step_indices_list,
        train_sampling_steps,
        torch_rng,
        accelerator,
        jitter_ratio=0.3,
        tail_margin=1,
    ):
        indices = list(denoising_step_indices_list)
        n = len(indices)
        tail_margin = max(int(tail_margin), 1)

        # The head stays fixed at the pure-noise start; the remaining steps jitter
        # symmetrically around their base values so the expected schedule is unchanged.
        result = [indices[0]]
        for i in range(1, n):
            gap_upper = indices[i - 1] - indices[i]
            if i + 1 < n:
                gap_lower = indices[i] - indices[i + 1]
                max_jitter = int(min(gap_upper, gap_lower) * jitter_ratio)
            else:
                # Tail step: no lower neighbor (the base value sits at the clean end),
                # so the jitter budget comes from the upward gap and the downward side
                # is clamped by tail_margin. Keeping the tail off the schedule's cleanest
                # position guarantees Decoupled DMD's tau_CA always has a cleaner slot.
                max_jitter = int(gap_upper * jitter_ratio)

            if max_jitter > 0:
                # NB: torch_rng may live on CUDA while randint's default output is CPU;
                # use the global CPU RNG here, the result is broadcast from rank 0 anyway.
                jitter = torch.randint(
                    -max_jitter, max_jitter + 1, (1,)
                ).item()
            else:
                jitter = 0

            value = indices[i] + jitter
            # Strict monotonicity by construction (no post-hoc clamp repair).
            value = min(value, result[i - 1] - 1)
            if i == n - 1:
                value = max(value, tail_margin)
            result.append(value)

        result = [max(1, min(train_sampling_steps, x)) for x in result]
        result = torch.tensor(result)

        if dist.is_initialized():
            result = result.to(accelerator.device)
            dist.broadcast(result, src=0)
            result = result.cpu()
        return result

    for epoch in range(first_epoch, args.num_train_epochs):
        train_dmd_loss = 0.0
        # Number of generator backward contributions since the last log flush; the
        # generator only backprops every gen_update_interval batches, so its metrics
        # must be averaged by contribution count, not by gradient_accumulation_steps.
        train_gen_log_count = 0
        train_denoising_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            # Data batch sanity check
            if args.train_mode != "normal" and epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.mp4", rescale=True)

                clip_pixel_values, mask_pixel_values, texts = batch['clip_pixel_values'].cpu(), batch['mask_pixel_values'].cpu(), batch['text']
                mask_pixel_values = rearrange(mask_pixel_values, "b f c h w -> b c f h w")
                for idx, (clip_pixel_value, pixel_value, text) in enumerate(zip(clip_pixel_values, mask_pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    Image.fromarray(np.uint8(clip_pixel_value)).save(f"{args.output_dir}/sanity_check/clip_{gif_name[:10] if not text == '' else f'{global_step}-{idx}'}.png")
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/mask_{gif_name[:10] if not text == '' else f'{global_step}-{idx}'}.mp4", rescale=True)

            with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                if args.train_mode != "normal" or args.use_teacher_forcing:
                    # Convert images to latent space
                    pixel_values = batch["pixel_values"].to(weight_dtype)

                    # Increase the batch size when the length of the latent sequence of the current sample is small
                    if args.auto_tile_batch_size and args.training_with_video_token_length and zero_stage != 3:
                        if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            pixel_values = torch.tile(pixel_values, (4, 1, 1, 1, 1))
                            if args.enable_text_encoder_in_dataloader:
                                batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (4, 1, 1))
                                batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (4, 1))
                                batch['neg_encoder_hidden_states'] = torch.tile(batch['neg_encoder_hidden_states'], (4, 1, 1))
                                batch['neg_encoder_attention_mask'] = torch.tile(batch['neg_encoder_attention_mask'], (4, 1))
                            else:
                                batch['text'] = batch['text'] * 4
                        elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            pixel_values = torch.tile(pixel_values, (2, 1, 1, 1, 1))
                            if args.enable_text_encoder_in_dataloader:
                                batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (2, 1, 1))
                                batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (2, 1))
                                batch['neg_encoder_hidden_states'] = torch.tile(batch['neg_encoder_hidden_states'], (2, 1, 1))
                                batch['neg_encoder_attention_mask'] = torch.tile(batch['neg_encoder_attention_mask'], (2, 1))
                            else:
                                batch['text'] = batch['text'] * 2
                    if args.train_mode != "normal":
                        clip_pixel_values = batch["clip_pixel_values"].to(weight_dtype)
                        mask_pixel_values = batch["mask_pixel_values"].to(weight_dtype)
                        mask = batch["mask"].to(weight_dtype)
                        # Increase the batch size when the length of the latent sequence of the current sample is small
                        if args.auto_tile_batch_size and args.training_with_video_token_length and zero_stage != 3:
                            if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                                clip_pixel_values = torch.tile(clip_pixel_values, (4, 1, 1, 1))
                                mask_pixel_values = torch.tile(mask_pixel_values, (4, 1, 1, 1, 1))
                                mask = torch.tile(mask, (4, 1, 1, 1, 1))
                            elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                                clip_pixel_values = torch.tile(clip_pixel_values, (2, 1, 1, 1))
                                mask_pixel_values = torch.tile(mask_pixel_values, (2, 1, 1, 1, 1))
                                mask = torch.tile(mask, (2, 1, 1, 1, 1))

                    if args.random_frame_crop:
                        def _create_special_list(length):
                            if length == 1:
                                return [1.0]
                            if length >= 2:
                                last_element = 0.90
                                remaining_sum = 1.0 - last_element
                                other_elements_value = remaining_sum / (length - 1)
                                special_list = [other_elements_value] * (length - 1) + [last_element]
                                return special_list
                        select_frames = [_tmp for _tmp in list(range(sample_n_frames_bucket_interval + 1, args.video_sample_n_frames + sample_n_frames_bucket_interval, sample_n_frames_bucket_interval))]
                        select_frames_prob = np.array(_create_special_list(len(select_frames)))
                        
                        if len(select_frames) != 0:
                            if rng is None:
                                temp_n_frames = np.random.choice(select_frames, p = select_frames_prob)
                            else:
                                temp_n_frames = rng.choice(select_frames, p = select_frames_prob)
                        else:
                            temp_n_frames = 1

                        # Magvae needs the number of frames to be 4n + 1.
                        temp_n_frames = (temp_n_frames - 1) // sample_n_frames_bucket_interval + 1

                        pixel_values = pixel_values[:, :temp_n_frames, :, :]
                        mask_pixel_values = mask_pixel_values[:, :temp_n_frames, :, :]
                        mask = mask[:, :temp_n_frames, :, :]
                        
                    # Keep all node same token length to accelerate the traning when resolution grows.
                    if args.keep_all_node_same_token_length:
                        if args.token_sample_size > 256:
                            numbers_list = list(range(256, args.token_sample_size + 1, 128))

                            if numbers_list[-1] != args.token_sample_size:
                                numbers_list.append(args.token_sample_size)
                        else:
                            numbers_list = [256]
                        numbers_list = [_number * _number * args.video_sample_n_frames for _number in  numbers_list]
                
                        actual_token_length = index_rng.choice(numbers_list)
                        actual_video_length = (min(
                                actual_token_length / pixel_values.size()[-1] / pixel_values.size()[-2], args.video_sample_n_frames
                        ) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1
                        actual_video_length = int(max(actual_video_length, 1))

                        # Magvae needs the number of frames to be 4n + 1.
                        actual_video_length = (actual_video_length - 1) // sample_n_frames_bucket_interval + 1

                        pixel_values = pixel_values[:, :actual_video_length, :, :]
                        mask_pixel_values = mask_pixel_values[:, :actual_video_length, :, :]
                        mask = mask[:, :actual_video_length, :, :]

                    if args.low_vram:
                        torch.cuda.empty_cache()
                        vae.to(accelerator.device)
                        if args.train_mode != "normal":
                            clip_image_encoder.to(accelerator.device)
                        real_score_transformer3d = real_score_transformer3d.to("cpu")
                        if not args.enable_text_encoder_in_dataloader:
                            text_encoder.to("cpu")

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
                        if args.use_teacher_forcing:
                            clean_latents = _batch_encode_vae(pixel_values)
                        else:
                            clean_latents = None

                        if args.train_mode != "normal":
                            # Encode inpaint latents.
                            mask_latents = _batch_encode_vae(mask_pixel_values)
                            if vae_stream_2 is not None:
                                torch.cuda.current_stream().wait_stream(vae_stream_2) 

                            mask = rearrange(mask, "b f c h w -> b c f h w")
                            mask = torch.concat(
                                [
                                    torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2), 
                                    mask[:, :, 1:]
                                ], dim=2
                            )
                            mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4])
                            mask = mask.transpose(1, 2)
                            mask = resize_mask(1 - mask, mask_latents)

                            inpaint_latents = torch.concat([mask, mask_latents], dim=1)

                            clip_context = []
                            for clip_pixel_value in clip_pixel_values:
                                clip_image = Image.fromarray(np.uint8(clip_pixel_value.float().cpu().numpy()))
                                clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(clip_image_encoder.device, weight_dtype)
                                _clip_context = clip_image_encoder([clip_image[:, None, :, :]])
                                clip_context.append(_clip_context)
                            clip_context = torch.cat(clip_context)

                    if args.use_teacher_forcing:
                        target_shape = clean_latents.size()
                    else:
                        target_shape = mask_latents.size()
                else:
                    text = batch['text']
                    if args.fix_sample_size is not None:
                        local_sample_size = [int(x / 16) * 16 for x in args.fix_sample_size]
                        num_frames = args.video_sample_n_frames
                    else:
                        if args.random_hw_adapt and args.training_with_video_token_length:
                            # Get token length
                            target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
                            length_to_frame_num = get_length_to_frame_num(target_token_length)

                            if rng is None:
                                local_length = np.random.choice(list(length_to_frame_num.keys()))
                            else:
                                local_length = rng.choice(list(length_to_frame_num.keys()))
                            num_frames = length_to_frame_num[local_length]

                            aspect_ratio_sample_size = {key : [x / 512 * local_length for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                            if rng is None:
                                aspect_ratio_key = np.random.choice(list(aspect_ratio_sample_size.keys()))
                            else:
                                aspect_ratio_key = rng.choice(list(aspect_ratio_sample_size.keys()))
                            local_sample_size = aspect_ratio_sample_size[aspect_ratio_key]
                        else:
                            num_frames = args.video_sample_n_frames

                            aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                            if rng is None:
                                aspect_ratio_key = np.random.choice(list(aspect_ratio_sample_size.keys()))
                            else:
                                aspect_ratio_key = rng.choice(list(aspect_ratio_sample_size.keys()))
                            local_sample_size = aspect_ratio_sample_size[aspect_ratio_key]
                        local_sample_size = [int(x / 16) * 16 for x in local_sample_size]

                    # Compute latent frame count
                    latent_num_frames = int((num_frames - 1) // vae.temporal_compression_ratio + 1)

                    # Align latent_num_frames to num_frame_per_block for KV cache training
                    if args.use_kv_cache_training:
                        if args.independent_first_frame:
                            # latent_frames - 1 must be divisible by num_frame_per_block
                            k = latent_num_frames - 1
                            k = (k // args.num_frame_per_block) * args.num_frame_per_block
                            latent_num_frames = k + 1
                        else:
                            # latent_frames must be divisible by num_frame_per_block
                            latent_num_frames = (latent_num_frames // args.num_frame_per_block) * args.num_frame_per_block
                        latent_num_frames = max(latent_num_frames, args.num_frame_per_block)

                    target_shape = (
                        len(text),
                        vae.latent_channels, 
                        latent_num_frames, 
                        int(local_sample_size[0] // vae.spatial_compression_ratio),
                        int(local_sample_size[1] // vae.spatial_compression_ratio), 
                    )
                    clean_latents = None

                if args.low_vram:
                    vae.to('cpu')
                    real_score_transformer3d = real_score_transformer3d.to("cpu")
                    if args.train_mode != "normal":
                        clip_image_encoder.to('cpu')
                    torch.cuda.empty_cache()
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to(accelerator.device)

                if args.enable_text_encoder_in_dataloader:
                    prompt_embeds = batch['encoder_hidden_states'].to(device=accelerator.device)
                    neg_prompt_embeds = batch['neg_encoder_hidden_states'].to(device=accelerator.device)
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
                        prompt_embeds = text_encoder(text_input_ids.to(accelerator.device), attention_mask=prompt_attention_mask.to(accelerator.device))[0]
                        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

                        neg_txt = [
                            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" for text in batch['text']
                        ]
                        neg_prompt_ids = tokenizer(
                            neg_txt, 
                            padding="max_length", 
                            max_length=args.tokenizer_max_length, 
                            truncation=True, 
                            add_special_tokens=True, 
                            return_tensors="pt"
                        )
                        neg_text_input_ids = neg_prompt_ids.input_ids
                        neg_prompt_attention_mask = neg_prompt_ids.attention_mask

                        neg_seq_lens = neg_prompt_attention_mask.gt(0).sum(dim=1).long()
                        neg_prompt_embeds = text_encoder(neg_text_input_ids.to(accelerator.device), attention_mask=neg_prompt_attention_mask.to(accelerator.device))[0]
                        neg_prompt_embeds = [u[:v] for u, v in zip(neg_prompt_embeds, neg_seq_lens)]

                if args.low_vram:
                    generator_transformer3d = generator_transformer3d.to(accelerator.device)
                    real_score_transformer3d = real_score_transformer3d.to(accelerator.device)
                    fake_score_transformer3d = fake_score_transformer3d.to(accelerator.device)
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to('cpu')
                        torch.cuda.empty_cache()

            generator_update = step % args.gen_update_interval == 0
            # Enter the generator's accumulation context only on batches that actually
            # backprop through the generator. Entering it on every batch would advance
            # the accumulation counter gen_update_interval times faster than real
            # generator gradients are produced; whenever gcd(gradient_accumulation_steps,
            # gen_update_interval) > 1 the sync flag would then never coincide with a
            # generator-update batch and optimizer.step() would silently never fire.
            generator_accumulate_ctx = (
                accelerator.accumulate(generator_transformer3d) if generator_update else contextlib.nullcontext()
            )
            with generator_accumulate_ctx:
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

                def add_noise(latents, noise, timesteps):
                    sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                    return (1.0 - sigmas) * latents + sigmas * noise

                def generate_and_sync_list(num_denoising_steps, device):
                    indices = torch.randint(low=0, high=num_denoising_steps, size=(1,), generator=torch_rng, device=device)
                    if dist.is_initialized():
                        dist.broadcast(indices, src=0)
                    return indices.tolist()

                def convert_flow_pred_to_x0(
                    scheduler,
                    flow_pred: torch.Tensor,
                    xt: torch.Tensor,
                    timestep: torch.Tensor
                ) -> torch.Tensor:
                    """
                    Convert flow matching's prediction to x0 prediction.
                    Supports both 4D [B, C, H, W] and 5D [B, C, F, H, W] inputs.
                    """
                    original_dtype = flow_pred.dtype
                    device = flow_pred.device

                    flow_pred = flow_pred.double()
                    xt = xt.double()
                    timesteps = scheduler.timesteps.to(device).double()
                    sigmas = scheduler.sigmas.to(device).double()
                    timestep = timestep.to(device).double()

                    timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
                    sigma_t = sigmas[timestep_id]

                    ndim = flow_pred.ndim
                    if ndim == 4:
                        sigma_t = sigma_t.view(-1, 1, 1, 1)
                    elif ndim == 5:
                        sigma_t = sigma_t.view(-1, 1, 1, 1, 1)
                    else:
                        raise ValueError(f"Expected 4D or 5D input, got {ndim}D tensor.")

                    x0_pred = xt - sigma_t * flow_pred
                    return x0_pred.to(original_dtype)

                # Create discrete denoising steps (per-step, with optional randomization)
                if getattr(args, 'randomize_step_indices', False):
                    random_indices = randomize_denoising_step_indices(
                        args.denoising_step_indices_list,
                        args.train_sampling_steps,
                        torch_rng,
                        accelerator,
                        jitter_ratio=args.index_jitter_ratio,
                    )
                else:
                    random_indices = torch.tensor(args.denoising_step_indices_list)

                denoising_step_list = noise_scheduler.timesteps[args.train_sampling_steps - random_indices]

                # --- Main Training Logic ---
                bsz, channel, num_frames, height, width = target_shape
                # Flex-Forcing partition ladder for this iteration (arXiv
                # 2607.03509 3.1-3.3), drawn by whichever rollout branch runs
                # below. The critic draws its own rather than reusing this one
                # because it samples its own clip length. None = Flex-Forcing off.
                flex_partitions = None
                if generator_update:  # generator_update computed before the accumulate ctx above
                    if args.use_kv_cache_training:
                        # Calculate frame_seq_length
                        patch_h, patch_w = accelerator.unwrap_model(generator_transformer3d).config.patch_size[1:]
                        frame_seq_length = (target_shape[3] * target_shape[4]) // (patch_h * patch_w)
                        
                        # Determine block structure with variable-length support
                        if not args.independent_first_frame:
                            assert num_frames % args.num_frame_per_block == 0
                            max_num_blocks = num_frames // args.num_frame_per_block
                            assert args.score_num_frames % args.num_frame_per_block == 0
                            min_num_blocks = args.score_num_frames // args.num_frame_per_block
                        else:
                            assert (num_frames - 1) % args.num_frame_per_block == 0
                            max_num_blocks = (num_frames - 1) // args.num_frame_per_block
                            if args.score_num_frames > 1:
                                assert (args.score_num_frames - 1) % args.num_frame_per_block == 0
                                min_num_blocks = (args.score_num_frames - 1) // args.num_frame_per_block
                            else:
                                min_num_blocks = 0

                        # Random sample number of blocks (Self-Forcing variable-length training)
                        if args.min_length_prob_bias > 0.0 and max_num_blocks > min_num_blocks:
                            # Weighted sampling: give min_num_blocks a higher probability
                            num_options = max_num_blocks - min_num_blocks + 1
                            bias = min(args.min_length_prob_bias, 0.99)
                            remaining_prob = (1.0 - bias) / (num_options - 1)
                            probs = [remaining_prob] * num_options
                            probs[0] = bias  # min_num_blocks gets the bias
                            probs_tensor = torch.tensor(probs, device=accelerator.device)
                            block_indices = torch.multinomial(probs_tensor, 1, generator=torch_rng)
                            num_generated_blocks = (min_num_blocks + block_indices).item()
                        else:
                            num_generated_blocks = torch.randint(
                                min_num_blocks, max_num_blocks + 1, (1,),
                                generator=torch_rng, device=accelerator.device
                            ).item()
                        if dist.is_initialized():
                            _sync = torch.tensor([num_generated_blocks], device=accelerator.device)
                            dist.broadcast(_sync, src=0)
                            num_generated_blocks = _sync.item()

                        all_num_frames = [args.num_frame_per_block] * num_generated_blocks
                        if args.independent_first_frame:
                            all_num_frames = [1] + all_num_frames

                        num_generated_frames = sum(all_num_frames)
                        # 3.1/3.2: keep the sampled clip length but re-cut the
                        # frame axis into this iteration's partition ladder - the
                        # same draw the block-mask path makes, so both training
                        # modes cover the same layouts. Level 0 is where the walk
                        # below starts; the finer levels are the sub-spans it
                        # descends into. Returns None with Flex-Forcing off, and
                        # broadcasts level 0 itself, so every rank rolls out the
                        # same layout.
                        flex_partitions = sample_flex_partitions(
                            args, num_generated_frames, len(denoising_step_list),
                            torch_rng, accelerator.device,
                            accelerator.is_main_process)
                        if flex_partitions is not None:
                            all_num_frames = flex_partitions[0]
                            assert sum(all_num_frames) == num_generated_frames
                        
                        # Initialize KV cache
                        num_layers = generator_transformer3d.config.num_layers
                        num_heads = generator_transformer3d.config.num_heads
                        head_dim = generator_transformer3d.config.dim // num_heads
                        text_len = 512  # T5 sequence length
                        
                        kv_cache = initialize_kv_cache_for_training(
                            batch_size=bsz,
                            num_frames=num_frames,
                            frame_seq_length=frame_seq_length,
                            num_layers=num_layers,
                            num_heads=num_heads,
                            head_dim=head_dim,
                            dtype=weight_dtype,
                            device=accelerator.device
                        )
                        
                        crossattn_cache = initialize_crossattn_cache_for_training(
                            batch_size=bsz,
                            text_len=text_len,
                            num_layers=num_layers,
                            num_heads=num_heads,
                            head_dim=head_dim,
                            dtype=weight_dtype,
                            device=accelerator.device
                        )
                        
                        # Block-by-block generation
                        generator_noise = torch.randn(target_shape, device=accelerator.device, generator=torch_rng, dtype=weight_dtype)
                        num_input_frames = 0  # T2V mode
                        
                        # Use actual batch size from generator_noise (may differ due to SP)
                        actual_bsz = generator_noise.shape[0]
                        output_pred = torch.zeros_like(generator_noise)
                        
                        # Decide whether to use teacher forcing for this video (once per video, not per block)
                        use_teacher_forcing_step = (
                            args.use_teacher_forcing and 
                            torch.rand(1, generator=torch_rng, device=accelerator.device).item() < args.teacher_forcing_prob
                        )
                        
                        # Same exit step across all spans (matches original Self-Forcing default)
                        num_denoising_steps = len(denoising_step_list)
                        final_step_index = generate_and_sync_list(num_denoising_steps, device=accelerator.device)[0]
                        
                        # Only spans in the last score_num_frames get gradient at exit step
                        # (matches Self-Forcing: start_gradient_frame_index = num_output_frames - 21)
                        start_gradient_frame_index = num_generated_frames - args.score_num_frames
                        
                        # The walk, mirroring the one in
                        # `pipeline_wan_flex_forcing.py` so that training and
                        # inference roll the same ladder out the same way. A work
                        # item is (span, step_idx, commit): a frame range, the
                        # schedule position it sits at, and whether its x0 has to
                        # go into the KV cache once it is done. Two rules, from
                        # 3.2:
                        #
                        #   * the span still splits at the next level -> one
                        #     *buffered* step over the whole span, re-noised back
                        #     into `generator_noise`, then resume the sub-spans.
                        #     Pushing them back reversed keeps the walk in
                        #     temporal order, which the cache's frame bookkeeping
                        #     relies on;
                        #   * otherwise -> run the rest of the schedule up to the
                        #     exit step over the span in one go, and that last
                        #     step is the gradient-carrying one.
                        #
                        # A single-level ladder never splits (`subs` is the span
                        # itself), so this degenerates exactly to the block-major
                        # Self-Forcing loop it replaces.
                        ladder = (flex_partitions if flex_partitions is not None
                                  else [all_num_frames])
                        # The two things the inference rollout does before it
                        # touches the cache, mirrored here. The nesting check is
                        # not decoration: this walk overwrites a coarse span's
                        # cache slot with its own sub-spans, which is only safe
                        # because every finer chunk falls inside its parent. The
                        # scalar chunk size is what every self-attn layer copies
                        # out of the model on each KV forward (and what
                        # Forcing-KV's AR stride then divides by), so it has to
                        # describe this iteration's level 0 and not the
                        # launch-time value; the block-mask path already keeps it
                        # fresh through `set_flex_chunk_sizes`.
                        validate_nested_partitions(ladder, num_generated_frames)
                        accelerator.unwrap_model(
                            generator_transformer3d
                        ).num_frame_per_block = max(ladder[0])
                        top = chunk_boundaries(ladder[0])
                        stack = [(span, 0, idx < len(top) - 1)
                                 for idx, span in reversed(list(enumerate(top)))]
                        while stack:
                            (span_start, span_end), step_idx, commit = stack.pop()
                            # The partition one level down, i.e. the sub-spans this
                            # span would split into. Clamped: a ladder shorter
                            # than the schedule keeps its finest level, and then
                            # every span is its own only sub-span and runs to the
                            # exit step as a leaf.
                            level = ladder[min(step_idx + 1, len(ladder) - 1)]
                            subs = [s for s in chunk_boundaries(level)
                                    if s[0] >= span_start and s[1] <= span_end]
                            splits = step_idx < final_step_index and len(subs) > 1
                            # A splitting span takes a single buffered step; a leaf
                            # runs whatever is left of the schedule up to the exit
                            # step. `step_idx` is where the slice starts, which is
                            # what lets the re-noising below read the right
                            # (t_i, t_i+1) pair when the walk resumes a span
                            # mid-schedule.
                            schedule = (denoising_step_list[step_idx:step_idx + 1] if splits
                                        else denoising_step_list[step_idx:final_step_index + 1])

                            current_num_frames = span_end - span_start
                            current_start_frame = span_start
                            # Extract noise for the current span, out of the shared
                            # buffer its parent's buffered step already re-noised.
                            start_idx = current_start_frame - num_input_frames
                            end_idx = start_idx + current_num_frames
                            noisy_input = generator_noise[:, :, start_idx:end_idx]
                            
                            for local_idx, current_timestep in enumerate(schedule):
                                global_idx = step_idx + local_idx
                                is_final_step = (global_idx == final_step_index)
                                timestep = torch.full(
                                    [bsz, current_num_frames],
                                    current_timestep,
                                    device=noisy_input.device,
                                    dtype=torch.int64
                                )
                                
                                # Gradient only on the exit step, and only on a
                                # span that reaches into the scored window. The
                                # test is on the *overlap*, not on the span's
                                # start: under a coarse layout a span can be the
                                # whole clip, and a start-based test would then
                                # drop the gradient from every forward at once,
                                # leaving the DMD loss with nothing to
                                # differentiate. For a uniform partition the two
                                # agree, since both the window offset and every
                                # span start are multiples of the block width.
                                if not is_final_step or span_end <= start_gradient_frame_index:
                                    context_manager = torch.no_grad()
                                else:
                                    context_manager = contextlib.nullcontext()
                                
                                with context_manager:
                                    # Convert noisy_input to list format
                                    noisy_input_list = [noisy_input[i] for i in range(bsz)]
                                    
                                    # Use full seq_len (consistent with inference code)
                                    full_seq_len = frame_seq_length * num_frames
                                    
                                    generator_pred_block = generator_transformer3d(
                                        x=noisy_input_list,
                                        context=prompt_embeds,
                                        t=timestep,
                                        seq_len=full_seq_len,
                                        kv_cache=kv_cache,
                                        crossattn_cache=crossattn_cache,
                                        current_start=current_start_frame * frame_seq_length,
                                        cache_start=None,
                                        y=inpaint_latents if args.train_mode != "normal" else None,
                                        clip_fea=clip_context if args.train_mode != "normal" else None,
                                    )
                                    
                                    # Stack list output to tensor: [B, C, F, H, W]
                                    if isinstance(generator_pred_block, list):
                                        generator_pred_block = torch.stack(generator_pred_block, dim=0)
                                    
                                    # Flatten timestep for convert_flow_pred_to_x0: [B, F] -> [B*F]
                                    if not args.flow_euler_rollout or is_final_step:
                                        generator_pred_block = convert_flow_pred_to_x0(
                                            scheduler=noise_scheduler,
                                            flow_pred=generator_pred_block,
                                            xt=noisy_input,
                                            timestep=timestep[:, 0]
                                        )
                                
                                if is_final_step:
                                    break
                                
                                # Add noise for next step
                                next_timestep = denoising_step_list[global_idx + 1] * torch.ones(
                                    bsz, dtype=torch.long, device=noisy_input.device
                                )
                                if args.flow_euler_rollout:
                                    # Same Euler ODE step as the block-mask path below.
                                    sigma_t = get_sigmas(timestep[:, 0], n_dim=noisy_input.ndim, dtype=torch.float32)
                                    sigma_next = get_sigmas(next_timestep, n_dim=noisy_input.ndim, dtype=torch.float32)
                                    noisy_input = (
                                        noisy_input.float() + (sigma_next - sigma_t) * generator_pred_block.float()
                                    ).to(noisy_input.dtype)
                                else:
                                    noisy_input = add_noise(
                                        generator_pred_block,
                                        torch.randn(generator_pred_block.shape, dtype=generator_pred_block.dtype, device=generator_pred_block.device, generator=torch_rng),
                                        next_timestep
                                    )
                            
                            if splits:
                                # The next level consumes the *re-noised* buffer,
                                # not x0: the sub-spans pick the schedule up where
                                # this step left it. Written back into the shared
                                # noise buffer so each sub-span reads its own slice
                                # of it. The trailing sub-chunk only needs
                                # committing when the parent span itself is
                                # followed by a sibling at some higher level.
                                generator_noise[:, :, start_idx:end_idx] = noisy_input
                                stack.extend(reversed([
                                    (sub, step_idx + 1, k < len(subs) - 1 or commit)
                                    for k, sub in enumerate(subs)]))
                                continue
                            
                            # Record output
                            output_pred[:, :, current_start_frame:current_start_frame + current_num_frames] = generator_pred_block
                            
                            # Update KV cache with clean context (consistent with inference: feed denoised_pred directly).
                            # Leaves only: a span that split already had its
                            # sub-spans committed at their own finer granularity,
                            # and re-committing it as one coarse chunk would
                            # rewrite those keys under an attention pattern their
                            # x0 was never produced with.
                            if commit:
                                context_timestep = torch.ones([bsz, current_num_frames], device=accelerator.device, dtype=torch.int64) * args.context_noise
                                
                                # Use clean latents for teacher forcing, otherwise use denoised prediction directly
                                if use_teacher_forcing_step and clean_latents is not None:
                                    context_input = clean_latents[:, :, start_idx:end_idx]
                                else:
                                    context_input = generator_pred_block
                                
                                context_input_list = [context_input[i] for i in range(bsz)]
                                
                                # Use full seq_len (consistent with inference code)
                                full_seq_len = frame_seq_length * num_frames
                                
                                with torch.no_grad():
                                    generator_transformer3d(
                                        x=context_input_list,
                                        context=prompt_embeds,
                                        t=context_timestep,
                                        seq_len=full_seq_len,
                                        kv_cache=kv_cache,
                                        crossattn_cache=crossattn_cache,
                                        current_start=current_start_frame * frame_seq_length,
                                        cache_start=None,
                                        y=inpaint_latents if args.train_mode != "normal" else None,
                                        clip_fea=clip_context if args.train_mode != "normal" else None,
                                    )
                        
                        # Final output — slice generated frames (may be < num_frames for variable-length)
                        generator_pred_full = output_pred[:, :, :num_generated_frames]

                        # Gradient mask: first block gets no gradient when generating > min frames
                        # (matches Self-Forcing reference: model/base.py L182-L190)
                        min_num_frames_score = args.score_num_frames
                        need_gradient_mask = (num_generated_frames != min_num_frames_score)
                        gradient_mask = None
                        if need_gradient_mask:
                            gradient_mask = torch.ones_like(generator_pred_full, dtype=torch.bool)
                            if args.independent_first_frame:
                                gradient_mask[:, :, :1] = False
                            else:
                                # The first *block* gets no gradient, so this has to
                                # track the actual first chunk: identical to
                                # `args.num_frame_per_block` for a uniform layout and
                                # the sampled size under Flex-Forcing. Under a
                                # pyramid it has to be the first *leaf*, not level
                                # 0's first chunk: the coarse planning band makes
                                # that the whole clip, which would mask every frame
                                # and leave the DMD loss averaging over nothing.
                                leaf_sizes = ladder[min(final_step_index, len(ladder) - 1)]
                                # Degenerate case the coarse band can draw: the
                                # first leaf *is* the whole clip, when the exit
                                # step is reached at level 0. There is then no
                                # leading context chunk to protect, and masking it
                                # would clear the score set instead of trimming
                                # its first chunk.
                                if leaf_sizes[0] < num_generated_frames:
                                    gradient_mask[:, :, :leaf_sizes[0]] = False

                        # Slice for score computation: last score_num_frames frames
                        if num_generated_frames > args.score_num_frames:
                            # Re-encode boundary for cleaner score input
                            generator_pred_for_score, score_num_frames, _ = slice_for_score(
                                generator_pred_full, vae, weight_dtype,
                                score_num_frames=args.score_num_frames,
                                independent_first_frame=args.independent_first_frame,
                            )
                        else:
                            generator_pred_for_score = generator_pred_full
                            score_num_frames = num_generated_frames

                        # Compute score_mask for DMD loss (matches Self-Forcing: dmd.py L199-204)
                        score_mask = None
                        if gradient_mask is not None:
                            mask_offset = num_generated_frames - score_num_frames
                            score_mask = gradient_mask[:, :, mask_offset:mask_offset + score_num_frames]

                        # generator_pred = the sliced version for DMD loss
                        generator_pred = generator_pred_for_score
                        seq_len = frame_seq_length * score_num_frames  # Score always on fixed window
                    
                    else:
                        # === Block mask training (flex attention, no KV cache) ===
                        # Block mask training: use flex attention to process entire video at once
                        # Note: for long videos, use KV cache mode instead
                        score_mask = None  # Block mask mode: no gradient mask needed
                        if num_frames > args.score_num_frames:
                            raise ValueError(
                                f"Block mask mode does not support variable-length training "
                                f"(video produces {num_frames} latent frames > score_num_frames={args.score_num_frames}). "
                                f"Use --use_kv_cache_training for long video training."
                            )
                        
                        patch_h_bm, patch_w_bm = accelerator.unwrap_model(generator_transformer3d).config.patch_size[1:]
                        frame_seqlen_bm = (height * width) // (patch_h_bm * patch_w_bm)
                        
                        # Standard backward simulation training
                        generator_noise = torch.randn(target_shape, device=accelerator.device, generator=torch_rng, dtype=weight_dtype)
                        num_denoising_steps = len(denoising_step_list)
                        final_step_index = generate_and_sync_list(num_denoising_steps, device=generator_noise.device)[0]

                        # Precompute seq_len once (same for all steps)
                        seq_len = frame_seqlen_bm * num_frames

                        # Decide whether to use teacher forcing for this step
                        use_teacher_forcing_step = (
                            args.use_teacher_forcing and 
                            torch.rand(1, generator=torch_rng, device=accelerator.device).item() < args.teacher_forcing_prob
                        )
                        
                        # Flex-Forcing: draw this iteration's partition ladder
                        # (3.1/3.3) and install level 0 before the mask builders
                        # below, which dispatch on it. With Flex-Forcing off this
                        # stays None and the inherited uniform masks are built
                        # exactly as before.
                        if flex_partitions is None:
                            flex_partitions = sample_flex_partitions(
                                args, num_frames, len(denoising_step_list), torch_rng,
                                accelerator.device, accelerator.is_main_process)
                        flex_model = accelerator.unwrap_model(generator_transformer3d)
                        install_flex_partition(flex_model, flex_partitions, 0)

                        # Decide clean_x / aug_t first: the mask-builder closure keys
                        # off them, and 3.2 has to be able to rebuild the mask at
                        # every denoising step without repeating this branch.
                        if use_teacher_forcing_step and clean_latents is not None:
                            # Teacher forcing: clean + noisy sequence mask
                            clean_x = [clean_latents[i] for i in range(clean_latents.size(0))]
                            aug_t = torch.zeros(bsz, device=accelerator.device, dtype=torch.int64)
                        else:
                            # Standard causal mask
                            clean_x = None
                            aug_t = None
                        # 3.3 on prompt-only data: there is no ground-truth clean
                        # half, so the loop below substitutes the model's own
                        # previous-step x0 prediction. Decided up front because it
                        # doubles the sequence, i.e. the mask has to be the
                        # teacher-forcing one from the very first step.
                        use_self_context = flex_self_context_enabled(
                            args, clean_x, final_step_index)
                        if use_self_context:
                            aug_t = torch.zeros(bsz, device=accelerator.device, dtype=torch.int64)
                        self_clean_x = None
                        build_block_mask = make_flex_mask_builder(
                            flex_model, args, num_frames, frame_seqlen_bm,
                            accelerator.device,
                            clean_x is not None or use_self_context)
                        build_block_mask()

                        for index, current_timestep in enumerate(denoising_step_list):
                            is_final_step = (index == final_step_index)
                            # 3.2: a pyramid ladder swaps in the finer partition of
                            # this step and rebuilds the mask. Level 0 is already
                            # installed above, hence the `index > 0` guard.
                            if index > 0 and flex_partitions is not None and len(flex_partitions) > 1:
                                install_flex_partition(flex_model, flex_partitions, index)
                                build_block_mask()
                            timestep = torch.full(
                                generator_noise.shape[:1],
                                current_timestep,
                                device=generator_noise.device,
                                dtype=torch.int64
                            )
                            
                            with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                                context_manager = torch.no_grad() if not is_final_step else contextlib.nullcontext()
                                
                                with context_manager:
                                    # Convert to list format for transformer
                                    generator_noise_list = [generator_noise[i] for i in range(bsz)]
                                    if clean_x is not None:
                                        clean_x_list = [clean_latents[i] for i in range(bsz)]
                                    elif use_self_context and self_clean_x is not None:
                                        # 3.3: the previous step's own x0 prediction,
                                        # already detached by the no_grad it was made
                                        # under, so gradients reach Pi's parameters
                                        # without flowing back into the context.
                                        clean_x_list = [self_clean_x[i] for i in range(bsz)]
                                    else:
                                        clean_x_list = None
                                    
                                    # Use block_mask for causal training (一次性处理整个视频)
                                    generator_pred = generator_transformer3d(
                                        x=generator_noise_list,
                                        context=prompt_embeds,
                                        t=timestep,
                                        seq_len=seq_len,
                                        y=inpaint_latents if args.train_mode != "normal" else None,
                                        clip_fea=clip_context if args.train_mode != "normal" else None,
                                        clean_x=clean_x_list,
                                        aug_t=aug_t,
                                    )
                                    # An Euler rollout carries the prediction in
                                    # flow/velocity space between steps; only the
                                    # final step needs x0, which is what the DMD
                                    # objective is defined on.
                                    if not args.flow_euler_rollout or is_final_step:
                                        generator_pred = convert_flow_pred_to_x0(
                                            scheduler=noise_scheduler,
                                            flow_pred=generator_pred,
                                            xt=generator_noise,
                                            timestep=timestep
                                        )
                                
                                if is_final_step:
                                    break
                                # Keep this step's x0 prediction as the clean context
                                # the next, gradient-carrying step will read.
                                if use_self_context:
                                    if args.flow_euler_rollout:
                                        # `generator_pred` is still in flow space here,
                                        # but 3.3's clean context is by definition x0, so
                                        # convert a detached copy for that purpose only.
                                        self_clean_x = convert_flow_pred_to_x0(
                                            scheduler=noise_scheduler,
                                            flow_pred=generator_pred.detach(),
                                            xt=generator_noise,
                                            timestep=timestep
                                        )
                                    else:
                                        self_clean_x = generator_pred.detach()

                                next_timestep = denoising_step_list[index + 1] * torch.ones(
                                    generator_noise.shape[:1], dtype=torch.long, device=generator_noise.device
                                )
                                if args.flow_euler_rollout:
                                    # Deterministic Euler ODE step in fp32, matching
                                    # LightX2V's WanStepDistillScheduler.step_post:
                                    # x_next = x_t - sigma_t * v + sigma_next * v.
                                    sigma_t = get_sigmas(timestep, n_dim=generator_noise.ndim, dtype=torch.float32)
                                    sigma_next = get_sigmas(next_timestep, n_dim=generator_noise.ndim, dtype=torch.float32)
                                    generator_noise = (
                                        generator_noise.float() + (sigma_next - sigma_t) * generator_pred.float()
                                    ).to(generator_noise.dtype)
                                else:
                                    generator_noise = add_noise(
                                        generator_pred,
                                        torch.randn(generator_pred.shape, dtype=generator_pred.dtype, device=generator_pred.device, generator=torch_rng),
                                        next_timestep
                                    )

                    # Common code for both KV cache and block mask training
                    indices = idx_sampling(bsz, generator=torch_rng, device=accelerator.device).long().cpu()
                    generator_timestep = noise_scheduler.timesteps[indices].to(device=accelerator.device)
                    generator_denoised_input = add_noise(
                        generator_pred,
                        torch.randn(generator_pred.shape, dtype=generator_pred.dtype, device=generator_pred.device, generator=torch_rng),
                        generator_timestep
                    ).detach().to(accelerator.device, dtype=weight_dtype)

                    # Compute fake score
                    with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device), torch.no_grad():
                        fake_score_main_cond = fake_score_transformer3d(
                            x=generator_denoised_input,
                            context=prompt_embeds,
                            t=generator_timestep,
                            seq_len=seq_len,
                            y=inpaint_latents if args.train_mode != "normal" else None,
                            clip_fea=clip_context if args.train_mode != "normal" else None,
                        )
                        fake_score_main_cond = convert_flow_pred_to_x0(
                            scheduler=noise_scheduler,
                            flow_pred=fake_score_main_cond,
                            xt=generator_denoised_input,
                            timestep=generator_timestep
                        )

                        if args.fake_guidance_scale != 0.0:
                            fake_score_main_uncond = fake_score_transformer3d(
                                x=generator_denoised_input,
                                context=neg_prompt_embeds,
                                t=generator_timestep,
                                seq_len=seq_len,
                                y=inpaint_latents if args.train_mode != "normal" else None,
                                clip_fea=clip_context if args.train_mode != "normal" else None,
                            )
                            fake_score_main_uncond = convert_flow_pred_to_x0(
                                scheduler=noise_scheduler,
                                flow_pred=fake_score_main_uncond,
                                xt=generator_denoised_input,
                                timestep=generator_timestep
                            )
                            fake_score_main = fake_score_main_uncond + (
                                fake_score_main_cond - fake_score_main_uncond
                            ) * args.fake_guidance_scale
                        else:
                            fake_score_main = fake_score_main_cond

                        # Compute real score
                        real_score_main_cond = real_score_transformer3d(
                            x=generator_denoised_input,
                            context=prompt_embeds,
                            t=generator_timestep,
                            seq_len=seq_len,
                            y=inpaint_latents if args.train_mode != "normal" else None,
                            clip_fea=clip_context if args.train_mode != "normal" else None,
                        )
                        real_score_main_cond = convert_flow_pred_to_x0(
                            scheduler=noise_scheduler,
                            flow_pred=real_score_main_cond,
                            xt=generator_denoised_input,
                            timestep=generator_timestep
                        )

                        real_score_main_uncond = real_score_transformer3d(
                            x=generator_denoised_input,
                            context=neg_prompt_embeds,
                            t=generator_timestep,
                            seq_len=seq_len,
                            y=inpaint_latents if args.train_mode != "normal" else None,
                            clip_fea=clip_context if args.train_mode != "normal" else None,
                        )
                        real_score_main_uncond = convert_flow_pred_to_x0(
                            scheduler=noise_scheduler,
                            flow_pred=real_score_main_uncond,
                            xt=generator_denoised_input,
                            timestep=generator_timestep
                        )

                        real_score_main = real_score_main_uncond + (
                            real_score_main_cond - real_score_main_uncond
                        ) * args.real_guidance_scale

                    # DMD loss
                    fake_to_real_grad = fake_score_main - real_score_main
                    generator_to_real_norm = generator_pred - real_score_main
                    normalizer = torch.abs(generator_to_real_norm).mean(dim=[1, 2, 3, 4], keepdim=True)
                    fake_to_real_grad = fake_to_real_grad / normalizer
                    fake_to_real_grad = torch.nan_to_num(fake_to_real_grad)

                    # Apply gradient mask: only compute loss on unmasked elements
                    # (matches Self-Forcing dmd.py: F.mse_loss(x[mask], target[mask]))
                    if score_mask is not None:
                        dmd_loss = 0.5 * F.mse_loss(
                            generator_pred.double()[score_mask],
                            (generator_pred.double() - fake_to_real_grad.double()).detach()[score_mask],
                            reduction="mean"
                        )
                    else:
                        dmd_loss = 0.5 * F.mse_loss(
                            generator_pred.double(),
                            (generator_pred.double() - fake_to_real_grad.double()).detach(),
                            reduction="mean"
                        )
                        
                    avg_dmd_loss = accelerator.gather(dmd_loss.repeat(args.train_batch_size)).mean()
                    train_dmd_loss += avg_dmd_loss.item()
                    train_gen_log_count += 1

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
                
                if args.use_kv_cache_training:
                    # KV cache mode: block-by-block generation
                    fake_score_critic_noise = torch.randn(target_shape, device=accelerator.device, generator=torch_rng, dtype=weight_dtype)
                    
                    # Calculate frame_seq_length
                    frame_seq_length = (target_shape[3] * target_shape[4]) // (patch_h * patch_w)
                    
                    # Determine block structure (variable-length, mirrors generator branch)
                    if not args.independent_first_frame:
                        max_num_blocks_critic = num_frames // args.num_frame_per_block
                        min_num_blocks_critic = args.score_num_frames // args.num_frame_per_block
                    else:
                        max_num_blocks_critic = (num_frames - 1) // args.num_frame_per_block
                        if args.score_num_frames > 1:
                            min_num_blocks_critic = (args.score_num_frames - 1) // args.num_frame_per_block
                        else:
                            min_num_blocks_critic = 0

                    # Random sample number of blocks (mirrors generator's variable-length training)
                    if args.min_length_prob_bias > 0.0 and max_num_blocks_critic > min_num_blocks_critic:
                        num_options = max_num_blocks_critic - min_num_blocks_critic + 1
                        bias = min(args.min_length_prob_bias, 0.99)
                        remaining_prob = (1.0 - bias) / (num_options - 1)
                        probs = [remaining_prob] * num_options
                        probs[0] = bias  # min_num_blocks_critic gets the bias
                        probs_tensor = torch.tensor(probs, device=accelerator.device)
                        block_indices = torch.multinomial(probs_tensor, 1, generator=torch_rng)
                        num_generated_blocks_critic = (min_num_blocks_critic + block_indices).item()
                    else:
                        num_generated_blocks_critic = torch.randint(
                            min_num_blocks_critic, max_num_blocks_critic + 1, (1,),
                            generator=torch_rng, device=accelerator.device
                        ).item()
                    if dist.is_initialized():
                        _sync = torch.tensor([num_generated_blocks_critic], device=accelerator.device)
                        dist.broadcast(_sync, src=0)
                        num_generated_blocks_critic = _sync.item()

                    all_num_frames = [args.num_frame_per_block] * num_generated_blocks_critic
                    if args.independent_first_frame:
                        all_num_frames = [1] + all_num_frames
                    num_generated_frames_critic = sum(all_num_frames)
                    # Same re-cut as the generator's KV-cache rollout above; the
                    # critic samples its own clip length, so it cannot reuse it.
                    critic_partitions = sample_flex_partitions(
                        args, num_generated_frames_critic,
                        len(denoising_step_list), torch_rng,
                        accelerator.device, accelerator.is_main_process)
                    if critic_partitions is not None:
                        all_num_frames = critic_partitions[0]
                        assert sum(all_num_frames) == num_generated_frames_critic
                    
                    # Initialize KV cache
                    num_layers = generator_transformer3d.config.num_layers
                    num_heads = generator_transformer3d.config.num_heads
                    head_dim = generator_transformer3d.config.dim // num_heads
                    text_len = 512
                    
                    critic_kv_cache = initialize_kv_cache_for_training(
                        batch_size=bsz,
                        num_frames=num_frames,
                        frame_seq_length=frame_seq_length,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dtype=weight_dtype,
                        device=accelerator.device
                    )
                    
                    critic_crossattn_cache = initialize_crossattn_cache_for_training(
                        batch_size=bsz,
                        text_len=text_len,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dtype=weight_dtype,
                        device=accelerator.device
                    )
                    
                    num_input_frames = 0
                    output_pred = torch.zeros_like(fake_score_critic_noise)
                    
                    # Decide whether to use teacher forcing for this video
                    use_teacher_forcing_step = (
                        args.use_teacher_forcing and 
                        torch.rand(1, generator=torch_rng, device=accelerator.device).item() < args.teacher_forcing_prob
                    )
                    
                    # Same exit step across all spans (matches original Self-Forcing default)
                    num_denoising_steps = len(denoising_step_list)
                    final_step_index = generate_and_sync_list(num_denoising_steps, device=accelerator.device)[0]
                    
                    # The same 3.2 walk as the generator's rollout above, spelled
                    # out again here because the critic has to re-roll it exactly:
                    # a span that still splits takes one buffered step, re-noises
                    # it back into the shared noise buffer and pushes its sub-spans
                    # back reversed; a leaf runs the rest of the schedule up to the
                    # exit step. All of it is under no_grad, so the generator's
                    # gradient gate has no counterpart here.
                    ladder = (critic_partitions if critic_partitions is not None
                              else [all_num_frames])
                    # Same two refreshes as the generator's rollout: the critic
                    # re-rolls with the generator's weights, so it reads the same
                    # model-level chunk size, and it must not overwrite it with a
                    # value the generator's next rollout could not agree with.
                    validate_nested_partitions(ladder, num_generated_frames_critic)
                    accelerator.unwrap_model(
                        generator_transformer3d
                    ).num_frame_per_block = max(ladder[0])
                    top = chunk_boundaries(ladder[0])
                    stack = [(span, 0, idx < len(top) - 1)
                             for idx, span in reversed(list(enumerate(top)))]
                    while stack:
                        (span_start, span_end), step_idx, commit = stack.pop()
                        level = ladder[min(step_idx + 1, len(ladder) - 1)]
                        subs = [s for s in chunk_boundaries(level)
                                if s[0] >= span_start and s[1] <= span_end]
                        splits = step_idx < final_step_index and len(subs) > 1
                        schedule = (denoising_step_list[step_idx:step_idx + 1] if splits
                                    else denoising_step_list[step_idx:final_step_index + 1])

                        current_num_frames = span_end - span_start
                        current_start_frame = span_start
                        start_idx = current_start_frame - num_input_frames
                        end_idx = start_idx + current_num_frames
                        noisy_input = fake_score_critic_noise[:, :, start_idx:end_idx]
                        
                        for local_idx, current_timestep in enumerate(schedule):
                            global_idx = step_idx + local_idx
                            is_final_step = (global_idx == final_step_index)
                            timestep = torch.full(
                                [bsz, current_num_frames],
                                current_timestep,
                                device=noisy_input.device,
                                dtype=torch.int64
                            )
                            
                            context_manager = torch.no_grad()
                            
                            with context_manager:
                                noisy_input_list = [noisy_input[i] for i in range(bsz)]
                                
                                # Use full seq_len (consistent with inference code)
                                full_seq_len = frame_seq_length * num_frames
                                
                                fake_score_denoised_pred_block = generator_transformer3d(
                                    x=noisy_input_list,
                                    context=prompt_embeds,
                                    t=timestep,
                                    seq_len=full_seq_len,
                                    kv_cache=critic_kv_cache,
                                    crossattn_cache=critic_crossattn_cache,
                                    current_start=current_start_frame * frame_seq_length,
                                    cache_start=None,
                                    y=inpaint_latents if args.train_mode != "normal" else None,
                                    clip_fea=clip_context if args.train_mode != "normal" else None,
                                )
                                
                                # Stack list output to tensor: [B, C, F, H, W]
                                if isinstance(fake_score_denoised_pred_block, list):
                                    fake_score_denoised_pred_block = torch.stack(fake_score_denoised_pred_block, dim=0)
                                
                                if not args.flow_euler_rollout or is_final_step:
                                    fake_score_denoised_pred_block = convert_flow_pred_to_x0(
                                        scheduler=noise_scheduler,
                                        flow_pred=fake_score_denoised_pred_block,
                                        xt=noisy_input,
                                        timestep=timestep[:, 0]
                                    )
                            
                            if is_final_step:
                                break
                            
                            next_timestep = denoising_step_list[global_idx + 1] * torch.ones(
                                bsz, dtype=torch.long, device=noisy_input.device
                            )
                            if args.flow_euler_rollout:
                                # The critic must re-roll the generator exactly, so it
                                # follows the same Euler step as the generator above.
                                sigma_t = get_sigmas(timestep[:, 0], n_dim=noisy_input.ndim, dtype=torch.float32)
                                sigma_next = get_sigmas(next_timestep, n_dim=noisy_input.ndim, dtype=torch.float32)
                                noisy_input = (
                                    noisy_input.float() + (sigma_next - sigma_t) * fake_score_denoised_pred_block.float()
                                ).to(noisy_input.dtype)
                            else:
                                noisy_input = add_noise(
                                    fake_score_denoised_pred_block,
                                    torch.randn(fake_score_denoised_pred_block.shape, dtype=fake_score_denoised_pred_block.dtype, device=fake_score_denoised_pred_block.device, generator=torch_rng),
                                    next_timestep
                                )
                        
                        if splits:
                            # The next level consumes the *re-noised* buffer, not
                            # x0: the sub-spans pick the schedule up where this
                            # step left it.
                            fake_score_critic_noise[:, :, start_idx:end_idx] = noisy_input
                            stack.extend(reversed([
                                (sub, step_idx + 1, k < len(subs) - 1 or commit)
                                for k, sub in enumerate(subs)]))
                            continue
                        
                        output_pred[:, :, current_start_frame:current_start_frame + current_num_frames] = fake_score_denoised_pred_block
                        
                        # Update KV cache with clean context (consistent with inference: feed denoised_pred directly).
                        # Leaves only, and only when a later chunk will read it.
                        if commit:
                            context_timestep = torch.ones([bsz, current_num_frames], device=accelerator.device, dtype=torch.int64) * args.context_noise
                            
                            # Use clean latents for teacher forcing, otherwise use denoised prediction directly
                            if use_teacher_forcing_step and clean_latents is not None:
                                context_input = clean_latents[:, :, start_idx:end_idx]
                            else:
                                context_input = fake_score_denoised_pred_block
                            
                            context_input_list = [context_input[i] for i in range(bsz)]
                            
                            # Use full seq_len (consistent with inference code)
                            full_seq_len = frame_seq_length * num_frames
                            
                            with torch.no_grad():
                                generator_transformer3d(
                                    x=context_input_list,
                                    context=prompt_embeds,
                                    t=context_timestep,
                                    seq_len=full_seq_len,
                                    kv_cache=critic_kv_cache,
                                    crossattn_cache=critic_crossattn_cache,
                                    current_start=current_start_frame * frame_seq_length,
                                    cache_start=None,
                                    y=inpaint_latents if args.train_mode != "normal" else None,
                                    clip_fea=clip_context if args.train_mode != "normal" else None,
                                )
                    
                    fake_score_denoised_pred_full = output_pred[:, :, :num_generated_frames_critic]

                    # Slice for critic score: last score_num_frames frames
                    if num_generated_frames_critic > args.score_num_frames:
                        fake_score_denoised_pred, critic_score_num_frames, _ = slice_for_score(
                            fake_score_denoised_pred_full, vae, weight_dtype,
                            score_num_frames=args.score_num_frames,
                            independent_first_frame=args.independent_first_frame,
                        )
                    else:
                        fake_score_denoised_pred = fake_score_denoised_pred_full
                        critic_score_num_frames = num_generated_frames_critic

                    seq_len = frame_seq_length * critic_score_num_frames
                    
                else:
                    with torch.no_grad():
                        # Block mask mode: use flex attention to process entire video at once
                        
                        patch_h_bm, patch_w_bm = accelerator.unwrap_model(generator_transformer3d).config.patch_size[1:]
                        frame_seqlen_bm = (height * width) // (patch_h_bm * patch_w_bm)
                        seq_len = frame_seqlen_bm * num_frames
                        
                        fake_score_critic_noise = torch.randn(target_shape, device=accelerator.device, generator=torch_rng, dtype=weight_dtype)
                        num_denoising_steps = len(denoising_step_list)
                        final_step_index = generate_and_sync_list(num_denoising_steps, device=fake_score_critic_noise.device)[0]

                        # Decide whether to use teacher forcing for this step
                        use_teacher_forcing_step = (
                            args.use_teacher_forcing and 
                            torch.rand(1, generator=torch_rng, device=accelerator.device).item() < args.teacher_forcing_prob
                        )
                        
                        # The critic re-rolls the generator under `no_grad`, so it
                        # has to run the *same* partition the generator used this
                        # iteration; only critic-only updates draw one themselves.
                        if flex_partitions is None:
                            flex_partitions = sample_flex_partitions(
                                args, num_frames, len(denoising_step_list), torch_rng,
                                accelerator.device, accelerator.is_main_process)
                        flex_model = accelerator.unwrap_model(generator_transformer3d)
                        install_flex_partition(flex_model, flex_partitions, 0)

                        if use_teacher_forcing_step and clean_latents is not None:
                            # Teacher forcing: clean + noisy sequence mask
                            clean_x = [clean_latents[i] for i in range(clean_latents.size(0))]
                            aug_t = torch.zeros(bsz, device=accelerator.device, dtype=torch.int64)
                        else:
                            # Standard causal mask
                            clean_x = None
                            aug_t = None
                        # 3.3: mirror the generator's decision exactly. The critic
                        # re-rolls that same generator, so diverging here would put
                        # the two sides of the DMD loss on different attention.
                        use_self_context = flex_self_context_enabled(
                            args, clean_x, final_step_index)
                        if use_self_context:
                            aug_t = torch.zeros(bsz, device=accelerator.device, dtype=torch.int64)
                        self_clean_x = None
                        build_block_mask = make_flex_mask_builder(
                            flex_model, args, num_frames, frame_seqlen_bm,
                            accelerator.device,
                            clean_x is not None or use_self_context)
                        build_block_mask()

                        for index, current_timestep in enumerate(denoising_step_list):
                            is_final_step = (index == final_step_index)
                            # 3.2, mirroring the generator's step loop above.
                            if index > 0 and flex_partitions is not None and len(flex_partitions) > 1:
                                install_flex_partition(flex_model, flex_partitions, index)
                                build_block_mask()
                            timestep = torch.full(
                                fake_score_critic_noise.shape[:1], 
                                current_timestep,
                                device=fake_score_critic_noise.device,
                                dtype=torch.int64
                            )
                            
                            
                            with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                                # Convert to list format for transformer
                                fake_score_critic_noise_list = [fake_score_critic_noise[i] for i in range(bsz)]
                                if clean_x is not None:
                                    clean_x_list = [clean_latents[i] for i in range(bsz)]
                                elif use_self_context and self_clean_x is not None:
                                    # 3.3, mirroring the generator's step loop above.
                                    clean_x_list = [self_clean_x[i] for i in range(bsz)]
                                else:
                                    clean_x_list = None
                                
                                fake_score_denoised_pred = generator_transformer3d(
                                    x=fake_score_critic_noise_list,
                                    context=prompt_embeds,
                                    t=timestep,
                                    seq_len=seq_len,
                                    y=inpaint_latents if args.train_mode != "normal" else None,
                                    clip_fea=clip_context if args.train_mode != "normal" else None,
                                    clean_x=clean_x_list,
                                    aug_t=aug_t,
                                )
                                if not args.flow_euler_rollout or is_final_step:
                                    fake_score_denoised_pred = convert_flow_pred_to_x0(
                                        scheduler=noise_scheduler,
                                        flow_pred=fake_score_denoised_pred,
                                        xt=fake_score_critic_noise,
                                        timestep=timestep
                                    )
                                
                                if is_final_step:
                                    break
                                # 3.3: same substitution the generator made, so the
                                # rollout the critic scores is the one being trained.
                                if use_self_context:
                                    if args.flow_euler_rollout:
                                        # Flow space here too, so convert a detached x0
                                        # exactly as the generator's loop did.
                                        self_clean_x = convert_flow_pred_to_x0(
                                            scheduler=noise_scheduler,
                                            flow_pred=fake_score_denoised_pred.detach(),
                                            xt=fake_score_critic_noise,
                                            timestep=timestep
                                        )
                                    else:
                                        self_clean_x = fake_score_denoised_pred.detach()

                                next_timestep = denoising_step_list[index + 1] * torch.ones(
                                    fake_score_critic_noise.shape[:1], 
                                    dtype=torch.long,
                                    device=fake_score_critic_noise.device
                                )
                                
                                if args.flow_euler_rollout:
                                    sigma_t = get_sigmas(timestep, n_dim=fake_score_critic_noise.ndim, dtype=torch.float32)
                                    sigma_next = get_sigmas(next_timestep, n_dim=fake_score_critic_noise.ndim, dtype=torch.float32)
                                    fake_score_critic_noise = (
                                        fake_score_critic_noise.float()
                                        + (sigma_next - sigma_t) * fake_score_denoised_pred.float()
                                    ).to(fake_score_critic_noise.dtype)
                                else:
                                    fake_score_critic_noise = add_noise(
                                        fake_score_denoised_pred,
                                        torch.randn(fake_score_denoised_pred.shape, dtype=fake_score_denoised_pred.dtype, device=fake_score_denoised_pred.device, generator=torch_rng),
                                        next_timestep
                                    )

                indices = idx_sampling(bsz, generator=torch_rng, device=accelerator.device).long().cpu()
                critic_timestep = noise_scheduler.timesteps[indices].to(device=accelerator.device)
                critic_noise = torch.randn(fake_score_denoised_pred.shape, dtype=fake_score_denoised_pred.dtype, device=fake_score_denoised_pred.device, generator=torch_rng)

                fake_score_denoised_input = add_noise(
                    fake_score_denoised_pred,
                    critic_noise,
                    critic_timestep
                )

                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    fake_score_denoised_output = fake_score_transformer3d(
                        x=fake_score_denoised_input,
                        context=prompt_embeds,
                        t=critic_timestep,
                        seq_len=seq_len,
                        y=inpaint_latents if args.train_mode != "normal" else None,
                        clip_fea=clip_context if args.train_mode != "normal" else None,
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

                denoising_loss = custom_mse_loss(fake_score_denoised_output, critic_noise - fake_score_denoised_pred)
                avg_denoising_loss = accelerator.gather(denoising_loss.repeat(args.train_batch_size)).mean()
                train_denoising_loss += avg_denoising_loss.item() / args.gradient_accumulation_steps
                
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
                accelerator.log({"train_denoising_loss": train_denoising_loss, "train_dmd_loss": train_dmd_loss / max(train_gen_log_count, 1)}, step=global_step)
                train_dmd_loss = 0.0
                train_gen_log_count = 0
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
                        # Keep the checkpoint out of the progress bar rate: a minute-long save would
                        # otherwise land in the next step's interval and be shown as a slow step. The
                        # save also stages the whole state in host RAM and leaves the freed blocks in
                        # the allocator caches, so the cache flushes run inside the same window.
                        with progress_bar.paused():
                            accelerator.save_state(save_path)
                            accelerator_fake_score_transformer3d.save_state(fake_score_save_path)
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        logger.info(f"Saved state to {save_path}")

                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    with progress_bar.paused():
                        log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            clip_image_encoder,
                            generator_transformer3d,
                            args,
                            config,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"denoising_loss": denoising_loss.detach().item(), "dmd_loss": dmd_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            with progress_bar.paused():
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    clip_image_encoder,
                    generator_transformer3d,
                    args,
                    config,
                    accelerator,
                    weight_dtype,
                    global_step,
                )

    # Close the bar before the end-of-run checkpoint: tqdm keeps redrawing a live bar whenever
    # something else writes to the console. PauseAwareTqdm.close() rebases the closing line onto
    # the smoothed rate, so the worker warm-up and the first dataloader fetch do not dilute it.
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
