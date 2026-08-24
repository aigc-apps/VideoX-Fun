# Modified from scripts/minimax_h3/train.py for VACE-style control training, porting the control branch of
# scripts/z_image_fun/train_control.py to the MiniMax-H3 packed-sequence transformer.
#
# Control training of the packed-sequence transformer on the *video and audio* rows together: the target video is
# guided by a paired control video (pose / depth / canny ...) carried by `VideoSpeechControlDataset`. The control
# signal enters through `MiniMaxH3ControlTransformer3DModel`'s zero-initialised side branch: the clean control
# latents are patchified exactly like the target video, embedded by `control_proj_in` and injected as per-layer
# skips through the zero-initialised `before_proj` / `after_proj`, so a freshly initialised model is numerically
# identical to the base MiniMax-H3 model and only the control parameters (`--trainable_modules control`) need
# training.
#
# Mirroring `scripts/z_image_fun/train_control.py`:
#   * 10% of the batches zero the control latents, keeping the unconditional path trainable (CFG),
#   * the audio stream keeps the joint video + audio flow-matching loss of `scripts/minimax_h3/train.py`.
#
# MiniMax-H3's rectified-flow convention is the *opposite* of Wan's and is reproduced here from
# `MiniMaxH3Scheduler.scale_noise` / `MiniMaxH3Scheduler.step`, the single source of truth:
#   * noising: `x_t = t * x0 + (1 - t) * noise` with `t = 1` clean, `t = 1 - sigma`,
#   * the sigma grid is exponentially shifted, `sigma' = s * sigma / (1 + (s - 1) * sigma)`, `s = 12.0` for video and `3.0` for audio,
#   * the transformer predicts a data-ward velocity, so the regression target is `x0 - noise`.
#
# Usage:
#   accelerate launch scripts/minimax_h3_fun/train_control.py \
#       --pretrained_model_name_or_path=/root/MiniMax-H3 \
#       --gradient_checkpointing --low_vram --trainable_modules "control"

import argparse
import contextlib
import gc
import inspect
import logging
import math
import os
import pickle
import random
import shutil
import sys
import warnings

import accelerate
import datasets
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
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers.utils import ContextManagers

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.data import (ASPECT_RATIO_512,
                             AspectRatioBatchImageVideoSampler,
                             ImageVideoSampler, VideoSpeechControlDataset,
                             get_closest_ratio, get_random_mask)
from videox_fun.models import (AutoencoderKLMiniMaxH3,
                               AutoencoderKLMiniMaxH3Audio,
                               MiniMaxH3ControlTransformer3DModel,
                               Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor)
from videox_fun.data import (ASPECT_RATIO_512, ASPECT_RATIO_RANDOM_CROP_512,
                             ASPECT_RATIO_RANDOM_CROP_PROB,
                             AspectRatioBatchImageVideoSampler,
                             ImageVideoDataset, ImageVideoSampler,
                             RandomSampler, get_closest_ratio)
from videox_fun.pipeline.pipeline_minimax_h3 import (
    MINIMAX_H3_FPS, MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_TEXT_ENCODER_LAYER, MINIMAX_H3_TEXT_TAG, _offload_scope,
    align_num_frames, audio_latent_num_frames, build_packed_sequence,
    build_row_timesteps, patchify_video_latents, unpatchify_video_tokens,
    video_latent_num_frames)
from videox_fun.pipeline import MiniMaxH3ControlPipeline
from videox_fun.utils import MiniMaxH3Scheduler
from videox_fun.utils.utils import (get_video_to_video_latent,
                                    save_videos_grid,
                                    save_videos_with_audio_grid)

# Silences diffusers' `randn_tensor` notice about CPU generators producing CUDA tensors (the tensor is created
# on CPU and moved to GPU; harmless, only a marginal speed note).
warnings.filterwarnings("ignore", message="The passed generator was created on")

def _mm_token_type_ids(tokenizer, token_ids):
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_pad_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    return [1 if t == image_pad_id else 2 if t == video_pad_id else 0 for t in token_ids]


def resample_waveform_to_span(waveform, target_length):
    r"""Rescale a waveform onto the 24 fps timeline of a `target_length`-frame clip.

    The dataset filters clips whose frame rate falls outside the 24 fps tolerance at the source and slices the
    waveform over a span that already matches the layout, so this rescale is normally a near-identity pass; it
    absorbs the remaining rounding slack between the sliced waveform and the layout's `target_length` instead of
    letting it surface as an audio-latent mismatch.
    """
    mono = waveform.ndim == 1
    wave = waveform[None] if mono else waveform
    resampled = F.interpolate(
        wave[None].float(), size=target_length, mode="linear", align_corners=False,
    )[0]
    resampled = resampled.to(dtype=waveform.dtype)
    return resampled[0] if mono else resampled


def encode_prompt(text_encoder, tokenizer, processor, prompt, device, dtype):
    r"""Build MiniMax-H3's presentation of a text-only request and encode it.

    Control training conditions on the control video through the transformer's side branch, so the presentation is
    always the verbatim prompt — no keyframe vision blocks, which keeps the text stream of every sample plain text
    tagged `MINIMAX_H3_TEXT_TAG`.
    """
    num_layers = text_encoder.config.text_config.num_hidden_layers
    if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER:
        raise ValueError(
            f"MiniMax-H3 conditions on `hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}]` of its Qwen3-VL "
            f"conditioner, which needs more than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers, but "
            f"`text_encoder` has {num_layers}. The last hidden state of a stack truncated to exactly "
            f"{MINIMAX_H3_TEXT_ENCODER_LAYER} layers is post-norm and is not the conditioning MiniMax-H3 expects."
        )

    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not token_ids:
        # An empty prompt (e.g. the dataset's text drop for classifier-free guidance) tokenizes to zero tokens,
        # and Qwen3-VL's `get_rope_index` cannot reduce over a zero-length sequence dimension; a single
        # whitespace token stands in for the dropped text.
        token_ids = tokenizer(" ", add_special_tokens=False)["input_ids"]
    token_tags = [MINIMAX_H3_TEXT_TAG] * len(token_ids)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    encoder_kwargs = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        pixel_values=None,
        image_grid_thw=None,
        use_cache=False,
        output_hidden_states=True,
    )
    model_module = text_encoder.model
    inner_forward = getattr(getattr(model_module, "module", model_module), "forward", model_module.forward)
    if "mm_token_type_ids" in inspect.signature(inner_forward).parameters:
        encoder_kwargs["mm_token_type_ids"] = torch.tensor(
            [_mm_token_type_ids(tokenizer, token_ids)], dtype=torch.long, device=device
        )
    with _offload_scope(text_encoder):
        outputs = text_encoder.model(**encoder_kwargs)
        prompt_embeds = outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].to(device=device, dtype=dtype)
    return prompt_embeds, torch.tensor(token_tags, dtype=torch.long)


def shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    r"""The exponential sigma shift of `MiniMaxH3Scheduler`, `sigma' = s*sigma / (1 + (s-1)*sigma)`."""
    return shift * sigma / (1 + (shift - 1) * sigma)


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value


def snap_num_frames(actual_num_frames, max_num_frames):
    """
    Pick the generation length from the control video instead of padding a short one: the largest `17 * n + 5`
    the video VAE can decode that does not exceed the frames actually read (capped by `max_num_frames`), snapping
    down so no tail frame is ever repeated. A control video below 5 frames is raised to 5, the smallest count
    the video VAE can encode.
    """
    num_frames = min(actual_num_frames, max_num_frames)
    num_frames = (num_frames - 5) // 17 * 17 + 5
    return max(num_frames, 5)


logger = get_logger(__name__, log_level="INFO")


@contextlib.contextmanager
def restore_frozen_requires_grad(model, trainable_module_names, enabled):
    r"""Narrow `requires_grad` back to the real trainable set for the duration of an `accelerator` checkpoint call.

    FSDP training keeps `requires_grad` uniformly `True` so that every wrapped unit is resharded by a normal
    post-backward hook (see the `use_fsdp` branch of `main`), but FSDP's optimizer state helpers read
    `requires_grad` off every original parameter of a flat parameter (`_get_fqn_to_fsdp_param_info`) and demand each
    gradient-requiring one be in the optimizer state. A frozen parameter sharing its unit with a trainable one —
    `proj_in.weight` next to `control_proj_in.weight` in the root unit — therefore aborts the save with
    "proj_in.weight is not in the optimizer state", and a resume hits the same check. Dropping `requires_grad` on
    the frozen parameters narrows that walk to the optimizer's own parameters; it is restored afterwards so the next
    backward keeps its prompt reshard.
    """
    if not enabled:
        yield
        return

    frozen_params = [
        param for name, param in model.named_parameters()
        if not any(trainable_module_name in name for trainable_module_name in trainable_module_names)
    ]
    # A wrap without `use_orig_params` exposes flat parameters only, whose names match no trainable module: narrowing
    # would then freeze the whole model, so leave `requires_grad` alone and let the save report its own error.
    if len(frozen_params) == len(list(model.named_parameters())):
        logger.warning(
            f"No parameter of the model matches {trainable_module_names}, so `requires_grad` is left as it is for "
            "the checkpoint."
        )
        yield
        return

    for param in frozen_params:
        param.requires_grad = False
    try:
        yield
    finally:
        for param in frozen_params:
            param.requires_grad = True


def log_validation(
    vae, audio_vae, text_encoder, tokenizer, processor, transformer,
    scheduler, audio_scheduler, args, accelerator, weight_dtype, global_step,
):
    r"""Run the inference pipeline over the validation pairs and save one video with its soundtrack per prompt.

    The denoise loop, the control-row construction and both decodes are `MiniMaxH3ControlPipeline`'s own, the way
    `scripts/minimax_h3/train.py` validates through `MiniMaxH3Pipeline`, so a validation sample reproduces
    `examples/minimax_h3_fun/predict_v2v_control.py` exactly — including the audio stream and the inpaint
    zero-padding of an `--enable_inpaint` checkpoint. A validation pair without a control path runs with
    `control_video=None`, the base-model branch.
    """
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=weight_dtype):
            logger.info("Running validation... ")
            pipeline = MiniMaxH3ControlPipeline(
                vae=vae,
                audio_vae=audio_vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                processor=processor,
                # Under FSDP the transformer must keep its wrapper so `_pre_forward_unshard` materializes the
                # sharded FlatParameters during inference; unwrapping leaves weights as 1-D shard views.
                transformer=accelerator.unwrap_model(transformer) if type(transformer).__name__ == 'DistributedDataParallel' else transformer,
                scheduler=scheduler,
                audio_scheduler=audio_scheduler,
            )
            pipeline = pipeline.to(accelerator.device)

            if args.seed is None:
                generator = None
            else:
                rank_seed = args.seed + accelerator.process_index
                generator = torch.Generator(device=accelerator.device).manual_seed(rank_seed)
                logger.info(f"Rank {accelerator.process_index} using seed: {rank_seed}")

            validation_paths = args.validation_paths if args.validation_paths else [""] * len(args.validation_prompts)
            for i in range(len(args.validation_prompts)):
                control_video = None
                num_frames = args.video_sample_n_frames
                if validation_paths[i]:
                    # The exact preprocessing of `examples/minimax_h3_fun/predict_v2v_control.py` (fps resample,
                    # canvas resize + crop, `[0, 1]` `(1, 3, F, H, W)` layout), then generate at the control
                    # video's actual length instead of padding a short one.
                    control_video, _, _, _ = get_video_to_video_latent(
                        validation_paths[i],
                        video_length=args.video_sample_n_frames,
                        sample_size=(args.video_sample_size, args.video_sample_size),
                        fps=MINIMAX_H3_FPS,
                        ref_image=None,
                        keep_aspect_ratio=True,
                    )
                    num_frames = snap_num_frames(control_video.shape[2], args.video_sample_n_frames)

                output = pipeline(
                    prompt=args.validation_prompts[i],
                    control_video=control_video,
                    height=args.video_sample_size,
                    width=args.video_sample_size,
                    num_frames=num_frames,
                    num_inference_steps=args.validation_sampling_steps,
                    generator=generator,
                    output_type="pt",
                )

                os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                save_videos_with_audio_grid(
                    output.videos,
                    output.audio,
                    os.path.join(
                        args.output_dir,
                        f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.mp4",
                    ),
                    fps=24,
                    audio_sample_rate=output.sampling_rate,
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
        # The full traceback, not just `str(e)`: a validation that keeps failing silently leaves no sample and no
        # clue, which is indistinguishable from a validation that never proves the control branch works.
        logger.exception(f"Eval error on rank {accelerator.process_index}")
        vae.to(accelerator.device if not args.low_vram else "cpu")
        text_encoder.to(accelerator.device if not args.low_vram else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMax-H3 control training (video + audio, VACE-style side branch).")
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
        "--config_path",
        type=str,
        default=None,
        help="The config of the model in training, e.g. config/minimax_h3/minimax_h3_control.yaml.",
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
        help="A csv/json containing the training data.",
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
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--enable_inpaint",
        action="store_true",
        help=(
            "Feed a random inpaint mask through the control branch alongside the control video: the control rows "
            "carry the visibility map + masked-video latents on top of `in_channels` (WanFun's mask recipe of "
            "scripts/wan2.1_fun/train_lora.py), so the yaml at `--config_path` must pin the matching widened "
            "`control_in_dim`, and checkpoints of a mask-less control branch no longer load."
        ),
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
        "--output_dir",
        type=str,
        default="samples/minimax-h3-control",
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
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
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
        default=["control"],
        help='Enter a list of trainable modules'
    )
    parser.add_argument(
        '--trainable_modules_low_learning_rate',
        nargs='+',
        default=[],
        help='Enter a list of trainable modules with lower learning rate'
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
    # MiniMax-H3 specific
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=1,
        help="Frame sampling stride (MiniMax-H3 is 24 fps).",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=124,
        help="Number of frames (form 17*n+5).",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=1,
        help="Repeat video entries to balance ratio.",
    )
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="Keep VAE and conditioner on CPU, move to GPU only while encoding.",
    )
    parser.add_argument(
        "--offload_every_step",
        action="store_true",
        help="Move transformer through CPU between steps (cards far below 62 GB).",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="minimax_h3_control",
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
        "--validation_paths",
        type=str,
        default=None,
        nargs="+",
        help=("A set of control videos evaluated every `--validation_steps` / `--validation_epochs` and logged to `--report_to`."),
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
    parser.add_argument(
        "--validation_sampling_steps",
        type=int,
        default=50,
        help="Number of denoising steps of the validation sampling loop.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    if args.video_sample_size % 32:
        raise ValueError(
            f"`video_sample_size` {args.video_sample_size} must be a multiple of 32: the canvas is patched "
            "2x2 into the transformer and its RoPE grid keys off that."
        )
    aligned_frames = align_num_frames(int(args.video_sample_n_frames))
    if aligned_frames != int(args.video_sample_n_frames):
        raise ValueError(
            f"`video_sample_n_frames` has to be of the form 17 * n + 5 the video VAE encodes, got "
            f"{args.video_sample_n_frames} (nearest is {aligned_frames})."
        )

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
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
    else:
        rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    if args.seed is not None:
        print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")
    else:
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

    # ------------------------------------------------------------------ models
    # The control variant of the transformer. The released MiniMax-H3 layout has no control-branch entries, and
    # `from_pretrained` initialises them itself: every control block from the main block it is attached to and
    # `control_proj_in` from `proj_in`, with before_proj / after_proj zeroed. The side branch therefore starts as an
    # identity — the freshly loaded model is numerically identical to the base MiniMax-H3 model — while its blocks
    # still receive gradient.
    # `--enable_inpaint` feeds a random inpaint mask through the side branch alongside the control video: on top
    # of the control video's `in_channels` the control rows carry the visibility map and the masked-video latents
    # (WanFun's mask recipe of scripts/wan2.1_fun/train_lora.py). `control_proj_in` then no longer matches
    # `proj_in` and `materialize_missing_control_params` initialises it off the fixed seed; the widened projection
    # also stops a mask-less control checkpoint from loading.
    # `--config_path` pins that layout (config/minimax_h3/minimax_h3_control.yaml, mirroring flux2's
    # `transformer_additional_kwargs`): `control_blocks_places` selects the layers the control blocks attach to and
    # `control_in_dim` the channels the control rows carry, both overriding the registered config at
    # `from_pretrained` time. With `--enable_inpaint` the yaml's `control_in_dim` must cover the mask channels;
    # the model default (`in_channels`) does not, and the forward of `control_proj_in` rejects the mismatch.
    transformer_load_kwargs = {}
    if args.config_path is not None:
        config = OmegaConf.load(args.config_path)
        transformer_load_kwargs.update(
            OmegaConf.to_container(config["transformer_additional_kwargs"], resolve=True)
        )
    transformer = MiniMaxH3ControlTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", low_cpu_mem_usage=True, torch_dtype=weight_dtype,
        **transformer_load_kwargs,
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
    # `from_pretrained`. So the two VAEs and the Qwen3-VL conditioner will not enjoy the parameter
    # sharding across multiple gpus and only the transformer will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # The two VAEs stay float32 as released (the encode/decode recipe is float16 autocast over float32
        # weights), so they are loaded without `torch_dtype`; the mixed-precision loader mixin restores the
        # pinned fp32 modules anyway.
        vae = AutoencoderKLMiniMaxH3.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", low_cpu_mem_usage=True,
        )
        # The audio VAE encodes the paired waveform to packed audio rows of the packed sequence.
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="audio_vae", low_cpu_mem_usage=True,
        )
        tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(args.pretrained_model_name_or_path, "tokenizer"))
        processor = Qwen3VLProcessor.from_pretrained(os.path.join(args.pretrained_model_name_or_path, "processor"))
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, "text_encoder"), low_cpu_mem_usage=True, torch_dtype=weight_dtype,
        )
        text_encoder = text_encoder.eval()
    scheduler = MiniMaxH3Scheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    audio_scheduler = MiniMaxH3Scheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="audio_scheduler")

    # Freeze the VAEs and the conditioner; the transformer's trainable parameters are selected below through
    # `--trainable_modules` (the control branch by default).
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    audio_vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
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
            from safetensors.torch import load_file, safe_open
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

    # FSDP1 does not reshard a *fully frozen* wrapped unit through a post-backward hook: it registers a `mode="all"`
    # multi-grad hook over the gradient-requiring inputs of the unit's forward instead, so the unsharded flat
    # parameters of the main stack can stay alive far into the backward pass — tens of GB for a 50-block model.
    # Keeping `requires_grad` uniformly `True` gives every unit a normal post-backward hook, i.e. a reshard plus
    # reduce-scatter right after its own backward. The frozen parameters therefore carry a sharded gradient (model
    # size / world size), which is never read and never handed to the optimizer: the trainable set stays the one
    # selected above and is matched by name everywhere below.
    if args.use_fsdp:
        transformer.requires_grad_(True)

    # Create EMA for the transformer.
    if args.use_ema:
        if zero_stage == 3:
            raise NotImplementedError("FSDP does not support EMA.")

        ema_transformer = MiniMaxH3ControlTransformer3DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", **transformer_load_kwargs
        ).to(weight_dtype)

        ema_transformer = EMAModel(ema_transformer.parameters(), model_cls=MiniMaxH3ControlTransformer3DModel, model_config=ema_transformer.config)

    # ------------------------------------------------------------------ save / load hooks
    # `accelerate` 0.16.0+ supports custom saving hooks; the full transformer is serialized in the diffusers
    # layout (`transformer/`) so the predict scripts and the pipeline load it with zero changes.
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

                    # Serialized in the diffusers layout (`transformer/` plus its `config.json`) so the predict
                    # scripts load the checkpoint with `from_pretrained(..., subfolder="transformer")` instead of a
                    # hand-rolled `load_state_dict(..., strict=False)`, which silently accepts a mismatched file.
                    save_directory = os.path.join(output_dir, "transformer")
                    os.makedirs(save_directory, exist_ok=True)
                    safetensor_save_path = os.path.join(save_directory, f"diffusion_pytorch_model.safetensors")
                    # MiniMax-H3 ships a mixed-precision checkpoint, and the modules the model pins in float32
                    # (`_keep_in_fp32_modules`: the patch projections — `control_proj_in` included, it matches the
                    # `proj_in` substring — the timestep MLP and the two output heads) have to stay float32 here;
                    # casting them to `weight_dtype` would quantize exactly the weights whose precision the
                    # released model depends on.
                    fp32_patterns = MiniMaxH3ControlTransformer3DModel._keep_in_fp32_modules
                    accelerate_state_dict = {
                        k: v.to(dtype=torch.float32 if any(p in k for p in fp32_patterns) else weight_dtype)
                        for k, v in accelerate_state_dict.items()
                    }
                    save_file(accelerate_state_dict, safetensor_save_path, metadata={"format": "pt"})
                    accelerator.unwrap_model(models[-1]).save_config(save_directory)

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
                        ema_transformer.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                    models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                    if not args.use_deepspeed:
                        weights.pop()

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

            def load_model_hook(models, input_dir):
                if args.use_ema:
                    ema_path = os.path.join(input_dir, "transformer_ema")
                    _, ema_kwargs = MiniMaxH3ControlTransformer3DModel.load_config(ema_path, return_unused_kwargs=True)
                    load_model = MiniMaxH3ControlTransformer3DModel.from_pretrained(
                        input_dir, subfolder="transformer_ema",
                    )
                    load_model = EMAModel(load_model.parameters(), model_cls=MiniMaxH3ControlTransformer3DModel, model_config=load_model.config)
                    load_model.load_state_dict(ema_kwargs)

                    ema_transformer.load_state_dict(load_model.state_dict())
                    ema_transformer.to(accelerator.device)
                    del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = MiniMaxH3ControlTransformer3DModel.from_pretrained(
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
        transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    # Matched by name rather than by `requires_grad`: under FSDP every parameter carries `requires_grad=True` so that
    # the frozen units are resharded promptly (see above), so `requires_grad` no longer marks the trainable set.
    trainable_module_names = args.trainable_modules + args.trainable_modules_low_learning_rate
    trainable_params = [
        param for name, param in transformer.named_parameters()
        if any(trainable_module_name in name for trainable_module_name in trainable_module_names)
    ]
    trainable_params_optim = [
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.learning_rate / 2},
    ]
    in_already = []
    for name, param in transformer.named_parameters():
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
    num_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable: {len(trainable_params)} tensors, {num_trainable / 1e6:.2f} M parameters.")

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
        trainable_params_optim,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ------------------------------------------------------------------ data
    # MiniMax-H3 control training keeps the paired waveform of `scripts/minimax_h3/train.py`, so the dataset must
    # carry video + control + audio; `VideoSpeechControlDataset` reads the `control_file_path` / `audio_path`
    # fields of the training meta and resamples the waveform to the audio VAE's sample rate (32 kHz, 40 latents/s),
    # fixed by the VAE's 800-sample hop against that 40-latents/s grid.
    audio_sr = getattr(audio_vae.config, "sampling_rate", 32000)
    train_dataset = VideoSpeechControlDataset(
        args.train_data_meta, args.train_data_dir,
        video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride,
        video_sample_n_frames=args.video_sample_n_frames, enable_bucket=args.enable_bucket, enable_inpaint=False,
        audio_sr=audio_sr,
        audio_native_sr_resample=True,
        audio_stereo=True,
        audio_span_includes_last_frame=True,
        # The video VAE encodes 17n + 5 frames, so a clip yielding fewer than 5 sampled frames can never form a
        # valid batch; the dataset skips it and retries with another sample rather than passing it on.
        min_video_sample_n_frames=5,
        # MiniMax-H3 reads its frames on a fixed 24 fps timeline, so a clip sampled at another rate would play at the
        # wrong speed against its own soundtrack; the dataset skips those too. The tolerance keeps 23.976 / 23.81 fps
        # (which is most of a webvid-style corpus) while rejecting 25 and 29.97 / 30 fps.
        target_video_sample_fps=MINIMAX_H3_FPS,
    )

    # The packed-sequence layout (text tokens + condition + audio + video rows) is per-sample, so a batch larger
    # than one would need a sample loop; batch-level training (mirroring `scripts/minimax_h3/train.py`) therefore
    # pins the batch size to one for now.
    if args.train_batch_size != 1:
        raise ValueError("MiniMax-H3 packed-sequence training requires --train_batch_size=1.")

    # The MiniMax-H3 video VAE encodes 17n + 5 frames, so bucket frame counts bucket in steps of 17.
    sample_n_frames_bucket_interval = 17

    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            print(f"worker_init_fn with {_seed + worker_id}")
            np.random.seed(_seed + worker_id)
            random.seed(_seed + worker_id)
        return _worker_init_fn

    if args.enable_bucket:
        aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset, 
            batch_size=args.train_batch_size, train_folder=args.train_data_dir, drop_last=True,
            aspect_ratios=aspect_ratio_sample_size,
        )

        def collate_fn(examples):
            def get_length_to_frame_num(token_length):
                if args.video_sample_size > 256:
                    sample_sizes = list(range(256, args.video_sample_size + 1, 128))

                    if sample_sizes[-1] != args.video_sample_size:
                        sample_sizes.append(args.video_sample_size)
                else:
                    sample_sizes = [args.video_sample_size]

                length_to_frame_num = {
                    sample_size: (min(token_length / sample_size / sample_size, args.video_sample_n_frames) - 5) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 5 for sample_size in sample_sizes
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
                    return np.random.choice(number_list, p=number_list_prob)
                else:
                    return rng.choice(number_list, p=number_list_prob)

            # Get token length
            target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
            length_to_frame_num = get_length_to_frame_num(target_token_length)

            # Create new output
            new_examples                          = {}
            new_examples["pixel_values"]          = []
            new_examples["control_pixel_values"]  = []
            new_examples["text"]                  = []
            new_examples["audio"]                 = []
            new_examples["fps"]                   = []

            # Used in Inpaint mode (`--enable_inpaint`)
            if args.enable_inpaint:
                new_examples["mask_pixel_values"] = []
                new_examples["mask"]              = []

            # Get downsample ratio in image and videos
            pixel_value     = examples[0]["pixel_values"]
            f, h, w, c      = np.shape(pixel_value)

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

            closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
            closest_size = [int(x / 32) * 32 for x in closest_size]

            min_example_length = min(
                [example["pixel_values"].shape[0] for example in examples]
            )
            # The 17n + 5 rounding below cannot invent frames: a sample shorter than the 5 the video VAE needs makes
            # it land on a negative count, the floor to 5 further down then hides that, and the slice still yields
            # only the few frames the sample holds — which surfaces much later as `video_latent_num_frames` rejecting
            # a frame count that is not 17n + 5. Fail here, where the cause is still legible.
            if min_example_length < 5:
                raise ValueError(
                    f"The shortest sample in this batch holds {min_example_length} frames; MiniMax-H3's video VAE "
                    "encodes 17 * n + 5 frames and so needs at least 5. Drop the clips that yield fewer than 5 "
                    "sampled frames from the training meta, or lower `--video_sample_stride`."
                )
            batch_video_length = int(min(batch_video_length, min_example_length))

            # The MiniMax-H3 video VAE encodes 17n + 5 frames.
            batch_video_length = (batch_video_length - 5) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 5

            if batch_video_length < 5:
                batch_video_length = 5

            for example in examples:
                # To 0~1
                pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.

                # The control video goes through the exact same geometry as the target video, so the patchified
                # control rows align one-to-one with the video rows of the packed sequence.
                control_pixel_values = torch.from_numpy(example["control_pixel_values"]).permute(0, 3, 1, 2).contiguous()
                control_pixel_values = control_pixel_values / 255.

                # Get adapt hw for resize
                if closest_size[0] / h > closest_size[1] / w:
                    resize_size = closest_size[0], int(w * closest_size[0] / h)
                else:
                    resize_size = int(h * closest_size[1] / w), closest_size[1]

                transform = transforms.Compose([
                    transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                    transforms.CenterCrop(closest_size),
                ])

                pixel_values = transform(pixel_values)[:batch_video_length]
                new_examples["pixel_values"].append(pixel_values)
                new_examples["control_pixel_values"].append(transform(control_pixel_values)[:batch_video_length])
                new_examples["text"].append(example["text"])

                # The mask follows WanFun's recipe (scripts/wan2.1_fun/train_lora.py): a random inpaint mask
                # over the sampled clip, and the masked video (masked pixels zeroed) that the VAE encodes as the
                # inpaint latents.
                if args.enable_inpaint:
                    mask = get_random_mask(pixel_values.size()).float()
                    new_examples["mask_pixel_values"].append(pixel_values * (1 - mask))
                    new_examples["mask"].append(mask)

                # Slice the waveform like the frames: the dataset sliced it across the sample's full span, but the
                # batch may keep fewer frames (the bucket minimum), so cut the audio to the kept span first and
                # then rescale it onto the 24 fps timeline of `batch_video_length` (a no-op when the lengths
                # already match within rounding; it also absorbs clips whose metadata fps disagrees with the real
                # frame rate, which the dataset's span check missed).
                # The waveform is `(channels, num_samples)` on the stereo route, so the length is the last axis.
                audio_length = example["audio"].shape[-1]
                example_frames = example["pixel_values"].shape[0]
                batch_audio_length = int(audio_length / example_frames * batch_video_length)
                # The `num_frames / fps` span the audio latent grid keys off, as in the inference pipeline.
                target_audio_length = int(round(batch_video_length / MINIMAX_H3_FPS * audio_sr))
                new_examples["audio"].append(
                    resample_waveform_to_span(example["audio"][..., :batch_audio_length], target_audio_length)
                )
                new_examples["fps"].append(example.get("fps", 24))

            # Limit the number of frames to the same
            new_examples["pixel_values"] = torch.stack([example for example in new_examples["pixel_values"]])
            new_examples["control_pixel_values"] = torch.stack([example for example in new_examples["control_pixel_values"]])
            if args.enable_inpaint:
                new_examples["mask_pixel_values"] = torch.stack([example for example in new_examples["mask_pixel_values"]])
                new_examples["mask"] = torch.stack([example for example in new_examples["mask"]])

            # Pad audio to same length and stack
            max_audio_length = max(audio.shape[-1] for audio in new_examples["audio"])
            new_examples["audio"] = torch.stack([
                F.pad(audio, (0, max_audio_length - audio.shape[-1]))
                for audio in new_examples["audio"]
            ])
            new_examples["fps"] = new_examples["fps"]
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
        batch_sampler_generator = torch.Generator()
        if args.seed is not None:
            batch_sampler_generator.manual_seed(args.seed)
        batch_sampler = ImageVideoSampler(
            RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, args.train_batch_size
        )

        def collate_fn(examples):
            # `VideoSpeechControlDataset` returns `[-1, 1]` pixels; MiniMax-H3 wants `[0, 1]` and ImageNet-normalizes
            # the encoder input itself, so hand the loop `[0, 1]` and drop the rest. The control video gets the same
            # treatment. The audio waveform is sliced to the video span and right-padded to a common length so the
            # batch stacks.
            min_example_length = min(
                [example["pixel_values"].shape[0] for example in examples]
            )
            if min_example_length < 5:
                raise ValueError(f"The shortest sample holds {min_example_length} frames; MiniMax-H3 needs at least 5.")
            batch_video_length = int(min(args.video_sample_n_frames, min_example_length))

            # The MiniMax-H3 video VAE encodes 17n + 5 frames.
            batch_video_length = (batch_video_length - 5) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 5

            if batch_video_length < 5:
                batch_video_length = 5

            # Create new output
            new_examples                          = {}
            new_examples["pixel_values"]          = []
            new_examples["control_pixel_values"]  = []
            new_examples["text"]                  = []
            new_examples["audio"]                 = []
            new_examples["fps"]                   = []

            # Used in Inpaint mode (`--enable_inpaint`)
            if args.enable_inpaint:
                new_examples["mask_pixel_values"] = []
                new_examples["mask"]              = []

            for example in examples:
                # To 0~1
                pixel_values = example["pixel_values"][:batch_video_length] * 0.5 + 0.5
                new_examples["pixel_values"].append(pixel_values)
                control_pixel_values = example["control_pixel_values"][:batch_video_length]
                new_examples["control_pixel_values"].append(control_pixel_values * 0.5 + 0.5)
                new_examples["text"].append(example["text"])

                # The mask mirrors scripts/flux2_fun/train_control.py: a random inpaint mask over the sliced
                # clip, and the masked video (masked pixels zeroed) that the VAE encodes as the inpaint latents.
                if args.enable_inpaint:
                    mask = get_random_mask(pixel_values.size()).float()
                    new_examples["mask_pixel_values"].append(pixel_values * (1 - mask))
                    new_examples["mask"].append(mask)

                # Slice the waveform like the frames: the dataset sliced it across the sample's full span, but the
                # batch may keep fewer frames (the bucket minimum), so cut the audio to the kept span first and
                # then rescale it onto the 24 fps timeline of `batch_video_length` (a no-op when the lengths
                # already match within rounding; it also absorbs clips whose metadata fps disagrees with the real
                # frame rate, which the dataset's span check missed).
                # The waveform is `(channels, num_samples)` on the stereo route, so the length is the last axis.
                audio_length = example["audio"].shape[-1]
                example_frames = example["pixel_values"].shape[0]
                batch_audio_length = int(audio_length / example_frames * batch_video_length)
                # The `num_frames / fps` span the audio latent grid keys off, as in the inference pipeline.
                target_audio_length = int(round(batch_video_length / MINIMAX_H3_FPS * audio_sr))
                new_examples["audio"].append(
                    resample_waveform_to_span(example["audio"][..., :batch_audio_length], target_audio_length)
                )
                new_examples["fps"].append(example.get("fps", 24))

            # Limit the number of frames to the same
            new_examples["pixel_values"] = torch.stack([example for example in new_examples["pixel_values"]])
            new_examples["control_pixel_values"] = torch.stack([example for example in new_examples["control_pixel_values"]])
            if args.enable_inpaint:
                new_examples["mask_pixel_values"] = torch.stack([example for example in new_examples["mask_pixel_values"]])
                new_examples["mask"] = torch.stack([example for example in new_examples["mask"]])

            # Pad audio to same length and stack
            max_audio_length = max(audio.shape[-1] for audio in new_examples["audio"])
            new_examples["audio"] = torch.stack([
                F.pad(audio, (0, max_audio_length - audio.shape[-1]))
                for audio in new_examples["audio"]
            ])
            new_examples["fps"] = new_examples["fps"]
            return new_examples

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index),
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

    # Cast to `weight_dtype` *before* prepare so FSDP flattens a uniform-dtype parameter set (the conversion mixin
    # pins a few modules in float32 for inference precision).
    transformer.gradient_checkpointing_save_on_cpu = args.gradient_checkpointing_save_on_cpu
    transformer = transformer.to(weight_dtype)

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Shard the frozen text encoder *after* prepare (mirrors `scripts/minimax_h3/train.py`): the Qwen3-VL
    # conditioner is wrapped per decoder layer so the per-step unshard footprint stays small, and a post-prepare
    # shard keeps the text encoder out of the trainable FSDP unit.
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
    # The two VAEs stay float32 (mirrors the pipeline: float32 weights, float16 autocast only at the
    # encode/decode call site), so they are moved without a dtype cast.
    vae.to(device if not args.low_vram else "cpu")
    audio_vae.to(device if not args.low_vram else "cpu")
    transformer.to(device, dtype=weight_dtype)
    text_encoder.to(device if not args.low_vram else "cpu", dtype=weight_dtype)

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
    latents_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    patch_size = tuple(unwrapped_transformer.config.patch_size)
    latent_channels = unwrapped_transformer.config.in_channels
    audio_channels = unwrapped_transformer.config.audio_in_channels
    video_shift = float(scheduler.shift)
    audio_shift = float(audio_scheduler.shift)
    audio_latents_mean = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, -1, 1)
    audio_latents_std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
    train_generator = torch.Generator(device="cpu")
    if args.seed is not None:
        train_generator.manual_seed(args.seed)

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
            with restore_frozen_requires_grad(transformer, trainable_module_names, args.use_fsdp):
                accelerator.load_state(checkpoint_folder_path)
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        train_video_loss = 0.0
        train_audio_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            # Sanity check: save the first batch so a glance at output_dir/sanity_check confirms the data pipe
            # (target video *and* its paired control video, plus the paired waveform muxed into the target video).
            if epoch == first_epoch and step == 0:
                pixel_values, control_pixel_values, texts = batch["pixel_values"].cpu(), batch["control_pixel_values"].cpu(), batch["text"]
                audios = batch["audio"].cpu()
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, control_pixel_value, text) in enumerate(zip(pixel_values, control_pixel_values, texts)):
                    pixel_value = pixel_value[None].permute(0, 2, 1, 3, 4)
                    control_pixel_value = control_pixel_value[None].permute(0, 2, 1, 3, 4)
                    gif_name = "-".join(text.replace("/", "").split()[:10]) if not text == "" else f"{global_step}-{idx}"
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.mp4", rescale=False)
                    save_videos_grid(control_pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}-control.mp4", rescale=False)
                    # Audio check: the collate slices the waveform onto the `num_frames / 24` span and rescales it
                    # onto the batch timeline, so muxing it back into the target video exposes any span / alignment
                    # breakage (drift, silent audio, wrong sample rate) before it surfaces as a latent mismatch.
                    # The keep-dim wrap makes `audio[0]` inside the saver see the per-sample `(C, T)` waveform.
                    save_videos_with_audio_grid(
                        pixel_value, audios[idx : idx + 1],
                        f"{args.output_dir}/sanity_check/{gif_name[:10]}-audio.mp4",
                        fps=MINIMAX_H3_FPS,
                        audio_sample_rate=audio_sr,
                        rescale=False,
                    )
                    if args.enable_inpaint:
                        mask_pixel_value = batch["mask_pixel_values"][idx].cpu()[None].permute(0, 2, 1, 3, 4)
                        mask_value = batch["mask"][idx].cpu()[None].permute(0, 2, 1, 3, 4).repeat(1, 3, 1, 1, 1)
                        save_videos_grid(mask_pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}-mask_pixel.mp4", rescale=False)
                        save_videos_grid(mask_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}-mask.mp4", rescale=False)

            with accelerator.accumulate(transformer):
                # Batch-level training (bs=1): the packed-sequence layout (text tokens + condition + audio +
                # video rows) is per-sample, so a batch larger than one would need a sample loop. With bs=1 the
                # batch *is* the sample, and the encode / noise / forward / loss mirror
                # `scripts/minimax_h3/train.py` without a sample loop.
                pixel_values = batch["pixel_values"][0]
                control_pixel_values = batch["control_pixel_values"][0]
                text = batch["text"][0]
                audio = batch["audio"][0]
                if args.enable_inpaint:
                    mask_pixel_values = batch["mask_pixel_values"][0]
                    mask = batch["mask"][0]
                # MiniMax-H3 has no fps input: its temporal rotary grid (`_temporal_position_grid`) and its audio
                # latent grid (`audio_latent_num_frames`, 40 latents/s against 24 fps) are both hard-wired to 24 fps,
                # and the control video is read on the same timeline. `batch["fps"]` cannot police that: the dataset
                # floors it (`int(fps // stride)`), so the very common 23.976 fps arrives as 23 and is
                # indistinguishable from a genuine 23 fps source. The audio latent count further down is the real
                # gate — it measures the video / audio span mismatch directly, in the units the layout keys off.

                # The 33 B transformer alone fills ~66 GB, so under `low_vram` it yields the GPU while the VAE and
                # the conditioner encode, and moves back for the forward / backward.
                if args.low_vram:
                    transformer.to("cpu")
                    torch.cuda.empty_cache()

                # ---- encode (per-sample, bs=1 so batch-level) ----
                num_frames, _, height, width = pixel_values.shape
                num_latent_frames = video_latent_num_frames(num_frames)
                latent_height = height // vae.spatial_compression_ratio
                latent_width = width // vae.spatial_compression_ratio

                # Video latents: MiniMax-H3's encoder input is `[0, 1]` pixels ImageNet-normalized; the VAE stays
                # float32 and the encode runs under float16 autocast, mirroring the decode recipe. The dataset
                # hands over `(F, C, H, W)` rows, the VAE wants `(B, C, F, H, W)`.
                pixels = pixel_values.to(device).permute(1, 0, 2, 3)[None]
                pixels = (pixels - pixel_mean) / pixel_std
                control_pixels = control_pixel_values.to(device).permute(1, 0, 2, 3)[None]
                control_pixels = (control_pixels - pixel_mean) / pixel_std
                if args.enable_inpaint:
                    mask_pixels = mask_pixel_values.to(device).permute(1, 0, 2, 3)[None]
                    mask_pixels = (mask_pixels - pixel_mean) / pixel_std

                # Under `low_vram`, load both VAEs at once and keep them on GPU for the video, control and audio
                # encodes in one session.
                if args.low_vram:
                    vae.to(device)
                    audio_vae.to(device)

                # Encode in `vae_mini_batch` mini batches, mirroring `_batch_encode_vae` in train.py. The target
                # latents are sampled; the control latents take the posterior mode (deterministic conditioning,
                # mirroring the pipeline's keyframe recipe).
                def _batch_encode_vae(pixels, posterior_mode=False):
                    bs = args.vae_mini_batch
                    new_pixel_values = []
                    for i in range(0, pixels.shape[0], bs):
                        pixels_bs = pixels[i : i + bs]
                        posterior = vae.encode(pixels_bs.float()).latent_dist
                        latents_bs = posterior.mode() if posterior_mode else posterior.sample()
                        new_pixel_values.append((latents_bs.float() - latents_mean) / latents_std)
                    return torch.cat(new_pixel_values, dim=0)

                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    target_latents = _batch_encode_vae(pixels)
                    control_latents = _batch_encode_vae(control_pixels, posterior_mode=True)
                    if args.enable_inpaint:
                        # Inpaint latents: the masked video takes the posterior mode like the control video
                        # (deterministic conditioning).
                        mask_latents = _batch_encode_vae(mask_pixels, posterior_mode=True)
                if args.low_vram:
                    target_latents = target_latents.cpu()
                    control_latents = control_latents.cpu()
                    if args.enable_inpaint:
                        mask_latents = mask_latents.cpu()

                # Control rows: patchified exactly like the target video. 10% of the batches zero them, keeping the
                # unconditional path trainable for classifier-free guidance (mirrors z_image_fun/train_control.py).
                control_rows = patchify_video_latents(control_latents, patch_size)
                if rng is None:
                    control_keep = np.random.choice([0, 1], p=[0.10, 0.90])
                else:
                    control_keep = rng.choice([0, 1], p=[0.10, 0.90])
                if not control_keep:
                    control_rows = torch.zeros_like(control_rows)

                if args.enable_inpaint:
                    # The mask handling mirrors WanFun's inpaint recipe (scripts/wan2.1_fun/train_lora.py, the
                    # `mask = resize_mask(1 - mask, latents)` block): the visibility map 1 - mask trilinearly
                    # resized onto the latent grid, and the VAE-encoded masked video behind it. Wan's step packs
                    # 4 pixel frames into every latent frame before the resize and splits the first frame off, both
                    # keyed off its causal 4x VAE layout (latent frames = (F - 1) / 4 + 1, where the causal
                    # convolution makes the first pixel frame fill the first latent frame alone); MiniMax-H3's VAE
                    # runs a 17 -> 5 chunked time grid (frames 17n + 5, latents 5n + 2) whose first chunk encodes
                    # 5 frames into 2 latents, so neither packing nor a first-frame split yields its layout and the
                    # visibility map goes straight from pixel frames to the latent grid. The inpaint rows stay
                    # controlnet-style — appended to the control rows along the channel columns.
                    mask_5d = rearrange(mask, "f c h w -> c f h w")[None]
                    mask_condition = F.interpolate(
                        1 - mask_5d, size=mask_latents.size()[2:], mode="trilinear", align_corners=False,
                    )

                    # Encode inpaint latents.
                    mask_condition_rows = patchify_video_latents(mask_condition, patch_size)
                    mask_latent_rows = patchify_video_latents(mask_latents, patch_size)
                    inpaint_rows = torch.cat(
                        [
                            mask_condition_rows.to(control_rows.device),
                            mask_latent_rows.to(control_rows.device),
                        ],
                        dim=-1,
                    )
                    # A fully masked clip carries nothing of the original, so 90% of those batches drop the whole
                    # inpaint info (WanFun's `t2v_flag` zeroes the concatenated mask map and masked-video latents
                    # alike): the all-zero mask channels then read as pure generation, and the same all-zero layout
                    # is what validation pads in.
                    if bool((mask == 1).all()):
                        if rng is None:
                            mask_keep = np.random.choice([0, 1], p=[0.90, 0.10])
                        else:
                            mask_keep = rng.choice([0, 1], p=[0.90, 0.10])
                        if not mask_keep:
                            inpaint_rows = torch.zeros_like(inpaint_rows)
                    control_rows = torch.cat([control_rows, inpaint_rows], dim=-1)

                # The conditioner reads `hidden_states[50]` of Qwen3-VL; control training conditions on the
                # verbatim prompt alone (no keyframe vision blocks). An FSDP-sharded text encoder tolerates
                # symmetric `.to` moves, so it is brought on-device right before the encode and back to CPU
                # afterwards.
                if args.low_vram:
                    text_encoder.to(device)
                with torch.no_grad():
                    prompt_embeds, text_token_tags = encode_prompt(
                        text_encoder, tokenizer, processor,
                        text, device=device, dtype=weight_dtype
                    )
                if args.low_vram:
                    text_encoder.to("cpu")
                    torch.cuda.empty_cache()
                if args.low_vram:
                    prompt_embeds = prompt_embeds.cpu()

                # Audio latents: the waveform autoencoder is mono, stereo carried as two batch items; the dataset
                # hands over the stereo waveform on the inference-aligned route (a mono clip arrives upmixed),
                # which is encoded to `[2, 32, T]`, normalized and packed to `[2*T, 32]` rows.
                # The audio VAE is still on GPU from the joint onload under `low_vram`.
                num_audio_latents = audio_latent_num_frames(num_frames)
                audio_wave = audio.to(device).float()
                if audio_wave.ndim == 1:
                    audio_wave = audio_wave.unsqueeze(0).expand(2, -1)
                audio_wave = audio_wave.unsqueeze(1)
                with torch.no_grad():
                    audio_posterior = audio_vae.encode(audio_wave).latent_dist
                    audio_latents = audio_posterior.mode()
                audio_latents = (audio_latents.float() - audio_latents_mean) / audio_latents_std
                # On the inference-aligned route the waveform covers the `num_frames / fps` span the layout keys
                # off, so the encode usually lands exactly on `num_audio_latents`; pad or truncate below covers only
                # the encoder's rounding at the 800-sample hop and the collate's rescale of a shorter batch. Keep
                # the one-frame window as the guard it always was: a count outside it means the waveform does not
                # cover the same span as the frames, which at these lengths is what a source fps other than 24
                # looks like; padding that away with zeros would teach the model to end every clip on silence and
                # desynchronize the soundtrack from the picture.
                audio_latent_low = audio_latent_num_frames(num_frames - 1) - 1
                audio_latent_high = audio_latent_num_frames(num_frames) + 1
                if not audio_latent_low <= audio_latents.shape[2] <= audio_latent_high:
                    raise ValueError(
                        f"The waveform encodes to {audio_latents.shape[2]} audio latents, outside the "
                        f"[{audio_latent_low}, {audio_latent_high}] that {num_frames} frames span at "
                        f"{MINIMAX_H3_FPS} fps, so the audio does not cover the same span as the frames. This is "
                        f"most often a source fps other than {MINIMAX_H3_FPS} (the dataset floors fps, so 23.976 "
                        "shows up as 23): re-encode the clip to 24 fps or drop it from the training meta."
                    )
                if audio_latents.shape[2] < num_audio_latents:
                    audio_latents = F.pad(
                        audio_latents, (0, num_audio_latents - audio_latents.shape[2]), value=0,
                    )
                elif audio_latents.shape[2] > num_audio_latents:
                    audio_latents = audio_latents[:, :, :num_audio_latents]
                if args.low_vram:
                    vae.to("cpu")
                    audio_vae.to("cpu")
                    torch.cuda.empty_cache()
                audio_rows = audio_latents.permute(0, 2, 1).reshape(-1, audio_channels)
                if args.low_vram:
                    audio_rows = audio_rows.cpu()

                if args.low_vram:
                    transformer.to(device)
                    # Move encoded latents back to GPU for noising and forward.
                    target_latents = target_latents.to(device)
                    control_rows = control_rows.to(device)
                    prompt_embeds = prompt_embeds.to(device)
                    audio_rows = audio_rows.to(device)

                # ---- noising ----
                # 1. Rows: target `x0`, the noise, the noised `x_t` and the regression target `x0 - noise`.
                x0_rows = patchify_video_latents(target_latents, patch_size)
                noise = torch.randn(
                    target_latents.shape, generator=train_generator, device="cpu", dtype=torch.float32
                ).to(device)
                noise_rows = patchify_video_latents(noise, patch_size)

                # 2. A time on the video schedule pushed through the exponential shift of `MiniMaxH3Scheduler`,
                # `t = 1 - sigma`; the sigma itself comes from the ltx2.3 sampling recipe.
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
                t = 1.0 - shifted_sigma(video_shift, sigma)
                xt_rows = t * x0_rows + (1.0 - t) * noise_rows
                target_rows = x0_rows - noise_rows

                # 2b. Audio noising on the audio schedule (same sigma, audio shift); audio has no condition rows.
                audio_x0_rows = audio_rows
                audio_noise = torch.randn(
                    audio_x0_rows.shape, generator=train_generator, device="cpu", dtype=torch.float32
                ).to(device)
                audio_t = 1.0 - shifted_sigma(audio_shift, sigma)
                audio_xt_rows = audio_t * audio_x0_rows + (1.0 - audio_t) * audio_noise
                audio_target_rows = audio_x0_rows - audio_noise

                # 3. The packed layout and its per-row timestep plan; audio rows are packed alongside the video
                # rows and share one forward pass. Control training runs the t2v layout (no keyframe rows): the
                # control signal enters through the transformer's side branch instead of conditioning rows.
                layout = build_packed_sequence(
                    text_token_tags,
                    num_latent_frames,
                    latent_height,
                    latent_width,
                    num_audio_latents,
                    patch_size,
                    (),
                )
                unique_timesteps, timestep_indices = build_row_timesteps(
                    layout, float(t), float(audio_t), float(t), 1.0
                )

                # 4. One forward over the packed sequence, carrying the clean control rows into the
                # zero-initialised side branch. The transformer aligns every input with the dtype of its
                # projection itself (the patch projections are float32 in the checkpoint), so no autocast.
                video_output, audio_output = transformer(
                    hidden_states=xt_rows[None],
                    audio_hidden_states=audio_xt_rows[None],
                    encoder_hidden_states=prompt_embeds,
                    timestep=unique_timesteps.to(device),
                    timestep_indices=timestep_indices.to(device),
                    token_tags=layout.token_tags.to(device),
                    position_ids=layout.position_ids.to(device),
                    video_indices=layout.video_indices.to(device),
                    audio_indices=layout.audio_indices.to(device),
                    text_indices=layout.text_indices.to(device),
                    control_rows=control_rows[None],
                    return_dict=False,
                )

                # 5. MSE on the generated rows alone, in float32. Video and audio losses are weighted equally
                # (mirrors `scripts/minimax_h3/train.py`). Note that the two `mean` reductions are taken over very
                # different row counts (~5e5 video rows against ~5e3 audio rows), so an equal weight gives every
                # audio element roughly a hundred times the gradient of a video element; the two terms are logged
                # separately below so a run dragged by one modality is visible instead of hidden in `train_loss`.
                video_loss = F.mse_loss(
                    video_output[0].float(), target_rows.float(), reduction="mean"
                )
                audio_loss = F.mse_loss(
                    audio_output[0].float(), audio_target_rows.float(), reduction="mean"
                )
                loss = 0.5 * video_loss + 0.5 * audio_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                train_video_loss += (
                    accelerator.gather(video_loss.detach().repeat(args.train_batch_size)).mean().item()
                    / args.gradient_accumulation_steps
                )
                train_audio_loss += (
                    accelerator.gather(audio_loss.detach().repeat(args.train_batch_size)).mean().item()
                    / args.gradient_accumulation_steps
                )

                # Backpropagate
                if global_step == initial_global_step:
                    print(
                        f"[mem probe] rank={accelerator.process_index} before-backward "
                        f"mem={torch.cuda.memory_allocated() / 1e9:.1f}GB",
                        flush=True,
                    )
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
                    else:
                        norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)
                    if not args.use_deepspeed and not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        writer.add_scalar(f'gradients/norm_sum', norm_sum, global_step=global_step)
                        writer.add_scalar(f'gradients/actual_max_grad_norm', actual_max_grad_norm, global_step=global_step)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if args.use_fsdp and accelerator.sync_gradients:
                    # The frozen parameters keep a gradient of their own under the uniform `requires_grad` above and
                    # the optimizer does not own them, so clear them here: FSDP folds a leftover `.grad` into the
                    # next backward's accumulation (`prepare_gradient_for_backward`) instead of overwriting it. Gated
                    # on `sync_gradients` like `AcceleratedOptimizer.zero_grad`, since `Module.zero_grad` would
                    # otherwise drop the trainable parameters' partial sum mid-accumulation.
                    transformer.zero_grad(set_to_none=True)
                del target_latents, control_latents, control_rows, x0_rows, noise_rows, xt_rows, layout, video_output, audio_output
                if args.enable_inpaint:
                    del mask_pixels, mask_latents, mask_condition_rows, mask_latent_rows, inpaint_rows

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_transformer.step(transformer.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "video_loss": train_video_loss,
                        "audio_loss": train_audio_loss,
                    },
                    step=global_step,
                )
                train_loss = 0.0
                train_video_loss = 0.0
                train_audio_loss = 0.0

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
                        with restore_frozen_requires_grad(transformer, trainable_module_names, args.use_fsdp):
                            accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    if args.use_ema:
                        # Store the transformer parameters temporarily and load the EMA parameters to perform inference.
                        ema_transformer.store(transformer.parameters())
                        ema_transformer.copy_to(transformer.parameters())
                    log_validation(
                        vae, audio_vae, text_encoder, tokenizer, processor, transformer,
                        scheduler, audio_scheduler, args, accelerator, weight_dtype, global_step,
                    )
                    if args.use_ema:
                        # Switch back to the original transformer parameters.
                        ema_transformer.restore(transformer.parameters())

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            if args.use_ema:
                # Store the transformer parameters temporarily and load the EMA parameters to perform inference.
                ema_transformer.store(transformer.parameters())
                ema_transformer.copy_to(transformer.parameters())
            log_validation(
                vae, audio_vae, text_encoder, tokenizer, processor, transformer,
                scheduler, audio_scheduler, args, accelerator, weight_dtype, global_step,
            )
            if args.use_ema:
                # Switch back to the original transformer parameters.
                ema_transformer.restore(transformer.parameters())

        if global_step >= args.max_train_steps:
            break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        if args.use_ema:
            ema_transformer.copy_to(transformer.parameters())

    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        with restore_frozen_requires_grad(transformer, trainable_module_names, args.use_fsdp):
            accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
