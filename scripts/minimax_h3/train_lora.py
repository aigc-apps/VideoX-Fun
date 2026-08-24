# Modified from scripts/wan2.1_fun/train_lora.py for MiniMax-H3, aligned to the videox-fun unified training
# scaffold (parameter set, peft/kohya LoRA switch, comfyui-compatible save, sanity check, checkpointing).
#
# LoRA finetuning of the packed-sequence transformer on the *video and audio* rows together, covering `t2v`
# (text only), `fl2v` (first-frame keyframe conditioning, the keyframe taken from the training sample itself)
# and `ref2va` (reference image / video / audio conditioning, loaded from `transformer_ref`).
# The layout mirrors `scripts/ltx2.3/train_lora.py`: batch-level training (bs=1, the packed layout is per-sample),
# video + audio flow-matching loss weighted 0.5 / 0.5, FSDP + offload composable.
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
#   accelerate launch scripts/minimax_h3/train_lora.py \
#       --pretrained_model_name_or_path=/root/MiniMax-H3 \
#       --train_mode=fl2v --gradient_checkpointing --low_vram

import argparse
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
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers.utils.torch_utils import is_compiled_module
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers.utils import ContextManagers

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.data import (ASPECT_RATIO_512,
                             AspectRatioBatchImageVideoSampler,
                             ImageVideoSampler, RandomSampler,
                             VideoSpeechDataset, get_closest_ratio)
from videox_fun.models import (AutoencoderKLMiniMaxH3,
                               AutoencoderKLMiniMaxH3Audio,
                               MiniMaxH3Transformer3DModel, Qwen2TokenizerFast,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor)
from videox_fun.pipeline import MiniMaxH3Pipeline
from videox_fun.pipeline.pipeline_minimax_h3 import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED, MINIMAX_H3_FPS, MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD, MINIMAX_H3_TEXT_ENCODER_LAYER, MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_SAMPLE_FPS, MINIMAX_H3_VIDEO_TAG,
    _offload_scope, align_num_frames, audio_latent_num_frames,
    build_packed_sequence, build_ref2va_packed_sequence, build_row_timesteps,
    check_ref2va_references, keyframe_condition_noise,
    normalize_ref2va_references, patchify_video_latents, prepare_keyframe_image,
    ref2va_condition_rows, video_latent_num_frames)
from videox_fun.utils import MiniMaxH3Scheduler
from videox_fun.utils.lora_utils import convert_peft_lora_to_kohya_lora, create_network
from videox_fun.utils.utils import save_videos_grid

# Silences diffusers' `randn_tensor` notice about CPU generators producing CUDA tensors (the tensor is created
# on CPU and moved to GPU; harmless, only a marginal speed note).
warnings.filterwarnings("ignore", message="The passed generator was created on")

def _mm_token_type_ids(tokenizer, token_ids):
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_pad_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    return [1 if t == image_pad_id else 2 if t == video_pad_id else 0 for t in token_ids]


def gather_ref2va_vision_features(processor, references):
    r"""Run references' pixels through the conditioner's processors, batched per modality.

    Returns (vision_inputs, image_token_counts, video_block_token_counts, video_block_timestamps).
    """
    merge_size = processor.image_processor.merge_size**2
    vision_inputs = {}
    image_token_counts = []
    video_block_token_counts = []
    video_block_timestamps = []

    images = [ref.image for ref in references if ref.kind == "image"]
    if images:
        image_features = processor.image_processor(images=images, return_tensors="pt")
        vision_inputs["pixel_values"] = image_features["pixel_values"]
        vision_inputs["image_grid_thw"] = image_features["image_grid_thw"]
        image_token_counts = [
            int(grid.prod()) // merge_size for grid in image_features["image_grid_thw"]
        ]

    videos = [ref for ref in references if ref.kind == "video"]
    if videos:
        temporal_patch = processor.video_processor.temporal_patch_size
        sampled = [
            MiniMaxH3Pipeline._sample_ref2va_condition_frames(
                ref.frames, float(ref.fps), MINIMAX_H3_VIDEO_SAMPLE_FPS, temporal_patch
            )
            for ref in videos
        ]
        video_block_timestamps = [timestamps for _, timestamps in sampled]
        video_features = processor.video_processor(
            videos=[np.stack(frames) for frames, _ in sampled], do_sample_frames=False, return_tensors="pt"
        )
        vision_inputs["pixel_values_videos"] = video_features["pixel_values_videos"]
        vision_inputs["video_grid_thw"] = video_features["video_grid_thw"]
        video_block_token_counts = [
            int(grid[1]) * int(grid[2]) // merge_size for grid in video_features["video_grid_thw"]
        ]
        for timestamps, grid in zip(video_block_timestamps, video_features["video_grid_thw"]):
            if int(grid[0]) != len(timestamps):
                raise ValueError(
                    f"The processor merged a reference video into {int(grid[0])} vision blocks, but MiniMax-H3 "
                    f"labels {len(timestamps)} of them."
                )
    return vision_inputs, image_token_counts, video_block_token_counts, video_block_timestamps


def build_ref2va_presentation(tokenizer, references, image_token_counts, video_block_token_counts,
                              video_block_timestamps, prompt):
    r"""Tokenize MiniMax-H3's presentation of a `ref2va` request."""

    def text(value):
        ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        return ids, [MINIMAX_H3_TEXT_TAG] * len(ids)

    def vision(pad_token, num_tokens):
        ids = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        return ids, [MINIMAX_H3_VIDEO_TAG] * len(ids)

    token_ids, token_tags = [], []
    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            ids, tags = text(f"<Audio {counts['audio']}>: ")
            token_ids += ids
            token_tags += tags
        if reference.kind == "image":
            counts["image"] += 1
            ids, tags = text(f"<Picture {counts['image']}>: ")
            token_ids += ids
            token_tags += tags
            ids, tags = vision("<|image_pad|>", image_token_counts[counts["image"] - 1])
            token_ids += ids
            token_tags += tags
        elif reference.kind == "video":
            counts["video"] += 1
            ids, tags = text(f"<Video {counts['video']}>: ")
            token_ids += ids
            token_tags += tags
            for timestamp in video_block_timestamps[counts["video"] - 1]:
                ids, tags = text(f"<{timestamp:.1f} seconds>")
                token_ids += ids
                token_tags += tags
                ids, tags = vision("<|video_pad|>", video_block_token_counts[counts["video"] - 1])
                token_ids += ids
                token_tags += tags
    ids, tags = text(prompt)
    token_ids += ids
    token_tags += tags
    return token_ids, token_tags


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


def encode_prompt(
    text_encoder, tokenizer, processor, prompt,
    images=None, references=None, device=None, dtype=None,
):
    r"""Build MiniMax-H3's presentation of a request and encode it.

    The presentation is the verbatim prompt for `t2va`. Every keyframe prepends a `"<Picture i>: "` label and a
    vision block (`<|vision_start|>`, one `<|image_pad|>` per vision patch, `<|vision_end|>`) — no chat template
    and no special tokens. The rows of a vision block are tagged as *video* rather than text, which is what the
    transformer's AdaLN modulation keys off.

    When `references` is given, the `ref2va` presentation is built instead and `images` is ignored.
    """
    num_layers = text_encoder.config.text_config.num_hidden_layers
    if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER:
        raise ValueError(
            f"MiniMax-H3 conditions on `hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}]` of its Qwen3-VL "
            f"conditioner, which needs more than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers, but "
            f"`text_encoder` has {num_layers}. The last hidden state of a stack truncated to exactly "
            f"{MINIMAX_H3_TEXT_ENCODER_LAYER} layers is post-norm and is not the conditioning MiniMax-H3 expects."
        )

    pixel_values, image_grid_thw = None, None
    vision_inputs = {}
    token_ids, token_tags = [], []
    if references:
        # The ref2va presentation: vision features gathered per modality, tokenization in request order.
        vision_inputs, image_token_counts, video_block_token_counts, video_block_timestamps = (
            gather_ref2va_vision_features(processor, references)
        )
        token_ids, token_tags = build_ref2va_presentation(
            tokenizer, references, image_token_counts, video_block_token_counts,
            video_block_timestamps, prompt,
        )
        pixel_values = vision_inputs.get("pixel_values")
        image_grid_thw = vision_inputs.get("image_grid_thw")
    elif images:
        vision = processor.image_processor(images=images, return_tensors="pt")
        pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
        merge_size = processor.image_processor.merge_size**2
        for index in range(len(images)):
            num_image_tokens = int(image_grid_thw[index].prod()) // merge_size
            label_ids = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
            vision_ids = (
                [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            token_ids += label_ids + vision_ids
            token_tags += [MINIMAX_H3_TEXT_TAG] * len(label_ids) + [MINIMAX_H3_VIDEO_TAG] * len(vision_ids)
    prompt_ids = [] if references else tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_ids += prompt_ids
    token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)
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
        pixel_values=None if pixel_values is None else pixel_values.to(device, text_encoder.dtype),
        image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
        use_cache=False,
        output_hidden_states=True,
    )
    if "pixel_values_videos" in vision_inputs:
        encoder_kwargs["pixel_values_videos"] = vision_inputs["pixel_values_videos"].to(
            device, text_encoder.dtype
        )
        encoder_kwargs["video_grid_thw"] = vision_inputs["video_grid_thw"].to(device)
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


def encode_keyframes(vae, patch_size, images, device):
    r"""Encode the `fl2va` keyframes into packed conditioning rows.

    The keyframes go through the video VAE's spatial encoder only — they are single frames, so none of its
    17-frame temporal chunking applies — and the posterior is *sampled*, under a generator seeded with 42
    independently of the request seed. The sampled latent is rounded to float16 before being normalized, as in the
    reference implementation; both are part of reproducing the released model's conditioning.
    """
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)

    rows = []
    with _offload_scope(vae):
        for image in images:
            pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None, :, None]
            pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
            moments = vae._encode_clip(pixels)
            posterior = DiagonalGaussianDistribution(moments)
            latents = posterior.sample(generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
            latents = latents.to(torch.float16).float().cpu()
            rows.append(patchify_video_latents((latents - latents_mean) / latents_std, patch_size))
    return torch.cat(rows)


def encode_reference_latents_for_training(
    vae, audio_vae, references, patch_size, device,
    audio_latent_channels=None,
):
    r"""Encode the `ref2va` references for training.

    Image and video references go through the video VAE (sampled posterior, float16 rounding, normalized);
    audio references go through the audio VAE (posterior mean, normalized). Mirrors
    `MiniMaxH3Pipeline.encode_reference_latents` without requiring a pipeline instance.
    """
    if audio_latent_channels is None:
        audio_latent_channels = getattr(audio_vae.config, "latent_channels", 32)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    frames_per_chunk = getattr(vae, "frames_per_chunk", 17)
    latents_per_chunk = getattr(vae, "latents_per_chunk", 5)

    def encode_pixels(pixels):
        pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
        posterior = DiagonalGaussianDistribution(vae._encode(pixels))
        latents = posterior.sample(
            generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
        )
        latents = latents.to(torch.float16).float().cpu()
        return (latents - latents_mean) / latents_std

    condition_latents = []
    with _offload_scope(vae):
        for reference in references:
            if reference.kind == "image":
                pixels = torch.from_numpy(np.array(reference.image)).to(device).permute(2, 0, 1)[None, :, None]
                condition_latents.append(encode_pixels(pixels))
            elif reference.kind == "video":
                num_frames = reference.frames.shape[0]
                num_frames = (
                    max(1, (num_frames - latents_per_chunk) // frames_per_chunk) * frames_per_chunk
                    + latents_per_chunk
                )
                pixels = (
                    torch.from_numpy(reference.frames[:num_frames].copy()).to(device).permute(3, 0, 1, 2)[None]
                )
                condition_latents.append(encode_pixels(pixels))

    audio_latents_mean = torch.tensor(audio_vae.config.latents_mean).view(1, 1, -1)
    audio_latents_std = torch.tensor(audio_vae.config.latents_std).view(1, 1, -1)
    audio_condition_latents = []
    with _offload_scope(audio_vae):
        for reference in references:
            if reference.has_audio:
                posterior = audio_vae.encode(reference.audio.to(device)[:, None], return_dict=False)[0]
                latents = posterior.mode().float().cpu().transpose(1, 2)
                normalized = (latents - audio_latents_mean) / audio_latents_std
                audio_condition_latents.append(normalized.reshape(-1, audio_latent_channels))
    return condition_latents, audio_condition_latents


def shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    r"""The exponential sigma shift of `MiniMaxH3Scheduler`, `sigma' = s*sigma / (1 + (s-1)*sigma)`."""
    return shift * sigma / (1 + (shift - 1) * sigma)


logger = get_logger(__name__, log_level="INFO")


def log_validation(
    vae, audio_vae, text_encoder, tokenizer, processor, transformer,
    scheduler, audio_scheduler, args, accelerator, weight_dtype, global_step,
):
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=weight_dtype):
            logger.info("Running validation... ")
            pipeline = MiniMaxH3Pipeline(
                vae=vae,
                audio_vae=audio_vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                processor=processor,
                transformer=accelerator.unwrap_model(transformer),
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

            for i in range(len(args.validation_prompts)):
                output = pipeline(
                    args.validation_prompts[i],
                    height=args.video_sample_size,
                    width=args.video_sample_size,
                    num_frames=args.video_sample_n_frames,
                    num_inference_steps=50,
                    generator=generator,
                )
                sample = output.videos
                os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                save_videos_grid(
                    sample,
                    os.path.join(
                        args.output_dir,
                        f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.mp4",
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


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMax-H3 LoRA training (video + audio, t2v / fl2v / ref2va).")
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
        default="samples/minimax-h3-lora",
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
    parser.add_argument("--save_state", action="store_true", help="Whether or not to save state.")
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
    parser.add_argument(
        "--use_peft_lora", action="store_true", help="Whether or not to use peft lora."
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
        "--target_name",
        type=str,
        default="to_,ff,linear_",
        help=("The module is trained in loras."),
    )
    parser.add_argument(
        "--lora_skip_name",
        type=str,
        default=None,
        help=("The module is not trained in loras."),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    # MiniMax-H3 specific
    parser.add_argument(
        "--train_mode",
        type=str,
        default="fl2v",
        choices=["t2v", "fl2v", "ref2va"],
        help="t2v (text only), fl2v (first-frame keyframe conditioning), or ref2va (reference to video+audio).",
    )
    parser.add_argument(
        "--t2v_ratio",
        type=float,
        default=0.0,
        help=("Under --train_mode=fl2v, the fraction of steps that drop the keyframe and train t2v instead, so one "
              "run keeps both conditionings. 0 trains fl2v only."),
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
        default="minimax_h3_lora",
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

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    if args.train_mode not in ("t2v", "fl2v", "ref2va"):
        raise ValueError(f"`train_mode` must be 't2v', 'fl2v' or 'ref2va', got {args.train_mode!r}.")
    if not 0.0 <= args.t2v_ratio <= 1.0:
        raise ValueError(f"`t2v_ratio` is a probability and must be in [0, 1], got {args.t2v_ratio}.")
    if args.t2v_ratio > 0.0 and args.train_mode != "fl2v":
        raise ValueError(
            f"`t2v_ratio` mixes t2v steps into an fl2v run, so it only applies to `--train_mode=fl2v`, but "
            f"`train_mode` is {args.train_mode!r}. Drop `--t2v_ratio` to train {args.train_mode} only."
        )
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
    # `pretrained_model_name_or_path` may point at a converted diffusers layout or at an *original* MiniMax-H3
    # partition; every component's `from_pretrained` auto-detects the layout and stream-converts the original
    # shards on the fly, so the caller never branches on the format itself.
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", low_cpu_mem_usage=True, torch_dtype=weight_dtype,
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

    # Freeze everything; the LoRA modules created below are the only trainable parameters.
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    audio_vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Conditioning (`encode_prompt` / `encode_keyframes`) runs through the module-level helpers; the pipeline is
    # only constructed inside `log_validation` for inference, mirroring `scripts/ltx2/train.py`.

    # ------------------------------------------------------------------ LoRA
    # peft path: inject adapters straight into the transformer; kohya path: build a LoRANetwork and apply it.
    if args.use_peft_lora:
        from peft import (LoraConfig, get_peft_model_state_dict,
                          inject_adapter_in_model)
        lora_config = LoraConfig(r=args.rank, lora_alpha=args.network_alpha, target_modules=args.target_name.split(","))
        transformer = inject_adapter_in_model(lora_config, transformer)
        network = None
    else:
        network = create_network(
            1.0,
            args.rank,
            args.network_alpha,
            text_encoder,
            transformer,
            neuron_dropout=None,
            target_name=args.target_name,
            skip_name=args.lora_skip_name,
        )
        network = network.to(weight_dtype)
        network.apply_to(text_encoder, transformer, args.train_text_encoder, True)

        # Re-parent each LoRA module under its target layer so FSDP shards it with the enclosing
        # MiniMaxH3TransformerBlock; keeping every LoRA weight in the root unit would leave them
        # unsharded from forward through the whole backward. The duplicate registration under `network`
        # must be dropped, otherwise the same parameter would live in two FSDP units. The org module
        # reference was deleted by `apply_to`, but the patched bound method still carries it.
        path_by_id = {id(m): n for n, m in transformer.named_modules()}
        lora_save_keys = []
        for lora in network.unet_loras:
            org_module = lora.org_forward.__self__
            setattr(org_module, "lora_adapter", lora)
            network._modules.pop(lora.lora_name, None)
            lora_save_keys.append((f"{path_by_id[id(org_module)]}.lora_adapter", lora.lora_name))

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

    # ------------------------------------------------------------------ save / load hooks
    # `accelerate` 0.16.0+ supports custom saving hooks. peft weights are exported in both the native and the
    # kohya (`lora_unet_*`) spellings so the predict scripts and ComfyUI load them with zero changes.
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        if fsdp_stage != 0 or zero_stage == 3:
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    from safetensors.torch import save_file
                    safetensor_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model.safetensors")
                    if args.use_peft_lora:
                        network_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(models[-1]), accelerate_state_dict)
                        network_state_dict_kohya = convert_peft_lora_to_kohya_lora(network_state_dict)
                        safetensor_kohya_format_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model_compatible_with_comfyui.safetensors")
                        save_model(safetensor_kohya_format_save_path, network_state_dict_kohya)
                    else:
                        # The LoRA modules were re-parented under their target layers, so rebuild the original
                        # kohya key names (`lora_unet_*`) from the recorded mapping instead of filtering `network`.
                        network_state_dict = {}
                        for new_prefix, lora_name in lora_save_keys:
                            prefix = new_prefix + "."
                            for key, value in accelerate_state_dict.items():
                                if key.startswith(prefix):
                                    network_state_dict[f"{lora_name}.{key[len(prefix):]}"] = value.to(weight_dtype)
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
                    if args.use_peft_lora:
                        network_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(models[-1]), accelerate_state_dict)
                        network_state_dict_kohya = convert_peft_lora_to_kohya_lora(network_state_dict)
                        safetensor_kohya_format_save_path = os.path.join(output_dir, f"lora_diffusion_pytorch_model_compatible_with_comfyui.safetensors")
                        save_model(safetensor_kohya_format_save_path, network_state_dict_kohya)
                    else:
                        # The LoRA modules were re-parented under their target layers, so rebuild the original
                        # kohya key names (`lora_unet_*`) from the recorded mapping instead of filtering `network`.
                        network_state_dict = {}
                        for new_prefix, lora_name in lora_save_keys:
                            prefix = new_prefix + "."
                            for key, value in accelerate_state_dict.items():
                                if key.startswith(prefix):
                                    network_state_dict[f"{lora_name}.{key[len(prefix):]}"] = value.to(weight_dtype)
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
        transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    trainable_params = [p for p in transformer.parameters() if p.requires_grad] if args.use_peft_lora else [p for lora in network.unet_loras for p in lora.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"LoRA created: {len(trainable_params)} tensors, {num_trainable / 1e6:.2f} M parameters (peft={bool(args.use_peft_lora)}).")

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
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ------------------------------------------------------------------ data
    # MiniMax-H3 phase two trains video *and* audio rows of the packed sequence, so the dataset must carry the paired
    # waveform; `VideoSpeechDataset` reads the `audio_path` field of the training meta. The three audio flags put the
    # waveform on the inference route of `normalize_reference_audio`: sliced at the file's native rate and resampled
    # once with the pipeline's torchaudio pass onto the audio VAE's sample rate (32 kHz, 40 latents/s), stereo kept
    # as released, over the `num_frames / fps` span the audio latent grid keys off.
    audio_sr = getattr(audio_vae.config, "sampling_rate", 32000)
    train_dataset = VideoSpeechDataset(
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
        enable_ref2va=(args.train_mode == "ref2va"),
    )

    # The packed-sequence layout (text tokens + condition + audio + video rows) is per-sample, so a batch larger
    # than one would need a sample loop; batch-level training (mirroring `scripts/ltx2.3/train_lora.py`) therefore
    # pins the batch size to one for now.
    if args.train_batch_size != 1:
        raise ValueError("MiniMax-H3 packed-sequence training requires --train_batch_size=1.")

    # The MiniMax-H3 video VAE encodes 17n + 5 frames, so bucket frame counts bucket in steps of 17
    # (ltx2.3's magvae equivalent is `vae.config.temporal_compression_ratio` with the 4n + 1 form).
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
            new_examples                 = {}
            new_examples["pixel_values"] = []
            new_examples["text"]         = []
            new_examples["audio"]        = []
            new_examples["fps"]          = []

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

                # Get adapt hw for resize
                if closest_size[0] / h > closest_size[1] / w:
                    resize_size = closest_size[0], int(w * closest_size[0] / h)
                else:
                    resize_size = int(h * closest_size[1] / w), closest_size[1]

                transform = transforms.Compose([
                    transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                    transforms.CenterCrop(closest_size),
                ])

                new_examples["pixel_values"].append(transform(pixel_values)[:batch_video_length])
                new_examples["text"].append(example["text"])

                # The waveform is `(channels, num_samples)` on the stereo route, so the length is the last axis.
                audio_length = example["audio"].shape[-1]
                batch_audio_length = int(audio_length / pixel_values.size()[0] * batch_video_length)
                # The `num_frames / fps` span the audio latent grid keys off, as in the inference pipeline.
                target_audio_length = int(round(batch_video_length / MINIMAX_H3_FPS * audio_sr))
                new_examples["audio"].append(
                    resample_waveform_to_span(example["audio"][..., :batch_audio_length], target_audio_length)
                )
                new_examples["fps"].append(example.get("fps", 24))

            # Limit the number of frames to the same
            new_examples["pixel_values"] = torch.stack([example for example in new_examples["pixel_values"]])

            # Pad audio to same length and stack
            max_audio_length = max(audio.shape[-1] for audio in new_examples["audio"])
            new_examples["audio"] = torch.stack([
                F.pad(audio, (0, max_audio_length - audio.shape[-1]))
                for audio in new_examples["audio"]
            ])
            new_examples["fps"] = new_examples["fps"]
            # Under `ref2va`, pass the references through (bs=1 so always one entry).
            if args.train_mode == "ref2va":
                new_examples["references"] = [example.get("references") for example in examples]
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
            # `VideoSpeechDataset` returns `[-1, 1]` pixels; MiniMax-H3 wants `[0, 1]` and ImageNet-normalizes the
            # encoder input itself, so hand the loop `[0, 1]` and drop the rest. The audio waveform is sliced to the
            # video span and right-padded to a common length so the batch stacks.
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
            new_examples                 = {}
            new_examples["pixel_values"] = []
            new_examples["text"]         = []
            new_examples["audio"]        = []
            new_examples["fps"]          = []

            for example in examples:
                # To 0~1
                pixel_values = example["pixel_values"][:batch_video_length]
                new_examples["pixel_values"].append(pixel_values * 0.5 + 0.5)
                new_examples["text"].append(example["text"])

                audio_length = example["audio"].shape[-1]
                batch_audio_length = int(audio_length / example["pixel_values"].shape[0] * batch_video_length)
                target_audio_length = int(round(batch_video_length / MINIMAX_H3_FPS * audio_sr))
                new_examples["audio"].append(
                    resample_waveform_to_span(example["audio"][..., :batch_audio_length], target_audio_length)
                )
                new_examples["fps"].append(example.get("fps", 24))

            # Limit the number of frames to the same
            new_examples["pixel_values"] = torch.stack([example for example in new_examples["pixel_values"]])

            # Pad audio to same length and stack
            max_audio_length = max(audio.shape[-1] for audio in new_examples["audio"])
            new_examples["audio"] = torch.stack([
                F.pad(audio, (0, max_audio_length - audio.shape[-1]))
                for audio in new_examples["audio"]
            ])
            new_examples["fps"] = new_examples["fps"]
            # Under `ref2va`, pass the references through (bs=1 so always one entry).
            if args.train_mode == "ref2va":
                new_examples["references"] = [example.get("references") for example in examples]
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

    # Attach the network to the transformer so `accelerator.prepare` shards / wraps them together and the saving
    # hook can read the LoRA weights out of the prepared model. Cast to `weight_dtype` *before* prepare so FSDP
    # flattens a uniform-dtype parameter set (the conversion mixin pins a few modules in float32 for inference
    # precision and kohya LoRA modules default to float32).
    if network is not None:
        transformer.network = network
    transformer.gradient_checkpointing_save_on_cpu = args.gradient_checkpointing_save_on_cpu
    transformer = transformer.to(weight_dtype)

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Shard the frozen text encoder *after* prepare (mirrors `scripts/ltx2.3/train_lora.py`): the Qwen3-VL
    # conditioner (~62 GB) is wrapped per decoder layer so the per-step unshard footprint stays small, and a
    # post-prepare shard keeps the text encoder out of the trainable FSDP unit.
    sharded_text_encoder = fsdp_stage != 0 or zero_stage != 0
    if sharded_text_encoder:
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
    transformer.to(device)
    if not sharded_text_encoder:
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
    # The t2v / fl2v draw of a mixed run gets its own generator, seeded per rank (mirroring `log_validation`) so the
    # ranks of one global batch do not all land on the same conditioning and every step mixes the two.
    mode_generator = torch.Generator(device="cpu")
    if args.seed is not None:
        mode_generator.manual_seed(args.seed + accelerator.process_index)

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
    logger.info(f"  Video / audio loss weights = {args.video_loss_weight} / {args.audio_loss_weight}")

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
                if network is not None:
                    m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                else:
                    m, u = accelerator.unwrap_model(transformer).load_state_dict(state_dict, strict=False)
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
        train_t2v_share = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            # Sanity check: save the first batch so a glance at output_dir/sanity_check confirms the data pipe.
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch["pixel_values"].cpu(), batch["text"]
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None].permute(0, 2, 1, 3, 4)
                    gif_name = "-".join(text.replace("/", "").split()[:10]) if not text == "" else f"{global_step}-{idx}"
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.mp4", rescale=False)

            with accelerator.accumulate(transformer):
                # Batch-level training (bs=1): the packed-sequence layout (text tokens + condition + audio +
                # video rows) is per-sample, so a batch larger than one would need a sample loop. With bs=1 the
                # batch *is* the sample, and the encode / noise / forward / loss mirror
                # `scripts/ltx2.3/train_lora.py` without a sample loop.
                pixel_values = batch["pixel_values"][0]
                text = batch["text"][0]
                audio = batch["audio"][0]
                # MiniMax-H3 has no fps input: its temporal rotary grid (`_temporal_position_grid`) and its audio
                # latent grid (`audio_latent_num_frames`, 40 latents/s against 24 fps) are both hard-wired to 24 fps,
                # unlike ltx2.3 which conditions on fps through `prepare_video_coords`. `batch["fps"]` cannot police
                # that: the dataset floors it (`int(fps // stride)`), so the very common 23.976 fps arrives as 23 and
                # is indistinguishable from a genuine 23 fps source. The audio latent count further down is the real
                # gate — it measures the video / audio span mismatch directly, in the units the layout keys off.

                # The 33 B transformer alone fills ~66 GB, so under `low_vram` it yields the GPU while the VAE and
                # the conditioner encode, and moves back for the forward / backward. Under FSDP it stays put: a
                # `.to()` on the wrapped model breaks the FlatParameter sharding (every rank ends up holding a full
                # copy), and a sharded 33 B model is only ~10 GB per rank on 8 cards, so it does not need offloading.
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

                # Under `low_vram`, load both VAEs at once and keep them on GPU for the video, keyframe and
                # audio encodes in one session — the video VAE was previously loaded twice (once for video
                # encode, once for keyframe encode) and the audio VAE was loaded separately.
                if args.low_vram:
                    vae.to(device)
                    audio_vae.to(device)

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
                    target_latents = _batch_encode_vae(pixels)
                if args.low_vram:
                    target_latents = target_latents.cpu()

                # The fl2v keyframe is the sample's own first frame, prepared onto the canvas exactly like
                # inference does (stretch: it is the geometry anchor). The keyframe image is a CPU-side PIL
                # conversion, so the video VAE stays idle on GPU for a moment under `low_vram`.
                # `--t2v_ratio` drops the keyframe on that fraction of steps: without the keyframe the presentation
                # loses its vision block and the packed sequence loses its condition rows, which is exactly a t2v
                # step, so one run can hold on to both conditionings instead of drifting to fl2v alone.
                keyframe, keyframe_anchors = None, ()
                step_mode = args.train_mode
                references = None
                if step_mode == "fl2v" and args.t2v_ratio > 0.0:
                    if float(torch.rand((), generator=mode_generator)) < args.t2v_ratio:
                        step_mode = "t2v"
                elif step_mode == "ref2va":
                    # The dataset hands over `MiniMaxH3Reference` objects; a sample without references or with an
                    # unparseable entry falls back to t2v so the retry of `__getitem__` does not have to.
                    raw_refs = batch.get("references", [None])[0]
                    if raw_refs:
                        references = list(raw_refs)
                        references = check_ref2va_references(references)
                        references = normalize_ref2va_references(references, num_frames, audio_sr)
                    else:
                        step_mode = "t2v"
                if step_mode == "fl2v":
                    keyframe = Image.fromarray(
                        (pixel_values[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    ).convert("RGB")
                    keyframe = prepare_keyframe_image(keyframe, height, width, stretch=True)
                    keyframe_anchors = ("first",)

                # The conditioner reads `hidden_states[50]` of Qwen3-VL; the presentation of an `fl2v` request
                # carries the keyframe's vision block ahead of the prompt, tagged as video rows. An FSDP-sharded
                # text encoder tolerates symmetric `.to` moves, so it is brought on-device right before the encode
                # and back to CPU afterwards.
                if args.low_vram:
                    text_encoder.to(device)
                with torch.no_grad():
                    if step_mode == "ref2va" and references is not None:
                        prompt_embeds, text_token_tags = encode_prompt(
                            text_encoder, tokenizer, processor,
                            text, references=references, device=device, dtype=weight_dtype,
                        )
                    else:
                        prompt_embeds, text_token_tags = encode_prompt(
                            text_encoder, tokenizer, processor,
                            text, None if keyframe is None else [keyframe], device=device, dtype=weight_dtype
                        )
                if args.low_vram:
                    text_encoder.to("cpu")
                    torch.cuda.empty_cache()
                    prompt_embeds = prompt_embeds.cpu()

                # Conditioning rows: the keyframe latent, sampled under the released seed-42 / float16-rounding
                # contract and noise-augmented to MiniMax-H3's conditioning level. The video VAE is still on GPU
                # from the video encode above under `low_vram` — no second onload needed.
                condition_rows = None
                ref_condition_latents = None
                ref_audio_condition_latents = []
                if keyframe is not None:
                    with torch.no_grad():
                        condition_rows = encode_keyframes(vae, patch_size, [keyframe], device=device)
                    if args.low_vram:
                        condition_rows = condition_rows.cpu()
                elif step_mode == "ref2va" and references is not None:
                    # Encode the references: images and videos through the video VAE, soundtracks through the audio
                    # VAE. The video VAE is still on GPU from the target encode under `low_vram`.
                    with torch.no_grad():
                        ref_condition_latents, ref_audio_condition_latents = encode_reference_latents_for_training(
                            vae, audio_vae, references, patch_size, device,
                            audio_latent_channels=audio_channels,
                        )
                    if args.low_vram:
                        ref_condition_latents = [c.cpu() for c in ref_condition_latents]
                        ref_audio_condition_latents = [a.cpu() for a in ref_audio_condition_latents]

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
                    prompt_embeds = prompt_embeds.to(device)
                    audio_rows = audio_rows.to(device)
                    if ref_condition_latents is not None:
                        ref_condition_latents = [c.to(device) for c in ref_condition_latents]
                    if ref_audio_condition_latents:
                        ref_audio_condition_latents = [a.to(device) for a in ref_audio_condition_latents]

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

                # 3. Prepend the conditioning rows, noise-augmented and pinned at their augmentation level.
                num_condition_rows = 0
                num_condition_audio_rows = 0
                condition_timestep = float(t)
                if condition_rows is not None:
                    condition_rows = condition_rows.to(device)
                    condition_noise = keyframe_condition_noise(
                        ((1, latent_height, latent_width),),
                        patch_size,
                        latent_channels,
                        generator=train_generator,
                        device=device,
                    )
                    condition_rows = scheduler.scale_noise(
                        condition_rows, MINIMAX_H3_KEYFRAME_NOISE_AUG, condition_noise
                    )
                    xt_rows = torch.cat([condition_rows, xt_rows])
                    num_condition_rows = condition_rows.shape[0]
                    condition_timestep = max(float(t), MINIMAX_H3_KEYFRAME_NOISE_AUG)
                elif ref_condition_latents is not None:
                    # ref2va: noise the visual conditions to t=0.999 and prepend; audio conditions ride clean
                    # at t=0, packed with the target audio rows.
                    condition_rows = ref2va_condition_rows(
                        scheduler, ref_condition_latents, patch_size,
                        generator=train_generator, device=device,
                    )
                    xt_rows = torch.cat([condition_rows, xt_rows])
                    num_condition_rows = condition_rows.shape[0]
                    condition_timestep = max(float(t), MINIMAX_H3_KEYFRAME_NOISE_AUG)
                    if ref_audio_condition_latents:
                        audio_condition_rows = torch.cat([
                            rows.to(device) for rows in ref_audio_condition_latents
                        ])
                        audio_xt_rows = torch.cat([audio_condition_rows, audio_xt_rows])
                        num_condition_audio_rows = audio_condition_rows.shape[0]

                # 4. The packed layout and its per-row timestep plan; audio rows are packed alongside the video
                # rows and share one forward pass.
                if step_mode == "ref2va" and references is not None:
                    layout = build_ref2va_packed_sequence(
                        text_token_tags,
                        references,
                        ref_condition_latents,
                        ref_audio_condition_latents,
                        num_latent_frames,
                        latent_height,
                        latent_width,
                        num_audio_latents,
                        patch_size,
                    )
                else:
                    layout = build_packed_sequence(
                        text_token_tags,
                        num_latent_frames,
                        latent_height,
                        latent_width,
                        num_audio_latents,
                        patch_size,
                        keyframe_anchors,
                    )
                unique_timesteps, timestep_indices = build_row_timesteps(
                    layout, float(t), float(audio_t), condition_timestep, 1.0
                )

                # 5. One forward over the packed sequence. The transformer aligns every input with the dtype of
                # its projection itself (the patch projections are float32 in the checkpoint), so no autocast.
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
                    return_dict=False,
                )

                # 6. MSE on the generated rows alone, in float32: the conditioning rows are re-imposed by
                # construction and never supervised. The two losses are weighted by `--video_loss_weight` /
                # `--audio_loss_weight` (0.5 / 0.5 by default, the split of `scripts/ltx2.3/train_lora.py`). Note
                # that the two `mean` reductions are taken over very different row counts (~5e5 video rows against
                # ~5e3 audio rows), so an equal weight gives every audio element roughly a hundred times the
                # gradient of a video element; the two terms are logged separately below so a run that is being
                # dragged by one modality is visible instead of hidden inside `train_loss`.
                video_loss = F.mse_loss(
                    video_output[0, num_condition_rows:].float(), target_rows.float(), reduction="mean"
                )
                audio_loss = F.mse_loss(
                    audio_output[0, num_condition_audio_rows:].float(), audio_target_rows.float(), reduction="mean"
                )
                loss = args.video_loss_weight * video_loss + args.audio_loss_weight * audio_loss

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
                # The realized share of t2v steps across all ranks, so the log shows what the run actually mixed
                # rather than what was requested.
                train_t2v_share += (
                    accelerator.gather(
                        torch.full((args.train_batch_size,), float(step_mode == "t2v"), device=accelerator.device)
                    ).mean().item()
                    / args.gradient_accumulation_steps
                )

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                del target_latents, x0_rows, noise_rows, xt_rows, layout, video_output, audio_output

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "video_loss": train_video_loss,
                        "audio_loss": train_audio_loss,
                        "t2v_share": train_t2v_share,
                    },
                    step=global_step,
                )
                train_loss = 0.0
                train_video_loss = 0.0
                train_audio_loss = 0.0
                train_t2v_share = 0.0

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
                        if not args.save_state:
                            if args.use_peft_lora:
                                safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                                network_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(transformer))
                                save_model(safetensor_save_path, network_state_dict)

                                safetensor_kohya_format_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-compatible_with_comfyui.safetensors")
                                network_state_dict_kohya = convert_peft_lora_to_kohya_lora(network_state_dict)
                                save_model(safetensor_kohya_format_save_path, network_state_dict_kohya)
                                logger.info(f"Saved safetensor to {safetensor_save_path}")
                            else:
                                safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                                save_model(safetensor_save_path, accelerator.unwrap_model(network))
                                logger.info(f"Saved safetensor to {safetensor_save_path}")
                        else:
                            accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(accelerator_save_path)
                            logger.info(f"Saved state to {accelerator_save_path}")

                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    log_validation(
                        vae, audio_vae, text_encoder, tokenizer, processor, transformer,
                        scheduler, audio_scheduler, args, accelerator, weight_dtype, global_step,
                    )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            log_validation(
                vae, audio_vae, text_encoder, tokenizer, processor, transformer,
                scheduler, audio_scheduler, args, accelerator, weight_dtype, global_step,
            )

        if global_step >= args.max_train_steps:
            break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if not args.save_state:
            if args.use_peft_lora:
                safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                network_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(transformer))
                save_model(safetensor_save_path, network_state_dict)

                safetensor_kohya_format_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-compatible_with_comfyui.safetensors")
                network_state_dict_kohya = convert_peft_lora_to_kohya_lora(network_state_dict)
                save_model(safetensor_kohya_format_save_path, network_state_dict_kohya)
                logger.info(f"Saved safetensor to {safetensor_save_path}")
            else:
                safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                save_model(safetensor_save_path, accelerator.unwrap_model(network))
                logger.info(f"Saved safetensor to {safetensor_save_path}")
        else:
            accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(accelerator_save_path)
            logger.info(f"Saved state to {accelerator_save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
