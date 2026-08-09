# Modified from the MiniMax-H3 modular blocks of
# https://github.com/huggingface/diffusers/tree/main/src/diffusers/modular_pipelines/minimax_h3
# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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
# limitations under the License.

r"""
MiniMax-H3 text/keyframe to video + audio pipeline.

MiniMax-H3 generates a video and its soundtrack **jointly**: one transformer denoises a single packed sequence that
holds the text conditioning, the keyframe conditioning rows, the target audio rows and the target video rows at once,
with full self-attention over all of it. There is no separate vocoder and no audio post-hoc pass.

The row order of a `t2va` / `fl2va` request is

```
[ text (L) | keyframe conditions (C) | target audio (A) | target video (V) ]
```

and the geometry helpers of this module exist to place a row in that sequence and to give it its `(t, h, w)` rotary
coordinate. The coordinates are built in float64 because video and audio share one 40-units-per-second rotary clock —
video advances `5/3` rotary units per pixel frame at 24 fps, audio advances one unit per latent at 40 latents/s — and
that shared clock *is* the audio/video alignment.

The released checkpoint is guidance-distilled: guidance is baked into the weights, so the default `guidance_scale`
of `1` runs exactly one forward pass per step with no CFG. A `guidance_scale` above `1` enables classifier-free
guidance with a `negative_prompt`, running two forward passes per step.
"""

import contextlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from PIL import Image, ImageOps

from ..models import (AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio,
                      MiniMaxH3Transformer3DModel)
from ..utils import MiniMaxH3Scheduler

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Per-row modality tags. They index the transformer's AdaLN table, so the values are a checkpoint contract.
MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2

# MiniMax-H3 generates at a fixed 24 fps and was released for a 768 pixel short edge only, with a soft area cap of
# 768x1344 and both axes rounded to a multiple of 32.
MINIMAX_H3_FPS = 24
MINIMAX_H3_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4
MINIMAX_H3_MAX_ASPECT_RATIO = 4
MINIMAX_H3_MIN_DURATION = 5.0
MINIMAX_H3_MAX_DURATION = 15.0

# The video VAE encodes 17 pixel frames per chunk and drops the 3 trailing latent frames of every chunk, so
# `17 * n + 5` pixel frames map to `5 * n + 2` latent frames.
MINIMAX_H3_FRAMES_PER_CHUNK = 17
MINIMAX_H3_LATENTS_PER_CHUNK = 5

# The pixel convention of the video VAE: ImageNet-normalized RGB over a `[0, 1]` base range.
MINIMAX_H3_PIXEL_MEAN = (0.485, 0.456, 0.406)
MINIMAX_H3_PIXEL_STD = (0.229, 0.224, 0.225)

# MiniMax-H3 conditions on the *unnormalized* hidden state its Qwen3-VL conditioner produces after the 50th of its 64
# decoder layers, i.e. `hidden_states[50]` (`hidden_states[0]` being the embedding output).
MINIMAX_H3_TEXT_ENCODER_LAYER = 50

# The audio VAE hops 800 samples at 32 kHz, i.e. 40 latents per second. Stereo is carried as two channel-major
# blocks of audio rows (and as two batch items at the audio VAE boundary, which is mono).
MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40
MINIMAX_H3_AUDIO_CHANNELS = 2

# Conditioning rows are not fully clean: the released model noises keyframe latents to `t = 0.999` and runs them at
# that timestep for every denoising step.
MINIMAX_H3_KEYFRAME_NOISE_AUG = 0.999

# The seeded posterior sample of the keyframe VAE encode. Fixed at 42 independently of the request seed.
MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42

# Rotary-time constants. One latent frame spans `5/3 * frames_per_latent` rotary units, where the pattern
# `(1, 4, 4, 4, 4)` mirrors the VAE's 17-pixel-frames-to-5-latent-frames grouping; the spatial axes are normalized
# by the square root of the latent area and scaled by 32.
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


@contextlib.contextmanager
def _offload_scope(module: torch.nn.Module):
    r"""
    Fire the top-level CPU-offload hook of `module` around a call that bypasses its `forward`.

    `enable_model_cpu_offload` registers an `AlignDevicesHook` on the top-level module and wraps its `forward` alone,
    so calling a method such as `decode` — or a submodule such as `text_encoder.model` — never fires it: the module
    would be used while still on the CPU, or, once onloaded, stay on the GPU forever and starve the next component.
    Fire the hook by hand around those calls instead, so `pre_forward` onloads the module and `post_forward` offloads
    it again, symmetrically.

    Modes that hook leaves instead (`sequential_cpu_offload`) or that keep no top-level `_hf_hook`
    (`model_full_load`, `model_group_offload`) are unaffected: the scope is a no-op there.
    """
    hook = getattr(module, "_hf_hook", None)
    if hook is None or not hasattr(hook, "pre_forward"):
        yield
        return
    hook.pre_forward(module)
    try:
        yield
    finally:
        # `ModelHook.post_forward` takes the forward output it would hand back; the scoped call bypassed `forward`,
        # so there is none.
        hook.post_forward(module, None)


@dataclass
class MiniMaxH3PackedSequence:
    r"""
    The structural description of one packed MiniMax-H3 sequence.

    Attributes:
        sequence_length (`int`):
            Total number of rows, `L + C + A + V`.
        position_ids (`torch.Tensor` of shape `(sequence_length, 3)`, float64):
            The `(t, h, w)` rotary coordinate of every row.
        token_tags (`torch.Tensor` of shape `(sequence_length,)`):
            The modality tag of every row.
        video_indices (`torch.Tensor`):
            Sequence positions of the video rows: the keyframe conditioning rows first, then the target rows.
        audio_indices (`torch.Tensor`):
            Sequence positions of the audio rows.
        text_indices (`torch.Tensor`):
            Sequence positions of the text rows.
        num_condition_video_rows (`int`):
            How many leading entries of `video_indices` are conditioning rows rather than generated rows.
        num_condition_audio_rows (`int`):
            How many leading entries of `audio_indices` are reference rows rather than generated rows.
    """

    sequence_length: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int


@dataclass
class MiniMaxH3PipelineOutput(BaseOutput):
    r"""
    Output of [`MiniMaxH3Pipeline`].

    Args:
        videos (`torch.Tensor`, `np.ndarray` or `list[list[PIL.Image.Image]]`):
            The generated video, at 24 fps.
        audio (`torch.Tensor`):
            The generated soundtrack, of shape `(batch_size, 2, num_samples)`.
        sampling_rate (`int`):
            Sample rate of the soundtrack in Hz.
    """

    videos: torch.Tensor
    audio: torch.Tensor
    sampling_rate: int


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> Tuple[int, int]:
    r"""
    Resolve a display aspect ratio into a MiniMax-H3 canvas.

    The short edge starts at 768, the area is capped at `768 * 1344` and both axes are then rounded to the nearest
    multiple of 32 — so the final area may end up slightly above the pre-rounding budget. Only the ratio of the two
    arguments matters; pass either the aspect ratio (`16, 9`) or the source dimensions of a keyframe.

    Args:
        aspect_width (`float`): Width of the target ratio.
        aspect_height (`float`): Height of the target ratio.

    Returns:
        `tuple[int, int]`: the `(height, width)` of the canvas.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")

    ratio = aspect_width / aspect_height
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got {aspect_width}:{aspect_height} ({ratio:g})."
        )

    if ratio >= 1.0:
        width, height = MINIMAX_H3_SHORT_EDGE * ratio, float(MINIMAX_H3_SHORT_EDGE)
    else:
        width, height = float(MINIMAX_H3_SHORT_EDGE), MINIMAX_H3_SHORT_EDGE / ratio

    area = width * height
    if area > MINIMAX_H3_MAX_PIXELS:
        scale = (MINIMAX_H3_MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale

    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return max(multiple, round(height / multiple) * multiple), max(multiple, round(width / multiple) * multiple)


def align_num_frames(num_frames: int) -> int:
    r"""
    Snap a frame count up to the next `17 * n + 5` the video VAE can encode.

    Args:
        num_frames (`int`): The requested number of frames.

    Returns:
        `int`: The aligned number of frames.
    """
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    while num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int) -> int:
    r"""
    The number of latent frames the video VAE produces for a `17 * n + 5` frame count.

    Args:
        num_frames (`int`): An aligned number of frames.

    Returns:
        `int`: The number of latent frames, `5 * n + 2`.
    """
    if num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        raise ValueError(f"`num_frames` must be of the form 17 * n + 5, got {num_frames}.")
    return (
        num_frames - MINIMAX_H3_LATENTS_PER_CHUNK
    ) // MINIMAX_H3_FRAMES_PER_CHUNK * MINIMAX_H3_LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    r"""
    The number of audio latents that covers a video of `num_frames` frames at 24 fps.

    Args:
        num_frames (`int`): The number of video frames.

    Returns:
        `int`: The number of audio latents, rounded at the 40 Hz latent grid.
    """
    return int(round(num_frames / MINIMAX_H3_FPS * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND))


def prepare_keyframe_image(image: Image.Image, height: int, width: int, stretch: bool) -> Image.Image:
    r"""
    Put a keyframe onto the target canvas.

    The first keyframe of a request is the geometry anchor and is *stretched* onto the canvas, while a second
    keyframe follows that canvas and is cover-cropped (aspect-preserving max-scale LANCZOS resize plus a centre
    crop). An image that already is the canvas is returned untouched, without a resampling pass.

    Args:
        image (`PIL.Image.Image`): The keyframe, in RGB and already EXIF-transposed.
        height (`int`): Canvas height.
        width (`int`): Canvas width.
        stretch (`bool`): Whether to stretch (geometry anchor) instead of cover-cropping (follower).

    Returns:
        `PIL.Image.Image`: The prepared keyframe.
    """
    if image.size == (width, height):
        return image
    if stretch:
        return image.resize((width, height), Image.Resampling.LANCZOS)

    scale = max(width / image.size[0], height / image.size[1])
    resized_size = (max(width, round(image.size[0] * scale)), max(height, round(image.size[1] * scale)))
    left = max(0, (resized_size[0] - width) // 2)
    top = max(0, (resized_size[1] - height) // 2)
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    return resized.crop((left, top, left + width, top + height))


def patchify_video_latents(latents: torch.Tensor, patch_size: Tuple[int, int, int]) -> torch.Tensor:
    r"""
    Pack video latents into transformer rows.

    Args:
        latents (`torch.Tensor` of shape `(batch_size, channels, num_frames, height, width)`):
            The latents to pack.
        patch_size (`tuple[int, int, int]`): The `(t, h, w)` patch.

    Returns:
        `torch.Tensor` of shape `(batch_size * num_patches, channels * prod(patch_size))`: The packed rows, ordered
        frame-major then row-major.
    """
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}.")

    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video_tokens(
    rows: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int,
    patch_size: Tuple[int, int, int],
) -> torch.Tensor:
    r"""
    Unpack transformer rows back into video latents. The inverse of [`patchify_video_latents`].

    Args:
        rows (`torch.Tensor` of shape `(num_patches, channels * prod(patch_size))`): The packed rows.
        num_latent_frames (`int`): Number of latent frames.
        latent_height (`int`): Latent height.
        latent_width (`int`): Latent width.
        channels (`int`): Number of latent channels.
        patch_size (`tuple[int, int, int]`): The `(t, h, w)` patch.

    Returns:
        `torch.Tensor` of shape `(batch_size, channels, num_latent_frames, latent_height, latent_width)`.
    """
    patch_t, patch_h, patch_w = patch_size
    rows = rows.reshape(
        -1,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(-1, channels, num_latent_frames, latent_height, latent_width).contiguous()


def unpack_audio_tokens(rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
    r"""
    Unpack the channel-major audio rows into audio VAE latents.

    Args:
        rows (`torch.Tensor` of shape `(num_audio_latents * 2, latent_channels)`): The packed audio rows.
        num_audio_latents (`int`): Number of audio latents per channel.

    Returns:
        `torch.Tensor` of shape `(2, latent_channels, num_audio_latents)`: One batch item per stereo channel, which
        is what the mono audio VAE consumes.
    """
    rows = rows.reshape(MINIMAX_H3_AUDIO_CHANNELS, num_audio_latents, rows.shape[-1])
    return rows.permute(0, 2, 1).contiguous()


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    r"""
    One aspect-normalized spatial rotary axis: `dim // patch` coordinates centred on the unit interval, scaled up by
    32. The right endpoint is excluded, so a square canvas spans `[0, 32)`.
    """
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # Built with numpy: `np.linspace(..., endpoint=False)` is `start + arange(num) * (stop - start) / num`, which is
    # not what `torch.linspace` computes, and the float64 grid has to be reproduced exactly.
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    r"""The rotary time of every latent frame, starting at `origin`. Spacing is non-uniform: `5/3 * (1, 4, 4, 4, 4)`."""
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _temporal_position_span(num_latent_frames: int) -> float:
    r"""
    The rotary time spanned by `num_latent_frames` latent frames.

    Summed by numpy (pairwise summation) rather than sequentially: the reference computes the keyframe anchor this
    way and the two summation orders differ in the last ulp from 16 latent frames onwards.
    """
    spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
    for index in range(len(_ROPE_FRAMES_PER_LATENT)):
        spans[index :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[index]
    return float(spans.sum())


def build_packed_sequence(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: Tuple[int, int, int],
    keyframe_anchors: Tuple[str, ...] = (),
) -> MiniMaxH3PackedSequence:
    r"""
    Build the `[text | keyframe conditions | target audio | target video]` layout used by the `t2va` and `fl2va`
    tasks.

    Args:
        text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
            The modality tag of every text row. Text is tagged `1`, except for the rows of a keyframe's vision block,
            which MiniMax-H3 tags `0` (video).
        num_latent_frames (`int`): Number of target latent frames.
        latent_height (`int`): Target latent height.
        latent_width (`int`): Target latent width.
        num_audio_latents (`int`): Number of target audio latents per channel.
        patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.
        keyframe_anchors (`tuple[str, ...]`):
            One entry per keyframe conditioning block, in packed order: `"first"` anchors the block at the first
            latent frame, `"last"` at the last one.

    Returns:
        [`MiniMaxH3PackedSequence`]
    """
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = text_token_tags.shape[0]
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    # 1. The (t, h, w) grid. Text rows sit on the time axis at their row index, and the media rows continue the time
    # axis from there, so text length shifts the whole media clock.
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    frame_grid = torch.stack([grid.reshape(-1) for grid in torch.meshgrid(height_grid, width_grid, indexing="ij")], -1)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            anchor_time = float(num_text_tokens) + _temporal_position_span(num_latent_frames) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    # Audio rows are channel-major and share the video's rotary clock: one unit per latent at 40 latents/s equals
    # 24 fps * 5/3. They carry no height coordinate and are pinned to the two extremes of the width grid.
    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )

    video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    # 2. Row indices and modality tags.
    video_indices = torch.cat([torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)])
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_rows,
        num_condition_audio_rows=0,
    )


def build_row_timesteps(
    layout: MiniMaxH3PackedSequence,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Assign a timestep to every row of the packed sequence and reduce it to the transformer's `(timestep,
    timestep_indices)` pair.

    One forward serves rows at different noise levels: the generated video and audio rows step down their own
    schedules while the conditioning rows stay pinned at their noise-augmentation level. Text rows never reach an
    output head and inherit the video timestep.

    Args:
        layout ([`MiniMaxH3PackedSequence`]): The packed layout.
        video_timestep (`float`): Timestep of the generated video rows.
        audio_timestep (`float`): Timestep of the generated audio rows.
        condition_video_timestep (`float`): Timestep of the video conditioning rows.
        condition_audio_timestep (`float`): Timestep of the audio reference rows.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: the distinct timesteps, sorted, and the index of every row into them.
    """
    row_timesteps = torch.full((layout.sequence_length,), video_timestep, dtype=torch.float32)
    row_timesteps[layout.video_indices[: layout.num_condition_video_rows]] = condition_video_timestep
    row_timesteps[layout.audio_indices[layout.num_condition_audio_rows :]] = audio_timestep
    row_timesteps[layout.audio_indices[: layout.num_condition_audio_rows]] = condition_audio_timestep
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


def keyframe_condition_noise(
    condition_latent_shapes: Tuple[Tuple[int, int, int], ...],
    patch_size: Tuple[int, int, int],
    latent_channels: int,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Draw the noise that the keyframe conditioning rows are mixed with.

    One draw per condition, in packed order, off the request's generator. The conditioning rows are prepared before
    the target rows, so these are the *first* draws of a request, ahead of the video and audio noise of
    [`~MiniMaxH3Pipeline.prepare_latents`] — the order is part of what a generator reproduces.

    Args:
        condition_latent_shapes (`tuple[tuple[int, int, int], ...]`):
            The `(num_latent_frames, latent_height, latent_width)` of every condition, in packed order.
        patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.
        latent_channels (`int`): Number of video latent channels.
        generator (`torch.Generator`, *optional*): The generator of the request.
        device (`torch.device`, *optional*): The device the noise is drawn on.
        dtype (`torch.dtype`, defaults to `torch.float32`): The dtype of the noise.

    Returns:
        `torch.Tensor` of shape `(num_condition_rows, latent_channels * prod(patch_size))`: the noise rows,
        concatenated in packed order.
    """
    rows = []
    for num_latent_frames, latent_height, latent_width in condition_latent_shapes:
        noise = randn_tensor(
            (1, latent_channels, num_latent_frames, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        rows.append(patchify_video_latents(noise, patch_size))
    return torch.cat(rows)



class MiniMaxH3Pipeline(DiffusionPipeline):
    r"""
    Pipeline for joint video + audio generation with MiniMax-H3, covering the `t2va` (text only) and `fl2va` (first
    and/or last keyframe) tasks.

    MiniMax-H3 denoises **one packed sequence** that holds the text conditioning, the keyframe conditioning latents,
    the audio latents and the video latents at once, which is why the pipeline passes a row layout around rather than
    per-modality tensors, and why it carries two schedulers (`shift = 12.0` for video, `shift = 3.0` for audio) that
    are stepped inside a single transformer call.

    Args:
        vae ([`AutoencoderKLMiniMaxH3`]):
            The video autoencoder. Its latents are normalized with `latents_mean` / `latents_std`.
        audio_vae ([`AutoencoderKLMiniMaxH3Audio`]):
            The waveform autoencoder. It is mono: stereo is carried as two batch items.
        text_encoder ([`Qwen3VLForConditionalGeneration`]):
            The conditioner. MiniMax-H3 reads the *unnormalized* hidden state after its 50th decoder layer and never
            uses the language-model head.
        tokenizer ([`Qwen2TokenizerFast`]):
            Tokenizer of the conditioner.
        processor ([`Qwen3VLProcessor`]):
            Processor of the conditioner, used for the vision blocks of the keyframes.
        transformer ([`MiniMaxH3Transformer3DModel`]):
            The denoiser of the packed sequence.
        scheduler ([`MiniMaxH3Scheduler`]):
            Schedule of the video latents (`shift = 12.0` in the released checkpoints).
        audio_scheduler ([`MiniMaxH3Scheduler`]):
            Schedule of the audio latents (`shift = 3.0` in the released checkpoints).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae->audio_vae"
    _callback_tensor_inputs = ["latents", "audio_latents", "prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKLMiniMaxH3,
        audio_vae: AutoencoderKLMiniMaxH3Audio,
        text_encoder,
        tokenizer,
        processor,
        transformer: MiniMaxH3Transformer3DModel,
        scheduler: MiniMaxH3Scheduler,
        audio_scheduler: MiniMaxH3Scheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
            audio_scheduler=audio_scheduler,
        )
        # The video VAE decodes into ImageNet-normalized RGB over a [0, 1] base range, which this pipeline reverts
        # itself, so the processor must not denormalize a second time.
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_spatial_compression_ratio, do_normalize=False
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""Load a [`MiniMaxH3Pipeline`] from any of the three MiniMax-H3 on-disk layouts.

        The layout is auto-detected, so the same entry point covers every release:

        * an **original** MiniMax-H3 partition (`_minimax_h3.sigma_shift_scales` in `model_index.json` or fused
          `qkv_proj` keys in the transformer shards) is stream-converted through
          [`~MiniMaxH3Pipeline.from_pretrained_original`] with no intermediate diffusers copy;
        * a **standard diffusers** folder that ships a root `model_index.json` is delegated to
          `DiffusionPipeline.from_pretrained`, which wires the components itself;
        * a **diffusers-format snapshot without a root `model_index.json`** -- e.g. the ModelScope
          `MiniMax/MiniMax-H3` download, which carries a `modular_model_index.json` for the modular pipeline instead --
          is assembled here from its component subfolders with the `videox_fun` class registry, so it loads without
          running `convert_minimax_h3_to_diffusers.py` first.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                A local folder in any of the layouts above, or a repo id for `DiffusionPipeline.from_pretrained`.
            torch_dtype (`torch.dtype`, *optional*):
                Dtype of the transformer and the text encoder; `None` keeps the released bfloat16. The two VAEs always
                stay float32 as released, regardless of this argument.
        """
        import os

        from ..models.minimax_h3_conversion import is_raw_minimax_h3_format

        path = pretrained_model_name_or_path
        torch_dtype = kwargs.pop("torch_dtype", kwargs.pop("dtype", None))

        # Original MiniMax-H3 partition: stream-convert without writing a diffusers copy on disk.
        if is_raw_minimax_h3_format(path):
            return cls.from_pretrained_original(path, torch_dtype=torch_dtype)

        # Non-local (repo id) or a folder that already carries a `model_index.json`: let diffusers wire it. A snapshot
        # without `model_index.json` falls through to the subfolder assembly below.
        if not os.path.isdir(path) or os.path.isfile(os.path.join(os.fspath(path), "model_index.json")):
            return super().from_pretrained(path, torch_dtype=torch_dtype, **kwargs)

        from ..models import (Qwen2TokenizerFast,
                              Qwen3VLForConditionalGeneration,
                              Qwen3VLProcessor)

        def _subfolder(name):
            folder = os.path.join(os.fspath(path), name)
            if not os.path.isdir(folder):
                raise FileNotFoundError(
                    f"`{name}` subfolder not found under {path}; expected a diffusers-format MiniMax-H3 snapshot "
                    f"with `transformer/`, `vae/`, `audio_vae/`, `text_encoder/`, `tokenizer/`, `processor/`, "
                    f"`scheduler/` and `audio_scheduler/`."
                )
            return folder

        transformer = MiniMaxH3Transformer3DModel.from_pretrained(
            _subfolder("transformer"), torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        # The two VAEs stay float32 as released (the decode recipe is float16 autocast over float32 weights), so they
        # are loaded without `torch_dtype`; the mixed-precision loader mixin restores the pinned fp32 modules anyway.
        vae = AutoencoderKLMiniMaxH3.from_pretrained(_subfolder("vae"))
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(_subfolder("audio_vae"))
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            _subfolder("text_encoder"), low_cpu_mem_usage=True, torch_dtype=torch_dtype
        ).eval()
        tokenizer = Qwen2TokenizerFast.from_pretrained(_subfolder("tokenizer"))
        processor = Qwen3VLProcessor.from_pretrained(_subfolder("processor"))
        scheduler = MiniMaxH3Scheduler.from_pretrained(_subfolder("scheduler"))
        audio_scheduler = MiniMaxH3Scheduler.from_pretrained(_subfolder("audio_scheduler"))

        return cls(
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
            audio_scheduler=audio_scheduler,
        )

    @classmethod
    def from_pretrained_original(cls, checkpoint_path, torch_dtype=None):
        r"""
        Assemble a full [`MiniMaxH3Pipeline`] from an *original* MiniMax-H3 checkpoint partition (e.g.
        `MiniMax-H3/FL2VA`) **without converting it on disk first**.

        The transformer and both VAEs are built empty on the meta device and assembled by streaming the original
        shards through the shared key / tensor mapping of `minimax_h3_conversion`, so peak memory stays the models
        plus one shard and no intermediate diffusers copy is written. The conditioner (`text_encoder` / `tokenizer` /
        `processor`) is already shipped in the HuggingFace layout and is loaded as is, and the two schedules are
        built from the `_minimax_h3.sigma_shift_scales` block of the checkpoint's `model_index.json`
        (`12.0` video, `3.0` audio as released).

        Args:
            checkpoint_path (`str` or `os.PathLike`):
                An original MiniMax-H3 partition folder, holding `model_index.json`, `transformer/`, `video_vae/`,
                `audio_vae/` and the conditioner folders.
            torch_dtype (`torch.dtype`, *optional*):
                The dtype of the transformer and the conditioner; `None` keeps the released bfloat16. The two VAEs
                always stay float32, as released.
        """
        import os

        from ..models import (Qwen2TokenizerFast,
                              Qwen3VLForConditionalGeneration,
                              Qwen3VLProcessor)
        from ..models.minimax_h3_conversion import read_original_sigma_shifts

        shifts = read_original_sigma_shifts(checkpoint_path)

        transformer = MiniMaxH3Transformer3DModel.from_pretrained_original(
            checkpoint_path, torch_dtype=torch_dtype
        )
        vae = AutoencoderKLMiniMaxH3.from_pretrained_original(checkpoint_path)
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained_original(checkpoint_path)

        tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(checkpoint_path, "tokenizer"))
        processor = Qwen3VLProcessor.from_pretrained(os.path.join(checkpoint_path, "processor"))
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(checkpoint_path, "text_encoder"),
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
        ).eval()

        scheduler = MiniMaxH3Scheduler(shift=float(shifts["video"]))
        audio_scheduler = MiniMaxH3Scheduler(shift=float(shifts["audio"]))

        return cls(
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
            audio_scheduler=audio_scheduler,
        )

    @property
    def vae_spatial_compression_ratio(self) -> int:
        if getattr(self, "vae", None) is not None:
            return self.vae.spatial_compression_ratio
        return 16

    @property
    def vae_latent_channels(self) -> int:
        if getattr(self, "vae", None) is not None:
            return self.vae.config.latent_channels
        return 24

    @property
    def audio_sampling_rate(self) -> int:
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.config.sampling_rate
        return 32000

    @property
    def audio_latent_channels(self) -> int:
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.config.latent_channels
        return 32

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        if getattr(self, "transformer", None) is not None:
            return tuple(self.transformer.config.patch_size)
        return (1, 2, 2)

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    def check_inputs(self, prompt, height, width, num_frames, num_inference_steps):
        if not isinstance(prompt, str):
            raise ValueError(
                f"MiniMax-H3 packs one request into one sequence, so `prompt` must be a single string, got "
                f"{type(prompt)}."
            )
        if num_inference_steps < 1:
            raise ValueError(
                "`num_inference_steps` is a number of denoising steps, so it must be at least 1, got "
                f"{num_inference_steps}."
            )
        if (height is None) != (width is None):
            raise ValueError("`height` and `width` have to be passed together, or neither of them.")
        if height is not None and (height % MINIMAX_H3_CANVAS_MULTIPLE or width % MINIMAX_H3_CANVAS_MULTIPLE):
            raise ValueError(
                f"`height` and `width` must be multiples of {MINIMAX_H3_CANVAS_MULTIPLE}, got {height}x{width}."
            )
        # The duration the request generates is the one of the *aligned* frame count, so that is what the ceiling has
        # to hold for: 346 frames would otherwise pass the check and then be rounded up to 362, i.e. 15.083 seconds.
        aligned_num_frames = align_num_frames(num_frames)
        duration = aligned_num_frames / MINIMAX_H3_FPS
        if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
            raise ValueError(
                f"MiniMax-H3 generates between {MINIMAX_H3_MIN_DURATION} and {MINIMAX_H3_MAX_DURATION} seconds at "
                f"{MINIMAX_H3_FPS} fps, so `num_frames`, rounded up to the next `17 * n + 5` the video VAE can "
                f"encode, must be between {int(MINIMAX_H3_MIN_DURATION * MINIMAX_H3_FPS)} and "
                f"{int(MINIMAX_H3_MAX_DURATION * MINIMAX_H3_FPS)}, got {num_frames} (rounded up to "
                f"{aligned_num_frames})."
            )

    def _mm_token_type_ids(self, token_ids: List[int]) -> List[int]:
        r"""
        The per-token modality run Qwen3-VL lays its 3D rotary positions out over: `0` text, `1` image, `2` video.
        Transformers versions that do not take `mm_token_type_ids` derive the same runs from the vision pad ids in
        `input_ids` themselves, so this is only handed over when the conditioner accepts it.
        """
        image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_pad_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        return [1 if token == image_pad_id else 2 if token == video_pad_id else 0 for token in token_ids]

    def encode_prompt(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Build MiniMax-H3's presentation of a request and encode it.

        The presentation is the verbatim prompt for `t2va`. Every keyframe prepends a `"<Picture i>: "` label and a
        vision block (`<|vision_start|>`, one `<|image_pad|>` per vision patch, `<|vision_end|>`) — no chat template
        and no special tokens. The rows of a vision block are tagged as *video* rather than text, which is what the
        transformer's AdaLN modulation keys off.

        Args:
            prompt (`str`): The prompt to encode.
            images (`list[PIL.Image.Image]`, *optional*):
                The keyframes, already prepared onto the target canvas, in packed order.
            device (`torch.device`, *optional*): The device to run the conditioner on.
            dtype (`torch.dtype`, *optional*): The dtype of the returned embeddings.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the `(1, num_text_tokens, 5120)` hidden states and the
            `(num_text_tokens,)` per-row modality tags.
        """
        device = device or self._execution_device
        dtype = dtype or self.transformer.dtype

        num_layers = self.text_encoder.config.text_config.num_hidden_layers
        if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER:
            raise ValueError(
                f"MiniMax-H3 conditions on `hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}]` of its Qwen3-VL "
                f"conditioner, which needs more than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers, but "
                f"`text_encoder` has {num_layers}. The last hidden state of a stack truncated to exactly "
                f"{MINIMAX_H3_TEXT_ENCODER_LAYER} layers is post-norm and is not the conditioning MiniMax-H3 expects."
            )

        pixel_values, image_grid_thw = None, None
        token_ids, token_tags = [], []
        if images:
            vision = self.processor.image_processor(images=images, return_tensors="pt")
            pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
            merge_size = self.processor.image_processor.merge_size**2
            for index in range(len(images)):
                num_image_tokens = int(image_grid_thw[index].prod()) // merge_size
                label_ids = self.tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                vision_ids = (
                    [self.tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                    + [self.tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                    + [self.tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                )
                token_ids += label_ids + vision_ids
                token_tags += [MINIMAX_H3_TEXT_TAG] * len(label_ids) + [MINIMAX_H3_VIDEO_TAG] * len(vision_ids)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)
        if not token_ids:
            # An empty prompt (e.g. the default empty negative prompt under CFG) tokenizes to zero tokens, and
            # Qwen3-VL's `get_rope_index` cannot reduce over a zero-length sequence dimension; a single
            # whitespace token stands in for the dropped text.
            token_ids = self.tokenizer(" ", add_special_tokens=False)["input_ids"]
            token_tags = [MINIMAX_H3_TEXT_TAG] * len(token_ids)

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        encoder_kwargs = dict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pixel_values=None if pixel_values is None else pixel_values.to(device, self.text_encoder.dtype),
            image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(device),
            use_cache=False,
            output_hidden_states=True,
        )
        # `text_encoder.model` may be an FSDP wrapper whose own `forward` is `(*args, **kwargs)`; follow its module
        # attribute down to the real model before inspecting the signature.
        model_module = self.text_encoder.model
        inner_forward = getattr(getattr(model_module, "module", model_module), "forward", model_module.forward)
        if "mm_token_type_ids" in inspect.signature(inner_forward).parameters:
            encoder_kwargs["mm_token_type_ids"] = torch.tensor(
                [self._mm_token_type_ids(token_ids)], dtype=torch.long, device=device
            )
        # `text_encoder.model` is a submodule, and a CPU-offload hook wraps the *top-level* module's `forward` alone,
        # so calling the submodule directly would leave the conditioner on the CPU. Fire the hook by hand instead of
        # routing through `text_encoder(...)`: MiniMax-H3 reads `hidden_states[50]` and never uses the language-model
        # head, whose vocabulary-wide projection over every token is all the top-level forward would add. The scope
        # also fires `post_forward`, so the conditioner is offloaded again once the embeddings are drawn.
        with _offload_scope(self.text_encoder):
            outputs = self.text_encoder.model(**encoder_kwargs)
            prompt_embeds = outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].to(device=device, dtype=dtype)
        return prompt_embeds, torch.tensor(token_tags, dtype=torch.long)

    def encode_keyframes(self, images: List[Image.Image], device: Optional[torch.device] = None) -> torch.Tensor:
        r"""
        Encode the `fl2va` keyframes into packed conditioning rows.

        The keyframes go through the video VAE's spatial encoder only — they are single frames, so none of its
        17-frame temporal chunking applies — and the posterior is *sampled*, under a generator seeded with 42
        independently of the request seed. The sampled latent is rounded to float16 before being normalized, as in the
        reference implementation; both are part of reproducing the released model's conditioning.

        Args:
            images (`list[PIL.Image.Image]`):
                The keyframes, already prepared onto the target canvas, in packed order.
            device (`torch.device`, *optional*): The device to run the VAE on.

        Returns:
            `torch.Tensor` of shape `(num_condition_rows, latent_channels * prod(patch_size))`: the float32
            conditioning rows.
        """
        device = device or self._execution_device
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)

        rows = []
        # `_encode_clip` is a method call, not the VAE's `forward`, so the top-level CPU-offload hook never fires
        # around it on its own: scope the whole encode, once, instead of per keyframe.
        with _offload_scope(self.vae):
            for image in images:
                pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None, :, None]
                pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
                # `vae.encode` chunks along time for videos; a keyframe is one frame and is encoded by the (tiled)
                # spatial encoder alone, which is what the released model conditions on.
                moments = self.vae._encode_clip(pixels)
                posterior = DiagonalGaussianDistribution(moments)
                latents = posterior.sample(generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
                # The sampled latent is rounded to float16 before it is normalized: ~11 bits of every conditioning
                # latent, so the released model's conditioning cannot be reproduced without it.
                latents = latents.to(torch.float16).float().cpu()
                rows.append(patchify_video_latents((latents - latents_mean) / latents_std, self.patch_size))
        return torch.cat(rows)

    def prepare_latents(
        self,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        audio_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Draw the initial noise of both modalities and pack it into transformer rows.

        A request draws every stream from the one generator it is given, and the order is part of what that generator
        reproduces: the conditioning noise of the keyframes first (one draw per condition, in
        [`keyframe_condition_noise`]), then the video noise here, as a latent tensor that is patchified afterwards,
        then the audio noise, directly in row layout. Passing `latents` or `audio_latents` skips its draw.

        Args:
            num_latent_frames (`int`): Number of video latent frames.
            latent_height (`int`): Latent height.
            latent_width (`int`): Latent width.
            num_audio_latents (`int`): Number of audio latents per channel.
            device (`torch.device`): The device the rows are drawn on.
            generator (`torch.Generator`, *optional*): The generator of the request.
            latents (`torch.Tensor`, *optional*):
                Pre-generated video noise of shape `(1, latent_channels, num_latent_frames, latent_height,
                latent_width)`, used instead of the draw.
            audio_latents (`torch.Tensor`, *optional*):
                Pre-generated audio noise of shape `(2, audio_latent_channels, num_audio_latents)`.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the video rows and the channel-major audio rows.
        """
        if latents is None:
            latents = randn_tensor(
                (1, self.vae_latent_channels, num_latent_frames, latent_height, latent_width),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        video_rows = patchify_video_latents(latents.to(torch.float32), self.patch_size)

        if audio_latents is None:
            audio_rows = randn_tensor(
                (num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS, self.audio_latent_channels),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            audio_rows = audio_latents.to(torch.float32).permute(0, 2, 1).reshape(-1, self.audio_latent_channels)
        return video_rows.to(device), audio_rows.to(device)

    def decode_latents(
        self,
        latents: torch.Tensor,
        num_condition_video_rows: int,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        output_type: str = "pt",
    ):
        r"""
        Unpack the generated video rows back into latents, denormalize them and decode them into video.

        The spatial tiling of the video VAE covers the canvas exactly, so the decoded frames need no crop back, but
        the decode itself runs under float16 autocast even though the VAE weights are float32, and the VAE produces
        ImageNet-normalized RGB that is reverted here.
        """
        device = self._execution_device
        latents = unpatchify_video_tokens(
            latents[num_condition_video_rows:],
            num_latent_frames,
            latent_height,
            latent_width,
            self.vae_latent_channels,
            self.patch_size,
        )
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        latents = latents * latents_std + latents_mean

        if output_type == "latent":
            return latents

        # `decode` is reached as a method call, so the top-level CPU-offload hook is fired by hand around it.
        with _offload_scope(self.vae), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            video = self.vae.decode(latents, return_dict=False)[0]
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
        video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
        return self.video_processor.postprocess_video(video, output_type=output_type)

    def decode_audio_latents(
        self,
        audio_latents: torch.Tensor,
        num_condition_audio_rows: int,
        num_audio_latents: int,
        output_type: str = "pt",
    ) -> torch.Tensor:
        r"""
        Unpack the generated audio rows back into latents, denormalize them and decode them into a stereo waveform.
        The audio VAE is mono and takes the two stereo channels as two batch items.
        """
        device = self._execution_device
        audio_latents = unpack_audio_tokens(audio_latents[num_condition_audio_rows:], num_audio_latents)
        audio_latents_mean = torch.tensor(self.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
        audio_latents_std = torch.tensor(self.audio_vae.config.latents_std, device=device).view(1, -1, 1)
        audio_latents = audio_latents * audio_latents_std + audio_latents_mean

        if output_type == "latent":
            return audio_latents

        # `decode` is reached as a method call, so the top-level CPU-offload hook is fired by hand around it.
        with _offload_scope(self.audio_vae):
            audio = self.audio_vae.decode(audio_latents, return_dict=False)[0]
        return audio.float().permute(1, 0, 2)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str = None,
        image: Optional[Image.Image] = None,
        last_image: Optional[Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 124,
        num_inference_steps: int = 50,
        flow_shift: Optional[float] = None,
        audio_flow_shift: Optional[float] = None,
        guidance_scale: float = 1.0,
        negative_prompt: Optional[str] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        audio_latents: Optional[torch.Tensor] = None,
        output_type: str = "pt",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        r"""
        Generate a video and its soundtrack.

        Args:
            prompt (`str`):
                The prompt to guide generation. MiniMax-H3 packs one request into one sequence, so a batch of prompts
                is not a thing.
            image (`PIL.Image.Image`, *optional*):
                Keyframe the video starts from. It is *stretched* onto the target canvas, which by default is derived
                from its own aspect ratio.
            last_image (`PIL.Image.Image`, *optional*):
                Keyframe the video ends on. Can be passed on its own to generate *up to* a frame. Combined with
                `image` it is the follower of the two and is cover-cropped onto the canvas.
            height (`int`, *optional*):
                Height of the generated video in pixels, a multiple of 32. Defaults to MiniMax-H3's own canvas for the
                aspect ratio of the first keyframe, or 16:9 without one.
            width (`int`, *optional*):
                Width of the generated video in pixels, a multiple of 32.
            num_frames (`int`, defaults to `124`):
                Number of frames to generate, at the fixed 24 fps. Snapped up to the next `17 * n + 5` the video VAE
                can decode; the resulting duration must stay between 5 and 15 seconds.
            num_inference_steps (`int`, defaults to `50`):
                Number of denoising steps, i.e. of model evaluations. The sigma grid it is built from holds one more
                point than that, the terminal `0`.
            flow_shift (`float`, *optional*):
                Overrides the video schedule's exponential shift (`12.0` in the released checkpoints).
            audio_flow_shift (`float`, *optional*):
                Overrides the audio schedule's exponential shift (`3.0` in the released checkpoints).
            guidance_scale (`float`, defaults to `1.0`):
                Classifier-free guidance scale. The released checkpoint is guidance-distilled, so the default `1.0`
                disables CFG and runs one forward pass per step. A value above `1.0` enables CFG with
                `negative_prompt`, running two forward passes per step.
            negative_prompt (`str`, *optional*):
                The prompt that guides what to exclude from generation, used when `guidance_scale > 1`. Defaults to an
                empty string when `guidance_scale > 1` and `negative_prompt` is `None`.
            generator (`torch.Generator`, *optional*):
                The generator of the request. A request draws the keyframe conditioning noise first, then the video
                noise, then the audio noise, so two runs from the same generator state return the same video and
                soundtrack.
            latents (`torch.Tensor`, *optional*):
                Pre-generated video noise of shape `(1, 24, num_latent_frames, latent_height, latent_width)`, used
                instead of the draw.
            audio_latents (`torch.Tensor`, *optional*):
                Pre-generated audio noise of shape `(2, 32, num_audio_latents)`.
            output_type (`str`, defaults to `"pt"`):
                Output format: `"pil"`, `"np"`, `"pt"`, or `"latent"` for the raw latents.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`MiniMaxH3PipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that, if specified, may carry a `scale` entry which is applied to the LoRA layers.
            callback_on_step_end (`Callable`, *optional*):
                A function called at the end of every denoising step.
            callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["latents"]`):
                The tensors of the loop the callback is handed.

        Returns:
            [`MiniMaxH3PipelineOutput`] or `tuple`:
                The generated video, the stereo soundtrack of shape `(1, 2, num_samples)` and its sample rate. Muxing
                the two into one file is left to the caller, e.g. with `save_videos_with_audio_grid`.
        """
        self.check_inputs(prompt, height, width, num_frames, num_inference_steps)
        self._attention_kwargs = attention_kwargs
        device = self._execution_device

        # 1. Resolve the plan: the canvas, the frame count the video VAE can decode, the latent geometry every later
        # step keys off, and the keyframes put onto that canvas.
        keyframes = [
            ImageOps.exif_transpose(keyframe).convert("RGB")
            for keyframe in (image, last_image)
            if keyframe is not None
        ]
        keyframe_anchors = tuple(
            anchor for anchor, keyframe in (("first", image), ("last", last_image)) if keyframe is not None
        )
        if height is None:
            height, width = resolve_canvas_size(*(keyframes[0].size if keyframes else (16, 9)))

        aligned_num_frames = align_num_frames(num_frames)
        if aligned_num_frames != num_frames:
            logger.warning(
                f"`num_frames` has to be of the form 17 * n + 5 for the video VAE; rounding {num_frames} up to "
                f"{aligned_num_frames}."
            )
            num_frames = aligned_num_frames

        num_latent_frames = video_latent_num_frames(num_frames)
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        num_audio_latents = audio_latent_num_frames(num_frames)
        keyframes = [
            prepare_keyframe_image(keyframe, height, width, stretch=index == 0)
            for index, keyframe in enumerate(keyframes)
        ]

        # 2. Encode MiniMax-H3's presentation of the request. The released checkpoint is guidance-distilled, so the
        # default guidance_scale of 1 runs one forward pass per step with no CFG; a guidance_scale above 1 enables
        # classifier-free guidance with a negative prompt.
        do_cfg = guidance_scale > 1.0
        prompt_embeds, text_token_tags = self.encode_prompt(
            prompt, keyframes, device=device, dtype=self.transformer.dtype
        )
        if do_cfg:
            negative_prompt = negative_prompt if negative_prompt is not None else ""
            negative_prompt_embeds, negative_text_token_tags = self.encode_prompt(
                negative_prompt, keyframes, device=device, dtype=self.transformer.dtype
            )

        # 3. Encode the keyframes into conditioning rows and noise them to MiniMax-H3's conditioning level. They are
        # the anchors of the whole denoising loop: the loop only ever writes the generated rows.
        condition_latents = None
        if keyframes:
            condition_latents = self.encode_keyframes(keyframes, device=device)
            noise = keyframe_condition_noise(
                ((1, latent_height, latent_width),) * len(keyframes),
                self.patch_size,
                self.vae_latent_channels,
                generator=generator,
                device=device,
            )
            condition_latents = self.scheduler.scale_noise(
                condition_latents.to(device), MINIMAX_H3_KEYFRAME_NOISE_AUG, noise
            )

        # 4. Build the packed layout and its fp64 rotary grid.
        layout = build_packed_sequence(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            self.patch_size,
            keyframe_anchors,
        )
        position_ids = layout.position_ids.to(device)
        token_tags = layout.token_tags.to(device)
        video_indices = layout.video_indices.to(device)
        audio_indices = layout.audio_indices.to(device)
        text_indices = layout.text_indices.to(device)
        num_condition_video_rows = layout.num_condition_video_rows
        num_condition_audio_rows = layout.num_condition_audio_rows

        if do_cfg:
            negative_layout = build_packed_sequence(
                negative_text_token_tags,
                num_latent_frames,
                latent_height,
                latent_width,
                num_audio_latents,
                self.patch_size,
                keyframe_anchors,
            )
            negative_position_ids = negative_layout.position_ids.to(device)
            negative_token_tags = negative_layout.token_tags.to(device)
            negative_video_indices = negative_layout.video_indices.to(device)
            negative_audio_indices = negative_layout.audio_indices.to(device)
            negative_text_indices = negative_layout.text_indices.to(device)

        # 5. Draw the noise of the generated rows and prepend the conditioning rows.
        latents, audio_latents = self.prepare_latents(
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            device,
            generator,
            latents,
            audio_latents,
        )
        if condition_latents is not None:
            latents = torch.cat([condition_latents, latents])

        # 6. Initialize the two schedules and stage the row-to-timestep plan of every step. One forward serves every
        # modality and every noise level at once: the generated rows step down their own schedule while the
        # conditioning rows stay pinned at their noise-augmentation level.
        if flow_shift is not None:
            self.scheduler.set_shift(flow_shift)
        if audio_flow_shift is not None:
            self.audio_scheduler.set_shift(audio_flow_shift)
        # `set_timesteps` counts sigma grid points and the terminal `0` is one of them, so `num_inference_steps + 1`
        # points are what drives exactly `num_inference_steps` model evaluations.
        self.scheduler.set_timesteps(num_inference_steps + 1, device=device)
        self.audio_scheduler.set_timesteps(num_inference_steps + 1, device=device)
        timesteps = self.scheduler.timesteps
        audio_timesteps = self.audio_scheduler.timesteps
        # Both schedules collapse consecutive duplicates after their sigma shift; if the two shifts collapse a
        # different number of points the step loop below would zip schedules of unequal length and silently drop
        # the tail of the longer one, so fail loudly instead.
        if len(timesteps) != len(audio_timesteps):
            raise ValueError(
                f"The video schedule holds {len(timesteps)} steps but the audio schedule holds "
                f"{len(audio_timesteps)} after their sigma shifts collapsed duplicates, and one forward serves "
                "both modalities per step. Pick `flow_shift` / `audio_flow_shift` (or `num_inference_steps`) so "
                "the two schedules stay the same length."
            )

        row_timestep_plan = [
            tuple(
                tensor.to(device)
                for tensor in build_row_timesteps(
                    layout,
                    float(timestep),
                    float(audio_timestep),
                    max(float(timestep), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                    1.0,
                )
            )
            for timestep, audio_timestep in zip(timesteps, audio_timesteps)
        ]
        if do_cfg:
            negative_row_timestep_plan = [
                tuple(
                    tensor.to(device)
                    for tensor in build_row_timesteps(
                        negative_layout,
                        float(timestep),
                        float(audio_timestep),
                        max(float(timestep), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                        1.0,
                    )
                )
                for timestep, audio_timestep in zip(timesteps, audio_timesteps)
            ]

        # 7. Denoise the packed sequence over the two schedules.
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                unique_timesteps, timestep_indices = row_timestep_plan[i]
                noise_pred, audio_noise_pred = self.transformer(
                    hidden_states=latents[None],
                    audio_hidden_states=audio_latents[None],
                    encoder_hidden_states=prompt_embeds,
                    timestep=unique_timesteps,
                    timestep_indices=timestep_indices,
                    token_tags=token_tags,
                    position_ids=position_ids,
                    video_indices=video_indices,
                    audio_indices=audio_indices,
                    text_indices=text_indices,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )

                if do_cfg:
                    neg_unique_timesteps, neg_timestep_indices = negative_row_timestep_plan[i]
                    neg_noise_pred, neg_audio_noise_pred = self.transformer(
                        hidden_states=latents[None],
                        audio_hidden_states=audio_latents[None],
                        encoder_hidden_states=negative_prompt_embeds,
                        timestep=neg_unique_timesteps,
                        timestep_indices=neg_timestep_indices,
                        token_tags=negative_token_tags,
                        position_ids=negative_position_ids,
                        video_indices=negative_video_indices,
                        audio_indices=negative_audio_indices,
                        text_indices=negative_text_indices,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)
                    audio_noise_pred = neg_audio_noise_pred + guidance_scale * (
                        audio_noise_pred - neg_audio_noise_pred
                    )

                # The conditioning rows are re-imposed by construction: only the generated rows are ever written, so
                # the anchors survive the whole loop.
                latents[num_condition_video_rows:] = self.scheduler.step(
                    noise_pred[0, num_condition_video_rows:].float(),
                    t,
                    latents[num_condition_video_rows:],
                    return_dict=False,
                )[0]
                audio_latents[num_condition_audio_rows:] = self.audio_scheduler.step(
                    audio_noise_pred[0, num_condition_audio_rows:].float(),
                    audio_timesteps[i],
                    audio_latents[num_condition_audio_rows:],
                    return_dict=False,
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for tensor_name in callback_on_step_end_tensor_inputs:
                        callback_kwargs[tensor_name] = locals()[tensor_name]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs) or {}
                    latents = callback_outputs.pop("latents", latents)
                    audio_latents = callback_outputs.pop("audio_latents", audio_latents)

                progress_bar.update()

        # 8. Decode both modalities.
        videos = self.decode_latents(
            latents, num_condition_video_rows, num_latent_frames, latent_height, latent_width, output_type
        )
        audio = self.decode_audio_latents(
            audio_latents, num_condition_audio_rows, num_audio_latents, output_type
        )

        self.maybe_free_model_hooks()

        if not return_dict:
            return (videos, audio, self.audio_sampling_rate)
        return MiniMaxH3PipelineOutput(videos=videos, audio=audio, sampling_rate=self.audio_sampling_rate)
