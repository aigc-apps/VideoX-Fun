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
MiniMax-H3 text/keyframe/reference to video + audio pipeline.

MiniMax-H3 generates a video and its soundtrack **jointly**: one transformer denoises a single packed sequence that
holds the text conditioning, the conditioning rows, the target audio rows and the target video rows at once, with
full self-attention over all of it. There is no separate vocoder and no audio post-hoc pass.

The row order of a `t2va` / `fl2va` request is

```
[ text (L) | keyframe conditions (C) | target audio (A) | target video (V) ]
```

and a `ref2va` request interleaves one block per reference — image, video with its soundtrack, or audio clip — in
request order between the text and the generated rows. The order is semantic twice over: it labels the references
in the prompt presentation and it advances the shared audio/video rotary clock, so a different order is a different
request.

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
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse

import numpy as np
import requests
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

# The reference budgets the released checkpoint documents for `ref2va`: per-modality caps and a total cap, and an
# audio reference cannot stand alone — it has to be paired with at least one image or video reference.
MINIMAX_H3_MAX_IMAGE_REFERENCES = 9
MINIMAX_H3_MAX_VIDEO_REFERENCES = 3
MINIMAX_H3_MAX_AUDIO_REFERENCES = 3
MINIMAX_H3_MAX_REFERENCES = 12

# An image reference is encoded at a short edge of its own — 2048 for the released checkpoint, upscaling included
# and with no area cap — unlike video references and the target itself, which share the one canvas rule.
MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048

# The rate the conditioner reads a video reference at: every `MINIMAX_H3_FPS / MINIMAX_H3_VIDEO_SAMPLE_FPS`-th of
# the normalized 24 fps frames, merged afterwards in pairs by the conditioner's temporal patch.
MINIMAX_H3_VIDEO_SAMPLE_FPS = 2.0


@dataclass
class MiniMaxH3Reference:
    r"""
    Base class of the three references a `ref2va` request conditions on: [`MiniMaxH3ImageReference`],
    [`MiniMaxH3VideoReference`] and [`MiniMaxH3AudioReference`].

    References are passed to the pipeline as a list, **in the order the model should read them**: the order labels
    them in the prompt presentation and lays them out on the shared rotary clock, so a different order is a
    different request.

    Every reference holds in-memory media, and the rate that media carries where there is one — MiniMax-H3 resamples
    a reference onto its own 24 fps and onto the audio VAE's sample rate, so a rate lost on the way in is a request
    conditioned at the wrong speed. Each class decodes a file through its `from_file` classmethod, along with the
    rates.
    """


@dataclass
class MiniMaxH3ImageReference(MiniMaxH3Reference):
    r"""
    A subject, style or scene reference: at most 9 per request.

    Attributes:
        image (`PIL.Image.Image`, `np.ndarray` or `torch.Tensor`):
            The reference image: an image, a `(height, width, 3)` array or a `(3, height, width)` tensor, `uint8` or
            floating point over `[0, 1]`. It never binds the generated geometry — it is encoded at a short edge of
            its own, 2048 for the released checkpoint, whatever canvas the request generates at.
    """

    image: Union[Image.Image, np.ndarray, torch.Tensor]

    kind = "image"
    has_audio = False

    @classmethod
    def from_file(cls, media) -> "MiniMaxH3ImageReference":
        r"""
        Load an image file into a [`MiniMaxH3ImageReference`].

        Args:
            media (`str` or `os.PathLike`): Path or URL of the image.
        """
        with _local_media_file(media) as path:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return cls(image=image)


@dataclass
class MiniMaxH3VideoReference(MiniMaxH3Reference):
    r"""
    A motion and camera reference: at most 3 per request, conditioned on together with its own soundtrack.

    Attributes:
        frames (`list[PIL.Image.Image]`, `np.ndarray` or `torch.Tensor`):
            The reference frames: a list of images, a `(num_frames, height, width, 3)` array or a `(num_frames, 3,
            height, width)` tensor, `uint8` or floating point over `[0, 1]`.
        fps (`float`, *optional*, defaults to 24.0):
            The frame rate `frames` carries, which is what places the reference's vision blocks on the conditioner's
            2 fps grid. MiniMax-H3's own clock is 24 fps, so any other rate is resampled onto it by dropping and
            duplicating whole frames — which makes this the field to get right when the frames came from a file.
        audio (`torch.Tensor` of shape `(channels, num_samples)`, *optional*):
            This video's soundtrack, mono or stereo, conditioned on as the reference's own rather than as a reference
            of its own. Left out, the reference conditions on motion alone.
        sample_rate (`int`, *optional*):
            The rate `audio` carries its samples at. Left out, it is the audio VAE's own, which leaves the samples
            untouched; any other rate is resampled onto it.
    """

    frames: Union[List[Image.Image], np.ndarray, torch.Tensor]
    fps: Optional[float] = None
    audio: Optional[torch.Tensor] = None
    sample_rate: Optional[int] = None

    kind = "video"

    def __post_init__(self):
        if self.fps is None:
            self.fps = float(MINIMAX_H3_FPS)

    @property
    def has_audio(self) -> bool:
        r"""Whether this reference contributes audio rows, i.e. whether it carries a soundtrack."""
        return self.audio is not None

    @classmethod
    def from_file(cls, media) -> "MiniMaxH3VideoReference":
        r"""
        Decode a video file into a [`MiniMaxH3VideoReference`], at the resolution, the frame rate and the soundtrack
        it carries.

        The rates land on the reference, which is the point of decoding this way: MiniMax-H3 resamples a reference
        onto its own 24 fps, so a frame rate lost on the way in is a request conditioned at the wrong speed, with
        nothing to raise about it. A container whose metadata is wrong is corrected by overriding `fps` or
        `sample_rate` on the returned reference.

        Needs [PyAV](https://github.com/PyAV-Org/PyAV).

        Args:
            media (`str` or `os.PathLike`): Path or URL of the video.

        Returns:
            [`MiniMaxH3VideoReference`]: the `(num_frames, height, width, 3)` `uint8` frames at the frame rate the
            container reports, carrying its soundtrack and that soundtrack's own sample rate when it has an audio
            stream.
        """
        frames, fps, audio, sample_rate = _decode_video_file(media)
        return cls(frames=frames, fps=fps, audio=audio, sample_rate=sample_rate)


@dataclass
class MiniMaxH3AudioReference(MiniMaxH3Reference):
    r"""
    A voice or music reference: at most 3 per request, and never on its own — an audio reference has to be paired
    with at least one image or video reference. It never reaches the conditioner and is encoded by the audio VAE
    alone.

    Attributes:
        audio (`torch.Tensor` of shape `(channels, num_samples)`):
            The reference waveform, mono or stereo.
        sample_rate (`int`, *optional*):
            The rate `audio` carries its samples at. Left out, it is the audio VAE's own, which leaves the samples
            untouched; any other rate is resampled onto it.
    """

    audio: torch.Tensor
    sample_rate: Optional[int] = None

    kind = "audio"
    has_audio = True

    @classmethod
    def from_file(cls, media) -> "MiniMaxH3AudioReference":
        r"""
        Decode an audio file into a [`MiniMaxH3AudioReference`], at the sample rate it carries.

        Needs [PyAV](https://github.com/PyAV-Org/PyAV).

        Args:
            media (`str` or `os.PathLike`): Path or URL of the audio, or of a video whose soundtrack is taken.

        Returns:
            [`MiniMaxH3AudioReference`]: the `(channels, num_samples)` float32 waveform at the sample rate the
            container reports.
        """
        audio, sample_rate = _decode_audio_file(media)
        return cls(audio=audio, sample_rate=sample_rate)


@contextlib.contextmanager
def _local_media_file(media):
    r"""The reference media as a local file: a URL is downloaded to a temporary file, removed on the way out."""
    path = str(media)
    if not path.startswith(("http://", "https://")):
        if not os.path.isfile(path):
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {path} is not a valid path."
            )
        yield path
        return

    response = requests.get(path, stream=True, timeout=60)
    if response.status_code != 200:
        raise ValueError(f"Failed to download {path}. Status code: {response.status_code}")
    suffix = os.path.splitext(os.path.basename(unquote(urlparse(path).path)))[1]
    download = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        with download as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        yield download.name
    finally:
        os.remove(download.name)


def _import_av():
    r"""PyAV, the soft dependency a media file is decoded with."""
    try:
        import av
    except ImportError as error:
        raise ImportError(
            "Decoding a MiniMax-H3 reference from a file needs PyAV. You can install it with `pip install av`, or "
            "build the reference from decoded media itself: frames and the `fps` they carry for a video, a "
            "`(channels, num_samples)` waveform and its `sample_rate` for audio."
        ) from error
    return av


def _decode_reference_soundtrack(av, container, stream) -> Tuple[torch.Tensor, int]:
    r"""
    An audio stream's samples as a `(channels, num_samples)` float32 waveform, at the rate the container carries
    them.

    Args:
        av (`module`): PyAV.
        container (`av.container.InputContainer`): The open container.
        stream (`av.audio.stream.AudioStream`): The stream to decode.

    Returns:
        `tuple[torch.Tensor, int]`: the waveform and its sample rate.
    """
    sample_rate = int(stream.codec_context.sample_rate)
    # Planar float is a format conversion only: the sample rate and the channel layout stay the container's own,
    # and a mono soundtrack is upmixed later, by the setup step's audio normalization.
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout=stream.layout, rate=sample_rate)
    chunks = []
    for frame in container.decode(stream):
        chunks += [torch.from_numpy(resampled.to_ndarray()) for resampled in resampler.resample(frame)]
    # Whatever the resampler is still holding.
    chunks += [torch.from_numpy(resampled.to_ndarray()) for resampled in resampler.resample(None)]
    return torch.cat(chunks, dim=-1).to(torch.float32), sample_rate


def _stream_rotation(av, container, stream, first_frame) -> float:
    r"""
    The display matrix rotation of a video stream, in degrees, whatever the installed PyAV exposes: PyAV 13+ surfaces
    it on every frame, older releases only carry the legacy `rotate` stream metadata tag (or nothing at all).
    """
    rotation = getattr(first_frame, "rotation", None)
    if rotation is not None:
        return float(rotation)
    side_data = getattr(stream, "side_data", None)
    if side_data is not None:
        display_matrix = getattr(side_data, "get", lambda *_: None)("DISPLAYMATRIX")
        rotation = getattr(display_matrix, "rotation", None)
        if rotation is not None:
            return float(rotation)
    return float(stream.metadata.get("rotate", 0.0))


def _decode_video_file(media) -> Tuple[np.ndarray, float, Optional[torch.Tensor], Optional[int]]:
    r"""
    A video file's frames as `(num_frames, height, width, 3)` `uint8`, the frame rate the container reports, and its
    soundtrack with that soundtrack's sample rate (`None, None` without an audio stream). The machinery behind
    [`MiniMaxH3VideoReference.from_file`].
    """
    av = _import_av()
    with _local_media_file(media) as path, av.open(path) as container:
        stream = container.streams.video[0]
        frames, rotation = [], 0.0
        for frame in container.decode(stream):
            # The display matrix rotation belongs to the stream; read it off the first frame.
            if not frames:
                rotation = _stream_rotation(av, container, stream, frame)
            frames.append(frame.to_ndarray(format="rgb24"))
        frame_rate = float(stream.average_rate or stream.guessed_rate)
        soundtrack = None
        if container.streams.audio:
            # Decoding the frames drained the container, so the soundtrack is read in a second pass over it.
            container.seek(0)
            soundtrack = _decode_reference_soundtrack(av, container, container.streams.audio[0])

    if not frames:
        raise ValueError(f"No video frames to decode in {media}.")
    frames = np.stack(frames)
    # `ffmpeg` displays a frame upright by undoing the counterclockwise rotation the display matrix carries, which
    # is what this reproduces, snapped to the nearest quarter turn.
    turns = round(rotation / 90.0) % 4
    if turns:
        frames = np.ascontiguousarray(np.rot90(frames, k=-turns, axes=(1, 2)))
    waveform, sample_rate = soundtrack if soundtrack is not None else (None, None)
    return frames, frame_rate, waveform, sample_rate


def _decode_audio_file(media) -> Tuple[torch.Tensor, int]:
    r"""
    An audio file's `(channels, num_samples)` float32 waveform, at the sample rate the container reports. The
    machinery behind [`MiniMaxH3AudioReference.from_file`].
    """
    av = _import_av()
    with _local_media_file(media) as path, av.open(path) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream to decode in {media}.")
        waveform, sample_rate = _decode_reference_soundtrack(av, container, container.streams.audio[0])
    return waveform, sample_rate

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
        video_view_tags (`torch.Tensor`, *optional*):
            One view id per entry of `video_indices` — multiview layouts only. `None` for the base layouts.
    """

    sequence_length: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int
    video_view_tags: Optional[torch.Tensor] = None


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


def normalize_reference_video(frames, fps: float, num_frames: int) -> np.ndarray:
    r"""
    Normalize a video reference's frames: any accepted layout, onto `uint8` at 24 fps, truncated to the generated
    frame count, on the canvas its own aspect ratio resolves to.

    The two passes reproduce the reference implementation's `ffmpeg` decode, in the same order: the constant frame
    rate resample first (dropping and duplicating whole frames, as `ffmpeg`'s `fps` filter does), the LANCZOS
    rescale second. Frames handed over at 24 fps and already at the canvas their own aspect ratio resolves to flow
    through untouched, which is the parity-exact route.

    Args:
        frames (`list[PIL.Image.Image]`, `np.ndarray` or `torch.Tensor`):
            The reference frames: a list of images, a `(num_frames, height, width, 3)` array or a `(num_frames, 3,
            height, width)` tensor, `uint8` or floating point over `[0, 1]`.
        fps (`float`): The frame rate `frames` carries.
        num_frames (`int`): The generated frame count the reference is truncated to.

    Returns:
        `np.ndarray` of shape `(num_frames, height, width, 3)`: the normalized `uint8` RGB frames.
    """
    # Any accepted layout onto `uint8` THWC. A `torch.Tensor` is channels-first, as everywhere else, and a
    # `np.ndarray` channels-last; floating point values are read over `[0, 1]`.
    if isinstance(frames, list):
        frames = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
    if isinstance(frames, torch.Tensor):
        frames = frames.movedim(-3, -1).cpu().numpy()
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = (frames * 255.0).round().clip(0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(
            f"A reference video must be `(num_frames, height, width, 3)` RGB frames, got {tuple(frames.shape)}."
        )

    # Onto MiniMax-H3's 24 fps grid: every frame is held until the slot of the next one, and the last one until
    # the slot the stream's end rounds to.
    if fps <= 0:
        raise ValueError(f"A reference video must have a positive frame rate, got {fps}.")
    if fps != MINIMAX_H3_FPS:
        scale = MINIMAX_H3_FPS / fps
        slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
        frames = np.repeat(frames, np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5)), axis=0)

    # Truncated to the generated frame count and put on the canvas of its *own* aspect ratio — the same rule the
    # target canvas follows, unlike an image reference.
    frames = frames[:num_frames]
    height, width = resolve_canvas_size(frames.shape[2], frames.shape[1])
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames]
    )


def normalize_reference_audio(
    waveform: torch.Tensor, sample_rate: int, target_sample_rate: int, max_duration: float
) -> torch.Tensor:
    r"""
    Normalize a reference soundtrack onto the audio VAE's sample rate, as a stereo waveform.

    The reference implementation extracts a soundtrack at a native rate, truncates it there and resamples it once,
    in torch, which this mirrors: the truncation is applied at `sample_rate` and the resampling is a single
    `torchaudio` pass. A mono waveform is upmixed by repeating its channel.

    Args:
        waveform (`torch.Tensor` of shape `(channels, num_samples)`): The soundtrack, mono or stereo.
        sample_rate (`int`): The sample rate `waveform` carries its samples at.
        target_sample_rate (`int`): The audio VAE's sample rate, i.e. what the waveform is resampled to.
        max_duration (`float`): Truncate the reference to this many seconds.

    Returns:
        `torch.Tensor` of shape `(2, num_samples)`: the float32 waveform.
    """
    waveform = torch.as_tensor(waveform)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
        raise ValueError(
            "A reference soundtrack must be a `(channels, num_samples)` mono or stereo waveform, got "
            f"{tuple(waveform.shape)}."
        )
    waveform = waveform.to(torch.float32)[:, : int(max_duration * sample_rate)]
    if waveform.shape[0] != 2:
        waveform = waveform.expand(2, -1).contiguous()
    if sample_rate == target_sample_rate:
        return waveform

    try:
        import torchaudio
    except ImportError as error:
        raise ImportError(
            f"Resampling a MiniMax-H3 reference soundtrack from {sample_rate} Hz to {target_sample_rate} Hz "
            "needs `torchaudio`. Pass a waveform already at the audio VAE's sample rate to do without it."
        ) from error
    return torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)


def normalize_reference_image(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
    r"""
    Normalize an image reference onto the resolution it is encoded at: a short edge of its own — 2048 for the
    released checkpoint, upscaling included and with *no* area cap — unlike video references and the target itself,
    which share the one canvas rule.

    Args:
        image (`PIL.Image.Image`, `np.ndarray` or `torch.Tensor`):
            The reference image: an image, a `(height, width, 3)` array or a `(3, height, width)` tensor, `uint8` or
            floating point over `[0, 1]`.

    Returns:
        `PIL.Image.Image`: the normalized RGB image.
    """
    # Any accepted layout onto a PIL image. A `torch.Tensor` is channels-first, as everywhere else, and a
    # `np.ndarray` channels-last; both carry floating point over `[0, 1]`.
    if isinstance(image, torch.Tensor):
        image = image.movedim(-3, -1).cpu().numpy()
    if isinstance(image, np.ndarray):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"A reference image must be `(height, width, 3)` RGB pixels, got {tuple(image.shape)}.")
        if image.dtype != np.uint8:
            image = (image * 255.0).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
    image = image.convert("RGB")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {image.size}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    scale = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    target_height = max(multiple, round(height * scale / multiple) * multiple)
    target_width = max(multiple, round(width * scale / multiple) * multiple)
    if image.size != (target_width, target_height):
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return image


def check_ref2va_references(references: List[MiniMaxH3Reference]) -> List[MiniMaxH3Reference]:
    r"""
    Validate the references of a `ref2va` request against the budgets of the released checkpoint: per-modality
    caps, a total cap, and an audio reference that cannot stand alone.
    """
    if not references:
        raise ValueError("`ref2va` needs at least one reference; use the `t2va` workflow for text-only requests.")
    for index, entry in enumerate(references):
        if not isinstance(entry, MiniMaxH3Reference):
            raise ValueError(
                f"`references[{index}]` must be a [`MiniMaxH3ImageReference`], [`MiniMaxH3VideoReference`] or "
                f"[`MiniMaxH3AudioReference`], got {type(entry)}. A request that holds paths decodes them first, "
                "with each class's `from_file` classmethod."
            )
    kinds = [entry.kind for entry in references]
    for kind, limit in (
        ("image", MINIMAX_H3_MAX_IMAGE_REFERENCES),
        ("video", MINIMAX_H3_MAX_VIDEO_REFERENCES),
        ("audio", MINIMAX_H3_MAX_AUDIO_REFERENCES),
    ):
        if kinds.count(kind) > limit:
            raise ValueError(f"MiniMax-H3 accepts at most {limit} {kind} references, got {kinds.count(kind)}.")
    if len(kinds) > MINIMAX_H3_MAX_REFERENCES:
        raise ValueError(
            f"MiniMax-H3 accepts at most {MINIMAX_H3_MAX_REFERENCES} references in total, got {len(kinds)}."
        )
    if set(kinds) == {"audio"}:
        raise ValueError(
            "An audio reference has to be paired with at least one image or video reference and cannot be used "
            "on its own."
        )
    return references


def normalize_ref2va_references(
    references: List[MiniMaxH3Reference], num_frames: int, audio_sampling_rate: int
) -> List[MiniMaxH3Reference]:
    r"""
    Normalize the references of a `ref2va` request onto MiniMax-H3's own rates and resolutions, in packed order:
    an image resized to its own 2048 pixel short edge, a video resampled onto 24 fps and onto the canvas its own
    aspect ratio resolves to, and a soundtrack put on the audio VAE's sample rate and truncated to the generated
    duration.

    Args:
        references (`list[MiniMaxH3Reference]`): The references, validated by [`check_ref2va_references`].
        num_frames (`int`): The resolved frame count, of the form `17 * n + 5`.
        audio_sampling_rate (`int`): The audio VAE's sample rate.

    Returns:
        `list[MiniMaxH3Reference]`: the normalized references, same public types, in packed order.
    """
    max_duration = num_frames / MINIMAX_H3_FPS
    normalized = []
    for entry in references:
        waveform = None
        if entry.has_audio:
            sample_rate = entry.sample_rate if entry.sample_rate is not None else audio_sampling_rate
            waveform = normalize_reference_audio(entry.audio, sample_rate, audio_sampling_rate, max_duration)

        if entry.kind == "image":
            normalized.append(MiniMaxH3ImageReference(image=normalize_reference_image(entry.image)))
        elif entry.kind == "video":
            normalized.append(
                MiniMaxH3VideoReference(
                    frames=normalize_reference_video(entry.frames, float(entry.fps), num_frames),
                    fps=float(MINIMAX_H3_FPS),
                    audio=waveform,
                    sample_rate=None if waveform is None else audio_sampling_rate,
                )
            )
        else:
            normalized.append(MiniMaxH3AudioReference(audio=waveform, sample_rate=audio_sampling_rate))
    return normalized


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


def build_multiview_packed_sequence(
    text_token_tags: torch.Tensor,
    num_views: int,
    num_latent_frames_per_view: int,
    latent_height: int,
    latent_width: int,
    patch_size: Tuple[int, int, int],
    num_audio_latents: int = 0,
) -> MiniMaxH3PackedSequence:
    r"""
    Build the `[text | target audio | target video]` layout of a multiview request.

    The target video rows hold the views concatenated in view-major order — all rows of view 0 first, then view 1,
    and so on — which is how `VideoMultiViewsDataset` concatenates the views along time and how the multi-view
    transformer's `view_tags` address them. The rotary time axis restarts at the same origin for every view, so
    each view keeps the single-view temporal RoPE the base model was pre-trained with, and the full self-attention
    of the packed sequence couples the views. `video_view_tags` carries the view id of every video row for the
    transformer's view embeddings.

    Args:
        text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
            The modality tag of every text row, as produced by the prompt encoding.
        num_views (`int`): Number of views of the scene.
        num_latent_frames_per_view (`int`): Number of target latent frames of each view.
        latent_height (`int`): Target latent height.
        latent_width (`int`): Target latent width.
        patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.
        num_audio_latents (`int`, defaults to `0`):
            Number of target audio latents per channel. The nuScenes rig has no soundtrack, so the multiview
            layout defaults to an audio-free sequence; a nonzero count packs the audio rows between the text and
            the video exactly like `build_packed_sequence` does.

    Returns:
        [`MiniMaxH3PackedSequence`] whose `video_view_tags` tags every video row with its view id.
    """
    if num_views < 1:
        raise ValueError(f"`num_views` must be positive, got {num_views}.")
    if num_latent_frames_per_view < 1:
        raise ValueError(f"`num_latent_frames_per_view` must be positive, got {num_latent_frames_per_view}.")

    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = text_token_tags.shape[0]
    num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    rows_per_view = num_latent_frames_per_view * rows_per_frame
    num_video_rows = num_views * rows_per_view
    sequence_length = num_text_tokens + num_audio_rows + num_video_rows

    audio_start = num_text_tokens
    video_start = audio_start + num_audio_rows

    # 1. The (t, h, w) grid. Text rows sit on the time axis at their row index, and the media rows continue the
    # time axis from there, so text length shifts the whole media clock — as in `build_packed_sequence`. Every
    # view restarts the temporal grid at that origin, so each view keeps its single-view rotary positions.
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    frame_grid = torch.stack([grid.reshape(-1) for grid in torch.meshgrid(height_grid, width_grid, indexing="ij")], -1)

    if num_audio_rows:
        audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
        position_ids[audio_start:video_start, 0] = audio_time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
        position_ids[audio_start:video_start, 2] = torch.cat(
            [
                torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
                torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
            ]
        )

    view_time_grid = _temporal_position_grid(num_latent_frames_per_view, float(num_text_tokens))
    view_position_ids = torch.empty(num_latent_frames_per_view, rows_per_frame, 3, dtype=torch.float64)
    view_position_ids[:, :, 0] = view_time_grid[:, None]
    view_position_ids[:, :, 1:] = frame_grid[None]
    view_position_ids = view_position_ids.reshape(-1, 3)
    for view in range(num_views):
        position_ids[video_start + view * rows_per_view : video_start + (view + 1) * rows_per_view] = view_position_ids

    # 2. Row indices, modality tags and per-row view ids.
    video_indices = torch.arange(video_start, sequence_length)
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

    video_view_tags = torch.arange(num_views, dtype=torch.long).repeat_interleave(rows_per_view)

    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=0,
        num_condition_audio_rows=0,
        video_view_tags=video_view_tags,
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


def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""The `(h, w)` rotary coordinates of one latent frame, and the width axis they were built from."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _fill_audio_positions(
    position_ids: torch.Tensor,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: torch.Tensor,
    audio_channels: int,
) -> None:
    r"""
    Place one channel-major audio block.

    Audio rows carry no height coordinate and are pinned to the two extremes of the width grid of *their own* block —
    the target grid for a standalone audio reference, the video's grid for a soundtrack.
    """
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(audio_channels)
    position_ids[rows, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )


def build_ref2va_packed_sequence(
    text_token_tags: torch.Tensor,
    references: List[MiniMaxH3Reference],
    condition_latents: List[torch.Tensor],
    audio_condition_latents: List[torch.Tensor],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: Tuple[int, int, int],
) -> MiniMaxH3PackedSequence:
    r"""
    Build the `[text | reference blocks | target audio | target video]` layout of the `ref2va` task.

    The reference order advances the shared audio/video rotary clock, so it is part of the layout rather than a
    detail of the presentation: each block pushes the clock forward by the time it occupies — one integer slot for
    an image, its latent count for a standalone audio reference, and `max(soundtrack latents, video rotary span)`
    for a video reference, whose soundtrack rows are packed immediately before its video rows and share their
    origin.

    Args:
        text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
            The modality tag of every text row. Text is tagged `1`, except for the rows of a reference's vision
            block, which MiniMax-H3 tags `0` (video).
        references (`list[MiniMaxH3Reference]`):
            The references, in packed order. Only their modality is read here; the geometry comes from the latents.
        condition_latents (`list[torch.Tensor]`):
            One `(1, channels, num_latent_frames, latent_height, latent_width)` tensor per image and video
            reference, in packed order, as [`~MiniMaxH3Pipeline.encode_reference_latents`] produced them.
        audio_condition_latents (`list[torch.Tensor]`):
            One `(num_audio_latents * 2, audio_latent_channels)` tensor per audio-bearing reference, in packed
            order.
        num_latent_frames (`int`): Number of target latent frames.
        latent_height (`int`): Target latent height.
        latent_width (`int`): Target latent width.
        num_audio_latents (`int`): Number of target audio latents per channel.
        patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.

    Returns:
        [`MiniMaxH3PackedSequence`]
    """
    _, patch_h, patch_w = patch_size
    audio_channels = MINIMAX_H3_AUDIO_CHANNELS
    num_text_tokens = text_token_tags.shape[0]
    num_target_video_rows = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    num_target_audio_rows = num_audio_latents * audio_channels

    # The geometry of every reference block is the shape of what the encoder produced for it, so the two can never
    # disagree. Both lists are in packed order but skip the references they do not apply to, so they are consumed
    # as iterators alongside the reference list rather than indexed by it.
    visual_geometry = iter(tuple(latents.shape[2:5]) for latents in condition_latents)
    audio_row_counts = iter(rows.shape[0] for rows in audio_condition_latents)
    num_reference_video_rows = sum(
        frames * (height // patch_h) * (width // patch_w)
        for frames, height, width in (tuple(latents.shape[2:5]) for latents in condition_latents)
    )
    num_reference_audio_rows = sum(rows.shape[0] for rows in audio_condition_latents)
    sequence_length = (
        num_text_tokens
        + num_reference_video_rows
        + num_reference_audio_rows
        + num_target_audio_rows
        + num_target_video_rows
    )

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    # Reference blocks, in request order. `rotary_time` is the shared audio/video clock: it starts where the text
    # rows end and every block pushes it forward by the time that block occupies.
    video_indices, audio_indices = [], []
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for reference in references:
        if reference.kind == "image":
            num_latent_frames_, reference_height, reference_width = next(visual_geometry)
            num_video_rows = num_latent_frames_ * (reference_height // patch_h) * (reference_width // patch_w)
            rows = slice(cursor, cursor + num_video_rows)
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(reference_height, reference_width, patch_h, patch_w)
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            # An image is a single frame and takes a single integer rotary slot, not a latent frame's 5/3 units.
            rotary_time += 1.0
        elif reference.kind == "audio":
            num_audio_rows = next(audio_row_counts)
            reference_audio_latents = num_audio_rows // audio_channels
            rows = slice(cursor, cursor + num_audio_rows)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(
                position_ids, rows, reference_audio_latents, rotary_time, target_width_grid, audio_channels
            )
            rotary_time += float(reference_audio_latents)
        elif reference.kind == "video":
            # A video reference's soundtrack rows are packed immediately before its video rows and share their
            # origin, so the two are rotary-aligned exactly as the generated audio and video are.
            num_audio_rows = next(audio_row_counts) if reference.has_audio else 0
            reference_audio_latents = num_audio_rows // audio_channels
            num_latent_frames_, reference_height, reference_width = next(visual_geometry)
            num_video_rows = num_latent_frames_ * (reference_height // patch_h) * (reference_width // patch_w)
            audio_rows = slice(cursor, cursor + num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + num_video_rows)
            cursor = video_rows.stop
            audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))

            frame_grid, width_grid = _frame_position_grid(reference_height, reference_width, patch_h, patch_w)
            _fill_audio_positions(
                position_ids, audio_rows, reference_audio_latents, rotary_time, width_grid, audio_channels
            )
            frame_time = _temporal_position_grid(num_latent_frames_, rotary_time)
            position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
            position_ids[video_rows, 1:] = frame_grid.repeat(num_latent_frames_, 1)
            # The rotary time this reference advances the clock by, summed sequentially in float64. That is *not*
            # how `_temporal_position_span` sums the same series — that one reproduces a numpy pairwise sum, and
            # the two orders differ in the last ulp from 16 latent frames onwards. The reference implementation
            # keeps both, one per call site, so the port has to as well.
            video_span = sum(
                _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
                for index in range(num_latent_frames_)
            )
            rotary_time += max(float(reference_audio_latents), video_span)
        else:
            raise ValueError(f"A reference must be an 'image', a 'video' or an 'audio', got {reference.kind!r}.")

    # The generated rows. Target audio and target video share the origin the reference blocks left behind.
    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(
        position_ids,
        slice(audio_start, video_start),
        num_audio_latents,
        rotary_time,
        target_width_grid,
        audio_channels,
    )
    frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

    video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
    audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
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
        num_condition_video_rows=num_reference_video_rows,
        num_condition_audio_rows=num_reference_audio_rows,
    )


def ref2va_condition_rows(
    scheduler,
    condition_latents: List[torch.Tensor],
    patch_size: Tuple[int, int, int],
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    r"""
    Noise the encoded `ref2va` visual conditioning to MiniMax-H3's conditioning level and pack it into rows.

    One draw per reference, in packed order, off the request's generator — these are the *first* draws of a
    request, ahead of the video and audio noise of [`~MiniMaxH3Pipeline.prepare_latents`], and the order is part
    of what a generator reproduces. Each reference is packed on its own because references are encoded at their
    own resolutions, so their latents do not share a shape.

    Args:
        scheduler ([`MiniMaxH3Scheduler`]): The video schedule, whose `scale_noise` mixes at `t = 0.999`.
        condition_latents (`list[torch.Tensor]`):
            One `(1, latent_channels, num_latent_frames, latent_height, latent_width)` tensor per image and video
            reference, in packed order.
        patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.
        generator (`torch.Generator`, *optional*): The generator of the request.
        device (`torch.device`, *optional*): The device the noise is drawn on.

    Returns:
        `torch.Tensor`: the noised conditioning rows, concatenated in packed order.
    """
    packed = []
    for condition in condition_latents:
        noise = randn_tensor(condition.shape, generator=generator, device=device, dtype=torch.float32)
        # The anchors are not fully clean: the released model noises them to `t = 0.999` and holds them there for
        # every step. Mixing before the patchify is the same arithmetic, since patchify only permutes.
        noised = scheduler.scale_noise(condition.to(device), MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)
        packed.append(patchify_video_latents(noised, patch_size))
    return torch.cat(packed)



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

    # The duration window a request must fit into; the control variant lowers the bound, since it follows the
    # control video's actual length without padding a short one.
    _min_duration = MINIMAX_H3_MIN_DURATION
    _max_duration = MINIMAX_H3_MAX_DURATION

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
            transformer_subfolder (`str`, *optional*, defaults to `"transformer"`):
                The subfolder the transformer is loaded from, read by the subfolder assembly below. The `ref2va`
                weights ship in `transformer_ref`, same architecture, so pass `transformer_subfolder="transformer_ref"`
                for them; a snapshot without that folder only carries the base weights.
        """
        import os

        from ..models.minimax_h3_conversion import is_raw_minimax_h3_format

        path = pretrained_model_name_or_path
        torch_dtype = kwargs.pop("torch_dtype", kwargs.pop("dtype", None))
        transformer_subfolder = kwargs.pop("transformer_subfolder", "transformer")

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
            _subfolder(transformer_subfolder), torch_dtype=torch_dtype, low_cpu_mem_usage=True
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
    def vae_frames_per_chunk(self) -> int:
        if getattr(self, "vae", None) is not None:
            return self.vae.config.clip_length
        return 17

    @property
    def vae_latents_per_chunk(self) -> int:
        if getattr(self, "vae", None) is not None:
            return self.vae.tokens_chunk_size
        return 5

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
        if not self._min_duration <= duration <= self._max_duration:
            raise ValueError(
                f"MiniMax-H3 generates between {self._min_duration} and {self._max_duration} seconds at "
                f"{MINIMAX_H3_FPS} fps, so `num_frames`, rounded up to the next `17 * n + 5` the video VAE can "
                f"encode, must be between {int(self._min_duration * MINIMAX_H3_FPS)} and "
                f"{int(self._max_duration * MINIMAX_H3_FPS)}, got {num_frames} (rounded up to "
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

    @staticmethod
    def _sample_ref2va_condition_frames(
        frames: np.ndarray, fps: float, sample_fps: float, temporal_patch: int
    ) -> Tuple[List[np.ndarray], List[float]]:
        r"""
        Sample the frames the conditioner sees from a normalized reference video, and label their vision blocks.

        The conditioner reads a reference at `sample_fps`: every `fps / sample_fps`-th frame, deduplicated. Qwen3-VL
        then merges the sampled frames in groups of `temporal_patch` — repeating the last one when the count does not
        divide — and a merged group is labelled with the mean of its timestamps, which `"<{timestamp:.1f} seconds>"`
        renders with Python's round-half-to-even, so the first block of a 2 fps pair is `"<0.2 seconds>"` rather than
        `"<0.3 seconds>"`.

        Returns:
            `tuple[list[np.ndarray], list[float]]`: the sampled frames and one timestamp per vision block.
        """
        stride = fps / sample_fps
        indices, cursor = [], 0.0
        while round(cursor) < frames.shape[0]:
            if not indices or round(cursor) > indices[-1]:
                indices.append(round(cursor))
            cursor += stride
        if len(indices) < temporal_patch:
            minimum = round((temporal_patch - 1) * stride) + 1
            raise ValueError(
                f"A reference video is read at {sample_fps:g} fps and its sampled frames are merged in groups of "
                f"{temporal_patch}, so it must run at least {minimum} frames at {fps:g} fps "
                f"({minimum / fps:.2g} seconds), got {frames.shape[0]}."
            )

        timestamps = [index / sample_fps for index in range(len(indices))]
        timestamps += [timestamps[-1]] * (-len(timestamps) % temporal_patch)
        block_timestamps = [
            (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2
            for index in range(0, len(timestamps), temporal_patch)
        ]
        return [frames[index] for index in indices], block_timestamps

    def _gather_ref2va_vision_features(self, references: List[MiniMaxH3Reference]) -> Dict[str, torch.Tensor]:
        r"""
        Run the references' pixels through the conditioner's processors, batched per modality.

        Audio contributes nothing — a waveform never reaches the conditioner. Returns the vision tensors keyed by
        the conditioner's parameter names.
        """
        merge_size = self.processor.image_processor.merge_size**2
        vision_inputs = {}

        images = [reference.image for reference in references if reference.kind == "image"]
        if images:
            image_features = self.processor.image_processor(images=images, return_tensors="pt")
            vision_inputs["pixel_values"] = image_features["pixel_values"]
            vision_inputs["image_grid_thw"] = image_features["image_grid_thw"]
            # The presentation only needs the grid of each image reference to size its vision block; the counts
            # themselves are re-derived there from the same grids.
            self._ref2va_image_token_counts = [
                int(grid.prod()) // merge_size for grid in image_features["image_grid_thw"]
            ]

        videos = [reference for reference in references if reference.kind == "video"]
        if videos:
            temporal_patch = self.processor.video_processor.temporal_patch_size
            sampled = [
                self._sample_ref2va_condition_frames(
                    reference.frames, float(reference.fps), MINIMAX_H3_VIDEO_SAMPLE_FPS, temporal_patch
                )
                for reference in videos
            ]
            self._ref2va_video_block_timestamps = [timestamps for _, timestamps in sampled]
            video_features = self.processor.video_processor(
                videos=[np.stack(frames) for frames, _ in sampled], do_sample_frames=False, return_tensors="pt"
            )
            vision_inputs["pixel_values_videos"] = video_features["pixel_values_videos"]
            vision_inputs["video_grid_thw"] = video_features["video_grid_thw"]
            self._ref2va_video_block_token_counts = [
                int(grid[1]) * int(grid[2]) // merge_size for grid in video_features["video_grid_thw"]
            ]
            for timestamps, grid in zip(self._ref2va_video_block_timestamps, video_features["video_grid_thw"]):
                if int(grid[0]) != len(timestamps):
                    raise ValueError(
                        f"The processor merged a reference video into {int(grid[0])} vision blocks, but MiniMax-H3 "
                        f"labels {len(timestamps)} of them."
                    )
        return vision_inputs

    def _build_ref2va_presentation(
        self, prompt: str, references: List[MiniMaxH3Reference]
    ) -> Tuple[List[int], List[int]]:
        r"""
        Tokenize MiniMax-H3's presentation of a `ref2va` request.

        Every reference prepends a label, in packed order and numbered per modality: `"<Picture i>: "` plus a vision
        block for an image, `"<Audio j>: "` alone for audio and `"<Video k>: "` plus one timestamped vision block per
        merged frame pair for a video. A video that carries sound is labelled `"<Audio j>: "` *before* `"<Video k>: "`,
        mirroring the order its rows are packed in. The prompt follows verbatim, with no chat template and no special
        tokens.
        """
        image_token_counts = getattr(self, "_ref2va_image_token_counts", [])
        video_block_token_counts = getattr(self, "_ref2va_video_block_token_counts", [])
        video_block_timestamps = getattr(self, "_ref2va_video_block_timestamps", [])

        def text(value: str) -> Tuple[List[int], List[int]]:
            ids = self.tokenizer(value, add_special_tokens=False)["input_ids"]
            return ids, [MINIMAX_H3_TEXT_TAG] * len(ids)

        def vision(pad_token: str, num_tokens: int) -> Tuple[List[int], List[int]]:
            ids = (
                [self.tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [self.tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
                + [self.tokenizer.convert_tokens_to_ids("<|vision_end|>")]
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
                    # `"{:.1f}"` rounds half to even, so the mean of a 2 fps pair renders as "<0.2 seconds>".
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

    def encode_prompt(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        references: Optional[List["MiniMaxH3Reference"]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Build MiniMax-H3's presentation of a request and encode it.

        The presentation is the verbatim prompt for `t2va`. Every keyframe prepends a `"<Picture i>: "` label and a
        vision block (`<|vision_start|>`, one `<|image_pad|>` per vision patch, `<|vision_end|>`) — no chat template
        and no special tokens. The rows of a vision block are tagged as *video* rather than text, which is what the
        transformer's AdaLN modulation keys off.

        When `references` is given, the `ref2va` presentation is built instead and `images` is ignored: every
        reference prepends a label numbered per modality — `"<Audio j>: "` first when it carries sound, then
        `"<Picture i>: "` plus a vision block, or `"<Video k>: "` plus one timestamped vision block per merged frame
        pair — and the prompt follows verbatim. Audio-only references contribute nothing: a waveform never reaches
        the conditioner.

        Args:
            prompt (`str`): The prompt to encode.
            images (`list[PIL.Image.Image]`, *optional*):
                The keyframes, already prepared onto the target canvas, in packed order.
            references (`list[MiniMaxH3Reference]`, *optional*):
                The references, normalized by the setup of [`~MiniMaxH3Pipeline.__call__`], in packed order.
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
        vision_inputs = {}
        token_ids, token_tags = [], []
        if references:
            # The vision tensors are batched per modality while the presentation is tokenized in request order; the
            # two agree because the filtering preserves relative order within each modality and Qwen3-VL fills the
            # n-th pad *run* of a modality with the n-th entry of that modality's batch. The vision features go
            # first because their gather caches the vision block timestamps the presentation reads.
            vision_inputs = self._gather_ref2va_vision_features(references)
            token_ids, token_tags = self._build_ref2va_presentation(prompt, references)
            pixel_values = vision_inputs.get("pixel_values")
            image_grid_thw = vision_inputs.get("image_grid_thw")
        elif images:
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
        prompt_ids = [] if references else self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
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
        if "pixel_values_videos" in vision_inputs:
            encoder_kwargs["pixel_values_videos"] = vision_inputs["pixel_values_videos"].to(
                device, self.text_encoder.dtype
            )
            encoder_kwargs["video_grid_thw"] = vision_inputs["video_grid_thw"].to(device)
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

    def encode_reference_latents(
        self, references: List[MiniMaxH3Reference], device: Optional[torch.device] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        r"""
        Encode the `ref2va` references: image and video references through the video VAE, soundtracks through the
        audio VAE.

        They are the anchors of the whole denoising loop, which only ever writes the generated rows: the visual ones
        are noised to MiniMax-H3's conditioning level and packed by [`ref2va_condition_rows`], while the soundtracks
        ride along clean at `t = 0`. The latent geometry of every reference is the shape of what this returns, which
        is what the packed layout is built from.

        A video reference is encoded *down* to the nearest `17 * n + 5` frame count the video VAE chunks over, so it
        is encoded without padding — this only bites when the reference is shorter than the target, whose own frame
        count already has that form. Its posterior is sampled under the same fixed generator as a keyframe's; a
        soundtrack takes the posterior *mean* and is never sampled.

        Args:
            references (`list[MiniMaxH3Reference]`):
                The references, normalized by the setup of [`~MiniMaxH3Pipeline.__call__`], in packed order.
            device (`torch.device`, *optional*): The device to run the VAEs on.

        Returns:
            `tuple[list[torch.Tensor], list[torch.Tensor]]`: one `(1, latent_channels, num_latent_frames,
            latent_height, latent_width)` float32 CPU tensor per image and video reference in packed order, and one
            `(num_audio_latents * 2, audio_latent_channels)` row tensor per audio-bearing reference in packed order.
        """
        device = device or self._execution_device
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)

        def encode_pixels(pixels: torch.Tensor) -> torch.Tensor:
            pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
            posterior = DiagonalGaussianDistribution(self.vae._encode(pixels))
            latents = posterior.sample(generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
            # The sampled latent is rounded to float16 before it is normalized, as for the keyframes: ~11 bits of
            # every conditioning latent, so the released model's conditioning cannot be reproduced without it.
            latents = latents.to(torch.float16).float().cpu()
            return (latents - latents_mean) / latents_std

        condition_latents = []
        # `_encode` is a method call, not the VAE's `forward`, so the top-level CPU-offload hook never fires around
        # it on its own: scope the whole encode, once, instead of per reference.
        with _offload_scope(self.vae):
            for reference in references:
                if reference.kind == "image":
                    pixels = torch.from_numpy(np.array(reference.image)).to(device).permute(2, 0, 1)[None, :, None]
                    condition_latents.append(encode_pixels(pixels))
                elif reference.kind == "video":
                    # Snap *down* to `frames_per_chunk * n + latents_per_chunk` so the VAE encodes without padding.
                    num_frames = reference.frames.shape[0]
                    frames_per_chunk, latents_per_chunk = self.vae_frames_per_chunk, self.vae_latents_per_chunk
                    num_frames = (
                        max(1, (num_frames - latents_per_chunk) // frames_per_chunk) * frames_per_chunk
                        + latents_per_chunk
                    )
                    pixels = (
                        torch.from_numpy(reference.frames[:num_frames].copy()).to(device).permute(3, 0, 1, 2)[None]
                    )
                    condition_latents.append(encode_pixels(pixels))

        audio_latents_mean = torch.tensor(self.audio_vae.config.latents_mean).view(1, 1, -1)
        audio_latents_std = torch.tensor(self.audio_vae.config.latents_std).view(1, 1, -1)
        audio_condition_latents = []
        with _offload_scope(self.audio_vae):
            for reference in references:
                if reference.has_audio:
                    posterior = self.audio_vae.encode(reference.audio.to(device)[:, None], return_dict=False)[0]
                    # Channel-major rows: the two stereo channels are two batch items of the mono audio VAE.
                    latents = posterior.mode().float().cpu().transpose(1, 2)
                    normalized = (latents - audio_latents_mean) / audio_latents_std
                    audio_condition_latents.append(normalized.reshape(-1, self.audio_latent_channels))
        return condition_latents, audio_condition_latents

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
        references: Optional[List[MiniMaxH3Reference]] = None,
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
        Generate a video and its soundtrack, as a `t2va`, an `fl2va` or a `ref2va` request.

        Args:
            prompt (`str`):
                The prompt to guide generation. MiniMax-H3 packs one request into one sequence, so a batch of prompts
                is not a thing.
            image (`PIL.Image.Image`, *optional*):
                Keyframe the video starts from. It is *stretched* onto the target canvas, which by default is derived
                from its own aspect ratio. Mutually exclusive with `references`.
            last_image (`PIL.Image.Image`, *optional*):
                Keyframe the video ends on. Can be passed on its own to generate *up to* a frame. Combined with
                `image` it is the follower of the two and is cover-cropped onto the canvas. Mutually exclusive with
                `references`.
            references (`list[MiniMaxH3Reference]`, *optional*):
                The `ref2va` references to condition on, **in the order the model should read them**: the order
                labels them in the prompt presentation and lays them out on the shared rotary clock, so a different
                order is a different request. One dataclass per modality, all holding in-memory media — a
                [`MiniMaxH3ImageReference`] (at most 9), a [`MiniMaxH3VideoReference`] at its own `fps` (at most 3,
                whose `audio` soundtrack is conditioned on as well), or a [`MiniMaxH3AudioReference`] at its own
                `sample_rate` (at most 3) — for at most 12 references in total, and audio references cannot be the
                only ones. Decode files with each class's `from_file` classmethod, which brings the rates along. The
                `ref2va` checkpoint is guidance-distilled with no unconditional branch, so `references` needs
                `guidance_scale <= 1`, and it needs a transformer loaded from `transformer_ref`.
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

        # `ref2va` is a task of its own: the keyframes of `fl2va` are mutually exclusive with it, and the released
        # `ref2va` checkpoint is guidance-distilled with no unconditional branch, so there is no CFG to run.
        do_ref2va = bool(references)
        if do_ref2va:
            if image is not None or last_image is not None:
                raise ValueError(
                    "`references` is the `ref2va` task, which is mutually exclusive with the `image` / `last_image` "
                    "keyframes of `fl2va`."
                )
            if guidance_scale > 1.0:
                raise ValueError(
                    "The `ref2va` checkpoint is guidance-distilled and has no unconditional branch, so `references` "
                    f"needs `guidance_scale <= 1`, got {guidance_scale}."
                )
            references = check_ref2va_references(list(references))

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
        if do_ref2va:
            # The references never bind the generated geometry: they are normalized onto their own resolutions, with
            # soundtracks truncated to the resolved duration.
            references = normalize_ref2va_references(references, num_frames, self.audio_sampling_rate)
        else:
            keyframes = [
                prepare_keyframe_image(keyframe, height, width, stretch=index == 0)
                for index, keyframe in enumerate(keyframes)
            ]

        # 2. Encode MiniMax-H3's presentation of the request. The released checkpoint is guidance-distilled, so the
        # default guidance_scale of 1 runs one forward pass per step with no CFG; a guidance_scale above 1 enables
        # classifier-free guidance with a negative prompt.
        do_cfg = guidance_scale > 1.0
        if do_ref2va:
            prompt_embeds, text_token_tags = self.encode_prompt(
                prompt, references=references, device=device, dtype=self.transformer.dtype
            )
        else:
            prompt_embeds, text_token_tags = self.encode_prompt(
                prompt, keyframes, device=device, dtype=self.transformer.dtype
            )
            if do_cfg:
                negative_prompt = negative_prompt if negative_prompt is not None else ""
                negative_prompt_embeds, negative_text_token_tags = self.encode_prompt(
                    negative_prompt, keyframes, device=device, dtype=self.transformer.dtype
                )

        # 3. Encode the conditioning and noise it to MiniMax-H3's conditioning level. The anchors are the whole
        # denoising loop's invariant: the loop only ever writes the generated rows.
        audio_condition_latents = []
        condition_latents = None
        if do_ref2va:
            condition_latents, audio_condition_latents = self.encode_reference_latents(references, device=device)
        elif keyframes:
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
        if do_ref2va:
            layout = build_ref2va_packed_sequence(
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
        else:
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

        # 5. Draw the noise of the generated rows and prepend the conditioning rows. The reference noise is the
        # request's *first* draw — ahead of the video and audio noise below — and the order is part of what the
        # generator reproduces.
        if do_ref2va:
            condition_rows = ref2va_condition_rows(
                self.scheduler, condition_latents, self.patch_size, generator=generator, device=device
            )
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
        if do_ref2va:
            latents = torch.cat([condition_rows, latents])
            if audio_condition_latents:
                num_reference_audio_rows = sum(rows.shape[0] for rows in audio_condition_latents)
                if num_reference_audio_rows != num_condition_audio_rows:
                    raise ValueError(
                        f"The layout reserved {num_condition_audio_rows} reference audio rows but the encoded "
                        f"soundtracks pack into {num_reference_audio_rows}. The references the layout was built from "
                        "and the ones the audio conditioning was encoded from do not agree."
                    )
                # Soundtracks are never noised: a reference soundtrack conditions clean, at `t = 0`.
                audio_latents = torch.cat(
                    [rows.to(device) for rows in audio_condition_latents] + [audio_latents]
                )
        elif condition_latents is not None:
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
