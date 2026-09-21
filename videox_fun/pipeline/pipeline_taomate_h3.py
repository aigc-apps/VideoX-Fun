# Modified from https://github.com/TaoLiveAIGC/TaoMate-H3 (Apache-2.0 / MiniMax H3 Community License).
# Copyright 2026 The TaoMate authors and the VideoX-Fun team. All rights reserved.
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
TaoMate-H3 streaming inference on top of [`MiniMaxH3Pipeline`].

TaoMate turns one long T2VA request into a sequence of five-second *stream requests*, and every stream request is
split into four audio/video chunks (`(2, 2, 2, 1)` seventeen-frame groups, plus a five-frame / two-latent affine
prefix on the first one) that are denoised one after another. Each chunk is a **future-free packed sequence**
(`[text | audio | video]`, no padding) laid out on a *global* rotary timeline, so a chunk sees its past through the
rotary positions instead of a growing context; the clean audio/video K/V of every finished chunk is **committed**
to a persistent KV cache — the first chunk's video rows kept as a sink plus the two most recent chunks, with the
audio history dropped every 12 requests — and the next chunk attends over it. This is the
`persistent_kv = clean_audio_video` contract.

The soundtrack does not run a long schedule at all: a Base10 teacher (an offline artifact of three clean audio
milestones, captured at states 3 / 6 / 9 of a 10-state shifted schedule) is copied into the audio rows after each
of the three student steps (`fl2va_tail40_rollover`). Clean video rows are renormalized to the first chunk's
per-channel mean / std (`affine_to_first_stream_chunk`), and every request after the first is generated on the
*canonical continuation* geometry (119 native frames / 35 video latents / 198-199 audio latents, spliced behind the
previous request through a two-latent video and a nine-latent audio transport prefix). Both VAE decodes run
**once**, over the spliced full timeline (`one_shot_audio_video_vae`).

The per-chunk arithmetic — sigma schedules, the Euler update, the packed layout and the transformer call — is
shared with [`MiniMaxH3Pipeline`]; the streaming-specific state machine lives in `MiniMaxH3StreamingKVCache` /
`MiniMaxH3StreamingKVCacheAttnProcessor` (`videox_fun/models/minimax_h3_transformer3d.py`). Single-GPU only: the
persistent cache holds whole-sequence K/V while sequence parallelism shards heads across ranks, so the two cannot
be combined.
"""

import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from diffusers.utils import BaseOutput, logging

from ..models.minimax_h3_transformer3d import (install_minimax_h3_streaming_kv_cache,
                                               uninstall_minimax_h3_streaming_kv_cache)
from .pipeline_minimax_h3 import (MINIMAX_H3_AUDIO_TAG, MINIMAX_H3_VIDEO_TAG,
                                  MiniMaxH3PackedSequence, MiniMaxH3Pipeline,
                                  build_packed_sequence, build_row_timesteps,
                                  patchify_video_latents)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# The streaming contract. These values are deliberately not options: changing one of them creates a different
# inference method (the TaoMate `DirectPolicy`).
TAOMATE_H3_VIDEO_FPS = 24
TAOMATE_H3_AUDIO_LATENT_RATE = 40
TAOMATE_H3_VIDEO_GROUP_FRAMES = 17
TAOMATE_H3_VIDEO_GROUP_LATENTS = 5
TAOMATE_H3_VIDEO_PREFIX_FRAMES = 5
TAOMATE_H3_VIDEO_PREFIX_LATENTS = 2
TAOMATE_H3_CHUNK_GROUP_COUNTS = (2, 2, 2, 1)
TAOMATE_H3_REQUEST_FRAMES = 124
TAOMATE_H3_REQUEST_SECONDS = 5
TAOMATE_H3_STEADY_FRAMES = TAOMATE_H3_REQUEST_FRAMES - TAOMATE_H3_VIDEO_PREFIX_FRAMES
TAOMATE_H3_REQUEST_VIDEO_LATENTS = 37
TAOMATE_H3_REQUEST_AUDIO_LATENTS = 207
TAOMATE_H3_AUDIO_TRANSPORT_PREFIX_LATENTS = 9
TAOMATE_H3_AUDIO_KV_RESET_WINDOW_REQUESTS = 12
TAOMATE_H3_SUPPORTED_SHORT_EDGES = (480, 768, 1088)
TAOMATE_H3_VIDEO_SIGMA_SHIFT = 12.0
TAOMATE_H3_AUDIO_SIGMA_SHIFT = 3.0
TAOMATE_H3_SIGMA_GRID_STEPS = 50
TAOMATE_H3_DISTILLED_STATE_INDICES = (0, 16, 33, 49)
TAOMATE_H3_TEACHER_STATE_NUMBERS = (3, 6, 9)
TAOMATE_H3_VIDEO_NOISE_REQUEST_STRIDE = 1_000_003
TAOMATE_H3_AUDIO_LATENT_CHANNELS = 32
TAOMATE_H3_ROLLOVER_REFERENCE_LATENTS = 40


# Streaming geometry: chunk boundaries and the global rotary timeline.
def taomate_h3_round_half_even(value: Fraction) -> int:
    r"""Round a fraction to the nearest integer, ties to even."""
    quotient, remainder = divmod(value.numerator, value.denominator)
    doubled = remainder * 2
    if doubled < value.denominator:
        return quotient
    if doubled > value.denominator:
        return quotient + 1
    return quotient + (quotient & 1)


def taomate_h3_audio_latent_boundary(frame: int) -> int:
    r"""The audio latent index aligned with native frame `frame` (40 Hz audio clock)."""
    return taomate_h3_round_half_even(Fraction(frame * TAOMATE_H3_AUDIO_LATENT_RATE, TAOMATE_H3_VIDEO_FPS))


def taomate_h3_video_temporal_position(latent_index: int) -> Fraction:
    r"""
    The absolute H3 rotary time of one video latent on the global timeline.

    Latent frames advance non-uniformly: `5/3` of a time unit per native frame at 24 fps, grouped `1, 4, 4, 4, 4`
    frames per latent (one standalone first frame plus four 17-frame group quarters).
    """
    if latent_index < 0:
        raise ValueError("latent_index must be non-negative")
    weights = (1, 4, 4, 4, 4)
    return sum(
        (Fraction(5, 3) * weights[index % len(weights)] for index in range(latent_index)),
        start=Fraction(0),
    )


def taomate_h3_video_temporal_positions(start: int, count: int) -> Tuple[Fraction, ...]:
    r"""The rotary times of `count` consecutive video latents from `start`."""
    if count <= 0:
        raise ValueError("count must be a positive integer")
    return tuple(taomate_h3_video_temporal_position(index) for index in range(start, start + count))


@dataclass(frozen=True)
class TaomateH3StreamPhase:
    r"""One audio/video chunk of a stream request."""

    index: int
    group_count: int
    frame_start: int
    frame_stop: int
    video_latent_start: int
    video_latent_stop: int
    audio_latent_start: int
    audio_latent_stop: int

    @property
    def frame_count(self) -> int:
        return self.frame_stop - self.frame_start

    @property
    def video_latent_count(self) -> int:
        return self.video_latent_stop - self.video_latent_start

    @property
    def audio_latent_count(self) -> int:
        return self.audio_latent_stop - self.audio_latent_start

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaomateH3StreamPlan:
    r"""The phase plan of one stream request: native frames split into chunks."""

    native_frame_count: int
    phases: Tuple[TaomateH3StreamPhase, ...]

    @property
    def duration_seconds(self) -> float:
        return self.native_frame_count / TAOMATE_H3_VIDEO_FPS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "native_frame_count": self.native_frame_count,
            "duration_seconds": self.duration_seconds,
            "phase_count": len(self.phases),
            "phases": [phase.to_dict() for phase in self.phases],
        }


def taomate_h3_direct_5s_plan() -> TaomateH3StreamPlan:
    r"""
    The validated four-chunk plan of the first five-second stream request.

    Chunk 0 carries the five-frame / two-latent affine prefix on top of its two 17-frame groups; the audio
    boundaries round the native frame timeline to the 40 Hz audio clock half-to-even.
    """
    phases: List[TaomateH3StreamPhase] = []
    frame_stop = 0
    video_stop = 0
    for index, groups in enumerate(TAOMATE_H3_CHUNK_GROUP_COUNTS):
        frame_start = frame_stop
        video_start = video_stop
        frame_stop += groups * TAOMATE_H3_VIDEO_GROUP_FRAMES
        video_stop += groups * TAOMATE_H3_VIDEO_GROUP_LATENTS
        if index == 0:
            frame_stop += TAOMATE_H3_VIDEO_PREFIX_FRAMES
            video_stop += TAOMATE_H3_VIDEO_PREFIX_LATENTS
        phases.append(
            TaomateH3StreamPhase(
                index=index,
                group_count=groups,
                frame_start=frame_start,
                frame_stop=frame_stop,
                video_latent_start=video_start,
                video_latent_stop=video_stop,
                audio_latent_start=taomate_h3_audio_latent_boundary(frame_start),
                audio_latent_stop=taomate_h3_audio_latent_boundary(frame_stop),
            )
        )
    return TaomateH3StreamPlan(native_frame_count=TAOMATE_H3_REQUEST_FRAMES, phases=tuple(phases))


def taomate_h3_canonical_continuation_plan(
    base_plan: TaomateH3StreamPlan, *, request_index: int
) -> TaomateH3StreamPlan:
    r"""
    The steady-state geometry of every stream request after the first.

    A continuation does not regenerate the five-frame prefix: it advances seven native 17-frame groups (35 video
    latents / 119 frames), while the audio boundaries round on the *global* frame timeline, so the per-request
    audio target is 198 or 199 latents rather than a repeated 207.
    """
    if request_index <= 0:
        raise ValueError("canonical continuation requires request_index > 0")
    if base_plan.native_frame_count <= TAOMATE_H3_VIDEO_PREFIX_FRAMES:
        raise ValueError("base stream has no steady continuation geometry")

    steady_frame_count = base_plan.native_frame_count - TAOMATE_H3_VIDEO_PREFIX_FRAMES
    global_frame_start = base_plan.native_frame_count + (request_index - 1) * steady_frame_count
    global_audio_start = taomate_h3_audio_latent_boundary(global_frame_start)

    phases: List[TaomateH3StreamPhase] = []
    frame_stop = 0
    video_stop = 0
    for index, base_phase in enumerate(base_plan.phases):
        frame_start = frame_stop
        video_start = video_stop
        frame_stop += base_phase.group_count * TAOMATE_H3_VIDEO_GROUP_FRAMES
        video_stop += base_phase.group_count * TAOMATE_H3_VIDEO_GROUP_LATENTS
        phases.append(
            TaomateH3StreamPhase(
                index=index,
                group_count=base_phase.group_count,
                frame_start=frame_start,
                frame_stop=frame_stop,
                video_latent_start=video_start,
                video_latent_stop=video_stop,
                audio_latent_start=(
                    taomate_h3_audio_latent_boundary(global_frame_start + frame_start) - global_audio_start
                ),
                audio_latent_stop=(
                    taomate_h3_audio_latent_boundary(global_frame_start + frame_stop) - global_audio_start
                ),
            )
        )
    return TaomateH3StreamPlan(native_frame_count=steady_frame_count, phases=tuple(phases))


# Sigma schedules: the 50-state shifted grids the checkpoint was distilled on, sampled at the four retained states
# (`0, 16, 33, 49`) -> three steps.
def taomate_h3_time_shift_sigmas(*, num_steps: int = TAOMATE_H3_SIGMA_GRID_STEPS, shift_scale: float) -> List[float]:
    r"""`linspace(1, 0, N)` pushed through `s * base / (1 + (s - 1) * base)`."""
    if num_steps <= 1 or not (0.0 < shift_scale < float("inf")):
        raise ValueError("sigma schedule requires num_steps > 1 and shift_scale > 0")
    base = torch.linspace(1.0, 0.0, num_steps, dtype=torch.float32, device="cpu")
    shifted = shift_scale * base / (1.0 + (shift_scale - 1.0) * base)
    shifted = torch.unique_consecutive(shifted)
    if int(shifted.numel()) != num_steps:
        raise ValueError("shifted sigma schedule changed cardinality")
    return [float(item) for item in shifted.tolist()]


def taomate_h3_select_time_shift_sigmas(
    *, shift_scale: float, state_indices: Optional[Sequence[int]] = None, num_steps: int = TAOMATE_H3_SIGMA_GRID_STEPS
) -> List[float]:
    r"""Sample the shifted sigma grid at the retained state indices."""
    schedule = taomate_h3_time_shift_sigmas(num_steps=num_steps, shift_scale=shift_scale)
    if state_indices is None:
        return schedule
    indices = tuple(state_indices)
    if (
        len(indices) < 2
        or indices[0] != 0
        or indices[-1] != num_steps - 1
        or tuple(sorted(set(indices))) != indices
    ):
        raise ValueError("retained sigma indices must be sorted, unique, and span the schedule")
    return [schedule[index] for index in indices]


def taomate_h3_streaming_sigma_schedules() -> Tuple[List[float], List[float]]:
    r"""The `(video, audio)` three-step schedules of the streaming Stage3 contract."""
    sigmas_video = taomate_h3_select_time_shift_sigmas(
        shift_scale=TAOMATE_H3_VIDEO_SIGMA_SHIFT, state_indices=TAOMATE_H3_DISTILLED_STATE_INDICES
    )
    sigmas_audio = taomate_h3_select_time_shift_sigmas(
        shift_scale=TAOMATE_H3_AUDIO_SIGMA_SHIFT, state_indices=TAOMATE_H3_DISTILLED_STATE_INDICES
    )
    return sigmas_video, sigmas_audio


# The offline Base10 audio teacher: three clean audio milestones per request, produced ahead of inference (see
# scripts/taomate_h3/generate_base10_teacher.py).
TAOMATE_H3_TEACHER_GEOMETRY_KEYS = ("width", "height", "video_latent_h", "video_latent_w")


class TaomateH3TeacherError(RuntimeError):
    r"""The external Base10 teacher artifact does not match the current run."""


def taomate_h3_teacher_geometry(width: int, height: int, *, video_latent_h: int, video_latent_w: int) -> Dict[str, int]:
    r"""The spatial geometry identity stored in a teacher artifact."""
    return {
        "width": int(width),
        "height": int(height),
        "video_latent_h": int(video_latent_h),
        "video_latent_w": int(video_latent_w),
    }


def _taomate_h3_teacher_json(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaomateH3TeacherError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise TaomateH3TeacherError(f"{label} must be a JSON object")
    return value


def _taomate_h3_teacher_geometry(value: Any, *, label: str) -> Dict[str, int]:
    if not isinstance(value, dict):
        raise TaomateH3TeacherError(f"{label} is absent")
    result: Dict[str, int] = {}
    for key in TAOMATE_H3_TEACHER_GEOMETRY_KEYS:
        item = value.get(key)
        if type(item) is not int or item <= 0:
            raise TaomateH3TeacherError(f"{label}.{key} must be a positive integer")
        result[key] = item
    return result


class TaomateH3TeacherArtifact:
    r"""
    A completed Base10 teacher set consumed request by request.

    The artifact root holds a `complete.json` marker and one `request_XX.pt` file per stream request: a mapping
    with the request's `prompt` / `seed` / `audio_latent_count`, the contract keys (`teacher_state_numbers =
    (3, 6, 9)`, `stage3_target_state_indices = (16, 33, 49)`) and `milestones` — three
    `(2 * audio_latent_count, 32)` float32 tensors, the clean audio rows after teacher steps 3, 6 and 9. A producer
    may contain more requests than the current run; only the ordered prompt / seed prefix passed to [`open`] is
    consumed.
    """

    def __init__(
        self,
        root: Path,
        prompts: Tuple[str, ...],
        seeds: Tuple[int, ...],
        producer_geometry: Dict[str, int],
        artifact_request_count: int,
        audio_noise_seeds: Tuple[int, ...],
    ) -> None:
        self.root = root
        self.prompts = prompts
        self.seeds = seeds
        self.producer_geometry = producer_geometry
        self.artifact_request_count = artifact_request_count
        self.audio_noise_seeds = audio_noise_seeds

    @classmethod
    def open(
        cls,
        root: Union[str, Path],
        *,
        prompts: Sequence[str],
        seeds: Sequence[int],
        consumer_geometry: Dict[str, int],
    ) -> "TaomateH3TeacherArtifact":
        r"""Validate the artifact against the current run's prompts, seeds and canvas."""
        normalized_prompts = tuple(prompts)
        normalized_seeds = tuple(seeds)
        if not normalized_prompts or any(
            not isinstance(prompt, str) or not prompt.strip() for prompt in normalized_prompts
        ):
            raise TaomateH3TeacherError("teacher prompts must be non-empty strings")
        if len(normalized_seeds) != len(normalized_prompts) or any(
            type(seed) is not int or seed < 0 for seed in normalized_seeds
        ):
            raise TaomateH3TeacherError("one non-negative teacher seed is required per prompt")

        resolved = Path(root).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise TaomateH3TeacherError("Base10 teacher root must be a directory")
        completion = _taomate_h3_teacher_json(resolved / "complete.json", label="Base10 completion marker")

        request_count = completion.get("request_count")
        if type(request_count) is not int or request_count < len(normalized_prompts):
            raise TaomateH3TeacherError("Base10 artifact has fewer requests than the current run")
        if completion.get("base_precision") != "bf16":
            raise TaomateH3TeacherError("external Base10 teacher must use BF16 base weights")
        if (
            completion.get("strategy") != "previous_clean_audio_tail_reference_then_new_noise"
            or completion.get("partition") != "fl2va"
        ):
            raise TaomateH3TeacherError("Base10 teacher must use the FL2VA tail40 rollover contract")
        audio_noise_seed_sequence = completion.get("audio_noise_seed_sequence")
        consumed_audio_noise_seeds: Tuple[int, ...] = ()
        if (
            isinstance(audio_noise_seed_sequence, list)
            and len(audio_noise_seed_sequence) >= len(normalized_prompts)
        ):
            consumed = audio_noise_seed_sequence[: len(normalized_prompts)]
            if all(type(value) is int and value >= 0 for value in consumed):
                consumed_audio_noise_seeds = tuple(consumed)
        if len(consumed_audio_noise_seeds) != len(normalized_prompts):
            raise TaomateH3TeacherError("Base10 audio-noise seed sequence is invalid")

        producer_geometry = _taomate_h3_teacher_geometry(completion.get("producer_geometry"), label="producer geometry")
        expected_geometry = _taomate_h3_teacher_geometry(consumer_geometry, label="consumer geometry")
        if producer_geometry != expected_geometry:
            raise TaomateH3TeacherError("Base10 producer geometry differs from current inference geometry")

        return cls(
            root=resolved,
            prompts=normalized_prompts,
            seeds=normalized_seeds,
            producer_geometry=producer_geometry,
            artifact_request_count=request_count,
            audio_noise_seeds=consumed_audio_noise_seeds,
        )

    @property
    def request_count(self) -> int:
        return len(self.prompts)

    def audio_noise_seed(self, request_index: int) -> int:
        r"""The audio-noise seed of one request; the student must reuse it exactly."""
        return self.audio_noise_seeds[request_index]

    def load_request(
        self, *, request_index: int, audio_latent_count: int, device: torch.device
    ) -> Dict[str, Any]:
        r"""Load one request's three float32 milestone tensors onto `device`."""
        if not 0 <= request_index < self.request_count:
            raise TaomateH3TeacherError("teacher request index is outside the current run")
        if type(audio_latent_count) is not int or audio_latent_count <= 0:
            raise TaomateH3TeacherError("audio latent count must be a positive integer")

        tensor_path = self.root / f"request_{request_index:02d}.pt"
        expected_request_values = {
            "prompt": self.prompts[request_index],
            "seed": self.seeds[request_index],
            "audio_latent_count": audio_latent_count,
        }
        if not tensor_path.is_file():
            raise TaomateH3TeacherError(f"Base10 request {request_index} tensor file is absent: {tensor_path}")
        try:
            payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise TaomateH3TeacherError(f"cannot load Base10 request {request_index} tensors: {exc}") from exc
        if not isinstance(payload, dict):
            raise TaomateH3TeacherError(f"Base10 request {request_index} tensor payload must be a mapping")
        if any(payload.get(key) != value for key, value in expected_request_values.items()):
            raise TaomateH3TeacherError(f"Base10 request {request_index} tensor metadata differs from current request")
        if tuple(payload.get("teacher_state_numbers", ())) != TAOMATE_H3_TEACHER_STATE_NUMBERS:
            raise TaomateH3TeacherError(
                f"Base10 request {request_index} must contain states {TAOMATE_H3_TEACHER_STATE_NUMBERS}"
            )
        if tuple(payload.get("stage3_target_state_indices", ())) != TAOMATE_H3_DISTILLED_STATE_INDICES[1:]:
            raise TaomateH3TeacherError(
                f"Base10 teacher states do not target Stage3 states {TAOMATE_H3_DISTILLED_STATE_INDICES[1:]}"
            )

        values = payload.get("milestones")
        if not isinstance(values, (list, tuple)) or len(values) != len(TAOMATE_H3_TEACHER_STATE_NUMBERS):
            raise TaomateH3TeacherError(
                f"Base10 request {request_index} must contain {len(TAOMATE_H3_TEACHER_STATE_NUMBERS)} milestones"
            )
        expected_shape = (2 * audio_latent_count, TAOMATE_H3_AUDIO_LATENT_CHANNELS)
        for stage_index, value in enumerate(values):
            if not isinstance(value, torch.Tensor):
                raise TaomateH3TeacherError(
                    f"Base10 request {request_index} milestone {stage_index} is not a tensor"
                )
            if tuple(value.shape) != expected_shape or value.dtype != torch.float32:
                raise TaomateH3TeacherError(
                    f"Base10 request {request_index} milestone {stage_index} must be float32 with shape "
                    f"{expected_shape}"
                )

        milestones = {
            index: value.to(device=device, dtype=torch.float32)
            for index, value in enumerate(values)
        }
        return {"milestones": milestones}

    def receipt(self, *, request_index: int, audio_latent_count: int) -> Dict[str, Any]:
        r"""The provenance receipt of one consumed teacher request."""
        return {
            "mode": "base10_fl2va_tail40_rollover",
            "model": "external_base_h3_bf16",
            "base_precision": "bf16",
            "artifact_storage_dtype": "float32",
            "producer_executed_forwards": 9,
            "full_state_count": 10,
            "captured_base_state_numbers": list(TAOMATE_H3_TEACHER_STATE_NUMBERS),
            "stage3_target_state_indices": list(TAOMATE_H3_DISTILLED_STATE_INDICES[1:]),
            "audio_latents_per_channel": audio_latent_count,
            "reference_latents_per_channel": 0 if request_index == 0 else TAOMATE_H3_ROLLOVER_REFERENCE_LATENTS,
            "prefix_source_request": None if request_index == 0 else request_index - 1,
            "audio_noise_seed": self.audio_noise_seed(request_index),
            "producer_geometry": dict(self.producer_geometry),
            "artifact_request_count": self.artifact_request_count,
            "consumed_request_count": self.request_count,
        }


# Per-chunk packed layout: a padless `[text | audio | video]` sequence laid out on the global rotary timeline.
def taomate_h3_phase_layout(
    text_token_tags: torch.Tensor,
    phase: TaomateH3StreamPhase,
    latent_height: int,
    latent_width: int,
    patch_size: Tuple[int, int, int],
    *,
    media_time_origin: int,
    video_latent_offset: int,
    audio_latent_offset: int,
) -> MiniMaxH3PackedSequence:
    r"""
    Build one future-free T2VA chunk on the global canonical timeline.

    The packed layout itself is the plain `[text | audio | video]` T2VA sequence of the chunk (no padding — the
    persistent KV cache spans whole rows, and single-GPU inference has no sequence-parallel alignment to satisfy).
    Its rotary time axis is then rewritten onto the *global* clock: the prompt rows are right-aligned so they end
    where the request's first video latent begins, the video rows sit at their global latent positions, and the
    audio rows at their global 40 Hz indices.
    """
    _, patch_h, patch_w = patch_size
    frame_rows = (latent_height // patch_h) * (latent_width // patch_w)
    layout = build_packed_sequence(
        text_token_tags,
        phase.video_latent_count,
        latent_height,
        latent_width,
        phase.audio_latent_count,
        patch_size,
    )
    position_ids = layout.position_ids
    text_len = int(layout.text_indices.shape[0])

    prompt_start = float(media_time_origin + taomate_h3_video_temporal_position(video_latent_offset) - text_len)
    position_ids[layout.text_indices, 0] = prompt_start + torch.arange(text_len, dtype=torch.float64)

    video_positions = taomate_h3_video_temporal_positions(
        video_latent_offset + phase.video_latent_start, phase.video_latent_count
    )
    video_times = torch.tensor(
        [float(media_time_origin + position) for position in video_positions], dtype=torch.float64
    )
    video_rows = layout.video_indices[layout.num_condition_video_rows :]
    position_ids[video_rows, 0] = video_times.repeat_interleave(frame_rows)

    audio_times = media_time_origin + torch.arange(
        audio_latent_offset + phase.audio_latent_start, audio_latent_offset + phase.audio_latent_stop,
        dtype=torch.float64,
    )
    audio_pairs = layout.audio_indices.view(2, phase.audio_latent_count)
    position_ids[audio_pairs[0], 0] = audio_times
    position_ids[audio_pairs[1], 0] = audio_times
    return layout


def taomate_h3_phase_initial_rows(
    initial_video_rows: torch.Tensor,
    initial_audio_rows: torch.Tensor,
    phase: TaomateH3StreamPhase,
    *,
    frame_rows: int,
    total_audio_latents: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Slice one phase's target noise rows out of the request's initial noise."""
    video = initial_video_rows[
        phase.video_latent_start * frame_rows : phase.video_latent_stop * frame_rows
    ].contiguous()
    audio = initial_audio_rows.view(2, total_audio_latents, -1)[
        :, phase.audio_latent_start : phase.audio_latent_stop
    ].contiguous()
    return video, audio.view(-1, int(audio.shape[-1]))


def taomate_h3_teacher_phase_audio_rows(
    milestone: torch.Tensor, *, phase: TaomateH3StreamPhase, total_audio_latents: int
) -> torch.Tensor:
    r"""Slice the phase's rows out of a channel-major request-wide milestone."""
    return (
        milestone.view(2, total_audio_latents, -1)[:, phase.audio_latent_start : phase.audio_latent_stop]
        .contiguous()
        .view(-1, int(milestone.shape[-1]))
    )


def taomate_h3_join_phase_audio(rows: Sequence[torch.Tensor], phases: Sequence[TaomateH3StreamPhase]) -> torch.Tensor:
    r"""Stitch the per-phase audio rows back into one channel-major request tensor."""
    by_channel: Tuple[List[torch.Tensor], List[torch.Tensor]] = ([], [])
    for value, phase in zip(rows, phases):
        current = value.view(2, phase.audio_latent_count, -1)
        by_channel[0].append(current[0])
        by_channel[1].append(current[1])
    return torch.cat((torch.cat(by_channel[0], dim=0), torch.cat(by_channel[1], dim=0)), dim=0)


def taomate_h3_parse_resolution(value: str) -> Tuple[int, int, str]:
    r"""Parse a `WIDTHxHEIGHT` canvas: a 480 / 768 / 1088 short edge, 32-aligned."""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError("resolution must use WIDTHxHEIGHT, for example 480x864")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("resolution width and height must be integers") from exc
    if min(width, height) not in TAOMATE_H3_SUPPORTED_SHORT_EDGES or width % 32 or height % 32:
        raise ValueError("streaming inference requires a 480-, 768-, or 1088-pixel short edge and 32-pixel alignment")
    return width, height, f"{width}x{height}"


@dataclass
class MiniMaxH3StreamingPipelineOutput(BaseOutput):
    r"""
    Output of [`MiniMaxH3StreamingPipeline`].

    Args:
        videos (`torch.Tensor`, `np.ndarray` or `list[list[PIL.Image.Image]]`):
            The generated video over the whole stream, at 24 fps.
        audio (`torch.Tensor`):
            The generated soundtrack over the whole stream, of shape `(batch_size, 2, num_samples)`.
        sampling_rate (`int`):
            Sample rate of the soundtrack in Hz.
        request_receipts (`list[dict]`):
            One provenance receipt per stream request: the phase plan, the noise seeds, the transport prefix and
            the persistent-KV state.
    """

    videos: Any
    audio: Any
    sampling_rate: int
    request_receipts: List[Dict[str, Any]]


class _TaomateH3StreamSession:
    r"""The cross-request state of one streaming `__call__` run."""

    def __init__(self) -> None:
        self.media_time_origin: Optional[int] = None
        self.video_latent_offset = 0
        self.audio_latent_offset = 0
        self.renorm_anchor: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.transport_video_rows: Optional[torch.Tensor] = None
        self.transport_audio_rows: Optional[torch.Tensor] = None

    def renorm_clean_video_rows(self, rows: torch.Tensor) -> torch.Tensor:
        r"""Match generated chunks to the first chunk's per-row-frame statistics."""
        current = rows.detach().float()
        mean = current.mean(dim=0, keepdim=True)
        std = current.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        if self.renorm_anchor is None:
            self.renorm_anchor = (mean, std)
            return rows
        anchor_mean, anchor_std = self.renorm_anchor
        normalized = (current - mean).div(std).mul(anchor_std).add(anchor_mean)
        return normalized.to(dtype=rows.dtype)


class MiniMaxH3StreamingPipeline(MiniMaxH3Pipeline):
    r"""
    TaoMate-H3 streaming T2VA pipeline: any number of five-second stream requests denoised chunk by chunk with a
    persistent clean audio/video KV cache and a Base10 audio teacher, published through one VAE decode.

    Components, prompt encoding, latent packing and VAE decoding are shared with [`MiniMaxH3Pipeline`]. The
    streaming contract adds the persistent KV cache processor (installed for the duration of a call), the
    canonical chunk geometry and the teacher artifact. Single-GPU only.

    Examples:

    ```py
    import torch
    from videox_fun.pipeline import MiniMaxH3StreamingPipeline

    pipe = MiniMaxH3StreamingPipeline.from_pretrained("path/to/MiniMax-H3")
    pipe.to("cuda")
    output = pipe(
        prompt="a cat plays piano on a stage",
        width=480, height=864,
        request_count=4,
        seed=0,
        audio_teacher_dir="outputs/taomate_h3_teacher",
    )
    ```
    """

    def _check_streaming_inputs(
        self,
        prompts: List[str],
        request_count: int,
        seed: Union[int, Sequence[int]],
        height: int,
        width: int,
    ) -> Tuple[int, ...]:
        r"""Validate the streaming request contract; return the per-request seeds."""
        if request_count < 1:
            raise ValueError(f"`request_count` must be at least 1, got {request_count}.")
        if len(prompts) != request_count:
            raise ValueError(
                f"`prompt` must be one string or one string per request: got {len(prompts)} prompts for "
                f"request_count={request_count}."
            )
        for index, prompt in enumerate(prompts):
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"`prompt` item {index} must be a non-empty string.")
        if min(width, height) not in TAOMATE_H3_SUPPORTED_SHORT_EDGES or width % 32 or height % 32:
            raise ValueError(
                "streaming inference requires a 480-, 768-, or 1088-pixel short edge and 32-pixel alignment, "
                f"got {width}x{height}."
            )
        if isinstance(seed, int):
            seeds = (seed,) * request_count
        else:
            seeds = tuple(seed)
            if len(seeds) != request_count or any(type(item) is not int or item < 0 for item in seeds):
                raise ValueError("`seed` must be one non-negative integer or one per request.")
        return seeds

    def _stream_request(
        self,
        *,
        request_index: int,
        prompt: str,
        seed: int,
        teacher: TaomateH3TeacherArtifact,
        base_plan: TaomateH3StreamPlan,
        session: _TaomateH3StreamSession,
        cache: Any,
        sigmas_video: List[float],
        sigmas_audio: List[float],
        latent_height: int,
        latent_width: int,
        frame_rows: int,
        attention_kwargs: Optional[Dict[str, Any]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        r"""Generate one five-second stream request and advance the session state."""
        canonical_continuation = request_index > 0
        active_plan = (
            taomate_h3_canonical_continuation_plan(base_plan, request_index=request_index)
            if canonical_continuation
            else base_plan
        )
        official_video_latents = TAOMATE_H3_REQUEST_VIDEO_LATENTS
        official_audio_latents = TAOMATE_H3_REQUEST_AUDIO_LATENTS
        total_video_latents = active_plan.phases[-1].video_latent_stop
        total_audio_latents = active_plan.phases[-1].audio_latent_stop
        video_prefix_latents = official_video_latents - total_video_latents
        audio_prefix_latents = official_audio_latents - total_audio_latents
        phases = active_plan.phases

        audio_teacher = teacher.load_request(
            request_index=request_index,
            audio_latent_count=total_audio_latents,
            device=device,
        )
        audio_noise_seed = teacher.audio_noise_seed(request_index)
        video_noise_seed = (seed + request_index * TAOMATE_H3_VIDEO_NOISE_REQUEST_STRIDE) % (2**63)

        # The official noise order: the audio generator first draws (and discards) the full video noise — so audio
        # sampling depends on the resolution — then draws the audio rows; the video rows use their own request
        # substream. Both draws stay on the CPU.
        audio_generator = torch.Generator(device="cpu").manual_seed(audio_noise_seed)
        torch.randn(
            1,
            self.vae_latent_channels,
            official_video_latents,
            latent_height,
            latent_width,
            generator=audio_generator,
            dtype=torch.float32,
            device="cpu",
        )
        initial_audio = torch.randn(
            official_audio_latents * 2,
            self.audio_latent_channels,
            generator=audio_generator,
            dtype=torch.float32,
            device="cpu",
        )
        video_generator = torch.Generator(device="cpu").manual_seed(video_noise_seed)
        raw_video = torch.randn(
            1,
            self.vae_latent_channels,
            official_video_latents,
            latent_height,
            latent_width,
            generator=video_generator,
            dtype=torch.float32,
            device="cpu",
        )
        request_video_rows = patchify_video_latents(raw_video, self.patch_size)

        transport_video_rows = session.transport_video_rows
        transport_audio_rows = session.transport_audio_rows
        if canonical_continuation:
            if transport_video_rows is None or list(transport_video_rows.shape) != [
                video_prefix_latents * frame_rows, int(request_video_rows.shape[-1])
            ]:
                raise RuntimeError("canonical continuation has no exact prior video transport prefix")
            if transport_audio_rows is None:
                raise RuntimeError("canonical continuation has no prior audio transport prefix")
            previous_audio_latents = int(transport_audio_rows.shape[0]) // 2
            if previous_audio_latents < audio_prefix_latents:
                raise RuntimeError("canonical continuation has no exact prior audio transport prefix")
            # A continuation slices its own fresh noise down to the steady geometry; the prefix of the published
            # request is the previous request's tail, spliced back at the end.
            request_video_rows = request_video_rows[video_prefix_latents * frame_rows :].contiguous()
            initial_audio = (
                initial_audio.view(2, official_audio_latents, -1)[:, audio_prefix_latents:]
                .contiguous()
                .view(-1, int(initial_audio.shape[-1]))
            )

        prompt_embeds, text_token_tags = self.encode_prompt(prompt, device=device, dtype=self.transformer.dtype)
        text_len = int(text_token_tags.shape[0])
        media_time_origin = text_len if session.media_time_origin is None else session.media_time_origin

        audio_kv_reset_applied = request_index > 0 and request_index % TAOMATE_H3_AUDIO_KV_RESET_WINDOW_REQUESTS == 0
        audio_kv_reset_tokens = cache.drop_audio_history() if audio_kv_reset_applied else 0
        starting_history_tokens = cache.history_tokens

        # Re-arm both schedules per request: each phase runs the same three-step plan, and `set_timesteps` resets
        # the step indices.
        self.scheduler.set_timesteps(sigmas=sigmas_video, device=device)
        self.audio_scheduler.set_timesteps(sigmas=sigmas_audio, device=device)
        video_timesteps = self.scheduler.timesteps
        audio_timesteps = self.audio_scheduler.timesteps

        phase_video_outputs: List[torch.Tensor] = []
        phase_audio_outputs: List[torch.Tensor] = []
        for phase in phases:
            # The two schedules are shared across the four phases while `set_timesteps` runs only once per
            # request, so the step cursor would otherwise carry past the three-entry sigma grid. Re-arm it per
            # phase; the first `step()` re-locates it via the passed timestep.
            self.scheduler._step_index = None
            self.audio_scheduler._step_index = None
            layout = taomate_h3_phase_layout(
                text_token_tags,
                phase,
                latent_height,
                latent_width,
                self.patch_size,
                media_time_origin=media_time_origin,
                video_latent_offset=session.video_latent_offset,
                audio_latent_offset=session.audio_latent_offset,
            )
            position_ids = layout.position_ids.to(device)
            token_tags = layout.token_tags.to(device)
            video_indices = layout.video_indices.to(device)
            audio_indices = layout.audio_indices.to(device)
            text_indices = layout.text_indices.to(device)

            phase_video, phase_audio = taomate_h3_phase_initial_rows(
                request_video_rows, initial_audio, phase, frame_rows=frame_rows, total_audio_latents=total_audio_latents
            )
            phase_video = phase_video.to(device)
            phase_audio = phase_audio.to(device)

            def transformer_call(video_rows, audio_rows, unique_timesteps, timestep_indices):
                return self.transformer(
                    hidden_states=video_rows[None],
                    audio_hidden_states=audio_rows[None],
                    encoder_hidden_states=prompt_embeds,
                    timestep=unique_timesteps.to(video_rows.device),
                    timestep_indices=timestep_indices.to(video_rows.device),
                    token_tags=token_tags,
                    position_ids=position_ids,
                    video_indices=video_indices,
                    audio_indices=audio_indices,
                    text_indices=text_indices,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )

            # 1. Denoise this chunk in `len(video_timesteps)` steps. Text and video rows ride the video clock,
            # audio rows the audio clock; after every step the Base10 milestone replaces the audio rows.
            for step in range(len(video_timesteps)):
                unique_timesteps, timestep_indices = build_row_timesteps(
                    layout, float(video_timesteps[step]), float(audio_timesteps[step]),
                    float(video_timesteps[step]), float(audio_timesteps[step]),
                )
                cache.begin_live(token_tags)
                video_velocity, audio_velocity = transformer_call(
                    phase_video, phase_audio, unique_timesteps, timestep_indices
                )
                cache.end_forward()
                phase_video = self.scheduler.step(
                    video_velocity[0].float(), video_timesteps[step], phase_video, return_dict=False
                )[0]
                phase_audio = self.audio_scheduler.step(
                    audio_velocity[0].float(), audio_timesteps[step], phase_audio, return_dict=False
                )[0]
                milestone = audio_teacher["milestones"].get(step)
                if milestone is not None:
                    phase_audio.copy_(
                        taomate_h3_teacher_phase_audio_rows(
                            milestone, phase=phase, total_audio_latents=total_audio_latents
                        )
                    )

            # 2. Renormalize the clean video rows to the first chunk's stats.
            phase_video = session.renorm_clean_video_rows(phase_video)

            # 3. Commit the chunk's clean audio/video K/V to the persistent history. The forward itself is
            # output-free: only the K/V the processors stage on the way through matters.
            commit_mask = (token_tags == MINIMAX_H3_VIDEO_TAG) | (token_tags == MINIMAX_H3_AUDIO_TAG)
            unique_timesteps, timestep_indices = build_row_timesteps(layout, 1.0, 1.0, 1.0, 1.0)
            cache.begin_clean_commit(token_tags, commit_mask)
            transformer_call(phase_video, phase_audio, unique_timesteps, timestep_indices)
            cache.commit()
            cache.retain_sink_and_recent_commits()

            phase_video_outputs.append(phase_video)
            phase_audio_outputs.append(phase_audio)

        joined_video = torch.cat(phase_video_outputs, dim=0)
        joined_audio = taomate_h3_join_phase_audio(phase_audio_outputs, phases)
        if list(joined_video.shape) != [total_video_latents * frame_rows, int(request_video_rows.shape[-1])]:
            raise RuntimeError("reassembled streaming video rows changed request shape")
        if list(joined_audio.shape) != [2 * total_audio_latents, self.audio_latent_channels]:
            raise RuntimeError("reassembled streaming audio rows changed request shape")
        # The published soundtrack *is* the teacher's final clean audio state.
        final_teacher = audio_teacher["milestones"][len(TAOMATE_H3_TEACHER_STATE_NUMBERS) - 1]
        if not torch.equal(joined_audio, final_teacher):
            raise RuntimeError("published streaming audio differs from Base teacher clean latent")

        # 4. Save the transport prefix the next request splices behind: the last two video latents and the last
        # (up to) nine audio latents.
        session.transport_video_rows = (
            joined_video[-TAOMATE_H3_VIDEO_PREFIX_LATENTS * frame_rows :]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        session.transport_audio_rows = (
            joined_audio.view(2, total_audio_latents, -1)[
                :, -min(TAOMATE_H3_AUDIO_TRANSPORT_PREFIX_LATENTS, total_audio_latents) :
            ]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .view(-1, int(joined_audio.shape[-1]))
        )
        session.media_time_origin = media_time_origin

        returned_video = joined_video
        returned_audio = joined_audio
        if canonical_continuation:
            returned_video = torch.cat(
                (
                    transport_video_rows.to(device=joined_video.device, dtype=joined_video.dtype),
                    joined_video,
                ),
                dim=0,
            )
            previous_audio = transport_audio_rows.view(
                2, -1, int(joined_audio.shape[-1])
            )[:, -audio_prefix_latents:]
            returned_audio = (
                torch.cat(
                    (
                        previous_audio.to(device=joined_audio.device, dtype=joined_audio.dtype),
                        joined_audio.view(2, total_audio_latents, int(joined_audio.shape[-1])),
                    ),
                    dim=1,
                )
                .contiguous()
                .view(-1, int(joined_audio.shape[-1]))
            )
        if list(returned_video.shape) != [official_video_latents * frame_rows, int(request_video_rows.shape[-1])]:
            raise RuntimeError("streaming output no longer matches official request rows")
        if list(returned_audio.shape) != [2 * official_audio_latents, self.audio_latent_channels]:
            raise RuntimeError("streaming audio output no longer matches official request rows")

        session.video_latent_offset += total_video_latents
        session.audio_latent_offset += total_audio_latents

        receipt: Dict[str, Any] = {
            "request_index": request_index,
            "prompt": prompt,
            "seed": seed,
            "video_noise_seed": video_noise_seed,
            "audio_noise_seed": audio_noise_seed,
            "canonical_continuation": canonical_continuation,
            "plan": active_plan.to_dict(),
            "frame_rows": frame_rows,
            "video_transport_prefix_latents": video_prefix_latents,
            "audio_transport_prefix_latents_per_channel": audio_prefix_latents,
            "published_video_latents": total_video_latents,
            "published_audio_latents_per_channel": total_audio_latents,
            "text_token_count": text_len,
            "media_time_origin": media_time_origin,
            "starting_history_tokens": starting_history_tokens,
            "retained_history_tokens": cache.history_tokens,
            "retained_history_audio_tokens": cache.history_audio_tokens,
            "retained_history_video_tokens": cache.history_video_tokens,
            "audio_kv_reset_applied": audio_kv_reset_applied,
            "audio_kv_reset_tokens": audio_kv_reset_tokens,
            "denoise_forwards_per_phase": len(video_timesteps),
            "clean_forwards_per_phase": 1,
            "teacher": teacher.receipt(request_index=request_index, audio_latent_count=total_audio_latents),
        }
        return returned_video, returned_audio, receipt

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        audio_teacher_dir: Union[str, Path],
        height: int = 864,
        width: int = 480,
        request_count: int = 1,
        seed: Union[int, Sequence[int]] = 0,
        output_type: str = "pt",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_request_end: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ):
        r"""
        Generate a continuous video with its soundtrack, five seconds at a time.

        Args:
            prompt (`str` or `list[str]`):
                One prompt per stream request; a single string repeats over every request. Each prompt is
                right-aligned on the shared rotary timeline.
            audio_teacher_dir (`str` or `os.PathLike`):
                The offline Base10 audio-teacher artifact directory (see
                `scripts/taomate_h3/generate_base10_teacher.py`). The soundtrack *is* the teacher's clean audio;
                the artifact must match the prompts, seeds and canvas.
            height (`int`, defaults to `864`):
                Canvas height in pixels. The short edge must be 480, 768 or 1088; both edges 32-aligned.
            width (`int`, defaults to `480`):
                Canvas width in pixels.
            request_count (`int`, defaults to `1`):
                How many five-second stream requests to generate. The first one runs the direct 124-frame plan,
                every following one the canonical 119-frame continuation spliced behind its predecessor.
            seed (`int` or `list[int]`, defaults to `0`):
                The authored seed(s). Request `i` draws its video noise from `seed + i * 1000003`; the audio
                noise seeds come from the teacher artifact.
            output_type (`str`, defaults to `"pt"`):
                Output format: `"pil"`, `"np"`, `"pt"`, or `"latent"` for the denormalized latents.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`MiniMaxH3StreamingPipelineOutput`].
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that, if specified, may carry a `scale` entry which is applied to the LoRA layers.
            callback_on_request_end (`Callable`, *optional*):
                A function called as `callback(request_index, receipt)` once a stream request finishes.

        Returns:
            [`MiniMaxH3StreamingPipelineOutput`] or `tuple`:
                The generated video and the stereo soundtrack of the spliced timeline, its sample rate, and one
                receipt per stream request.
        """
        prompts = [prompt] * request_count if isinstance(prompt, str) else list(prompt)
        seeds = self._check_streaming_inputs(prompts, request_count, seed, height, width)
        self._attention_kwargs = attention_kwargs
        device = self._execution_device

        sp_world_size = int(getattr(self.transformer, "sp_world_size", 1) or 1)
        if sp_world_size > 1:
            raise ValueError(
                "TaoMate-H3 streaming inference keeps whole-sequence K/V on one device and requires "
                f"single-GPU inference, got sp_world_size={sp_world_size}."
            )

        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        _, patch_h, patch_w = self.patch_size
        frame_rows = (latent_height // patch_h) * (latent_width // patch_w)

        teacher = TaomateH3TeacherArtifact.open(
            audio_teacher_dir,
            prompts=prompts,
            seeds=seeds,
            consumer_geometry=taomate_h3_teacher_geometry(
                width, height, video_latent_h=latent_height, video_latent_w=latent_width
            ),
        )
        sigmas_video, sigmas_audio = taomate_h3_streaming_sigma_schedules()
        base_plan = taomate_h3_direct_5s_plan()

        cache = install_minimax_h3_streaming_kv_cache(self.transformer, dtype=self.transformer.dtype)
        session = _TaomateH3StreamSession()
        try:
            video_segments: List[torch.Tensor] = []
            audio_segments: List[torch.Tensor] = []
            video_prefix_latents: List[int] = []
            audio_prefix_latents: List[int] = []
            request_receipts: List[Dict[str, Any]] = []
            with self.progress_bar(total=request_count * len(base_plan.phases)) as progress_bar:
                for request_index in range(request_count):
                    returned_video, returned_audio, receipt = self._stream_request(
                        request_index=request_index,
                        prompt=prompts[request_index],
                        seed=seeds[request_index],
                        teacher=teacher,
                        base_plan=base_plan,
                        session=session,
                        cache=cache,
                        sigmas_video=sigmas_video,
                        sigmas_audio=sigmas_audio,
                        latent_height=latent_height,
                        latent_width=latent_width,
                        frame_rows=frame_rows,
                        attention_kwargs=attention_kwargs,
                        device=device,
                    )
                    video_segments.append(returned_video)
                    audio_segments.append(returned_audio)
                    video_prefix_latents.append(receipt["video_transport_prefix_latents"])
                    audio_prefix_latents.append(receipt["audio_transport_prefix_latents_per_channel"])
                    request_receipts.append(receipt)
                    for _ in range(len(base_plan.phases)):
                        progress_bar.update()
                    if callback_on_request_end is not None:
                        callback_on_request_end(request_index, receipt)
        finally:
            uninstall_minimax_h3_streaming_kv_cache(self.transformer)
            cache.clear()

        # One-shot publication: splice the requests (dropping the transport prefix each continuation repeats) and
        # decode the full timeline once.
        video_rows = torch.cat(
            [
                segment if index == 0 else segment[video_prefix_latents[index] * frame_rows :]
                for index, segment in enumerate(video_segments)
            ],
            dim=0,
        )
        audio_rows = torch.cat(
            [
                segment if index == 0 else segment[2 * audio_prefix_latents[index] :]
                for index, segment in enumerate(audio_segments)
            ],
            dim=0,
        )
        total_video_latents = int(video_rows.shape[0]) // frame_rows
        total_audio_latents = int(audio_rows.shape[0]) // 2
        # The official runtime ships the video VAE with tile-parallel decoding enabled; a full-timeline decode
        # without tiling holds several ~40 GiB intermediate feature maps and cannot fit on one 80 GiB card next
        # to anything else. Split the canvas exactly as the checkpoint expects.
        self.vae.enable_tiling()
        videos = self.decode_latents(video_rows, 0, total_video_latents, latent_height, latent_width, output_type)
        audio = self.decode_audio_latents(audio_rows, 0, total_audio_latents, output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (videos, audio, self.audio_sampling_rate, request_receipts)
        return MiniMaxH3StreamingPipelineOutput(
            videos=videos,
            audio=audio,
            sampling_rate=self.audio_sampling_rate,
            request_receipts=request_receipts,
        )
