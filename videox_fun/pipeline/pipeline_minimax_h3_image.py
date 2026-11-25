# Modified from the MiniMax-H3-Image modular blocks of
# https://github.com/huggingface/diffusers/tree/main/src/diffusers/modular_pipelines/minimax_h3_image
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

r"""MiniMax-H3-Image text-to-image and image-editing pipeline.

MiniMax-H3-Image is MiniMax-H3 without the audio branch: the same transformer denoises **one packed sequence** that
holds the text conditioning, the rows of the reference images and the rows of the generated image — which is a single
latent frame — with full self-attention over all of it. The row order is

```
[ text (L) | reference images (R) | generated image (V) ]
```

and, as in the video model, it is semantic: it labels the references in the prompt presentation and it lays the
rotary clock out, so a different order is a different request. The layout container, the patch packing and the
per-row timestep logic are therefore reused from [`pipeline_minimax_h3`]; only the image-specific placement
(`build_image_packed_sequence`, where every image is one frame on a clock that the text rows also occupy) is added
here.

The default is the original single output projection, with an experimental configurable step count and sigma shift.
Explicit PDD mode uses the released fixed six-step schedule. The checkpoint is guidance-distilled in both cases;
the pipeline does not expose CFG or a negative prompt.
"""

import inspect
import math
from numbers import Integral
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from ..models import (AutoencoderKLMiniMaxH3Image,
                      MiniMaxH3ImageTransformer2DModel)
from ..utils import MiniMaxH3Scheduler
from ..utils.lora_utils_pdd import pdd_head_timestep_grid
from .pipeline_minimax_h3 import (MINIMAX_H3_KEYFRAME_ENCODE_SEED,
                                  MINIMAX_H3_KEYFRAME_NOISE_AUG,
                                  MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD,
                                  MiniMaxH3PackedSequence, MiniMaxH3Pipeline,
                                  _frame_position_grid, _offload_scope,
                                  build_row_timesteps, logger,
                                  normalize_reference_image,
                                  patchify_video_latents,
                                  ref2va_condition_rows,
                                  unpatchify_video_tokens)

# An image row is tagged exactly like a video row: the AdaLN table of the released checkpoint keeps the three-way
# `(timestep, modality)` layout of MiniMax-H3, with image rows in the video slot and no audio rows at all.
MINIMAX_H3_IMAGE_TAG = 0

# The sampling grid the checkpoint's six output heads were distilled on, terminal `0` included, i.e. the `sigmas` of
# the partition's `pdd_config.json`. One model evaluation per head, in this order.
MINIMAX_H3_IMAGE_SAMPLE_SIGMAS = (
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
)

def head_timestep_grid(num_pdd_heads: int) -> torch.Tensor:
    """Training nodes for Image's released pre-fused PDD heads."""
    return pdd_head_timestep_grid(num_pdd_heads, MINIMAX_H3_IMAGE_SAMPLE_SIGMAS)


# How many tokens of the prompt the release keeps: it is used as given, truncated to 4096 tokens. The cap is on the
# prompt alone, not on the presentation it ends up in — a reference is stamped ahead of it as a vision block of
# thousands of rows of its own.
MINIMAX_H3_IMAGE_MAX_PROMPT_TOKENS = 4096

# The image checkpoint documents one budget the video model does not share: an editing request may condition on up to
# 16 reference images.
MINIMAX_H3_IMAGE_MAX_REFERENCES = 16

# The smallest area, in pixels, an image is generated at: a smaller request is generated on a canvas scaled up to
# this area at the same aspect ratio and resized back after decoding.
MINIMAX_H3_IMAGE_MIN_GENERATION_AREA = 2048 * 2048

# The canvas a request without an explicit size is generated at: a square 1024, which the minimum generation area
# above then scales up to its own budget.
MINIMAX_H3_IMAGE_DEFAULT_SIZE = 1024


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def prepare_image_timesteps(scheduler, device, output_head_type, num_pdd_heads,
                            num_inference_steps=None, flow_shift=None):
    """Choose the ordinary single-head grid or the explicitly selected fixed PDD grid."""
    if output_head_type == "pdd":
        if num_inference_steps is not None and (
            isinstance(num_inference_steps, bool) or not isinstance(num_inference_steps, Integral)
            or int(num_inference_steps) != len(MINIMAX_H3_IMAGE_SAMPLE_SIGMAS) - 1
        ):
            raise ValueError("The released PDD image model only supports its fixed six-step grid")
        if flow_shift is not None:
            raise ValueError("flow_shift cannot be changed for the fixed PDD image schedule")
        timesteps, _ = retrieve_timesteps(scheduler, sigmas=list(MINIMAX_H3_IMAGE_SAMPLE_SIGMAS), device=device)
        if len(timesteps) != num_pdd_heads:
            raise ValueError(f"PDD schedule has {len(timesteps)} evaluations but the model has {num_pdd_heads} heads")
        return timesteps
    if output_head_type != "single" or num_pdd_heads != 1:
        raise ValueError(f"Invalid output head configuration: {output_head_type!r}, {num_pdd_heads} heads")
    steps = 40 if num_inference_steps is None else num_inference_steps
    if isinstance(steps, bool) or not isinstance(steps, Integral) or int(steps) < 1:
        raise ValueError("num_inference_steps must be a positive integer")
    shift = 12.0 if flow_shift is None else float(flow_shift)
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("flow_shift must be a finite positive number")
    if not hasattr(scheduler, "set_shift"):
        raise ValueError("Single-head sampling requires a scheduler supporting set_shift")
    scheduler.set_shift(shift)
    # MiniMaxH3Scheduler counts sigma points including terminal zero, so request N+1 for N forwards.
    timesteps, _ = retrieve_timesteps(scheduler, num_inference_steps=int(steps) + 1, device=device)
    if len(timesteps) != int(steps):
        raise ValueError(f"Requested {steps} forwards but scheduler produced {len(timesteps)}")
    return timesteps


class MiniMaxH3ImagePipelineOutput(BaseOutput):
    r"""
    Output of [`MiniMaxH3ImagePipeline`].

    Args:
        images (`torch.Tensor`, `np.ndarray` or `list[PIL.Image.Image]`):
            The generated images.
    """

    images: Union[torch.Tensor, np.ndarray]


def get_image_canvas_size(height: int, width: int, min_generation_area: int, multiple: int) -> Tuple[int, int]:
    r"""
    The canvas MiniMax-H3-Image generates a `height x width` image on.

    An image below `min_generation_area` pixels is generated larger at the same aspect ratio: both sides are scaled
    by `sqrt(min_generation_area / (height * width))`. Either way both sides round to the nearest `multiple`.

    Returns:
        `tuple[int, int]`: the `(height, width)` of the canvas.
    """
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def build_image_packed_sequence(
    text_token_tags: torch.Tensor,
    reference_grids: List[Tuple[int, int]],
    latent_height: int,
    latent_width: int,
    patch_size: Tuple[int, int, int],
) -> MiniMaxH3PackedSequence:
    r"""
    Build the `[text | reference images | generated image]` layout of MiniMax-H3-Image.

    Text rows sit on the rotary time axis at their row index and every image is one frame on the same clock, which
    starts where the text rows end: a reference takes one integer slot each and the generated image takes the slot
    after the last reference. The video layout of [`build_packed_sequence`] shares the rule for its keyframe blocks,
    but advances the clock by a video's temporal span and interleaves audio rows, so the image placement is its own
    function; the `(h, w)` grid every image block is stamped with is the shared `_frame_position_grid`.

    The layout comes back as a [`MiniMaxH3PackedSequence`] with an empty audio block, which is what lets
    [`build_row_timesteps`] and the pipeline reuse the video pipeline's per-row timestep logic verbatim:
    `video_indices` are the image rows, reference rows first, and `num_condition_video_rows` their count.

    Args:
        text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
            The modality tag of every text row. Text is tagged `1`, except for the rows of a reference's vision
            block, which are tagged `0` (image).
        reference_grids (`list[tuple[int, int]]`):
            The `(latent_height, latent_width)` of every reference image, in packed order. Empty for `t2i`.
        latent_height (`int`): Latent height of the generated image.
        latent_width (`int`): Latent width of the generated image.
        patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.

    Returns:
        [`MiniMaxH3PackedSequence`]
    """
    _, patch_h, patch_w = patch_size
    num_text_tokens = text_token_tags.shape[0]
    num_reference_rows = sum((height // patch_h) * (width // patch_w) for height, width in reference_grids)
    num_target_rows = (latent_height // patch_h) * (latent_width // patch_w)
    sequence_length = num_text_tokens + num_reference_rows + num_target_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    # Every image is one frame on the rotary clock, which starts where the text rows end.
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for height, width in reference_grids:
        rows = slice(cursor, cursor + (height // patch_h) * (width // patch_w))
        cursor = rows.stop
        frame_grid, _ = _frame_position_grid(height, width, patch_h, patch_w)
        position_ids[rows, 0] = rotary_time
        position_ids[rows, 1:] = frame_grid
        rotary_time += 1.0
    frame_grid, _ = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    position_ids[cursor:, 0] = rotary_time
    position_ids[cursor:, 1:] = frame_grid

    text_indices = torch.arange(num_text_tokens)
    image_indices = torch.arange(num_text_tokens, sequence_length)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[image_indices] = MINIMAX_H3_IMAGE_TAG

    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=image_indices,
        audio_indices=torch.arange(0, dtype=torch.long),
        text_indices=text_indices,
        num_condition_video_rows=num_reference_rows,
        num_condition_audio_rows=0,
    )


class MiniMaxH3ImagePipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation and editing with MiniMax-H3-Image, covering the `t2i` (text only) and `edit`
    (reference images) tasks.

    Args:
        vae ([`AutoencoderKLMiniMaxH3Image`]):
            The image autoencoder. Its latents are normalized with `latents_mean` / `latents_std` and its pixels are
            ImageNet-normalized RGB over a `[0, 1]` base range.
        text_encoder ([`Qwen3VLForConditionalGeneration`]):
            The conditioner. MiniMax-H3 reads the *unnormalized* hidden state after its 50th decoder layer and never
            uses the language-model head.
        tokenizer ([`Qwen2TokenizerFast`]):
            Tokenizer of the conditioner.
        processor ([`Qwen3VLProcessor`]):
            Processor of the conditioner, used for the vision blocks of the reference images.
        transformer ([`MiniMaxH3ImageTransformer2DModel`]):
            The denoiser of the packed sequence, with one output head per step of
            [`MINIMAX_H3_IMAGE_SAMPLE_SIGMAS`].
        scheduler ([`MiniMaxH3Scheduler`]):
            The rectified-flow Euler schedule, only ever set to the checkpoint's fixed sampling grid.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def _transformer_is_fsdp(self):
        r"""
        True when sequence-parallel inference wrapped the denoiser in FSDP, in which case the whole-model offload
        entry points below must not put accelerate hooks on the shards. Same contract as
        [`MiniMaxH3Pipeline._transformer_is_fsdp`].
        """
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except Exception:
            return False
        return isinstance(getattr(self, "transformer", None), FSDP)

    def _fsdp_offload_components(self, device):
        from ..dist.fsdp import offload_components_cpu
        names = [n for n in ("vae",) if isinstance(getattr(self, n, None), torch.nn.Module)]
        return offload_components_cpu(self, device, names)

    def enable_model_cpu_offload(self, gpu_id=None, device="cuda"):
        if self._transformer_is_fsdp():
            return self._fsdp_offload_components(device)
        return super().enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    def enable_sequential_cpu_offload(self, gpu_id=None, device="cuda"):
        if self._transformer_is_fsdp():
            return self._fsdp_offload_components(device)
        return super().enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)

    def __init__(
        self,
        vae: AutoencoderKLMiniMaxH3Image,
        text_encoder,
        tokenizer,
        processor,
        transformer: MiniMaxH3ImageTransformer2DModel,
        scheduler: MiniMaxH3Scheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
        )
        # The image VAE decodes into ImageNet-normalized RGB over a [0, 1] base range, which this pipeline reverts
        # itself, so the processor must not denormalize a second time. It is only ever used to resize a canvas image
        # back to the size that was requested.
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_spatial_compression_ratio, do_normalize=False
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""Load a [`MiniMaxH3ImagePipeline`] from a MiniMax-H3-Image snapshot.

        Components are assembled locally because the released root `model_index.json` names native classes.
        An original `transformer/` is preferred and streamed through the native loader; otherwise the converted
        `transformer_diffusers/` is used. A non-local path (a repo id) is left to diffusers' own wiring.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                A local MiniMax-H3-Image snapshot folder or a repo id.
            torch_dtype (`torch.dtype`, *optional*):
                Dtype of the transformer and the text encoder; `None` keeps the released bfloat16. The VAE always
                stays float32 as released, regardless of this argument.
            output_head_type (`str`, defaults to `"single"`): Select `"pdd"` to use the released six PDD heads.
        """
        import os

        path = pretrained_model_name_or_path
        torch_dtype = kwargs.pop("torch_dtype", kwargs.pop("dtype", None))
        output_head_type = kwargs.pop("output_head_type", "single")

        if not os.path.isdir(path):
            if output_head_type != "single":
                raise ValueError("Explicit Image output-head selection requires a local model snapshot")
            return super().from_pretrained(path, torch_dtype=torch_dtype, **kwargs)

        from ..models import (Qwen2TokenizerFast,
                              Qwen3VLForConditionalGeneration,
                              Qwen3VLProcessor)
        from ..models.minimax_h3_image_transformer2d import resolve_image_transformer_component

        def _folder(*names):
            for name in names:
                folder = os.path.join(os.fspath(path), name)
                if os.path.isdir(folder):
                    return folder
            raise FileNotFoundError(
                f"None of {names} was found under {path}; expected a MiniMax-H3-Image snapshot with "
                f"`transformer_diffusers` (or `transformer`), `vae_diffusers` (or `vae`), `text_encoder`, "
                f"`tokenizer`, `processor` and `scheduler`."
            )

        transformer = MiniMaxH3ImageTransformer2DModel.from_pretrained(
            resolve_image_transformer_component(path), torch_dtype=torch_dtype, low_cpu_mem_usage=True,
            output_head_type=output_head_type,
        )
        # The VAE stays float32 as released, so it is loaded without `torch_dtype`; the mixed-precision loader mixin
        # restores the pinned fp32 modules anyway.
        vae = AutoencoderKLMiniMaxH3Image.from_pretrained(_folder("vae_diffusers", "vae"))
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            _folder("text_encoder"), low_cpu_mem_usage=True, torch_dtype=torch_dtype
        ).eval()
        tokenizer = Qwen2TokenizerFast.from_pretrained(_folder("tokenizer"))
        processor = Qwen3VLProcessor.from_pretrained(_folder("processor"))
        scheduler = MiniMaxH3Scheduler.from_pretrained(_folder("scheduler"))

        return cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
        )

    @property
    def _execution_device(self):
        r"""
        The device the pipeline executes on, pinned to a CUDA component when the registered order does not yield one.
        Same contract as [`MiniMaxH3Pipeline._execution_device`]: under `--fsdp_offload_params` the transformer keeps
        its flat param on the CPU at rest, so resolving through `self.device` can come back `cpu` nondeterministically
        and feed CPU tensors to GPU-resident params.
        """
        device = super()._execution_device
        if device.type != "cpu":
            return device
        for model in self.components.values():
            if not isinstance(model, torch.nn.Module):
                continue
            model_device = getattr(model, "device", None)
            if model_device is not None and model_device.type != "cpu":
                return model_device
        return device

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
    def patch_size(self) -> Tuple[int, int, int]:
        if getattr(self, "transformer", None) is not None:
            return tuple(self.transformer.config.patch_size)
        return (1, 2, 2)

    @property
    def canvas_multiple(self) -> int:
        r"""
        What the generated height and width have to be a multiple of, 32 for the released checkpoint. A canvas has to
        survive the VAE's spatial compression and still be a whole number of patch rows wide, so the multiple is the
        product of the two.
        """
        return self.vae_spatial_compression_ratio * self.patch_size[2]

    @property
    def num_pdd_heads(self) -> int:
        r"""Number of sampling steps the transformer was distilled to, i.e. of output heads."""
        if getattr(self, "transformer", None) is not None:
            return int(self.transformer.config.num_pdd_heads)
        return 1

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    def check_inputs(self, prompt, height, width, min_generation_area):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(
                "MiniMax-H3-Image packs one request into one sequence, so `prompt` must be a single non-empty "
                f"string, got {prompt!r}."
            )
        if height <= 0 or width <= 0:
            raise ValueError(f"`height` and `width` must be positive, got {height}x{width}.")
        if not 1024 <= min_generation_area <= 4096 * 4096:
            raise ValueError(
                f"`min_generation_area` must be within [1024, {4096 * 4096}], got {min_generation_area}."
            )

    def encode_prompt(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Build MiniMax-H3-Image's presentation of a request and encode it.

        The presentation is MiniMax-H3's: every reference image prepends a `"<Picture i>: "` label and a vision block
        (`<|vision_start|>`, one `<|image_pad|>` per merged vision patch, `<|vision_end|>`), tagged as *image* rows
        rather than text rows, and the prompt follows verbatim, with no chat template. The rows a request conditions
        on therefore come from the conditioner's own vision merge, at the reference resolutions, while the reference
        *pixels* additionally reach the transformer as VAE rows — see [`~MiniMaxH3ImagePipeline.encode_reference_images`].

        The video pipeline builds exactly this presentation for its keyframes, so it is reused verbatim rather than
        ported a second time: the reference images are handed to it as keyframes, which is the same thing in this
        model (single frames, image-tagged vision blocks). Only the prompt cap of the image release is applied here,
        and it is applied to the prompt alone: a reference's vision block costs thousands of rows ahead of the
        instruction, so truncating the assembled presentation would drop the instruction of every editing request.

        Args:
            prompt (`str`): The prompt to encode.
            images (`list[PIL.Image.Image]`, *optional*):
                The reference images, normalized by [`~MiniMaxH3ImagePipeline.__call__`], in packed order.
            device (`torch.device`, *optional*): The device to run the conditioner on.
            dtype (`torch.dtype`, *optional*): The dtype of the returned embeddings.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the `(1, num_text_tokens, 5120)` hidden states and the
            `(num_text_tokens,)` per-row modality tags.
        """
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"] if prompt else []
        if len(prompt_tokens) > MINIMAX_H3_IMAGE_MAX_PROMPT_TOKENS:
            logger.warning(
                f"The prompt holds {len(prompt_tokens)} tokens, over the "
                f"{MINIMAX_H3_IMAGE_MAX_PROMPT_TOKENS} the released model keeps; the trailing tokens are dropped."
            )
            prompt = self.tokenizer.decode(prompt_tokens[:MINIMAX_H3_IMAGE_MAX_PROMPT_TOKENS])

        return MiniMaxH3Pipeline.encode_prompt(self, prompt, images, None, device, dtype)

    def encode_reference_images(
        self, images: List[Image.Image], device: Optional[torch.device] = None
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        r"""
        Encode the editing references into conditioning latents, one single-frame latent per reference.

        The recipe is the video pipeline's for its keyframes and `ref2va` image references, kept bit for bit: the
        posterior is *sampled* under a generator seeded with 42 independently of the request seed, the sampled latent
        is rounded to float16 before it is normalized, and the normalization is
        `(latent - latents_mean) / latents_std`. The image VAE is a 2D model here, so a reference goes through the
        (tiled) encoder as `(1, 3, height, width)` and comes back as `(1, latent_channels, 1, latent_height,
        latent_width)` — the frame axis is the one latent frame its causal encoder produces from a single frame.

        Args:
            images (`list[PIL.Image.Image]`): The normalized reference images, in packed order.
            device (`torch.device`, *optional*): The device to run the VAE on.

        Returns:
            `tuple[list[torch.Tensor], list[tuple[int, int]]]`: one `(1, latent_channels, 1, latent_height,
            latent_width)` float32 CPU tensor per reference, and the `(latent_height, latent_width)` of every
            reference, which is what the packed layout is built from.
        """
        device = device or self._execution_device
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1)
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1)

        condition_latents, reference_grids = [], []
        # `_encode` is a method call, not the VAE's `forward`, so the top-level CPU-offload hook never fires around
        # it on its own: scope the whole encode, once, instead of per reference.
        with _offload_scope(self.vae):
            for image in images:
                pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None]
                pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
                # Tiling is on by default here, so a reference is encoded on its own overlapping-tile grid.
                posterior = DiagonalGaussianDistribution(self.vae._encode(pixels))
                latents = posterior.sample(generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
                # The sampled latent is rounded to float16 before it is normalized: ~11 bits of every conditioning
                # latent, so the released model's conditioning cannot be reproduced without it.
                latents = latents.to(torch.float16).float().cpu()
                latents = (latents - latents_mean) / latents_std
                condition_latents.append(latents[:, :, None])
                reference_grids.append((latents.shape[-2], latents.shape[-1]))
        return condition_latents, reference_grids

    def prepare_latents(
        self,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Draw the initial noise of the generated image and pack it into transformer rows.

        The draw is a `(1, latent_channels, 1, latent_height, latent_width)` latent tensor that is patchified
        afterwards, exactly as the video pipeline draws its target, so the row layout of a single frame is the same
        one a video frame gets. Passing `latents` skips the draw.

        Args:
            latent_height (`int`): Latent height of the generated image.
            latent_width (`int`): Latent width of the generated image.
            device (`torch.device`): The device the rows are drawn on.
            generator (`torch.Generator`, *optional*): The generator of the request.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noise of shape `(1, latent_channels, 1, latent_height, latent_width)`.

        Returns:
            `torch.Tensor` of shape `(num_rows, latent_channels * prod(patch_size))`: the packed noise rows.
        """
        if latents is None:
            latents = randn_tensor(
                (1, self.vae_latent_channels, 1, latent_height, latent_width),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        return patchify_video_latents(latents.to(torch.float32), self.patch_size).to(device)

    def decode_latents(
        self,
        latents: torch.Tensor,
        num_condition_rows: int,
        latent_height: int,
        latent_width: int,
        height: int,
        width: int,
        output_type: str = "pil",
    ):
        r"""
        Unpack the generated rows back into latents, denormalize them and decode them into an image.

        The decode runs under float16 autocast over float32 weights, as the video decode does, and the VAE produces
        ImageNet-normalized RGB that is reverted here. When the request was generated on a larger canvas, the decoded
        canvas is resized back to the size that was asked for with LANCZOS, which the released recipe does through
        PIL rather than through torch's interpolator.
        """
        device = self._execution_device
        image_latents = unpatchify_video_tokens(
            latents[num_condition_rows:], 1, latent_height, latent_width, self.vae_latent_channels, self.patch_size
        )
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1)
        image_latents = (image_latents[:, :, 0] * latents_std + latents_mean).float()

        if output_type == "latent":
            return image_latents

        # `decode` is reached as a method call, so the top-level CPU-offload hook is fired by hand around it.
        with _offload_scope(self.vae), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            image = self.vae.decode(image_latents, return_dict=False)[0]
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1)
        image = (image.float() * pixel_std + pixel_mean).clamp(0, 1)

        if (image.shape[-2], image.shape[-1]) != (height, width):
            image = self.image_processor.numpy_to_pil(self.image_processor.pt_to_numpy(image))
            image = [self.image_processor.resize(frame, height, width) for frame in image]
            if output_type == "pil":
                return image
            image = self.image_processor.pil_to_tensor(image).permute(0, 3, 1, 2).float() / 255.0
        return self.image_processor.postprocess(image, output_type=output_type)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str = None,
        images: Optional[List[Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        min_generation_area: int = MINIMAX_H3_IMAGE_MIN_GENERATION_AREA,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        num_inference_steps: Optional[int] = None,
        flow_shift: Optional[float] = None,
    ):
        r"""
        Generate an image from text, or edit one or more reference images.

        A request without `images` is a `t2i` request; with `images` it is an `edit` request, and the order of the
        references is part of the request: it labels them `<Picture 1>`, `<Picture 2>`, ... in the presentation and
        lays them out on the rotary clock in that order.

        The default single projection uses a configurable step count and sigma shift. When PDD is selected, the
        released model runs one fixed evaluation per head. Both modes run without CFG.

        Args:
            prompt (`str`):
                The prompt to guide generation, used as given and truncated to
                `MINIMAX_H3_IMAGE_MAX_PROMPT_TOKENS` tokens. MiniMax-H3-Image packs one request into one sequence,
                so a batch of prompts is not a thing.
            images (`list[PIL.Image.Image]`, *optional*):
                The reference images to edit, **in the order the model should read them**, at most
                `MINIMAX_H3_IMAGE_MAX_REFERENCES` of them. Each is normalized onto a short edge of its own, as in the
                video model's image references, whatever the requested size is.
            height (`int`, *optional*):
                Height of the generated image in pixels. Defaults to a square `MINIMAX_H3_IMAGE_DEFAULT_SIZE`.
            width (`int`, *optional*):
                Width of the generated image in pixels. The returned image has exactly this size.
            min_generation_area (`int`, defaults to `MINIMAX_H3_IMAGE_MIN_GENERATION_AREA`):
                The smallest area, in pixels, the image is generated at. A smaller request is generated on a canvas
                scaled up to this area at the same aspect ratio and resized back with LANCZOS; pass `1024` to turn
                the upscaling off.
            num_inference_steps (`int`, *optional*):
                Number of forwards in single-head mode (default 40). The released PDD mode remains fixed at 6.
            flow_shift (`float`, *optional*):
                Sigma shift in single-head mode (default 12); not used by the fixed released PDD grid.
            generator (`torch.Generator`, *optional*):
                The generator of the request. A request draws the reference conditioning noise first, one draw per
                reference, and the image noise after that, so the order is part of what it reproduces.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noise of shape `(1, 24, 1, latent_height, latent_width)`, used instead of the draw.
            output_type (`str`, defaults to `"pil"`): Output format: `"pil"`, `"np"`, `"pt"` or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`MiniMaxH3ImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that, if specified, may carry a `scale` entry which is applied to the LoRA layers.
            callback_on_step_end (`Callable`, *optional*): A function called at the end of every denoising step.
            callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["latents"]`):
                The tensors of the loop the callback is handed.

        Returns:
            [`MiniMaxH3ImagePipelineOutput`] or `tuple`: the generated images.
        """
        # 1. Resolve the plan: the canvas the request is generated on, which is the requested size scaled up to
        # `min_generation_area` at the same aspect ratio, and the references put onto their own resolutions.
        height = MINIMAX_H3_IMAGE_DEFAULT_SIZE if height is None else height
        width = MINIMAX_H3_IMAGE_DEFAULT_SIZE if width is None else width
        self.check_inputs(prompt, height, width, min_generation_area)
        self._attention_kwargs = attention_kwargs
        device = self._execution_device

        references = list(images) if images else []
        if len(references) > MINIMAX_H3_IMAGE_MAX_REFERENCES:
            raise ValueError(
                f"MiniMax-H3-Image accepts at most {MINIMAX_H3_IMAGE_MAX_REFERENCES} reference images, got "
                f"{len(references)}."
            )
        references = [normalize_reference_image(reference) for reference in references]

        canvas_height, canvas_width = get_image_canvas_size(
            height, width, min_generation_area, self.canvas_multiple
        )
        latent_height = canvas_height // self.vae_spatial_compression_ratio
        latent_width = canvas_width // self.vae_spatial_compression_ratio

        # 2. Encode the presentation of the request: the text rows, which hold the reference images' vision blocks
        # too, and their modality tags.
        prompt_embeds, text_token_tags = self.encode_prompt(
            prompt, references, device=device, dtype=self.transformer.dtype
        )

        # 3. Encode the references into conditioning latents and noise them to MiniMax-H3's conditioning level. The
        # conditioning noise is the request's *first* draw, ahead of the image noise, and the anchors are the whole
        # loop's invariant: the loop only ever writes the generated rows.
        condition_rows = None
        if references:
            condition_latents, reference_grids = self.encode_reference_images(references, device=device)
            condition_rows = ref2va_condition_rows(
                self.scheduler, condition_latents, self.patch_size, generator=generator, device=device
            )
        else:
            reference_grids = []

        # 4. Build the packed layout and its fp64 rotary grid.
        layout = build_image_packed_sequence(
            text_token_tags, reference_grids, latent_height, latent_width, self.patch_size
        )
        if condition_rows is not None and condition_rows.shape[0] != layout.num_condition_video_rows:
            raise ValueError(
                f"The layout reserved {layout.num_condition_video_rows} reference rows but the encoded references "
                f"pack into {condition_rows.shape[0]}. The references the layout was built from and the ones the "
                "conditioning was encoded from do not agree."
            )
        position_ids = layout.position_ids.to(device)
        token_tags = layout.token_tags.to(device)
        image_indices = layout.video_indices.to(device)
        text_indices = layout.text_indices.to(device)
        num_condition_rows = layout.num_condition_video_rows

        # 5. Draw the noise of the generated rows and prepend the conditioning rows.
        latents = self.prepare_latents(latent_height, latent_width, device, generator, latents)
        if condition_rows is not None:
            latents = torch.cat([condition_rows, latents])

        # 6. Stage the row-to-timestep plan of every step on the selected sampling grid. One forward serves
        # the generated rows and the reference rows at once: the generated rows step down the grid while the
        # reference rows stay pinned at their noise-augmentation level.
        output_head_type = getattr(self.transformer.config, "output_head_type", "single")
        timesteps = prepare_image_timesteps(
            self.scheduler, device, output_head_type, self.num_pdd_heads,
            num_inference_steps=num_inference_steps, flow_shift=flow_shift,
        )
        row_timestep_plan = [
            tuple(
                tensor.to(device)
                for tensor in build_row_timesteps(
                    layout,
                    float(timestep),
                    0.0,
                    max(float(timestep), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                    1.0,
                )
            )
            for timestep in timesteps
        ]

        # 7. Denoise with the shared projection or the selected PDD head.
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                unique_timesteps, timestep_indices = row_timestep_plan[i]
                noise_pred = self.transformer(
                    hidden_states=latents[None],
                    encoder_hidden_states=prompt_embeds,
                    timestep=unique_timesteps,
                    timestep_indices=timestep_indices,
                    token_tags=token_tags,
                    position_ids=position_ids,
                    image_indices=image_indices,
                    text_indices=text_indices,
                    pdd_step_index=0 if output_head_type == "single" else i,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                latents[num_condition_rows:] = self.scheduler.step(
                    noise_pred[0, num_condition_rows:].float(),
                    t,
                    latents[num_condition_rows:],
                    return_dict=False,
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for tensor_name in callback_on_step_end_tensor_inputs:
                        callback_kwargs[tensor_name] = locals()[tensor_name]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs) or {}
                    latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        # 8. Decode the generated rows, on the canvas, and put the result back at the requested size.
        images = self.decode_latents(
            latents, num_condition_rows, latent_height, latent_width, height, width, output_type
        )

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)
        return MiniMaxH3ImagePipelineOutput(images=images)
