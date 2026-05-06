# Modified from https://github.com/guandeh17/Self-Forcing/blob/main/pipeline/causal_diffusion_inference.py
import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from ..models import (AutoencoderKLWan, AutoTokenizer,
                              WanT5EncoderModel, WanTransformer3DModel)
from ..utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                get_sampling_sigmas)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""


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


@dataclass
class WanSelfForcingPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    videos: torch.Tensor


class WanSelfForcingPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: WanTransformer3DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Shape: [B, C, F, H, W] (standard PyTorch format)
        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.cpu().float().numpy()
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    def _initialize_kv_cache(self, batch_size, dtype, device, frame_seq_length):
        """
        Initialize KV cache for causal self-attention.
        """
        kv_cache_pos = []
        kv_cache_neg = []
        # Use the default KV cache size (32760 tokens for global attention)
        local_attn_size = getattr(self.transformer.config, 'local_attn_size', -1)
        if local_attn_size != -1:
            kv_cache_size = local_attn_size * frame_seq_length
        else:
            kv_cache_size = 32760
        
        num_heads = self.transformer.config.num_heads
        head_dim = self.transformer.config.dim // num_heads
        
        for _ in range(self.transformer.config.num_layers):
            kv_cache_pos.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_neg.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        
        self.kv_cache_pos = kv_cache_pos
        self.kv_cache_neg = kv_cache_neg

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize cross-attention cache.
        """
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        text_len = self.transformer.config.text_len
        num_heads = self.transformer.config.num_heads
        head_dim = self.transformer.config.dim // num_heads
        
        for _ in range(self.transformer.config.num_layers):
            crossattn_cache_pos.append({
                "k": torch.zeros([batch_size, text_len, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, text_len, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })
            crossattn_cache_neg.append({
                "k": torch.zeros([batch_size, text_len, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, text_len, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })
        
        self.crossattn_cache_pos = crossattn_cache_pos
        self.crossattn_cache_neg = crossattn_cache_neg

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        comfyui_progressbar: bool = False,
        shift: int = 5,
        initial_latent: Optional[torch.FloatTensor] = None,
        start_frame_index: int = 0,
        num_frame_per_block: int = 1,
        independent_first_frame: bool = True,
        context_noise: int = 0,
    ) -> Union[WanSelfForcingPipelineOutput, Tuple]:
        r"""
        Function invoked when calling the pipeline for Self-Forcing causal generation.
        
        Args:
            initial_latent: Optional initial latent frames for I2V/video extension.
                Shape: (batch_size, num_input_frames, channels, height, width)
            start_frame_index: Starting frame index for long video generation.
                Used when continuing generation from a previous segment.
            num_frame_per_block: Number of frames to generate per block.
            independent_first_frame: Whether to generate the first frame independently (T2V mode).
            context_noise: Context noise level for KV cache update (matches training config).
        
        Examples:
            ```python
            pass
            ```
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else:
            in_prompt_embeds = prompt_embeds

        # 4. Prepare timesteps
        self.scheduler.sigma_min = 0.0
        self.scheduler.config.shift_terminal = 0.625

        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler,
                device=device,
                sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 1)

        # 5. Prepare latents (noise) and output buffer separately
        latent_channels = self.transformer.config.in_channels
        
        # For I2V: num_frames_to_generate is frames to generate (not including input frames)
        num_frames_to_generate = num_frames
        if initial_latent is not None:
            # In I2V mode, num_frames is total frames, but noise should only be the new frames
            # VAE compression: num_latent_frames = (num_frames - 1) // temporal_compression + 1
            num_input_frames_temp = initial_latent.shape[2]  # [B, C, F, H, W]
            total_latent_frames = (num_frames - 1) // self.vae.temporal_compression_ratio + 1
            input_latent_frames = num_input_frames_temp
            num_frames_to_generate = total_latent_frames - input_latent_frames
        
        # Prepare noise (only for frames to generate)
        noise = self.prepare_latents(
            batch_size,
            latent_channels,
            num_frames_to_generate,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )
        
        # Calculate total output frames (input + generated)
        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0  # [B, C, F, H, W]
        num_output_frames = num_frames_to_generate + num_input_frames
        
        # Allocate output buffer: [B, C, F_total, H, W]
        output = torch.zeros_like(
            noise,
            device=device,
            dtype=weight_dtype
        )
        
        if comfyui_progressbar:
            pbar.update(1)

        # 6. Calculate sequence length and frame_seq_length
        target_shape = (
            self.vae.latent_channels,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            width // self.vae.spatial_compression_ratio,
            height // self.vae.spatial_compression_ratio,
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2])
            * target_shape[1]
        )
        
        # Calculate frame_seq_length: tokens per frame
        frame_seq_length = (target_shape[2] * target_shape[3]) // (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2])

        # 7. Causal generation loop - block by block
        # num_latent_frames is the number of frames after VAE compression
        num_latent_frames = target_shape[1]
        
        # Determine num_blocks based on mode (T2V vs I2V)
        # Reference: causal_inference.py line 70-78
        if not independent_first_frame or (independent_first_frame and initial_latent is not None):
            # I2V mode: even with independent_first_frame, if initial_latent is provided, frames should be divisible
            assert num_latent_frames % num_frame_per_block == 0, \
                f"num_latent_frames ({num_latent_frames}) must be divisible by num_frame_per_block ({num_frame_per_block})"
            num_blocks = num_latent_frames // num_frame_per_block
        else:
            # T2V mode: no initial_latent, use [1, 4, 4, ...] pattern
            assert (num_latent_frames - 1) % num_frame_per_block == 0, \
                f"num_latent_frames-1 ({num_latent_frames - 1}) must be divisible by num_frame_per_block ({num_frame_per_block})"
            num_blocks = (num_latent_frames - 1) // num_frame_per_block
        
        # Self-Forcing causal state (reset per call)
        current_start_frame = start_frame_index
        cache_start_frame = 0

        # 8. Initialize KV cache and cross-attention cache
        # Reset caches if they exist (for multiple inference calls)
        if self.kv_cache_pos is not None:
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=device)
                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=device)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=device)
            for block_index in range(len(self.crossattn_cache_pos)):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
        else:
            self._initialize_kv_cache(batch_size=batch_size, dtype=weight_dtype, device=device, frame_seq_length=frame_seq_length)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=weight_dtype, device=device)

        # Build all_num_frames list
        # Self-Forcing: T2V with independent_first_frame uses [1, 4, 4, 4, ...] pattern
        # I2V mode uses [4, 4, 4, ...] pattern (first frame is provided)
        all_num_frames = [num_frame_per_block] * num_blocks
        if independent_first_frame and initial_latent is None:
            # First frame is generated independently (standard Self-Forcing T2V pattern)
            all_num_frames = [1] + all_num_frames

        for block_idx, current_num_frames in enumerate(all_num_frames):
            # Extract noise for current block and convert to list format
            # noise only contains frames to generate, indexed from 0
            # current_start_frame tracks global position (including input frames for I2V)
            # Need to offset by num_input_frames to get index in noise
            start_idx = current_start_frame - num_input_frames
            end_idx = start_idx + current_num_frames
            noisy_input = noise[:, :, start_idx:end_idx]

            # Denoising loop for current block
            # Reset scheduler state for each block (required for causal generation)
            # For Euler scheduler, resetting _step_index is sufficient.
            # For multi-step schedulers (UniPC, DPM++), also clear accumulated model outputs.
            self.scheduler._step_index = None
            if hasattr(self.scheduler, 'model_outputs'):
                self.scheduler.model_outputs = []
            
            for step_idx, t in enumerate(timesteps):
                
                # Per-frame timesteps for causal generation
                timestep = torch.ones([batch_size, current_num_frames], device=device, dtype=torch.long) * t

                if do_classifier_free_guidance:
                    # Conditional path
                    with torch.cuda.amp.autocast(dtype=weight_dtype):
                        flow_pred_cond = self.transformer(
                            x=noisy_input,
                            context=prompt_embeds,
                            t=timestep,
                            seq_len=seq_len,
                            kv_cache=self.kv_cache_pos,
                            crossattn_cache=self.crossattn_cache_pos,
                            current_start=current_start_frame * frame_seq_length,
                            cache_start=None,
                        )
                    
                    # Unconditional path
                    with torch.cuda.amp.autocast(dtype=weight_dtype):
                        flow_pred_uncond = self.transformer(
                            x=noisy_input,
                            context=negative_prompt_embeds,
                            t=timestep,
                            seq_len=seq_len,
                            kv_cache=self.kv_cache_neg,
                            crossattn_cache=self.crossattn_cache_neg,
                            current_start=current_start_frame * frame_seq_length,
                            cache_start=None,
                        )
                    
                    # CFG guidance
                    # Transformer output shape check
                    if flow_pred_cond.dim() == 5:
                        # Already [B, C, F, H, W]
                        flow_pred = flow_pred_uncond + guidance_scale * (flow_pred_cond - flow_pred_uncond)
                    elif flow_pred_cond.dim() == 4:
                        # [F, C, H, W], need to add batch dim
                        flow_pred_cond = flow_pred_cond.unsqueeze(0).permute(0, 2, 1, 3, 4)
                        flow_pred_uncond = flow_pred_uncond.unsqueeze(0).permute(0, 2, 1, 3, 4)
                        flow_pred = flow_pred_uncond + guidance_scale * (flow_pred_cond - flow_pred_uncond)
                    else:
                        raise ValueError(f"Unexpected flow_pred_cond dim: {flow_pred_cond.dim()}, shape: {flow_pred_cond.shape}")
                else:
                    # Forward pass with KV cache
                    with torch.cuda.amp.autocast(dtype=weight_dtype):
                        flow_pred = self.transformer(
                            x=noisy_input,
                            context=in_prompt_embeds,
                            t=timestep,
                            seq_len=seq_len,
                            kv_cache=self.kv_cache_pos,
                            crossattn_cache=self.crossattn_cache_pos,
                            current_start=current_start_frame * frame_seq_length,
                            cache_start=None,
                        )

                    # Transformer output shape check
                    if flow_pred.dim() == 4:
                        # [F, C, H, W], need to add batch dim and permute
                        flow_pred = flow_pred.unsqueeze(0).permute(0, 2, 1, 3, 4)
                    # If already 5D [B, C, F, H, W], no need to permute
                
                # Get current sigma for x0 conversion
                sigma_t = self.scheduler.sigmas[step_idx]
                
                # Convert to x0: x0 = x_t - sigma_t * flow_pred (matches original wan_wrapper.py line 192)
                denoised_pred = noisy_input - sigma_t * flow_pred  # [B*F, C, H, W]
                
                if step_idx < len(timesteps) - 1:
                    # Not the last step: add noise for next timestep
                    next_t = timesteps[step_idx + 1]
                    
                    # Add noise using flow matching formula: x_{t+1} = (1-sigma_{t+1}) * x0 + sigma_{t+1} * noise
                    next_sigma = self.scheduler.sigmas[step_idx + 1]
                    local_noise = torch.randn(denoised_pred.shape, device=denoised_pred.device, dtype=denoised_pred.dtype, generator=generator)
                    noisy_input = (1 - next_sigma) * denoised_pred + next_sigma * local_noise
                else:
                    noisy_input = denoised_pred

            # Update output with denoised block
            output[:, :, cache_start_frame:cache_start_frame + current_num_frames] = denoised_pred

            # Update KV cache with clean context (timestep=context_noise) for next block
            # Reference: causal_inference.py line 227 - uses context_noise for KV cache update
            if block_idx < len(all_num_frames) - 1:
                context_timestep = torch.ones([batch_size, current_num_frames], device=device, dtype=torch.long) * context_noise
                
                if do_classifier_free_guidance:
                    # Update both positive and negative caches
                    with torch.cuda.amp.autocast(dtype=weight_dtype):
                        self.transformer(
                            x=denoised_pred,
                            context=prompt_embeds,
                            t=context_timestep,
                            seq_len=seq_len,
                            kv_cache=self.kv_cache_pos,
                            crossattn_cache=self.crossattn_cache_pos,
                            current_start=current_start_frame * frame_seq_length,
                            cache_start=None,
                        )
                        self.transformer(
                            x=denoised_pred,
                            context=negative_prompt_embeds,
                            t=context_timestep,
                            seq_len=seq_len,
                            kv_cache=self.kv_cache_neg,
                            crossattn_cache=self.crossattn_cache_neg,
                            current_start=current_start_frame * frame_seq_length,
                            cache_start=None,
                        )
                else:
                    with torch.cuda.amp.autocast(dtype=weight_dtype):
                        self.transformer(
                            x=denoised_pred,
                            context=in_prompt_embeds,
                            t=context_timestep,
                            seq_len=seq_len,
                            kv_cache=self.kv_cache_pos,
                            crossattn_cache=self.crossattn_cache_pos,
                            current_start=current_start_frame * frame_seq_length,
                            cache_start=None,
                        )

            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, block_idx, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)

            if comfyui_progressbar:
                pbar.update(1)

        # 9. Decode output
        
        if output_type == "pil":
            video = self.decode_latents(output)
            video = torch.from_numpy(video)
        else:
            video = output

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanSelfForcingPipelineOutput(videos=video)
