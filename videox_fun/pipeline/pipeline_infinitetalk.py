# Modified from https://github.com/MeiGen-AI/InfiniteTalk/blob/main/wan/multitalk.py
import copy
import inspect
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from tqdm import tqdm

from ..models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                      InfiniteTalkTransformer3DModel, InfiniteTalkAudioEncoder,
                      WanT5EncoderModel)
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


class MomentumBuffer:
    """Momentum buffer for APG (Adaptive Projected Guidance)."""
    def __init__(self, momentum=0.5):
        self.momentum = momentum
        self.buffer = None
    
    def update(self, grad):
        if self.buffer is None:
            self.buffer = grad.clone()
        else:
            self.buffer = self.momentum * self.buffer + (1 - self.momentum) * grad
        return self.buffer


def adaptive_projected_guidance(diff_uncond, noise_pred_cond, momentum_buffer=None, norm_threshold=1.0):
    """Adaptive Projected Guidance for better CFG."""
    if momentum_buffer is not None:
        diff_uncond = momentum_buffer.update(diff_uncond)
    
    cond_norm = torch.norm(noise_pred_cond)
    diff_norm = torch.norm(diff_uncond)
    
    if diff_norm > norm_threshold * cond_norm:
        diff_uncond = diff_uncond * (norm_threshold * cond_norm / diff_norm)
    
    return diff_uncond


def match_and_blend_colors(videos, original_color_reference, strength=0.5):
    """Match and blend colors between generated video and reference."""
    if strength <= 0.0:
        return videos
    
    # Calculate mean and std for color matching
    orig_mean = original_color_reference.mean(dim=[2, 3, 4], keepdim=True)
    orig_std = original_color_reference.std(dim=[2, 3, 4], keepdim=True) + 1e-8
    
    gen_mean = videos.mean(dim=[2, 3, 4], keepdim=True)
    gen_std = videos.std(dim=[2, 3, 4], keepdim=True) + 1e-8
    
    # Normalize and blend
    videos_normalized = (videos - gen_mean) / gen_std
    videos_matched = videos_normalized * orig_std.to(videos_normalized.device) + orig_mean.to(videos_normalized.device)
    
    # Blend with original strength
    videos = (1 - strength) * videos + strength * videos_matched
    return videos


@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    videos: torch.Tensor


class InfiniteTalkPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _exclude_from_cpu_offload = ["audio_encoder"]
    _optional_components = ["audio_encoder", "clip_image_encoder"]
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
        audio_encoder: InfiniteTalkAudioEncoder,
        vae: AutoencoderKLWan,
        transformer: InfiniteTalkTransformer3DModel,
        scheduler: FlowMatchEulerDiscreteScheduler = None,
        clip_image_encoder: CLIPModel = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, 
            scheduler=scheduler, audio_encoder=audio_encoder,
            clip_image_encoder=clip_image_encoder
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.spatial_compression_ratio, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.motion_frames = 9

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
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None, num_length_latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1 if num_length_latents is None else num_length_latents,
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

    def prepare_control_latents(
        self, control, control_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the control to latents shape as we concatenate the control to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        if control is not None:
            control = control.to(device=device, dtype=dtype)
            bs = 1
            new_control = []
            for i in range(0, control.shape[0], bs):
                control_bs = control[i : i + bs]
                control_bs = self.vae.encode(control_bs)[0]
                control_bs = control_bs.mode()
                new_control.append(control_bs)
            control = torch.cat(new_control, dim = 0)

        if control_image is not None:
            control_image = control_image.to(device=device, dtype=dtype)
            bs = 1
            new_control_pixel_values = []
            for i in range(0, control_image.shape[0], bs):
                control_pixel_values_bs = control_image[i : i + bs]
                control_pixel_values_bs = self.vae.encode(control_pixel_values_bs)[0]
                control_pixel_values_bs = control_pixel_values_bs.mode()
                new_control_pixel_values.append(control_pixel_values_bs)
            control_image_latents = torch.cat(new_control_pixel_values, dim = 0)
        else:
            control_image_latents = None

        return control, control_image_latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        # frames = frames.cpu().float().numpy()
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

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        ref_image: Union[torch.FloatTensor] = None,
        clip_image = None,  # Add clip_image parameter
        audio_path = None,
        segment_frame_length: int = 81,
        num_inference_steps: int = 40,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 5.0,
        audio_guide_scale: float = 4.0,
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
        shift: float = 5.0,
        fps: int = 16,
        max_frames_num: int = 1000,
        color_correction_strength: float = 0.0,
        use_apg: bool = False,
        apg_momentum: float = 0.5,
        apg_norm_threshold: float = 1.0,
        progress: bool = True,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.
        Args:

        Examples:

        Returns:

        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
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

        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 2)

        # 4. Prepare latents.
        latent_channels = self.vae.config.latent_channels
        if comfyui_progressbar:
            pbar.update(1)

        # Ref_image b c 1 h w
        video_length = ref_image.shape[2]
        ref_image = self.image_processor.preprocess(rearrange(ref_image, "b c f h w -> (b f) c h w"), height=height, width=width) 
        ref_image = ref_image.to(dtype=torch.float32)
        ref_image = rearrange(ref_image, "(b f) c h w -> b c f h w", f=video_length)
        
        # Ref_image b 16 1 h/8 w/8
        ref_image_latentes = self.prepare_control_latents(
            None,
            ref_image,
            batch_size,
            height,
            width,
            weight_dtype,
            device,
            generator,
            do_classifier_free_guidance
        )[1]
        ref_image_latentes = ref_image_latentes[:, :, :1]

        # Store original color reference for color correction (in pixel space, not latent)
        original_color_reference = None
        if color_correction_strength > 0.0:
            original_color_reference = ref_image.clone()  # [B, C, T, H, W] in pixel space

        # Extract raw audio features
        # full_audio_emb shape: [total_audio_frames, num_layers=12, dim=768]
        # Following official InfiniteTalk: audio embeddings should be pre-extracted and aligned to video frames
        # [125, 12, 768]
        full_audio_emb = self.audio_encoder.extract_audio_feat(
            audio_path, return_all_layers=True
        )
        full_audio_emb = full_audio_emb.to(device, weight_dtype)

        # Prepare CLIP features
        if clip_image is not None:
            # PIL Image, convert to tensor [C, H, W] then add batch + temporal dims
            clip_image_tensor = TF.to_tensor(clip_image).sub_(0.5).div_(0.5)
            clip_image_tensor = clip_image_tensor.unsqueeze(0).unsqueeze(2).to(device, weight_dtype)
            clip_fea = self.clip_image_encoder(clip_image_tensor).to(weight_dtype)
        else:
            # If no clip_image, create zero features
            clip_fea = torch.zeros(1, 768, dtype=weight_dtype, device=device)

        # For single person mode: create full mask and resize
        ref_target_masks_pixel = torch.ones([1, height, width], dtype=weight_dtype, device=device)
        ref_target_masks = F.interpolate(
            ref_target_masks_pixel.unsqueeze(0), 
            size=(height // self.vae.spatial_compression_ratio, width // self.vae.spatial_compression_ratio), 
            mode='nearest'
        ).squeeze(0)
        # Stack for 3 channels (official expects [3, H, W])
        ref_target_masks = ref_target_masks.repeat(3, 1, 1)  # [3, lat_h, lat_w]

        # InfiniteTalk iterative generation
        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        clip_length = segment_frame_length
        is_first_clip = True
        arrive_last_frame = False
        cur_motion_frames_num = 1
        audio_start_idx = 0
        audio_end_idx = audio_start_idx + clip_length
        gen_video_list = []

        if comfyui_progressbar:
            pbar.update(1)

        # 5. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Initialize APG momentum buffers if enabled
        text_momentumbuffer = MomentumBuffer(apg_momentum) if use_apg else None
        audio_momentumbuffer = MomentumBuffer(apg_momentum) if use_apg else None

        # Iterative generation loop
        while True:
            # 6. Prepare timesteps
            if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
                timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None, mu=1)
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
                timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None)
            
            # Apply timestep transform
            self._num_timesteps = len(timesteps)

            # Calculate audio indices for this segment
            # full_audio_emb shape from encoder: [total_frames, num_layers=12, dim=768]
            # Transformer expects: [B, T, audio_window, blocks=12, channels=768]
            center_indices = torch.arange(
                audio_start_idx,
                audio_end_idx,
                1,
            ).unsqueeze(1) + indices.unsqueeze(0)  
            # [segment_frames, audio_window]
            # [81, 5]
            center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0]-1)
            
            # Index into audio embeddings
            # full_audio_emb[center_indices]: [segment_frames, audio_window, num_layers, dim]
            # audio_emb_input: [1, 81, 5, 12, 768]
            audio_emb_input = full_audio_emb[center_indices].unsqueeze(0).to(device, weight_dtype)

            # lat_motion_frames: number of latent motion frames
            lat_motion_frames = (cur_motion_frames_num + 3) // 4
            # Following official: lat_target_frames is the total latent frames for segment_frame_length
            lat_target_frames = (segment_frame_length - 1) // self.vae.temporal_compression_ratio + 1
            
            # 7. Prepare latents
            # [B, C, T, H, W] = [1, 16, lat_target_frames, h // 8, w // 8]
            lat_h_target = height // self.vae.spatial_compression_ratio
            lat_w_target = width // self.vae.spatial_compression_ratio
            latents = torch.randn(
                1,  # B
                self.vae.config.latent_channels,  # C = 16
                lat_target_frames,  # T
                lat_h_target,  # H
                lat_w_target,  # W
                dtype=weight_dtype,
                device=device,
                generator=generator,
            )
            
            # Calculate seq_len for transformer
            # seq_len = number of tokens = (H_lat * W_lat) / (patch_h * patch_w) * T_lat
            seq_len = math.ceil((lat_h_target * lat_w_target) / (self.transformer.patch_size[1] * self.transformer.patch_size[2]) * lat_target_frames)

            # Prepare y: mask + VAE encoded condition (following original multitalk.py)
            # Official: y is 4D tensor [4+C, T_latent, H_latent, W_latent]
            lat_h_local = height // self.vae.spatial_compression_ratio
            lat_w_local = width // self.vae.spatial_compression_ratio
            
            # Step 1: Create mask in pixel frame dimension
            # Mask should have the same temporal dimension as y_latent (lat_target_frames)
            msk_pixel = torch.ones(1, segment_frame_length, lat_h_local, lat_w_local, device=device, dtype=weight_dtype)
            msk_pixel[:, 1:] = 0  # First frame = 1, rest = 0
            # Repeat first frame 4 times (temporal compression) and reshape
            msk_pixel = torch.concat([
                torch.repeat_interleave(msk_pixel[:, 0:1], repeats=4, dim=1),
                msk_pixel[:, 1:]
            ], dim=1)
            # [1, 4, T_latent, H_latent, W_latent]
            msk = msk_pixel.view(1, msk_pixel.shape[1] // 4, 4, lat_h_local, lat_w_local)
            msk = msk.transpose(1, 2)  
            
            # Step 2: Encode condition video (ref_image in pixel space + zero frames)
            # ref_image is [B, C, T, H, W] normalized to [-1, 1]
            cond_video_pixels = torch.zeros(1, 3, segment_frame_length, height, width, dtype=weight_dtype, device=device)
            cond_video_pixels[:, :, :1] = ref_image[:, :, :1]
            y_latent_output = self.vae.encode(cond_video_pixels)
            # [B, 16, T_latent, H_latent, W_latent]
            y_latent = y_latent_output[0].mode().to(weight_dtype) 
            
            # Step 3: Concat mask and latent in channel dimension
            # msk: [1, 4, T, H, W] + y_latent: [1, C, T, H, W]
            # y: [1, 20, T, H, W]
            y = torch.cat([msk, y_latent], dim=1)  

            # Encode motion_latents following official: use ref_image/cond_frame
            if is_first_clip:
                # First clip: encode ref_image (official Line 577: self.vae.encode(cond_image))
                motion_latents_pixels = ref_image.to(device, weight_dtype)
            else:
                # Subsequent clips: encode cond_frame (official Line 579: self.vae.encode(cond_frame))
                motion_latents_pixels = cond_frame.to(device, weight_dtype)
            
            motion_latents_output = self.vae.encode(motion_latents_pixels)
            motion_latents = motion_latents_output[0].mode()  # [B, C, T, H, W]
            
            # 8. Denoising loop
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self.transformer.num_inference_steps = num_inference_steps
            
            progress_wrap = partial(tqdm, total=len(timesteps)) if progress else (lambda x: x)
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(progress_wrap(timesteps)):
                    self.transformer.current_steps = i

                    if self.interrupt:
                        continue

                    # For InfiniteTalk, CFG is handled by multiple transformer calls, not batching
                    latent_model_input = latents.to(weight_dtype)
                    if hasattr(self.scheduler, "scale_model_input"):
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    
                    # Inject clean motion frames at the beginning of each step (following official Line 711)
                    # latents/motion_latents are 5D [B, C, T, H, W], index on T dimension
                    latents[:, :, :lat_motion_frames] = motion_latents[:, :, :lat_motion_frames]
                    
                    # Expand timestep to match batch size (1 for InfiniteTalk)
                    # [1] tensor for transformer
                    timestep = t.expand(1)

                    # Forward Code
                    # Transformer accepts 5D tensors [B, C, T, H, W] directly
                    with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                        # context is already a list from encode_prompt
                        context_input = prompt_embeds if isinstance(prompt_embeds, list) else [prompt_embeds]
                        context_neg = negative_prompt_embeds if isinstance(negative_prompt_embeds, list) else [negative_prompt_embeds]
                        
                        noise_pred_cond = self.transformer(
                            x=latents,  # 5D [B, C, T, H, W]
                            t=timestep,
                            context=context_input,
                            seq_len=seq_len,
                            clip_fea=clip_fea,
                            y=y,  # 5D [B, 4+C, T, H, W]
                            audio=audio_emb_input,
                            ref_target_masks=ref_target_masks,
                            cond_flag=True,
                        )
                    
                    # CFG with APG support
                    if math.isclose(guidance_scale, 1.0):
                        # Only audio guidance
                        noise_pred_drop_audio = self.transformer(
                            x=latents,
                            t=timestep,
                            context=context_input,
                            seq_len=seq_len,
                            clip_fea=clip_fea,
                            y=y,
                            audio=audio_emb_input * 0,  # zero audio
                            ref_target_masks=ref_target_masks,
                            cond_flag=False,
                        )
                        
                        if use_apg:
                            diff_uncond_audio = noise_pred_cond - noise_pred_drop_audio
                            noise_pred = noise_pred_cond + (audio_guide_scale - 1) * adaptive_projected_guidance(
                                diff_uncond_audio, noise_pred_cond,
                                momentum_buffer=audio_momentumbuffer,
                                norm_threshold=apg_norm_threshold
                            )
                        else:
                            noise_pred = noise_pred_drop_audio + audio_guide_scale * (noise_pred_cond - noise_pred_drop_audio)
                    else:
                        # Full CFG with text and audio
                        noise_pred_drop_text = self.transformer(
                            x=latents,
                            t=timestep,
                            context=context_neg,
                            seq_len=seq_len,
                            clip_fea=clip_fea,
                            y=y,
                            audio=audio_emb_input,
                            ref_target_masks=ref_target_masks,
                            cond_flag=False,
                        )
                        noise_pred_uncond = self.transformer(
                            x=latents,
                            t=timestep,
                            context=context_neg,
                            seq_len=seq_len,
                            clip_fea=clip_fea,
                            y=y,
                            audio=audio_emb_input * 0,  # zero audio
                            ref_target_masks=ref_target_masks,
                            cond_flag=False,
                        )
                        
                        if use_apg:
                            diff_uncond_text = noise_pred_cond - noise_pred_drop_text
                            diff_uncond_audio = noise_pred_drop_text - noise_pred_uncond
                            noise_pred = noise_pred_cond + (guidance_scale - 1) * adaptive_projected_guidance(
                                diff_uncond_text, noise_pred_cond,
                                momentum_buffer=text_momentumbuffer,
                                norm_threshold=apg_norm_threshold
                            ) + (audio_guide_scale - 1) * adaptive_projected_guidance(
                                diff_uncond_audio, noise_pred_cond,
                                momentum_buffer=audio_momentumbuffer,
                                norm_threshold=apg_norm_threshold
                            )
                        else:
                            noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_cond - noise_pred_drop_text) + \
                                audio_guide_scale * (noise_pred_drop_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    # Always inject clean motion at the end (following official Line 773)
                    latents[:, :, :lat_motion_frames] = motion_latents[:, :, :lat_motion_frames]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                    if comfyui_progressbar:
                        pbar.update(1)

            # Decode latents (following official Line 782: x0 = [latent])
            # latents is 5D [B, C, T, H, W], already contains motion frames (injected in denoising loop)
            image = self.vae.decode(latents.to(weight_dtype)).sample

            # Apply color correction
            if color_correction_strength > 0.0 and original_color_reference is not None:
                image = match_and_blend_colors(image, original_color_reference, color_correction_strength)

            # Cache generated video (following official Line 791-794)
            if is_first_clip:
                gen_video_list.append(image)
            else:
                gen_video_list.append(image[:, :, cur_motion_frames_num:])

            # Decide whether done
            if arrive_last_frame:
                break

            # Update for next iteration
            is_first_clip = False
            cur_motion_frames_num = self.motion_frames

            # Update cond_frame for next clip (following official Line 803)
            cond_frame = image[:, :, -cur_motion_frames_num:].to(dtype=weight_dtype, device=device)

            # Update audio indices
            audio_start_idx += (segment_frame_length - cur_motion_frames_num)
            audio_end_idx = audio_start_idx + clip_length

            # Check if reached last frame and pad audio if needed (following official Line 814-828)
            if audio_end_idx >= min(max_frames_num, full_audio_emb.shape[0]):
                arrive_last_frame = True
                # Pad audio by flipping the last frames
                if audio_end_idx >= full_audio_emb.shape[0]:
                    miss_length = audio_end_idx - full_audio_emb.shape[0] + 3
                    add_audio_emb = torch.flip(full_audio_emb[-miss_length:], dims=[0])
                    full_audio_emb = torch.cat([full_audio_emb, add_audio_emb], dim=0)

            if max_frames_num <= segment_frame_length:
                break

        # Concatenate all generated video segments
        videos = torch.cat(gen_video_list, dim=2)[:, :, :int(max_frames_num)]
        videos = (videos / 2 + 0.5).clamp(0, 1).float().cpu()
        
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return videos

        return WanPipelineOutput(videos=videos)
