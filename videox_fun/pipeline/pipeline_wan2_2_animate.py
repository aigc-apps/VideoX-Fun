import inspect
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import copy
import torch
import cv2
import torch.nn.functional as F
from einops import rearrange
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from decord import VideoReader

from ..models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                              WanT5EncoderModel, Wan2_2Transformer3DModel_Animate)
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


class Wan2_2AnimatePipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _optional_components = ["transformer_2", "clip_image_encoder"]
    model_cpu_offload_seq = "text_encoder->clip_image_encoder->transformer_2->transformer->vae"

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
        transformer: Wan2_2Transformer3DModel_Animate,
        transformer_2: Wan2_2Transformer3DModel_Animate = None,
        clip_image_encoder: CLIPModel = None,
        scheduler: FlowMatchEulerDiscreteScheduler = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, 
            transformer_2=transformer_2, clip_image_encoder=clip_image_encoder, scheduler=scheduler
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.spatial_compression_ratio, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

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

    def padding_resize(self, img_ori, height=512, width=512, padding_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
        ori_height = img_ori.shape[0]
        ori_width = img_ori.shape[1]
        channel = img_ori.shape[2]

        img_pad = np.zeros((height, width, channel))
        if channel == 1:
            img_pad[:, :, 0] = padding_color[0]
        else:
            img_pad[:, :, 0] = padding_color[0]
            img_pad[:, :, 1] = padding_color[1]
            img_pad[:, :, 2] = padding_color[2]

        if (ori_height / ori_width) > (height / width):
            new_width = int(height / ori_height * ori_width)
            img = cv2.resize(img_ori, (new_width, height), interpolation=interpolation)
            padding = int((width - new_width) / 2)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]  
            img_pad[:, padding: padding + new_width, :] = img
        else:
            new_height = int(width / ori_width * ori_height)
            img = cv2.resize(img_ori, (width, new_height), interpolation=interpolation)
            padding = int((height - new_height) / 2)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]  
            img_pad[padding: padding + new_height, :, :] = img

        img_pad = np.uint8(img_pad)

        return img_pad

    def inputs_padding(self, x, target_len):
        ndim = x.ndim

        if ndim == 4:
            f = x.shape[0]
            if target_len <= f:
                return [deepcopy(x[i]) for i in range(target_len)]
            
            idx = 0
            flip = False
            target_array = []
            while len(target_array) < target_len:
                target_array.append(deepcopy(x[idx]))
                if flip:
                    idx -= 1
                else:
                    idx += 1
                if idx == 0 or idx == f - 1:
                    flip = not flip
            return target_array[:target_len]

        elif ndim == 5:
            b, c, f, h, w = x.shape

            if target_len <= f:
                return x[:, :, :target_len, :, :]

            indices = []
            idx = 0
            flip = False
            while len(indices) < target_len:
                indices.append(idx)
                if flip:
                    idx -= 1
                else:
                    idx += 1
                if idx == 0 or idx == f - 1:
                    flip = not flip
            indices = indices[:target_len]

            if isinstance(x, torch.Tensor):
                indices_tensor = torch.tensor(indices, device=x.device, dtype=torch.long)
                return x[:, :, indices_tensor, :, :]
            else:
                indices_array = np.array(indices)
                return x[:, :, indices_array, :, :]
        
        else:
            raise ValueError(f"Unsupported input dimension: {ndim}. Expected 4D or 5D.")

    def get_valid_len(self, real_len, clip_len=81, overlap=1):
        real_clip_len = clip_len - overlap
        last_clip_num = (real_len - overlap) % real_clip_len
        if last_clip_num == 0:
            extra = 0
        else:
            extra = real_clip_len - last_clip_num
        target_len = real_len + extra
        return target_len

    def prepare_source(self, src_pose_path, src_face_path, src_ref_path):
        pose_video_reader = VideoReader(src_pose_path)
        pose_len = len(pose_video_reader)
        pose_idxs = list(range(pose_len))
        pose_video = pose_video_reader.get_batch(pose_idxs).asnumpy()

        face_video_reader = VideoReader(src_face_path)
        face_len = len(face_video_reader)
        face_idxs = list(range(face_len))
        face_video = face_video_reader.get_batch(face_idxs).asnumpy()
        height, width = pose_video[0].shape[:2]

        ref_image = cv2.imread(src_ref_path)[..., ::-1]
        ref_image = self.padding_resize(ref_image, height=height, width=width)
        return pose_video, face_video, ref_image
    
    def prepare_source_for_replace(self, src_bg_path, src_mask_path):
        bg_video_reader = VideoReader(src_bg_path)
        bg_len = len(bg_video_reader)
        bg_idxs = list(range(bg_len))
        bg_video = bg_video_reader.get_batch(bg_idxs).asnumpy()

        mask_video_reader = VideoReader(src_mask_path)
        mask_len = len(mask_video_reader)
        mask_idxs = list(range(mask_len))
        mask_video = mask_video_reader.get_batch(mask_idxs).asnumpy()
        mask_video = mask_video[:, :, :, 0] / 255
        return bg_video, mask_video

    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)
        return msk

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
        clip_len=77,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        pose_video = None,
        face_video = None,
        ref_image = None,
        bg_video = None,
        mask_video = None,
        replace_flag = True,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        boundary: float = 0.875,
        comfyui_progressbar: bool = False,
        shift: int = 5,
        refert_num = 1,
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
            pbar = ProgressBar(num_inference_steps + 1)

        # 4. Prepare latents
        if pose_video is not None:
            video_length = pose_video.shape[2]
            pose_video = self.image_processor.preprocess(rearrange(pose_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
            pose_video = pose_video.to(dtype=torch.float32)
            pose_video = rearrange(pose_video, "(b f) c h w -> b c f h w", f=video_length)
        else:
            pose_video = None

        if face_video is not None:
            video_length = face_video.shape[2]
            face_video = self.image_processor.preprocess(rearrange(face_video, "b c f h w -> (b f) c h w")) 
            face_video = face_video.to(dtype=torch.float32)
            face_video = rearrange(face_video, "(b f) c h w -> b c f h w", f=video_length)
        else:
            face_video = None

        real_frame_len = pose_video.size()[2]
        target_len = self.get_valid_len(real_frame_len, clip_len, overlap=refert_num)
        print('real frames: {} target frames: {}'.format(real_frame_len, target_len))
        pose_video = self.inputs_padding(pose_video, target_len).to(device, weight_dtype) 
        face_video = self.inputs_padding(face_video, target_len).to(device, weight_dtype) 
        ref_image = self.padding_resize(np.array(ref_image), height=height, width=width)
        ref_image = torch.tensor(ref_image / 127.5 - 1).unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0).to(device, weight_dtype) 

        if replace_flag:
            if bg_video is not None:
                video_length = bg_video.shape[2]
                bg_video = self.image_processor.preprocess(rearrange(bg_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
                bg_video = bg_video.to(dtype=torch.float32)
                bg_video = rearrange(bg_video, "(b f) c h w -> b c f h w", f=video_length)
            else:
                bg_video = None
            bg_video = self.inputs_padding(bg_video, target_len).to(device, weight_dtype) 
            mask_video = self.inputs_padding(mask_video, target_len).to(device, weight_dtype) 

        if comfyui_progressbar:
            pbar.update(1)

        # 5. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (self.vae.latent_channels, (num_frames - 1) // self.vae.temporal_compression_ratio + 1, width // self.vae.spatial_compression_ratio, height // self.vae.spatial_compression_ratio)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1]) 
        
        # 6. Denoising loop
        start           = 0
        end             = clip_len
        all_out_frames  = []
        copy_timesteps  = copy.deepcopy(timesteps)
        copy_latents    = copy.deepcopy(latents)
        bs              = pose_video.size()[0]
        while True:
            if start + refert_num >= pose_video.size()[2]:
                break

            # Prepare timesteps
            if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
                timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, copy_timesteps, mu=1)
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
                timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, copy_timesteps)
            self._num_timesteps = len(timesteps)

            latent_channels = self.transformer.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                latent_channels,
                num_frames,
                height,
                width,
                weight_dtype,
                device,
                generator,
                copy_latents,
            )

            if start == 0:
                mask_reft_len = 0
            else:
                mask_reft_len = refert_num

            conditioning_pixel_values   = pose_video[:, :, start:end]
            face_pixel_values           = face_video[:, :, start:end]
            ref_pixel_values            = ref_image.clone().detach()
            if start > 0:
                refer_t_pixel_values = out_frames[:, :, -refert_num:].clone().detach()
                refer_t_pixel_values = (refer_t_pixel_values - 0.5) / 0.5
            else:
                refer_t_pixel_values = torch.zeros(bs, 3, refert_num, height, width)
            refer_t_pixel_values = refer_t_pixel_values.to(device=device, dtype=weight_dtype)
            
            pose_latents, ref_latents = self.prepare_control_latents(
                conditioning_pixel_values,
                ref_pixel_values,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )

            mask_ref    = self.get_i2v_mask(1, target_shape[-1], target_shape[-2], 1, device=device)
            y_ref       = torch.concat([mask_ref, ref_latents], dim=1).to(device=device, dtype=weight_dtype)
            if mask_reft_len > 0:
                if replace_flag:
                    # Image.fromarray(np.array((refer_t_pixel_values[0, :, 0].permute(1,2,0) * 0.5 + 0.5).float().cpu().numpy() *255, np.uint8)).save("1.jpg")
                    bg_pixel_values = bg_video[:, :, start:end]
                    y_reft = self.vae.encode(
                        torch.concat(
                            [
                                refer_t_pixel_values[:, :, :mask_reft_len], 
                                bg_pixel_values[:, :, mask_reft_len:]
                            ], dim=2
                        ).to(device=device, dtype=weight_dtype)
                    )[0].mode()

                    mask_pixel_values = 1 - mask_video[:, :, start:end]
                    mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
                    mask_pixel_values = F.interpolate(mask_pixel_values, size=(target_shape[-1], target_shape[-2]), mode='nearest')
                    mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b c t h w", b = bs)[:, 0]
                    msk_reft = self.get_i2v_mask(
                        int((clip_len - 1) // self.vae.temporal_compression_ratio + 1), target_shape[-1], target_shape[-2], mask_reft_len, mask_pixel_values=mask_pixel_values, device=device
                    )
                else:
                    refer_t_pixel_values = rearrange(refer_t_pixel_values[:, :, :mask_reft_len], "b c t h w -> (b t) c h w")
                    refer_t_pixel_values = F.interpolate(refer_t_pixel_values, size=(height, width), mode="bicubic")
                    refer_t_pixel_values = rearrange(refer_t_pixel_values, "(b t) c h w -> b c t h w", b = bs)

                    y_reft = self.vae.encode(
                        torch.concat(
                            [
                                refer_t_pixel_values,
                                torch.zeros(bs, 3, clip_len - mask_reft_len, height, width).to(device=device, dtype=weight_dtype),
                            ], dim=2,
                        ).to(device=device, dtype=weight_dtype)
                    )[0].mode()
                    msk_reft = self.get_i2v_mask(
                        int((clip_len - 1) // self.vae.temporal_compression_ratio + 1), target_shape[-1], target_shape[-2], mask_reft_len, device=device
                    )
            else:
                if replace_flag:
                    bg_pixel_values = bg_video[:, :, start:end]
                    y_reft = self.vae.encode(
                        bg_pixel_values.to(device=device, dtype=weight_dtype)
                    )[0].mode()

                    mask_pixel_values = 1 - mask_video[:, :, start:end]
                    mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
                    mask_pixel_values = F.interpolate(mask_pixel_values, size=(target_shape[-1], target_shape[-2]), mode='nearest')
                    mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b c t h w", b = bs)[:, 0]
                    msk_reft = self.get_i2v_mask(
                        int((clip_len - 1) // self.vae.temporal_compression_ratio + 1), target_shape[-1], target_shape[-2], mask_reft_len, mask_pixel_values=mask_pixel_values, device=device
                    )
                else:
                    y_reft = self.vae.encode(
                        torch.zeros(1, 3, clip_len - mask_reft_len, height, width).to(device=device, dtype=weight_dtype)
                    )[0].mode()
                    msk_reft = self.get_i2v_mask(
                        int((clip_len - 1) // self.vae.temporal_compression_ratio + 1), target_shape[-1], target_shape[-2], mask_reft_len, device=device
                    )

            y_reft = torch.concat([msk_reft, y_reft], dim=1).to(device=device, dtype=weight_dtype)
            y = torch.concat([y_ref, y_reft], dim=2)

            clip_context = self.clip_image_encoder([ref_pixel_values[0, :, :, :]]).to(device=device, dtype=weight_dtype)

            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self.transformer.num_inference_steps = num_inference_steps
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    self.transformer.current_steps = i

                    if self.interrupt:
                        continue

                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    if hasattr(self.scheduler, "scale_model_input"):
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    y_in = torch.cat([y] * 2) if do_classifier_free_guidance else y
                    clip_context_input = (
                        torch.cat([clip_context] * 2) if do_classifier_free_guidance else clip_context
                    )
                    pose_latents_input = (
                        torch.cat([pose_latents] * 2) if do_classifier_free_guidance else pose_latents
                    )
                    face_pixel_values_input = (
                        torch.cat([torch.ones_like(face_pixel_values) * -1] + [face_pixel_values]) if do_classifier_free_guidance else face_pixel_values
                    )

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])
                    
                    if self.transformer_2 is not None:
                        if t >= boundary * self.scheduler.config.num_train_timesteps:
                            local_transformer = self.transformer_2
                        else:
                            local_transformer = self.transformer
                    else:
                        local_transformer = self.transformer

                    # predict noise model_output
                    with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                        noise_pred = local_transformer(
                            x=latent_model_input,
                            context=in_prompt_embeds,
                            t=timestep,
                            seq_len=seq_len,
                            y=y_in,
                            clip_fea=clip_context_input,
                            pose_latents=pose_latents_input,
                            face_pixel_values=face_pixel_values_input,
                        )

                    # Perform guidance
                    if do_classifier_free_guidance:
                        if self.transformer_2 is not None and (isinstance(self.guidance_scale, (list, tuple))):
                            sample_guide_scale = self.guidance_scale[1] if t >= self.transformer_2.config.boundary * self.scheduler.config.num_train_timesteps else self.guidance_scale[0]
                        else:
                            sample_guide_scale = self.guidance_scale
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_text - noise_pred_uncond)

                    # Compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

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

            out_frames = self.decode_latents(latents[:, :, 1:])
            if start != 0:
                out_frames = out_frames[:, :, refert_num:]
            all_out_frames.append(out_frames.cpu())
            start += clip_len - refert_num
            end += clip_len - refert_num

        videos = torch.cat(all_out_frames, dim=2)[:, :, :real_frame_len]

        # Offload all models
        self.maybe_free_model_hooks()

        return WanPipelineOutput(videos=videos.float().cpu())
