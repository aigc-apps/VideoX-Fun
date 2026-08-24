# Modified from https://github.com/Robbyant/lingbot-video/blob/main/lingbot_video/pipeline_lingbot_video.py
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from torchvision.transforms.functional import \
    normalize as normalize_image_tensor

from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


TOKEN_LENGTH = 37698
HIDDEN_STATE_SKIP_LAYER = 0
LOW_NOISE_TAIL_V1_DEFAULT_STEPS = 2

PROMPT_TEMPLATE = (
    "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
    "a text prompt with an image reference, or a text prompt with a video reference "
    "or a video reference alone, generate an \"Enhanced prompt\" that provides detailed "
    "visual descriptions suitable for video generation. Evaluate the level of detail "
    "in the user's input: if it is simple, enrich it by adding specifics about colors, "
    "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
    "progression, and spatial relationships to create vivid, concrete, and temporally "
    "coherent scenes to create vivid and concrete scenes. Please generate only the "
    "enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
IMG_PROMPT_TEMPLATE = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PROMPT_TEMPLATE = "<|vision_start|><|video_pad|><|vision_end|>"

DEFAULT_NEGATIVE_PROMPT = (
    '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], "temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static object with sudden jump", "frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}'
)

# Still-image default (t2i): drops the whole temporal/motion block and the video-only
# codec/temporal terms that cannot apply to a single frame.
DEFAULT_NEGATIVE_PROMPT_IMAGE = (
    '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "pillarboxed", "side bars", "portrait image in landscape frame"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "distorted reflections"]}}'
)


@dataclass
class LingBotVideoPipelineOutput(BaseOutput):
    r"""
    Output class for LingBotVideo pipelines.

    Args:
        videos (`torch.Tensor`):
            Tensor of shape `(batch_size, channels, num_frames, height, width)` with values in `[0, 1]`.
    """

    videos: torch.Tensor


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _transformer_timestep(timestep: torch.Tensor, transformer_dtype: torch.dtype) -> torch.Tensor:
    sigma = timestep.float() / 1000.0
    if transformer_dtype in {torch.bfloat16, torch.float16}:
        sigma = sigma.to(transformer_dtype)
    return (sigma * 1000.0).float()


def _transformer_autocast(device: torch.device, transformer_dtype: torch.dtype):
    if device.type != "cuda" or transformer_dtype not in {torch.bfloat16, torch.float16}:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=transformer_dtype)


def compute_refiner_sigmas(
    sigma_max: float,
    sigma_min: float,
    num_inference_steps: int,
    shift: float,
    t_thresh: Optional[float],
    tail_steps: int = 0,
) -> Optional[np.ndarray]:
    """Truncated sigma schedule for the low-noise refiner: keep the shifted sigmas
    below `t_thresh` and optionally append a short low-noise tail."""
    if t_thresh is None:
        return None
    t_value = float(t_thresh)
    if not (0.0 < t_value <= 1.0):
        raise ValueError(f"refiner t_thresh must lie in (0, 1], got {t_value}")
    steps = int(num_inference_steps)
    if steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {steps}")
    tail = int(tail_steps or 0)
    if tail < 0:
        raise ValueError(f"refiner_sigma_tail_steps must be >= 0, got {tail}")

    base = np.linspace(float(sigma_max), float(sigma_min), steps + 1).copy()[:-1]
    shift_value = float(shift)
    shifted = shift_value * base / (1.0 + (shift_value - 1.0) * base)
    eps = 1e-6
    sigmas = shifted[shifted <= t_value + eps]
    if sigmas.size == 0 or abs(float(sigmas[0]) - t_value) > eps:
        sigmas = np.concatenate([[t_value], sigmas])
    if tail > 0:
        start = float(sigmas[-1])
        stop = min(float(sigma_min), start)
        extra = np.linspace(start, stop, tail + 2, dtype=np.float64)[1:-1]
        sigmas = np.concatenate([sigmas, extra])
    return sigmas.astype(np.float32)


def prepare_refiner_latent(
    x_up: torch.Tensor,
    noise: torch.Tensor,
    t_thresh: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Mix a clean (upsampled) latent with noise at level `t_thresh` for refinement."""
    if not torch.is_tensor(t_thresh):
        t_thresh = torch.tensor(float(t_thresh), device=x_up.device, dtype=x_up.dtype)
    while t_thresh.ndim < x_up.ndim:
        t_thresh = t_thresh.view(*t_thresh.shape, *([1] * (x_up.ndim - t_thresh.ndim)))
    return (1.0 - t_thresh) * x_up + t_thresh * noise


class LingBotVideoPipeline(DiffusionPipeline):
    r"""
    LingBot-Video t2v/t2i pipeline.

    CFG runs as two independent transformer forwards. Prompts are expected to be
    structured JSON captions produced by the LingBot-Video rewriter, but plain
    natural-language prompts also work through the built-in chat template.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(self, transformer, vae, text_encoder, processor, scheduler):
        super().__init__()
        if (
            scheduler is not None
            and scheduler.__class__.__name__ != FlowUniPCMultistepScheduler.__name__
        ):
            raise TypeError(
                "LingBotVideoPipeline requires FlowUniPCMultistepScheduler; "
                f"got {scheduler.__class__.__name__}."
            )
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
            scheduler=scheduler,
        )
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8
        self.token_length = TOKEN_LENGTH
        self.hidden_state_skip_layer = HIDDEN_STATE_SKIP_LAYER
        self.prompt_template = PROMPT_TEMPLATE
        self.img_prompt_template = IMG_PROMPT_TEMPLATE
        self.video_prompt_template = VIDEO_PROMPT_TEMPLATE
        self._crop_start: Optional[int] = None

    @staticmethod
    def check_inputs(height: int, width: int, num_frames: int) -> None:
        if num_frames != 1 and (num_frames - 1) % 4 != 0:
            raise ValueError(f"`num_frames` must be 1 or 4n+1, got {num_frames}.")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be multiples of 16, got {height}x{width}.")

    @staticmethod
    def _apply_inpainting(latents: torch.Tensor, cond_latent: torch.Tensor) -> torch.Tensor:
        cond_t = cond_latent.shape[2]
        latents[:, :, :cond_t] = cond_latent.float()
        return latents

    @staticmethod
    def apply_text_to_template(text: str, template: str = PROMPT_TEMPLATE) -> str:
        return template.format(text)

    def _compute_crop_start(self) -> int:
        if self._crop_start is None:
            marker = "<|USER_INPUT_MARKER|>"
            marked = self.prompt_template.format(marker)
            marker_pos = marked.find(marker)
            if marker_pos < 0:
                self._crop_start = 0
            else:
                prefix = self.processor(
                    text=marked[:marker_pos],
                    images=None,
                    videos=None,
                    return_tensors="pt",
                )
                self._crop_start = int(prefix["input_ids"].shape[1])
        return self._crop_start

    def _build_prompt_inputs(
        self,
        prompt: Union[str, List[str]],
        images: Optional[Any] = None,
        videos: Optional[Any] = None,
        video_metadata: Optional[Any] = None,
        video_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)

        visual_template = ""
        if images is not None:
            visual_template = self.img_prompt_template
        elif videos is not None:
            visual_template = self.video_prompt_template

        texts = [
            self.apply_text_to_template(visual_template + text, self.prompt_template)
            for text in prompts
        ]
        kwargs = dict(video_kwargs or {})
        return self.processor(
            text=texts,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            do_resize=False,
            truncation=True,
            max_length=self.token_length,
            padding="longest",
            return_tensors="pt",
            **kwargs,
        )

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        images: Optional[Any] = None,
        videos: Optional[Any] = None,
        video_metadata: Optional[Any] = None,
        video_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if self.text_encoder is None or self.processor is None:
            raise ValueError("`text_encoder` and `processor` are required for encode_prompt().")

        device = torch.device(device) if device is not None else self._execution_device
        inputs = self._build_prompt_inputs(
            prompt,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            video_kwargs=video_kwargs,
        )
        inputs = inputs.to(device)
        # Released-checkpoint semantics (repo/lingbot-video @ release commit):
        # call the full ForConditionalGeneration model with
        # `output_hidden_states=True` and read
        # `hidden_states[-(skip_layer + 1)]`. With the default skip_layer=0
        # this is the final-layer output *before* the terminal RMSNorm
        # (std ~14). Experiments show the DiT text_embedder's leading
        # RMSNorm absorbs the scale difference, so the inner model's
        # post-norm `last_hidden_state` (std ~1) also works; we keep the
        # release semantics for exact parity with the official runner.
        # The cpu-offload hook is registered on the top-level `text_encoder`;
        # ensure the module is resident on the execution device first.
        self.text_encoder.to(device)
        outputs = self.text_encoder(
            **inputs,
            output_hidden_states=self.hidden_state_skip_layer is not None,
        )
        if self.hidden_state_skip_layer is not None:
            prompt_embeds = outputs.hidden_states[-(self.hidden_state_skip_layer + 1)]
        else:
            prompt_embeds = outputs.last_hidden_state

        prompt_mask = inputs["attention_mask"]
        crop_start = self._compute_crop_start()
        if crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_mask = prompt_mask[:, crop_start:]

        # Batch=1 can drop right padding before DiT inference.
        if prompt_embeds.shape[0] == 1:
            true_len = int(prompt_mask[0].sum().item())
            prompt_embeds = prompt_embeds[:, :true_len]
            prompt_mask = prompt_mask[:, :true_len]

        offload_hook = getattr(self.text_encoder, "_hf_hook", None)
        if offload_hook is not None and getattr(offload_hook, "offload", False):
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        return prompt_embeds, prompt_mask

    def prepare_latents(
        self,
        num_frames: int,
        height: int,
        width: int,
        generator: Optional[torch.Generator],
        latents: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (
            1,
            self.transformer.config.in_channels,
            latent_frames,
            latent_height,
            latent_width,
        )
        if latents is None:
            return randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        if tuple(latents.shape) != shape:
            raise ValueError(f"`latents` shape must be {shape}, got {tuple(latents.shape)}.")
        return latents.to(device=device, dtype=torch.float32)

    def _latents_mean_std_inv(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=torch.float32)
        std_inv = 1.0 / torch.tensor(self.vae.config.latents_std, device=device, dtype=torch.float32)
        return mean.view(1, -1, 1, 1, 1), std_inv.view(1, -1, 1, 1, 1)

    def _dit_latent_to_vae(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std_inv = self._latents_mean_std_inv(latents.device)
        return latents.float() / std_inv + mean

    def _vae_latent_to_dit(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std_inv = self._latents_mean_std_inv(latents.device)
        return (latents.float() - mean) * std_inv

    @torch.no_grad()
    def encode_video_latent(
        self,
        video: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        # video: (B, C, T, H, W) in [0, 1]
        if self.vae is None:
            raise ValueError("`vae` is required to encode video latents.")
        # `_execution_device` (not the current parameter device) is the onload device:
        # with cpu/group offload the VAE still sits on CPU here and only moves to the
        # accelerator inside its forward hook.
        vae_device = self._execution_device
        vae_dtype = _module_dtype(self.vae)
        video = video.to(device=vae_device, dtype=torch.float32)
        bsz, channels, frames, height, width = video.shape
        flat_video = video.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channels, height, width)
        norm_flat_video = normalize_image_tensor(
            flat_video,
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            inplace=False,
        )
        norm_video = (
            norm_flat_video.reshape(bsz, frames, channels, height, width)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=vae_device.type == "cuda",
        ):
            encoded = self.vae.encode(norm_video.to(vae_dtype))
        if hasattr(encoded, "latent_dist"):
            latents = encoded.latent_dist.sample(generator)
        else:
            latents = encoded[0] if isinstance(encoded, tuple) else encoded
        return self._vae_latent_to_dit(latents).to(latents)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae_device = self._execution_device
        vae_dtype = _module_dtype(self.vae)
        vae_latents = self._dit_latent_to_vae(latents).to(device=vae_device, dtype=torch.float32)
        autocast_dtype = (
            vae_dtype
            if vae_device.type == "cuda" and vae_dtype in {torch.bfloat16, torch.float16}
            else None
        )
        with torch.autocast(
            "cuda",
            dtype=autocast_dtype or torch.bfloat16,
            enabled=autocast_dtype is not None,
        ):
            decoded = self.vae.decode(vae_latents.to(vae_dtype))
        frames = decoded[0] if isinstance(decoded, tuple) else decoded.sample
        frames = frames.float().clamp_(-1, 1)
        frames = (frames + 1.0) / 2.0
        return frames.cpu()  # (B, C, T, H, W) in [0, 1]

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 3.0,
        shift: float = 3.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        cond_latent: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_mask: Optional[torch.Tensor] = None,
        t_thresh: Optional[float] = None,
        refiner_sigma_tail_steps: int = LOW_NOISE_TAIL_V1_DEFAULT_STEPS,
        output_type: str = "pt",
        return_dict: bool = True,
    ) -> Union[LingBotVideoPipelineOutput, Tuple]:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str`):
                The prompt to guide the video generation.
            negative_prompt (`str`, *optional*, defaults to the built-in default):
                The prompt to guide the video generation to avoid.
            height (`int`, *optional*, defaults to 480):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to 832):
                The width in pixels of the generated video.
            num_frames (`int`, *optional*, defaults to 81):
                The number of frames of the generated video.
            num_inference_steps (`int`, *optional*, defaults to 40):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 3.0):
                The guidance scale for classifier-free guidance.
            shift (`float`, *optional*, defaults to 3.0):
                The sigma shift applied to the timestep schedule.
            generator (`torch.Generator`, *optional*):
                A random number generator for reproducible generation.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            cond_latent (`torch.Tensor`, *optional*):
                Clean temporal-prefix condition latent (e.g. the ti2v first-frame
                latent), re-injected after every scheduler step.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-encoded prompt embeddings, ``prompt_mask`` is required.
            prompt_mask (`torch.Tensor`, *optional*):
                Validity mask of ``prompt_embeds``.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-encoded negative prompt embeddings.
            negative_prompt_mask (`torch.Tensor`, *optional*):
                Validity mask of ``negative_prompt_embeds``.
            t_thresh (`float`, *optional*):
                Threshold for the two-stage (refiner) sigma schedule.
            refiner_sigma_tail_steps (`int`, *optional*):
                Number of low-noise tail steps of the refiner sigma schedule.
            output_type (`str`, *optional*, defaults to "pt"):
                The output format, "pt" for pixel values or "latent".
            return_dict (`bool`, *optional*, defaults to True):
                Whether or not to return a [`LingBotVideoPipelineOutput`] instead
                of a plain tuple.

        Returns:
            [`LingBotVideoPipelineOutput`] or `tuple`:
                The generated video.
        """
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(height, width, num_frames)
        if self.transformer is None or self.scheduler is None:
            raise ValueError("`transformer` and `scheduler` are required for generation.")

        # 2. Default call parameters
        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._interrupt = False

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if prompt_embeds is not None:
            if prompt_mask is None:
                raise ValueError("`prompt_mask` is required when `prompt_embeds` is provided.")
            prompt_embeds = prompt_embeds.to(device=device)
            prompt_mask = prompt_mask.to(device=device)
        if negative_prompt_embeds is not None:
            if negative_prompt_mask is None:
                raise ValueError(
                    "`negative_prompt_mask` is required when `negative_prompt_embeds` is provided."
                )
            negative_prompt_embeds = negative_prompt_embeds.to(device=device)
            negative_prompt_mask = negative_prompt_mask.to(device=device)

        # 3. Encode input prompt
        if prompt_embeds is None:
            prompt_embeds, prompt_mask = self.encode_prompt(prompt, device=device)
        if do_classifier_free_guidance:
            if negative_prompt_embeds is not None:
                negative_embeds, negative_mask = negative_prompt_embeds, negative_prompt_mask
            else:
                negative_embeds, negative_mask = self.encode_prompt(negative_prompt, device=device)

        # 4. Prepare timesteps
        sigmas = compute_refiner_sigmas(
            sigma_max=float(self.scheduler.sigma_max),
            sigma_min=float(self.scheduler.sigma_min),
            num_inference_steps=num_inference_steps,
            shift=shift,
            t_thresh=t_thresh,
            tail_steps=refiner_sigma_tail_steps,
        )
        if sigmas is None:
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        else:
            self.scheduler.set_timesteps(
                int(sigmas.shape[0]),
                device=device,
                sigmas=sigmas,
                shift=1.0,
            )
        transformer_dtype = _module_dtype(self.transformer)

        # 5. Prepare latents
        latents = self.prepare_latents(num_frames, height, width, generator, latents, device)
        # Clean temporal-prefix condition latent: written into the latent before
        # sampling and after every scheduler step, so the fixed frames stay clean
        # while the rest denoise against them through attention.
        if cond_latent is not None:
            cond_latent = cond_latent.to(device=device, dtype=torch.float32)
            latents = self._apply_inpainting(latents, cond_latent)

        # 6. Denoising loop
        with self.progress_bar(total=len(self.scheduler.timesteps)) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                if self.interrupt:
                    continue

                latent_model_input = latents

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = _transformer_timestep(t, transformer_dtype).expand(latent_model_input.shape[0]).to(device)

                # predict noise model_output
                prompt_model_input = prompt_embeds.to(transformer_dtype)
                with _transformer_autocast(device, transformer_dtype):
                    noise_pred = self.transformer(
                        latent_model_input,
                        timestep,
                        prompt_model_input,
                        encoder_attention_mask=prompt_mask,
                        return_dict=False,
                    )[0].float()

                # perform guidance
                if do_classifier_free_guidance:
                    negative_model_input = negative_embeds.to(transformer_dtype)
                    with _transformer_autocast(device, transformer_dtype):
                        noise_pred_uncond = self.transformer(
                            latent_model_input,
                            timestep,
                            negative_model_input,
                            encoder_attention_mask=negative_mask,
                            return_dict=False,
                        )[0].float()
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    return_dict=False,
                    generator=generator,
                )[0]
                if cond_latent is not None:
                    latents = self._apply_inpainting(latents, cond_latent)

                progress_bar.update()

        # 7. Decode latents
        if output_type == "latent":
            video = latents
        else:
            video = self.decode_latents(latents)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return video

        return LingBotVideoPipelineOutput(videos=video)
