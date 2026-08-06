# Modified from https://github.com/Robbyant/lingbot-video/blob/main/lingbot_video/pipeline_lingbot_video_i2v.py
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .pipeline_lingbot_video import (DEFAULT_NEGATIVE_PROMPT,
                                     LingBotVideoPipeline,
                                     LingBotVideoPipelineOutput, _module_dtype,
                                     _transformer_autocast,
                                     _transformer_timestep)

IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[int, int]:
    max_pixels = max_pixels if max_pixels is not None else IMAGE_MAX_TOKEN_NUM * factor**2
    min_pixels = min_pixels if min_pixels is not None else IMAGE_MIN_TOKEN_NUM * factor**2
    if max_pixels < min_pixels:
        raise ValueError("max_pixels must be greater than or equal to min_pixels.")
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}.")

    resized_height = max(factor, _round_by_factor(height, factor))
    resized_width = max(factor, _round_by_factor(width, factor))
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = _floor_by_factor(height / beta, factor)
        resized_width = _floor_by_factor(width / beta, factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * beta, factor)
        resized_width = _ceil_by_factor(width * beta, factor)
    return resized_height, resized_width


def _pixel_tensor_to_pil(pixel: torch.Tensor) -> Image.Image:
    """Match torchvision.transforms.ToPILImage for a float CHW image in [0, 1]."""
    frame = pixel[0, :, 0].detach().cpu().clamp(0, 1)
    array = frame.permute(1, 2, 0).mul(255).byte().numpy()
    return Image.fromarray(array, mode="RGB")


class LingBotVideoI2VPipeline(LingBotVideoPipeline):
    r"""
    LingBot-Video ti2v pipeline.

    The condition frame is used twice: as visual input for Qwen3-VL and as a
    clean latent that is written into the beginning of the diffusion latent
    before sampling and after every scheduler step.
    """

    def preprocess_image(self, image: Image.Image, height: int, width: int) -> torch.Tensor:
        if image is None:
            raise ValueError("`image` is required when `image_tensor` is not provided.")
        raw = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).contiguous()
        old_h, old_w = raw.shape[-2:]
        scale = max(height / old_h, width / old_w)
        new_h = max(math.ceil(old_h * scale), height)
        new_w = max(math.ceil(old_w * scale), width)
        resized = F.interpolate(raw.float(), size=(new_h, new_w), mode="bilinear", align_corners=False)
        top = int(round((new_h - height) / 2.0))
        left = int(round((new_w - width) / 2.0))
        cropped = resized[:, :, top : top + height, left : left + width].float() / 255.0
        return cropped.unsqueeze(2)  # (1, C, 1, H, W)

    def _vision_patch_size(self) -> int:
        for obj in (
            getattr(getattr(self.text_encoder, "config", None), "vision_config", None),
            getattr(getattr(self.processor, "image_processor", None), "config", None),
            getattr(self.processor, "image_processor", None),
        ):
            patch = getattr(obj, "patch_size", None)
            if patch is not None:
                return int(patch)
        return 16

    def _vlm_image(self, pixel: torch.Tensor) -> Image.Image:
        image = _pixel_tensor_to_pil(pixel)
        patch_factor = self._vision_patch_size() * SPATIAL_MERGE_SIZE
        width, height = image.size
        resized_height, resized_width = smart_resize(height, width, factor=patch_factor)
        return image.resize((resized_width, resized_height))

    @torch.no_grad()
    def encode_image_latent(
        self,
        pixel: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.vae is None:
            raise ValueError("`vae` is required to encode image latents.")
        # See decode_latents: use the onload device, not the parameter device.
        device = self._execution_device
        vae_dtype = _module_dtype(self.vae)
        pixel = pixel.to(device=device, dtype=torch.float32)
        norm_pixel = (pixel - 0.5) / 0.5
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            latents = self.vae.encode(norm_pixel.to(vae_dtype)).latent_dist.sample(generator)
        return self._vae_latent_to_dit(latents).to(latents)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        image_tensor: Optional[torch.Tensor] = None,
        cond_latent: Optional[torch.Tensor] = None,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 3.0,
        shift: float = 3.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_mask: Optional[torch.Tensor] = None,
        output_type: str = "pt",
        return_dict: bool = True,
    ) -> Union[LingBotVideoPipelineOutput, Tuple]:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str`):
                The prompt to guide the video generation.
            image (`PIL.Image.Image`, *optional*):
                The first-frame condition image.
            image_tensor (`torch.Tensor`, *optional*):
                Preprocessed first-frame condition tensor, used instead of ``image``.
            cond_latent (`torch.Tensor`, *optional*):
                Pre-encoded first-frame condition latent; when None it is encoded
                from the condition image.
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
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-encoded prompt embeddings, ``prompt_mask`` is required.
            prompt_mask (`torch.Tensor`, *optional*):
                Validity mask of ``prompt_embeds``.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-encoded negative prompt embeddings.
            negative_prompt_mask (`torch.Tensor`, *optional*):
                Validity mask of ``negative_prompt_embeds``.
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

        pixel = image_tensor if image_tensor is not None else self.preprocess_image(image, height, width)
        pixel = pixel.to(device=device, dtype=torch.float32)
        vlm_image = self._vlm_image(pixel)

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

        # 3. Encode input prompt (with the first-frame visual tokens)
        if prompt_embeds is None:
            prompt_embeds, prompt_mask = self.encode_prompt(prompt, images=[vlm_image], device=device)
        if do_classifier_free_guidance:
            if negative_prompt_embeds is not None:
                negative_embeds, negative_mask = negative_prompt_embeds, negative_prompt_mask
            else:
                negative_embeds, negative_mask = self.encode_prompt(
                    negative_prompt, images=[vlm_image], device=device
                )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        transformer_dtype = _module_dtype(self.transformer)

        # 5. Prepare latents
        if cond_latent is None:
            cond_latent = self.encode_image_latent(pixel, generator=generator)
        cond_latent = cond_latent.to(device=device, dtype=torch.float32)

        latents = self.prepare_latents(num_frames, height, width, generator, latents, device)
        # Clean temporal-prefix condition latent: written into the latent before
        # sampling and after every scheduler step, so the first frame stays clean
        # while the rest denoise against it through attention.
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
