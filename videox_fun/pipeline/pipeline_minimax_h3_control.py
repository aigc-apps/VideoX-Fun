# Control variant of `pipeline_minimax_h3.py`, mirroring the VACE-style control pipeline of
# `pipeline_wan_vace.py`: the control video is encoded by the video VAE, patchified exactly like the target video
# and handed to `MiniMaxH3ControlTransformer3DModel` as `control_rows`, which injects zero-gated per-layer skips
# into the main block stack. Everything else — the packed layout, the two schedules, the audio branch — is the
# base pipeline's.

r"""
MiniMax-H3 control to video + audio pipeline.

The control video guides the generated video through the transformer's side branch rather than through the packed
sequence: it never occupies rows of its own, so the layout, the rotary grid and the two schedules are exactly the
ones of [`MiniMaxH3Pipeline`]. The control rows line up one-to-one with the video rows of the packed sequence, and
the released control checkpoints are trained on the `t2va` layout alone, so this pipeline covers text plus control
video only: a keyframe request (`image` / `last_image`) adds conditioning *video* rows and belongs to
[`MiniMaxH3Pipeline`].

A freshly initialised control branch is an identity (`after_proj` is zero), so a control model that has not been
trained yet reproduces the base model regardless of `control_context_scale`.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from .pipeline_minimax_h3 import (MINIMAX_H3_KEYFRAME_NOISE_AUG,
                                  MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD,
                                  MiniMaxH3Pipeline, MiniMaxH3PipelineOutput,
                                  _offload_scope, align_num_frames,
                                  audio_latent_num_frames,
                                  build_packed_sequence, build_row_timesteps,
                                  logger, patchify_video_latents,
                                  resolve_canvas_size,
                                  video_latent_num_frames)


class MiniMaxH3ControlPipeline(MiniMaxH3Pipeline):
    r"""
    Pipeline for joint video + audio generation with MiniMax-H3 under the guidance of a control video (pose, depth,
    canny, ...).

    Same components and same contract as [`MiniMaxH3Pipeline`], with a `transformer` that carries the control branch
    ([`MiniMaxH3ControlTransformer3DModel`]) and a `control_video` argument in
    [`~MiniMaxH3ControlPipeline.__call__`]. Keyframe conditioning is not part of it, see the module docstring.

    Note that [`~MiniMaxH3Pipeline.from_pretrained`] and [`~MiniMaxH3Pipeline.from_pretrained_original`] build a
    *base* transformer for the on-disk layouts they assemble by hand, so the pipeline is meant to be constructed
    from components, as `examples/minimax_h3_fun/predict_v2v_control.py` does; `__init__` rejects a base transformer
    rather than letting the control arguments fail deep inside the forward.
    """

    def __init__(
        self,
        vae,
        audio_vae,
        text_encoder,
        tokenizer,
        processor,
        transformer,
        scheduler,
        audio_scheduler,
    ):
        if not hasattr(transformer, "control_blocks"):
            raise ValueError(
                f"{type(self).__name__} needs a transformer with a control branch "
                f"(`MiniMaxH3ControlTransformer3DModel`), got {type(transformer).__name__}. Load the control "
                "variant, or use `MiniMaxH3Pipeline` for the base model."
            )
        super().__init__(
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
            audio_scheduler=audio_scheduler,
        )

    # The control branch follows the control video's actual length instead of padding a short one, so the base
    # pipeline's 5-second lower bound does not apply here; the ceiling stays, as the model was trained up to it.
    _min_duration = 0.0

    def _fit_video_to_canvas(
        self,
        pixels: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        name: str,
        warn_frames: bool = True,
    ) -> torch.Tensor:
        r"""
        Pad / truncate a `(1, C, num_frames, H, W)` video tensor to `num_frames` frames and bilinearly resize it
        onto the `(height, width)` canvas, mirroring the training collate's geometry for every conditioning input.
        """
        num_input_frames = pixels.shape[2]
        if num_input_frames < num_frames:
            if warn_frames:
                logger.warning(
                    f"The {name} holds {num_input_frames} frames but the request generates {num_frames}; "
                    "repeating its last frame to fill the tail."
                )
            tail = pixels[:, :, -1:].expand(-1, -1, num_frames - num_input_frames, -1, -1)
            pixels = torch.cat([pixels, tail], dim=2)
        elif num_input_frames > num_frames:
            if warn_frames:
                logger.warning(
                    f"The {name} holds {num_input_frames} frames but the request generates {num_frames}; "
                    "dropping the tail."
                )
            pixels = pixels[:, :, :num_frames]

        if pixels.shape[-2:] != (height, width):
            logger.warning(
                f"Resizing the {name} from {tuple(pixels.shape[-2:])} to the {(height, width)} canvas of the "
                "request."
            )
            frames = F.interpolate(
                pixels[0].permute(1, 0, 2, 3), size=(height, width), mode="bilinear", align_corners=False
            )
            pixels = frames.permute(1, 0, 2, 3)[None]
        return pixels

    def encode_control_video(
        self,
        control_video: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        r"""
        Encode the control video into the clean, patchified rows the control branch reads.

        The recipe is the one the control branch was trained on (see `scripts/minimax_h3_fun/train_control.py`): the
        control video goes through the exact same geometry and normalization as the target video, the posterior is
        taken at its *mode* (deterministic conditioning, unlike the sampled keyframe conditioning) and the latents
        are normalized with the VAE's `latents_mean` / `latents_std`.

        Args:
            control_video (`torch.Tensor` of shape `(1, 3, num_frames, height, width)`):
                The control video in the `[0, 1]` range, e.g. the first return value of
                `videox_fun.utils.utils.get_video_to_video_latent`.
            height (`int`): Height of the target canvas.
            width (`int`): Width of the target canvas.
            num_frames (`int`):
                Number of frames the request generates, already snapped to `17 * n + 5`. A shorter control video is
                extended by repeating its last frame, a longer one is truncated.
            device (`torch.device`, *optional*): The device to run the VAE on.

        Returns:
            `torch.Tensor` of shape `(num_video_rows, latent_channels * prod(patch_size))`: the float32 control rows.
        """
        device = device or self._execution_device
        if control_video.ndim != 5 or control_video.shape[0] != 1 or control_video.shape[1] != 3:
            raise ValueError(
                "`control_video` must be a `(1, 3, num_frames, height, width)` tensor in the [0, 1] range, got "
                f"{list(control_video.shape)}."
            )
        pixels = self._fit_video_to_canvas(
            control_video.to(device=device, dtype=torch.float32), height, width, num_frames, "control video"
        )

        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
        pixels = (pixels - pixel_mean) / pixel_std

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        # `encode` is reached as a method call, so the top-level CPU-offload hook is fired by hand around it, and the
        # encode runs under float16 autocast even though the VAE weights are float32, as in training.
        with _offload_scope(self.vae), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            control_latents = self.vae.encode(pixels).latent_dist.mode()
        control_latents = (control_latents.float() - latents_mean) / latents_std
        return patchify_video_latents(control_latents, self.patch_size)

    def encode_inpaint_condition(
        self,
        mask: torch.Tensor,
        inpaint_video: Optional[torch.Tensor],
        height: int,
        width: int,
        num_frames: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        r"""
        Build the inpaint rows an `--enable_inpaint` control checkpoint reads on top of the control rows.

        Mirrors the training recipe of `scripts/minimax_h3_fun/train_control.py`: the visibility map `1 - mask` is
        trilinearly resized straight onto the latent grid (no first-frame split — MiniMax-H3's chunked VAE encodes
        its first 5 frames into 2 latents, so the Wan causal-VAE packing has no counterpart here), the masked video
        `inpaint_video * (1 - mask)` is VAE-encoded at its posterior mode, and the two are patchified and
        concatenated along the channel columns, visibility map first.

        Args:
            mask (`torch.Tensor` of shape `(1, 1, num_frames, height, width)`):
                The inpaint mask in the `[0, 1]` range; `1` marks the regions to regenerate, `0` the regions the
                `inpaint_video` content is kept in.
            inpaint_video (`torch.Tensor` of shape `(1, 3, num_frames, height, width)`, *optional*):
                The source video behind the mask in the `[0, 1]` range. `None` leaves nothing behind the visible
                regions (zeroed masked pixels).
            height (`int`): Height of the target canvas.
            width (`int`): Width of the target canvas.
            num_frames (`int`): Number of frames the request generates, already snapped to `17 * n + 5`.
            device (`torch.device`, *optional*): The device to run the VAE on.

        Returns:
            `torch.Tensor` of shape `(num_video_rows, (1 + latent_channels) * prod(patch_size))`: the float32
            inpaint rows, in the training order (visibility map, then masked-video latents).
        """
        device = device or self._execution_device
        if mask.ndim != 5 or mask.shape[0] != 1 or mask.shape[1] != 1:
            raise ValueError(
                "`mask` must be a `(1, 1, num_frames, height, width)` tensor in the [0, 1] range, got "
                f"{list(mask.shape)}."
            )
        # Training only ever sees hard {0, 1} masks at pixel resolution (`get_random_mask`), softened once by the
        # trilinear drop onto the latent grid; binarize so a caller-supplied soft or resampled mask lands back in
        # that distribution instead of half-zeroing pixels.
        mask = (mask.to(device=device, dtype=torch.float32) > 0.5).to(torch.float32)
        mask = self._fit_video_to_canvas(
            mask, height, width, num_frames, "mask", warn_frames=False
        )
        # `_fit_video_to_canvas` resizes bilinearly when the mask misses the canvas; re-harden the smeared edges.
        mask = (mask > 0.5).to(torch.float32)

        if inpaint_video is not None:
            if inpaint_video.ndim != 5 or inpaint_video.shape[0] != 1 or inpaint_video.shape[1] != 3:
                raise ValueError(
                    "`inpaint_video` must be a `(1, 3, num_frames, height, width)` tensor in the [0, 1] range, got "
                    f"{list(inpaint_video.shape)}."
                )
            masked_pixels = self._fit_video_to_canvas(
                inpaint_video.to(device=device, dtype=torch.float32),
                height, width, num_frames, "inpaint video", warn_frames=False,
            )
            masked_pixels = masked_pixels * (1 - mask)
        else:
            logger.warning("No `inpaint_video` given: the visible regions behind the mask carry no content.")
            masked_pixels = torch.zeros_like(mask.expand(-1, 3, -1, -1, -1))

        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
        masked_pixels = (masked_pixels - pixel_mean) / pixel_std

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        with _offload_scope(self.vae), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            mask_latents = self.vae.encode(masked_pixels).latent_dist.mode()
        mask_latents = (mask_latents.float() - latents_mean) / latents_std

        # The visibility map goes straight from pixel frames to the latent grid, exactly as in the training loop.
        mask_condition = F.interpolate(
            1 - mask, size=mask_latents.shape[2:], mode="trilinear", align_corners=False
        )
        mask_condition_rows = patchify_video_latents(mask_condition, self.patch_size)
        mask_latent_rows = patchify_video_latents(mask_latents, self.patch_size)
        return torch.cat([mask_condition_rows, mask_latent_rows], dim=-1)

    def align_control_rows_width(self, control_rows: torch.Tensor) -> torch.Tensor:
        r"""
        Match the control rows to the `control_proj_in` width of the loaded checkpoint. An `--enable_inpaint`
        checkpoint is widened with the mask channels (`control_in_dim` above the video latent channels); when the
        rows carry fewer columns the mask channels are zero-padded — the all-zero mask channels are the
        pure-generation layout the training loop drops fully masked batches to — and when they carry more the
        checkpoint cannot read them, so fail loudly instead of truncating trained channels away.
        """
        control_in_dim = getattr(self.transformer.config, "control_in_dim", None) or self.vae_latent_channels
        patch_columns = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        expected_columns = control_in_dim * patch_columns
        if control_rows.shape[-1] < expected_columns:
            control_rows = F.pad(control_rows, (0, expected_columns - control_rows.shape[-1]))
        elif control_rows.shape[-1] > expected_columns:
            raise ValueError(
                f"The control rows carry {control_rows.shape[-1]} columns but the checkpoint's `control_in_dim` "
                f"({control_in_dim}) expects {expected_columns}."
            )
        return control_rows

    @torch.no_grad()
    def __call__(
        self,
        prompt: str = None,
        control_video: Optional[torch.Tensor] = None,
        control_context_scale: float = 1.0,
        mask_video: Optional[torch.Tensor] = None,
        inpaint_video: Optional[torch.Tensor] = None,
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
        Generate a video and its soundtrack under the guidance of a control video.

        Same contract as [`~MiniMaxH3Pipeline.__call__`], minus the keyframe arguments (`image` / `last_image`), plus:

        Args:
            control_video (`torch.Tensor` of shape `(1, 3, num_frames, height, width)`, *optional*):
                The control video in the `[0, 1]` range. `None` disables the control branch, which makes the call
                identical to the base pipeline.
            control_context_scale (`float`, defaults to `1.0`):
                Scale applied to every control skip before it is added to the main branch. `0.0` switches the branch
                off, values below `1.0` weaken the guidance of the control video.
            mask_video (`torch.Tensor` of shape `(1, 1, num_frames, height, width)`, *optional*):
                The inpaint mask in the `[0, 1]` range; `1` marks the regions to regenerate, `0` the regions whose
                `inpaint_video` content is kept. Only for checkpoints trained with `--enable_inpaint` (a widened
                `control_in_dim`); when the checkpoint is inpaint-capable but no mask is given, the mask channels
                are zero-padded, which the model reads as pure generation. Without a `control_video` the control
                channels are zeroed instead — the layout the training drop of the control rows covers — so the
                mask works on its own.
            inpaint_video (`torch.Tensor` of shape `(1, 3, num_frames, height, width)`, *optional*):
                The source video behind the mask in the `[0, 1]` range, e.g. the first return value of
                `videox_fun.utils.utils.get_video_to_video_latent`. Only read when `mask_video` is given; `None`
                leaves nothing behind the visible regions.

        Returns:
            [`MiniMaxH3PipelineOutput`] or `tuple`:
                The generated video, the stereo soundtrack of shape `(1, 2, num_samples)` and its sample rate.
        """
        self.check_inputs(prompt, height, width, num_frames, num_inference_steps)
        self._attention_kwargs = attention_kwargs
        device = self._execution_device

        # 1. Resolve the plan: the canvas, the frame count the video VAE can decode and the latent geometry every
        # later step keys off. The layout is the `t2va` one — no keyframe conditioning rows — so its video rows are
        # the generated rows alone and the control rows cover them one-to-one.
        if height is None:
            height, width = resolve_canvas_size(16, 9)

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

        # 2. Encode MiniMax-H3's presentation of the request. The released checkpoint is guidance-distilled, so the
        # default guidance_scale of 1 runs one forward pass per step with no CFG; a guidance_scale above 1 enables
        # classifier-free guidance with a negative prompt. The control rows are handed to both passes.
        do_cfg = guidance_scale > 1.0
        prompt_embeds, text_token_tags = self.encode_prompt(
            prompt, [], device=device, dtype=self.transformer.dtype
        )
        if do_cfg:
            negative_prompt = negative_prompt if negative_prompt is not None else ""
            negative_prompt_embeds, negative_text_token_tags = self.encode_prompt(
                negative_prompt, [], device=device, dtype=self.transformer.dtype
            )

        # 3. Build the packed layout and its fp64 rotary grid.
        layout = build_packed_sequence(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            self.patch_size,
            (),
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
                (),
            )
            negative_position_ids = negative_layout.position_ids.to(device)
            negative_token_tags = negative_layout.token_tags.to(device)
            negative_video_indices = negative_layout.video_indices.to(device)
            negative_audio_indices = negative_layout.audio_indices.to(device)
            negative_text_indices = negative_layout.text_indices.to(device)

        # 4. Encode the control video into the clean rows the side branch reads. They are fixed for the whole loop.
        # An `--enable_inpaint` checkpoint widens `control_proj_in` with the mask channels, so the rows are either
        # extended with the encoded mask (visibility map + masked-video latents) or zero-padded — mirroring the
        # training validation, where the all-zero mask channels read as pure generation. A mask without a control
        # video zero-fills the control channels instead, the layout training reaches when it drops the control
        # rows, so the inpaint condition stands on its own.
        control_in_dim = getattr(self.transformer.config, "control_in_dim", None) or self.vae_latent_channels
        if control_video is None and mask_video is not None and control_in_dim == self.vae_latent_channels:
            raise ValueError(
                "`mask_video` was given but the checkpoint's `control_in_dim` covers the video latents "
                "alone; it was not trained with `--enable_inpaint` and cannot read the mask channels."
            )
        control_rows = None
        if control_video is not None or mask_video is not None:
            if control_video is not None:
                control_rows = self.encode_control_video(control_video, height, width, num_frames, device=device)
            else:
                patch_columns = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
                control_rows = torch.zeros(
                    video_indices.shape[0], self.vae_latent_channels * patch_columns,
                    device=device, dtype=torch.float32,
                )
            if control_rows.shape[0] != video_indices.shape[0]:
                raise ValueError(
                    f"The control video maps to {control_rows.shape[0]} rows but the packed sequence holds "
                    f"{video_indices.shape[0]} video rows. One control row per video row is required, in the same "
                    "order."
                )
            if mask_video is not None:
                inpaint_rows = self.encode_inpaint_condition(
                    mask_video, inpaint_video, height, width, num_frames, device=device
                )
                control_rows = torch.cat([control_rows, inpaint_rows], dim=-1)
            control_rows = self.align_control_rows_width(control_rows)
            control_rows = control_rows[None].to(device)

        # 5. Draw the noise of the generated rows.
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

        # 6. Initialize the two schedules and stage the row-to-timestep plan of every step. One forward serves every
        # modality and every noise level at once.
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

        # 7. Denoise the packed sequence over the two schedules, with the control rows fixed.
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
                    control_rows=control_rows,
                    control_context_scale=control_context_scale,
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
                        control_rows=control_rows,
                        control_context_scale=control_context_scale,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)
                    audio_noise_pred = neg_audio_noise_pred + guidance_scale * (
                        audio_noise_pred - neg_audio_noise_pred
                    )

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
