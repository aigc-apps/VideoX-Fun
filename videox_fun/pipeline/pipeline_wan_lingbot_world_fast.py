# Modified from https://github.com/guandeh17/Self-Forcing/blob/main/pipeline/causal_diffusion_inference.py and https://github.com/Robbyant/lingbot-world/blob/main/wan/image2video_fast.py
#
# LingBot-World (camera-pose controlled) causal streaming inference pipeline.
# Reference: repo/lingbot-world/wan/image2video_fast.py
#
# Mirrors the lingbot-world fast (Self-Forcing distilled) I2V flow: block-wise
# autoregressive denoising with a KV-cache, per-block plücker camera injection,
# and a flow-matching x0 resample schedule. Reuses WanSelfForcingPipeline
# helpers (T5 encode, KV/cross-attn cache init, VAE decode).

import inspect
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from ..models import WanTransformer3DModel_LingbotWorldFast
from ..data.utils import (compute_relative_poses, get_Ks_transformed,
                          get_plucker_embeddings, interpolate_camera_poses)
from ..utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                get_sampling_sigmas)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .pipeline_wan_self_forcing import (WanSelfForcingPipeline,
                                        WanSelfForcingPipelineOutput)


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
    Calls the scheduler's ``set_timesteps`` method and retrieves timesteps from the
    scheduler after the call. Handles custom timesteps / sigmas. Any kwargs will be
    supplied to ``scheduler.set_timesteps``.
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


class WanFunLingbotWorldFastPipeline(WanSelfForcingPipeline):
    r"""
    Pipeline for camera-controlled image-to-video generation using the
    lingbot-world fast (Self-Forcing distilled) transformer.

    This pipeline inherits from [`WanSelfForcingPipeline`] and reuses its helpers
    (T5 prompt encode, KV / cross-attention cache init, VAE decode). Generation
    proceeds block by block (``num_frame_per_block`` latent frames per block). For each
    block, a short flow-matching schedule denoises the noise
    to an ``x0`` prediction; the per-block plücker camera embedding is injected via
    ``dit_cond_dict`` and the clean block is written into the KV cache for the next
    block to attend to.
    """

    def _prepare_generic_timesteps(self, num_inference_steps, shift, device):
        r"""Build (and reset the state of) the generic scheduler schedule used by
        the non-stochastic fallback path.

        Dispatches by scheduler type exactly like the standard Wan pipelines.
        Re-invoked at the start of each causal block so the multistep solvers
        (UniPC / DPM++) restart their ``step_index`` and model-output history
        for every independent block denoise.
        """
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, _ = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, None, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler, device=device, sigmas=sampling_sigmas)
        else:
            timesteps, _ = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, None)
        return timesteps

    def _prepare_camera_plucker(self, action_path, control_type, frame_num,
                                lat_f, lat_h, lat_w, height, width, device, dtype):
        r"""
        Build the plücker camera embedding for the whole trajectory.

        Args:
            action_path (`str`): Directory with ``poses.npy`` / ``intrinsics.npy``.
            control_type (`str`): Camera control type ('cam').
            frame_num (`int`): Number of pixel-space frames to cover.
            lat_f / lat_h / lat_w (`int`): Latent temporal / spatial sizes.
            height / width (`int`): Output spatial size (must match intrinsics resize).
            device (`torch.device`): Target device.
            dtype (`torch.dtype`): Target dtype.

        Returns:
            Tensor: Plücker embedding of shape [1, C, lat_f, lat_h, lat_w], where C
                folds the VAE spatial stride into the plücker channels
                (6 * stride_h * stride_w).
        """
        # Load the camera-to-world poses (opencv coordinate) and intrinsics
        c2ws = np.load(f"{action_path}/poses.npy")
        c2ws = c2ws[:frame_num]

        Ks = torch.from_numpy(np.load(f"{action_path}/intrinsics.npy")).float()
        # Intrinsics are for the original (480p) size; transform to (height, width)
        Ks = get_Ks_transformed(
            Ks, height_org=480, width_org=832,
            height_resize=height, width_resize=width,
            height_final=height, width_final=width)
        Ks = Ks[0]

        # Interpolate the trajectory to the latent temporal length
        len_c2ws = len(c2ws)
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=np.linspace(0, len_c2ws - 1, lat_f),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        Ks = Ks.repeat(len(c2ws_infer), 1)

        # Compute the plücker embedding and fold the VAE spatial stride into channels
        c2ws_infer = c2ws_infer.to(device)
        Ks = Ks.to(device)
        c2ws_plucker_emb = get_plucker_embeddings(
            c2ws_infer, Ks, height, width, only_rays_d=False)
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(height // lat_h), c2=int(width // lat_w))
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb, 'b (f h w) c -> b c f h w',
            f=lat_f, h=lat_h, w=lat_w).to(dtype)
        return c2ws_plucker_emb

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        action_path: Optional[str] = None,
        control_type: str = "cam",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_frame_per_block: int = 3,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        stochastic_sampling: bool = True,
        shift: float = 10.0,
        num_train_timesteps: int = 1000,
        timesteps_index: Optional[List[int]] = None,
        generator: Optional[torch.Generator] = None,
        max_sequence_length: int = 512,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[WanSelfForcingPipelineOutput, tuple]:
        r"""
        Function invoked when calling the pipeline for camera-controlled causal
        image-to-video generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt to guide video generation.
            image (`PIL.Image.Image`):
                The reference first-frame image.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide video generation. Only used when
                ``guidance_scale`` > 1. NOTE: the distilled few-step model is
                trained without classifier-free guidance, so enabling CFG is off
                the training distribution and may degrade quality; it is exposed
                for interface parity with the non-distilled pipeline.
            action_path (`str`, *optional*):
                Directory with ``poses.npy`` / ``intrinsics.npy`` for the camera
                trajectory. When None, no camera condition is injected.
            control_type (`str`, *optional*, defaults to 'cam'):
                Camera control type.
            height (`int`, *optional*, defaults to 480):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to 832):
                The width in pixels of the generated video.
            num_frames (`int`, *optional*, defaults to 81):
                The number of frames to generate (4n + 1).
            num_frame_per_block (`int`, *optional*, defaults to 3):
                The number of latent frames generated per causal block.
            num_inference_steps (`int`, *optional*, defaults to 4):
                Number of denoising steps for the generic scheduler path
                (``stochastic_sampling=False``). When ``stochastic_sampling`` is
                True the step count is fixed by ``len(timesteps_index)`` (the
                calibrated 4-point grid), so this value is ignored there.
            guidance_scale (`float`, *optional*, defaults to 1.0):
                Classifier-free guidance scale. ``1.0`` disables CFG (the native
                distilled behavior). Values > 1 run an extra unconditional pass
                per step and combine via
                ``uncond + guidance_scale * (cond - uncond)``.
            stochastic_sampling (`bool`, *optional*, defaults to True):
                When True (default), use the native calibrated lingbot-world fast
                few-step schedule — the fixed-index FlowUniPC grid the distilled
                model was trained on. This is the CORRECT path and is required to
                reproduce reference quality. When False, fall back to the generic
                scheduler dispatch (retrieve_timesteps / set_timesteps by
                scheduler type), which the distilled weights were NOT calibrated
                on.
            shift (`float`, *optional*, defaults to 5.0):
                The flow-matching schedule shift.
            num_train_timesteps (`int`, *optional*, defaults to 1000):
                The number of training timesteps used to build the schedule.
            generator (`torch.Generator`, *optional*):
                A generator to make generation deterministic.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length for the text prompt.
            output_type (`str`, *optional*, defaults to "pil"):
                The output format ("pil" to decode to frames, otherwise latents).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`WanSelfForcingPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`WanSelfForcingPipelineOutput`] or `tuple`:
                When ``return_dict`` is True, returns a [`WanSelfForcingPipelineOutput`]
                whose ``videos`` has shape [B, C, F, H, W]; otherwise a tuple.
        """
        device = self._execution_device
        weight_dtype = self.text_encoder.dtype
        param_dtype = self.transformer.dtype

        # Classifier-free guidance is disabled by default (guidance_scale == 1.0)
        # because the distilled few-step model is trained without CFG. When
        # guidance_scale > 1 an extra unconditional path is run per step for
        # interface parity with the non-distilled pipeline.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt (and the negative prompt when CFG is enabled)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=weight_dtype,
        )
        context = [prompt_embeds[0]]
        context_null = [negative_prompt_embeds[0]] if do_classifier_free_guidance else None

        # 2. Preprocess the reference image to [3, H, W] in [-1, 1]
        img = TF.to_tensor(image).sub_(0.5).div_(0.5).to(device)

        # 3. Derive the latent shape; trim latent frames to a multiple of num_frame_per_block
        vae_stride = (
            self.vae.temporal_compression_ratio,
            self.vae.spatial_compression_ratio,
            self.vae.spatial_compression_ratio,
        )
        patch_size = self.transformer.config.patch_size
        frame_num = ((num_frames - 1) // 4) * 4 + 1
        if action_path is not None:
            c2ws_all = np.load(f"{action_path}/poses.npy")
            len_c2ws = ((len(c2ws_all) - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)

        lat_h = height // vae_stride[1]
        lat_w = width // vae_stride[2]
        lat_f = (frame_num - 1) // vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % num_frame_per_block))
        assert lat_f >= num_frame_per_block, \
            f"latent frames ({lat_f}) < num_frame_per_block ({num_frame_per_block}); increase num_frames"
        frame_num = (lat_f - 1) * vae_stride[0] + 1

        frame_seq_length = (lat_h * lat_w) // (patch_size[1] * patch_size[2])
        max_seq_len = num_frame_per_block * frame_seq_length

        # 4. Prepare noise (only for the frames to generate)
        noise = randn_tensor(
            (self.vae.latent_channels, lat_f, lat_h, lat_w),
            generator=generator, device=device, dtype=torch.float32)

        # 5. I2V conditioning: mask (4 channels after temporal fold) + VAE-encoded frame
        msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        img_frames = torch.concat([
            F.interpolate(img[None].cpu(), size=(height, width), mode='bicubic').transpose(0, 1),
            torch.zeros(3, frame_num - 1, height, width)
        ], dim=1).to(device)
        y = self.vae.encode(img_frames[None].to(self.vae.dtype)).latent_dist.mode()[0]
        y = torch.concat([msk, y.to(device)])

        # 6. Prepare the plücker camera embedding for the whole trajectory
        c2ws_plucker_emb = None
        if action_path is not None:
            c2ws_plucker_emb = self._prepare_camera_plucker(
                action_path, control_type, frame_num, lat_f, lat_h, lat_w,
                height, width, device, param_dtype)

        # 7. Prepare timesteps.
        #
        # stochastic_sampling=True is the CORRECT path for the distilled
        # lingbot-world fast model: build the FULL num_train_timesteps FlowUniPC
        # grid (with the runtime shift) and pick the tuned, non-uniformly spaced
        # indices. With the calibrated shift=10.0, indexing the 1000-step grid at
        # [0, 179, 358, 679] yields timesteps [999, 978, 947, 825] (sigmas
        # [0.9999, 0.9785, 0.947, 0.8252]) — the exact schedule the model was
        # trained on (image2video_fast.py + wan_i2v_A14B.py sample_shift=10.0).
        # These indices are an empirical distillation constant (NOT uniformly
        # spaced, not derivable from a formula); override via `timesteps_index`.
        #
        # Every other case keeps the original generic scheduler dispatch
        # (retrieve_timesteps / set_timesteps by scheduler type). The distilled
        # weights were NOT calibrated on those grids, so they are kept only for
        # experimentation / non-distilled schedulers and will look
        # grainy/painterly with the fast model.
        if stochastic_sampling:
            self.scheduler.set_timesteps(num_train_timesteps, device=device, shift=shift)
            _ti = timesteps_index if timesteps_index is not None else [0, 179, 358, 679]
            timesteps = self.scheduler.timesteps[_ti]
            # Append a trailing 0 so the resample can read timesteps[i + 1]; on the
            # final step t_{i+1}=0 collapses the latent to the x0 prediction.
            timesteps = torch.cat([timesteps, timesteps.new_zeros(1)])
        else:
            timesteps = self._prepare_generic_timesteps(
                num_inference_steps, shift, device)

        # 8. Initialize KV / cross-attention caches (single conditional path)
        num_latent_frames = lat_f
        self._initialize_kv_cache(
            batch_size=1, dtype=param_dtype, device=device,
            frame_seq_length=frame_seq_length, num_latent_frames=num_latent_frames)
        self._initialize_crossattn_cache(batch_size=1, dtype=param_dtype, device=device)

        # 9. Causal generation loop - block by block
        latents_block = noise.split(num_frame_per_block, dim=1)
        condition_block = y.split(num_frame_per_block, dim=1)
        cam_block = (c2ws_plucker_emb.split(num_frame_per_block, dim=2)
                     if c2ws_plucker_emb is not None else [None] * len(latents_block))
        num_blocks = len(latents_block)

        pred_latent_blocks = []
        for block_index in range(num_blocks):
            current_latent = latents_block[block_index].to(device)
            current_condition = condition_block[block_index]
            current_num_frames = current_latent.shape[1]

            # Camera condition for the current block
            dit_cond_dict = None
            if cam_block[block_index] is not None:
                dit_cond_dict = {"c2ws_plucker_emb": (cam_block[block_index],)}
            current_start = block_index * num_frame_per_block * frame_seq_length

            # For the generic (non-stochastic) fallback, restart the scheduler's
            # internal step_index / multistep history so each causal block
            # denoises from a fresh state (the stochastic path is stateless).
            if not stochastic_sampling:
                self._prepare_generic_timesteps(num_inference_steps, shift, device)

            # Denoise the block with the few-step schedule.
            denoise_timesteps = timesteps[:-1] if stochastic_sampling else timesteps
            with self.progress_bar(total=len(denoise_timesteps)) as progress_bar:
                for step_idx in range(len(denoise_timesteps)):
                    # Build the per-frame timestep in float32 so the exact schedule
                    # timestep reaches the model's time embedding (bf16 would round it).
                    timestep = torch.ones([1, current_num_frames], device=device,
                                          dtype=torch.float32) * denoise_timesteps[step_idx]
                    with torch.cuda.amp.autocast(dtype=param_dtype):
                        flow_pred = self.transformer(
                            x=[current_latent],
                            t=timestep,
                            context=context,
                            seq_len=max_seq_len,
                            y=[current_condition],
                            dit_cond_dict=dit_cond_dict,
                            kv_cache=self.kv_cache_pos,
                            crossattn_cache=self.crossattn_cache_pos,
                            current_start=current_start,
                            cache_start=None,
                        )[0]
                        if do_classifier_free_guidance:
                            # Unconditional path uses the negative context and its
                            # own KV / cross-attn caches so it does not pollute the
                            # conditional cache.
                            flow_pred_uncond = self.transformer(
                                x=[current_latent],
                                t=timestep,
                                context=context_null,
                                seq_len=max_seq_len,
                                y=[current_condition],
                                dit_cond_dict=dit_cond_dict,
                                kv_cache=self.kv_cache_neg,
                                crossattn_cache=self.crossattn_cache_neg,
                                current_start=current_start,
                                cache_start=None,
                            )[0]
                            flow_pred = flow_pred_uncond + guidance_scale * (flow_pred - flow_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    if stochastic_sampling:
                        # Native distilled few-step sampler. For this shifted
                        # flow-matching grid sigma == t / num_train_timesteps, so
                        # predict x0 from the velocity and renoise to the next
                        # timestep with fresh Gaussian noise. On the last step
                        # t_i_1 == 0, collapsing the latent to the x0 prediction.
                        t_i = (timesteps[step_idx] / self.scheduler.config.num_train_timesteps).to(weight_dtype)
                        t_i_1 = (timesteps[step_idx + 1] / self.scheduler.config.num_train_timesteps).to(weight_dtype)
                        x0 = current_latent - flow_pred * t_i
                        current_latent = (1 - t_i_1) * x0 + t_i_1 * torch.randn(
                            x0.shape, dtype=x0.dtype, device=device, generator=generator)
                    else:
                        # Generic fallback: delegate to the scheduler's own step()
                        # so each sampler (Flow Euler, UniPC, DPM++) applies its
                        # correct multi-step formula.
                        t = denoise_timesteps[step_idx]
                        x0 = self.scheduler.step(
                            flow_pred, t, current_latent, return_dict=False)[0]
                        current_latent = x0
                    progress_bar.update()

            pred_latent_blocks.append(x0)

            # Write the clean block into the KV cache (context timestep = 0) so the
            # next block can attend to it
            if block_index < num_blocks - 1:
                context_timestep = torch.zeros([1, current_num_frames], device=device,
                                               dtype=torch.float32)
                with torch.cuda.amp.autocast(dtype=param_dtype):
                    self.transformer(
                        x=[x0],
                        t=context_timestep,
                        context=context,
                        seq_len=max_seq_len,
                        y=[current_condition],
                        dit_cond_dict=dit_cond_dict,
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=current_start,
                        cache_start=None,
                    )
                    if do_classifier_free_guidance:
                        # Mirror the clean block into the unconditional caches too.
                        self.transformer(
                            x=[x0],
                            t=context_timestep,
                            context=context_null,
                            seq_len=max_seq_len,
                            y=[current_condition],
                            dit_cond_dict=dit_cond_dict,
                            kv_cache=self.kv_cache_neg,
                            crossattn_cache=self.crossattn_cache_neg,
                            current_start=current_start,
                            cache_start=None,
                        )

        pred_latents = torch.cat(pred_latent_blocks, dim=1)[None]

        # 10. Decode the latents to frames
        if output_type == "pil":
            video = self.decode_latents(pred_latents)
            video = torch.from_numpy(video)
        else:
            video = pred_latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)
        return WanSelfForcingPipelineOutput(videos=video)
