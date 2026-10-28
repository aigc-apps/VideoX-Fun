# Flex-Forcing: Towards a Unified Autoregressive and Bidirectional Video
# Diffusion Model (arXiv 2607.03509) - inference pipeline.
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import BaseOutput, logging, replace_example_docstring

from ..models import (AutoencoderKLWan, AutoTokenizer, WanT5EncoderModel,
                      WanTransformer3DModel_FlexForcing)
from ..utils.flex_chunking import (build_full_then_blocks_partitions,
                                   build_pyramid_partitions, chunk_boundaries,
                                   normalize_chunk_spec, uniform_chunks,
                                   validate_nested_partitions)
from ..utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                get_sampling_sigmas)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .pipeline_wan_self_forcing import (WanSelfForcingPipeline,
                                        WanSelfForcingPipelineOutput,
                                        retrieve_timesteps,
                                        stochastic_sampling_timesteps)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# `denoise_mode` strings that select the "first step full, every later step
# block-major" ladder built by `build_full_then_blocks_partitions`, i.e. one
# bidirectional planning chunk followed by the uniform `num_frame_per_block`
# partition. Kept as a tuple so the aliases stay in one place.
FULL_THEN_BLOCKS_MODES = ("full_then_blocks", "full_first", "one_shot")

# The partitions evaluated in the paper (5 s / 21 latent frames and the long
# video regimes), kept as a reference for choosing `chunk_spec`. Any list of
# positive ints summing to the latent frame count is accepted.
PAPER_CHUNK_CONFIGS: Tuple[Tuple[int, ...], ...] = (
    (21,),
    (18, 3),
    (15, 3, 3),
    (12, 6, 3),
    (11, 10),
    (10, 11),
    (9, 9, 3),
    (8, 13),
    (8, 8, 5),
    (7, 7, 7),
    (9, 6, 6),
    (6, 5, 5, 5),
    (5, 4, 4, 4),
    (3, 3, 3, 3, 3, 3, 3),
    (3, 2, 2, 2, 2, 2, 2, 2, 2, 2),
)

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> # One bidirectional planning step, then autoregressive refinement
        >>> pipe = WanFlexForcingPipeline.from_pretrained(...)
        >>> video = pipe(prompt="a cat surfing a wave", num_frames=81,
        ...              num_inference_steps=2, denoise_mode="pyramid").videos
        ```
"""


@dataclass
class WanFlexForcingPipelineOutput(BaseOutput):
    r"""
    Output class for Flex-Forcing pipelines.

    Args:
        videos (`torch.Tensor` or `np.ndarray`):
            Generated frames, matching :class:`WanSelfForcingPipelineOutput`.
    """

    videos: torch.Tensor


class WanFlexForcingPipeline(WanSelfForcingPipeline):
    r"""
    Pipeline for Flex-Forcing (arXiv 2607.03509) video generation.

    Inherits from [`WanSelfForcingPipeline`]; see its documentation for the
    shared arguments. The extra knobs select the frame partition (3.1) and the
    pyramid ladder (3.2). Without a partition the call is delegated verbatim to
    the inherited Self-Forcing rollout; with one, :meth:`__call__` drives the
    flexible order itself, in the coarse-to-fine walk at its end.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: WanTransformer3DModel_FlexForcing,
        scheduler,
    ):
        super().__init__(tokenizer=tokenizer, text_encoder=text_encoder,
                         vae=vae, transformer=transformer, scheduler=scheduler)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
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
        shift: float = 5.0,
        initial_latent: Optional[torch.FloatTensor] = None,
        start_frame_index: int = 0,
        num_frame_per_block: int = 1,
        independent_first_frame: bool = True,
        context_noise: int = 0,
        stochastic_sampling: bool = True,
        streaming: bool = False,
        decode_callback: Optional[Callable[[torch.Tensor, int], None]] = None,
        forcing_kv_enable: Optional[bool] = None,
        forcing_kv_head_profile: Optional[Union[str, Dict[str, Any]]] = None,
        forcing_kv_ar_start: Optional[int] = None,
        forcing_kv_spatial_context_length: Optional[int] = None,
        forcing_kv_temporal_context_length: Optional[int] = None,
        forcing_kv_dynamic_context_length: Optional[int] = None,
        forcing_kv_num_frame_patch: Optional[int] = None,
        forcing_kv_sim_retention_ratio: Optional[float] = None,
        # The Flex-Forcing knobs come last so that every inherited argument
        # keeps exactly the positional index it has in
        # `WanSelfForcingPipeline.__call__`.
        chunk_spec: Union[None, int, str, Sequence[int]] = None,
        denoise_mode: Union[str, int] = "fixed",
        min_num_frame_per_block: int = 1,
    ) -> Union[WanFlexForcingPipelineOutput, Tuple]:
        r"""
        Generate a video with a flexible frame partition and, optionally, a
        pyramid of nested partitions.

        Args:
            chunk_spec: Partition of the latent frame axis (Flex-Forcing 3.1).
                Accepts ``[11, 10]``, ``"15-3-3"``, ``"uniform:3"``, ``"ar"``
                (``[1] * F``), ``"bidir"`` (``[F]``) or an int for uniform
                chunks. ``None`` keeps the inherited ``num_frame_per_block`` /
                ``independent_first_frame`` schedule untouched.
                The uniform forms (``None``, an int, ``"uniform[:N]"``) honour
                ``independent_first_frame`` - which defaults to ``True`` here -
                so ``"uniform:3"`` yields ``[1, 3, 3, ...]``, matching the layout
                an IFF-trained checkpoint expects. Explicit size lists are taken
                verbatim: write the leading ``1`` yourself if you want it.
            denoise_mode: Does the frame partition change with the noise level?
                ``"fixed"`` (default) holds one partition across every denoising
                step, i.e. the block-major Self-Forcing / 3.1 schedule.
                ``"pyramid"`` runs 3.2: one nested level per denoising step,
                coarse (planning) to fine (refinement), with the depth taken
                from ``num_inference_steps`` so the two never need syncing. An
                int pins a truncated pyramid of exactly that many levels, e.g.
                ``2`` for the paper's ``[21] -> [11, 10]`` ladder on a 5 s clip.
                ``"full_then_blocks"`` (aliases ``"full_first"`` /
                ``"one_shot"``) runs a fixed two-level ladder instead: the first
                step plans the whole clip in one bidirectional chunk and every
                later step reuses the uniform ``num_frame_per_block`` partition.
            min_num_frame_per_block: Block size the ladder stops refining at -
                every level-0 chunk is binary-split until it is at or below this
                size. ``1`` lets the pyramid reach fully causal (single-frame)
                leaves; a larger value keeps the leaves block-major instead.

        The 3.3 K-Projection deliberately has no runtime knob: it is applied
        whenever the transformer was built with one, and building it with
        ``flex_kproj_mode='none'`` is the only way to leave it out.

        Every other argument is spelled out with the same name, type and default
        it has in :meth:`WanSelfForcingPipeline.__call__` - see its docs for
        what they mean. They are listed explicitly rather than gathered into
        ``*args`` / ``**kwargs`` so this signature stays introspectable, IDE
        completion works, and the inherited positional order is preserved.

        Examples:
            ```python
            pass
            ```
        """
        # `denoise_mode` -> ladder depth, i.e. how many nested partitions.
        #   "fixed"            -> 1: one partition held for every denoising step.
        #   "pyramid"          -> one level per denoising step (3.2), so the caller
        #                         never has to keep two numbers in sync. The ladder
        #                         stops earlier at its fixed point; asking for more
        #                         levels than the partition can yield is harmless.
        #   "full_then_blocks" -> 2 levels, built specially below: the first step
        #                         plans the whole clip in one bidirectional chunk,
        #                         every later step reuses the uniform
        #                         `num_frame_per_block` partition.
        #   int                -> an explicitly pinned pyramid depth, for callers
        #                         that must run a truncated pyramid (e.g. the
        #                         trainer's `--flex_pyramid_levels`).
        ladder_mode = (denoise_mode.strip().lower()
                       if isinstance(denoise_mode, str) else None)
        if ladder_mode is not None:
            if ladder_mode == "fixed":
                depth = 1
            elif ladder_mode == "pyramid":
                depth = max(1, int(num_inference_steps or 1))
            elif ladder_mode in FULL_THEN_BLOCKS_MODES:
                depth = 2
            else:
                raise ValueError(
                    "denoise_mode must be 'fixed', 'pyramid', 'full_then_blocks' "
                    f"or an int >= 1, got {denoise_mode!r}")
        else:
            depth = int(denoise_mode)
            if depth < 1:
                raise ValueError(
                    "denoise_mode must be 'fixed', 'pyramid', 'full_then_blocks' "
                    f"or an int >= 1, got {denoise_mode!r}")

        if chunk_spec is None and depth <= 1:
            # Nothing Flex-Forcing specific was asked for: hand the call to the
            # inherited Self-Forcing rollout untouched, which is why that base
            # class carries no Flex-Forcing knob at all. Only the output is
            # normalised to this pipeline's class on the way back.
            inherited = super().__call__(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                num_videos_per_prompt=num_videos_per_prompt,
                eta=eta,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                callback_on_step_end=callback_on_step_end,
                attention_kwargs=attention_kwargs,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
                comfyui_progressbar=comfyui_progressbar,
                shift=shift,
                initial_latent=initial_latent,
                start_frame_index=start_frame_index,
                num_frame_per_block=num_frame_per_block,
                independent_first_frame=independent_first_frame,
                context_noise=context_noise,
                stochastic_sampling=stochastic_sampling,
                streaming=streaming,
                decode_callback=decode_callback,
                forcing_kv_enable=forcing_kv_enable,
                forcing_kv_head_profile=forcing_kv_head_profile,
                forcing_kv_ar_start=forcing_kv_ar_start,
                forcing_kv_spatial_context_length=forcing_kv_spatial_context_length,
                forcing_kv_temporal_context_length=forcing_kv_temporal_context_length,
                forcing_kv_dynamic_context_length=forcing_kv_dynamic_context_length,
                forcing_kv_num_frame_patch=forcing_kv_num_frame_patch,
                forcing_kv_sim_retention_ratio=forcing_kv_sim_retention_ratio,
            )
            # Present the inherited output under this pipeline's output class.
            if isinstance(inherited, WanSelfForcingPipelineOutput):
                return WanFlexForcingPipelineOutput(videos=inherited.videos)
            return inherited

        # Turn the chunk knobs into a validated ladder: `chunk_sizes` is the
        # level-0 partition, `partitions` the full ladder (`None` when no
        # pyramid was requested).
        #
        # `latent_frames` is the same quantity step 6 recomputes as
        # `num_latent_frames`; the ladder has to be sized before that geometry
        # block runs, because steps 7 and 8 read `max(chunk_sizes)`.
        latent_frames = (
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1)
        if initial_latent is not None:
            # The inherited I2V accounting subtracts latent input frames but
            # `prepare_latents` compresses the remainder again, so an explicit
            # partition cannot be reconciled with it. Fail loudly instead of
            # silently mis-sizing the rollout.
            raise ValueError(
                "Flex-Forcing partitions describe the frames this call "
                "generates and currently require T2V mode "
                "(`initial_latent=None`).")

        base = normalize_chunk_spec(chunk_spec, latent_frames,
                                    independent_first_frame=independent_first_frame) \
            if chunk_spec is not None else None
        if depth > 1:
            if ladder_mode in FULL_THEN_BLOCKS_MODES:
                # "First step full, every later step block-major": a fixed
                # two-level ladder whose fine level is the uniform
                # `num_frame_per_block` partition rather than a binary split, so
                # it does not go through `build_pyramid_partitions`.
                # [NOTE] This ladder is fully determined by `num_frame_per_block`:
                # it ignores both `chunk_spec` (level 0 is always the whole clip)
                # and `min_num_frame_per_block` (the fine level is the uniform
                # grid, not a binary split), so say so instead of dropping the
                # caller's explicit arguments silently.
                if chunk_spec is not None:
                    logger.warning(
                        "`denoise_mode=%r` ignores `chunk_spec=%r`: its level 0 is "
                        "always one chunk over the whole clip and its fine level "
                        "is the uniform `num_frame_per_block` partition.",
                        ladder_mode, chunk_spec)
                if min_num_frame_per_block != 1:
                    logger.warning(
                        "`denoise_mode=%r` ignores `min_num_frame_per_block=%r`: "
                        "the finest level is fixed by `num_frame_per_block`.",
                        ladder_mode, min_num_frame_per_block)
                partitions = build_full_then_blocks_partitions(
                    latent_frames, num_frame_per_block,
                    independent_first_frame=independent_first_frame)
            else:
                # Level 0 defaults to a single chunk over the whole clip - the
                # paper's "high-level planning" step - unless the caller pinned it.
                partitions = build_pyramid_partitions(
                    latent_frames, num_levels=int(depth),
                    min_num_frame_per_block=min_num_frame_per_block, base_chunks=base,
                    independent_first_frame=independent_first_frame)
            validate_nested_partitions(partitions, latent_frames)
            chunk_sizes = partitions[0]
        else:
            # `chunk_spec` cannot be None here: that case returned above.
            chunk_sizes = normalize_chunk_spec(
                chunk_spec, latent_frames,
                independent_first_frame=independent_first_frame)
            partitions = None

        local_attn_size = getattr(self.transformer.config, 'local_attn_size', -1)
        if partitions is not None:
            # A level change hands the next level a buffer that sits exactly at
            # `timesteps[step_idx + 1]`, and the finer level restarts at frame
            # 0, so `current_start` moves backwards. Both are safe against a
            # full KV cache and neither is against a rolling one, which evicts
            # by `current_start` deltas.
            #
            # The handoff itself is not tied to `stochastic_sampling`. The
            # re-noising branch defines the buffer as
            # `(1 - t_i_1) * x0 + t_i_1 * eps`; the deterministic branch
            # defines it as the Euler increment
            # `x + (sigma_next - sigma_t) * flow_pred`, which lands on the
            # very same sigma because `sigma_next` is read off the grid at
            # `global_idx + 1` rather than off a scheduler counter. That
            # increment is what `--flow_euler_rollout` trains, and the
            # trainer's own KV-cache walk pops the same `(span, step_idx,
            # commit)` stack, so a ladder under a deterministic schedule is
            # legitimate - and is the rollout that actually matches a
            # checkpoint distilled with `--flow_euler_rollout`.
            if local_attn_size != -1:
                raise ValueError(
                    "A per-step partition ladder (`denoise_mode='pyramid'` or "
                    "'full_then_blocks') requires `local_attn_size=-1` (full KV "
                    "cache). A rolling window evicts by `current_start` deltas, "
                    "which a level change moves backwards.")
            # Forcing-KV (arXiv 2605.09681) rewrites the cache by grouping heads
            # into static/dynamic histories on a *clean* pass, and it anchors its
            # `ar_step` on a fixed block size. A ladder breaks both premises: the
            # partition changes between steps, so neither the per-block clean
            # commit nor a fixed stride survives it.
            if self._forcing_kv_enabled():
                raise ValueError(
                    "A per-step partition ladder (`denoise_mode='pyramid'` or "
                    "'full_then_blocks') cannot be combined with Forcing-KV head "
                    "compression: compression commits a clean pass per block and "
                    "derives its step from a fixed block size, neither of which "
                    "survives a partition that changes between steps.")
        # Applies to every mode, not just the pyramid. The partition defines one
        # bidirectional chunk, so a frame must see the whole of its own chunk. A
        # rolling window is anchored at the chunk *end* (`window_start =
        # local_end_index - local_attn_size * frame_seqlen`), so once a chunk is
        # wider than the window its earliest frames cannot attend to its latest
        # ones and the chunk silently degrades to partial causality. `sink_size`
        # does not rescue this: it pins the clip start, not the neighbourhood of
        # the chunk being denoised.
        # `partitions` is the ladder (a list of levels, each a partition) when
        # one is on; otherwise the single `chunk_sizes` partition is the only
        # level. Widest chunk over every level a rollout can reach.
        levels = partitions if partitions is not None else [chunk_sizes]
        widest = max(max(level) for level in levels)
        if local_attn_size != -1 and widest > local_attn_size:
            raise ValueError(
                f"Partition breaks the local attention window: the widest chunk "
                f"is {widest} latent frames but `local_attn_size="
                f"{local_attn_size}`, so a chunk's first frames cannot attend to "
                f"its last ones and the intra-chunk bidirectionality that "
                f"Flex-Forcing 3.1 relies on is lost. Either use "
                f"`local_attn_size=-1`, or keep every chunk at or below "
                f"`local_attn_size` frames.")

        # Below is the Flex-Forcing rollout proper. It performs the same setup
        # the inherited `__call__` does - prompt encoding, timesteps, latents,
        # KV cache allocation, Forcing-KV config - then replaces the fixed
        # `num_frame_per_block` schedule with `chunk_sizes` / the pyramid
        # `partitions` and runs the coarse-to-fine walk below. T2V only.
        if isinstance(callback_on_step_end,
                      (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs
        self.check_inputs(prompt, height, width, negative_prompt,
                          callback_on_step_end_tensor_inputs, prompt_embeds,
                          negative_prompt_embeds)
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
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, negative_prompt, do_classifier_free_guidance,
            num_videos_per_prompt=1, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length, device=device)
        in_prompt_embeds = (negative_prompt_embeds + prompt_embeds
                            if do_classifier_free_guidance else prompt_embeds)

        # 4. Prepare timesteps
        if stochastic_sampling:
            timesteps = stochastic_sampling_timesteps(
                num_inference_steps, shift, device)
        elif isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler, device=device, sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents (noise) and output buffer. T2V only: every latent
        #    frame is generated, so noise covers the whole clip.
        latent_channels = self.transformer.config.in_channels
        noise = self.prepare_latents(
            batch_size, latent_channels, num_frames, height, width,
            weight_dtype, device, generator, latents)
        output = torch.zeros_like(noise, device=device, dtype=weight_dtype)

        # 6. Sequence geometry
        patch_size = self.transformer.config.patch_size
        target_shape = (
            self.vae.latent_channels,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            width // self.vae.spatial_compression_ratio,
            height // self.vae.spatial_compression_ratio,
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) / (patch_size[1] * patch_size[2])
            * target_shape[1])
        frame_seq_length = (target_shape[2] * target_shape[3]) // (
            patch_size[1] * patch_size[2])
        num_latent_frames = target_shape[1]

        # 7. Forcing-KV config before the cache is budget-sized, then anchor the
        #    attention window to the largest chunk of the partition.
        if forcing_kv_enable is not None:
            fkv_kwargs = {}
            if forcing_kv_ar_start is not None:
                fkv_kwargs["ar_start"] = forcing_kv_ar_start
            if forcing_kv_spatial_context_length is not None:
                fkv_kwargs["spatial_context_length"] = forcing_kv_spatial_context_length
            if forcing_kv_temporal_context_length is not None:
                fkv_kwargs["temporal_context_length"] = forcing_kv_temporal_context_length
            if forcing_kv_dynamic_context_length is not None:
                fkv_kwargs["dynamic_context_length"] = forcing_kv_dynamic_context_length
            if forcing_kv_num_frame_patch is not None:
                fkv_kwargs["num_frame_patch"] = forcing_kv_num_frame_patch
            if forcing_kv_sim_retention_ratio is not None:
                fkv_kwargs["sim_retention_ratio"] = forcing_kv_sim_retention_ratio
            self.set_forcing_kv_config(
                forcing_kv_enable,
                head_profile=forcing_kv_head_profile,
                **fkv_kwargs)
        self._unwrap_transformer().num_frame_per_block = max(chunk_sizes)

        # 8. Initialize (or reset) the KV / cross-attention caches
        required_kv_size = num_latent_frames * frame_seq_length
        if (self._forcing_kv_enabled()
                and getattr(self.transformer.config, 'local_attn_size', -1) != -1):
            fkv_cache_size = self._forcing_kv_cache_tokens(frame_seq_length)
            if fkv_cache_size is not None:
                required_kv_size = fkv_cache_size
        if (self.kv_cache_pos is not None
                and self.kv_cache_pos[0]["k"].shape[1] >= required_kv_size):
            for block_index in range(len(self.kv_cache_pos)):
                for cache in (self.kv_cache_pos[block_index],
                              self.kv_cache_neg[block_index]):
                    cache["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=device)
                    cache["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=device)
                    for key in ("forcing_kv", "_fkv_last_q", "_fkv_window_start",
                                "_fkv_local_end"):
                        cache.pop(key, None)
            for block_index in range(len(self.crossattn_cache_pos)):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
        else:
            self._initialize_kv_cache(
                batch_size=batch_size, dtype=weight_dtype, device=device,
                frame_seq_length=frame_seq_length,
                num_latent_frames=num_latent_frames)
            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=weight_dtype, device=device)

        pbar = None
        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(len(chunk_sizes) * num_inference_steps + 1)
            pbar.update(1)

        # 9. Run the flexible rollout (3.1 / 3.2).
        #
        # Everything below reads the setup above straight out of this frame, and
        # every transformer call is spelled out at its own site the way the
        # inherited rollout spells it - no state object, no helper class and no
        # helper closure to chase. `state` is the paper's temporary buffer: a
        # copy of the initial noise in which every frame sits at its current
        # noise level, and which the walk both reads from and writes back into.
        state = noise.clone()
        # The stochastic grid carries a trailing element that the re-noising
        # reads as `timesteps[global_idx + 1]`, so it is not a step and gets
        # sliced off. The deterministic grid needs no tail of its own: its
        # transfer reads `scheduler.sigmas`, which `set_timesteps` already
        # terminates with a 0.
        denoise_timesteps = timesteps[:-1] if stochastic_sampling else timesteps
        last_step = len(denoise_timesteps) - 1
        block_idx = 0
        streamed_videos: List[torch.Tensor] = []

        # A single-level ladder is exactly the block-major Self-Forcing rollout
        # over `chunk_sizes`; the pyramid just appends finer levels.
        ladder = [list(chunk_sizes)] + (
            [list(p) for p in partitions[1:]] if partitions else [])
        validate_nested_partitions(ladder, num_latent_frames)

        # The walk. A work item is (span, step_idx, commit): a frame range, the
        # schedule position it sits at, and whether its x0 has to go into the KV
        # cache once it is done. Two rules, from 3.2:
        #
        #   * the span still splits at the next level -> one *buffered*
        #     bidirectional step over the whole span, then resume the sub-spans
        #     autoregressively. Pushing them back reversed keeps the walk in
        #     temporal order, which both the KV cache and the streaming decode
        #     rely on;
        #   * otherwise -> run the rest of the schedule over the span in one go.
        #
        # Both rules are the same loop over the same span; they differ only in
        # how much of the schedule they take and which of its two results they
        # keep, so the loop is written out once below rather than hidden in a
        # helper.
        top = chunk_boundaries(ladder[0])
        stack = [(span, 0, idx < len(top) - 1)
                 for idx, span in reversed(list(enumerate(top)))]
        if streaming:
            self.vae.clear_cache()
        while stack:
            (start, end), step_idx, commit = stack.pop()
            level = ladder[min(step_idx + 1, len(ladder) - 1)]
            subs = [s for s in chunk_boundaries(level)
                    if s[0] >= start and s[1] <= end]
            splits = step_idx < last_step and len(subs) > 1

            # A splitting span takes a single buffered step; a leaf runs
            # whatever is left of the schedule to completion. `step_idx` is
            # where the slice starts inside `timesteps`, which is what lets the
            # transfer below pick up the right (t_i, t_i+1) pair when the walk
            # resumes a span mid-schedule - indexed into `timesteps` directly
            # under a stochastic rollout, resolved by the scheduler from the `t`
            # it is handed under a deterministic one.
            schedule = (denoise_timesteps[step_idx:step_idx + 1] if splits
                        else denoise_timesteps[step_idx:])

            # Reset scheduler state for each span (required for causal
            # generation). For Euler, clearing `_step_index` is enough;
            # multi-step schedulers (UniPC, DPM++) also accumulate
            # `model_outputs`. Spelled inline, as the inherited rollout does.
            self.scheduler._step_index = None
            if hasattr(self.scheduler, "model_outputs"):
                self.scheduler.model_outputs = []

            # `span_frames` is this span's width in *latent* frames, deliberately
            # not called `num_frames`: that name belongs to this call's
            # pixel-frame count.
            span_frames = end - start
            current_start = (start_frame_index + start) * frame_seq_length
            noisy_input = state[:, :, start:end]
            denoised_pred = noisy_input
            t = None
            # The buffered step is a single forward pass, so it gets no bar of
            # its own; the leaf's bar counts down the rest of its schedule.
            progress_ctx = (nullcontext() if splits
                            else self.progress_bar(total=len(schedule)))
            with progress_ctx as progress_bar:
                for local_idx, t in enumerate(schedule):
                    timestep = torch.ones([batch_size, span_frames],
                                          device=device,
                                          dtype=weight_dtype) * t
                    if pbar is not None:
                        pbar.update(1)
                    # One CFG-combined transformer call, written out as the
                    # inherited rollout writes it. `flex_state=None` throughout:
                    # the 4.2 any-order window is an editing knob and T2V
                    # generation never narrows it.
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
                                current_start=current_start,
                                cache_start=None,
                                forcing_kv_state=None,
                                flex_state=None,
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
                                current_start=current_start,
                                cache_start=None,
                                forcing_kv_state=None,
                                flex_state=None,
                            )
                        # CFG guidance, with the legacy 4-dim [F, C, H, W]
                        # return shape folded in.
                        if flow_pred_cond.dim() == 5:
                            flow_pred = flow_pred_uncond + guidance_scale * (
                                flow_pred_cond - flow_pred_uncond)
                        elif flow_pred_cond.dim() == 4:
                            flow_pred_cond = flow_pred_cond.unsqueeze(0).permute(
                                0, 2, 1, 3, 4)
                            flow_pred_uncond = flow_pred_uncond.unsqueeze(0).permute(
                                0, 2, 1, 3, 4)
                            flow_pred = flow_pred_uncond + guidance_scale * (
                                flow_pred_cond - flow_pred_uncond)
                        else:
                            raise ValueError(
                                f"Unexpected flow_pred_cond dim: "
                                f"{flow_pred_cond.dim()}, "
                                f"shape: {flow_pred_cond.shape}")
                    else:
                        with torch.cuda.amp.autocast(dtype=weight_dtype):
                            flow_pred = self.transformer(
                                x=noisy_input,
                                context=in_prompt_embeds,
                                t=timestep,
                                seq_len=seq_len,
                                kv_cache=self.kv_cache_pos,
                                crossattn_cache=self.crossattn_cache_pos,
                                current_start=current_start,
                                cache_start=None,
                                forcing_kv_state=None,
                                flex_state=None,
                            )
                        if flow_pred.dim() == 4:
                            flow_pred = flow_pred.unsqueeze(0).permute(
                                0, 2, 1, 3, 4)
                        elif flow_pred.dim() != 5:
                            # The inherited rollout lets an unexpected rank fall
                            # through here and die later as a broadcast error;
                            # naming it at the source is cheaper to debug.
                            raise ValueError(
                                f"Unexpected flow_pred dim: {flow_pred.dim()}, "
                                f"shape: {flow_pred.shape}")

                    if stochastic_sampling:
                        global_idx = step_idx + local_idx
                        t_i = (timesteps[global_idx] / 1000).to(weight_dtype)
                        t_i_1 = (timesteps[global_idx + 1] / 1000).to(weight_dtype)
                        denoised_pred = noisy_input - flow_pred * t_i
                        noisy_input = (1 - t_i_1) * denoised_pred + t_i_1 * torch.randn(
                            denoised_pred.shape, dtype=denoised_pred.dtype,
                            device=device, generator=generator)
                    else:
                        # Delegate to the scheduler's own step() so each sampler
                        # (Flow Euler, UniPC, DPM++) applies its multi-step
                        # formula, exactly as the inherited rollout does.
                        #
                        # This holds under a ladder as well, which is not
                        # obvious: the walk revisits the schedule out of order,
                        # but the per-span `_step_index = None` reset above
                        # makes step() re-derive its position by looking the
                        # handed-over `t` up in `scheduler.timesteps` instead of
                        # counting calls, and inside one span the counter then
                        # advances in step with `schedule`. `sigmas` carries the
                        # trailing 0 `set_timesteps` appends, so the leaf's last
                        # step lands exactly on x0.
                        #
                        # Known and accepted: this grid is
                        # `set_timesteps(num_inference_steps)`, which at shift=5
                        # gives [1000, 909.7, 717.3, 24.4], whereas the trainer
                        # distils on [1000, 937.5, 833.3, 625] - the grid
                        # `stochastic_sampling_timesteps` reproduces entry for
                        # entry. A deterministic ladder therefore samples noise
                        # levels this checkpoint was not distilled at. Use
                        # `stochastic_sampling=True` when the samples have to
                        # sit on the trained trajectory.
                        noisy_input = self.scheduler.step(
                            flow_pred, t, noisy_input, return_dict=False)[0]
                        denoised_pred = noisy_input
                    if progress_bar is not None:
                        progress_bar.update()

            if splits:
                # The next level consumes the *re-noised* buffer, not x0: the
                # sub-spans pick the schedule up where this step left it. The
                # trailing sub-chunk only needs committing when the parent span
                # itself is followed by a sibling at some higher level.
                state[:, :, start:end] = noisy_input
                stack.extend(reversed([
                    (sub, step_idx + 1, k < len(subs) - 1 or commit)
                    for k, sub in enumerate(subs)]))
                continue

            x0 = denoised_pred
            state[:, :, start:end] = x0
            output[:, :, start:end] = x0

            # Leaves complete in temporal order (the walk never revisits an
            # earlier span), so per-leaf streaming decode stays seam-free.
            if streaming:
                video_chunk = self.decode_latents_stream(x0)
                if decode_callback is not None:
                    decode_callback(video_chunk, block_idx)
                else:
                    streamed_videos.append(video_chunk)
            if commit:
                # Write x0 into the KV cache as clean context for later chunks,
                # run at `context_noise` - the level the cache is trained to be
                # read back from, so the next chunk attends to clean keys.
                #
                # Leaves only. A span that split already had its sub-spans
                # committed at their own finer granularity, and re-committing it
                # as one coarse chunk would rewrite those keys under an
                # attention pattern their x0 was never produced with.
                commit_timestep = torch.ones([batch_size, span_frames],
                                             device=device,
                                             dtype=torch.long) * context_noise
                if do_classifier_free_guidance:
                    # Update both positive and negative caches.
                    with torch.cuda.amp.autocast(dtype=weight_dtype):
                        self.transformer(
                            x=x0,
                            context=prompt_embeds,
                            t=commit_timestep,
                            seq_len=seq_len,
                            kv_cache=self.kv_cache_pos,
                            crossattn_cache=self.crossattn_cache_pos,
                            current_start=current_start,
                            cache_start=None,
                            forcing_kv_state={"clean_pass": True},
                            flex_state=None,
                        )
                        self.transformer(
                            x=x0,
                            context=negative_prompt_embeds,
                            t=commit_timestep,
                            seq_len=seq_len,
                            kv_cache=self.kv_cache_neg,
                            crossattn_cache=self.crossattn_cache_neg,
                            current_start=current_start,
                            cache_start=None,
                            forcing_kv_state={"clean_pass": True},
                            flex_state=None,
                        )
                else:
                    with torch.cuda.amp.autocast(dtype=weight_dtype):
                        self.transformer(
                            x=x0,
                            context=in_prompt_embeds,
                            t=commit_timestep,
                            seq_len=seq_len,
                            kv_cache=self.kv_cache_pos,
                            crossattn_cache=self.crossattn_cache_pos,
                            current_start=current_start,
                            cache_start=None,
                            forcing_kv_state={"clean_pass": True},
                            flex_state=None,
                        )

            if callback_on_step_end is not None:
                # `check_inputs` has already restricted the names to
                # `_callback_tensor_inputs`. The inherited rollout spells this
                # `locals()[k]` and hands over its own live buffer as `latents`;
                # `state` is ours, so a returned `latents` has to be copied back
                # for the edit to reach the steps that follow.
                available = {"latents": state,
                             "prompt_embeds": prompt_embeds,
                             "negative_prompt_embeds": negative_prompt_embeds}
                callback_outputs = callback_on_step_end(
                    self, block_idx, t,
                    {k: available[k]
                     for k in callback_on_step_end_tensor_inputs})
                if callback_outputs:
                    returned = callback_outputs.pop("latents", None)
                    if returned is not None:
                        state.copy_(returned)
            block_idx += 1

        # 10. Decode output
        if streaming:
            # Close the VAE causal cache the rollout opened before its first
            # leaf.
            self.vae.clear_cache()
            if decode_callback is not None:
                video = output.new_zeros(0)
            else:
                video = torch.cat(streamed_videos, dim=2)
        elif output_type == "pil":
            video = torch.from_numpy(self.decode_latents(output))
        else:
            video = output

        self.maybe_free_model_hooks()
        if not return_dict:
            return (video,)
        return WanFlexForcingPipelineOutput(videos=video)

    # ------------------------------------------------------------------
    # 4.2 Any-order / any-timestep editing
    # ------------------------------------------------------------------
    @torch.no_grad()
    def edit_video(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        video: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        edit_span: Optional[Tuple[int, int]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 2,
        edit_steps: int = 1,
        shift: float = 5.0,
        context_noise: int = 0,
        stochastic_sampling: bool = True,
        num_frame_per_block: Optional[int] = None,
        max_sequence_length: int = 512,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[WanFlexForcingPipelineOutput, Tuple]:
        r"""
        Edit an already generated clip in place (Flex-Forcing 4.2).

        The clip is committed to the KV cache as clean context, one span is
        re-noised to a *low-level refinement* timestep and denoised there while
        the high-level planning timesteps stay untouched. The attention window
        is widened to the whole clip, so the edited span conditions on clean
        tokens from both its past and its future - the any-order capability a
        strictly causal rollout cannot offer.

        Args:
            prompt: Text condition for the edit.
            video: Clip to edit, ``[B, C, F, H, W]`` pixels in ``[0, 1]`` - the
                range :meth:`decode_latents` returns and
                :func:`videox_fun.utils.utils.get_video_to_video_latent`
                produces, so a generated clip can be fed straight back in.
                Mutually exclusive with ``latents``.
            latents: Pre-encoded clip, ``[B, C, F, H, W]`` latent frames.
            edit_span: Half-open ``(start, end)`` range of **latent** frames to
                regenerate. ``None`` edits the whole clip.
            num_inference_steps: Length of the schedule the refinement steps are
                taken from (2 for the paper's DMD model).
            edit_steps: How many *trailing* steps of that schedule to run. Keep
                it small - editing at a planning timestep would restructure the
                clip instead of refining it.
            stochastic_sampling: Which rollout the refinement runs on - the same
                knob :meth:`__call__` takes. ``True`` re-noises between steps on
                the hardcoded ``stochastic_sampling_timesteps`` grid. ``False``
                delegates each step to ``self.scheduler`` (Flow Euler, UniPC,
                DPM++), which is what a checkpoint distilled with
                ``--flow_euler_rollout`` was trained on.
            num_frame_per_block: Granularity of the clean-context commit - the
                same uniform block size the inherited rollout uses, doing the
                same job. ``None`` commits the whole clip in one bidirectional
                pass, which is the memory-hungry end; an int commits chunk by
                chunk in temporal order. The trailing chunk absorbs the
                remainder, so ``num_latent_frames`` need not be divisible by it
                (the inherited rollout asserts that; this one does not).

        Returns:
            The edited clip, decoded when ``output_type == "pil"`` and returned
            as latents otherwise.
        """
        if (video is None) == (latents is None):
            raise ValueError("Provide exactly one of `video` or `latents`.")
        if getattr(self.transformer.config, 'local_attn_size', -1) != -1:
            raise ValueError(
                "Any-order editing requires `local_attn_size=-1`: the edited "
                "span must see clean tokens on both sides, which a rolling "
                "window may already have evicted.")

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        if latents is None:
            # `vae.encode` wants [-1, 1]; the user-facing range in this repo is
            # [0, 1] (see `decode_latents`), so normalise here - the same inline
            # convention `pipeline_lingbot_video_i2v` uses.
            pixels = video.to(device=device, dtype=torch.float32)
            pixels = ((pixels - 0.5) / 0.5).to(self.vae.dtype)
            latents = self.vae.encode(pixels)[0].mode()
        latents = latents.to(device=device, dtype=weight_dtype)
        batch_size, _, num_latent_frames = latents.shape[:3]

        edit_start, edit_end = (0, num_latent_frames) if edit_span is None \
            else (int(edit_span[0]), int(edit_span[1]))
        if not 0 <= edit_start < edit_end <= num_latent_frames:
            raise ValueError(
                f"edit_span {(edit_start, edit_end)} does not describe a "
                f"non-empty range inside {num_latent_frames} latent frames.")
        if not 1 <= int(edit_steps) <= int(num_inference_steps):
            raise ValueError(
                f"edit_steps must lie in [1, num_inference_steps="
                f"{num_inference_steps}], got {edit_steps}.")

        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, negative_prompt, do_classifier_free_guidance,
            num_videos_per_prompt=1, max_sequence_length=max_sequence_length,
            device=device)
        in_prompt_embeds = (negative_prompt_embeds + prompt_embeds
                            if do_classifier_free_guidance else prompt_embeds)

        patch_size = self.transformer.config.patch_size
        frame_seq_length = ((latents.shape[3] // patch_size[1])
                            * (latents.shape[4] // patch_size[2]))
        seq_len = frame_seq_length * num_latent_frames

        # The same schedule dispatch `__call__` makes: the stochastic grid is
        # the hardcoded few-step one, the deterministic grid comes from the
        # scheduler so that whichever sampler is configured supplies its own
        # multi-step formula. `num_inference_steps` may be rewritten by
        # `retrieve_timesteps` and `step_offset` below is derived from it, so
        # this has to run first.
        if stochastic_sampling:
            timesteps = stochastic_sampling_timesteps(
                num_inference_steps, shift, device)
        elif isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, None)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler, device=device, sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, None)
        self._num_timesteps = len(timesteps)

        self._initialize_kv_cache(
            batch_size=batch_size, dtype=weight_dtype, device=device,
            frame_seq_length=frame_seq_length,
            num_latent_frames=num_latent_frames)
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=weight_dtype, device=device)

        commit_chunks = ([num_latent_frames] if num_frame_per_block is None
                         else uniform_chunks(num_latent_frames,
                                             num_frame_per_block))
        # The transformer's own attribute has to cover the widest forward pass
        # of this call, so a span wider than the commit block widens it too.
        self._unwrap_transformer().num_frame_per_block = max(
            max(commit_chunks), edit_end - edit_start)

        # 1. Clean context for the whole clip, committed causally in temporal
        #    order at `context_noise` - the level the cache is trained to be
        #    read back from. `flex_state=None` keeps the 4.2 any-order window
        #    closed here; widening it would expose the tail of the cache that has
        #    not been written yet (zero keys).
        for start, end in chunk_boundaries(commit_chunks):
            timestep = torch.ones([batch_size, end - start], device=device,
                                  dtype=torch.long) * context_noise
            if do_classifier_free_guidance:
                # Update both positive and negative caches.
                with torch.cuda.amp.autocast(dtype=weight_dtype):
                    self.transformer(
                        x=latents[:, :, start:end],
                        context=prompt_embeds,
                        t=timestep,
                        seq_len=seq_len,
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=start * frame_seq_length,
                        cache_start=None,
                        forcing_kv_state={"clean_pass": True},
                        flex_state=None,
                    )
                    self.transformer(
                        x=latents[:, :, start:end],
                        context=negative_prompt_embeds,
                        t=timestep,
                        seq_len=seq_len,
                        kv_cache=self.kv_cache_neg,
                        crossattn_cache=self.crossattn_cache_neg,
                        current_start=start * frame_seq_length,
                        cache_start=None,
                        forcing_kv_state={"clean_pass": True},
                        flex_state=None,
                    )
            else:
                with torch.cuda.amp.autocast(dtype=weight_dtype):
                    self.transformer(
                        x=latents[:, :, start:end],
                        context=in_prompt_embeds,
                        t=timestep,
                        seq_len=seq_len,
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=start * frame_seq_length,
                        cache_start=None,
                        forcing_kv_state={"clean_pass": True},
                        flex_state=None,
                    )

        # 2. Re-noise only the edited span to its refinement level, leaving
        #    the planning timesteps (and every other frame) untouched.
        step_offset = int(num_inference_steps) - int(edit_steps)
        t_edit = (timesteps[step_offset] / 1000).to(weight_dtype)
        span_noise = torch.randn(
            latents[:, :, edit_start:edit_end].shape, dtype=weight_dtype,
            device=device, generator=generator)
        noisy_span = ((1 - t_edit) * latents[:, :, edit_start:edit_end]
                      + t_edit * span_noise)

        # 3. Denoise the span at the refinement levels only, with the whole clip
        #    visible so it conditions on past *and* future clean tokens - the
        #    any-order half of 4.2, which is what `flex_state` opens here and
        #    step 1 deliberately left closed. Either rollout runs the trailing
        #    `edit_steps` of the schedule; they differ in whether the grid
        #    carries a tail. The stochastic one does - the re-noising below reads
        #    it as `timesteps[global_idx + 1]` - so it is sliced off here. The
        #    deterministic one needs no tail of its own, because its transfer
        #    reads `scheduler.sigmas`, which `set_timesteps` already terminates
        #    with a 0.
        flex_state = {"attn_window": (0, seq_len)}
        denoise_timesteps = timesteps[:-1] if stochastic_sampling else timesteps
        schedule = denoise_timesteps[step_offset:]
        span_frames = edit_end - edit_start
        current_start = edit_start * frame_seq_length
        noisy_input = noisy_span
        edited = noisy_span
        # Only the deterministic branch touches `self.scheduler`, but that one
        # keeps its own position there and, for the multi-step samplers, a
        # history of model outputs. Start both fresh, the way `__call__` does at
        # the head of every span.
        self.scheduler._step_index = None
        if hasattr(self.scheduler, "model_outputs"):
            self.scheduler.model_outputs = []
        with self.progress_bar(total=len(schedule)) as progress_bar:
            for local_idx, t in enumerate(schedule):
                timestep = torch.ones([batch_size, span_frames], device=device,
                                      dtype=weight_dtype) * t
                # One CFG-combined transformer call over the widened window.
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
                            current_start=current_start,
                            cache_start=None,
                            forcing_kv_state=None,
                            flex_state=flex_state,
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
                            current_start=current_start,
                            cache_start=None,
                            forcing_kv_state=None,
                            flex_state=flex_state,
                        )
                    # CFG guidance, with the legacy 4-dim [F, C, H, W] return
                    # shape folded in.
                    if flow_pred_cond.dim() == 5:
                        flow_pred = flow_pred_uncond + guidance_scale * (
                            flow_pred_cond - flow_pred_uncond)
                    elif flow_pred_cond.dim() == 4:
                        flow_pred_cond = flow_pred_cond.unsqueeze(0).permute(
                            0, 2, 1, 3, 4)
                        flow_pred_uncond = flow_pred_uncond.unsqueeze(0).permute(
                            0, 2, 1, 3, 4)
                        flow_pred = flow_pred_uncond + guidance_scale * (
                            flow_pred_cond - flow_pred_uncond)
                    else:
                        raise ValueError(
                            f"Unexpected flow_pred_cond dim: "
                            f"{flow_pred_cond.dim()}, "
                            f"shape: {flow_pred_cond.shape}")
                else:
                    with torch.cuda.amp.autocast(dtype=weight_dtype):
                        flow_pred = self.transformer(
                            x=noisy_input,
                            context=in_prompt_embeds,
                            t=timestep,
                            seq_len=seq_len,
                            kv_cache=self.kv_cache_pos,
                            crossattn_cache=self.crossattn_cache_pos,
                            current_start=current_start,
                            cache_start=None,
                            forcing_kv_state=None,
                            flex_state=flex_state,
                        )
                    if flow_pred.dim() == 4:
                        flow_pred = flow_pred.unsqueeze(0).permute(
                            0, 2, 1, 3, 4)
                    elif flow_pred.dim() != 5:
                        raise ValueError(
                            f"Unexpected flow_pred dim: {flow_pred.dim()}, "
                            f"shape: {flow_pred.shape}")

                if stochastic_sampling:
                    global_idx = step_offset + local_idx
                    t_i = (timesteps[global_idx] / 1000).to(weight_dtype)
                    t_i_1 = (timesteps[global_idx + 1] / 1000).to(weight_dtype)
                    edited = noisy_input - flow_pred * t_i
                    noisy_input = (1 - t_i_1) * edited + t_i_1 * torch.randn(
                        edited.shape, dtype=edited.dtype, device=device,
                        generator=generator)
                else:
                    # Delegate to the scheduler's own step(), exactly as
                    # `__call__` does. This refinement is a single linear span
                    # rather than a walk, so the position `step()` re-derives
                    # from the `t` it is handed is the one the schedule is at,
                    # and the trailing sigma of 0 lands the last step on x0 -
                    # which is what `output` below writes back into the clip.
                    noisy_input = self.scheduler.step(
                        flow_pred, t, noisy_input, return_dict=False)[0]
                    edited = noisy_input
                progress_bar.update()

        output = latents.clone()
        output[:, :, edit_start:edit_end] = edited

        if output_type == "pil":
            result = torch.from_numpy(self.decode_latents(output))
        else:
            result = output

        self.maybe_free_model_hooks()
        if not return_dict:
            return (result,)
        return WanFlexForcingPipelineOutput(videos=result)
