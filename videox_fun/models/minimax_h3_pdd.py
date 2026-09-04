r"""Parallel Decoding Distillation (PDD) parts for MiniMax-H3.

The model-agnostic pieces — the plan math, [`PDDParallelHead`], [`PDDLoRALinear`], [`pdd_teacher_mode`], the checkpoint
resolution — live in `videox_fun/utils/lora_utils_pdd.py` and are re-exported below, so every existing
`from videox_fun.models.minimax_h3_pdd import ...` site keeps working. This file keeps only the MiniMax-H3 glue:
which layers become parallel heads, how the packed-sequence teacher forward is called, and how the pipeline's step
callback arms the plans.

PDD (arXiv 2607.26004) turns a pre-trained flow model into a *parallel decoder*: the sampling interval is discretized
into `N` intervals grouped into blocks of size `L`, and one network evaluation predicts the **mean velocity of every
interval of the next block** instead of the single instantaneous velocity. Generation then advances `L` intervals per
evaluation, i.e. `NFE = N / L`.

MiniMax-H3 has two final heads — `proj_out` for the video rows and `audio_proj_out` for the audio rows — and both are
repeated, as the paper does for the two towers of LTX-2.3. Video and audio ride the same block structure on two
schedules (`shift=12.0` / `3.0`), so both modalities take the stage together and each advances by its own step size.

The frozen teacher is the same module under [`pdd_teacher_mode`]: the low-rank updates of the backbone are switched off and
both heads fall back to the pre-trained weights they were built from. There is no second copy of the 33 B backbone.
"""

import os

import torch

from videox_fun.utils.lora_utils_pdd import (
    PDD_EMA_WEIGHTS_NAME, PDD_LEGACY_LIVE_WEIGHTS_NAME, PDD_WEIGHTS_NAME, PDDLoRALinear, PDDParallelHead, add_pdd_lora,
    load_pdd_config, merge_pdd_lora, pdd_num_inference_steps, pdd_sampling_plan, pdd_state_dict, pdd_time_grid,
    pdd_training_plan, pdd_teacher_mode, resolve_pdd_lora_path, shifted_sigma,
)

# The generic class was `MiniMaxH3ParallelHead` before the model-agnostic parts moved to
# `videox_fun/utils/lora_utils_pdd.py`; the alias is the same class object, so old imports and `isinstance` checks
# keep working.
MiniMaxH3ParallelHead = PDDParallelHead


def attach_parallel_decoder(transformer, num_steps: int) -> None:
    r"""
    Turn a `MiniMaxH3Transformer3DModel` into a PDD parallel decoder, in place.

    Both final heads are replaced by [`PDDParallelHead`]s of `num_steps` heads each, initialized from the weights they
    replace. Nothing else about the model changes: the two heads keep the names `proj_out` and `audio_proj_out`, so
    the float32 pinning of the mixed-precision checkpoint (`_keep_in_fp32_modules`) and the forward that reads
    `self.proj_out.weight.dtype` both still apply.

    Args:
        transformer (`MiniMaxH3Transformer3DModel`): The model to convert.
        num_steps (`int`): The PDD grid size `N`.
    """
    transformer.proj_out = PDDParallelHead(transformer.proj_out, num_steps)
    transformer.audio_proj_out = PDDParallelHead(transformer.audio_proj_out, num_steps)


def set_parallel_plan(transformer, video_plan: torch.Tensor, audio_plan: torch.Tensor) -> None:
    r"""Set the plans of both parallel heads for the next forward pass."""
    transformer.proj_out.set_plan(video_plan)
    transformer.audio_proj_out.set_plan(audio_plan)


def pdd_teacher_mean_velocity(teacher, forward_kwargs, video, audio, index, grids, solver: str):
    r"""
    A Runge-Kutta estimate of the teacher's mean velocity over interval `index` of the grid (eq. 5 / eq. 6).

    Video and audio ride the same block structure on two schedules, so both modalities take the stage together and
    each advances by its own step size. The caller must already have put the model in [`pdd_teacher_mode`].

    Args:
        teacher: The transformer, under [`pdd_teacher_mode`].
        forward_kwargs (`Callable[[float, float], dict]`):
            Builds everything but the two latent streams for a forward at a given `(video_time, audio_time)` — the
            conditioning, the row timesteps and the packed layout, all of which are the caller's business.
        video (`torch.Tensor`), audio (`torch.Tensor`): The state the mean velocity is estimated at.
        index (`int`): The grid interval.
        grids (`tuple`): `(video_grid, audio_grid, video_step_sizes, audio_step_sizes)`.
        solver (`str`): `"euler"` for one evaluation, `"midpoint"` for two.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: the video and audio mean velocities, in float32.
    """
    video_grid, audio_grid, video_steps, audio_steps = grids
    video_time, audio_time = float(video_grid[index]), float(audio_grid[index])
    velocity = teacher(
        hidden_states=video[None], audio_hidden_states=audio[None], **forward_kwargs(video_time, audio_time)
    )
    if solver == "euler":
        return velocity[0][0].float(), velocity[1][0].float()

    half_video, half_audio = 0.5 * float(video_steps[index]), 0.5 * float(audio_steps[index])
    mid_video = video + half_video * velocity[0][0].float()
    mid_audio = audio + half_audio * velocity[1][0].float()
    velocity = teacher(
        hidden_states=mid_video[None],
        audio_hidden_states=mid_audio[None],
        **forward_kwargs(video_time + half_video, audio_time + half_audio),
    )
    return velocity[0][0].float(), velocity[1][0].float()


def load_pdd_lora(transformer, pdd_lora_path):
    r"""
    Attach the parallel heads and LoRA, then load the resolved PDD weights into `transformer`.

    A checkpoint directory loads `pdd_ema.safetensors` when present (EMA inference export) and otherwise
    `pdd.safetensors`. Returns the config the predict scripts need to arm the heads and pick NFE.
    """
    path = resolve_pdd_lora_path(pdd_lora_path)
    config = load_pdd_config(path)
    add_pdd_lora(
        transformer,
        config["lora_targets"].split(","),
        int(config["lora_rank"]),
        float(config["lora_alpha"]),
    )
    attach_parallel_decoder(transformer, int(config["pdd_num_steps"]))
    if path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(path)
    else:
        state_dict = torch.load(path, map_location="cpu")
    _, unexpected = transformer.load_state_dict(state_dict, strict=False)
    print(f"From PDD checkpoint: {path} ({len(state_dict)} tensors, unexpected keys: {len(unexpected)})", flush=True)
    assert not unexpected, f"{path} holds keys the parallel decoder does not have, e.g. {unexpected[:3]}."
    return config


def pdd_step_callback(transformer, scheduler, audio_scheduler, config, num_inference_steps):
    r"""Arm the fused block-mean plan before each pipeline step. Call this, then pass the return value as `callback_on_step_end`."""
    video_steps = pdd_time_grid(scheduler.shift, int(config["pdd_num_steps"])).diff()
    audio_steps = pdd_time_grid(audio_scheduler.shift, int(config["pdd_num_steps"])).diff()
    block_size = int(config["pdd_num_steps"]) // int(num_inference_steps)

    def arm(step_index):
        start = step_index * block_size
        set_parallel_plan(
            transformer,
            pdd_sampling_plan(video_steps, start, block_size).float(),
            pdd_sampling_plan(audio_steps, start, block_size).float(),
        )

    arm(0)

    def callback(pipe, step_index, timestep, callback_kwargs):
        if step_index + 1 < int(num_inference_steps):
            arm(step_index + 1)
        return {}

    return callback
