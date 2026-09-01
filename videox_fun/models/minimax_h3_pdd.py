r"""Parallel Decoding Distillation (PDD) parts for MiniMax-H3.

PDD (arXiv 2607.26004) turns a pre-trained flow model into a *parallel decoder*: the sampling interval is discretized
into `N` intervals grouped into blocks of size `L`, and one network evaluation predicts the **mean velocity of every
interval of the next block** instead of the single instantaneous velocity. Generation then advances `L` intervals per
evaluation, i.e. `NFE = N / L`.

Architecturally this is the teacher's own backbone with the final linear layer repeated `N` times, one head per
interval of the grid, each initialized from the pre-trained head (§3, Figure 4 of the paper). MiniMax-H3 has two final
heads — `proj_out` for the video rows and `audio_proj_out` for the audio rows — and both are repeated, as the paper
does for the two towers of LTX-2.3.

Both training and generation only ever need *linear combinations* of the `N` heads over a contiguous range (the paper's
layer fusion, §3.1):

* generation advances a whole block, `X_{n+L} = X_n + sum_{l=n}^{n+L-1} h_l * u_l`, i.e. the combination with
  coefficients `h_l = t_{l+1} - t_l`;
* training needs the intra-block state `X_k = X_n + sum_{l=n}^{k-1} h_l * u_l` (same form, shorter range), the single
  velocity `u_k` the loss regresses (a one-hot combination) and the `L_min`-interval advance of the data-free
  algorithm's carried state.

[`MiniMaxH3ParallelHead`] therefore takes a *plan*: a `(num_directions, N)` coefficient matrix. It fuses the heads into
`num_directions` linear layers and returns their outputs stacked on the channel axis, so one backbone evaluation yields
every direction a step needs and the cost of the enlarged head never scales with `N`.

The frozen teacher is the same module under [`teacher_mode`]: the low-rank updates of the backbone are switched off and
both heads fall back to the pre-trained weights they were built from. There is no second copy of the 33 B backbone.
"""

import contextlib
import json
import os
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    r"""The exponential sigma shift of `MiniMaxH3Scheduler`, `sigma' = s*sigma / (1 + (s-1)*sigma)`."""
    return shift * sigma / (1 + (shift - 1) * sigma)


def pdd_time_grid(shift: float, num_steps: int) -> torch.Tensor:
    r"""
    The PDD time discretization `0 = t_0 < ... < t_N = 1` of a MiniMax-H3 schedule.

    The paper's shift reparameterization (eq. 16), `t_n = shift_s(n/N)` with `shift_s(t) = (t/s) / (1 + (1/s - 1) t)`,
    is algebraically the same grid as `MiniMaxH3Scheduler`'s `t = 1 - sigma'` over a uniform sigma grid, so the grid is
    built from the scheduler's own shift to keep one source of truth for the schedule. A consequence worth relying on:
    the block boundaries of this grid, taken every `L` indices, are exactly the grid `set_timesteps(N / L + 1)` builds,
    so PDD generation reuses the released scheduler unchanged.

    Args:
        shift (`float`): The exponential shift of the schedule (`12.0` video / `3.0` audio as released).
        num_steps (`int`): The grid size `N`.

    Returns:
        `torch.Tensor` of shape `(num_steps + 1,)`, float64: the grid, ascending from `0` to `1`.
    """
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shifted_sigma(shift, sigma)


def pdd_training_plan(step_sizes: torch.Tensor, start: int, targets: Sequence[int], advance: int) -> torch.Tensor:
    r"""
    Every direction one PDD training step needs, from a single backbone evaluation.

    Args:
        step_sizes (`torch.Tensor` of shape `(N,)`): The grid step sizes `h_l = t_{l+1} - t_l`.
        start (`int`): The block start `n`, i.e. the index the state is currently at.
        targets (`Sequence[int]`): The intra-block indices `k` the loss is evaluated at, each `n <= k < N`.
        advance (`int`): How many intervals the carried state moves after the step, i.e. `L_min`.

    Returns:
        `torch.Tensor` of shape `(2 * len(targets) + 1, N)`: for every target, the displacement from `X_n` to `X_k`
        followed by the row that selects `u_k`; then, last, the displacement from `X_n` to `X_{n+L_min}`.
    """
    plan = torch.zeros(2 * len(targets) + 1, step_sizes.shape[0], dtype=step_sizes.dtype, device=step_sizes.device)
    for position, target in enumerate(targets):
        plan[2 * position, start:target] = step_sizes[start:target]
        plan[2 * position + 1, target] = 1.0
    plan[-1, start : start + advance] = step_sizes[start : start + advance]
    return plan


def pdd_sampling_plan(step_sizes: torch.Tensor, start: int, block_size: int) -> torch.Tensor:
    r"""
    The single direction a PDD generation step needs: the *mean* velocity of the whole block.

    Normalizing the fused displacement by the block span turns it into the block's average velocity, which is what an
    ordinary Euler step over the block boundaries consumes — so `MiniMaxH3Scheduler` drives PDD generation unchanged.

    Args:
        step_sizes (`torch.Tensor` of shape `(N,)`): The grid step sizes `h_l = t_{l+1} - t_l`.
        start (`int`): The block start `n`.
        block_size (`int`): The block size `L`.

    Returns:
        `torch.Tensor` of shape `(1, N)`: the plan.
    """
    plan = torch.zeros(1, step_sizes.shape[0], dtype=step_sizes.dtype, device=step_sizes.device)
    span = step_sizes[start : start + block_size].sum()
    plan[0, start : start + block_size] = step_sizes[start : start + block_size] / span
    return plan


class MiniMaxH3ParallelHead(nn.Module):
    r"""
    The `N` per-interval output heads of a PDD parallel decoder, in place of one final linear layer.

    The heads are held as a single `(num_steps, out_features, in_features)` parameter, every slice initialized from the
    pre-trained layer this replaces — so at initialization every interval predicts exactly the teacher's velocity and
    the parallel decoder starts as the teacher. That pre-trained layer is also kept as a frozen buffer pair, which is
    what [`teacher_mode`] switches to: the teacher's instantaneous velocity stays available from the same module after
    the heads have moved.

    `forward` does not evaluate the heads one by one: it fuses them into the `num_directions` linear maps of the
    current `plan` and applies those, which is the paper's layer fusion (§3.1) and keeps the head's cost independent of
    `num_steps`.

    Args:
        source (`nn.Linear`): The pre-trained final layer to repeat.
        num_steps (`int`): The grid size `N`, i.e. how many heads to hold.
    """

    def __init__(self, source: nn.Linear, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = nn.Parameter(source.weight.detach()[None].repeat(num_steps, 1, 1).clone())
        self.bias = (
            None if source.bias is None else nn.Parameter(source.bias.detach()[None].repeat(num_steps, 1).clone())
        )
        self.register_buffer("teacher_weight", source.weight.detach().clone(), persistent=False)
        self.register_buffer(
            "teacher_bias", None if source.bias is None else source.bias.detach().clone(), persistent=False
        )
        self.teacher = False
        # `plan` is a plain attribute rather than a buffer: it is per-step control flow, not model state to
        # serialize or shard. The default reproduces the source layer, so an unplanned head is the teacher's head.
        self.plan = torch.zeros(1, num_steps)
        self.plan[0, 0] = 1.0

    def set_plan(self, plan: torch.Tensor) -> None:
        r"""
        Set the `(num_directions, num_steps)` coefficient matrix the next forward fuses the heads with.

        Args:
            plan (`torch.Tensor`): The plan. Row `p` weights the `N` heads into the `p`-th output direction.
        """
        if plan.ndim != 2 or plan.shape[1] != self.num_steps:
            raise ValueError(
                f"A PDD plan must be a `(num_directions, {self.num_steps})` matrix, got {list(plan.shape)}."
            )
        self.plan = plan

    @property
    def num_directions(self) -> int:
        return 1 if self.teacher else self.plan.shape[0]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(..., in_features)`): The backbone's final hidden state.

        Returns:
            `torch.Tensor` of shape `(..., num_directions * out_features)`: the planned directions, stacked on the
            channel axis in plan-row order. Under [`teacher_mode`] this is the single pre-trained direction.
        """
        if self.teacher:
            return F.linear(hidden_states, self.teacher_weight, self.teacher_bias)
        plan = self.plan.to(device=self.weight.device, dtype=self.weight.dtype)
        weight = torch.einsum("pn,noi->poi", plan, self.weight).flatten(0, 1)
        bias = None if self.bias is None else torch.einsum("pn,no->po", plan, self.bias).flatten()
        return F.linear(hidden_states, weight, bias)


class LoRALinear(nn.Module):
    r"""
    A frozen `nn.Linear` with a trainable low-rank update, `y = W x + b + (alpha / rank) * B A x`.

    The adapter parameters are held in float32 and cast to the activation dtype inside `forward`, so the optimizer sees
    float32 master weights while the matmuls stay at the backbone's precision. `B` starts at zero, so the wrapped
    module is exactly the frozen layer at initialization — and is again exactly the frozen layer whenever `enabled` is
    false, which is how [`teacher_mode`] recovers the teacher without a second copy of the backbone.

    Args:
        base (`nn.Linear`): The layer to wrap. It is frozen here.
        rank (`int`): The rank of the update.
        alpha (`float`): The scaling numerator; `alpha == rank` means a unit-scaled update.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scaling = alpha / rank
        self.enabled = True
        self.lora_down = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_up = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_down, a=5**0.5)

    # MiniMax-H3 reads `linear.weight.dtype` off its projections to align activations with the mixed-precision
    # checkpoint, so the wrapper has to present the wrapped layer's own tensors under the usual names.
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.base(hidden_states)
        if not self.enabled:
            return out
        update = F.linear(
            F.linear(hidden_states, self.lora_down.to(hidden_states.dtype)),
            self.lora_up.to(hidden_states.dtype),
        )
        return out + self.scaling * update.to(out.dtype)


def attach_parallel_decoder(transformer, num_steps: int) -> None:
    r"""
    Turn a `MiniMaxH3Transformer3DModel` into a PDD parallel decoder, in place.

    Both final heads are replaced by [`MiniMaxH3ParallelHead`]s of `num_steps` heads each, initialized from the
    weights they replace. Nothing else about the model changes: the two heads keep the names `proj_out` and
    `audio_proj_out`, so the float32 pinning of the mixed-precision checkpoint (`_keep_in_fp32_modules`) and the
    forward that reads `self.proj_out.weight.dtype` both still apply.

    Args:
        transformer (`MiniMaxH3Transformer3DModel`): The model to convert.
        num_steps (`int`): The PDD grid size `N`.
    """
    transformer.proj_out = MiniMaxH3ParallelHead(transformer.proj_out, num_steps)
    transformer.audio_proj_out = MiniMaxH3ParallelHead(transformer.audio_proj_out, num_steps)


def add_lora(module: nn.Module, target_names: Sequence[str], rank: int, alpha: float) -> int:
    r"""
    Wrap every `nn.Linear` whose qualified name ends in one of `target_names` with a [`LoRALinear`], in place.

    Args:
        module (`nn.Module`): The root to walk.
        target_names (`Sequence[str]`): Qualified-name suffixes to match, e.g. `("to_q", "ff.net.2")`.
        rank (`int`): The rank of every adapter.
        alpha (`float`): The scaling numerator of every adapter.

    Returns:
        `int`: The number of layers wrapped.
    """
    targets = [
        (name, child)
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear) and any(name.endswith(suffix) for suffix in target_names)
    ]
    for name, child in targets:
        parent_name, _, attribute = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attribute, LoRALinear(child, rank, alpha))
    return len(targets)


def set_parallel_plan(transformer, video_plan: torch.Tensor, audio_plan: torch.Tensor) -> None:
    r"""Set the plans of both parallel heads for the next forward pass."""
    transformer.proj_out.set_plan(video_plan)
    transformer.audio_proj_out.set_plan(audio_plan)


@contextlib.contextmanager
def teacher_mode(transformer):
    r"""
    Run `transformer` as the frozen pre-trained teacher.

    The low-rank updates of the backbone are switched off and both parallel heads fall back to the weights they were
    built from, so the forward is bit-for-bit the released model's instantaneous velocity — with a single output
    direction rather than the planned ones.
    """
    heads = [module for module in transformer.modules() if isinstance(module, MiniMaxH3ParallelHead)]
    adapters = [module for module in transformer.modules() if isinstance(module, LoRALinear)]
    for head in heads:
        head.teacher = True
    for adapter in adapters:
        adapter.enabled = False
    try:
        yield transformer
    finally:
        for head in heads:
            head.teacher = False
        for adapter in adapters:
            adapter.enabled = True


def pdd_teacher_mean_velocity(teacher, forward_kwargs, video, audio, index, grids, solver: str):
    r"""
    A Runge-Kutta estimate of the teacher's mean velocity over interval `index` of the grid (eq. 5 / eq. 6).

    Video and audio ride the same block structure on two schedules, so both modalities take the stage together and
    each advances by its own step size. The caller must already have put the model in [`teacher_mode`].

    Args:
        teacher: The transformer, under [`teacher_mode`].
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


def pdd_state_dict(transformer) -> dict:
    r"""
    The trainable PDD state of a parallel decoder: the two enlarged heads and every low-rank update.

    The frozen backbone is not included, so a checkpoint is a few gigabytes rather than the 62 GB of the base model.
    """
    trainable = {
        name
        for name, module in transformer.named_modules()
        if isinstance(module, (MiniMaxH3ParallelHead, LoRALinear))
    }
    return {
        name: value.detach().cpu()
        for name, value in transformer.state_dict().items()
        if any(name.startswith(f"{prefix}.") for prefix in trainable) and ".base." not in name
    }


PDD_WEIGHTS_NAME = "pdd.safetensors"
PDD_EMA_WEIGHTS_NAME = "pdd_ema.safetensors"
# Pre-rename checkpoints stored live weights here; resume still accepts it.
PDD_LEGACY_LIVE_WEIGHTS_NAME = "pdd_live.safetensors"


def resolve_pdd_lora_path(path):
    r"""
    A checkpoint directory or a weights file.

    A directory prefers `pdd_ema.safetensors` (the EMA inference export) and falls back to `pdd.safetensors`
    (live weights, or the EMA file on checkpoints written before the rename).
    """
    if path is None:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path):
        ema = os.path.join(path, PDD_EMA_WEIGHTS_NAME)
        live = os.path.join(path, PDD_WEIGHTS_NAME)
        if os.path.isfile(ema):
            path = ema
        elif os.path.isfile(live):
            path = live
        else:
            raise FileNotFoundError(
                f"PDD checkpoint directory {path} has neither {PDD_EMA_WEIGHTS_NAME} nor {PDD_WEIGHTS_NAME}."
            )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDD checkpoint does not exist: {path}")
    return path


def load_pdd_config(weights_path):
    r"""Rank / alpha / targets / grid next to the weights file (`pdd_config.json`)."""
    config = {
        "pdd_num_steps": 32,
        "pdd_block_size": 4,
        "lora_rank": 64,
        "lora_alpha": 64.0,
        "lora_targets": "to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear",
    }
    config_path = os.path.join(os.path.dirname(weights_path), "pdd_config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        aliases = {"lora_rank": "rank", "lora_alpha": "network_alpha", "lora_targets": "target_name"}
        for key in config:
            if key in saved:
                config[key] = saved[key]
            elif aliases.get(key) in saved:
                config[key] = saved[aliases[key]]
    if not isinstance(config["lora_targets"], str):
        config["lora_targets"] = ",".join(config["lora_targets"])
    return config


def load_pdd_lora(transformer, pdd_lora_path):
    r"""
    Attach the parallel heads and LoRA, then load the resolved PDD weights into `transformer`.

    A checkpoint directory loads `pdd_ema.safetensors` when present (EMA inference export) and otherwise
    `pdd.safetensors`. Returns the config the predict scripts need to arm the heads and pick NFE.
    """
    path = resolve_pdd_lora_path(pdd_lora_path)
    config = load_pdd_config(path)
    add_lora(
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


def pdd_num_inference_steps(config, num_inference_steps, teacher_default=None):
    r"""Keep `num_inference_steps` when it divides `N`; otherwise snap a leftover teacher default to `N / L`."""
    grid = int(config["pdd_num_steps"])
    steps = int(num_inference_steps)
    if grid % steps == 0:
        return steps
    block = int(config["pdd_block_size"])
    if teacher_default is not None and steps == int(teacher_default) and block > 0 and grid % block == 0:
        nfe = grid // block
        print(f"PDD checkpoint: using num_inference_steps {nfe} (grid {grid}, block {block})", flush=True)
        return nfe
    raise ValueError(f"num_inference_steps {steps} must divide PDD grid size {grid}.")


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
