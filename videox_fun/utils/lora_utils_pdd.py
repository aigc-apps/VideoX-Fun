r"""Model-agnostic parts of Parallel Decoding Distillation (PDD).

PDD (arXiv 2607.26004) turns a pre-trained flow model into a *parallel decoder*: the sampling interval is discretized
into `N` intervals grouped into blocks of size `L`, and one network evaluation predicts the **mean velocity of every
interval of the next block** instead of the single instantaneous velocity. Generation then advances `L` intervals per
evaluation, i.e. `NFE = N / L`.

Everything here works on plain `nn.Linear`s and tensors, with no reference to any particular transformer: the math of
the time grid and the head plans, the [`PDDParallelHead`] / [`PDDLoRALinear`] modules, the teacher switch, and the checkpoint
resolution. Porting a model supplies only the glue that knows where the final linear layers live and how a forward is
called — `videox_fun/models/minimax_h3_pdd.py` is the MiniMax-H3 reference: an attach step that swaps the final heads
for [`PDDParallelHead`]s, a teacher mean-velocity estimate in the model's calling convention, a step callback in the
pipeline's callback protocol, and a `load_pdd_lora` that ties them together.
"""

import contextlib
import json
import os
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    r"""The exponential sigma shift of a rectified-flow schedule, `sigma' = s*sigma / (1 + (s-1)*sigma)`."""
    return shift * sigma / (1 + (shift - 1) * sigma)


def pdd_time_grid(shift: float, num_steps: int) -> torch.Tensor:
    r"""
    The PDD time discretization `0 = t_0 < ... < t_N = 1` of a rectified-flow schedule with an exponential sigma shift.

    The paper's shift reparameterization (eq. 16), `t_n = shift_s(n/N)` with `shift_s(t) = (t/s) / (1 + (1/s - 1) t)`,
    is algebraically the same grid as `t = 1 - sigma'` over a uniform sigma grid — the convention of MiniMax-H3's
    scheduler, where `t = 1` is clean. A consequence worth relying on when the model's scheduler is a plain Euler
    rectified-flow one: the block boundaries of this grid, taken every `L` indices, are exactly the grid
    `set_timesteps(N / L + 1)` builds, so PDD generation reuses such a released scheduler unchanged.

    Args:
        shift (`float`): The exponential shift of the schedule (`12.0` video / `3.0` audio for MiniMax-H3; `1.0` is
            a uniform grid).
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
    ordinary Euler step over the block boundaries consumes — so a plain rectified-flow scheduler drives PDD
    generation unchanged.

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


class PDDParallelHead(nn.Module):
    r"""
    The `N` per-interval output heads of a PDD parallel decoder, in place of one final linear layer.

    The heads are held as a single `(num_steps, out_features, in_features)` parameter, every slice initialized from the
    pre-trained layer this replaces — so at initialization every interval predicts exactly the teacher's velocity and
    the parallel decoder starts as the teacher. That pre-trained layer is also kept as a frozen buffer pair, which is
    what [`pdd_teacher_mode`] switches to: the teacher's instantaneous velocity stays available from the same module after
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
            channel axis in plan-row order. Under [`pdd_teacher_mode`] this is the single pre-trained direction.
        """
        if self.teacher:
            return F.linear(hidden_states, self.teacher_weight, self.teacher_bias)
        plan = self.plan.to(device=self.weight.device, dtype=self.weight.dtype)
        weight = torch.einsum("pn,noi->poi", plan, self.weight).flatten(0, 1)
        bias = None if self.bias is None else torch.einsum("pn,no->po", plan, self.bias).flatten()
        return F.linear(hidden_states, weight, bias)


class PDDLoRALinear(nn.Module):
    r"""
    A frozen `nn.Linear` with a trainable low-rank update, `y = W x + b + (alpha / rank) * B A x`.

    The PDD counterpart of `lora_utils.py`'s `LoRAModule`, and deliberately not built on it: the adapter is a node in
    the model tree (so FSDP shards it and the base layer stays visible for dtype pinning), not an out-of-tree
    forward patch, and it must collapse to exactly the frozen layer when disabled.

    The adapter parameters are held in float32 and cast to the activation dtype inside `forward`, so the optimizer sees
    float32 master weights while the matmuls stay at the backbone's precision. `B` starts at zero, so the wrapped
    module is exactly the frozen layer at initialization — and is again exactly the frozen layer whenever `enabled` is
    false, which is how [`pdd_teacher_mode`] recovers the teacher without a second copy of the backbone.

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

    # Models may read `linear.weight.dtype` off their projections to align activations with a mixed-precision
    # checkpoint (MiniMax-H3 does), so the wrapper has to present the wrapped layer's own tensors under the usual
    # names.
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


def add_pdd_lora(module: nn.Module, target_names: Sequence[str], rank: int, alpha: float) -> int:
    r"""
    Wrap every `nn.Linear` whose qualified name ends in one of `target_names` with a [`PDDLoRALinear`], in place.

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
        setattr(parent, attribute, PDDLoRALinear(child, rank, alpha))
    return len(targets)


def merge_pdd_lora(module: nn.Module) -> int:
    r"""
    Fold every [`PDDLoRALinear`] into its frozen base layer and unwrap it, in place.

    An inference-only optimization: afterwards each wrapped layer is a plain `nn.Linear` whose weight already carries
    `scaling * B A`, so a forward costs one matmul instead of three. The [`PDDParallelHead`]s are left untouched — their
    effective weight changes every step with the `plan`, so they cannot be folded into a static layer.

    Two things to know before calling this. Merging overwrites the base weight, so it destroys the `enabled=False`
    fallback [`pdd_teacher_mode`] relies on: only merge on a pure student inference path. And the update is accumulated
    in float32 (the adapters' storage dtype) then cast back to the base weight's dtype, so the result is not
    bit-for-bit the un-merged forward — it rounds the low-rank delta into the backbone precision.

    Call this before any device offload / quantization / FSDP wrap is registered on the model, so no hook has to be
    rebuilt and the delta lands on the unquantized weight.

    Args:
        module (`nn.Module`): The root to walk, e.g. the transformer returned by `load_pdd_lora`.

    Returns:
        `int`: The number of adapters merged.
    """
    # Collect first, then mutate: replacing a child while walking `module.modules()` would perturb the traversal.
    adapters = [
        (parent, attribute, child)
        for parent in module.modules()
        for attribute, child in parent.named_children()
        if isinstance(child, PDDLoRALinear)
    ]
    for parent, attribute, adapter in adapters:
        base = adapter.base
        with torch.no_grad():
            weight = base.weight
            delta = adapter.scaling * (
                adapter.lora_up.to(weight.device) @ adapter.lora_down.to(weight.device)
            )
            weight.data = (weight.data.to(delta.dtype) + delta).to(weight.dtype)
        setattr(parent, attribute, base)
    return len(adapters)


@contextlib.contextmanager
def pdd_teacher_mode(transformer):
    r"""
    Run `transformer` as the frozen pre-trained teacher.

    The low-rank updates of the backbone are switched off and every [`PDDParallelHead`] falls back to the weights it was
    built from, so the forward is bit-for-bit the released model's instantaneous velocity — with a single output
    direction rather than the planned ones.
    """
    heads = [module for module in transformer.modules() if isinstance(module, PDDParallelHead)]
    adapters = [module for module in transformer.modules() if isinstance(module, PDDLoRALinear)]
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


def _strip_fsdp_wrapper(name: str) -> str:
    r"""Drop the `_fsdp_wrapped_module` path segments FSDP injects around each separately-wrapped child unit."""
    return ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module")


def pdd_state_dict(transformer, state_dict: Optional[dict] = None) -> dict:
    r"""
    The trainable PDD state of a parallel decoder: the enlarged heads and every low-rank update.

    The frozen backbone is not included, so a checkpoint is a few gigabytes rather than the full size of the base
    model.

    Args:
        transformer (`nn.Module`): The parallel decoder; its module tree decides which keys are trainable.
        state_dict (`dict`, optional): The weights to filter, defaulting to `transformer.state_dict()`. Under FSDP,
            pass an already-gathered `FULL_STATE_DICT` here, since the live module views are sharded.

    FSDP wraps every child unit in a `_fsdp_wrapped_module`, so a wrapped module's `named_modules` path
    (`blocks.0._fsdp_wrapped_module.attn.to_q`) never matches the clean keys of a gathered `FULL_STATE_DICT`
    (`blocks.0.attn.to_q.lora_down`) and only the root wrap unit would survive the filter. Both sides are normalized
    through [`_strip_fsdp_wrapper`] so the block LoRA and the parallel heads are kept too.
    """
    if state_dict is None:
        state_dict = transformer.state_dict()
    trainable = {
        _strip_fsdp_wrapper(name)
        for name, module in transformer.named_modules()
        if isinstance(module, (PDDParallelHead, PDDLoRALinear))
    }
    return {
        _strip_fsdp_wrapper(name): value.detach().cpu()
        for name, value in state_dict.items()
        if any(_strip_fsdp_wrapper(name).startswith(f"{prefix}.") for prefix in trainable) and ".base." not in name
    }


PDD_WEIGHTS_NAME = "pdd.safetensors"
PDD_EMA_WEIGHTS_NAME = "pdd_ema.safetensors"
# Pre-rename checkpoints stored live weights here; resume still accepts it.
PDD_LEGACY_LIVE_WEIGHTS_NAME = "pdd_live.safetensors"

# The released MiniMax-H3 recipe. Every field is meant to be overridden by the `pdd_config.json` a training run
# writes next to its weights; a port to another model passes its own `defaults` to [`load_pdd_config`] instead.
PDD_DEFAULT_CONFIG = {
    "pdd_num_steps": 32,
    "pdd_block_size": 4,
    "lora_rank": 64,
    "lora_alpha": 64.0,
    "lora_targets": "to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear",
}


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


def load_pdd_config(weights_path, defaults=None):
    r"""Rank / alpha / targets / grid next to the weights file (`pdd_config.json`), over `defaults`."""
    config = dict(PDD_DEFAULT_CONFIG if defaults is None else defaults)
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
