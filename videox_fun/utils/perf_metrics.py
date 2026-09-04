"""Environment-gated inference and training metrics for `videox_fun`.

This module is instrumentation only: it times, it never computes. Nothing here touches a tensor that feeds a
model, so a run with metrics on and a run with metrics off produce bit-identical outputs.

It is off unless `VIDEOX_PERF` is set, and "off" is literal: [`install`] and [`install_training`] return on their
first line, nothing is wrapped and no hook is registered, so a default run pays nothing at all. `import videox_fun`
does not even load this file with the variable unset, that bootstrap being guarded by it; importing
`videox_fun.utils` or `videox_fun.pipeline` does load it either way, which costs one bytecode load and no new
dependencies -- everything above is the standard library plus the `torch` those packages already import.

The inference half wires itself up in two layers:

* [`install`] wraps the `__call__` of every pipeline class exported by `videox_fun.pipeline`, which is what marks
  a *request* boundary -- where the counters are reset and where the one and only `cuda.synchronize` of the whole
  scheme happens, at a point the caller was about to synchronize anyway to save its output.
* At the start of every request, [`_attach_hooks`] walks `pipe.components` and hooks whichever module components
  are not hooked yet. Attaching *lazily* rather than at construction is what makes this work under FSDP and
  sequence parallel: the entry scripts reassign `pipeline.transformer = shard_fn(pipeline.transformer)` after
  building the pipeline, so a hook attached at construction would sit on a discarded object, while one attached per
  request lands on whatever the pipeline actually runs.

Per-step timings therefore come out of the natural granularity of the denoising loop -- one transformer forward
per step, two under CFG -- without the loop itself being touched.

The training half is [`install_training`], and it hangs off `accelerate` rather than off this repo's scripts,
because `Accelerator` is the one thing all of them have in common; see its docstring for the phase model. It adds
no synchronize at all.

Environment variables:
    VIDEOX_PERF: `1` for a per-request (inference) or per-window (training) summary plus an exit summary, `2` to
        also dump every step. Unset or `0` disables the module entirely.
    VIDEOX_PERF_JSON: path to append one JSON object per request / per window to. Suffixed with `.rank{N}` under
        multi-GPU so ranks never share a file. Each object carries a `kind` telling the two apart.
    VIDEOX_PERF_WARMUP: number of leading requests / global steps to exclude from the exit summary (they are still
        logged).
    VIDEOX_PERF_RANKS: `0` (default) to log from rank 0 only, `all` to log from every rank.
    VIDEOX_PERF_PEAK_TFLOPS: per-device hardware bf16 peak to compute MFU against, overriding the built-in device
        table. Under multi-GPU the MFU is taken against this times the world size.
    VIDEOX_PERF_DIT_PARAMS: exact transformer parameter count, overriding the FSDP-aware inference below.
    VIDEOX_PERF_FLOPS_ATTN: `0` to price only the linear layers, leaving out the quadratic core-attention term the
        FLOPs figures include by default. Worth reaching for on the causal models, whose masked attention costs about
        half of what the term charges them.
    VIDEOX_PERF_EVERY: training only, default 50. Global steps per aggregated log line; a line per step is
        unreadable over the tens of thousands of steps a real run takes.
    VIDEOX_PERF_TOTAL_STEPS: training only. Total planned steps, enabling a remaining-time estimate. Not guessed
        when unset -- `max_train_steps` lives in the entry script's argparse and cannot be read from here.
    VIDEOX_PERF_FLOPS_COEF: training only. Overrides the automatically chosen FLOPs multiplier; see
        [`flops_coef`]. Setting it collapses the reported MFU and HFU onto each other.

Note that when enabled this module calls `torch.cuda.reset_peak_memory_stats()` on the compute device once per
request (inference) or once per window (training), so any caller reading the peak memory counters itself sees them
scoped to the same interval.
"""

import atexit
import collections
import contextlib
import functools
import json
import logging
import os
import statistics
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

# Dense bf16 tensor-core peaks in TFLOPS, matched as substrings against `torch.cuda.get_device_name()`. Only
# devices with a published dense figure are listed; an unmatched device reports no MFU rather than a made-up one,
# and `VIDEOX_PERF_PEAK_TFLOPS` covers anything missing.
_PEAK_TFLOPS_BF16 = {
    "A100": 312.0,
    "A800": 312.0,
    "H100": 989.0,
    "H800": 989.0,
    "H200": 989.0,
    "H20": 148.0,
    "L40S": 362.0,
    "L20": 119.5,
    "RTX 4090": 165.2,
}

logger = logging.getLogger("videox_fun.perf")

_INSTALLED = False
_STATE: Optional["_MetricsState"] = None
# The request currently being measured, held per thread. The component hooks are permanent once attached, so they
# key off this to know whether they are inside a measured request -- and a `None` here is what makes them a no-op.
# It is thread-local because a server can have two requests in flight at once, and a shared slot would have them
# writing into one record: the hooks always fire on the thread that called the forward, so per-thread state keeps
# concurrent requests from corrupting each other's counts.
_TLS = threading.local()


def _current() -> Optional["_Request"]:
    return getattr(_TLS, "request", None)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class _StageStat:
    """Timings of one component across one request, as raw marker pairs resolved only at the end."""

    __slots__ = ("pairs", "tokens", "batch")

    def __init__(self):
        self.pairs: List[Tuple[Any, Any]] = []
        self.tokens: Optional[int] = None
        self.batch: int = 1

    def ready(self) -> bool:
        """Whether every event pair here has completed, so [`elapsed_ms`] can read all of them.

        Only the closing event of each pair is tested: the opening one was recorded earlier on the same stream, and
        a stream retires events in the order they were recorded, so a completed end implies a completed start.
        """
        for _, (_, event_end) in self.pairs:
            if event_end is not None and not event_end.query():
                return False
        return True

    def elapsed_ms(self) -> List[float]:
        out = []
        for (host_start, event_start), (host_end, event_end) in self.pairs:
            host_ms = (host_end - host_start) * 1000.0
            # `elapsed_time` on an event the device has not reached yet raises, so the readiness of the pair is a
            # precondition for reading it, not an optimisation. The inference path gets there via the synchronize
            # at the request boundary; the training path never synchronizes and instead defers settling a step
            # until its events have retired, falling back to the host clock for the rare pair that never does.
            if (
                event_start is not None
                and event_end is not None
                and event_start.query()
                and event_end.query()
            ):
                # The larger of the two is the one that saw the work; see [`_mark`].
                out.append(max(host_ms, event_start.elapsed_time(event_end)))
            else:
                out.append(host_ms)
        return out


class _Request:
    __slots__ = ("index", "warmup", "t0", "device", "stages", "open_marks")

    def __init__(self, index: int, warmup: bool, device: Optional[torch.device]):
        self.index = index
        self.warmup = warmup
        # Resolved once per request and carried here so that every CUDA call of the request -- the event records in
        # the hooks, the memory counters, the closing synchronize -- names the same device explicitly.
        self.device = device
        self.t0 = time.perf_counter()
        self.stages: Dict[str, _StageStat] = {}
        # Start markers of forwards that have not returned yet, keyed by stage; a list so that a re-entrant
        # component (a VAE decoding chunk by chunk inside an outer call) nests instead of losing its pair.
        self.open_marks: Dict[str, List[Any]] = {}

    def stage(self, name: str) -> _StageStat:
        stat = self.stages.get(name)
        if stat is None:
            stat = _StageStat()
            self.stages[name] = stat
        return stat


class _MetricsState:
    """Process-wide configuration and the accumulated history the exit summary is built from."""

    def __init__(self, level: int):
        self.level = level
        self._json_base = os.environ.get("VIDEOX_PERF_JSON") or None
        self.warmup = _env_int("VIDEOX_PERF_WARMUP", 0)
        self.log_all_ranks = (os.environ.get("VIDEOX_PERF_RANKS", "0").strip().lower() == "all")
        self.peak_tflops_override = _env_float("VIDEOX_PERF_PEAK_TFLOPS")
        self.dit_params_override = _env_int("VIDEOX_PERF_DIT_PARAMS", 0) or None
        # On by default: leaving core attention out understates a video step by two thirds and, worse, by a factor
        # that moves with the sequence length. Off is for the causal models, where charging the full square is an
        # overcount of nearly two, and for anyone who wants the old linear-only bound back.
        self.attn_flops = _env_int("VIDEOX_PERF_FLOPS_ATTN", 1) != 0
        self.num_requests = 0
        self.history: List[Dict[str, Any]] = []
        self._rank = 0
        self._world_size = 1
        self._topology_final = False
        self._reported = False
        self._configure_logger()
        atexit.register(self.report_summary)

    def _resolve_topology(self) -> None:
        """Settle rank and world size, preferring the process group over the environment once it exists.

        [`install`] runs while `videox_fun.pipeline` is being imported, which in the entry scripts is *before*
        `set_multi_gpus_devices` brings the process group up, so at that point the launcher's environment is all
        there is to go on. Re-resolving until the group is initialized means the numbers the FSDP parameter
        recovery and the per-rank json paths depend on are the real ones by the time a request runs.
        """
        if self._topology_final:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
            self._topology_final = True
        else:
            self._rank = _env_int("RANK", 0)
            self._world_size = _env_int("WORLD_SIZE", 1)

    @property
    def rank(self) -> int:
        self._resolve_topology()
        return self._rank

    @property
    def world_size(self) -> int:
        self._resolve_topology()
        return self._world_size

    @property
    def json_path(self) -> Optional[str]:
        if self._json_base is None:
            return None
        # Every rank keeps its own file: ranks share a filesystem, and appending from eight processes to one path
        # interleaves partial lines.
        return f"{self._json_base}.rank{self.rank}" if self.world_size > 1 else self._json_base

    def _configure_logger(self):
        # Own the handler outright instead of relying on the entry script's `basicConfig`: the metrics have to show
        # up whether the process was started by python, torchrun, accelerate or a server framework, and exactly
        # once.
        if not any(getattr(h, "_videox_perf", False) for h in logger.handlers):
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(message)s"))
            handler._videox_perf = True
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    @property
    def tag(self) -> str:
        return f"[Perf][rank{self.rank}]"

    def should_log(self) -> bool:
        return self.log_all_ranks or self.rank == 0

    def peak_tflops(self, device: Optional[torch.device] = None) -> Optional[float]:
        if self.peak_tflops_override is not None:
            return self.peak_tflops_override
        if device is None:
            # `get_device_name` with no argument reads device 0, and on a host with no context up it creates one;
            # see [`_perf_device`].
            return None
        name = torch.cuda.get_device_name(device)
        for key, value in _PEAK_TFLOPS_BF16.items():
            if key in name:
                return value
        return None

    def report_summary(self):
        # Guarded the way [`_TrainState.finish`] is: reporting explicitly should not then be reported again by the
        # `atexit` hook over the very same requests.
        if self._reported:
            return
        measured = [record for record in self.history if not record["warmup"]]
        if not measured or not self.should_log():
            return
        self._reported = True
        e2e = sorted(record["e2e_s"] for record in measured)
        steps = [ms for record in measured for ms in record.get("_step_ms", [])]
        parts = [
            f"{len(measured)} reqs",
            f"e2e p50 {_percentile(e2e, 50):.1f}s p95 {_percentile(e2e, 95):.1f}s",
        ]
        skipped = len(self.history) - len(measured)
        if skipped:
            parts[0] = f"{len(measured)} reqs ({skipped} warmup excluded)"
        if steps:
            parts.append(f"transformer mean {statistics.fmean(steps):.0f}ms/step")
        total = sum(record["e2e_s"] for record in measured)
        if total > 0:
            parts.append(f"{len(measured) / total * 3600.0:.1f} reqs/hour")
        logger.info(f"{self.tag} === {' | '.join(parts)} ===")


def _percentile(values_sorted: List[float], q: float) -> float:
    if not values_sorted:
        return float("nan")
    if len(values_sorted) == 1:
        return values_sorted[0]
    pos = (len(values_sorted) - 1) * q / 100.0
    low = int(pos)
    high = min(low + 1, len(values_sorted) - 1)
    return values_sorted[low] + (values_sorted[high] - values_sorted[low]) * (pos - low)


def _perf_device(pipe) -> Optional[torch.device]:
    """The CUDA device a pipeline computes on, or `None` to keep this module off the CUDA APIs altogether.

    Read off the pipeline's own modules rather than from `torch.cuda.current_device()`, because under multi-GPU the
    two are different devices here. `set_multi_gpus_devices` hands the entry script a `cuda:{local_rank}` to move
    the weights to but never calls `torch.cuda.set_device`, and the `set_device` inside xfuser's
    `init_distributed_environment` is guarded by `if not torch.distributed.is_initialized()`, which is already
    false by the time it runs -- the process group was brought up on the line before. So every rank keeps
    `current_device() == 0` while its model runs on `cuda:{local_rank}`, and a default-argument `synchronize`,
    `Event.record`, `reset_peak_memory_stats` or `max_memory_allocated` would every one of them aim at device 0:
    the synchronize would wait on an idle device instead of the busy one, the events would time an empty stream,
    the memory counters would report a device the pipeline never wrote to -- and each would in passing bring a
    context up on GPU 0 from all eight processes, taking memory from the rank that does have a model there.

    `None` covers a cpu-only run and a pipeline whose weights are offloaded to the host: no CUDA call is made at
    all, so metrics never create a context that the same run without them would not have.
    """
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return None
    for name in ("transformer", "transformer_2", "unet", "vae"):
        module = getattr(pipe, name, None)
        if isinstance(module, torch.nn.Module):
            for param in module.parameters():
                if param.device.type == "cuda":
                    return param.device
    try:
        # Under cpu offload the parameters rest on the host and only accelerate's hooks know where they run.
        device = pipe._execution_device
    except Exception:
        return None
    return device if getattr(device, "type", None) == "cuda" else None


def _mark(device: Optional[torch.device]) -> Tuple[float, Any]:
    """A timing marker: a host timestamp, plus a recorded CUDA event when the request runs on a device.

    Both are taken because neither alone is right for every component. A device forward returns to the host as soon
    as its kernels are *queued*, so the host clock on its own reports submission time; conversely a pipeline can
    hold a module that genuinely runs on the host -- a text encoder left on cpu, a vae under offload -- and a CUDA
    event pair around such a forward measures nothing, the two events executing back to back with no work between
    them. Resolving the pair with a `max` picks whichever of the two actually saw the work, per forward, with no
    need to guess where a module lives.

    The event is only *recorded* here, never waited on -- `cudaEventRecord` is a few microseconds against a step of
    several hundred milliseconds, and the elapsed time is read at the end of the request.
    """
    if device is not None:
        event = torch.cuda.Event(enable_timing=True)
        # Named stream rather than the default one: it pins the event to the pipeline's device (see
        # [`_perf_device`]) and follows any `with torch.cuda.stream(...)` the pipeline put the forward on.
        event.record(torch.cuda.current_stream(device))
        return time.perf_counter(), event
    return time.perf_counter(), None


def _infer_tokens(module: torch.nn.Module, args, kwargs) -> Optional[Tuple[int, int]]:
    """`(tokens per sample, batch)` of one transformer forward, or `None` when the inputs do not say.

    This only scales the analytic FLOPs, so it has to reflect what the blocks actually run over. Three input
    conventions appear across the families in this repo:

    * The Wan models are handed an explicit `seq_len`, the padded length their blocks run on, which is
      authoritative over anything derived from the latent's shape.
    * A packed sequence arrives already tokenized as `(B, tokens, D)`, as in MiniMax-H3.
    * A video latent arrives as `(B, C, F, H, W)` and only becomes tokens through the model's own patch size.

    The batch is returned separately rather than folded in, because it is a multiplier on the cost but not on the
    sequence length the log reports -- classifier-free guidance doubles the former and leaves the latter alone.
    Anything unrecognized reports nothing, so the FLOPs line is dropped rather than guessed.
    """
    hidden = kwargs.get("hidden_states")
    if hidden is None:
        hidden = kwargs.get("x")  # the Wan models name it `x`
    if hidden is None and args:
        hidden = args[0]
    if not torch.is_tensor(hidden) or hidden.ndim < 3:
        return None
    batch = int(hidden.shape[0])

    seq_len = kwargs.get("seq_len")
    if isinstance(seq_len, int) and seq_len > 0:
        return seq_len, batch
    if hidden.ndim == 3:
        return int(hidden.shape[1]), batch
    if hidden.ndim == 5:
        patch = getattr(getattr(module, "config", None), "patch_size", None)
        if isinstance(patch, int):
            patch = (1, patch, patch)
        if not isinstance(patch, (tuple, list)) or len(patch) != 3:
            return None
        _, _, frames, height, width = hidden.shape
        return int((frames // patch[0]) * (height // patch[1]) * (width // patch[2])), batch
    return None


def _is_sharded(module: torch.nn.Module) -> bool:
    for sub in module.modules():
        if hasattr(sub, "_fsdp_wrapped_module") or type(sub).__name__ == "FullyShardedDataParallel":
            return True
    for param in module.parameters():
        if type(param).__name__ in ("FlatParameter", "DTensor"):
            return True
    return False


def _count_params(module: torch.nn.Module, world_size: int) -> int:
    """Total parameter count of a component, undoing FSDP sharding.

    Under FSDP each rank only holds its shard, so the local `numel` is the full count divided by the world size;
    multiplying it back is right for the flat even sharding `shard_model` sets up. `VIDEOX_PERF_DIT_PARAMS` is the
    escape hatch for any layout where it is not.
    """
    total = sum(param.numel() for param in module.parameters())
    if world_size > 1 and _is_sharded(module):
        total *= world_size
    return total


# The naming conventions the attention projections go by across the families here: `q`/`k`/`v` in the Wan models,
# `to_q`/`to_k`/`to_v` in the diffusers ones, `q_proj`/`k_proj`/`v_proj` in the ones that came from a transformers
# tower.
_ATTN_PROJECTIONS = (("q", "to_q", "q_proj"), ("k", "to_k", "k_proj"), ("v", "to_v", "v_proj"))

# The fused form, where one matrix produces all three: `nn.Linear(dim, dim * 3)` in the LongCat and HiDream blocks.
# The query is the first of three equal shares of its output, which is why an output width that is not a multiple of
# three is not read as one of these.
_ATTN_FUSED = ("qkv", "to_qkv", "qkv_proj")

# Attention stacks that refine the *text* embedding before the blocks run. They are attention by structure, but they
# never see the latent sequence -- they run over a few hundred text rows -- so charging them the latent square is a
# pure overcount. Named in full rather than matched on "refiner": Z-Image has a `noise_refiner` beside its
# `context_refiner`, and that one does run over the latents, so a substring test would drop real work.
_ATTN_TEXT_TOWERS = ("token_refiner", "context_refiner")

# `QK^T` and `AV`, each a multiply-accumulate. The two products are what "core attention" means here, as against the
# q, k, v and output projections, whose cost is linear in the sequence and already inside the parameter count.
_ATTN_CORE_FACTOR = 4.0


def _query_width(module: torch.nn.Module) -> Optional[int]:
    """The width of one module's query projection if it is an attention, or `None` if it is not.

    Duck-typed on `out_features` rather than `isinstance(nn.Linear)`, so that a wrapped projection -- a peft
    `lora.Linear`, or anything else holding a base layer -- is still recognized. Requiring all three of q, k and v
    keeps the looser test from matching a module that merely happens to own an attribute named `v`.

    The separate q, k and v are looked for first and the fused matrix only after, because a model may have both: the
    HiDream tower fuses its own attention and holds an unfused `q_proj` elsewhere, and the unfused reading is the
    one that needs no assumption about how the output is divided.
    """
    found = []
    for candidates in _ATTN_PROJECTIONS:
        for attr in candidates:
            child = getattr(module, attr, None)
            if isinstance(getattr(child, "out_features", None), int):
                found.append(child)
                break
    if len(found) == len(_ATTN_PROJECTIONS):
        return found[0].out_features
    for attr in _ATTN_FUSED:
        width = getattr(getattr(module, attr, None), "out_features", None)
        if isinstance(width, int) and width % 3 == 0:
            return width // 3
    return None


def _attn_widths(module: torch.nn.Module) -> Dict[str, int]:
    """The query-projection width of a model's attention, split by what its keys and values run over.

    The analytic `2 * params * tokens` cost prices the linear layers and nothing else, and at video lengths the
    attention it leaves out is the larger half of the work: core attention grows with the square of the sequence
    while the linear part grows linearly, so it is a third of a step at 8k tokens and two thirds at 28k. That is
    also why a bound that omits it cannot be used to compare two runs at different lengths, which is what it was
    previously documented as being good for. This reads the widths needed to price it back, the way Megatron keeps
    its `self_attn_core_term` separate from its per-token terms rather than folding attention into a parameter
    count.

    Widths come from `in_features` / `out_features` rather than from any config, because those are set in
    `nn.Linear.__init__` and survive what a config does not: the families here name their dimensions a dozen
    different ways, and under FSDP the weights have been flattened away while these remain. They are also the
    *logical* widths, so unlike a parameter count they need no unsharding -- sharding in this repo splits weights
    across ranks (FSDP) or the sequence across ranks (ulysses, ring), and neither narrows a projection.

    Which modules count, and what their query width is, is [`_query_width`]; both the separate and the fused forms
    of the projection are recognized. The query width is what both `QK^T` and `AV` are wide in, including under GQA:
    the fewer key heads are repeated up to the query heads before the product, so the key width does not enter.

    The text-refiner stacks are skipped -- `token_refiner` in MiniMax-H3 and HunyuanVideo, `context_refiner` in
    Z-Image. They are attention over a few hundred text rows, not over the latents, so pricing them by the latent
    sequence overstates them by the ratio of the two lengths squared. They are matched by name in full because
    Z-Image also has a `noise_refiner`, which does run over the latents and must keep counting.

    Keys over the latent sequence and keys over the text are counted apart because only the first is quadratic. A
    module is taken to be cross-attention when its qualified name says so -- `cross_attn` in the Wan blocks, `attn2`
    in the diffusers convention -- and self-attention otherwise. Guessing self-attention is the conservative
    direction for the joint attention that MMDiT runs over text and latents concatenated: its true length is a
    little above the latent count, so charging it the latent count alone understates rather than inflates.

    Nothing here can see whether the attention is masked, and that is the one direction in which this overcounts.
    The models in this repo are bidirectional over the latent sequence and so pay the full square, but the causal
    variants -- `wan_flex_causal_attn`, the self-forcing transformers -- compute about half of it, and Megatron
    halves its own core term for exactly that reason. Reach for `VIDEOX_PERF_FLOPS_ATTN=0` on those runs to fall
    back to the linear-only bound rather than read a figure that is too high by nearly a factor of two.
    """
    widths = {"self": 0, "cross": 0, "modules": 0}
    for name, sub in module.named_modules():
        lowered = name.lower()
        if any(tower in lowered for tower in _ATTN_TEXT_TOWERS):
            continue
        query = _query_width(sub)
        if query is None:
            continue
        cross = "cross" in lowered or lowered.rsplit(".", 1)[-1] == "attn2"
        widths["cross" if cross else "self"] += query
        widths["modules"] += 1
    return widths


def _attn_flops(widths: Optional[Dict[str, int]], tokens: int, text_tokens: int) -> float:
    """Core attention FLOPs of one forward over `tokens` latent tokens, or zero when the widths were unreadable.

    Zero rather than a guess: a model whose attention this could not find falls back to the linear-only bound it
    always reported, which is wrong in a known direction by a known mechanism, and the caller says which of the two
    it used.

    The cross-attention term needs the text length, which is a property of the run and not of the module, so it is
    dropped when unknown. It is worth much less than the self term -- a few percent of the core at video lengths,
    being linear in the sequence where the other is quadratic -- so dropping it moves the total very little.
    """
    if not widths or not widths["self"]:
        return 0.0
    total = _ATTN_CORE_FACTOR * float(tokens) * float(tokens) * widths["self"]
    if text_tokens:
        total += _ATTN_CORE_FACTOR * float(tokens) * float(text_tokens) * widths["cross"]
    return total


def _text_tokens(module: torch.nn.Module) -> int:
    """The padded text length a model's cross-attention runs against, or 0 when it does not advertise one.

    Only the Wan family states it (`text_len`, 512), which is the family whose cross-attention is a separate module
    and therefore the family where the distinction changes anything.
    """
    value = getattr(module, "text_len", None)
    return int(value) if isinstance(value, int) and value > 0 else 0


def _attach_hooks(pipe) -> None:
    """Hook the module components of a pipeline, picking up any that were swapped since the last request.

    Run at the start of every request rather than once, because a component can be replaced after the pipeline was
    built: `pipeline.transformer = shard_fn(pipeline.transformer)` in the entry scripts is exactly that, and a
    wrapper applied between two requests would otherwise never be hooked and would drop its stage from the log
    without saying so. The guards below sit on the modules, so a repeat visit costs one `getattr` per component.
    """
    state = _STATE
    dit_params: Dict[str, int] = {}
    dit_attn: Dict[str, Tuple[Optional[Dict[str, int]], int]] = {}

    try:
        components = dict(pipe.components)
    except Exception:  # a pipeline may expose a component it cannot resolve; metrics must never break a run
        components = {}

    for name, component in components.items():
        if not isinstance(component, torch.nn.Module):
            continue  # tokenizer, processor, scheduler

        # Counted once per module and cached on it: walking the parameters of a 14B transformer on every request
        # would be wasted work, and the count cannot change once the sharding is in place.
        if name.startswith("transformer"):
            params = getattr(component, "_videox_perf_params", None)
            if params is None:
                params = state.dit_params_override or _count_params(component, state.world_size)
                component._videox_perf_params = params
            if params:
                dit_params[name] = params
            # Cached on the module beside the parameter count and for the same reason: reading the projection widths
            # walks every submodule, and they cannot change once the model is built.
            attn = getattr(component, "_videox_perf_attn", None)
            if attn is None:
                widths = _attn_widths(component) if state.attn_flops else None
                widths = widths if widths and widths["self"] else None
                attn = (widths, _text_tokens(component) if widths else 0)
                component._videox_perf_attn = attn
            dit_attn[name] = attn

        # The guard is on the module rather than on the pipeline, so that a module shared by two pipelines -- a
        # base and a refiner over the same vae -- is hooked once. Hooking it twice would append two marker pairs
        # per forward and double every count it appears in.
        if getattr(component, "_videox_perf_hooked", False):
            continue
        component._videox_perf_hooked = True

        if "vae" in name:
            # Pipelines call `vae.encode` / `vae.decode`; `vae.forward` never runs, so hooking it would report
            # nothing. Wrapping the two bound methods on the instance also splits the two directions apart. The
            # streaming pair shares those same two stages, being the same work done in chunks: `AutoencoderKLWan` is
            # the one vae that has them, and `pipeline_wan_self_forcing` reaches it only through `decode_stream`,
            # whose time would otherwise land in `other`. Neither streaming method calls the plain one, so sharing
            # the stage cannot double-count.
            for method_name, stage in (
                ("encode", f"{name}_enc"),
                ("decode", f"{name}_dec"),
                ("encode_stream", f"{name}_enc"),
                ("decode_stream", f"{name}_dec"),
            ):
                method = getattr(component, method_name, None)
                if callable(method):
                    setattr(component, method_name, _wrap_timed_method(method, stage))
            continue

        component.register_forward_pre_hook(_make_pre_hook(name), with_kwargs=True)
        component.register_forward_hook(_make_post_hook(name), with_kwargs=True)

    pipe._videox_perf_dit_params = dit_params
    pipe._videox_perf_dit_attn = dit_attn


def _make_pre_hook(name: str):
    def pre_hook(module, args, kwargs):
        request = _current()
        if request is None:
            return
        stat = request.stage(name)
        if name.startswith("transformer") and stat.tokens is None:
            shape = _infer_tokens(module, args, kwargs)
            if shape is not None:
                stat.tokens, stat.batch = shape
        request.open_marks.setdefault(name, []).append(_mark(request.device))

    return pre_hook


def _make_post_hook(name: str):
    def post_hook(module, args, kwargs, output):
        request = _current()
        if request is None:
            return
        marks = request.open_marks.get(name)
        if marks:
            request.stage(name).pairs.append((marks.pop(), _mark(request.device)))

    return post_hook


def _wrap_timed_method(method, stage: str):
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        request = _current()
        if request is None:
            return method(*args, **kwargs)
        start = _mark(request.device)
        try:
            return method(*args, **kwargs)
        finally:
            request.stage(stage).pairs.append((start, _mark(request.device)))

    wrapper._videox_perf = True
    return wrapper


def _begin_request(pipe) -> Optional[_Request]:
    if _current() is not None:
        # A pipeline invoked from inside another (e.g. a latent upsampler): let the outer request own the timings
        # instead of overwriting them.
        return None
    try:
        _attach_hooks(pipe)
        state = _STATE
        device = _perf_device(pipe)
        request = _Request(state.num_requests, warmup=state.num_requests < state.warmup, device=device)
        state.num_requests += 1
        if device is not None:
            torch.cuda.reset_peak_memory_stats(device)
    except Exception as error:  # never let measurement take down a generation run
        logger.warning(f"[Perf] failed to start metrics: {error!r}")
        return None
    _TLS.request = request
    return request


def _end_request(pipe, request: Optional[_Request]) -> None:
    if request is None:
        return
    _TLS.request = None
    try:
        if request.device is not None:
            # Before the wall clock is read, not after. A pipeline `__call__` returns once the last kernel is
            # *queued*, so an unsynchronized reading would time how long the host took to submit the work, not how
            # long the device took to do it -- and would come out below the device time it is meant to bound.
            # This is the only synchronize of the scheme. On the usual path it is free, the caller being about to
            # read the generated frames on the host anyway; with `output_type="latent"` it does add a wait the same
            # run without metrics would not have, which is the one place this module is not entirely free. It is a
            # local wait on one device, never a collective, so it cannot deadlock or skew a rank group.
            torch.cuda.synchronize(request.device)
        e2e = time.perf_counter() - request.t0
        _settle_request(pipe, request, e2e)
    except Exception as error:  # never let measurement take down a generation run
        logger.warning(f"{_STATE.tag} failed to report metrics: {error!r}")


def _settle_request(pipe, request: _Request, e2e: float) -> None:
    state = _STATE
    stages: Dict[str, Dict[str, float]] = {}
    step_ms: List[float] = []
    for name, stat in request.stages.items():
        values = stat.elapsed_ms()
        if not values:
            continue
        stages[name] = {
            "count": len(values),
            "total_ms": sum(values),
            "mean_ms": statistics.fmean(values),
            "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_ms": min(values),
            "max_ms": max(values),
        }
        if name.startswith("transformer"):
            step_ms.extend(values)

    dit = _dit_throughput(pipe, request, stages)
    record: Dict[str, Any] = {
        "kind": "request",
        "ts": time.time(),
        "rank": state.rank,
        "world_size": state.world_size,
        "req": request.index,
        "warmup": request.warmup,
        "e2e_s": e2e,
        "stages": stages,
        "dit": dit,
        "_step_ms": step_ms,
    }
    if request.device is not None:
        record["device"] = str(request.device)
        record["peak_alloc_bytes"] = torch.cuda.max_memory_allocated(request.device)
        record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(request.device)
    state.history.append(record)

    if state.should_log():
        logger.info(_format_record(state, record))
        if state.level >= 2:
            for name in sorted(stages):
                detail = ", ".join(f"{ms:.1f}" for ms in request.stages[name].elapsed_ms())
                logger.info(f"{state.tag}   {name} steps(ms): {detail}")
    if state.json_path:
        _append_json(state, record)


def _dit_throughput(pipe, request: _Request, stages: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
    """Achieved TFLOPS and MFU from an analytic cost of a transformer forward.

    The cost is `2 * params * tokens` for the linear layers plus a separate quadratic term for core attention, read
    from the model's projection widths by [`_attn_widths`] and dropped when those cannot be read -- in which case the
    figure is the linear-only lower bound this reported for every model before, understating a long video sequence by
    roughly two thirds. There is no MFU / HFU split here: inference has no backward and nothing to recompute, so the
    model and hardware costs coincide.

    The rate is a *job* rate, not a per-device one. Multi-GPU inference here splits the sequence across ranks with
    ulysses / ring attention, so the length one rank is handed covers the whole job while that rank computes only
    its shard of it -- which is also why the MFU below is taken against the aggregate peak of every device in the
    group. Comparing a single-device peak against a rate that eight devices produced would report an impossible
    figure well above 100%.
    """
    params_by_stage = getattr(pipe, "_videox_perf_dit_params", {}) or {}
    attn_by_stage = getattr(pipe, "_videox_perf_dit_attn", {}) or {}
    total_flops = 0.0
    attn_flops = 0.0
    total_ms = 0.0
    total_count = 0
    tokens_seen: List[int] = []
    batches_seen: List[int] = []
    for name, params in params_by_stage.items():
        stat = request.stages.get(name)
        summary = stages.get(name)
        if stat is None or summary is None or not stat.tokens:
            continue
        widths, text_tokens = attn_by_stage.get(name, (None, 0))
        per_forward = 2.0 * params * stat.tokens + _attn_flops(widths, stat.tokens, text_tokens)
        total_flops += per_forward * stat.batch * summary["count"]
        attn_flops += _attn_flops(widths, stat.tokens, text_tokens) * stat.batch * summary["count"]
        total_ms += summary["total_ms"]
        total_count += summary["count"]
        tokens_seen.append(stat.tokens)
        batches_seen.append(stat.batch)
    if not total_count or total_ms <= 0:
        return None
    achieved = total_flops / (total_ms / 1000.0) / 1e12
    devices = _STATE.world_size
    per_device_peak = _STATE.peak_tflops(request.device)
    peak = per_device_peak * devices if per_device_peak else None
    return {
        "params": sum(params_by_stage.values()),
        "tokens": max(tokens_seen),
        "batch": max(batches_seen),
        "devices": devices,
        "flops_per_fwd": total_flops / total_count,
        "attn_share": (attn_flops / total_flops) if total_flops else 0.0,
        "tflops": achieved,
        "peak_tflops": peak,
        "mfu": (achieved / peak) if peak else None,
    }


def _format_record(state: _MetricsState, record: Dict[str, Any]) -> str:
    parts = [f"req#{record['req']}{' warmup' if record['warmup'] else ''}", f"e2e {record['e2e_s']:.2f}s"]
    for name in sorted(record["stages"]):
        summary = record["stages"][name]
        seconds = summary["total_ms"] / 1000.0
        if summary["count"] == 1:
            parts.append(f"{name} {seconds:.2f}s(1x)")
        else:
            parts.append(
                f"{name} {seconds:.2f}s({summary['count']}x, {summary['mean_ms']:.0f}+-{summary['std_ms']:.0f}ms, "
                f"min {summary['min_ms']:.0f} max {summary['max_ms']:.0f})"
            )
    if "peak_alloc_bytes" in record:
        gib = 1024.0 ** 3
        parts.append(
            f"peak_alloc {record['peak_alloc_bytes'] / gib:.1f}GiB "
            f"peak_reserved {record['peak_reserved_bytes'] / gib:.1f}GiB"
        )
    dit = record.get("dit")
    if dit:
        segment = (
            f"DiT {dit['flops_per_fwd']:.2e} FLOPs/fwd (attn {dit['attn_share'] * 100:.0f}%) "
            f"-> {dit['tflops']:.1f} TFLOPS"
        )
        if dit["mfu"] is not None:
            over = f" over {dit['devices']} GPUs" if dit["devices"] > 1 else ""
            segment += f" (MFU {dit['mfu'] * 100:.1f}% @{dit['peak_tflops']:.0f}{over})"
        else:
            segment += " (MFU n/a)"
        parts.append(segment)
    return f"{state.tag} " + " | ".join(parts)


def _append_json(state: _MetricsState, record: Dict[str, Any]) -> None:
    payload = {key: value for key, value in record.items() if not key.startswith("_")}
    directory = os.path.dirname(state.json_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(state.json_path, "a") as handle:
        handle.write(json.dumps(payload) + "\n")


def _wrap_call(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        request = _begin_request(self)
        try:
            return fn(self, *args, **kwargs)
        finally:
            # `finally` so a failed request still releases the current-request slot; leaving it set would silently
            # attribute the next request's forwards to a dead record.
            _end_request(self, request)

    wrapper._videox_perf = True
    return wrapper


def _instrument_class(cls) -> bool:
    """Wrap the `__call__` a pipeline class defines itself. Returns whether anything was wrapped."""
    call = cls.__dict__.get("__call__")
    if call is None or getattr(call, "_videox_perf", False):
        # No own `__call__` means it inherits one, which its owner has already had wrapped -- wrapping here too
        # would time the same request twice.
        return False
    cls.__call__ = _wrap_call(call)
    return True


def install(level: Optional[int] = None) -> bool:
    """Wrap every `videox_fun` pipeline class so that requests are measured. No-op unless `VIDEOX_PERF` is set.

    Called at the end of `videox_fun.pipeline.__init__`, by which point every pipeline class is present in that
    module's namespace.
    """
    global _INSTALLED, _STATE
    if _INSTALLED:
        return True
    if level is None:
        level = _env_int("VIDEOX_PERF", 0)
    if level <= 0:
        return False

    # Marked installed before the import below, which can re-enter this function: called directly rather than from
    # the tail of `videox_fun.pipeline.__init__`, the import runs that module for the first time and its tail calls
    # `install` again. Without the flag set, that inner call would build a second state with a second atexit
    # summary and then have this one overwrite it, splitting the history across two objects.
    _INSTALLED = True
    # Adopted rather than replaced: in a training job both halves install, `videox_fun.__init__` running
    # `install_training` first and the entry script's `import videox_fun.pipeline` reaching here second -- and all
    # 110 training scripts do import it. Overwriting would leave `_TrainState` holding a state that is no longer the
    # module's, with two atexit summaries registered over it. The level is raised to the more verbose of the two
    # requests instead of being dropped, so that an explicit `install(level=2)` over a level-1 state still gets the
    # per-step detail its own log line promises.
    if _STATE is None:
        _STATE = _MetricsState(level)
    else:
        _STATE.level = max(_STATE.level, level)

    from diffusers import DiffusionPipeline

    from .. import pipeline as pipeline_package

    seen = set()
    wrapped = 0
    for obj in list(vars(pipeline_package).values()):
        # Dedupe on the class object, not its name: the package ends with aliases (`WanFunPipeline = WanPipeline`)
        # that would otherwise get the same class wrapped twice.
        if not isinstance(obj, type) or id(obj) in seen:
            continue
        seen.add(id(obj))
        if not issubclass(obj, DiffusionPipeline):
            continue  # the namespace also holds schedulers, models and helpers
        if _instrument_class(obj):
            wrapped += 1

    if _STATE.should_log():
        logger.info(f"{_STATE.tag} inference metrics enabled (level {level}) on {wrapped} pipeline classes")
    return True


def instrument_pipeline(pipe, level: int = 1):
    """Measure one pipeline instance regardless of `VIDEOX_PERF`, for programmatic use.

    Wraps the class that actually owns the `__call__` being run, then attaches the component hooks eagerly, so
    this should be called after any FSDP / offload wrapping has been applied.
    """
    global _STATE
    if _STATE is None:
        _STATE = _MetricsState(level)
    for cls in type(pipe).__mro__:
        if "__call__" in cls.__dict__:
            _instrument_class(cls)
            break
    _attach_hooks(pipe)
    return pipe


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# A global step is partitioned into these, in the order they occur. They are non-overlapping by construction and
# `other` is the residual of the step's wall clock against the sum of the rest, so the parts always add up to the
# whole: an unmeasured cost shows up as a fat `other` rather than disappearing.
_TRAIN_PHASES = (
    "data",
    "prep",
    "vae_enc",
    "vae_dec",
    "text_encoder",
    "aux",
    "fwd",
    "bwd",
    "clip",
    "opt",
    "other",
)

# How many later steps may close before an unsettled one is read off the host clock instead of its CUDA events.
# Zero extra lag would suffice for the loops in this repo, which sync on `gather(loss).item()` every micro-step;
# the margin is for a loop that does not.
_SETTLE_LAG = 2

_TRAIN_INSTALLED = False
_TRAIN: Optional["_TrainState"] = None


class _StepAccum:
    """Phase timings of one global step, accumulated over however many micro-steps it spans."""

    __slots__ = ("index", "t0", "total_s", "stages", "micro_steps", "tokens", "samples", "fwd_calls")

    def __init__(self, index: int, t0: float):
        self.index = index
        # The close of the *previous* step, not the top of this one's first micro-step, so that the dataloader wait
        # and the loop's own bookkeeping fall inside a step instead of between two of them.
        self.t0 = t0
        self.total_s = 0.0
        self.stages: Dict[str, _StageStat] = {}
        self.micro_steps = 0
        self.tokens: Optional[int] = None
        self.samples = 0
        self.fwd_calls = 0

    def add(self, name: str, start, end) -> None:
        stat = self.stages.get(name)
        if stat is None:
            stat = _StageStat()
            self.stages[name] = stat
        stat.pairs.append((start, end))

    def ready(self) -> bool:
        return all(stat.ready() for stat in self.stages.values())


class _TrainState:
    """Everything the training path keeps between steps: phase accumulation, windowing, and the model's size."""

    def __init__(self, state: _MetricsState):
        self.state = state
        self.every = max(1, _env_int("VIDEOX_PERF_EVERY", 50))
        self.total_steps = _env_int("VIDEOX_PERF_TOTAL_STEPS", 0) or None
        self.coef_override = _env_float("VIDEOX_PERF_FLOPS_COEF")

        self.num_steps = 0
        self.step: Optional[_StepAccum] = None
        self.pending: Any = collections.deque()
        self.window: List[Dict[str, Any]] = []
        self.measured_step_s: List[float] = []
        self.wrapped_classes = 0

        # Re-entrancy and attribution flags. Plain attributes rather than the thread-local state the inference
        # path needs: a training loop runs on one thread, and the dataloader's parallelism is in worker
        # *processes*, which hold their own copy of this module and never reach any of this.
        self.depth = 0
        self.in_backward = False
        self.step_owner: Optional[int] = None

        self.head_open = False
        self.micro_enter = 0.0
        self.prev_exit: Optional[float] = None

        self.accelerator = None
        self.dit_module: Optional[torch.nn.Module] = None
        self.dit_params = 0
        self.dit_trainable = 0
        self.attn_widths: Optional[Dict[str, int]] = None
        self.text_tokens = 0
        self._dit_resolved = False
        self._ckpt = False
        self._device: Optional[torch.device] = None
        self._device_seen = False
        self._finished = False
        atexit.register(self.finish)

    # -- device ------------------------------------------------------------

    @property
    def device(self) -> Optional[torch.device]:
        """The CUDA device this rank trains on, or `None` to stay off the CUDA APIs entirely.

        `accelerator.device` is both authoritative and, unlike the inference path (see [`_perf_device`]), backed by
        a real `torch.cuda.set_device`: accelerate's `PartialState` points the process at its own local index when
        it comes up. The device is still named explicitly on every call below rather than left to default, so that
        a script which never built an `Accelerator` cannot quietly aim this at device 0.
        """
        if self._device_seen:
            return self._device
        if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
            return None  # not resolved, not cached: CUDA may still come up later in the run
        device = getattr(self.accelerator, "device", None)
        if getattr(device, "type", None) == "cuda":
            self._device = device
            self._device_seen = True
        return self._device

    # -- step assembly -----------------------------------------------------

    def current(self, t0: Optional[float] = None) -> _StepAccum:
        if self.step is None:
            self.step = _StepAccum(self.num_steps, t0 if t0 is not None else time.perf_counter())
        return self.step

    def close_head(self, now: float) -> None:
        """End the `prep` window of the current micro-step, at the first measured work to start inside it.

        `prep` is thus literally what runs before any model does -- the noise and timestep sampling, the latent
        bookkeeping. It is taken off the host clock: the tensors it creates are small, and an event pair here would
        instead measure when the *previous* phase's queued kernels drained.
        """
        if self.head_open:
            self.head_open = False
            self.current().add("prep", (self.micro_enter, None), (now, None))

    def enter_micro(self) -> None:
        now = time.perf_counter()
        step = self.current(now)
        if self.prev_exit is not None:
            # From leaving the previous `accumulate` block to entering this one: the host blocked on the
            # dataloader, plus the loop bookkeeping around it. Host clock by definition -- nothing is submitted to
            # the device in that gap. It measures the main process's *wait*, not the work inside the workers,
            # which multiprocessing puts out of reach from here; a near-zero `data` still proves the input
            # pipeline is keeping up.
            step.add("data", (self.prev_exit, None), (now, None))
        step.micro_steps += 1
        self.head_open = True
        self.micro_enter = now

    def exit_micro(self) -> None:
        self.head_open = False
        self.prev_exit = time.perf_counter()

    def note_optimizer_step(self, optimizer, t_end: float) -> None:
        """Close the global step, if this call was the real one.

        The entry scripts call `optimizer.step()` on every micro-step and leave the decision to accelerate: the
        body of `AcceleratedOptimizer.step` is guarded by `sync_gradients`, so on an accumulation micro-step it is
        a no-op. Reading that same flag is what separates a genuine boundary from a pass-through, and it is still
        valid here -- `accumulate` sets it on entry and the loops themselves read it again after stepping.

        A script with several optimizers (the distillation and preference-tuning ones) steps more than one per
        iteration. The first instance to close a step owns the boundary from then on, so the count follows one of
        them instead of counting the same iteration several times over; the others' time still lands in `opt`.
        """
        try:
            sync = bool(optimizer.gradient_state.sync_gradients)
        except AttributeError:
            sync = True  # not an accelerate-managed optimizer, so every call is a step
        if not sync:
            return
        key = id(optimizer)
        if self.step_owner is None:
            self.step_owner = key
        if self.step_owner == key:
            self.close_step(t_end)

    def close_step(self, t_close: float) -> None:
        step = self.step
        if step is None:
            return
        step.total_s = t_close - step.t0
        self.num_steps += 1
        self.step = _StepAccum(self.num_steps, t_close)
        self.head_open = False
        self.pending.append(step)
        self._drain()

    # -- settling ----------------------------------------------------------

    def _drain(self, force: bool = False) -> None:
        """Settle the steps whose CUDA events have retired, and only those.

        This is where the promise of adding no synchronize is kept. A step's closing events are recorded a few
        microseconds before it closes and cannot be read yet; waiting on them would be exactly the stall this path
        exists to avoid, and reading them unretired raises. So a finished step waits in a queue until a later step
        proves the device has moved past it -- which the loops here do on their own, syncing on
        `gather(loss).item()` once per micro-step. A step that is still unready after `_SETTLE_LAG` more have
        closed is read off the host clock instead, which loses the device-side precision for that step but never
        blocks and never drops it.
        """
        while self.pending:
            step = self.pending[0]
            if not (force or step.ready() or self.num_steps - step.index > _SETTLE_LAG):
                break
            self.pending.popleft()
            self._settle(step)

    def _settle(self, step: _StepAccum) -> None:
        phases: Dict[str, float] = {}
        for name, stat in step.stages.items():
            values = stat.elapsed_ms()
            if values:
                phases[name] = sum(values) / 1000.0
        residual = step.total_s - sum(phases.values())
        phases["other"] = max(0.0, residual)
        record = {
            "step": step.index,
            "warmup": step.index < self.state.warmup,
            "total_s": step.total_s,
            "micro_steps": step.micro_steps,
            "phases": phases,
            # The measured phases can outrun the step when device work from one phase drains inside the next, since
            # each pair is resolved to whichever of its host and device spans is longer. Reported rather than
            # folded away, because a large one means the breakdown below should not be read too closely.
            "overrun_s": max(0.0, -residual),
            "tokens": step.tokens,
            "samples": step.samples,
            "fwd_calls": step.fwd_calls,
        }
        self.window.append(record)
        if not record["warmup"]:
            self.measured_step_s.append(step.total_s)
        if len(self.window) >= self.every:
            self.emit_window()

    # -- the model being trained -------------------------------------------

    def note_models(self, objects) -> None:
        """Size the trained model from what `prepare` handed back, not from the module the script built.

        Under FSDP the wrapper owns the flat sharded parameters and the inner module's own are emptied, so counting
        through the class the forward wrapper sees would report zero. The return value of `prepare` is the wrapper
        itself, which is the one place the shard and its `requires_grad` can both be read.

        The largest module wins, which is the DiT in every script here: the vae and the text encoder are not passed
        to `prepare` at all, and where a second network is (a discriminator, a fake-score model) it is the smaller.
        """
        for obj in objects:
            if not isinstance(obj, torch.nn.Module):
                continue
            total = self.state.dit_params_override or _count_params(obj, self.state.world_size)
            if total > self.dit_params:
                self.dit_params = total
                self.dit_module = obj
                self._dit_resolved = False

    def _resolve_dit(self) -> None:
        """Read the trainable fraction and the checkpointing flag, once, at the first window rather than at
        `prepare`.

        Deferred because a script is free to freeze weights or call `enable_gradient_checkpointing` after preparing,
        and by the first window -- fifty steps in by default -- whatever it was going to do it has done.
        """
        module = self.dit_module
        if self._dit_resolved or module is None:
            return
        self._dit_resolved = True
        world = self.state.world_size
        scale = world if world > 1 and _is_sharded(module) else 1
        self.dit_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad) * scale
        self._ckpt = any(getattr(sub, "gradient_checkpointing", False) for sub in module.modules())
        if self.state.attn_flops:
            # Walked once, here, for the same reason the trainable fraction is: it is a walk of every submodule of a
            # 14B transformer and nothing about it changes from one step to the next.
            widths = _attn_widths(module)
            self.attn_widths = widths if widths["self"] else None
            self.text_tokens = _text_tokens(module) if self.attn_widths else 0

    @property
    def dit_ckpt(self) -> bool:
        self._resolve_dit()
        return self._ckpt

    def flops_coef(self) -> Tuple[float, float, str]:
        """The two multipliers on a forward's cost that a whole step comes to, and why.

        A forward costs one pass over the model. A full-parameter backward costs two more -- one for the input
        gradients, one for the weight gradients -- so a full step is 3x the forward, which is the flat 3 Megatron
        applies as its `forward_backward_expansion_factor`. Freezing the base weights, as LoRA does, drops the
        weight-gradient pass over them and brings the backward to about 1x, for 2x total.

        Gradient checkpointing adds one more forward, recomputed during the backward. It is returned as a *second*
        coefficient rather than folded into the first, because the two answer different questions and the industry
        gave them different names. MFU, as PaLM defined it, is the work the model needs against the hardware peak,
        and it deliberately excludes recomputation: a run that recomputes has not become more useful for it. HFU is
        the work the hardware actually issued, recomputation included. Reporting one figure under the name of the
        other is what this did: every run here trains with checkpointing on, so every `mfu` it ever printed was an
        HFU, a quarter high. Megatron sidesteps the distinction by never counting recomputation at all -- its
        `num_floating_point_operations` has no term for it -- and by reporting TFLOP/s rather than a utilization.

        The trainable fraction decides between 3x and 2x, with 0.5 as the split: every LoRA configuration here trains
        well under a percent of the weights and every full-parameter one trains all of them, so nothing real lands
        near the threshold. `VIDEOX_PERF_FLOPS_COEF` overrides both coefficients at once, which collapses `mfu` and
        `hfu` onto each other by construction; it is what to reach for when a run mixes the two -- or when FSDP has
        flattened frozen and trainable weights into one parameter, where the fraction cannot be read apart.

        The fraction is `requires_grad`, not the set of weights the optimizer updates, and those come apart: the
        multiviews scripts pass `--trainable_modules view` yet flip `requires_grad_(True)` over the whole stack under
        FSDP, so that every wrapped unit gets a post-backward reshard. Those weight gradients really are computed
        before being discarded, so `requires_grad` is the fraction the FLOPs follow -- reading `trainable 100%` next
        to a narrow `--trainable_modules` is that, not a contradiction.
        """
        self._resolve_dit()
        ckpt = self._ckpt
        full = not self.dit_params or self.dit_trainable >= 0.5 * self.dit_params
        coef = 3.0 if full else 2.0
        hw_coef = coef + (1.0 if ckpt else 0.0)
        if self.attn_widths:
            attn = f"attn from dims ({self.attn_widths['modules']} modules)"
        elif self.state.attn_flops:
            attn = "attn omitted (widths unreadable)"
        else:
            attn = "attn omitted (disabled)"
        reason = f"{'full' if full else 'lora'}, ckpt {'on' if ckpt else 'off'} -> {coef:g}x/{hw_coef:g}x, {attn}"
        if self.coef_override is not None:
            override = self.coef_override
            return override, override, f"{reason}, overridden to {override:g}x"
        return coef, hw_coef, reason

    # -- windowing ---------------------------------------------------------

    def emit_window(self) -> None:
        window = self.window
        self.window = []
        if not window:
            return
        record = self._aggregate(window)
        if self.state.should_log():
            logger.info(_format_train_window(self.state, record))
            if self.state.level >= 2:
                for entry in window:
                    detail = " ".join(f"{name} {value:.3f}" for name, value in entry["phases"].items())
                    logger.info(f"{self.state.tag}   step {entry['step']} {entry['total_s']:.3f}s: {detail}")
        if self.state.json_path:
            _append_json(self.state, record)
        device = self.device
        if device is not None:
            # Scope the next window's peaks to the next window, the way the inference path scopes them to a request.
            torch.cuda.reset_peak_memory_stats(device)

    def _aggregate(self, window: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Warmup steps are logged but kept out of the statistics. If a whole window is warmup there is nothing to
        # fall back on but the window itself, which is better than printing nothing at all.
        measured = [entry for entry in window if not entry["warmup"]] or window
        totals = sorted(entry["total_s"] for entry in measured)
        phases = {}
        for name in _TRAIN_PHASES:
            values = [entry["phases"].get(name, 0.0) for entry in measured]
            if any(values):
                phases[name] = statistics.fmean(values)
        record: Dict[str, Any] = {
            "kind": "train_window",
            "ts": time.time(),
            "rank": self.state.rank,
            "world_size": self.state.world_size,
            "step_first": window[0]["step"],
            "step_last": window[-1]["step"],
            "steps": len(measured),
            "warmup_excluded": len(window) - len(measured),
            "step_s_p50": _percentile(totals, 50),
            "step_s_p95": _percentile(totals, 95),
            "phases_s": phases,
            "micro_steps": statistics.fmean([entry["micro_steps"] for entry in measured]),
            "overrun_s": statistics.fmean([entry["overrun_s"] for entry in measured]),
        }
        device = self.device
        if device is not None:
            record["device"] = str(device)
            record["peak_alloc_bytes"] = torch.cuda.max_memory_allocated(device)
            record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        record["throughput"] = self._throughput(measured)
        if self.total_steps:
            remaining = max(0, self.total_steps - window[-1]["step"] - 1)
            record["eta_s"] = remaining * record["step_s_p50"]
        return record

    def _throughput(self, measured: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Samples per second, achieved TFLOPS and MFU for the window.

        Every rate here is one sum over another -- the window's own work over the window's own wall clock -- and
        never an aggregate of one quantity divided by an aggregate of the other. The two agree while the steps in a
        window are alike and diverge badly when they are not, and under `--enable_bucket` they are not: the sampler
        mixes image steps with 81-frame video steps, so one window holds token counts that differ several-fold and
        step times with them. Taking the last step's tokens over the window's p50 step, as this did, reported 181.8
        TFLOPS for a window whose honest rate was a third of that -- the tokens having come from a video step and
        the p50 from an image one. The p50 the window reports is a description of the step times, not a divisor.

        A step that captured no token count contributes neither work nor time, rather than contributing time alone,
        which would pull the rate down by whatever share of the window it occupied.

        `flops_per_step` and `tokens` are window means, reported to say what the window contained. Dividing the
        first by the logged p50 will not reproduce `tflops` and is not meant to: a mean of products is not the
        product of the means, and the p50 spans steps that the FLOPs figure excludes. The two clocks the rates did
        divide by are reported as `wall_s` and `priced_s`, so that either rate can be checked against the record it
        came from. They differ only by the steps that carried no token count.

        The cost model is `2 * params * tokens` for the linear layers plus a separate quadratic term for core
        attention, read from the model's projection widths by [`_attn_widths`]. The split is Megatron's: it multiplies
        its per-token terms by the token count and its `self_attn_core_term` by the sum of the squared lengths,
        because one grows linearly in the sequence and the other quadratically. Folding attention into a parameter
        count, as this used to, understates a 28k-token video step by about two thirds and an 8k-token one by about a
        third -- which is why the old figure could not be used to compare two runs at different lengths, whatever its
        docstring claimed. When the widths cannot be read the attention term is dropped and `flops_coef_reason` says
        so; the figure is then the old lower bound.

        The linear half is priced from parameters, and that is loose in the opposite direction: it charges every
        weight against every latent token, while the cross-attention key and value projections run on the few hundred
        text tokens and the timestep and text embeddings on fewer still. On Wan that is worth about a fifth of the
        linear half, which the attention term now dwarfs.

        Two utilizations are reported and they are not interchangeable. `mfu` counts the forward and backward the
        model needs; `hfu` also counts the forward that gradient checkpointing recomputes. `tflops` pairs with the
        first and `hw_tflops` with the second, so that either rate divided by `peak_tflops` reproduces its own
        utilization. See [`flops_coef`] for why the two are kept apart.

        Two widths matter and they are not the same. `samples` is what this rank alone consumed, and it is reported
        as such: under sequence parallel one sample is spread over several ranks, so summing it across the world
        would claim several times the samples that were actually trained on. The FLOPs rate is instead reported for
        the whole job -- the local cost times the data-parallel width -- because that is the figure the aggregate
        peak below is comparable against. Under plain FSDP or DDP, where every rank holds its own samples, the two
        multiplications cancel and the MFU is exactly the per-device one.
        """
        wall_s = sum(entry["total_s"] for entry in measured)
        if wall_s <= 0:
            return None
        samples = statistics.fmean([entry["samples"] for entry in measured])
        dp = _dp_degree(self.state)
        result: Dict[str, Any] = {
            "samples_per_s": sum(entry["samples"] for entry in measured) / wall_s,
            "samples": samples,
            "wall_s": wall_s,
            "dp": dp,
        }
        priced = [entry for entry in measured if entry["tokens"] and entry["samples"]]
        priced_s = sum(entry["total_s"] for entry in priced)
        if not (priced and priced_s > 0 and self.dit_params):
            return result
        coef, hw_coef, reason = self.flops_coef()
        linear = 0.0
        attention = 0.0
        for entry in priced:
            tokens = entry["tokens"]
            linear += 2.0 * self.dit_params * tokens * entry["samples"]
            attention += _attn_flops(self.attn_widths, tokens, self.text_tokens) * entry["samples"]
        per_forward = linear + attention
        flops_total = coef * per_forward
        hw_flops_total = hw_coef * per_forward
        achieved = flops_total * dp / priced_s / 1e12
        hw_achieved = hw_flops_total * dp / priced_s / 1e12
        per_device_peak = self.state.peak_tflops(self.device)
        peak = per_device_peak * self.state.world_size if per_device_peak else None
        result.update(
            {
                "params": self.dit_params,
                "trainable": self.dit_trainable,
                "tokens": statistics.fmean([entry["tokens"] for entry in priced]),
                "priced_steps": len(priced),
                "priced_s": priced_s,
                "flops_coef": coef,
                "flops_coef_hw": hw_coef,
                "flops_coef_reason": reason,
                "flops_per_step": flops_total / len(priced),
                # What share of the figure is the quadratic term, so that a reader can see how much of it rests on
                # the width introspection rather than on the parameter count.
                "attn_share": (attention / per_forward) if per_forward else 0.0,
                "devices": self.state.world_size,
                "tflops": achieved,
                "hw_tflops": hw_achieved,
                "peak_tflops": peak,
                "mfu": (achieved / peak) if peak else None,
                "hfu": (hw_achieved / peak) if peak else None,
            }
        )
        return result

    def finish(self) -> None:
        """Flush at exit: force the queued steps out, emit the partial window, then one summary line.

        Guarded so that flushing explicitly does not then get flushed again by the `atexit` hook, which would print
        the summary twice over the same steps.
        """
        if self._finished:
            return
        self._finished = True
        try:
            self._drain(force=True)
            self.emit_window()
            if self.measured_step_s and self.state.should_log():
                totals = sorted(self.measured_step_s)
                logger.info(
                    f"{self.state.tag} === {len(totals)} steps | "
                    f"p50 {_percentile(totals, 50):.2f}s/step p95 {_percentile(totals, 95):.2f}s | "
                    f"{sum(totals) / 3600.0:.2f}h in-loop ==="
                )
        except Exception as error:  # an exit handler must not turn a finished run into a failed one
            logger.warning(f"[Perf] failed to report the training summary: {error!r}")


def _dp_degree(state: _MetricsState) -> int:
    """How many ranks hold *different* samples on a step.

    The world splits along two axes at once. Sequence parallel shares one sample across a group of ranks, each
    computing a slice of its tokens; data parallel gives each group its own samples. Only the latter multiplies the
    samples a step trains on, so it is the only one the throughput may be scaled by.

    xfuser is asked rather than assumed, and only if it is already imported: reaching into `sys.modules` avoids
    importing it in a run that does not use it, and the query is guarded because the accessor raises before the
    parallel state is initialized.
    """
    sp = 1
    module = sys.modules.get("xfuser.core.distributed.parallel_state")
    if module is not None:
        try:
            sp = max(1, int(module.get_sequence_parallel_world_size()))
        except Exception:
            sp = 1
    return max(1, state.world_size // sp)


def _model_stage(cls) -> str:
    """Which phase a model class's forward belongs to.

    Classified from the class rather than from a variable name in a script, because the training scripts are what
    this must not touch. The order matters: an autoencoder is one before it is anything else, and the vision towers
    are pulled out ahead of the text-encoder test they would otherwise pass.
    """
    from diffusers.models.modeling_utils import ModelMixin

    name = cls.__name__
    if "Autoencoder" in name or "AutoEncoder" in name or "VAE" in name:
        return "vae"
    if "Vision" in name:
        return "aux"
    if "T5Encoder" in name or "TextEncoder" in name or "CLIPTextModel" in name:
        return "text_encoder"
    if not issubclass(cls, ModelMixin):
        # What is left having reached here is a `transformers.PreTrainedModel`, which across this repo means an LLM
        # or T5 tower standing in for a text encoder.
        return "text_encoder"
    if "Transformer" in name or "UNet" in name or "Unet" in name or "LatentUpsampler" in name:
        return "fwd"
    # Audio encoders, vocoders, projection bridges and connectors: real cost, but none of the phases above.
    return "aux"


def _wrap_model_method(fn, stage: str, capture_tokens: bool = False):
    """Time one model method into `stage`, but only when it is the outermost such call of a training step.

    The nesting guard is what keeps the phases a true partition. Wrapped classes do contain each other -- an audio
    encoder inside a DiT, a decoder inside an autoencoder -- and timing both would count the inner one twice, once
    on its own and once inside its caller, so the phases would sum past the step they came from. Only the outermost
    is measured and the inner ones fold into it.

    Timing is likewise suppressed inside `Accelerator.backward`. Gradient checkpointing recomputes forwards during
    the backward pass; those are a real cost of the backward and belong in `bwd`, which is why `fwd` and `bwd` come
    out near 1:2 with checkpointing on rather than the 1:2 of the arithmetic being a coincidence.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        train = _TRAIN
        if train is None or train.depth or train.in_backward:
            return fn(self, *args, **kwargs)
        start = _mark(train.device)
        train.close_head(start[0])
        shape = _infer_tokens(self, args, kwargs) if capture_tokens else None
        train.depth += 1
        try:
            return fn(self, *args, **kwargs)
        finally:
            train.depth -= 1
            step = train.current()
            step.add(stage, start, _mark(train.device))
            if shape is not None:
                if step.tokens is None:
                    step.tokens = shape[0]
                step.samples += shape[1]
                step.fwd_calls += 1

    wrapper._videox_perf = True
    return wrapper


def _wrap_model_attr(cls, name: str, stage: str, capture_tokens: bool = False) -> bool:
    """Wrap `cls.name` unless it is already wrapped, resolving it through the class's bases.

    Resolving through the bases is what dedupes an inheritance chain for free: a subclass that does not define its
    own `forward` finds the parent's, which carries the marker if the parent has been done, and is skipped. Should
    the subclass be reached first instead, the guard inside [`_wrap_model_method`] keeps the resulting double
    wrapper from double-counting.
    """
    fn = getattr(cls, name, None)
    if not callable(fn) or getattr(fn, "_videox_perf", False):
        return False
    setattr(cls, name, _wrap_model_method(fn, stage, capture_tokens))
    return True


def _instrument_model_classes() -> Tuple[int, int]:
    """Wrap the model classes exported by `videox_fun.models`. Returns `(wrapped, skipped)`.

    Done at the *class* level, and at import time, because training has no pipeline object to walk: the entry
    scripts build their models themselves and hand them to `accelerate`, so there is no single place an instance
    can be caught. Wrapping the class before any instance exists also means it survives everything applied
    afterwards -- FSDP, DDP, peft -- since all of those end up calling the original class's forward.

    Membership is `ModelMixin` or `transformers.PreTrainedModel`, which is what separates a whole model from a
    building block: it is what keeps `WanSelfAttention` and `WanRMSNorm` out, and wrapping either of those would
    have the counters tick once per layer per step. The same test also leaves out four classes that are whole models
    but subclass neither -- `AutoencoderKLWan_`, `AutoencoderKLWan2_2_`, `MOVAModel` and `Wav2Vec2ModelWrapper` --
    and none of the four is a gap, each being held by an instrumented class that does the calling: the two inner
    vaes as the `self.model` of `AutoencoderKLWan` and `AutoencoderKLWan3_8`, and `Wav2Vec2ModelWrapper` as the
    `self.audio_encoder` of `LongCatVideoAudioEncoder`, so their time already lands in the holder's stage.
    `MOVAModel` defines no `forward` at all, only a `__call__` routing to four submodels that are themselves
    wrapped.
    """
    from diffusers.models.modeling_utils import ModelMixin
    from transformers import PreTrainedModel

    from .. import models as models_package

    seen = set()
    wrapped = 0
    skipped = 0
    for obj in list(vars(models_package).values()):
        # Dedupe on the class object rather than the name: the package exports aliases of the same class, and
        # wrapping one twice would append two marker pairs per call.
        if not isinstance(obj, type) or id(obj) in seen:
            continue
        seen.add(id(obj))
        if not issubclass(obj, torch.nn.Module):
            continue
        if not (issubclass(obj, ModelMixin) or issubclass(obj, PreTrainedModel)):
            skipped += 1
            continue
        stage = _model_stage(obj)
        if stage == "vae":
            # `forward` never runs on these: callers use `encode` and `decode`, and wrapping the two separately is
            # also what splits the two directions apart in the log. The streaming pair folds into those same two
            # stages, being the same work done in chunks, and neither of them calls the plain method, so nothing is
            # counted twice.
            done = _wrap_model_attr(obj, "encode", "vae_enc")
            done |= _wrap_model_attr(obj, "decode", "vae_dec")
            done |= _wrap_model_attr(obj, "encode_stream", "vae_enc")
            done |= _wrap_model_attr(obj, "decode_stream", "vae_dec")
        else:
            done = _wrap_model_attr(obj, "forward", stage, capture_tokens=(stage == "fwd"))
        wrapped += bool(done)
    return wrapped, skipped


def _patch_method(cls, name: str, factory) -> int:
    """Replace a method a class defines itself, once. Returns whether it did."""
    fn = cls.__dict__.get(name)
    if fn is None or getattr(fn, "_videox_perf", False):
        return 0
    setattr(cls, name, factory(fn))
    return 1


def _wrap_accumulate(fn):
    @functools.wraps(fn)
    @contextlib.contextmanager
    def accumulate(self, *models):
        train = _TRAIN
        if train is None:
            with fn(self, *models) as value:
                yield value
            return
        train.accelerator = self
        train.enter_micro()
        try:
            # Re-entered as a context manager, not called as a plain function. `Accelerator.accumulate` is a
            # `@contextmanager` generator, so `fn(self, *models)` hands back a context manager that has not yet run
            # a line of its body; driving it with `with` is what keeps `no_sync` wrapped around the micro-step, and
            # yielding its value through keeps an exception raised inside the block propagating as it did before.
            with fn(self, *models) as value:
                yield value
        finally:
            train.exit_micro()

    accumulate._videox_perf = True
    return accumulate


def _wrap_backward(fn):
    @functools.wraps(fn)
    def backward(self, *args, **kwargs):
        train = _TRAIN
        if train is None:
            return fn(self, *args, **kwargs)
        train.accelerator = self
        device = train.device
        start = _mark(device)
        train.close_head(start[0])
        train.in_backward = True
        try:
            return fn(self, *args, **kwargs)
        finally:
            train.in_backward = False
            train.current().add("bwd", start, _mark(device))

    backward._videox_perf = True
    return backward


def _wrap_clip(fn):
    @functools.wraps(fn)
    def clip_grad_norm_(self, *args, **kwargs):
        train = _TRAIN
        if train is None:
            return fn(self, *args, **kwargs)
        device = train.device
        start = _mark(device)
        try:
            return fn(self, *args, **kwargs)
        finally:
            # Worth its own phase rather than being left in `other`: under FSDP the global norm needs an all-reduce
            # across the shards, so this is where that collective becomes visible.
            train.current().add("clip", start, _mark(device))

    clip_grad_norm_._videox_perf = True
    return clip_grad_norm_


def _wrap_optimizer_step(fn):
    @functools.wraps(fn)
    def step(self, *args, **kwargs):
        train = _TRAIN
        if train is None:
            return fn(self, *args, **kwargs)
        device = train.device
        start = _mark(device)
        try:
            return fn(self, *args, **kwargs)
        finally:
            end = _mark(device)
            train.current().add("opt", start, end)
            train.note_optimizer_step(self, end[0])

    step._videox_perf = True
    return step


def _wrap_prepare(fn):
    @functools.wraps(fn)
    def prepare(self, *args, **kwargs):
        result = fn(self, *args, **kwargs)
        train = _TRAIN
        if train is not None:
            train.accelerator = self
            try:
                train.note_models(result if isinstance(result, tuple) else (result,))
            except Exception as error:  # never let measurement take down a training run
                logger.warning(f"[Perf] failed to size the prepared model: {error!r}")
        return result

    prepare._videox_perf = True
    return prepare


def _patch_accelerate() -> int:
    """Hang the step model off `accelerate`. Returns the number of methods patched.

    `accelerate` is the seam because it is the one thing all 113 training scripts in this repo share: every one of
    them builds an `Accelerator` and calls `prepare`, and all but a handful use `accumulate`, `backward` and
    `clip_grad_norm_`.

    The step boundary is `AcceleratedOptimizer.step` and deliberately not `torch.optim.AdamW.step`. Three optimizers
    are in use across these scripts -- torch's `AdamW`, bitsandbytes' `AdamW8bit` and `CAME` -- and accelerate's
    wrapper is the one point all three pass through. It also steps around a trap: `AdamW.step` overrides
    `Optimizer.step`, so patching the base class would silently miss it.
    """
    from accelerate import Accelerator
    from accelerate.optimizer import AcceleratedOptimizer

    patched = _patch_method(Accelerator, "accumulate", _wrap_accumulate)
    patched += _patch_method(Accelerator, "backward", _wrap_backward)
    patched += _patch_method(Accelerator, "clip_grad_norm_", _wrap_clip)
    patched += _patch_method(Accelerator, "prepare", _wrap_prepare)
    patched += _patch_method(AcceleratedOptimizer, "step", _wrap_optimizer_step)
    try:
        from accelerate.utils import DeepSpeedOptimizerWrapper
    except ImportError:  # pragma: no cover - depends on the accelerate build
        pass
    else:
        # It overrides `step` with a no-op, deepspeed having done the stepping inside `backward`, so the base class
        # patch above never runs for it and it needs its own to mark the boundary.
        patched += _patch_method(DeepSpeedOptimizerWrapper, "step", _wrap_optimizer_step)
    return patched


def _format_train_window(state: _MetricsState, record: Dict[str, Any]) -> str:
    span = f"step {record['step_first']}-{record['step_last']}"
    if record["warmup_excluded"]:
        span += f" ({record['warmup_excluded']} warmup excluded)"
    parts = [
        span,
        f"{record['step_s_p50']:.2f}s/step p95 {record['step_s_p95']:.2f}s",
        " ".join(f"{name} {seconds:.2f}s" for name, seconds in record["phases_s"].items()),
    ]
    if record["micro_steps"] > 1.0:
        parts.append(f"{record['micro_steps']:.0f} micro-steps")
    if record["overrun_s"] > 0.02 * max(record["step_s_p50"], 1e-9):
        # The phases came to more than the step. Said out loud rather than hidden, because past a couple of percent
        # it means the phase boundaries are blurred by device work draining across them.
        parts.append(f"overrun {record['overrun_s']:.2f}s")

    throughput = record.get("throughput")
    if throughput:
        parts.append(f"{throughput['samples_per_s']:.3f} samples/s (local, dp={throughput['dp']})")
    if "peak_alloc_bytes" in record:
        gib = 1024.0 ** 3
        parts.append(
            f"peak_alloc {record['peak_alloc_bytes'] / gib:.1f}GiB "
            f"peak_reserved {record['peak_reserved_bytes'] / gib:.1f}GiB"
        )
    if throughput and "params" in throughput:
        trainable = throughput["trainable"] / throughput["params"] * 100.0 if throughput["params"] else 0.0
        segment = (
            f"DiT {throughput['params'] / 1e9:.1f}B params "
            f"(trainable {trainable:.3g}%, {throughput['flops_coef_reason']}, "
            f"attn {throughput['attn_share'] * 100:.0f}%) "
            f"{throughput['flops_per_step']:.2e} FLOPs/step -> {throughput['tflops']:.1f} TFLOPS"
        )
        if throughput["mfu"] is not None:
            over = f" over {throughput['devices']} GPUs" if throughput["devices"] > 1 else ""
            # MFU first because it is the figure that compares across runs, HFU beside it because with gradient
            # checkpointing on the hardware really did issue that much and the gap between the two is the
            # recomputation. They coincide when checkpointing is off.
            segment += (
                f" (MFU {throughput['mfu'] * 100:.1f}% / HFU {throughput['hfu'] * 100:.1f}%"
                f" @{throughput['peak_tflops']:.0f}{over})"
            )
        else:
            segment += " (MFU n/a)"
        if throughput["priced_steps"] < record["steps"]:
            # Both the FLOPs and the rate cover only the steps that reported a token count. Said out loud when that is
            # fewer than the window held, so the figure is not read as a rate over the whole window.
            segment += f" [{throughput['priced_steps']}/{record['steps']} steps priced]"
        parts.append(segment)
    if "eta_s" in record:
        parts.append(f"ETA {record['eta_s'] / 3600.0:.1f}h")
    return f"{state.tag} " + " | ".join(parts)


def install_training(level: Optional[int] = None) -> bool:
    """Measure the training loop by wrapping `accelerate` and the `videox_fun.models` classes.

    No-op unless `VIDEOX_PERF` is set. Called at the end of `videox_fun.__init__`, which is early enough to wrap the
    model classes before any instance of one exists and covers every training script here -- the three that never
    import `videox_fun.pipeline` still import `videox_fun.models`.

    A global step is timed from the close of the previous optimizer step to the close of this one, and split into
    the phases in [`_TRAIN_PHASES`]. Nothing in this path calls `torch.cuda.synchronize`: the phase timings are CUDA
    events, which are recorded and then left alone until a later step has shown the device to be past them (see
    [`_TrainState._drain`]). Wall clock alone would not do here, because these loops synchronize on
    `gather(loss).item()` *before* `backward`, so the host returns from the backward long before the device is done
    with it and a host-only reading would push that work into the following step.
    """
    global _TRAIN_INSTALLED, _TRAIN, _STATE
    if _TRAIN_INSTALLED:
        return True
    if level is None:
        level = _env_int("VIDEOX_PERF", 0)
    if level <= 0:
        return False

    _TRAIN_INSTALLED = True
    if _STATE is None:
        _STATE = _MetricsState(level)
    train = _TrainState(_STATE)

    try:
        patched = _patch_accelerate()
        wrapped, skipped = _instrument_model_classes()
    except Exception as error:
        # A run that cannot be measured is still a run. Leave `_TRAIN` unset so the wrappers that did land, if any,
        # stay inert rather than half-reporting.
        logger.warning(f"[Perf] training metrics disabled: {error!r}")
        return False

    # Published last: every wrapper above reads this global and does nothing while it is `None`, so nothing is
    # measured until the whole set is in place and no half-installed state can produce a partial step.
    _TRAIN = train
    train.wrapped_classes = wrapped
    if _STATE.should_log():
        logger.info(
            f"{_STATE.tag} training metrics enabled (level {level}) on {patched} accelerate methods and "
            f"{wrapped} model classes ({skipped} non-model classes skipped), "
            f"reporting every {train.every} steps"
        )
    return True

