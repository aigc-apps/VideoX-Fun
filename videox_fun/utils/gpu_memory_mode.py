"""Unified dispatch for ``GPU_memory_mode`` (memory placement) and quantization.

Historically every inference/training script carried its own if/elif chain over the
``GPU_memory_mode`` string, which encoded *two orthogonal* concerns in one flat enum:
where the weights live (full load / cpu offload / group offload / sequential offload)
and at which precision the transformer runs (``_and_qfloat8``). Adding a quantization
method therefore meant adding a branch to the Cartesian product in ~200 copy-pasted
scripts, and the ordering knowledge (quantize before the offload hooks are installed,
never after ``torch.compile``) had to be re-encoded in each of them.

This module collapses that into a grammar plus a registry:

    ``<base>`` or ``<base>_and_<quant>``

with ``<base>`` in :data:`GPU_MEMORY_MODES` and ``<quant>`` in :data:`QUANT_MODES`.
The mode strings used across the repository keep their exact spelling, so existing
scripts, configs and UI dropdowns need no change; a new quantization method is one
:func:`register_quant_mode` decorated function in this file (or in its own module),
and every script picks it up by writing ``model_cpu_offload_and_<quant>``.
"""

import logging

import torch

from .fp4_optimization import (convert_float4_weight_dtype_wrapper,
                               convert_model_weight_to_float4)
from .fp8_optimization import (FLOAT8_DTYPE, convert_model_weight_to_float8,
                               convert_weight_dtype_wrapper,
                               replace_parameters_by_name)
from .group_offload import register_auto_device_hook, safe_enable_group_offload

logger = logging.getLogger("videox_fun")

# Memory placement modes: what happens to the pipeline after quantization is applied.
GPU_MEMORY_MODES = (
    "model_full_load",
    "model_cpu_offload",
    "model_group_offload",
    "sequential_cpu_offload",
    "text_encoder_offload",
)

# Attribute prefix of the conditioner components (`text_encoder`, plus `text_encoder_2` /
# `text_encoder_3` on the multi-encoder families).
TEXT_ENCODER_PREFIX = "text_encoder"

# Separator between the memory part and the quantization part of a mode string.
QUANT_SEPARATOR = "_and_"

# tag -> applier, filled by `register_quant_mode`.
QUANT_MODES = {}


def register_quant_mode(tag, saves_memory=True):
    """Register a quantization applier under `tag`.

    The applier is called as ``applier(model, device=..., weight_dtype=...,
    exclude_module_name=...)`` and must mutate `model` in place (returning it is
    optional). `saves_memory` records whether the method actually shrinks the stored
    weights: this distinction matters, because an FP8 *compute* backend (weights kept
    in bf16, only the GEMM runs in FP8) is a speed mode and must not be advertised as
    a memory saving, unlike a store-FP8 quantization.

    To add a method: write the applier, decorate it here, and use
    ``<base>_and_<tag>`` in any script -- no script needs editing.
    """
    def _decorator(applier):
        if tag in QUANT_MODES:
            raise ValueError(f"Quantization mode '{tag}' is already registered")
        applier.saves_memory = saves_memory
        QUANT_MODES[tag] = applier
        return applier
    return _decorator


@register_quant_mode("qfloat8")
def _quant_qfloat8(model, device=None, weight_dtype=None, exclude_module_name=("modulation",)):
    """Store the transformer weights in FP8 (scale-aware) and dequantize per forward.

    Two calls belong together and must stay in this order: the conversion rewrites the
    parameters into FP8 storage, and `convert_weight_dtype_wrapper` installs the forward
    hook that brings them back to `weight_dtype` on the fly. Only the wrapper half is
    safe on an already-sharded model -- the conversion rewrites `param.data`, which after
    `shard_model` is a flat-storage view. So this applier runs BEFORE sharding; on a
    sharded target `apply_gpu_memory_mode` downgrades to `qfloat8_preconverted` (wrapper
    only), and this conversion is never reached.
    """
    if weight_dtype is None:
        raise ValueError(
            "The qfloat8 mode needs `weight_dtype` (the dtype the FP8 weights are "
            "dequantized back to); pass the script's weight_dtype to "
            "apply_gpu_memory_mode().")
    convert_model_weight_to_float8(
        model, exclude_module_name=list(exclude_module_name), device=device)
    convert_weight_dtype_wrapper(model, weight_dtype)
    return model


@register_quant_mode("qfloat8_preconverted")
def _quant_qfloat8_preconverted(model, device=None, weight_dtype=None, exclude_module_name=("modulation",)):
    """Install only the dequantization wrapper on an already FP8-converted model.

    Some checkpoints store a mixed bag of dtypes and must run
    :func:`convert_model_weight_to_float8` *before* FSDP sharding, so the per-row scales
    are computed on the unsharded weights (the MiniMax-H3 / Taomate-H3 streaming
    pipelines). Those scripts call this applier through ``quant_tag`` after they have
    done the conversion themselves: re-running it here would either recompute the scales
    from sharded params or convert the deliberately-skipped float32 modules, so this mode
    installs only the forward dequant hook -- the second half of ``qfloat8``. The wrapper
    detects FSDP itself, so installing it on the already sharded ``pipeline.transformer``
    is the supported order. The user-facing ``GPU_memory_mode`` string keeps the
    ``_and_qfloat8`` spelling; only the internal tag differs.
    """
    if weight_dtype is None:
        raise ValueError(
            "The qfloat8_preconverted mode needs `weight_dtype` (the dtype the FP8 "
            "weights are dequantized back to); pass the script's weight_dtype to "
            "apply_gpu_memory_mode().")
    convert_weight_dtype_wrapper(model, weight_dtype)
    return model


@register_quant_mode("qfloat4")
def _quant_qfloat4(model, device=None, weight_dtype=None, exclude_module_name=("modulation",)):
    """Store the transformer's Linear weights in NVFP4 (E2M1, two-level scaled) per forward.

    Same shape as the FP8 mode: the packing rewrites ``weight`` into uint8 nibbles in
    NVFP4 (E2M1 elements, one E4M3 scale per 16-element block, plus a per-tensor fp32
    global scale), and the wrapper rebuilds the bf16 weight only for the duration of the
    matmul, so the compute dtype never changes and the stored bits cannot drift. Roughly
    3.6x smaller than bf16 and 1.8x smaller than ``qfloat8``, at a visible accuracy cost
    -- E2M1 has 16 levels, so this is the "fit it on the card" mode rather than the "run
    it faster" one. NVFP4 has no Hopper GEMM, so this stays weight-only storage.

    Only ``nn.Linear`` weights are packed (biases and norm scales stay as they are), and
    FSDP-sharded models are rejected by the packer instead of being silently skipped.
    """
    if weight_dtype is None:
        raise ValueError(
            "The qfloat4 mode needs `weight_dtype` (the dtype the FP4 weights are "
            "dequantized back to); pass the script's weight_dtype to "
            "apply_gpu_memory_mode().")
    convert_model_weight_to_float4(
        model, exclude_module_name=list(exclude_module_name), device=device)
    convert_float4_weight_dtype_wrapper(model, weight_dtype)
    return model


def split_gpu_memory_mode(mode):
    """Split ``<base>_and_<quant>`` into ``(base, quant_tag)``.

    Splits on the *last* separator so that a future base name containing ``_and_``
    cannot be mistaken for a quantization tag. Returns ``(mode, None)`` when there is
    no quantization part.
    """
    if not isinstance(mode, str):
        raise TypeError(f"GPU_memory_mode must be a string, got {type(mode)}")
    base, separator, tag = mode.rpartition(QUANT_SEPARATOR)
    if not separator:
        return mode, None
    return base, tag


def _locate_decoder_layers(model):
    """The conditioner's decoder-layer stack, or None when the layout is not recognised.

    Delegates to the canonical `fsdp.locate_decoder_layers`, shared with the FSDP2 sharding strategy so
    both the pinned-streaming path here and `shard_model_fsdp2` agree on which stack the encode walks.
    """
    from ..dist.fsdp import locate_decoder_layers

    return locate_decoder_layers(model)


def _stream_decoder_layers_pinned(model, layers, device):
    """Stream `layers` one-way off a pinned host mirror; keep the rest of `model` resident.

    Returns the number of streamed layers. Every layer's mirror is built first, so a `pin_memory`
    failure under a restricted `ulimit -l` raises before any parameter is moved or hooked and the
    caller's pageable fallback stays clean. Each layer is then repointed at its mirror (releasing
    the pageable original) and starts host-resident; the forward hooks load a layer to the card
    before it runs and drop it back afterwards, so the peak is one layer. The pieces outside the
    stack (embedding, final norm, vision tower -- ~2.6 GiB) move to the card once and stay, the
    replicated part the FSDP2 1-wide offload keeps on the device too.
    """
    from ..dist.fsdp import build_pinned_mirror, pinned_load, pinned_unload

    mirrors = [build_pinned_mirror(layer) for layer in layers]

    layer_module_ids = {id(module) for layer in layers for module in layer.modules()}
    for module in model.modules():
        if id(module) in layer_module_ids:
            continue
        for param in module.parameters(recurse=False):
            param.data = param.data.to(device)
        for buf in module.buffers(recurse=False):
            buf.data = buf.data.to(device)

    for layer, host in zip(layers, mirrors):
        for name, param in layer.named_parameters():
            param.data = host["p:" + name]
        for name, buf in layer.named_buffers():
            buf.data = host["b:" + name]

    def _onload(module, args):
        pinned_load(module, device)
        return None  # a non-None return would replace the module inputs

    def _offload(module, args, output):
        pinned_unload(module)
        return None  # a non-None return would replace the module output

    for layer in layers:
        layer.register_forward_pre_hook(_onload)
        layer.register_forward_hook(_offload)
    return len(layers)


def _stream_module_parameters(model, device, pinned=False):
    """Onload a module's own parameters around its forward and offload them again after it.

    A whole-component onload cannot work for the big conditioners: MiniMax-H3's is 63 GiB
    (Qwen3-VL, 31.5B parameters in bf16), so even beside a float8 transformer (33 GiB) it is
    96 GiB on an 80 GiB card. Moving every parameter-owning module on its own keeps the peak
    at the largest block (~0.8 GiB); the total transferred volume is the same, which is what
    makes this a placement for a component a request touches once.

    Hooking each module's `forward` also means no input tensor goes through accelerate's
    `send_to_device`, which trips over the meta-device tensors a `low_cpu_mem_usage`
    conditioner is built from.

    With `pinned=True` and a recognisable decoder stack the layers instead stream one-way off a
    pinned host mirror (`_stream_decoder_layers_pinned`): a request then pays no device-to-host
    leg, only the asynchronous host-to-card copy. An unrecognised layout, or a `pin_memory`
    failure, keeps the pageable per-module path below.
    """
    if pinned:
        layers = _locate_decoder_layers(model)
        if layers is not None:
            try:
                return _stream_decoder_layers_pinned(model, layers, device)
            except (RuntimeError, OSError) as exc:
                print("[text_encoder_offload] pinned layer mirror unavailable (%s); falling back "
                      "to pageable per-module streaming." % exc)

    def _onload(module, args):
        module.to(device)
        return None  # a non-None return would replace the module inputs

    def _offload(module, args, output):
        module.to("cpu")
        return None  # a non-None return would replace the module output

    streamed = 0
    for module in model.modules():
        if not any(True for _ in module.parameters(recurse=False)) and not any(
                True for _ in module.buffers(recurse=False)):
            continue
        module.register_forward_pre_hook(_onload)
        module.register_forward_hook(_offload)
        streamed += 1
    return streamed


def _place_without_text_encoder(pipeline, device, offload_names=None, offload_pinned=True):
    """Keep the transformer on `device`, stream the conditioners, and offload the one-shot VAE.

    `model_cpu_offload` offloads all components, the transformer included -- and the transformer
    is the one that runs once per denoising step, so its transfer dominates the request. A
    request touches the conditioner exactly once, before the denoising loop, so leaving only it
    in CPU RAM turns the largest component into a single one-shot transfer.

    The video VAE is also touched once per request (a single decode), and at 9.8 GiB fp32 it does
    not fit beside the transformer through a long decode, so it is offloaded by default through
    `offload_components_cpu` (`offload_names=None` -> ``("vae",)``; pass ``()`` to keep it
    resident). `offload_pinned` routes it through the pinned one-way mirror, dropping the
    device-to-host leg a two-way offload pays per request.

    FSDP-wrapped components are skipped: `shard_model` has already placed their flat-param
    shards, and a `.to()` would ask every rank for the unsharded parameters. A sharded VAE is
    therefore left to FSDP rather than offloaded here.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from ..dist.fsdp import offload_components_cpu

    offload_set = ("vae",) if offload_names is None else tuple(offload_names)

    conditioners = []
    deferred_offload = []
    for name, component in pipeline.components.items():
        if not isinstance(component, torch.nn.Module):
            continue
        if name.startswith(TEXT_ENCODER_PREFIX):
            conditioners.append((name, component))
            continue
        if isinstance(component, FSDP):
            continue
        if name in offload_set:
            deferred_offload.append(name)
            continue
        component.to(device)

    if not conditioners:
        raise ValueError(
            "GPU_memory_mode 'text_encoder_offload' needs a '%s' component to offload, and "
            "%s has none. Use 'model_full_load' or 'model_cpu_offload' for this pipeline."
            % (TEXT_ENCODER_PREFIX, type(pipeline).__name__))

    for name, component in conditioners:
        if _is_fsdp_wrapped(component):
            # Its shards live in FSDP's hands, not in the module's own parameters, so the per-module hooks
            # below would find nothing to move. When FSDP was wrapped with CPUOffload it already does what
            # this mode wants -- the shards stay in CPU RAM and stream to the card around each encode -- so
            # the component is left to FSDP instead of being hooked again here.
            if _fsdp_cpu_offload_enabled(component):
                print(
                    "GPU_memory_mode 'text_encoder_offload': '%s' is FSDP-sharded with CPUOffload, so its "
                    "shards already live in CPU RAM and stream to %s for the encode pass; left to FSDP."
                    % (name, device))
                continue
            # CPUOffload is fixed at wrap time and cannot be turned on afterwards, so the only fixes are
            # upstream: wrap the conditioner with offload, or leave it un-sharded for this mode to stream.
            raise ValueError(
                "GPU_memory_mode 'text_encoder_offload' cannot offload '%s': it is FSDP-sharded but was "
                "wrapped WITHOUT CPU offload, so its shards are GPU-resident and only FSDP can move them. "
                "FSDP's CPUOffload is set at wrap time and cannot be enabled afterwards -- wrap this "
                "component with `shard_model(..., offload_to_cpu=True)` (the minimax_h3 scripts derive it "
                "from GPU_memory_mode), or set the script's `fsdp_text_encoder = False` so this mode can "
                "stream the un-sharded conditioner." % name)
        component.to("cpu")
        streamed = _stream_module_parameters(component, device, pinned=offload_pinned)
        print(
            "GPU_memory_mode 'text_encoder_offload': '%s' kept in CPU RAM, %d streamed "
            "units (pinned=%s) loaded to %s for the single encode pass."
            % (name, streamed, offload_pinned, device))

    # Offload the one-shot VAE only after the conditioner guards above, so a raise there leaves no
    # half-attached hooks. This also pins `pipeline._execution_device` to the onload GPU.
    if deferred_offload:
        offload_components_cpu(pipeline, device, deferred_offload, pinned=offload_pinned)
        print("GPU_memory_mode 'text_encoder_offload': CPU-offloaded %s (pinned=%s); the "
              "transformer stays resident on %s." % (deferred_offload, offload_pinned, device))


def _is_fsdp_wrapped(component):
    """True when `component`, or the `.model` it exposes, is managed by FSDP.

    FSDP1 marks the sharded module with a wrapper class; FSDP2 (`fully_shard`) instead
    turns its parameters into DTensors, so both forms have to be checked. Looking one
    `.model` down covers the conditioners, which are sharded on their inner
    `Qwen3VLModel` because `encode_prompt` calls that submodule directly.

    The FSDP1 test scans the whole subtree and matches `FlatParameter`, not just the
    candidate's own class: `shard_model` returns a root wrapper today, so the candidate
    *is* the `FullyShardedDataParallel`, but a per-block wrap that left the root
    un-wrapped would only show FSDP on the children. A false negative here keeps the
    `qfloat8` conversion's `param.data` rewrite, which lands on a flat-storage view and
    corrupts it. This mirrors the sharding test in the TurboX converter's dispatch, so
    the downgrade guard in `apply_gpu_memory_mode` and the wrapper it downgrades to
    always agree on whether a model is sharded.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # torch without the DTensor package can only be FSDP1
        DTensor = None

    for candidate in (component, getattr(component, "model", None)):
        if not isinstance(candidate, torch.nn.Module):
            continue
        # `.modules()` includes the candidate itself, so a root wrapper is caught here
        # too, along with per-block wrappers deeper down.
        if any(isinstance(sub, FSDP) for sub in candidate.modules()):
            return True
        for param in candidate.parameters():
            # `FlatParameter` lives in a private module, so match on the type name.
            if type(param).__name__ == "FlatParameter":
                return True
            if DTensor is not None and isinstance(param, DTensor):
                return True
    return False


def _fsdp_cpu_offload_enabled(component):
    """True when the FSDP1 wrapper managing `component` was built with `CPUOffload(offload_params=True)`.

    `shard_model(..., offload_to_cpu=True)` keeps the flat-param shards in CPU RAM and streams them to the
    card around each wrapped forward, which is exactly the conditioner placement `text_encoder_offload` wants;
    when it is on, that component is left to FSDP rather than hooked again here. The wrapper may sit one
    `.model` down (the conditioner is sharded on the inner model `encode_prompt` calls), so both the component
    and its subtree are scanned. FSDP fixes CPUOffload at construction, so this only reports the setting the
    wrap already chose -- it cannot turn it on.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    for candidate in (component, getattr(component, "model", None)):
        if not isinstance(candidate, torch.nn.Module):
            continue
        for sub in candidate.modules():
            if isinstance(sub, FSDP):
                if getattr(getattr(sub, "cpu_offload", None), "offload_params", False):
                    return True
    return False


def _place_full_load(pipeline, device, sharded_offload, sharded_offload_pinned=True):
    """`model_full_load` across FSDP shards: move what is movable, leave the shards alone.

    `pipeline.to(device)` asks every rank for its unsharded parameters, because an FSDP
    flat param is not a module parameter a `.to()` can move -- the transient is the whole
    model and it OOMs. So when a component is FSDP-wrapped it is skipped: "fully loaded"
    for a shard means "already on the card", which is where `shard_model` put it. The
    remaining components are placed one by one, which is what `pipeline.to()` does
    anyway; the only difference is that a sharded pipeline is not asked for shards it
    does not hold.

    Returns without touching anything when nothing is sharded, keeping the historical
    whole-pipeline `.to(device)` for every non-FSDP caller.
    """
    from ..dist.fsdp import offload_components_cpu

    sharded = [name for name, component in pipeline.components.items()
               if isinstance(component, torch.nn.Module) and _is_fsdp_wrapped(component)]
    if not sharded:
        pipeline.to(device=device)
        return

    # The video VAE fits beside the shards at short clip lengths but not at long ones
    # (9.8 GiB fp32, plus the decode peak, on top of ~58 GiB of shards and activations).
    # The failure only surfaces on a second request, in decode, so it is offloaded by
    # default *when the pipeline is sharded*; `sharded_offload=()` turns that off.
    offload_names = ("vae",) if sharded_offload is None else tuple(sharded_offload)
    offloaded = [name for name in offload_names
                 if name not in sharded
                 and isinstance(getattr(pipeline, name, None), torch.nn.Module)]

    resident = []
    for name, component in pipeline.components.items():
        if not isinstance(component, torch.nn.Module):
            continue
        if name in sharded or name in offloaded:
            continue
        component.to(device)
        resident.append(name)

    if offloaded:
        offload_components_cpu(pipeline, device, offloaded, pinned=sharded_offload_pinned)

    print("GPU_memory_mode 'model_full_load': %s already on %s as FSDP shards and left "
          "there; resident %s; CPU-offloaded %s."
          % (sharded, device, resident, offloaded or "nothing"))


def _target_transformers(pipeline, transformers):
    """Modules the quantization / sequential-offload special cases apply to.

    Derived from the pipeline so MoE setups (a second high/low-noise transformer) are
    covered without every script repeating the ``transformer_2`` mirror block. Always
    read off the pipeline rather than the script's local variable: after FSDP sharding
    ``pipeline.transformer`` is the wrapped module, and the quantization helpers have to
    see the wrapper to detect the sharding.
    """
    if transformers is not None:
        return [model for model in transformers if model is not None]
    models = []
    for attr in ("transformer", "transformer_2"):
        model = getattr(pipeline, attr, None)
        if model is not None and not any(model is seen for seen in models):
            models.append(model)
    return models


def apply_gpu_memory_mode(pipeline, mode, device, weight_dtype=None, *,
                          transformers=None, exclude_module_name=("modulation",),
                          quant_tag=None, sharded_offload=None, sharded_offload_pinned=True,
                          strict=False):
    """Quantize (if the mode asks for it) and place `pipeline` on `device`.

    Replaces the per-script if/elif chain. Order is fixed here and is part of the
    contract: quantization runs first, then the memory placement, because the dtype
    wrapper captures the module ``forward`` it wraps and must not be installed on top of
    the offload hooks (and neither may run after ``torch.compile``).

    Args:
        pipeline: the diffusers pipeline; ``pipeline.transformer`` (and
            ``pipeline.transformer_2`` when present) are the quantization targets.
        mode: ``GPU_memory_mode`` string, ``<base>`` or ``<base>_and_<quant>``.
        device: target device for offload hooks and parameter replacement.
        weight_dtype: compute dtype handed to the quantization applier.
        transformers: explicit list of modules to quantize; only needed for pipelines
            whose transformer is not exposed as ``transformer`` / ``transformer_2``.
        exclude_module_name: sub-module name fragments the quantizer must leave at full
            precision (``modulation`` holds the small per-block time embeddings, whose
            FP8 storage costs more than it saves).
        quant_tag: override the ``<quant>`` tag parsed from `mode`. Scripts whose
            checkpoint is already quantized before this call (e.g. converted ahead of
            FSDP sharding) pass the wrapper-only tag ``'qfloat8_preconverted'`` here so
            the conversion is not repeated, while the user-facing `mode` keeps its
            ``_and_qfloat8`` spelling. ``None`` (default) uses the tag parsed from `mode`,
            except that a ``qfloat8`` target which is already FSDP-sharded is switched to
            ``qfloat8_preconverted`` automatically -- its conversion rewrites
            ``param.data``, which after sharding is a flat-storage view.
        sharded_offload: components to keep off the card. For ``model_full_load`` this applies
            only when the pipeline has FSDP-sharded parts (otherwise the mode keeps its
            whole-pipeline ``.to(device)`` behaviour); for ``text_encoder_offload`` it applies
            unconditionally, since that mode already leaves the transformer resident and streams
            the conditioner. ``None`` (default) offloads the video VAE, the one component too
            large (9.8 GiB fp32 on MiniMax-H3) to share the card through a long decode; pass
            ``()`` to keep everything resident, or a name tuple to choose.
        sharded_offload_pinned: offload the ``sharded_offload`` components through a pinned
            host mirror (one-way: the offload repoints ``.data`` at the pinned copy instead
            of a device-to-host ``.to("cpu")``), which drops the per-request device-to-host
            leg for a component decoded once such as the VAE. True by default; each component
            falls back to the two-way offload when pinning is unavailable (a restricted
            ``ulimit -l``). Used by ``model_full_load`` only when something is sharded, and by
            ``text_encoder_offload`` for its VAE offload.
        strict: raise on an unknown *base* mode instead of degrading. It is False by
            default, so an unfamiliar mode keeps the behaviour of the old if/elif chains,
            whose ``else`` branch plain-loaded the pipeline: a warning plus
            ``pipeline.to(device)``. Pass True in scripts that would rather fail than
            silently run without offloading. Note that this flag never weakens the
            quantization side: a ``<base>_and_<quant>`` mode whose ``<quant>`` is not
            registered always raises, because falling back there would mean generating
            with full-precision weights, which is exactly the memory spike the caller
            was trying to avoid.
    """
    base, tag = split_gpu_memory_mode(mode)
    if quant_tag is not None:
        tag = quant_tag
    if tag == "qfloat8":
        # Conversion must precede sharding: it rewrites `param.data` per row to compute
        # the scale, and after `shard_model` that attribute is a flat-storage view. Only
        # the wrapper half may run on a sharded model, so fall back to it and say so
        # instead of corrupting the flat param mid-run.
        sharded_targets = [model for model in _target_transformers(pipeline, transformers)
                           if _is_fsdp_wrapped(model)]
        if sharded_targets:
            tag = "qfloat8_preconverted"
            has_float8 = any(param.dtype == FLOAT8_DTYPE
                             for model in sharded_targets for param in model.parameters())
            if has_float8:
                logger.info("GPU_memory_mode '%s': transformer is already FSDP-sharded, "
                            "so only the dequantization wrapper is installed "
                            "(%s).", mode, tag)
            else:
                # "No float8 storage" is not the same as "no effect": a converter
                # patched in at runtime may decline to write FP8 into a model FSDP
                # already owns and quantize each all-gathered weight inside the
                # forward instead, which is a real FP8 GEMM but saves nothing. Only
                # the memory claim is unconditionally false here, so only say that.
                logger.warning(
                    "GPU_memory_mode '%s': transformer is FSDP-sharded but holds no "
                    "float8 weights, so the qfloat8 suffix saves no memory. The wrapper "
                    "installed now may still quantize each all-gathered weight inside "
                    "the forward -- a speed path, not a memory one, and only if the "
                    "live converter does that. Run convert_model_weight_to_float8 on "
                    "CPU BEFORE shard_model to actually save the memory.", mode)

    if tag is not None:
        applier = QUANT_MODES.get(tag)
        if applier is None:
            # Messages below use %-formatting across implicit string concatenation on
            # purpose: Python <= 3.11 has a legacy f-string tokenizer that rejects some
            # multi-line concatenated f-strings (quote + field combinations) which are
            # valid on 3.12+, and this module has to import on 3.10.
            raise NotImplementedError(
                "GPU_memory_mode '%s' asks for the quantization mode '%s', which is "
                "not registered. Registered modes: %s. Add it with "
                "@register_quant_mode('%s') in videox_fun/utils/gpu_memory_mode.py, "
                "which needs no change in the scripts."
                % (mode, tag, sorted(QUANT_MODES), tag))
        if base == "model_group_offload":
            raise ValueError(
                "GPU_memory_mode '%s' is not supported: the group offload hooks own "
                "the parameter storage of each layer group, so a per-forward "
                "dequantization wrapper on the same modules has no stable view to "
                "rewrite. Use '%s' with no quantization, or 'model_cpu_offload_and_%s'."
                % (mode, base, tag))
        for model in _target_transformers(pipeline, transformers):
            applier(model, device=device, weight_dtype=weight_dtype,
                    exclude_module_name=exclude_module_name)

    if base == "sequential_cpu_offload":
        # Per-layer offload moves every module to CPU, so the tensors that must stay on
        # the device (the modulation parameters and the RoPE frequency table) are pinned
        # explicitly before the hooks are installed.
        for model in _target_transformers(pipeline, transformers):
            replace_parameters_by_name(model, list(exclude_module_name), device=device)
            freqs = getattr(model, "freqs", None)
            if freqs is not None:
                model.freqs = freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif base == "model_group_offload":
        for model in _target_transformers(pipeline, transformers):
            register_auto_device_hook(model)
        safe_enable_group_offload(
            pipeline, onload_device=device, offload_device="cpu",
            offload_type="leaf_level", use_stream=True)
    elif base == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif base == "text_encoder_offload":
        _place_without_text_encoder(pipeline, device, sharded_offload, sharded_offload_pinned)
    elif base == "model_full_load":
        _place_full_load(pipeline, device, sharded_offload, sharded_offload_pinned)
    elif strict:
        raise ValueError(
            "Unknown GPU_memory_mode '%s'. Expected one of %s, optionally suffixed by "
            "'%s<quant>' with quant in %s."
            % (mode, GPU_MEMORY_MODES, QUANT_SEPARATOR, sorted(QUANT_MODES)))
    else:
        # Default path for an unrecognised base, mirroring the old if/elif chains whose
        # `else` branch silently full-loaded. Loud enough to be grepped in a log: the
        # consequence (no offloading) shows up later as an OOM with an unrelated stack.
        logger.warning(
            "Unknown GPU_memory_mode base '%s' (from '%s'); falling back to "
            "pipeline.to(device) with NO offloading and NO quantization. Expected one "
            "of %s, optionally suffixed by '%s<quant>' with quant in %s. Pass "
            "strict=True to turn this into an error."
            % (base, mode, GPU_MEMORY_MODES, QUANT_SEPARATOR, sorted(QUANT_MODES)))
        pipeline.to(device=device)
