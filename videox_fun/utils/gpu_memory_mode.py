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

from .fp4_optimization import (convert_float4_weight_dtype_wrapper,
                               convert_model_weight_to_float4)
from .fp8_optimization import (convert_model_weight_to_float8,
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
)

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
    hook that brings them back to `weight_dtype` on the fly. The wrapper detects FSDP
    itself (non-mutating output-side dequant), which is why installing it on the already
    sharded ``pipeline.transformer`` is the supported order.
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
                          quant_tag=None, strict=False):
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
            ``_and_qfloat8`` spelling. ``None`` (default) uses the tag parsed from `mode`.
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
    elif base == "model_full_load":
        pipeline.to(device=device)
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
