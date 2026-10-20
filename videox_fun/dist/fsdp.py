# Copied from https://github.com/Wan-Video/Wan2.1/blob/main/wan/distributed/fsdp.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import (lambda_auto_wrap_policy,
                                         transformer_auto_wrap_policy)
from torch.distributed.utils import _free_storage


def find_classes_in_model(model, class_names):
    """
    Recursively find unique module classes in the model that match the given class names.
    
    Args:
        model: The PyTorch model to traverse.
        class_names: A list of class name strings to look for.
        
    Returns:
        A set of matched class types.
    """
    found_classes = set()
    class_names_set = set(class_names)
    
    def traverse(module):
        if module.__class__.__name__ in class_names_set:
            found_classes.add(module.__class__)
        for child in module.children():
            traverse(child)
    
    traverse(model)
    print(f"Found transformer classes: {found_classes}")
    return found_classes


def create_transformer_auto_wrap_policy(
    model,
    transformer_layer_cls_to_wrap,
):
    """
    Creates an auto wrap policy that only wraps modules belonging to the specified 
    transformer layer classes.
    
    Args:
        model: The PyTorch model to analyze for class types.
        transformer_layer_cls_to_wrap: A list of class name strings to wrap.
        
    Returns:
        A callable auto wrap policy function.
    """
    # Dynamically find the actual class types corresponding to the provided names
    transformer_classes = find_classes_in_model(model, transformer_layer_cls_to_wrap)
    
    if not transformer_classes:
        raise ValueError(
            f"No modules found with class names {transformer_layer_cls_to_wrap}. "
            "Please check the class names or the model structure."
        )
    
    def transformer_policy(module, recurse, nonwrapped_numel=None, **kwargs):
        # Use the standard transformer auto wrap policy with the discovered classes
        policy_kwargs = dict(module=module, recurse=recurse, transformer_layer_cls=transformer_classes)
        if nonwrapped_numel is not None:
            policy_kwargs["nonwrapped_numel"] = nonwrapped_numel
        return transformer_auto_wrap_policy(**policy_kwargs)
    
    return transformer_policy


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
    module_to_wrapper=None,
    transformer_layer_cls_to_wrap=None,
    cast_dtype=True,
    ignored_modules=None,
    offload_to_cpu=False,
):  
    """
    Wraps the model with FSDP using the specified configuration.
    
    Args:
        model: The PyTorch model to shard.
        device_id: The CUDA device ID.
        param_dtype: Data type for parameters.
        reduce_dtype: Data type for gradient reduction.
        buffer_dtype: Data type for buffers.
        process_group: The process group for distributed training.
        sharding_strategy: The FSDP sharding strategy.
        sync_module_states: Whether to sync module states across ranks.
        module_to_wrapper: Specific modules to wrap if using lambda policy.
        transformer_layer_cls_to_wrap: List of class names to wrap using transformer policy.
        cast_dtype: Whether to cast the managed parameters to `param_dtype` before wrapping. Set False to
            keep a pre-quantized storage dtype (e.g. float8) — MixedPrecision still computes in `param_dtype`.
        ignored_modules: Modules excluded from FSDP (kept replicated in their own dtype via ignored_states).
            Use for modules a mixed-precision checkpoint pins to float32: casting them into the shard dtype
            makes AdaLN/output-head rounding accumulate coherently over the denoising trajectory (flicker).
        offload_to_cpu: When True, wrap with FSDP's native `CPUOffload(offload_params=True)` so each rank keeps
            its parameter shard in CPU RAM and streams it back to the GPU only around the wrapped module's
            forward. This is the offload path that is compatible with sequence-parallel multi-GPU inference
            (accelerate / diffusers module hooks cannot manage the flat-param shards, only FSDP itself can).
            Only shards the parameter memory, not activations. Default False keeps the previous fully-resident
            behaviour, so existing single-GPU / model_full_load callers are unaffected.
        
    Returns:
        The FSDP-wrapped model.
    """
    if transformer_layer_cls_to_wrap is not None:
        # Create policy based strictly on transformer layer classes
        auto_wrap_policy = create_transformer_auto_wrap_policy(
            model=model,
            transformer_layer_cls_to_wrap=transformer_layer_cls_to_wrap,
        )
    else:
        # Fallback to lambda policy if no transformer classes are specified
        auto_wrap_policy = partial(
            lambda_auto_wrap_policy, 
            lambda_fn=lambda m: m in (model.blocks if module_to_wrapper is None else module_to_wrapper)
        )

    # FSDP flattens each wrap unit's parameters into one flat buffer and requires a uniform dtype inside it.
    # Models that pin a few modules to fp32 for numerical precision (e.g. MiniMax-H3's embedders/output heads)
    # would otherwise fail with "Must flatten tensors with uniform dtype"; MixedPrecision computes in
    # `param_dtype` anyway, so cast the managed params up front. With `cast_dtype=False` the caller
    # intentionally keeps a different storage dtype (e.g. a pre-applied float8 quantization); FSDP keeps one
    # flat buffer per dtype and MixedPrecision casts to `param_dtype` for the compute. The ignored modules
    # keep their own dtype (typically float32) and stay replicated — a blanket model.to() on them would round
    # the AdaLN modulation and accumulate coherently over the sampling trajectory into temporal flicker.
    ignored_modules = list(ignored_modules) if ignored_modules else []
    ignored_param_ids = {id(p) for m in ignored_modules for p in m.parameters()}
    if cast_dtype and param_dtype is not None:
        for p in model.parameters():
            # A pre-applied float8 quantization stays as the storage dtype (MixedPrecision casts it to
            # `param_dtype` for the compute, matching the non-FSDP qfloat8 dequant wrapper numerics); only
            # the remaining dtypes (e.g. fp32 heads) are homogenized into `param_dtype`.
            if (p.dtype != param_dtype and id(p) not in ignored_param_ids
                    and p.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)):
                p.data = p.data.to(param_dtype)

    # With CPU offload, bound the prefetch depth so at most one extra all-gather is in flight; otherwise FSDP
    # would keep the gathered (unsharded) params of upcoming blocks resident and eat the savings.
    cpu_offload = CPUOffload(offload_params=True) if offload_to_cpu else None
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states,
        cpu_offload=cpu_offload,
        limit_all_gathers=offload_to_cpu,
        ignored_states=ignored_modules if ignored_modules else None)

    # `device_id`/`sync_module_states` only manage the sharded params; the ignored modules must be placed on
    # the device explicitly. They are small fp32 heads kept replicated (not sharded), so FSDP's CPUOffload does
    # not cover them: they stay GPU-resident even in offload mode, which is the intended numerical-precision pin.
    for m in ignored_modules:
        m.to(device_id)

    return model


def free_model(model):
    """
    Frees memory associated with the FSDP model.
    
    Args:
        model: The FSDP-wrapped model to free.
    """
    for m in model.modules():
        if isinstance(m, FSDP):
            _free_storage(m._handle.flat_param.data)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def offload_components_cpu(obj, device, component_names):
    r"""
    Attach top-level accelerate CPU-offload hooks to a chosen subset of a pipeline's whole-model components
    (by default the VAE / audio VAE), leaving every other component untouched.

    This is the offload companion to `shard_model(..., offload_to_cpu=True)` for sequence-parallel multi-GPU
    inference. There the transformer and the text encoder are wrapped by FSDP, whose flat-param shards are
    managed only by FSDP itself (enabled to live in CPU RAM at wrap time); accelerate's module hooks cannot
    manage those shards, and `pipeline.enable_model_cpu_offload()` would both `obj.to("cpu")` the FSDP-wrapped
    modules and register a second, conflicting hook on them. So under sequence parallelism the big sharded
    modules offload through FSDP and only the remaining non-FSDP components (VAE / audio VAE) are offloaded here.

    The hooks chain like `enable_model_cpu_offload` does: onloading a component offloads the previous one. The
    pipeline drives them through its `_offload_scope` helper, which fires `pre_forward` / `post_forward` around
    method calls (`encode` / `decode`) that bypass a module's `forward`.

    Also patches the pipeline's `_execution_device` to keep reporting the onload (GPU) device: the FSDP modules
    carry no `_hf_hook`, so the stock property would otherwise fall back to `self.device` (CPU) as soon as one
    of them leads the component order.
    """
    from accelerate import cpu_offload_with_hook

    onload_device = torch.device(device) if isinstance(device, str) else device

    hooks = []
    prev_hook = None
    for name in component_names:
        model = getattr(obj, name, None)
        if not isinstance(model, torch.nn.Module):
            continue
        _, hook = cpu_offload_with_hook(model, onload_device, prev_module_hook=prev_hook)
        hooks.append(hook)
        prev_hook = hook
    obj._cpu_offload_component_hooks = hooks

    # `_execution_device` must return the onload device while these component hooks are attached.
    if not hasattr(obj.__class__, "_execution_device_original"):
        obj.__class__._execution_device_original = obj.__class__._execution_device

    @property
    def _execution_device(self):
        # Return the onload device of any component that carries an accelerate CPU-offload hook; skip the
        # FSDP-wrapped modules (no `_hf_hook`) instead of bailing to `self.device` on them.
        for _, component in self.components.items():
            if isinstance(component, torch.nn.Module):
                hf_hook = getattr(component, "_hf_hook", None)
                if hf_hook is not None and getattr(hf_hook, "execution_device", None) is not None:
                    return torch.device(hf_hook.execution_device)
        # No component hook left: delegate to the saved original.
        return self.__class__._execution_device_original.fget(self)

    obj.__class__._execution_device = _execution_device

    return hooks