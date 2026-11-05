# Copied from https://github.com/Wan-Video/Wan2.1/blob/main/wan/distributed/fsdp.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
from functools import partial

import torch
from accelerate.big_modeling import UserCpuOffloadHook
from accelerate.hooks import ModelHook, add_hook_to_module, send_to_device
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
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

    # `device_id` only moves the managed params/buffers, but `sync_module_states` checks every buffer of the root
    # module, including those of the ignored modules (e.g. a computed rope `inv_freq`). Place the ignored modules
    # on the device before wrapping. They are small fp32 heads kept replicated (not sharded), so FSDP's CPUOffload
    # does not cover them: they stay GPU-resident even in offload mode, which is the intended numerical-precision pin.
    for m in ignored_modules:
        m.to(device_id)

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


def locate_decoder_layers(model):
    """The decoder-layer ModuleList of a conditioner, or None when the layout is not recognised.

    MiniMax-H3's Qwen3-VL nests it at `<inner>.language_model.layers`; the fallbacks cover
    `<inner>.model.layers` and a top-level `.layers`. Callers pass either the outer text encoder or
    its inner `Qwen3VLModel`, so both the module and its `.model` are probed as roots. The language
    model is probed before the vision tower so the stack the encode pass actually walks is returned.
    """
    for root in (model, getattr(model, "model", None)):
        if not isinstance(root, torch.nn.Module):
            continue
        for lm in (getattr(root, "language_model", None), getattr(root, "model", None), root):
            layers = getattr(lm, "layers", None)
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                return layers
    return None


def shard_model_fsdp2(model, device_id, param_dtype=None, cpu_offload=True,
                      reshard_after_forward=True):
    """FSDP2-shard a conditioner's decoder stack, optionally 1-wide with pinned CPU offload.

    This is the FSDP2 counterpart of `shard_model`'s `offload_to_cpu=True` path, for the text encoder
    under sequence-parallel inference. Only the decoder layers are sharded: wrapping a root would turn
    `embed_tokens` into a DTensor too, and its forward then mixes a DTensor weight with a plain
    `input_ids` (`aten.embedding.default got mixed torch.Tensor and DTensor`). The embedding, the final
    norm and the vision tower stay replicated (~2.6 GiB) and are moved to the card.

    With `cpu_offload=True` the shard mesh is 1 wide (`init_device_mesh("cuda", (1, world))["offload"]`),
    so every rank keeps a full pinned host copy and the per-request "gather" degenerates into a local
    copy plus an asynchronous host-to-card transfer -- no collective. A true `world`-wide shard needs one
    all-gather per decoder layer, which dominates the text stage on a PCIe-only box. The trade is host
    memory: `world` full copies instead of one sharded set, so this path needs the RAM to hold them and a
    `ulimit -l` large enough to pin them.

    FSDP1 and FSDP2 are mutually exclusive on the same module, so a model already wrapped by `shard_model`
    must not be passed here. `fully_shard` mutates the layers in place rather than wrapping the root, so
    the caller keeps using the same object it passed in, which is what is returned.

    NOT wired into the example scripts, and deliberately so on this repo's torch 2.7. Measured there
    (`_qtwork/_probe_fsdp2_mem.py`, 2 ranks, `no_grad`): the wrap does park the shards on the host (at-rest
    ~0 GiB on-card), but a forward unshards every wrapped layer to the GPU and -- inference having no
    backward to trigger the reshard -- leaves them all resident afterwards (after-forward ~ the whole
    decoder, params turned back into plain CUDA tensors). `shard_model(..., offload_to_cpu=True)`'s
    FSDP1 CPUOffload does not do this: its forward hooks reshard each layer to the host right after it
    runs (peak ~2 layers, after-forward ~at-rest), and on an NVLink box its per-layer all-gather is
    cheap, so FSDP1 is both correct and fast here. Wiring this FSDP2 path as the text-encoder offload
    would therefore leave a 58 GiB conditioner on-card after the first encode and OOM beside the DiT.
    It is kept for a torch whose FSDP2 reshards/offloads after a pure-forward pass (the source protocol
    measured 1.24 s on torch 2.12); verify with `_qtwork/_probe_fsdp2_mem.py` before wiring it anywhere.
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import (CPUOffloadPolicy, MixedPrecisionPolicy,
                                        fully_shard)
    from torch.distributed.tensor import DTensor

    layers = locate_decoder_layers(model)
    if layers is None:
        raise ValueError(
            "shard_model_fsdp2 cannot locate the decoder-layer stack on %s; pass a conditioner with a "
            "`.language_model.layers` / `.model.layers` / `.layers` ModuleList." % type(model).__name__)

    world = dist.get_world_size() if dist.is_initialized() else 1
    if cpu_offload:
        mesh = init_device_mesh("cuda", (1, world), mesh_dim_names=("offload", "replicate"))["offload"]
        offload_policy = CPUOffloadPolicy(pin_memory=True)
    else:
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("shard",))
        offload_policy = None
    # `cast_forward_inputs=False` keeps the plain `input_ids` / attention masks untouched; the layers'
    # params already carry the checkpoint dtype (bf16), so `param_dtype=None` adds no cast unless asked.
    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=None, output_dtype=None,
                              cast_forward_inputs=False)

    for layer in layers:
        kwargs = {"mesh": mesh, "mp_policy": mp, "reshard_after_forward": reshard_after_forward}
        if offload_policy is not None:
            kwargs["offload_policy"] = offload_policy
        fully_shard(layer, **kwargs)

    if cpu_offload:
        # The shards stay host-resident, so `model.to(device)` is not callable here (it would drag the
        # offloaded shards onto the card); move only the replicated non-DTensor params and the buffers.
        for p in model.parameters():
            if not isinstance(p, DTensor):
                p.data = p.data.to(device_id)
        for b in model.buffers():
            b.data = b.data.to(device_id)
    else:
        try:
            model.to(device_id)
        except Exception as exc:  # FSDP2 may already have placed the shards
            print("[shard_model_fsdp2] .to(device) skipped: %s" % exc, flush=True)

    shard_bytes = sum(p.to_local().numel() * p.to_local().element_size()
                      for p in model.parameters() if isinstance(p, DTensor))
    replicated_bytes = sum(p.numel() * p.element_size()
                           for p in model.parameters() if not isinstance(p, DTensor))
    print("[shard_model_fsdp2] %d layers sharded (cpu_offload=%s, shard-dim=%d, pinned=%s), local "
          "%.2f GiB sharded + %.2f GiB replicated."
          % (len(layers), cpu_offload, 1 if cpu_offload else world, cpu_offload,
             shard_bytes / 2**30, replicated_bytes / 2**30), flush=True)
    return model


# Attribute holding a module's pinned host mirror, installed by `build_pinned_mirror`.
_PINNED_HOST_ATTR = "_vxf_pinned_host"


def _pinned_cpu_copy(tensor):
    """A pinned CPU copy of `tensor`, reusing it when it is already a pinned host tensor."""
    src = tensor.detach()
    if src.device.type != "cpu":
        src = src.to("cpu")
    return src if src.is_pinned() else src.pin_memory()


def build_pinned_mirror(module):
    """Keep one pinned host copy per tensor of `module`; load/unload only repoint `.data`.

    A whole-component offload that parks the weights with `.to("cpu")` pays a device-to-host copy on every
    offload. For a component decoded once per request (the video / audio VAE) that copy is pure overhead: the
    host copy never changes, so it is built once, pinned, and reused. Onload allocates a fresh device tensor
    and copies it from the pinned buffer asynchronously; unload drops the device storage by repointing `.data`
    back at the mirror, with no device-to-host transfer at all.

    Inference only: the mirror captures the weights once and assumes nothing mutates them in place.
    `named_parameters()` / `named_buffers()` deduplicate tied tensors, so a shared weight is mirrored once and
    both names repoint to the same host tensor. Pinning needs a locked-memory limit (`ulimit -l`) large enough
    for the whole mirror; `pin_memory` raises when it is not, and the caller falls back to a two-way offload.
    """
    host = {}
    for name, param in module.named_parameters():
        host["p:" + name] = _pinned_cpu_copy(param.data)
    for name, buf in module.named_buffers():
        host["b:" + name] = _pinned_cpu_copy(buf.data)
    setattr(module, _PINNED_HOST_ATTR, host)
    return host


def pinned_load(module, device):
    """Point every tensor of `module` at a fresh device copy of its pinned host mirror."""
    host = getattr(module, _PINNED_HOST_ATTR, None)
    if host is None:
        return module.to(device)
    device = torch.device(device)
    for name, param in module.named_parameters():
        if param.data.device != device:
            dst = torch.empty_like(host["p:" + name], device=device)
            dst.copy_(host["p:" + name], non_blocking=True)
            param.data = dst
    for name, buf in module.named_buffers():
        if buf.data.device != device:
            dst = torch.empty_like(host["b:" + name], device=device)
            dst.copy_(host["b:" + name], non_blocking=True)
            buf.data = dst
    return module


def pinned_unload(module):
    """Repoint every device tensor of `module` back at its pinned host mirror; no device-to-host copy."""
    host = getattr(module, _PINNED_HOST_ATTR, None)
    if host is None:
        return module.to("cpu")
    for name, param in module.named_parameters():
        if param.data.device.type != "cpu":
            param.data = host["p:" + name]
    for name, buf in module.named_buffers():
        if buf.data.device.type != "cpu":
            buf.data = host["b:" + name]
    return module


class PinnedCpuOffload(ModelHook):
    """One-way CPU-offload hook backed by a pinned host mirror.

    Same contract and same chaining as accelerate's `CpuOffload` -- installed as the module's `_hf_hook` (so
    `_offload_scope` fires it around `encode` / `decode`), exposes `execution_device` (so `_execution_device`
    keeps reporting the onload GPU), and offloads the previous component from `pre_forward` -- so it drops into
    `offload_components_cpu` in place of `cpu_offload_with_hook`. Only the transfer direction differs: offload
    repoints `.data` at the pinned host copy instead of a device-to-host `.to("cpu")`, and onload copies back
    asynchronously. The device cache is deliberately left intact across an offload, so the next onload reuses
    the freed blocks.

    `init_hook` is what `UserCpuOffloadHook.offload()` calls, so it both builds the mirror on attach and
    re-parks the module on every chained offload; it is idempotent.
    """

    def __init__(self, execution_device=None, prev_module_hook=None):
        self.prev_module_hook = prev_module_hook
        self.execution_device = execution_device

    def init_hook(self, module):
        if getattr(module, _PINNED_HOST_ATTR, None) is None:
            build_pinned_mirror(module)
        pinned_unload(module)
        return module

    def pre_forward(self, module, *args, **kwargs):
        if self.prev_module_hook is not None:
            self.prev_module_hook.offload()
        pinned_load(module, self.execution_device)
        return send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device)


def _offload_component_pinned(model, onload_device, prev_hook):
    """Attach a `PinnedCpuOffload` hook, falling back to the two-way accelerate hook when pinning is unavailable.

    Returns the same `(model, UserCpuOffloadHook)` pair `cpu_offload_with_hook` does, so the caller chains and
    records it identically whichever path attached it. The mirror is built before the hook is added, so a
    `pin_memory` failure under a restricted `ulimit -l` leaves the module's `forward` untouched.
    """
    from accelerate import cpu_offload_with_hook

    try:
        build_pinned_mirror(model)
    except (RuntimeError, OSError) as exc:
        # `pin_memory` surfaces a locked-memory failure as RuntimeError, but a restricted `ulimit -l`
        # can also raise OSError from the underlying cudaHostRegister; either way fall back cleanly.
        print("[offload_components_cpu] pinned mirror unavailable (%s); falling back to two-way CPU "
              "offload for %s." % (exc, type(model).__name__))
        return cpu_offload_with_hook(model, onload_device, prev_module_hook=prev_hook)
    hook = PinnedCpuOffload(execution_device=onload_device, prev_module_hook=prev_hook)
    add_hook_to_module(model, hook, append=True)
    return model, UserCpuOffloadHook(model, hook)


def offload_components_cpu(obj, device, component_names, pinned=False):
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

    With `pinned=True` each named component is offloaded through a `PinnedCpuOffload` hook instead: a pinned
    host mirror is built once and the offload only repoints `.data` back at it, dropping the device-to-host leg
    a two-way `.to("cpu")` pays on every request. It falls back to the two-way hook per component when pinning
    is unavailable, so the offload still happens.
    """
    from accelerate import cpu_offload_with_hook

    onload_device = torch.device(device) if isinstance(device, str) else device

    hooks = []
    prev_hook = None
    for name in component_names:
        model = getattr(obj, name, None)
        if not isinstance(model, torch.nn.Module):
            continue
        if pinned:
            _, hook = _offload_component_pinned(model, onload_device, prev_hook)
        else:
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