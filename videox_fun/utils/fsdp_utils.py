"""Blockwise replay and opt-in CPU-offload memory management for FSDP1."""

import gc
from contextlib import contextmanager, nullcontext

import torch

__all__ = ['checkpoint_fsdp_forward', 'release_frozen_fsdp_gradients', 'release_completed_fsdp_gradients', 'release_offloaded_fsdp_gradients', 'release_fsdp_gradients_during_validation', 'release_fsdp_gradients_during_replay', 'configure_fsdp_pageable_cpu_parameters', 'configure_fsdp_precision']


def _require_cpu_offload_runtime():
    from packaging.version import Version

    if Version(torch.__version__).release[:2] != (2, 7):
        raise RuntimeError("FSDP CPU-offload memory patches support PyTorch 2.7; "
                           "validate private runtime contracts before enabling on another version")


def checkpoint_fsdp_forward(model, accumulated, *, context_fn=None, **kwargs):
    """Replay independent model forwards and accumulate parameter gradients.

    Share a fresh ``accumulated`` dictionary across all blocks of one rollout.
    Call ``loss.backward()`` once, then clip gradients and update the optimizer.
    Existing model gradients are cleared during replay; do not mix this helper
    with other gradient accumulation or direct parameter paths in the same loss.

    Distributed use requires FSDP1 FULL_SHARD with CPUOffload,
    ``use_orig_params=True``, and gradient synchronization enabled. FSDP2,
    ``no_sync``, higher-order gradients, and repeated backward are unsupported.
    CPU modules can use the same replay semantics without distributed wrapping.

    Inputs are keyword arguments containing tensors or immutable metadata.
    Tensors must be passed directly, not inside containers. Outputs must be a
    tensor or a tuple of tensors. All CUDA inputs must use one device.
    ``context_fn`` optionally snapshots mutable forward state and returns
    ``(forward_context, recompute_context)``, as in non-reentrant checkpointing.
    The model must not mutate parameters or buffers during replay.
    """
    keys = tuple(kwargs)
    values = tuple(kwargs.values())
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if not tensors:
        raise ValueError("FSDP replay requires at least one tensor input.")
    device = next((value.device for value in tensors if value.is_cuda), tensors[0].device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"FSDP replay supports CPU and CUDA inputs, got {device}.")
    if any(value.is_cuda and value.device != device for value in tensors):
        raise ValueError("FSDP replay requires all CUDA inputs on one device.")
    cuda_devices = [device.index] if device.type == "cuda" else []

    class Replay(torch.autograd.Function):
        @staticmethod
        def forward(ctx, anchor, *args):
            ctx.tensor_indices = [index for index, value in enumerate(args) if isinstance(value, torch.Tensor)]
            ctx.other_args = {index: value for index, value in enumerate(args) if not isinstance(value, torch.Tensor)}
            tensors = [args[index] for index in ctx.tensor_indices]
            ctx.input_requires_grad = [value.requires_grad for value in tensors]
            ctx.save_for_backward(*tensors)
            forward_context, ctx.replay_context = context_fn() if context_fn is not None else (nullcontext(), nullcontext())
            ctx.cpu_rng = torch.get_rng_state()
            ctx.cuda_rng = torch.cuda.get_rng_state(device) if cuda_devices else None
            ctx.autocast = torch.is_autocast_enabled(device.type)
            ctx.autocast_dtype = torch.get_autocast_dtype(device.type)
            ctx.set_materialize_grads(False)
            with forward_context:
                outputs = model(**dict(zip(keys, args)))
            if not isinstance(outputs, torch.Tensor) and not (
                isinstance(outputs, tuple) and all(isinstance(output, torch.Tensor) for output in outputs)
            ):
                raise TypeError("FSDP replay requires a tensor or a tuple of tensors as output.")
            return outputs

        @staticmethod
        def backward(ctx, *output_grads):
            parameters = list(model.parameters())
            # Clear gradient slots; the separate accumulator preserves earlier block gradients.
            previous_grads = [accumulated.get(parameter) for parameter in parameters]
            model.zero_grad(set_to_none=True)
            tensors = [value.detach().requires_grad_(requires_grad)
                       for value, requires_grad in zip(ctx.saved_tensors, ctx.input_requires_grad)]
            args = dict(ctx.other_args)
            args.update(zip(ctx.tensor_indices, tensors))
            with torch.random.fork_rng(devices=cuda_devices), torch.enable_grad(), \
                    torch.autocast(device.type, enabled=ctx.autocast, dtype=ctx.autocast_dtype):
                torch.set_rng_state(ctx.cpu_rng)
                if cuda_devices:
                    torch.cuda.set_rng_state(ctx.cuda_rng, device)
                with ctx.replay_context:
                    outputs = model(**{key: args[index] for index, key in enumerate(keys)})
                outputs = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
                active = [(output, grad) for output, grad in zip(outputs, output_grads)
                          if grad is not None and output.requires_grad]
                if active:
                    torch.autograd.backward([output for output, _ in active], [grad for _, grad in active])
            # FSDP completes reduce-scatter and offload before accumulating local shards.
            for parameter, previous in zip(parameters, previous_grads):
                if previous is not None:
                    if parameter.grad is None:
                        parameter.grad = previous
                    else:
                        parameter.grad.add_(previous)
                if parameter.grad is not None:
                    if previous is None:
                        accumulated[parameter] = parameter.grad.detach().clone()
                    else:
                        # Keep one independent accumulator per shard across blocks.
                        # Replacing it with a clone retains both full copies until
                        # previous_grads leaves this backward call.
                        previous.copy_(parameter.grad)
            result = [None] * len(keys)
            for index, tensor in zip(ctx.tensor_indices, tensors):
                result[index] = tensor.grad
            return (None, *result)

    with torch.autograd.graph.save_on_cpu():
        return Replay.apply(torch.empty(0, device=device, requires_grad=True), *values)


def release_frozen_fsdp_gradients(model):
    """Release unused CPU gradient buffers after permanently frozen forwards.

    FSDP1 allocates CPU-offloaded gradients even for frozen parameters. Clearing
    each unit after forward bounds their lifetime; FSDP supports None and will
    allocate the buffer again on its next forward. Teacher/EMA/text weights and
    saved activations remain live. This must not be used on trainable models.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    _require_cpu_offload_runtime()

    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('Frozen FSDP gradient release requires all parameters to be frozen')
    if getattr(model, '_videox_frozen_gradient_hooks', None) is not None:
        return model
    hooks = []

    def release(module, inputs, output):
        handle = module._handle
        if handle is not None and handle._offload_params:
            if handle.flat_param.requires_grad:
                raise RuntimeError('A permanently frozen FSDP model was made trainable')
            handle.flat_param._cpu_grad = None

    def release_after_init(handle, initialize):
        # Root lazy initialization visits every handle before the first forward.
        # Drop each frozen unit's unused buffer immediately so those allocations
        # do not accumulate into another whole model during lazy initialization.
        def initialize_frozen(*args, **kwargs):
            if handle.flat_param.requires_grad:
                raise RuntimeError('A permanently frozen FSDP model was made trainable')
            result = initialize(*args, **kwargs)
            handle.flat_param._cpu_grad = None
            return result
        return initialize_frozen

    for module in model.modules():
        if isinstance(module, FSDP):
            handle = module._handle
            if handle is not None and handle._offload_params:
                handle.init_flat_param_attributes = release_after_init(handle, handle.init_flat_param_attributes)
            hooks.append(module.register_forward_hook(release))

    def trim(module, inputs, output):
        torch._C._host_emptyCache()

    hooks.append(model.register_forward_hook(trim))
    model._videox_frozen_gradient_hooks = hooks
    return model


def release_completed_fsdp_gradients(*models):
    """Free unused CPU gradient storage after zero_grad or before resume.

    FSDP recreates these buffers on the next forward. This bounds the save
    peak when saving multiple optimizers without discarding accumulated gradients.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    _require_cpu_offload_runtime()

    for model in models:
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError('Cannot release gradients before optimizer zero_grad(set_to_none=True)')
    released = 0
    for model in models:
        for module in model.modules():
            if not isinstance(module, FSDP):
                continue
            handle = module._handle
            if handle is None or not handle._offload_params:
                continue
            parameter = handle.flat_param
            gradient = getattr(parameter, '_cpu_grad', None)
            if gradient is not None:
                released += gradient.numel() * gradient.element_size()
            parameter.grad = None
            parameter._cpu_grad = None
            del gradient
    # Per-forward autograd Function classes can keep completed replay accumulators
    # alive through Python cycles even after their loss/output variables are gone.
    gc.collect()
    torch._C._host_emptyCache()
    return released


def release_offloaded_fsdp_gradients(model):
    """Drop redundant CUDA grad shards after FSDP1 has offloaded them to CPU.

    PyTorch 2.7 retains ``_saved_grad_shard`` until the end of backward even
    with CPUOffload. The optimizer consumes ``_cpu_grad`` on this path.
    FSDP records the D2H copy's stream before this callback returns, so the
    allocator keeps the CUDA storage alive until the asynchronous copy ends.
    Only explicitly marked modules use this compatibility workaround.
    """
    import functools
    import torch.distributed.fsdp._runtime_utils as runtime

    _require_cpu_offload_runtime()

    for module in model.modules():
        module._videox_release_cpu_grad = True
    original = runtime._post_reduce_grad_callback
    if getattr(original, "_videox_cpu_grad_release", False):
        return

    @functools.wraps(original)
    def post_reduce(state, handle, grad_to_offload):
        original(state, handle, grad_to_offload)
        if not (getattr(state, "_videox_release_cpu_grad", False)
                and handle._offload_params and not handle._has_optim_in_backward):
            return
        parameter = handle.flat_param
        cpu_grad = getattr(parameter, "_cpu_grad", None)
        saved = getattr(parameter, "_saved_grad_shard", None)
        if isinstance(cpu_grad, torch.Tensor) and cpu_grad.device.type == "cpu" \
                and isinstance(saved, torch.Tensor) and saved.is_cuda:
            del parameter._saved_grad_shard

    post_reduce._videox_cpu_grad_release = True
    runtime._post_reduce_grad_callback = post_reduce


@contextmanager
def release_fsdp_gradients_during_validation(model):
    """Bound unused offloaded gradient buffers during a no-grad validation pass.

    Enter only between completed training steps, with gradients cleared. FSDP1
    allocates CPU gradient buffers even under no_grad; release them after each
    unit instead of retaining a full model's buffers until validation ends.
    Hooks are temporary and do not affect subsequent training/backward.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    units = [unit for unit in model.modules() if isinstance(unit, FSDP)
             and unit._handle is not None and unit._handle._offload_params]
    if not units:
        yield
        return
    if torch.is_grad_enabled():
        raise RuntimeError('Validation gradient release requires torch.no_grad()')
    release_completed_fsdp_gradients(model)
    hooks = []

    def release(unit, inputs, output):
        if torch.is_grad_enabled():
            raise RuntimeError('Validation gradient release cannot run with autograd enabled')
        unit.zero_grad(set_to_none=True)
        unit._handle.flat_param.grad = None
        unit._handle.flat_param._cpu_grad = None

    try:
        for unit in units:
            hooks.append(unit.register_forward_hook(release))
        yield
    finally:
        for hook in hooks:
            hook.remove()
        model.zero_grad(set_to_none=True)
        release_completed_fsdp_gradients(model)


@contextmanager
def release_fsdp_gradients_during_replay(model):
    """Bound idle CPU gradients during checkpoint_fsdp_forward rollouts.

    The caller may have autograd enabled to record the custom replay outputs,
    but every actual model forward must run under no_grad. Exit this context
    before backward; the replay then recreates the native gradient buffers.
    Pending gradients are rejected, and both hooks and initializers are restored.
    """
    from contextlib import ExitStack
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    def without_idle_gradients(unit, initialize):
        def wrapped(*args, **kwargs):
            if torch.is_grad_enabled():
                raise RuntimeError('FSDP replay buffer release requires no-grad model forwards')
            result = initialize(*args, **kwargs)
            unit.zero_grad(set_to_none=True)
            unit._handle.flat_param.grad = None
            unit._handle.flat_param._cpu_grad = None
            return result
        return wrapped

    def restore_initializer(handle, had_override, previous):
        if had_override:
            handle.init_flat_param_attributes = previous
        else:
            del handle.init_flat_param_attributes

    with ExitStack() as stack:
        # Only context entry bypasses the no-grad check. The installed per-unit
        # hooks still reject autograd-enabled forwards before clearing buffers.
        with torch.no_grad():
            stack.enter_context(release_fsdp_gradients_during_validation(model))
        for unit in FSDP.fsdp_modules(model):
            handle = unit._handle
            if handle is None or not handle._offload_params:
                continue
            previous = handle.init_flat_param_attributes
            had_override = 'init_flat_param_attributes' in handle.__dict__
            stack.callback(restore_initializer, handle, had_override, previous)
            handle.init_flat_param_attributes = without_idle_gradients(unit, previous)
        yield


def configure_fsdp_pageable_cpu_parameters(model):
    """Keep original CPU parameter shards in low-memory FSDP1 training.

    Install before lazy initialization. Native FSDP still creates pinned gradient
    buffers for asynchronous backward copies; only the parameter shard retains
    its existing pageable storage. H2D parameter transfers may be slower.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    _require_cpu_offload_runtime()

    def initialize_pageable(handle, initialize):
        def wrapped(*args, **kwargs):
            original = handle.flat_param.data
            result = initialize(*args, **kwargs)
            parameter = handle.flat_param
            if original.device.type == 'cpu' and not original.is_pinned():
                if (parameter.data_ptr() != original.data_ptr()
                        or parameter.shape != original.shape
                        or parameter.dtype != original.dtype):
                    raise RuntimeError('FSDP initialization unexpectedly replaced the CPU parameter shard')
                parameter._local_shard = original
            return result
        return wrapped

    handles = [unit._handle for unit in FSDP.fsdp_modules(model)
               if unit._handle is not None and unit._handle._offload_params
               and not getattr(unit._handle, '_videox_pageable_cpu_parameters', False)]
    if any(hasattr(handle.flat_param, '_local_shard') for handle in handles):
        raise RuntimeError('Configure pageable FSDP parameters before lazy initialization')
    for handle in handles:
        handle.init_flat_param_attributes = initialize_pageable(handle, handle.init_flat_param_attributes)
        handle._videox_pageable_cpu_parameters = True
    return model


def configure_fsdp_precision(model, accelerator, *, keep_in_fp32_modules=(), buffer_dtype=torch.float32):
    """Preserve optimizer identities and apply explicit per-module precision.

    Call before prepare. ``keep_in_fp32_modules`` contains exact module paths;
    an empty sequence works for models without special precision requirements.
    Reuse the original auto-wrap policy when configuring multiple models.
    """
    from dataclasses import replace
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import CustomPolicy

    plugin = accelerator.state.fsdp_plugin
    if getattr(plugin, "fsdp_version", 1) != 1:
        raise ValueError("Per-module FSDP precision configuration requires FSDP1")
    if getattr(plugin.cpu_offload, "offload_params", False):
        release_offloaded_fsdp_gradients(model)
    plugin.use_orig_params = True
    if not hasattr(plugin, "_videox_base_wrap_policy"):
        plugin.set_auto_wrap_policy(model)
        plugin._videox_base_wrap_policy = plugin.auto_wrap_policy
    base_policy = plugin._videox_base_wrap_policy
    pinned = {module for name, module in model.named_modules() if name in keep_in_fp32_modules}
    precision = plugin.mixed_precision_policy or MixedPrecision()
    plugin.mixed_precision_policy = replace(precision, buffer_dtype=buffer_dtype)

    def policy(module):
        if module in pinned and any(True for _ in module.parameters()):
            return {"mixed_precision": MixedPrecision(
                param_dtype=torch.float32, reduce_dtype=torch.float32,
                buffer_dtype=torch.float32, cast_forward_inputs=True,
            )}
        return bool(callable(base_policy) and base_policy(
            module=module, recurse=False, nonwrapped_numel=sum(p.numel() for p in module.parameters())
        ))

    if base_policy is not None and not callable(base_policy):
        # CustomPolicy / ModuleWrapPolicy expose the policy protocol in FSDP1.
        class PrecisionPolicy(CustomPolicy):
            def _run_policy(self, root_module, ignored_modules, root_kwargs):
                result = base_policy._run_policy(root_module, ignored_modules, root_kwargs)
                for module, kwargs in super()._run_policy(root_module, ignored_modules, root_kwargs).items():
                    # Preserve other overrides from the existing policy.
                    result.setdefault(module, dict(root_kwargs))["mixed_precision"] = kwargs["mixed_precision"]
                return result
        plugin.auto_wrap_policy = PrecisionPolicy(policy)
    else:
        plugin.auto_wrap_policy = CustomPolicy(policy)
