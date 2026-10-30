"""Blockwise autograd replay for memory-bounded FSDP training."""

from contextlib import nullcontext

import torch

__all__ = ["checkpoint_fsdp_forward"]


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
