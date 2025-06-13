from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)


non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)


def apply_checkpointing(model, block, p):
    """
    Apply selective activation checkpointing.

    Selectivity is defined as a percentage p, which means we apply ac
    on p of the total blocks. p is a floating number in the range of
    [0, 1].

    Some examples:
    p = 0: no ac for all blocks. same as `fsdp_activation_checkpointing=False`
    p = 1: apply ac on every block. i.e. "full ac".
    p = 1/2: [ac, no-ac, ac, no-ac, ...]
    p = 1/3: [no-ac, ac, no-ac,   no-ac, ac, no-ac,   ...]
    p = 2/3: [ac, no-ac, ac,    ac, no-ac, ac,    ...]
    Since blocks are homogeneous, we make ac blocks evenly spaced among
    all blocks.

    Implementation:
    For a given ac ratio p, we should essentially apply ac on every "1/p"
    blocks. The first ac block can be as early as the 0th block, or as
    late as the "1/p"th block, and we pick the middle one: (0.5p)th block.
    Therefore, we are essentially to apply ac on:
    (0.5/p)th block, (1.5/p)th block, (2.5/p)th block, etc., and of course,
    with these values rounding to integers.
    Since ac is applied recursively, we can simply use the following math
    in the code to apply ac on corresponding blocks.
    """
    block_idx = 0
    cut_off = 1 / 2
    # when passing p as a fraction number (e.g. 1/3), it will be interpreted
    # as a string in argv, thus we need eval("1/3") here for fractions.
    p = eval(p) if isinstance(p, str) else p

    def selective_checkpointing(submodule):
        nonlocal block_idx
        nonlocal cut_off

        if isinstance(submodule, block):
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )
