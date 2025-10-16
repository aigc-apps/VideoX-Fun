import importlib.util

from .cogvideox_xfuser import CogVideoXMultiGPUsAttnProcessor2_0
from .flux_xfuser import FluxMultiGPUsAttnProcessor2_0
from .fsdp import shard_model
from .fuser import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    get_world_group, init_distributed_environment,
                    initialize_model_parallel, sequence_parallel_all_gather,
                    sequence_parallel_chunk, set_multi_gpus_devices,
                    xFuserLongContextAttention)
from .qwen_xfuser import QwenImageMultiGPUsAttnProcessor2_0
from .wan_xfuser import usp_attn_forward, usp_attn_s2v_forward

# The pai_fuser is an internally developed acceleration package, which can be used on PAI.
if importlib.util.find_spec("paifuser") is not None:
    # --------------------------------------------------------------- #
    #   The simple_wrapper is used to solve the problem 
    #   about conflicts between cython and torch.compile
    # --------------------------------------------------------------- #
    def simple_wrapper(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner

    # --------------------------------------------------------------- #
    #   Sparse Attention Kernel
    # --------------------------------------------------------------- #
    from paifuser.models import parallel_magvit_vae
    from paifuser.ops import wan_usp_sparse_attention_wrapper

    from . import wan_xfuser

    # --------------------------------------------------------------- #
    #   Sparse Attention
    # --------------------------------------------------------------- #
    usp_sparse_attn_wrap_forward = simple_wrapper(wan_usp_sparse_attention_wrapper()(wan_xfuser.usp_attn_forward))
    wan_xfuser.usp_attn_forward = usp_sparse_attn_wrap_forward
    usp_attn_forward = usp_sparse_attn_wrap_forward
    print("Import PAI VAE Turbo and Sparse Attention")

    # --------------------------------------------------------------- #
    #   Fast Rope Kernel
    # --------------------------------------------------------------- #
    import types

    import torch
    from paifuser.ops import (ENABLE_KERNEL, usp_fast_rope_apply_qk,
                              usp_rope_apply_real_qk)

    def deepcopy_function(f):
        return types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__,closure=f.__closure__)

    local_rope_apply_qk = deepcopy_function(wan_xfuser.rope_apply_qk)

    if ENABLE_KERNEL:
        def adaptive_fast_usp_rope_apply_qk(q, k, grid_sizes, freqs):
            if torch.is_grad_enabled():
                return local_rope_apply_qk(q, k, grid_sizes, freqs)
            else:
                return usp_fast_rope_apply_qk(q, k, grid_sizes, freqs)

    else:
        def adaptive_fast_usp_rope_apply_qk(q, k, grid_sizes, freqs):
            return usp_rope_apply_real_qk(q, k, grid_sizes, freqs)
            
    wan_xfuser.rope_apply_qk = adaptive_fast_usp_rope_apply_qk
    rope_apply_qk = adaptive_fast_usp_rope_apply_qk
    print("Import PAI Fast rope")