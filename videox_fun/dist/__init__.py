import importlib.util

from .cogvideox_xfuser import CogVideoXMultiGPUsAttnProcessor2_0
from .fsdp import shard_model
from .fuser import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    get_world_group, init_distributed_environment,
                    initialize_model_parallel, set_multi_gpus_devices,
                    xFuserLongContextAttention)
from .wan_xfuser import usp_attn_forward

# The pai_fuser is an internally developed acceleration package, which can be used on PAI.
if importlib.util.find_spec("pai_fuser") is not None:
    from pai_fuser.core import parallel_magvit_vae
    from pai_fuser.core.attention import wan_usp_sparse_attention_wrapper
    from . import wan_xfuser
    
    # The simple_wrapper is used to solve the problem about conflicts between cython and torch.compile
    def simple_wrapper(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner

    wan_xfuser.usp_attn_forward = simple_wrapper(wan_usp_sparse_attention_wrapper()(wan_xfuser.usp_attn_forward))
    usp_attn_forward = simple_wrapper(wan_xfuser.usp_attn_forward)
    print("Import PAI VAE Turbo and Sparse Attention")

    from pai_fuser.core.rope import ENABLE_KERNEL, usp_fast_rope_apply_qk

    if ENABLE_KERNEL:
        wan_xfuser.rope_apply_qk = usp_fast_rope_apply_qk
        rope_apply_qk = usp_fast_rope_apply_qk
        print("Import PAI Fast rope")