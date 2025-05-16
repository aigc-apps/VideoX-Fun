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
    usp_attn_forward = wan_usp_sparse_attention_wrapper()(usp_attn_forward)
    print("Enable PAI VAE Turbo and Sparse Attention")