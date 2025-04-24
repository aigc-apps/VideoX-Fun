import torch
import torch.distributed as dist

from .fsdp import shard_model

try:
    try:
        import pai_fuser
        from pai_fuser.core.distributed import (
            get_sequence_parallel_rank, get_sequence_parallel_world_size,
            get_sp_group, get_world_group, init_distributed_environment,
            initialize_model_parallel)
        from pai_fuser.core.long_ctx_attention import \
            xFuserLongContextAttention
    except Exception as ex:
        import xfuser
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                             get_sequence_parallel_world_size,
                                             get_sp_group, get_world_group,
                                             init_distributed_environment,
                                             initialize_model_parallel)
        from xfuser.core.long_ctx_attention import xFuserLongContextAttention
except Exception as ex:
    get_sequence_parallel_world_size = None
    get_sequence_parallel_rank = None
    xFuserLongContextAttention = None
    get_sp_group = None
    get_world_group = None
    init_distributed_environment = None
    initialize_model_parallel = None

try: 
    from pai_fuser.core import parallel_magvit_vae
except:
    def parallel_magvit_vae(multi_gpus_overlap_scale, spatial_compression_ratio):
        def decorator(func):
            def wrapper(self, z, *args, **kwargs):
                decoded = func(self, z, *args, **kwargs)
                return decoded
            return wrapper
        return decorator

def set_multi_gpus_devices(ulysses_degree, ring_degree):
    if ulysses_degree > 1 or ring_degree > 1:
        if get_sp_group is None:
            raise RuntimeError("xfuser is not installed.")
        dist.init_process_group("nccl")
        print('parallel inference enabled: ulysses_degree=%d ring_degree=%d rank=%d world_size=%d' % (
            ulysses_degree, ring_degree, dist.get_rank(),
            dist.get_world_size()))
        assert dist.get_world_size() == ring_degree * ulysses_degree, \
                    "number of GPUs(%d) should be equal to ring_degree * ulysses_degree." % dist.get_world_size()
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(sequence_parallel_degree=dist.get_world_size(),
                ring_degree=ring_degree,
                ulysses_degree=ulysses_degree)
        # device = torch.device("cuda:%d" % dist.get_rank())
        device = torch.device(f"cuda:{get_world_group().local_rank}")
        print('rank=%d device=%s' % (get_world_group().rank, str(device)))
    else:
        device = "cuda"
    return device