import importlib.util

import torch
import torch.distributed as dist

try:
    # The pai_fuser is an internally developed acceleration package, which can be used on PAI.
    if importlib.util.find_spec("paifuser") is not None:
        import paifuser
        from paifuser.xfuser.core.distributed import (
            get_sequence_parallel_rank, get_sequence_parallel_world_size,
            get_sp_group, get_world_group, init_distributed_environment,
            initialize_model_parallel, model_parallel_is_initialized)
        from paifuser.xfuser.core.long_ctx_attention import \
            xFuserLongContextAttention
        print("Import PAI DiT Turbo")
    else:
        import xfuser
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                             get_sequence_parallel_world_size,
                                             get_sp_group, get_world_group,
                                             init_distributed_environment,
                                             initialize_model_parallel,
                                             model_parallel_is_initialized)
        from xfuser.core.long_ctx_attention import xFuserLongContextAttention
        print("Xfuser import sucessful")
except Exception as ex:
    get_sequence_parallel_world_size = None
    get_sequence_parallel_rank = None
    xFuserLongContextAttention = None
    get_sp_group = None
    get_world_group = None
    init_distributed_environment = None
    initialize_model_parallel = None

def set_multi_gpus_devices(ulysses_degree, ring_degree, classifier_free_guidance_degree=1):
    if ulysses_degree > 1 or ring_degree > 1 or classifier_free_guidance_degree > 1:
        if get_sp_group is None:
            raise RuntimeError("xfuser is not installed.")
        dist.init_process_group("nccl")
        print('parallel inference enabled: ulysses_degree=%d ring_degree=%d classifier_free_guidance_degree=% rank=%d world_size=%d' % (
            ulysses_degree, ring_degree, classifier_free_guidance_degree, dist.get_rank(),
            dist.get_world_size()))
        assert dist.get_world_size() == ring_degree * ulysses_degree * classifier_free_guidance_degree, \
                    "number of GPUs(%d) should be equal to ring_degree * ulysses_degree * classifier_free_guidance_degree." % dist.get_world_size()
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(sequence_parallel_degree=ring_degree * ulysses_degree,
                classifier_free_guidance_degree=classifier_free_guidance_degree,
                ring_degree=ring_degree,
                ulysses_degree=ulysses_degree)
        # device = torch.device("cuda:%d" % dist.get_rank())
        device = torch.device(f"cuda:{get_world_group().local_rank}")
        print('rank=%d device=%s' % (get_world_group().rank, str(device)))
    else:
        device = "cuda"
    return device

def sequence_parallel_chunk(x, dim=1):
    if get_sequence_parallel_world_size is None or not model_parallel_is_initialized():
        return x

    sp_world_size = get_sequence_parallel_world_size()
    if sp_world_size <= 1:
        return x

    sp_rank = get_sequence_parallel_rank()
    sp_group = get_sp_group()

    if x.size(1) % sp_world_size != 0:
        raise ValueError(f"Dim 1 of x ({x.size(1)}) not divisible by SP world size ({sp_world_size})")

    chunks = torch.chunk(x, sp_world_size, dim=1)
    x = chunks[sp_rank]

    return x

def sequence_parallel_all_gather(x, dim=1):
    if get_sequence_parallel_world_size is None or not model_parallel_is_initialized():
        return x

    sp_world_size = get_sequence_parallel_world_size()
    if sp_world_size <= 1:
        return x  # No gathering needed

    sp_group = get_sp_group()
    gathered_x = sp_group.all_gather(x, dim=dim)
    return gathered_x