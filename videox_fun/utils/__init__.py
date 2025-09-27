import importlib.util

from .fm_solvers import FlowDPMSolverMultistepScheduler
from .fm_solvers_unipc import FlowUniPCMultistepScheduler
from .fp8_optimization import (autocast_model_forward,
                               convert_model_weight_to_float8,
                               convert_weight_dtype_wrapper,
                               replace_parameters_by_name)
from .lora_utils import merge_lora, unmerge_lora
from .utils import (filter_kwargs, get_image_latent, get_image_to_video_latent, get_autocast_dtype,
                    get_video_to_video_latent, save_videos_grid)
from .cfg_optimization import cfg_skip
from .discrete_sampler import DiscreteSampling


# The pai_fuser is an internally developed acceleration package, which can be used on PAI.
if importlib.util.find_spec("paifuser") is not None:
    # --------------------------------------------------------------- #
    #   FP8 Linear Kernel
    # --------------------------------------------------------------- #
    from paifuser.ops import (convert_model_weight_to_float8,
                                convert_weight_dtype_wrapper)
    from . import fp8_optimization
    fp8_optimization.convert_model_weight_to_float8 = convert_model_weight_to_float8
    fp8_optimization.convert_weight_dtype_wrapper = convert_weight_dtype_wrapper
    convert_model_weight_to_float8 = fp8_optimization.convert_model_weight_to_float8
    convert_weight_dtype_wrapper = fp8_optimization.convert_weight_dtype_wrapper
    print("Import PAI Quantization Turbo")

    # --------------------------------------------------------------- #
    #   CFG Skip Turbo
    # --------------------------------------------------------------- #
    if importlib.util.find_spec("paifuser.accelerator") is not None:
        from paifuser.accelerator import (cfg_skip_turbo, disable_cfg_skip,
                                          enable_cfg_skip, share_cfg_skip)
    else:
        from paifuser import (cfg_skip_turbo, disable_cfg_skip,
                              enable_cfg_skip, share_cfg_skip)
    from . import cfg_optimization
    cfg_optimization.cfg_skip = cfg_skip_turbo
    cfg_skip = cfg_skip_turbo
    print("Import CFG Skip Turbo")