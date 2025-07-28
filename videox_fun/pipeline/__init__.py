from .pipeline_cogvideox_fun import CogVideoXFunPipeline
from .pipeline_cogvideox_fun_control import CogVideoXFunControlPipeline
from .pipeline_cogvideox_fun_inpaint import CogVideoXFunInpaintPipeline
from .pipeline_wan_fun import WanFunPipeline
from .pipeline_wan_fun_inpaint import WanFunInpaintPipeline
from .pipeline_wan_fun_control import WanFunControlPipeline
from .pipeline_wan_phantom import WanFunPhantomPipeline
from .pipeline_wan2_2 import Wan2_2Pipeline
from .pipeline_wan2_2_i2v import Wan2_2I2VPipeline

WanPipeline = WanFunPipeline
WanI2VPipeline = WanFunInpaintPipeline

import importlib.util

if importlib.util.find_spec("pai_fuser") is not None:
    from pai_fuser.core import sparse_reset

    WanFunInpaintPipeline.__call__ = sparse_reset(WanFunInpaintPipeline.__call__)
    WanFunPipeline.__call__ = sparse_reset(WanFunPipeline.__call__)
    WanFunControlPipeline.__call__ = sparse_reset(WanFunControlPipeline.__call__)
    WanI2VPipeline.__call__ = sparse_reset(WanI2VPipeline.__call__)
    WanPipeline.__call__ = sparse_reset(WanPipeline.__call__)
    WanFunPhantomPipeline.__call__ = sparse_reset(WanFunPhantomPipeline.__call__)
    Wan2_2Pipeline.__call__ = sparse_reset(Wan2_2Pipeline.__call__)
    Wan2_2I2VPipeline.__call__ = sparse_reset(Wan2_2I2VPipeline.__call__)