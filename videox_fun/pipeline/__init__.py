from .pipeline_cogvideox_fun import CogVideoXFunPipeline
from .pipeline_cogvideox_fun_control import CogVideoXFunControlPipeline
from .pipeline_cogvideox_fun_inpaint import CogVideoXFunInpaintPipeline
from .pipeline_flux import FluxPipeline
from .pipeline_fantasy_talking import FantasyTalkingPipeline
from .pipeline_qwenimage import QwenImagePipeline
from .pipeline_qwenimage_edit import QwenImageEditPipeline
from .pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from .pipeline_wan import WanPipeline
from .pipeline_wan2_2 import Wan2_2Pipeline
from .pipeline_wan2_2_fun_control import Wan2_2FunControlPipeline
from .pipeline_wan2_2_fun_inpaint import Wan2_2FunInpaintPipeline
from .pipeline_wan2_2_s2v import Wan2_2S2VPipeline
from .pipeline_wan2_2_ti2v import Wan2_2TI2VPipeline
from .pipeline_wan2_2_vace_fun import Wan2_2VaceFunPipeline
from .pipeline_wan_fun_control import WanFunControlPipeline
from .pipeline_wan_fun_inpaint import WanFunInpaintPipeline
from .pipeline_wan_phantom import WanFunPhantomPipeline
from .pipeline_wan_vace import WanVacePipeline

WanFunPipeline = WanPipeline
WanI2VPipeline = WanFunInpaintPipeline

Wan2_2FunPipeline = Wan2_2Pipeline
Wan2_2I2VPipeline = Wan2_2FunInpaintPipeline

import importlib.util

if importlib.util.find_spec("paifuser") is not None:
    # --------------------------------------------------------------- #
    #   Sparse Attention
    # --------------------------------------------------------------- #
    from paifuser.ops import sparse_reset

    # Wan2.1
    WanFunInpaintPipeline.__call__ = sparse_reset(WanFunInpaintPipeline.__call__)
    WanFunPipeline.__call__ = sparse_reset(WanFunPipeline.__call__)
    WanFunControlPipeline.__call__ = sparse_reset(WanFunControlPipeline.__call__)
    WanI2VPipeline.__call__ = sparse_reset(WanI2VPipeline.__call__)
    WanPipeline.__call__ = sparse_reset(WanPipeline.__call__)

    # Phantom
    WanFunPhantomPipeline.__call__ = sparse_reset(WanFunPhantomPipeline.__call__)

    # Wan2.2
    Wan2_2FunInpaintPipeline.__call__ = sparse_reset(Wan2_2FunInpaintPipeline.__call__)
    Wan2_2FunPipeline.__call__ = sparse_reset(Wan2_2FunPipeline.__call__)
    Wan2_2FunControlPipeline.__call__ = sparse_reset(Wan2_2FunControlPipeline.__call__)
    Wan2_2Pipeline.__call__ = sparse_reset(Wan2_2Pipeline.__call__)
    Wan2_2I2VPipeline.__call__ = sparse_reset(Wan2_2I2VPipeline.__call__)
    Wan2_2TI2VPipeline.__call__ = sparse_reset(Wan2_2TI2VPipeline.__call__)