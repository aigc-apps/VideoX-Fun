import importlib.util

from diffusers import AutoencoderKL
from transformers import (AutoTokenizer, CLIPImageProcessor, CLIPTextModel,
                          CLIPTokenizer, CLIPVisionModelWithProjection,
                          T5EncoderModel, T5Tokenizer, T5TokenizerFast)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor, Qwen2_5_VLConfig
except:
    Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer = None, None
    Qwen2VLProcessor, Qwen2_5_VLConfig = None, None
    print("Your transformers version is too old to load Qwen2_5_VLForConditionalGeneration and Qwen2Tokenizer. If you wish to use QwenImage, please upgrade your transformers package to the latest version.")

from .cogvideox_transformer3d import CogVideoXTransformer3DModel
from .cogvideox_vae import AutoencoderKLCogVideoX
from .flux_transformer2d import FluxTransformer2DModel
from .fantasytalking_transformer3d import FantasyTalkingTransformer3DModel
from .fantasytalking_audio_encoder import FantasyTalkingAudioEncoder
from .qwenimage_transformer2d import QwenImageTransformer2DModel
from .qwenimage_vae import AutoencoderKLQwenImage
from .wan_audio_encoder import WanAudioEncoder
from .wan_image_encoder import CLIPModel
from .wan_text_encoder import WanT5EncoderModel
from .wan_transformer3d import (Wan2_2Transformer3DModel, WanRMSNorm,
                                WanSelfAttention, WanTransformer3DModel)
from .wan_transformer3d_s2v import Wan2_2Transformer3DModel_S2V
from .wan_transformer3d_vace import VaceWanTransformer3DModel
from .wan_vae import AutoencoderKLWan, AutoencoderKLWan_
from .wan_vae3_8 import AutoencoderKLWan2_2_, AutoencoderKLWan3_8

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
    #   VAE Parallel Kernel
    # --------------------------------------------------------------- #
    from ..dist import parallel_magvit_vae
    AutoencoderKLWan_.decode = simple_wrapper(parallel_magvit_vae(0.4, 8)(AutoencoderKLWan_.decode))
    AutoencoderKLWan2_2_.decode = simple_wrapper(parallel_magvit_vae(0.4, 16)(AutoencoderKLWan2_2_.decode))

    # --------------------------------------------------------------- #
    #   Sparse Attention
    # --------------------------------------------------------------- #
    import torch
    from paifuser.ops import wan_sparse_attention_wrapper
    
    WanSelfAttention.forward = simple_wrapper(wan_sparse_attention_wrapper()(WanSelfAttention.forward))
    print("Import Sparse Attention")

    WanTransformer3DModel.forward = simple_wrapper(WanTransformer3DModel.forward)

    # --------------------------------------------------------------- #
    #   CFG Skip Turbo
    # --------------------------------------------------------------- #
    import os

    if importlib.util.find_spec("paifuser.accelerator") is not None:
        from paifuser.accelerator import (cfg_skip_turbo, disable_cfg_skip,
                                          enable_cfg_skip, share_cfg_skip)
    else:
        from paifuser import (cfg_skip_turbo, disable_cfg_skip,
                              enable_cfg_skip, share_cfg_skip)

    WanTransformer3DModel.enable_cfg_skip = enable_cfg_skip()(WanTransformer3DModel.enable_cfg_skip)
    WanTransformer3DModel.disable_cfg_skip = disable_cfg_skip()(WanTransformer3DModel.disable_cfg_skip)
    WanTransformer3DModel.share_cfg_skip = share_cfg_skip()(WanTransformer3DModel.share_cfg_skip)

    QwenImageTransformer2DModel.enable_cfg_skip = enable_cfg_skip()(QwenImageTransformer2DModel.enable_cfg_skip)
    QwenImageTransformer2DModel.disable_cfg_skip = disable_cfg_skip()(QwenImageTransformer2DModel.disable_cfg_skip)
    print("Import CFG Skip Turbo")

    # --------------------------------------------------------------- #
    #   RMS Norm Kernel
    # --------------------------------------------------------------- #
    from paifuser.ops import rms_norm_forward
    WanRMSNorm.forward = rms_norm_forward
    print("Import PAI RMS Fuse")

    # --------------------------------------------------------------- #
    #   Fast Rope Kernel
    # --------------------------------------------------------------- #
    import types

    import torch
    from paifuser.ops import (ENABLE_KERNEL, fast_rope_apply_qk,
                              rope_apply_real_qk)

    from . import wan_transformer3d

    def deepcopy_function(f):
        return types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__,closure=f.__closure__)

    local_rope_apply_qk = deepcopy_function(wan_transformer3d.rope_apply_qk)

    if ENABLE_KERNEL:
        def adaptive_fast_rope_apply_qk(q, k, grid_sizes, freqs):
            if torch.is_grad_enabled():
                return local_rope_apply_qk(q, k, grid_sizes, freqs)
            else:
                return fast_rope_apply_qk(q, k, grid_sizes, freqs)
    else:
        def adaptive_fast_rope_apply_qk(q, k, grid_sizes, freqs):
            return rope_apply_real_qk(q, k, grid_sizes, freqs)
            
    wan_transformer3d.rope_apply_qk = adaptive_fast_rope_apply_qk
    rope_apply_qk = adaptive_fast_rope_apply_qk
    print("Import PAI Fast rope")