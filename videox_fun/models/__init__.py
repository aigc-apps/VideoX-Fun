import importlib.util

from diffusers import AutoencoderKL
from transformers import (AutoTokenizer, CLIPImageProcessor, CLIPTextModel,
                          CLIPTokenizer, CLIPVisionModelWithProjection,
                          T5EncoderModel, T5Tokenizer, T5TokenizerFast)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
except:
    Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer = None, None
    print("Your transformers version is too old to load Qwen2_5_VLForConditionalGeneration and Qwen2Tokenizer. If you wish to use QwenImage, please upgrade your transformers package to the latest version.")

from .cogvideox_transformer3d import CogVideoXTransformer3DModel
from .cogvideox_vae import AutoencoderKLCogVideoX
from .flux_transformer2d import FluxTransformer2DModel
from .qwenimage_transformer2d import QwenImageTransformer2DModel
from .qwenimage_vae import AutoencoderKLQwenImage
from .wan_image_encoder import CLIPModel
from .wan_text_encoder import WanT5EncoderModel
from .wan_transformer3d import (Wan2_2Transformer3DModel, WanRMSNorm,
                                WanSelfAttention, WanTransformer3DModel)
from .wan_vae import AutoencoderKLWan, AutoencoderKLWan_
from .wan_vae3_8 import AutoencoderKLWan2_2_, AutoencoderKLWan3_8

# The pai_fuser is an internally developed acceleration package, which can be used on PAI.
if importlib.util.find_spec("pai_fuser") is not None:
    # The simple_wrapper is used to solve the problem about conflicts between cython and torch.compile
    def simple_wrapper(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner

    from ..dist import parallel_magvit_vae
    AutoencoderKLWan_.decode = simple_wrapper(parallel_magvit_vae(0.4, 8)(AutoencoderKLWan_.decode))
    AutoencoderKLWan2_2_.decode = simple_wrapper(parallel_magvit_vae(0.4, 16)(AutoencoderKLWan2_2_.decode))

    import torch
    from pai_fuser.core.attention import wan_sparse_attention_wrapper
    
    WanSelfAttention.forward = simple_wrapper(wan_sparse_attention_wrapper()(WanSelfAttention.forward))
    print("Import Sparse Attention")

    WanTransformer3DModel.forward = simple_wrapper(WanTransformer3DModel.forward)

    import os
    from pai_fuser.core import (cfg_skip_turbo, disable_cfg_skip,
                                enable_cfg_skip)

    WanTransformer3DModel.enable_cfg_skip = enable_cfg_skip()(WanTransformer3DModel.enable_cfg_skip)
    WanTransformer3DModel.disable_cfg_skip = disable_cfg_skip()(WanTransformer3DModel.disable_cfg_skip)
    print("Import CFG Skip Turbo")

    from pai_fuser.core.rope import ENABLE_KERNEL, fast_rope_apply_qk

    if ENABLE_KERNEL:
        import types
        from . import wan_transformer3d

        def deepcopy_function(f):
            return types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__,closure=f.__closure__)

        local_rope_apply_qk = deepcopy_function(wan_transformer3d.rope_apply_qk)
        def adaptive_fast_rope_apply_qk(q, k, grid_sizes, freqs):
            if torch.is_grad_enabled():
                return local_rope_apply_qk(q, k, grid_sizes, freqs)
            else:
                return fast_rope_apply_qk(q, k, grid_sizes, freqs)
            
        wan_transformer3d.rope_apply_qk = adaptive_fast_rope_apply_qk
        rope_apply_qk = adaptive_fast_rope_apply_qk
        print("Import PAI Fast rope")
