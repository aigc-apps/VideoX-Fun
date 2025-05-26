from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from .cogvideox_transformer3d import CogVideoXTransformer3DModel
from .cogvideox_vae import AutoencoderKLCogVideoX
from .wan_image_encoder import CLIPModel
from .wan_text_encoder import WanT5EncoderModel
from .wan_transformer3d import WanTransformer3DModel, WanSelfAttention
from .wan_vae import AutoencoderKLWan, AutoencoderKLWan_


import importlib.util

# The pai_fuser is an internally developed acceleration package, which can be used on PAI.
if importlib.util.find_spec("pai_fuser") is not None:
    from ..dist import parallel_magvit_vae
    AutoencoderKLWan_.decode = parallel_magvit_vae(0.2, 8)(AutoencoderKLWan_.decode)

    from pai_fuser.core.attention import wan_sparse_attention_wrapper
    import torch
    
    # The simple_wrapper is used to solve the problem about conflicts between cython and torch.compile
    def simple_wrapper(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner
    WanSelfAttention.forward = simple_wrapper(wan_sparse_attention_wrapper()(WanSelfAttention.forward))
    print("Import Sparse Attention")
    
    import os
    from pai_fuser.core import (cfg_skip_turbo, enable_cfg_skip, 
                                disable_cfg_skip)

    WanTransformer3DModel.enable_cfg_skip = enable_cfg_skip()(WanTransformer3DModel.enable_cfg_skip)
    WanTransformer3DModel.disable_cfg_skip = disable_cfg_skip()(WanTransformer3DModel.disable_cfg_skip)
    print("Import CFG Skip Turbo")

    from pai_fuser.core.rope import ENABLE_KERNEL, fast_rope_apply_qk

    if ENABLE_KERNEL:
        wan_transformer3d.rope_apply_qk = fast_rope_apply_qk
        rope_apply_qk = fast_rope_apply_qk
        print("Import PAI Fast rope")