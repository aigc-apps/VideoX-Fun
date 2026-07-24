import torch
import torch.cuda.amp as amp
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention

from .fuser import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    init_distributed_environment, initialize_model_parallel,
                    xFuserLongContextAttention)


def _pad_heads(tensor, sp_world_size):
    """Pad head dimension (dim=2) to be divisible by sp_world_size.
    Input shape: [B, S, H, D] -> pad H to next multiple of sp_world_size.
    """
    n_heads = tensor.shape[2]
    remainder = n_heads % sp_world_size
    if remainder == 0:
        return tensor, 0
    pad_amount = sp_world_size - remainder
    # F.pad pads from last dim backward: (D_left, D_right, H_left, H_right)
    tensor = F.pad(tensor, (0, 0, 0, pad_amount), value=0)
    return tensor, pad_amount


class ZMultiGPUsSingleStreamAttnProcessor:
    """
    Processor for Z-Image single stream attention that adapts the existing
    Attention class to match the behavior of the original Z-ImageAttention module.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "ZSingleStreamAttnProcessor requires PyTorch 2.0. "
                "To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                return x_out.type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # From [batch, seq_len] to [batch, 1, 1, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        half_dtypes = (torch.float16, torch.bfloat16)
        def half(x):
            return x if x.dtype in half_dtypes else x.to(torch.bfloat16)

        # Pad heads to be divisible by sp_world_size (e.g., 30 -> 32 for 8 GPUs)
        sp_world_size = get_sequence_parallel_world_size()
        query, head_pad = _pad_heads(query, sp_world_size)
        key, _ = _pad_heads(key, sp_world_size)
        value, _ = _pad_heads(value, sp_world_size)

        hidden_states = xFuserLongContextAttention()(
            None,
            half(query), half(key), half(value), dropout_p=0.0, causal=False,
        )

        # Trim padded heads back to original count, then reshape
        if head_pad > 0:
            hidden_states = hidden_states[:, :, :-head_pad, :]
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output
