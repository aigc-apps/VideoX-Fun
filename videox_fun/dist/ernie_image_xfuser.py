from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention

from .fuser import xFuserLongContextAttention


class ErnieImageMultiGPUsAttnProcessor:
    """
    Processor for Ernie-Image multi-GPU inference using sequence parallel attention.
    
    This processor adapts the single-stream attention mechanism to work with
    xFuserLongContextAttention for distributed inference across multiple GPUs.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "ErnieImageMultiGPUsAttnProcessor requires PyTorch 2.0. "
                "To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Step 1: QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Reshape to [batch, seq_len, heads, head_dim]
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Step 2: Apply QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Step 3: Apply rotary positional embeddings (RoPE)
        # Same rotate_half logic as ErnieImageSingleStreamAttnProcessor (rotary_interleaved=False)
        if freqs_cis is not None:
            def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
                rot_dim = freqs_cis.shape[-1]
                x, x_pass = x_in[..., :rot_dim], x_in[..., rot_dim:]
                cos_ = torch.cos(freqs_cis).to(x.dtype)
                sin_ = torch.sin(freqs_cis).to(x.dtype)
                # Non-interleaved rotate_half: [-x2, x1]
                x1, x2 = x.chunk(2, dim=-1)
                x_rotated = torch.cat((-x2, x1), dim=-1)
                return torch.cat((x * cos_ + x_rotated * sin_, x_pass), dim=-1)

            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # Step 4: Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # Step 5: Handle attention mask format conversion if needed
        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # Step 6: Perform distributed attention using xFuserLongContextAttention
        # This handles sequence parallelism automatically
        half_dtypes = (torch.float16, torch.bfloat16)
        
        def half(x):
            return x if x.dtype in half_dtypes else x.to(torch.bfloat16)

        hidden_states = xFuserLongContextAttention()(
            None,
            half(query),
            half(key),
            half(value),
            dropout_p=0.0,
            causal=False,
        )

        # Step 7: Reshape back and project output
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)
        
        output = attn.to_out[0](hidden_states)

        return output
