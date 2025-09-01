from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from .fuser import xFuserLongContextAttention


def _get_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    return _get_projections(attn, hidden_states, encoder_hidden_states)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        if sequence_dim == 2:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        elif sequence_dim == 1:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        else:
            raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, H, S, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class FluxMultiGPUsAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxMultiGPUsAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        text_seq_len: int = None,
    ) -> torch.FloatTensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)
        
        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            text_seq_len = encoder_query.shape[1]
            txt_query, txt_key, txt_value = query[:, :text_seq_len], key[:, :text_seq_len], value[:, :text_seq_len]
            img_query, img_key, img_value = query[:, text_seq_len:], key[:, text_seq_len:], value[:, text_seq_len:]
        else:
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
            txt_query, txt_key, txt_value = query[:, :text_seq_len], key[:, :text_seq_len], value[:, :text_seq_len]
            img_query, img_key, img_value = query[:, text_seq_len:], key[:, text_seq_len:], value[:, text_seq_len:]

        half_dtypes = (torch.float16, torch.bfloat16)
        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        hidden_states = xFuserLongContextAttention()(
            None,
            half(img_query), half(img_key), half(img_value), dropout_p=0.0, causal=False,
            joint_tensor_query=half(txt_query) if txt_query is not None else None,
            joint_tensor_key=half(txt_key) if txt_key is not None else None,
            joint_tensor_value=half(txt_value) if txt_value is not None else None,
            joint_strategy='front',
        )

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(img_query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states