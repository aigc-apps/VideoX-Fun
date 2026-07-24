from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from diffusers.models.embeddings import apply_rotary_emb

from .fuser import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    init_distributed_environment, initialize_model_parallel,
                    xFuserLongContextAttention)

def extract_seqlens_from_mask(attn_mask, text_seq_length):
    if attn_mask is None:
        return None
    
    if len(attn_mask.shape) == 4:
        bs, _, _, seq_len = attn_mask.shape
        
        if attn_mask.dtype == torch.bool:
            valid_mask = attn_mask.squeeze(1).squeeze(1)
        else:
            valid_mask = ~torch.isinf(attn_mask.squeeze(1).squeeze(1))
    elif len(attn_mask.shape) == 3:
        raise ValueError(
            "attn_mask should be 2D or 4D tensor, but got {}".format(
                attn_mask.shape))

    seqlens = valid_mask[:, -text_seq_length:].sum(dim=1)
    return seqlens

class HunyuanVideoMultiGPUsAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if xFuserLongContextAttention is not None:
            try:
                self.hybrid_seq_parallel_attn = xFuserLongContextAttention()
            except Exception:
                self.hybrid_seq_parallel_attn = None
        else:
            self.hybrid_seq_parallel_attn = None
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        # 5. Attention
        if encoder_hidden_states is not None:
            text_seq_length = encoder_hidden_states.size(1)

            q_lens = k_lens = extract_seqlens_from_mask(attention_mask, text_seq_length)

            img_q = query[:, :, :-text_seq_length].transpose(1, 2)
            txt_q = query[:, :, -text_seq_length:].transpose(1, 2)
            img_k = key[:, :, :-text_seq_length].transpose(1, 2)
            txt_k = key[:, :, -text_seq_length:].transpose(1, 2)
            img_v = value[:, :, :-text_seq_length].transpose(1, 2)
            txt_v = value[:, :, -text_seq_length:].transpose(1, 2)

            hidden_states = torch.zeros_like(query.transpose(1, 2))
            local_q_length = img_q.size()[1]
            for i in range(len(q_lens)):
                hidden_states[i][:local_q_length + q_lens[i]] = self.hybrid_seq_parallel_attn(
                    None,
                    img_q[i].unsqueeze(0), img_k[i].unsqueeze(0), img_v[i].unsqueeze(0), dropout_p=0.0, causal=False,
                    joint_tensor_query=txt_q[i][:q_lens[i]].unsqueeze(0),
                    joint_tensor_key=txt_k[i][:q_lens[i]].unsqueeze(0),
                    joint_tensor_value=txt_v[i][:q_lens[i]].unsqueeze(0),
                    joint_strategy='rear',
                )
        else:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            hidden_states = self.hybrid_seq_parallel_attn(
                None,
                query, key, value, dropout_p=0.0, causal=False
            )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states

