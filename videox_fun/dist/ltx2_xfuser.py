from typing import Tuple

import os
import torch

from .fuser import xFuserLongContextAttention

try:
    major, minor = torch.cuda.get_device_capability(0)
    if f"{major}.{minor}" == "8.0":
        from sageattention_sm80 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif f"{major}.{minor}" == "8.6":
        from sageattention_sm86 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif f"{major}.{minor}" == "8.9":
        from sageattention_sm89 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif f"{major}.{minor}" == "9.0":
        from sageattention_sm90 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif major > 9:
        from sageattention_sm120 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
except Exception:
    try:
        from sageattention import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    except Exception:
        sageattn = None
        SAGE_ATTENTION_AVAILABLE = False


class LTX2MultiGPUsAttnProcessor:
    """
    Multi-GPU sequence parallel attention processor for LTX2.
    Uses xFuserLongContextAttention for distributed attention computation.
    """

    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                "LTX2MultiGPUsAttnProcessor requires PyTorch 2.0 or later. "
                "Please upgrade your PyTorch installation."
            )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        query_rotary_emb=None,
        key_rotary_emb=None,
        perturbation_mask=None,
        all_perturbed=None,
    ) -> torch.Tensor:
        # Get sequence parallel info from attn module
        all_gather = getattr(attn, 'all_gather', None)
        
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # Determine if this is self-attention or cross-attention
        is_self_attn = encoder_hidden_states is None
        
        if is_self_attn:
            # Self-attention: use hidden_states for both Q and KV
            encoder_hidden_states = hidden_states

        # Calculate gate logits on original hidden_states if needed
        if attn.to_gate_logits is not None:
            gate_logits = attn.to_gate_logits(hidden_states)

        # Project to Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Apply normalization
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Apply rotary embeddings if provided
        # Note: RoPE freqs are already generated for the chunked sequence in transformer forward
        if query_rotary_emb is not None:
            query = query.unflatten(2, (attn.heads, -1))
            key = key.unflatten(2, (attn.heads, -1))
            
            if attn.rope_type == "interleaved":
                # freqs shape: [B, S_chunked, D]
                cos, sin = query_rotary_emb
                
                # Apply interleaved RoPE
                query_real, query_imag = query.unflatten(3, (-1, 2)).unbind(-1)
                query_rotated = torch.stack([-query_imag, query_real], dim=-1).flatten(3)
                query = (query.float() * cos.unsqueeze(2) + query_rotated.float() * sin.unsqueeze(2)).to(query.dtype)
                
                # Same for key (fall back to query_rotary_emb for self-attention)
                cos_k, sin_k = key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                key_real, key_imag = key.unflatten(3, (-1, 2)).unbind(-1)
                key_rotated = torch.stack([-key_imag, key_real], dim=-1).flatten(3)
                key = (key.float() * cos_k.unsqueeze(2) + key_rotated.float() * sin_k.unsqueeze(2)).to(key.dtype)
                    
            elif attn.rope_type == "split":
                # Apply split RoPE using the same logic as apply_split_rotary_emb
                # x: [B, S, H, D], freqs: [B, H, S, D//2]
                cos, sin = query_rotary_emb
                
                # Save original dtype
                query_dtype = query.dtype
                
                # Reshape query to match freqs dimensions
                b, s, h, d = query.shape
                # cos is (b, h, s, d//2) -> reshape query to (b, h, s, d)
                query = query.reshape(b, s, h, -1).transpose(1, 2)  # [B, H, S, D]
                
                # Split last dim into pairs
                r = d // 2
                split_query = query.reshape(b, h, s, 2, r).float()  # [B, H, S, 2, r]
                first_x = split_query[..., :1, :]  # [B, H, S, 1, r]
                second_x = split_query[..., 1:, :]  # [B, H, S, 1, r]
                
                # Apply rotation
                cos_u = cos.unsqueeze(-2)  # [B, H, S, 1, r//2]
                sin_u = sin.unsqueeze(-2)
                
                first_out = first_x * cos_u - second_x * sin_u
                second_out = second_x * cos_u + first_x * sin_u
                
                query = torch.cat([first_out, second_out], dim=-2).reshape(b, h, s, d)
                query = query.transpose(1, 2).reshape(b, s, h, d).to(query_dtype)  # [B, S, H, D]
                
                # Same for key (fall back to query_rotary_emb for self-attention)
                cos_k, sin_k = key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                key_dtype = key.dtype
                b_k, s_k, h_k, d_k = key.shape
                key = key.reshape(b_k, s_k, h_k, -1).transpose(1, 2)  # [B, H, S, D]
                
                r_k = d_k // 2
                split_key = key.reshape(b_k, h_k, s_k, 2, r_k).float()
                first_k = split_key[..., :1, :]
                second_k = split_key[..., 1:, :]
                
                cos_k_u = cos_k.unsqueeze(-2)
                sin_k_u = sin_k.unsqueeze(-2)
                
                first_k_out = first_k * cos_k_u - second_k * sin_k_u
                second_k_out = second_k * cos_k_u + first_k * sin_k_u
                
                key = torch.cat([first_k_out, second_k_out], dim=-2).reshape(b_k, h_k, s_k, d_k)
                key = key.transpose(1, 2).reshape(b_k, s_k, h_k, d_k).to(key_dtype)
        else:
            query = query.unflatten(2, (attn.heads, -1))
            key = key.unflatten(2, (attn.heads, -1))
        
        value = value.unflatten(2, (attn.heads, -1))

        # Use xFuserLongContextAttention for distributed attention
        half_dtypes = (torch.float16, torch.bfloat16)
        def half(x):
            return x if x.dtype in half_dtypes else x.to(torch.bfloat16)

        if is_self_attn:
            # Self-attention: Q, K, V are all chunked, use xFuser for communication
            hidden_states = xFuserLongContextAttention()(
                None,
                half(query), half(key), half(value),
                dropout_p=0.0,
                causal=False,
            )
        else:
            # Video-to-audio cross-attention: Q=audio(full), K,V=video(chunked).
            # Need to all_gather K,V across ranks before attention.
            if all_gather is not None:
                key = all_gather(key.contiguous(), dim=1)
                value = all_gather(value.contiguous(), dim=1)
            
            # Regular attention with [B, S, H, D] layout
            q_attn, k_attn, v_attn = half(query), half(key), half(value)
            attention_type = os.environ.get("VIDEOX_ATTENTION_TYPE", "FLASH_ATTENTION")
            if attention_type == "SAGE_ATTENTION" and SAGE_ATTENTION_AVAILABLE:
                hidden_states = sageattn(q_attn, k_attn, v_attn, tensor_layout="NHD", is_causal=False)
            else:
                q_attn = q_attn.transpose(1, 2)
                k_attn = k_attn.transpose(1, 2)
                v_attn = v_attn.transpose(1, 2)
                hidden_states = torch.nn.functional.scaled_dot_product_attention(
                    q_attn, k_attn, v_attn, is_causal=False, dropout_p=0.0)
                hidden_states = hidden_states.transpose(1, 2).contiguous()
        
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Apply gating if present
        if attn.to_gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))  # [B, T, H, D]
            gates = 2.0 * torch.sigmoid(gate_logits)  # [B, T, H]
            hidden_states = hidden_states * gates.unsqueeze(-1)
            hidden_states = hidden_states.flatten(2, 3)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states


class LTX2PerturbedMultiGPUsAttnProcessor:
    """
    Multi-GPU sequence parallel attention processor with perturbation support for LTX2.
    """

    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                "LTX2PerturbedMultiGPUsAttnProcessor requires PyTorch 2.0 or later. "
                "Please upgrade your PyTorch installation."
            )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        query_rotary_emb=None,
        key_rotary_emb=None,
        perturbation_mask=None,
        all_perturbed=None,
    ) -> torch.Tensor:
        # Get sequence parallel info
        all_gather = getattr(attn, 'all_gather', None)
        
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        is_self_attn = encoder_hidden_states is None
        
        if is_self_attn:
            encoder_hidden_states = hidden_states

        # Calculate gate logits
        if attn.to_gate_logits is not None:
            gate_logits = attn.to_gate_logits(hidden_states)

        value = attn.to_v(encoder_hidden_states)
        
        # Check if all tokens are perturbed
        if all_perturbed is None:
            all_perturbed = torch.all(perturbation_mask == 0) if perturbation_mask is not None else False

        if all_perturbed:
            # Skip attention, use value directly
            hidden_states = value
        else:
            # Project to Q, K
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)

            # Apply normalization
            query = attn.norm_q(query)
            key = attn.norm_k(key)

            # Apply RoPE (RoPE freqs are already generated for chunked sequence)
            if query_rotary_emb is not None:
                query = query.unflatten(2, (attn.heads, -1))
                key = key.unflatten(2, (attn.heads, -1))
                
                if attn.rope_type == "interleaved":
                    cos, sin = query_rotary_emb
                    
                    query_real, query_imag = query.unflatten(3, (-1, 2)).unbind(-1)
                    query_rotated = torch.stack([-query_imag, query_real], dim=-1).flatten(3)
                    query = (query.float() * cos.unsqueeze(2) + query_rotated.float() * sin.unsqueeze(2)).to(query.dtype)
                    
                    # Fall back to query_rotary_emb for self-attention
                    cos_k, sin_k = key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                    key_real, key_imag = key.unflatten(3, (-1, 2)).unbind(-1)
                    key_rotated = torch.stack([-key_imag, key_real], dim=-1).flatten(3)
                    key = (key.float() * cos_k.unsqueeze(2) + key_rotated.float() * sin_k.unsqueeze(2)).to(key.dtype)
                elif attn.rope_type == "split":
                    # Apply split RoPE
                    cos, sin = query_rotary_emb
                    
                    # Save original dtype
                    query_dtype = query.dtype
                    
                    b, s, h, d = query.shape
                    query = query.reshape(b, s, h, -1).transpose(1, 2)  # [B, H, S, D]
                    
                    r = d // 2
                    split_query = query.reshape(b, h, s, 2, r).float()
                    first_x = split_query[..., :1, :]
                    second_x = split_query[..., 1:, :]
                    
                    cos_u = cos.unsqueeze(-2)
                    sin_u = sin.unsqueeze(-2)
                    
                    first_out = first_x * cos_u - second_x * sin_u
                    second_out = second_x * cos_u + first_x * sin_u
                    
                    query = torch.cat([first_out, second_out], dim=-2).reshape(b, h, s, d)
                    query = query.transpose(1, 2).reshape(b, s, h, d).to(query_dtype)
                    
                    # Fall back to query_rotary_emb for self-attention
                    cos_k, sin_k = key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                    key_dtype = key.dtype
                    b_k, s_k, h_k, d_k = key.shape
                    key = key.reshape(b_k, s_k, h_k, -1).transpose(1, 2)
                    
                    r_k = d_k // 2
                    split_key = key.reshape(b_k, h_k, s_k, 2, r_k).float()
                    first_k = split_key[..., :1, :]
                    second_k = split_key[..., 1:, :]
                    
                    cos_k_u = cos_k.unsqueeze(-2)
                    sin_k_u = sin_k.unsqueeze(-2)
                    
                    first_k_out = first_k * cos_k_u - second_k * sin_k_u
                    second_k_out = second_k * cos_k_u + first_k * sin_k_u
                    
                    key = torch.cat([first_k_out, second_k_out], dim=-2).reshape(b_k, h_k, s_k, d_k)
                    key = key.transpose(1, 2).reshape(b_k, s_k, h_k, d_k).to(key_dtype)
            else:
                query = query.unflatten(2, (attn.heads, -1))
                key = key.unflatten(2, (attn.heads, -1))
            
            value = value.unflatten(2, (attn.heads, -1))

            # Use xFuserLongContextAttention
            half_dtypes = (torch.float16, torch.bfloat16)
            def half(x):
                return x if x.dtype in half_dtypes else x.to(torch.bfloat16)

            if is_self_attn:
                hidden_states = xFuserLongContextAttention()(
                    None,
                    half(query), half(key), half(value),
                    dropout_p=0.0,
                    causal=False,
                )
            else:
                # Video-to-audio cross-attention: Q=audio(full), K,V=video(chunked).
                # Need to all_gather K,V across ranks before attention.
                if all_gather is not None:
                    key = all_gather(key.contiguous(), dim=1)
                    value = all_gather(value.contiguous(), dim=1)
                
                # Regular attention with [B, S, H, D] layout
                q_attn, k_attn, v_attn = half(query), half(key), half(value)
                attention_type = os.environ.get("VIDEOX_ATTENTION_TYPE", "FLASH_ATTENTION")
                if attention_type == "SAGE_ATTENTION" and SAGE_ATTENTION_AVAILABLE:
                    hidden_states = sageattn(q_attn, k_attn, v_attn, tensor_layout="NHD", is_causal=False)
                else:
                    q_attn = q_attn.transpose(1, 2)
                    k_attn = k_attn.transpose(1, 2)
                    v_attn = v_attn.transpose(1, 2)
                    hidden_states = torch.nn.functional.scaled_dot_product_attention(
                        q_attn, k_attn, v_attn, is_causal=False, dropout_p=0.0)
                    hidden_states = hidden_states.transpose(1, 2).contiguous()

            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query.dtype)

            # Apply perturbation masking
            if perturbation_mask is not None:
                value = value.flatten(2, 3)
                hidden_states = torch.lerp(value, hidden_states, perturbation_mask)

        # Apply gating
        if attn.to_gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
            gates = 2.0 * torch.sigmoid(gate_logits)
            hidden_states = hidden_states * gates.unsqueeze(-1)
            hidden_states = hidden_states.flatten(2, 3)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states
