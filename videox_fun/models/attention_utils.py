"""Borrowed from https://github.com/svg-project/Sparse-VideoGen/tree/fbc0772/svg/models/wan.
"""
import math
from math import ceil
from functools import lru_cache

import torch
import triton
import triton.language as tl
if torch.__version__ >= "2.5.0":
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
    torch._dynamo.config.cache_size_limit = 192 * 3
    torch._dynamo.config.accumulated_cache_size_limit = 192 * 3
else:
    import warnings
    
    flex_attention = None
    warnings.warn(
        f"Flex attention is not available in torch {torch.__version__}. Please upgrade to torch 2.5.0 or above.", UserWarning
    )


def prepare_flexattention(cfg_size, num_head, head_dim, dtype, device, context_length, prompt_length, num_frame, frame_size, \
    diag_width=1, multiplier=2
):
    assert diag_width == multiplier, f"{diag_width} is not equivalent to {multiplier}"
    
    seq_len = context_length + num_frame * frame_size
    query, key, value = [torch.zeros((cfg_size, num_head, seq_len, head_dim), dtype=dtype, device=device) for _ in range(3)]

    mask_mod = generate_temporal_head_mask_mod(context_length, prompt_length, num_frame, frame_size, mul=multiplier)
    block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)

    hidden_states = flex_attention(query, key, value, block_mask=block_mask)

    return block_mask


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda", _compile=False):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, _compile=_compile)
    return block_mask


def generate_temporal_head_mask_mod(context_length: int = 226, prompt_length: int = 226, num_frames: int = 13, token_per_frame: int = 1350, mul: int = 2):
    
    def round_to_multiple(idx):
        return ceil(idx / 128) * 128
        
    def temporal_mask_mod(b, h, q_idx, kv_idx):
        two_frame = round_to_multiple(mul * token_per_frame)
        temporal_head_mask = (torch.abs(q_idx - kv_idx) <= two_frame)

        # return temporal_head_mask
        first_frame_mask = (kv_idx < token_per_frame)
        video_mask = first_frame_mask | temporal_head_mask
        return video_mask
    
    return temporal_mask_mod


def generate_dense_mask_mod():
    def dense_mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= 0) # True
    return dense_mask_mod


def sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len ** 2
    
    sparsity = (sparsity * total_elements - 2 * seq_len * context_length) / total_elements
      
    width = seq_len * (1 - math.sqrt(1 - sparsity))
    width_frame = width / frame_size
    
    return width_frame


def get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size):
    allocated = torch.cuda.memory_allocated() / 1e9
    # print(colored(f"Allocated Memory: {allocated:.2f} GB", "yellow"))

    attention_mask = torch.zeros((context_length + num_frame * frame_size, context_length + num_frame * frame_size), device="cpu")

    # TODO: fix hard coded mask
    if mask_name == "spatial":
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")
        
        pixel_attn_mask[:, :frame_size] = 1 # First Frame Sink
        
        block_size, block_thres = 128, frame_size * 2
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1
        attention_mask = pixel_attn_mask
    else:
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")

        pixel_attn_mask[:, :frame_size] = 1 # First Frame Sink
        
        block_size, block_thres = 128, frame_size * 2
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

        pixel_attn_mask = pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame).permute(1, 0, 3, 2).reshape(frame_size * num_frame, frame_size * num_frame)
        attention_mask = pixel_attn_mask

    attention_mask = attention_mask[:sample_mse_max_row].cuda()
    return attention_mask


def wan_token_reorder_to_token_major(tensor, fix_len, reorder_len, reorder_num_frame, frame_size):
    """Reorder it from frame major to token major!"""
    assert reorder_len == reorder_num_frame * frame_size
    assert tensor.shape[2] == fix_len + reorder_len

    tensor[:, :, :-fix_len, :] = tensor[:, :, :-fix_len:, :].reshape(tensor.shape[0], tensor.shape[1], reorder_num_frame, frame_size, tensor.shape[3]) \
                                                         .transpose(2, 3).reshape(tensor.shape[0], tensor.shape[1], reorder_len, tensor.shape[3])
    return tensor


def wan_token_reorder_to_frame_major(tensor, fix_len, reorder_len, reorder_num_frame, frame_size):
    """Reorder it from token major to frame major!"""
    assert reorder_len == reorder_num_frame * frame_size
    assert tensor.shape[2] == fix_len + reorder_len

    tensor[:, :, :-fix_len:, :] = tensor[:, :, :-fix_len:, :].reshape(tensor.shape[0], tensor.shape[1], frame_size, reorder_num_frame, tensor.shape[3]) \
                                                         .transpose(2, 3).reshape(tensor.shape[0], tensor.shape[1], reorder_len, tensor.shape[3])
    return tensor


@triton.jit
def wan_sparse_head_placement_kernel(
    query_ptr, key_ptr, value_ptr, # [cfg, num_heads, seq_len, head_dim] seq_len = context_length + num_frame * frame_size
    query_out_ptr, key_out_ptr, value_out_ptr, # [cfg, num_heads, seq_len, head_dim]
    best_mask_idx_ptr, # [cfg, num_heads]
    query_stride_b, query_stride_h, query_stride_s, query_stride_d,
    mask_idx_stride_b, mask_idx_stride_h,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    context_length: tl.constexpr,   
    num_frame: tl.constexpr,        
    frame_size: tl.constexpr,      
    BLOCK_SIZE: tl.constexpr
):
    # Copy query, key, value to output
    # range: [b, h, block_id * block_size: block_id * block_size + block_size, :]
    cfg = tl.program_id(0)
    head = tl.program_id(1)
    block_id = tl.program_id(2)

    start_id = block_id * BLOCK_SIZE
    end_id = start_id + BLOCK_SIZE
    end_id = tl.where(end_id > seq_len, seq_len, end_id) 

    # Load best mask idx (0 is spatial, 1 is temporal)
    is_temporal = tl.load(best_mask_idx_ptr + cfg * mask_idx_stride_b + head * mask_idx_stride_h)
    
    offset_token = tl.arange(0, BLOCK_SIZE) + start_id
    offset_mask = offset_token < seq_len
    offset_d = tl.arange(0, head_dim)

    if is_temporal:
        frame_id = offset_token // frame_size
        patch_id = offset_token - frame_id * frame_size
        offset_store_token = tl.where(offset_token >= seq_len - context_length, offset_token, patch_id * num_frame + frame_id)

        offset_load = (cfg * query_stride_b + head * query_stride_h + offset_token[:,None] * query_stride_s) + offset_d[None,:] * query_stride_d
        offset_query = query_ptr + offset_load
        offset_key = key_ptr + offset_load
        offset_value = value_ptr + offset_load

        offset_store = (cfg * query_stride_b + head * query_stride_h + offset_store_token[:,None] * query_stride_s) + offset_d[None,:] * query_stride_d
        offset_query_out = query_out_ptr + offset_store
        offset_key_out = key_out_ptr + offset_store
        offset_value_out = value_out_ptr + offset_store

        # Maybe tune the pipeline here
        query = tl.load(offset_query, mask=offset_mask[:,None])
        tl.store(offset_query_out, query, mask=offset_mask[:,None])
        key = tl.load(offset_key, mask=offset_mask[:,None])
        tl.store(offset_key_out, key, mask=offset_mask[:,None])
        value = tl.load(offset_value, mask=offset_mask[:,None])
        tl.store(offset_value_out, value, mask=offset_mask[:,None])


    else:
        offset_load = (cfg * query_stride_b + head * query_stride_h + offset_token[:,None] * query_stride_s) + offset_d[None,:] * query_stride_d
        offset_query = query_ptr + offset_load
        offset_key = key_ptr + offset_load
        offset_value = value_ptr + offset_load

        offset_store = offset_load
        offset_query_out = query_out_ptr + offset_store
        offset_key_out = key_out_ptr + offset_store
        offset_value_out = value_out_ptr + offset_store

        # Maybe tune the pipeline here
        query = tl.load(offset_query, mask=offset_mask[:,None])
        tl.store(offset_query_out, query, mask=offset_mask[:,None])
        key = tl.load(offset_key, mask=offset_mask[:,None])
        tl.store(offset_key_out, key, mask=offset_mask[:,None])
        value = tl.load(offset_value, mask=offset_mask[:,None])
        tl.store(offset_value_out, value, mask=offset_mask[:,None])


def wan_sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size):
    cfg, num_heads, seq_len, head_dim = query.shape
    BLOCK_SIZE = 128
    assert seq_len == context_length + num_frame * frame_size

    grid = (cfg, num_heads, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    wan_sparse_head_placement_kernel[grid](
        query, key, value, 
        query_out, key_out, value_out, 
        best_mask_idx,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        best_mask_idx.stride(0), best_mask_idx.stride(1),
        seq_len, head_dim, context_length, num_frame, frame_size, 
        BLOCK_SIZE
    )


@triton.jit
def wan_hidden_states_placement_kernel(
    hidden_states_ptr, # [cfg, num_heads, seq_len, head_dim] seq_len = context_length + num_frame * frame_size
    hidden_states_out_ptr, # [cfg, num_heads, seq_len, head_dim]
    best_mask_idx_ptr, # [cfg, num_heads]
    hidden_states_stride_b, hidden_states_stride_h, hidden_states_stride_s, hidden_states_stride_d,
    mask_idx_stride_b, mask_idx_stride_h,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    context_length: tl.constexpr,   
    num_frame: tl.constexpr,        
    frame_size: tl.constexpr,      
    BLOCK_SIZE: tl.constexpr
):
    # Copy hidden_states to output
    # range: [b, h, block_id * block_size: block_id * block_size + block_size, :]
    cfg = tl.program_id(0)
    head = tl.program_id(1)
    block_id = tl.program_id(2)

    start_id = block_id * BLOCK_SIZE
    end_id = start_id + BLOCK_SIZE
    end_id = tl.where(end_id > seq_len, seq_len, end_id) 

    # Load best mask idx (0 is spatial, 1 is temporal)
    is_temporal = tl.load(best_mask_idx_ptr + cfg * mask_idx_stride_b + head * mask_idx_stride_h)
    
    offset_token = tl.arange(0, BLOCK_SIZE) + start_id
    offset_mask = offset_token < seq_len
    offset_d = tl.arange(0, head_dim)

    if is_temporal:
        patch_id = offset_token // num_frame
        frame_id = offset_token - patch_id * num_frame
        offset_store_token = tl.where(offset_token >= seq_len - context_length, offset_token, frame_id * frame_size + patch_id)

        offset_load = (cfg * hidden_states_stride_b + head * hidden_states_stride_h + offset_token[:,None] * hidden_states_stride_s) + offset_d[None,:] * hidden_states_stride_d
        offset_hidden_states = hidden_states_ptr + offset_load

        offset_store = (cfg * hidden_states_stride_b + head * hidden_states_stride_h + offset_store_token[:,None] * hidden_states_stride_s) + offset_d[None,:] * hidden_states_stride_d
        offset_hidden_states_out = hidden_states_out_ptr + offset_store

        # Maybe tune the pipeline here
        hidden_states = tl.load(offset_hidden_states, mask=offset_mask[:,None])
        tl.store(offset_hidden_states_out, hidden_states, mask=offset_mask[:,None])
    else:
        offset_load = (cfg * hidden_states_stride_b + head * hidden_states_stride_h + offset_token[:,None] * hidden_states_stride_s) + offset_d[None,:] * hidden_states_stride_d
        offset_hidden_states = hidden_states_ptr + offset_load

        offset_store = offset_load
        offset_hidden_states_out = hidden_states_out_ptr + offset_store

        # Maybe tune the pipeline here
        hidden_states = tl.load(offset_hidden_states, mask=offset_mask[:,None])
        tl.store(offset_hidden_states_out, hidden_states, mask=offset_mask[:,None])


def wan_hidden_states_placement(hidden_states, hidden_states_out, best_mask_idx, context_length, num_frame, frame_size):
    cfg, num_heads, seq_len, head_dim = hidden_states.shape
    BLOCK_SIZE = 128
    assert seq_len == context_length + num_frame * frame_size

    grid = (cfg, num_heads, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)


    wan_hidden_states_placement_kernel[grid](
        hidden_states, 
        hidden_states_out, 
        best_mask_idx,
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3),
        best_mask_idx.stride(0), best_mask_idx.stride(1),
        seq_len, head_dim, context_length, num_frame, frame_size, 
        BLOCK_SIZE
    )

    return hidden_states_out
