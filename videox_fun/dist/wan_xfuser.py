import math

import torch
import torch.cuda.amp as amp

from .fuser import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    init_distributed_environment, initialize_model_parallel,
                    xFuserLongContextAttention)


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor

@amp.autocast(enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    dtype = x.dtype
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float32).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
        dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(dtype)

@amp.autocast(enabled=False)
@torch.compiler.disable()
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """
    Apply causal rotary positional embedding with frame offset support.
    
    This function applies RoPE with a starting frame offset, enabling causal
    inference where different frames can have different positional indices.
    
    Args:
        x: Input tensor with shape (batch, seq_len, n_channels, c*2)
        grid_sizes: Grid dimensions (f, h, w) for each sample
        freqs: Precomputed frequency parameters
        start_frame: Starting frame index for causal positioning
    
    Returns:
        Tensor with causal RoPE applied
    """
    n, c = x.size(2), x.size(3) // 2

    # Split freqs into temporal, height, and width components
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # Process each sample in the batch
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # Reshape and convert to complex numbers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        # Broadcast frequencies with start_frame offset for temporal dimension
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # Apply rotation: x * exp(i*freq)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        # Concatenate with padding tokens (if any)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # Append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

def rope_apply_qk(q, k, grid_sizes, freqs):
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)
    return q, k

def usp_attn_forward(self,
                     x,
                     seq_lens,
                     grid_sizes,
                     freqs,
                     dtype=torch.bfloat16, 
                     t=0):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q, k = rope_apply_qk(q, k, grid_sizes, freqs)

    # TODO: We should use unpaded q,k,v for attention.
    # k_lens = seq_lens // get_sequence_parallel_world_size()
    # if k_lens is not None:
    #     q = torch.cat([u[:l] for u, l in zip(q, k_lens)]).unsqueeze(0)
    #     k = torch.cat([u[:l] for u, l in zip(k, k_lens)]).unsqueeze(0)
    #     v = torch.cat([u[:l] for u, l in zip(v, k_lens)]).unsqueeze(0)

    x = xFuserLongContextAttention()(
        None,
        query=half(q),
        key=half(k),
        value=half(v),
        window_size=self.window_size)

    # TODO: padding after attention.
    # x = torch.cat([x, x.new_zeros(b, s - x.size(1), n, d)], dim=1)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x

@amp.autocast(enabled=False)
@torch.compiler.disable()
def s2v_rope_apply(x, grid_sizes, freqs):
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # loop over samples
    output = []
    for i, _ in enumerate(x):
        s = x.size(1)
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = freqs[i]
        freqs_i_rank = pad_freqs(freqs_i, s)
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])
        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

def s2v_rope_apply_qk(q, k, grid_sizes, freqs):
    q = s2v_rope_apply(q, grid_sizes, freqs)
    k = s2v_rope_apply(k, grid_sizes, freqs)
    return q, k

def usp_attn_s2v_forward(self,
                     x,
                     seq_lens,
                     grid_sizes,
                     freqs,
                     dtype=torch.bfloat16, 
                     t=0):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q, k = s2v_rope_apply_qk(q, k, grid_sizes, freqs)

    # TODO: We should use unpaded q,k,v for attention.
    # k_lens = seq_lens // get_sequence_parallel_world_size()
    # if k_lens is not None:
    #     q = torch.cat([u[:l] for u, l in zip(q, k_lens)]).unsqueeze(0)
    #     k = torch.cat([u[:l] for u, l in zip(k, k_lens)]).unsqueeze(0)
    #     v = torch.cat([u[:l] for u, l in zip(v, k_lens)]).unsqueeze(0)

    x = xFuserLongContextAttention()(
        None,
        query=half(q),
        key=half(k),
        value=half(v),
        window_size=self.window_size)

    # TODO: padding after attention.
    # x = torch.cat([x, x.new_zeros(b, s - x.size(1), n, d)], dim=1)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x

def usp_attn_self_forcing_forward(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    block_mask,
    kv_cache=None,
    current_start=0,
    cache_start=None,
    dtype=torch.bfloat16,
    t=0
):
    """
    USP attention forward for Self-Forcing with KV cache support.
    Combines sequence parallelism with causal KV cache inference.
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)
    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    
    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)
    
    if cache_start is None:
        cache_start = current_start
    
    # QKV computation
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v
    
    q, k, v = qkv_fn(x)
    
    # Inference mode with KV cache
    frame_seqlen = math.prod(grid_sizes[0][1:]).item()
    current_start_frame = current_start // frame_seqlen
    
    # Step 1: all_gather QKV to restore full sequence
    q_full = get_sp_group().all_gather(q, dim=1)  # [B, L_full, H, D]
    k_full = get_sp_group().all_gather(k, dim=1)
    v_full = get_sp_group().all_gather(v, dim=1)
    
    # Step 2: apply causal RoPE on full sequence with frame offset
    roped_query_full = causal_rope_apply(q_full, grid_sizes, freqs, 
                                         start_frame=current_start_frame).type_as(v_full)
    roped_key_full = causal_rope_apply(k_full, grid_sizes, freqs, 
                                       start_frame=current_start_frame).type_as(v_full)
    
    current_end = current_start + roped_query_full.shape[1]
    sink_tokens = self.sink_size * frame_seqlen
    kv_cache_size = kv_cache["k"].shape[1]
    num_new_tokens = roped_query_full.shape[1]
    
    # Step 3: KV cache update logic with full keys
    if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and \
       (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
        num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
        num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
        kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        local_end_index = kv_cache["local_end_index"].item() + current_end - \
            kv_cache["global_end_index"].item() - num_evicted_tokens
        local_start_index = local_end_index - num_new_tokens
        kv_cache["k"][:, local_start_index:local_end_index] = roped_key_full
        kv_cache["v"][:, local_start_index:local_end_index] = v_full
    else:
        local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
        local_start_index = local_end_index - num_new_tokens
        kv_cache["k"][:, local_start_index:local_end_index] = roped_key_full
        kv_cache["v"][:, local_start_index:local_end_index] = v_full
    
    # Step 4: chunk back to SP distribution for attention computation
    roped_query = torch.chunk(roped_query_full, sp_size, dim=1)[sp_rank]
    
    # Step 5: compute attention using xFuserLongContextAttention for sequence parallelism
    # Chunk KV cache window to match SP distribution
    kv_k_full = kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
    kv_v_full = kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
    kv_k = torch.chunk(kv_k_full, sp_size, dim=1)[sp_rank]
    kv_v = torch.chunk(kv_v_full, sp_size, dim=1)[sp_rank]
    
    x = xFuserLongContextAttention()(
        None,
        query=half(roped_query),
        key=half(kv_k),
        value=kv_v,
        window_size=self.window_size
    )
    
    kv_cache["global_end_index"].fill_(current_end)
    kv_cache["local_end_index"].fill_(local_end_index)
    
    # Output projection
    x = x.flatten(2)
    x = self.o(x)
    return x