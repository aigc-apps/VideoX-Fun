import torch
import torch.cuda.amp as amp
from einops import rearrange, repeat

from .fuser import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    init_distributed_environment, initialize_model_parallel,
                    xFuserLongContextAttention)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def usp_rope_longcatvideo_forward(self, q, k, grid_size, frame_index=None, num_ref_latents=None):
    """3D RoPE for sequence parallel.

    Args:
        query: [B, head, seq_local, head_dim]
        key: [B, head, seq_local, head_dim]
    Returns:
        query and key with the same shape as input.
    """
    # Handle both longcat (no key_name) and avatar (with key_name) versions
    if hasattr(self, 'register_grid_size'):
        import inspect
        sig = inspect.signature(self.register_grid_size)
        if 'key_name' in sig.parameters:
            # Avatar version: needs key_name
            key_name = '.'.join([str(i) for i in grid_size]) + f"-{str(frame_index)}-{str(num_ref_latents)}"
            if key_name not in self.freqs_dict:
                self.register_grid_size(grid_size, key_name, frame_index, num_ref_latents)
            freqs_cis = self.freqs_dict[key_name].to(q.device)
        else:
            # Longcat version: no key_name
            if grid_size not in self.freqs_dict:
                self.register_grid_size(grid_size)
            freqs_cis = self.freqs_dict[grid_size].to(q.device)
    else:
        # Fallback: try grid_size as key
        if grid_size not in self.freqs_dict:
            self.register_grid_size(grid_size)
        freqs_cis = self.freqs_dict[grid_size].to(q.device)
    q_, k_ = q.float(), k.float()
    freqs_cis = freqs_cis.float()
    cos, sin = freqs_cis.cos(), freqs_cis.sin()

    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    
    # q shape is [B, H, S_local, D], S_local = S_total / sp_size
    # freqs_cis shape is [S_total, D]
    # Need to extract the slice corresponding to current rank
    s_local = q.size(-2)
    cos = cos[(sp_rank * s_local):((sp_rank + 1) * s_local), :]
    sin = sin[(sp_rank * s_local):((sp_rank + 1) * s_local), :]
    
    cos, sin = rearrange(cos, 'n d -> 1 1 n d'), rearrange(sin, 'n d -> 1 1 n d')
    q_ = (q_ * cos) + (rotate_half(q_) * sin)
    k_ = (k_ * cos) + (rotate_half(k_) * sin)

    return q_.type_as(q), k_.type_as(k)


def _process_usp_attn(q, k, v, shape, dtype=torch.bfloat16):
    half_dtypes = (torch.float16, torch.bfloat16)
    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)
    q = rearrange(q, "B H S D -> B S H D")
    k = rearrange(k, "B H S D -> B S H D")
    v = rearrange(v, "B H S D -> B S H D")
    x = xFuserLongContextAttention()(
        None,
        query=half(q),
        key=half(k),
        value=half(v))
    x = rearrange(x, "B S H D -> B H S D")
    return x


def usp_attn_longcatvideo_forward(self, x: torch.Tensor, shape=None, num_cond_latents=None, return_kv=False):
    """
    Sequence parallel attention forward for LongCat-Video (non-avatar version).

    Key design:
    - xfuser's xFuserLongContextAttention handles Ring/Ulysses communication internally
    - Each rank holds [B, H, S_local, D], S_local = S_total / sp_size
    - Cond tokens: all_gather then independent computation (all ranks get same result)
    - Noise tokens: distributed attention, k,v need full context
    """
    B, N, C = x.shape
    qkv = self.qkv(x)

    qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
    qkv = qkv.view(qkv_shape).permute((2, 0, 3, 1, 4))  # [3, B, H, N, D]
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if return_kv:
        k_cache, v_cache = k.clone(), v.clone()

    # Apply RoPE (on SP-split sequence)
    q, k = self.rope_3d(q, k, shape)

    # Cond/Noise token separation mode
    if num_cond_latents is not None and num_cond_latents > 0:
        sp_world_size = get_sequence_parallel_world_size()
        sp_world_rank = get_sequence_parallel_rank()
        
        # Calculate cond token positions in global and local sequences
        # xfuser SP splits on 1D sequence (dim=1), each rank holds contiguous tokens
        #
        # Key issue: different ranks may have different number of cond tokens!
        #   - Earlier ranks may have cond tokens
        #   - Later ranks may have no cond tokens
        #
        # Solution: For cond attention, all ranks need complete cond tokens
        #   Use all_gather to collect global q,k,v, then extract complete cond part

        # First all_gather global q, k, v
        # q shape: [B, H, S_local, D], all_gather on dim=2 (sequence dimension)
        q_global = get_sp_group().all_gather(q.contiguous(), dim=2).contiguous()
        k_global = get_sp_group().all_gather(k.contiguous(), dim=2).contiguous()
        v_global = get_sp_group().all_gather(v.contiguous(), dim=2).contiguous()

        # Calculate global cond tokens count
        N_global = q_global.size(2)  # Full sequence length
        tokens_per_frame = N_global // shape[0]
        num_cond_global = num_cond_latents * tokens_per_frame

        # === Cond Tokens: Self-Attention (full sequence) ===
        # Extract cond tokens from global q,k,v
        q_cond = q_global[:, :, :num_cond_global].contiguous()
        k_cond = k_global[:, :, :num_cond_global].contiguous()
        v_cond = v_global[:, :, :num_cond_global].contiguous()

        # Cond tokens self-attention (all ranks compute same result)
        # _process_attn expects [B, H, S, D] format
        x_cond = self._process_attn(q_cond, k_cond, v_cond, shape)

        # Extract current rank's corresponding cond part
        rank_start = sp_world_rank * N
        rank_end = (sp_world_rank + 1) * N
        num_cond_local = max(0, min(num_cond_global, rank_end) - rank_start)

        if num_cond_local > 0:
            x_cond_local = x_cond[:, :, :num_cond_local].contiguous()
        else:
            x_cond_local = x_cond[:, :, :0].contiguous()

        # === Full sequence distributed Attention ===
        # All ranks uniformly compute complete attention (ensure same tensor size)
        x_full = _process_usp_attn(q, k, v, shape)

        # Replace cond part with correct cond attention result
        # Attention locality: each token's output only depends on its own q and global k,v
        # Replacing cond part doesn't affect noise part correctness
        x = x_full.clone()
        if num_cond_local > 0:
            x[:, :, :num_cond_local] = x_cond_local
    else:
        # Standard mode: directly use distributed attention
        x = _process_usp_attn(q, k, v, shape)

    x_output_shape = (B, N, C)
    x = x.transpose(1, 2)  # [B, H, N, D] -> [B, N, H, D]
    x = x.reshape(x_output_shape)  # [B, N, H, D] -> [B, N, C]
    x = self.proj(x)

    if return_kv:
        return x, (k_cache, v_cache)
    else:
        return x


def usp_attn_longcatvideo_avatar_forward(self, x: torch.Tensor, shape=None, num_cond_latents=None, return_kv=False, num_ref_latents=None, ref_img_index=None, mask_frame_range=None, ref_target_masks=None):
    """
    Sequence parallel attention forward for LongCat-Video Avatar.

    Key design:
    - xfuser's xFuserLongContextAttention handles Ring/Ulysses communication internally
    - Each rank holds [B, H, S_local, D], S_local = S_total / sp_size
    - Cond tokens: all_gather then independent computation (all ranks get same result)
    - Noise tokens: distributed attention, k,v need full context
    - Supports avatar-specific features: ref_target_masks, x_ref_attn_map
    """
    B, N, C = x.shape
    qkv = self.qkv(x)

    qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
    qkv = qkv.view(qkv_shape).permute((2, 0, 3, 1, 4))  # [3, B, H, N, D]
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if return_kv:
        k_cache, v_cache = k.clone(), v.clone()
    else:
        k_cache, v_cache = None, None

    # Apply RoPE (on SP-split sequence)
    q, k = self.rope_3d(q, k, shape, frame_index=ref_img_index, num_ref_latents=num_ref_latents)

    # Cond/Noise token separation mode
    if num_cond_latents is not None and num_cond_latents > 0:
        sp_world_size = get_sequence_parallel_world_size()
        sp_world_rank = get_sequence_parallel_rank()
        
        # Calculate cond token positions in global and local sequences
        # xfuser SP splits on 1D sequence (dim=1), each rank holds contiguous tokens
        #
        # Key issue: different ranks may have different number of cond tokens!
        #   - Earlier ranks may have cond tokens
        #   - Later ranks may have no cond tokens
        #
        # Solution: For cond attention, all ranks need complete cond tokens
        #   Use all_gather to collect global q,k,v, then extract complete cond part

        # First all_gather global q, k, v
        # q shape: [B, H, S_local, D], all_gather on dim=2 (sequence dimension)
        q_global = get_sp_group().all_gather(q.contiguous(), dim=2).contiguous()
        k_global = get_sp_group().all_gather(k.contiguous(), dim=2).contiguous()
        v_global = get_sp_group().all_gather(v.contiguous(), dim=2).contiguous()

        # Calculate global cond tokens count
        N_global = q_global.size(2)  # Full sequence length
        tokens_per_frame = N_global // shape[0]
        num_cond_global = num_cond_latents * tokens_per_frame

        # === Cond Tokens: Self-Attention (full sequence) ===
        # Extract cond tokens from global q,k,v
        q_cond = q_global[:, :, :num_cond_global].contiguous()
        k_cond = k_global[:, :, :num_cond_global].contiguous()
        v_cond = v_global[:, :, :num_cond_global].contiguous()

        # Cond tokens self-attention (all ranks compute same result)
        # _process_attn expects [B, H, S, D] format
        x_cond = self._process_attn(q_cond, k_cond, v_cond, shape)

        # Extract current rank's corresponding cond part
        rank_start = sp_world_rank * N
        rank_end = (sp_world_rank + 1) * N
        num_cond_local = max(0, min(num_cond_global, rank_end) - rank_start)

        if num_cond_local > 0:
            x_cond_local = x_cond[:, :, :num_cond_local].contiguous()
        else:
            x_cond_local = x_cond[:, :, :0].contiguous()

        # === Full sequence distributed Attention ===
        # All ranks uniformly compute complete attention (ensure same tensor size)
        x_full = _process_usp_attn(q, k, v, shape)

        # Replace cond part with correct cond attention result
        # Attention locality: each token's output only depends on its own q and global k,v
        # Replacing cond part doesn't affect noise part correctness
        x = x_full.clone()
        if num_cond_local > 0:
            x[:, :, :num_cond_local] = x_cond_local
    else:
        # Standard mode: directly use distributed attention
        x = _process_usp_attn(q, k, v, shape)

    x_output_shape = (B, N, C)
    x = x.transpose(1, 2)  # [B, H, N, D] -> [B, N, H, D]
    x = x.reshape(x_output_shape)  # [B, N, H, D] -> [B, N, C]
    x = self.proj(x)

    # Calculate attention mask for the given area in reference image
    # Note: x_ref_attn_map calculation requires full q,k which are not available in SP mode
    # For SP mode, we return None to maintain compatibility with the original interface
    x_ref_attn_map = None

    if return_kv:
        return x, (k_cache, v_cache), x_ref_attn_map
    else:
        return x, x_ref_attn_map


def usp_cross_attn_longcatvideo_forward(self, x, cond, kv_seqlen, num_cond_latents=None, shape=None):
    """
    Sequence parallel cross attention forward for LongCat-Video.

    Key design:
    - x (video tokens) is SP-split, cond (text tokens) is global
    - All ranks compute full cross attention, then zero out cond part
    - This ensures all ranks have same computation load
    """
    if num_cond_latents is None or num_cond_latents == 0:
        return self._process_cross_attn(x, cond, kv_seqlen)
    else:
        B, N, C = x.shape
        if num_cond_latents is not None and num_cond_latents > 0:
            assert shape is not None, "SHOULD pass in the shape"
            
            # In SP, need to calculate global num_cond_latents_thw
            sp_world_size = get_sequence_parallel_world_size()
            sp_world_rank = get_sequence_parallel_rank()
            
            # Global N and num_cond_latents_thw
            N_global = N * sp_world_size
            num_cond_global = num_cond_latents * (N_global // shape[0])
            
            # Current rank's sequence range
            rank_start = sp_world_rank * N
            rank_end = (sp_world_rank + 1) * N
            
            # Calculate overlap between current rank and cond region
            num_cond_local = max(0, min(num_cond_global, rank_end) - rank_start)
            
            # All ranks compute full cross attention
            output_full = self._process_cross_attn(x, cond, kv_seqlen)
            
            # Cond tokens don't participate in cross attention, zero them out
            # This ensures all ranks have same computation load
            if num_cond_local > 0:
                output_full[:, :num_cond_local, :] = 0
            
            output = output_full
        else:
            raise NotImplementedError
            
        return output

