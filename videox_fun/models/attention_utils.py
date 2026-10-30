import functools
import math
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

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


from .attention_kernel import (_sparse_linear_attention, get_block_map,
                               mean_pool, sparse_linear_attention_varlen)


def convert_qkv_dtype(q, k, v):
    try:
        """Unify the dtype of q, k, v tensors"""
        dtypes = {q.dtype, k.dtype, v.dtype}
        
        # If any tensor is float16/bfloat16
        if torch.float16 in dtypes or torch.bfloat16 in dtypes:
            target_dtype = torch.bfloat16 if torch.bfloat16 in dtypes else torch.float16
        # If all tensors are float32
        elif dtypes == {torch.float32}:
            target_dtype = torch.bfloat16 if (torch.cuda.is_available() and 
                                            torch.cuda.get_device_capability()[0] >= 8) else torch.float16
        else:
            return q, k, v  # No conversion for other cases
        
        return q.to(target_dtype), k.to(target_dtype), v.to(target_dtype)
    except Exception:
        return q, k, v


def _convert_attn_mask_to_lens(attn_mask):
    """
    Convert attention mask to sequence lengths for Flash Attention.
    
    Args:
        attn_mask: Attention mask, can be:
            - [B, L] with 1=valid, 0=padding
            - [B, 1, L] or [B, 1, 1, L] attention bias with 0=valid, -inf/-10000=padding
            - [B, H, Lq, Lk] full attention mask
    
    Returns:
        k_lens: [B] tensor of valid sequence lengths, or None if not a simple padding mask
    """
    if attn_mask is None:
        return None
    
    # Squeeze to simplest form
    while attn_mask.ndim > 2 and attn_mask.shape[1] == 1:
        attn_mask = attn_mask.squeeze(1)
    
    # Only handle [B, L] case (simple padding mask)
    if attn_mask.ndim != 2:
        return None
    
    # Check if it's attention bias format (0 and -inf/-10000) or binary mask (0/1)
    unique_vals = torch.unique(attn_mask)
    if len(unique_vals) > 2:
        return None  # Complex mask, can't convert
    
    # Determine which value means "valid"
    max_val = unique_vals.max().item()
    
    if max_val <= 0:  # Attention bias format: 0=valid, negative=padding
        valid_mask = (attn_mask >= -1.0)  # 0 is valid
    else:  # Binary format: 1=valid, 0=padding
        valid_mask = (attn_mask > 0.5)
    
    # Check if it's a simple left-padded or right-padded mask
    # For right-padding: [1,1,1,0,0] -> valid tokens are contiguous from start
    k_lens = valid_mask.sum(dim=-1).to(torch.int32)
    
    # Verify it's actually a contiguous padding mask by reconstruction
    B, L = valid_mask.shape
    reconstructed = torch.arange(L, device=valid_mask.device).unsqueeze(0) < k_lens.unsqueeze(1)
    if not torch.all(reconstructed == valid_mask):
        return None  # Not a simple contiguous padding mask
    
    return k_lens


def flash_attention_naive(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
):
    # apply attention
    if FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )[0]
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )

    # output
    return x.type(q.dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)
        if isinstance(x, (tuple, list)):
            x = x[0].unflatten(0, (b, lq))
        else:
            x = x.unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    attention_type=None,
    attn_mask=None,
):
    attention_type = os.environ.get("VIDEOX_ATTENTION_TYPE", "FLASH_ATTENTION") if attention_type is None else attention_type
    if fa_version is None and os.environ.get("VIDEOX_FA_VERSION") is not None:
        fa_version = int(os.environ["VIDEOX_FA_VERSION"])
    if torch.is_grad_enabled() and attention_type == "SAGE_ATTENTION":
        attention_type = "FLASH_ATTENTION"

    # Convert attn_mask to k_lens for Flash Attention if possible
    # Note: flash_attention doesn't support variable-length query, only set k_lens
    if attn_mask is not None and k_lens is None and attention_type == "FLASH_ATTENTION":
        converted_lens = _convert_attn_mask_to_lens(attn_mask)
        if converted_lens is not None:
            k_lens = converted_lens
            attn_mask = None  # Successfully converted, clear the mask
        else:
            # Conversion failed, fallback to SDPA which supports attn_mask
            attention_type = "SDPA"

    if attention_type == "SAGE_ATTENTION" and SAGE_ATTENTION_AVAILABLE:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using SAGE_ATTENTION. It can have a significant impact on performance.'
            )

        q, k, v = convert_qkv_dtype(q, k, v)
        out = sageattn(
            q, k, v, attn_mask=attn_mask, tensor_layout="NHD", is_causal=causal, dropout_p=dropout_p)

    elif attention_type == "FLASH_ATTENTION" and (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
    return out


class SparseLinearAttention(nn.Module):
    # Modified from https://github.com/thu-ml/TurboDiffusion/blob/main/turbodiffusion/SLA/kernel.py
    def __init__(self, head_dim, topk, feature_map='softmax', BLKQ=64, BLKK=64, use_bf16=True, tie_feature_map_qk=True):
        R'''
        Args:
            head_dim: dimension of each head.
            topk: ratio of keys selected for sparse attention, shared across all queries.
            feature_map: feature map for linear attention, one of ['hedgehog', 'elu', 'relu', 'softmax'].
            BLKQ: block size for query.
            BLKK: block size for key.
            use_bf16: whether to use bfloat16 (default) or float16 for computation. The conversion to bf16/fp16 is done inside the module.
            tie_feature_map_qk: whether to use the same feature map for query and key.
        '''
        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.float32)

        if feature_map == 'elu':
            def elu_feature_map(x):
                return torch.nn.functional.elu(x) + 1
            self.feature_map_q = elu_feature_map
            self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            def softmax_feature_map(x):
                return torch.nn.functional.softmax(x, dim=-1)
            self.feature_map_q = softmax_feature_map
            self.feature_map_k = softmax_feature_map
        else:
            raise NotImplementedError(f'Not supported feature map {feature_map}.')

        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q

        self.init_weights_()

    def init_weights_(self):
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)

    def forward(self, q, k, v, return_sparsity=False):
        R'''
        Args:
            q: queries  [B, L, H, D]
            k: keys     [B, L, H, D].
            v: values   [B, L, H, D].
            return_sparsity: whether to return the actual sparsity.
        '''
        dtype = q.dtype
        
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)
        o_s = _sparse_linear_attention.apply(q, k, v, sparse_map, lut, real_topk, self.BLKQ, self.BLKK)
        
        q = self.feature_map_q(q).contiguous().to(self.dtype) # c_q
        k = self.feature_map_k(k).contiguous().to(self.dtype) # c_k

        def calc_linear(q, k, v):
            kvsum = k.transpose(-1, -2) @ v
            ksum = torch.sum(k, dim=-2, keepdim=True)
            return (q @ kvsum) / (1e-5 + (q * ksum).sum(dim=-1, keepdim=True))
        o_l = calc_linear(q, k, v)

        with torch.amp.autocast('cuda', dtype=self.dtype):
            o_l = self.proj_l(o_l)
        o = (o_s + o_l).to(dtype).transpose(1, 2)

        if return_sparsity:
            return o, real_topk / sparse_map.shape[-1]
        else:
            return o


class RoLASparseAttention(nn.Module):
    R'''
    RoLa sparse + low-rank self-attention branch (the official SparkDiffusion default).

    Mirrors ``sparkdiffusion/networks/wan_rola_attention.py::WanSelfAttentionRoLa``:
      - sparse branch   : block-sparse exact softmax via the in-repo SLA Triton kernel
                          (same ``get_block_map`` + ``_sparse_linear_attention`` used by
                          ``SparseLinearAttention``; runs on post-RoPE q/k).
      - low-rank branch : ``proj_q``/``proj_k`` (head_dim -> rank) -> SiLU -> rank-truncated
                          3D RoPE -> linear attention ``Q_lr (K_lr^T V)`` -> RMSNorm
                          (runs on PRE-RoPE, qk-normed q/k; RoPE is applied OUTSIDE the
                          non-linear feature map, on the low-rank bottleneck).
      - token gate      : ``sigmoid(gate_proj(x) + gate_bias)`` scales the low-rank output
                          magnitude per token/head; ``gate`` is computed from the raw hidden
                          states ``x`` (not q/k/v).

    Fusion (per official)::

        scale = sqrt(mean(o_s^2) + 1e-5)          # per-token/head RMS of the sparse output
        out   = (o_s / scale + gate * o_l) * scale # == o_s + gate * o_l * scale

    New learnable params per layer (all under ``attn_op.*``): ``proj_q``, ``proj_k``,
    ``gate_proj``, ``gate_bias``. They are absent from a dense checkpoint (loaded with
    strict=False) and start at their init below. This replaces the earlier PureSLA path
    (single zero-init ``proj_l`` + softmax feature-map linear attention).

    All forward helpers take/return BHLD (``[B, H, L, *]``) tensors.
    '''
    def __init__(self, head_dim, num_heads, topk=0.03, rola_rank=64,
                 BLKQ=64, BLKK=64, use_bf16=True):
        R'''
        Args:
            head_dim: dimension of each head.
            num_heads: number of attention heads (sizes ``gate_bias``).
            topk: ratio of key blocks kept for the sparse branch (0.1 -> 90% sparse).
            rola_rank: low-rank bottleneck dimension of the linear branch.
            BLKQ/BLKK: query/key block sizes for the sparse kernel.
            use_bf16: compute the sparse kernel in bfloat16 (default) or float16.
        '''
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.topk = topk
        self.rola_rank = rola_rank
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16

        r = rola_rank
        self.proj_q = nn.Linear(head_dim, r, bias=False)
        self.proj_k = nn.Linear(head_dim, r, bias=False)
        self.gate_proj = nn.Linear(head_dim, 1, bias=False)
        # -1.946 -> sigmoid ~= 0.125, so the sparse branch dominates at init.
        self.gate_bias = nn.Parameter(torch.full((1, num_heads, 1, 1), -1.946))

        self.init_weights_()

    def init_weights_(self):
        nn.init.kaiming_uniform_(self.proj_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.proj_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate_proj.weight)
        with torch.no_grad():
            self.gate_bias.fill_(-1.946)

    def sparse(self, q, k, v):
        R'''Block-sparse exact softmax attention.

        Args:
            q, k: [B, H, L, D] POST-RoPE queries/keys.
            v:    [B, H, L, D] values.
        Returns:
            [B, H, L, D], or ``None`` when the sequence is shorter than one block
            (the caller then falls back to ``gate * o_l``).
        '''
        L = q.shape[2]
        if L < max(self.BLKQ, self.BLKK):
            return None
        orig_dtype = q.dtype
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        sparse_map, lut, real_topk = get_block_map(
            q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)
        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)
        o_s = _sparse_linear_attention.apply(
            q, k, v, sparse_map, lut, real_topk, self.BLKQ, self.BLKK)
        return o_s.to(orig_dtype)

    def lowrank(self, q_normed, k_normed, v, grid_sizes, freqs):
        R'''Low-rank linear attention branch with rank-truncated RoPE.

        Args:
            q_normed, k_normed: [B, H, L, D] PRE-RoPE, qk-normed queries/keys.
            v:                  [B, H, L, D] values.
            grid_sizes:         [B, 3] (F, H, W) for the rank RoPE expansion.
            freqs:              complex rotary table [M, head_dim // 2].
        Returns:
            [B, H, L, D], RMS-normalized.
        '''
        from .spark_transformer3d import rope_apply_lowrank

        q_lr = F.silu(self.proj_q(q_normed))  # [B, H, L, rank]
        k_lr = F.silu(self.proj_k(k_normed))
        # RoPE applied outside the non-linear feature map, on the rank bottleneck.
        q_lr = rope_apply_lowrank(q_lr, grid_sizes, freqs, self.rola_rank)
        k_lr = rope_apply_lowrank(k_lr, grid_sizes, freqs, self.rola_rank)
        c = k_lr.transpose(-2, -1) @ v        # [B, H, rank, D]
        out = q_lr @ c                        # [B, H, L, D]
        return F.rms_norm(out, [out.shape[-1]])

    def gate(self, x_bhld, head_slice=None):
        R'''Token-wise gate from the raw hidden states.

        Args:
            x_bhld:    [B, H, L, D] hidden states (attention input), head layout.
            head_slice: optional slice selecting this rank's head shard (multi-GPU
                        Ulysses head-parallel), so the per-head ``gate_bias`` lines up.
        Returns:
            [B, H, L, 1] gate in (0, 1).
        '''
        gb = self.gate_bias if head_slice is None else self.gate_bias[:, head_slice]
        return torch.sigmoid(self.gate_proj(x_bhld) + gb)


# Logical tile geometries keyed by tile volume -- the official VSA_H3_TILE_SHAPES. 64 is what the
# official defaults actually select for the checkpoints this repo runs: the Wan backend's
# VSA_TILE_SIZE is (4,4,4), and both released FastH3 inference contracts pin "vsa_tile_size": 64.
# (256/(4,8,8) is the H3 backend's *code* default, aimed at the Blackwell CuTe fastpath.) 128 and
# 256 stay selectable and reach the same 64-block kernel through the route-A expansion below.
_H3_TILE_SHAPES = {64: (4, 4, 4), 128: (4, 4, 8), 256: (4, 8, 8)}
_H3_TILE_ELEMS = 64
# Physical block size the Triton kernels are instantiated at: a logical tile of N elems is N // 64
# CONTIGUOUS 64-blocks in the padded tile buffer.
_H3_KERNEL_BLOCK = 64


def _h3_compute_topk(sparsity, num_blocks):
    """Key tiles to KEEP for a sparsity level, clamped to [1, num_blocks]."""
    return max(1, min(math.ceil((1.0 - sparsity) * num_blocks), num_blocks))


@functools.lru_cache(maxsize=32)
def _h3_tile_partition_indices(grid, tile_shape, device):
    """Region tile-order -> original flat region index (port of get_tile_partition_indices)."""
    T, H, W = grid
    ts, hs, ws = tile_shape
    indices = torch.arange(T * H * W, device=device, dtype=torch.long).reshape(T, H, W)
    parts = []
    for t in range(math.ceil(T / ts)):
        for h in range(math.ceil(H / hs)):
            for w in range(math.ceil(W / ws)):
                parts.append(indices[t * ts:min(t * ts + ts, T),
                                     h * hs:min(h * hs + hs, H),
                                     w * ws:min(w * ws + ws, W)].flatten())
    return torch.cat(parts, dim=0)


@functools.lru_cache(maxsize=32)
def _h3_variable_block_sizes(grid, num_tiles, device, tile_shape):
    """Valid (non-pad) token count inside every 3-D tile (port of construct_variable_block_sizes)."""
    t, h, w = grid
    ts_t, ts_h, ts_w = tile_shape
    n_t, n_h, n_w = num_tiles

    def _sizes(dim_len, tile, n):
        sizes = torch.full((n,), tile, dtype=torch.long, device=device)
        rem = dim_len - (n - 1) * tile
        sizes[-1] = rem if rem > 0 else tile
        return sizes

    t_s = _sizes(t, ts_t, n_t)
    h_s = _sizes(h, ts_h, n_h)
    w_s = _sizes(w, ts_w, n_w)
    return (t_s[:, None, None] * h_s[None, :, None] * w_s[None, None, :]).reshape(-1)


def _h3_non_pad_index(variable_block_sizes, tile_elems):
    """Padded-buffer slot index of every real row, in tile order (port of get_non_pad_index)."""
    n_win = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts = torch.arange(n_win, device=device) * tile_elems
    index_pad = starts[:, None] + torch.arange(tile_elems, device=device)[None, :]
    index_mask = torch.arange(tile_elems, device=device)[None, :] < variable_block_sizes[:, None]
    return index_pad[index_mask]


def _h3_validate_segment_geometry(segments, variable_block_sizes, untile_combined_index, tile_elems):
    R'''Fail synchronously on out-of-bounds tile geometry (port of the official guard).

    Invariants the block-sparse kernels trust without checking: every tile's valid size is in
    (0, tile_elems]; the sizes sum to the packed sequence length; and ``untile_combined_index``
    maps each packed row to exactly one non-pad slot of the padded tile buffer. A violation would
    otherwise surface only as an async device fault at some later kernel or collective (e.g. an
    FSDP all-gather), which is unattributable -- so raise here, once per cached geometry.
    '''
    total = sum(math.prod(seg) if isinstance(seg, tuple) else seg for seg in segments)
    n_pad = variable_block_sizes.numel() * tile_elems
    sizes_min = int(variable_block_sizes.min())
    sizes_max = int(variable_block_sizes.max())
    sizes_sum = int(variable_block_sizes.sum())
    if sizes_min < 1 or sizes_max > tile_elems or sizes_sum != total:
        raise ValueError(
            f"VSA-H3 tile sizes out of bounds for segments={segments}, tile_elems={tile_elems}: "
            f"min={sizes_min}, max={sizes_max}, sum={sizes_sum}, expected sum={total}.")
    if untile_combined_index.numel() != total:
        raise ValueError(
            f"VSA-H3 untile index has {untile_combined_index.numel()} entries for a packed "
            f"sequence of {total} rows (segments={segments}).")
    idx_min = int(untile_combined_index.min())
    idx_max = int(untile_combined_index.max())
    if idx_min < 0 or idx_max >= n_pad:
        # Range first: the pad-slot gather below would itself index out of bounds (the very async
        # fault this guard exists to preempt).
        raise ValueError(
            f"VSA-H3 untile index is not an injective map into non-pad slots: range "
            f"[{idx_min}, {idx_max}] vs padded length {n_pad} (segments={segments}).")
    in_tile_offset = untile_combined_index % tile_elems
    maps_into_pad = bool(
        (in_tile_offset >= variable_block_sizes[untile_combined_index // tile_elems]).any())
    if maps_into_pad or int(torch.unique(untile_combined_index).numel()) != total:
        raise ValueError(
            f"VSA-H3 untile index is not an injective map into non-pad slots: "
            f"pad-slot hit={maps_into_pad} (segments={segments}).")


@functools.lru_cache(maxsize=32)
def build_h3_tile_geometry(segments, device, tile_shape=_H3_TILE_SHAPES[_H3_TILE_ELEMS]):
    R'''Tile a packed sequence of interleaved dense segments and video regions.

    Port of the official ``_h3_segment_tile_geometry``. ``segments`` lists, in packed
    order, dense segment row counts (``int``) and video-region token grids (``(t,h,w)``
    triples). Returns ``(tile_partition_indices, variable_block_sizes,
    untile_combined_index, num_prefix_tiles, num_video_tiles, video_tile_spans,
    tile_elems)`` where ``video_tile_spans`` holds one ``[start, end)`` tile range per
    video region (prefix tiles included in the numbering).
    '''
    tile_elems = math.prod(tile_shape)
    ts_t, ts_h, ts_w = tile_shape
    prefix_sizes, prefix_rows, video_entries = [], [], []
    cursor = 0
    for seg in segments:
        if isinstance(seg, tuple):
            video_entries.append((cursor, seg))
            cursor += math.prod(seg)
            continue
        if seg <= 0:
            continue
        full, rem = divmod(seg, tile_elems)
        prefix_sizes.extend([tile_elems] * full)
        if rem:
            prefix_sizes.append(rem)
        prefix_rows.append(torch.arange(cursor, cursor + seg, device=device, dtype=torch.long))
        cursor += seg
    num_prefix_tiles = len(prefix_sizes)

    video_index_parts, video_size_parts, video_tile_spans = [], [], []
    tile_cursor = num_prefix_tiles
    for start, grid in video_entries:
        t, h, w = grid
        num_tiles = (math.ceil(t / ts_t), math.ceil(h / ts_h), math.ceil(w / ts_w))
        region_sizes = _h3_variable_block_sizes(grid, num_tiles, device, tile_shape)
        video_index_parts.append(_h3_tile_partition_indices(grid, tile_shape, device) + start)
        video_size_parts.append(region_sizes)
        video_tile_spans.append((tile_cursor, tile_cursor + int(region_sizes.numel())))
        tile_cursor += int(region_sizes.numel())
    num_video_tiles = tile_cursor - num_prefix_tiles

    tile_partition_indices = torch.cat(prefix_rows + video_index_parts)
    variable_block_sizes = torch.cat(
        [torch.tensor(prefix_sizes, dtype=torch.long, device=device)] + video_size_parts)
    non_pad_index = _h3_non_pad_index(variable_block_sizes, tile_elems)
    untile_combined_index = non_pad_index[torch.argsort(tile_partition_indices)]
    # One-time (lru-cached) synchronous bounds check; see _h3_validate_segment_geometry.
    _h3_validate_segment_geometry(segments, variable_block_sizes, untile_combined_index, tile_elems)
    return (tile_partition_indices, variable_block_sizes, untile_combined_index,
            num_prefix_tiles, num_video_tiles, tuple(video_tile_spans), tile_elems)


# Per-process reusable padded tile scratch for forward_h3's no-grad (inference) path.
# Mirrors the official FastVideo builder-owned tile_buf_holder: one buffer per
# (role, shape, dtype, device) shared across all layers and steps, so the per-layer
# torch.zeros((B,H,L_buf,D)) allocation+zeroing churn is paid once instead of every call.
# Keyed by role because q/k/v/gate are all live simultaneously within one forward_h3.
_H3_TILE_BUFS = {}


def _h3_tile_into_buf(t, dtype, idx, shape, device, role, reuse):
    R'''Scatter packed [B,L,H,D] ``t`` into a [B,H,L_buf,D] tile buffer (pad/dummy stay zero).

    Under ``reuse`` (no-grad inference) the buffer is cached in ``_H3_TILE_BUFS`` and only
    re-zeroed when the tile geometry (``idx`` identity) changes, exactly like the official
    holder: pad slots are never written and non-pad slots are fully overwritten per call, so
    a reused buffer stays valid. Callers consume the result before the next layer overwrites
    it. Under grad the buffer is fresh -- an in-place write into a buffer that outlives the
    step would anchor that step's autograd graph (the official tile-cache OOM).
    '''
    B, H, L_buf, D = shape
    if not reuse:
        buf = torch.zeros(shape, device=device, dtype=dtype)
        buf[:, :, idx, :] = t.transpose(1, 2).to(dtype)
        return buf
    key = (role, B, H, L_buf, D, dtype, str(device))
    ent = _H3_TILE_BUFS.get(key)
    if ent is None or ent[0].shape != shape or ent[0].dtype != dtype:
        ent = [torch.zeros(shape, device=device, dtype=dtype), None]
        _H3_TILE_BUFS[key] = ent
    if ent[1] is not idx:          # geometry identity changed -> clear stale valid rows once
        ent[0].zero_()
        ent[1] = idx
    ent[0][:, :, idx, :] = t.transpose(1, 2).to(dtype)
    return ent[0]


def _h3_pool_tiles(x, variable_block_sizes, tile_elems, n_tiles):
    R'''fp32 masked mean over each tile. ``x``: [B,H,L_buf,D] (pad slots zero) -> [B,H,n_tiles,D].

    Pad slots are guaranteed zero (zeros-init buffer, only non-pad slots written), so a
    plain fp32 sum divided by the true tile size is exactly the masked mean.
    '''
    B, H, _, D = x.shape
    xt = x[:, :, :n_tiles * tile_elems].reshape(B, H, n_tiles, tile_elems, D)
    pooled = xt.sum(dim=3, dtype=torch.float32)
    return pooled / variable_block_sizes.view(1, 1, -1, 1).to(torch.float32)


def _h3_build_block_mask(scores, num_prefix_tiles, sparsity, exempt,
                         video_tile_spans, span_sparsities):
    R'''scores [B,H,n_tiles,n_tiles] -> bool mask (port of official ``_build_block_mask``).

    Prefix rows/cols: with ``exempt`` every query keeps every prefix column, and every
    query keeps its own top-k of EACH video region under that region's sparsity.
    ``sparsity <= 0`` keeps every tile (dense).
    '''
    n_tiles = scores.shape[-1]
    if sparsity <= 0.0:
        span_sparsities = tuple(sparsity for _ in video_tile_spans)
    span_topk = [_h3_compute_topk(s, end - start)
                 for (start, end), s in zip(video_tile_spans, span_sparsities)]
    if all(kk == end - start for kk, (start, end) in zip(span_topk, video_tile_spans)):
        return torch.ones_like(scores, dtype=torch.bool)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if exempt or num_prefix_tiles == 0:
        for (start, end), k_span in zip(video_tile_spans, span_topk):
            if k_span == end - start:
                mask[..., start:end] = True
                continue
            idx = scores[..., start:end].topk(k_span, dim=-1).indices + start
            mask.scatter_(-1, idx, True)
        mask[..., :num_prefix_tiles] = True
    else:
        k_total = min(span_topk[0] + num_prefix_tiles, n_tiles)
        idx = scores.topk(k_total, dim=-1).indices
        mask.scatter_(-1, idx, True)
        mask[:, :, :num_prefix_tiles, :] = True
    return mask


def _h3_expand_mask_and_sizes_to_64(block_mask, variable_block_sizes, tile_elems):
    R'''Logical tile selection -> the physical 64-block map the Triton kernels run at.

    Port of the official ``_expand_mask_and_sizes_256_to_64`` / ``_128_to_64`` (their "route A"),
    generalized over ``r = tile_elems // 64``. A logical tile occupies ``r`` CONTIGUOUS 64-blocks in
    the padded buffer, so selecting it selects all r sub-blocks on both the query and the key axis
    (``repeat_interleave``), and its valid-token count chops into 64-token strides. The trailing
    sub-blocks of a partial logical tile legitimately get size 0, which every kernel treats as
    "contributes nothing" (the forward masks those columns to -inf, dQ skips them, dK/dV never
    stores them). ``tile_elems == 64`` is the identity.
    '''
    r = tile_elems // _H3_KERNEL_BLOCK
    if r == 1:
        return block_mask, variable_block_sizes
    mask_64 = block_mask.repeat_interleave(r, dim=2).repeat_interleave(r, dim=3)
    sizes = variable_block_sizes.to(torch.int32)
    offsets = torch.arange(r, device=sizes.device, dtype=torch.int32) * _H3_KERNEL_BLOCK
    sizes_64 = torch.clamp(sizes[:, None] - offsets[None, :],
                           min=0, max=_H3_KERNEL_BLOCK).reshape(-1)
    return mask_64, sizes_64


class VSASparseAttention(nn.Module):
    R'''
    Video Sparse Attention (VSA) -- the FastVideo/hao-ai-lab coarse-to-fine block-sparse
    attention (arXiv:2505.13389) used by the official FastWan2.1 sparse-distilled students.

    Two branches, mirroring the released FastWan ``to_gate_compress`` design:
      - sparse branch    : mean-pool q/k into blocks -> coarse block scores -> keep the top-k
                           key blocks per query block -> block-sparse EXACT softmax over the
                           selected blocks.
      - compressed branch: a global mean-pooled ("compressed") attention over ALL key blocks,
                           broadcast back to token resolution and scaled per token/head by the
                           learned gate ``to_gate_compress(x)`` (x = attention-input hidden
                           states). This is FastVideo's learned compensation for the context
                           lost to sparsification. ``to_gate_compress`` is zero-init, so a
                           checkpoint without those weights reduces to sparse-only::

        out = o_sparse + to_gate_compress(x) * o_compressed

    New learnable params per layer (all under ``attn_op.*``): ``to_gate_compress`` weight+bias.
    Absent from a dense checkpoint (loaded with strict=False); the official FastWan release
    provides them.

    NOTE on fidelity: the block-sparse forward IS the official FastVideo Triton kernel (vendored),
    the tile geometry / exempt mask / top-k are ports of the official H3 backend, and the tile
    volume is selectable over the official {64, 128, 256} -- 128/256 reach the same 64-block kernel
    through the official route-A expansion.

    ``forward`` accepts BSND (``[B, L, H, D]``) q/k/v like the sibling attention modules;
    internal helpers use BHLD (``[B, H, L, *]``).
    '''
    def __init__(self, head_dim, num_heads, dim, topk=0.2,
                 BLKQ=64, BLKK=64, use_bf16=True, tile_elems=_H3_TILE_ELEMS):
        R'''
        Args:
            head_dim: dimension of each head.
            num_heads: number of attention heads.
            dim: hidden size of the attention input (sizes ``to_gate_compress``).
            topk: keep-ratio of key blocks for the sparse branch (0.2 -> 80% sparse,
                matching the official ``--VSA-sparsity 0.8``).
            BLKQ/BLKK: query/key block sizes for the flat (non-segment) sparse kernel.
            use_bf16: compute the sparse kernel in bfloat16 (default) or float16.
            tile_elems: logical tile volume for the segment-aware path -- one of the official
                ``VSA_H3_TILE_SHAPES`` keys (64 / 128 / 256). Must match the volume the
                checkpoint was distilled at; the released FastWan and FastH3 both use 64.
        '''
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dim = dim
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        if tile_elems not in _H3_TILE_SHAPES:
            raise ValueError(
                f"Unsupported VSA tile volume {tile_elems}; supported volumes are "
                f"{sorted(_H3_TILE_SHAPES)} (the official VSA_H3_TILE_SHAPES).")
        self.tile_elems = tile_elems
        self.tile_shape = _H3_TILE_SHAPES[tile_elems]

        inner = num_heads * head_dim
        # Learned gate for the compressed global branch (FastVideo ``to_gate_compress``).
        self.to_gate_compress = nn.Linear(dim, inner, bias=True)
        self.init_weights_()

    def init_weights_(self):
        with torch.no_grad():
            nn.init.zeros_(self.to_gate_compress.weight)
            nn.init.zeros_(self.to_gate_compress.bias)

    def sparse(self, q, k, v):
        R'''Block-sparse exact softmax (VSA fine stage).

        Args:
            q, k: [B, H, L, D] POST-RoPE queries/keys.
            v:    [B, H, L, D] values.
        Returns:
            [B, H, L, D], or ``None`` when the sequence is shorter than one block.
        '''
        L = q.shape[2]
        if L < max(self.BLKQ, self.BLKK):
            return None
        orig_dtype = q.dtype
        with torch.no_grad():
            pq = self._pool_flat(q, self.BLKQ)
            pk = self._pool_flat(k, self.BLKK)
            scores = pq @ pk.transpose(-1, -2)
            real_topk = max(1, min(scores.shape[-1], math.ceil(self.topk * scores.shape[-1])))
            lut = scores.topk(real_topk, dim=-1, sorted=False).indices.contiguous()
            sparse_map = torch.zeros_like(scores, dtype=torch.bool)
            sparse_map.scatter_(-1, lut, True)
        if self.BLKQ % 64 or self.BLKK % 64:
            raise ValueError("VSA 的 flat block 大小必须是 64 的正整数倍。")
        B, H, L, D = q.shape
        n_tiles = math.ceil(L / 64)
        padding = n_tiles * 64 - L
        q, k, v = (F.pad(t.to(self.dtype), (0, 0, 0, padding)).contiguous() for t in (q, k, v))
        sizes = (L - torch.arange(n_tiles, device=q.device) * 64).clamp(max=64).to(torch.int32)
        mask = sparse_map.repeat_interleave(self.BLKQ // 64, dim=-2).repeat_interleave(
            self.BLKK // 64, dim=-1)[..., :n_tiles, :n_tiles].contiguous()
        o_s = sparse_linear_attention_varlen(q, k, v, mask, sizes)
        return o_s[:, :, :L].to(orig_dtype)

    @staticmethod
    def _pool_flat(x, block_size):
        B, H, L, D = x.shape
        n_blocks = math.ceil(L / block_size)
        padded = F.pad(x.float(), (0, 0, 0, n_blocks * block_size - L))
        counts = (L - torch.arange(n_blocks, device=x.device) * block_size).clamp(max=block_size)
        return padded.reshape(B, H, n_blocks, block_size, D).sum(3) / counts.view(1, 1, -1, 1)

    def compressed(self, q, k, v):
        R'''Mean-pooled global ("compressed") attention, broadcast to token resolution.

        Args:
            q, k, v: [B, H, L, D] POST-RoPE queries/keys/values.
        Returns:
            [B, H, L, D] compressed global context per token.
        '''
        B, H, L, D = q.shape
        pq = self._pool_flat(q, self.BLKQ)
        pk = self._pool_flat(k, self.BLKK)
        pv = self._pool_flat(v, self.BLKK)
        scale = D ** -0.5
        scores = (pq.float() @ pk.float().transpose(-1, -2)) * scale
        attn = torch.softmax(scores, dim=-1).to(pv.dtype)
        oc_blocks = attn @ pv                                 # [B,H,Lq,D]
        # Broadcast each query block's context to its BLKQ tokens, then trim to L.
        oc = oc_blocks.repeat_interleave(self.BLKQ, dim=2)[:, :, :L]
        return oc

    def forward_h3(self, q, k, v, x, segments, return_sparsity=False):
        R'''Segment-aware H3 VSA -- the official FastVideo segment/tile mask semantics.

        Args:
            q, k, v: [B, L, H, D] POST-RoPE queries/keys/values in PACKED order.
            x:       [B, L, C] attention-input hidden states (drives the compress gate).
            segments: packed layout -- dense row counts (``int``) and video region
                ``(t, h, w)`` token grid, in packed order. Built by the model forward.
        Returns:
            [B, L, H, D] attention output (before the output projection).

        Tiles the packed sequence into [prefix tiles][video cubic tiles] of ``self.tile_shape``,
        builds the exempt block mask on pooled tile scores, expands that logical selection down to
        the 64-block granularity the kernels run at, runs the block-sparse Triton kernel for the
        fine branch, adds the gated compressed branch on the same logical geometry, then untiles
        back to packed order.
        '''
        B, L, H, D = q.shape
        device = q.device
        (tile_partition_indices, variable_block_sizes, untile_combined_index,
         num_prefix_tiles, num_video_tiles, video_tile_spans, tile_elems) = \
            build_h3_tile_geometry(tuple(segments), device, self.tile_shape)
        n_tiles = num_prefix_tiles + num_video_tiles
        n_regions = len(video_tile_spans)
        # Logical tiles are laid end to end with no spare: the kernels take a per-64-block valid
        # count, so a partial tile needs no zero dummy partner tile any more.
        L_buf = n_tiles * tile_elems
        sparsity = 1.0 - self.topk                  # self.topk is the KEEP ratio
        vbs = variable_block_sizes.to(device)

        reuse = not torch.is_grad_enabled()
        buf_shape = (B, H, L_buf, D)

        def _tile(t, dtype, role):
            # [B,L,H,D] packed -> [B,H,L_buf,D] tile order (pad slots stay zero); under no-grad the
            # buffer is shared across layers/steps (see _h3_tile_into_buf).
            return _h3_tile_into_buf(t, dtype, untile_combined_index, buf_shape,
                                     device, role, reuse)

        qb = _tile(q, self.dtype, 'q')
        kb = _tile(k, self.dtype, 'k')
        vb = _tile(v, self.dtype, 'v')
        gate = self.to_gate_compress(x).view(B, L, H, D)
        gb = _tile(gate, gate.dtype, 'g')

        pq = _h3_pool_tiles(qb, vbs, tile_elems, n_tiles)     # [B,H,n_tiles,D] fp32
        pk = _h3_pool_tiles(kb, vbs, tile_elems, n_tiles)
        pv = _h3_pool_tiles(vb, vbs, tile_elems, n_tiles)

        # Block selection is a non-differentiable constant (exactly like the flat path's
        # get_block_map): build the mask under no_grad on detached pooled scores so no graph is
        # retained for the top-k.
        with torch.no_grad():
            scores = torch.matmul(pq.detach(), pk.detach().transpose(-2, -1)) / (D ** 0.5)
            mask = _h3_build_block_mask(scores, num_prefix_tiles, sparsity, True,
                                        video_tile_spans, (sparsity,) * n_regions)
            # Logical (tile_elems) selection -> the physical 64-block map + per-block counts.
            # Identity when tile_elems == 64.
            block_mask, block_sizes = _h3_expand_mask_and_sizes_to_64(mask, vbs, tile_elems)
        o_s = sparse_linear_attention_varlen(
            qb, kb, vb, block_mask, block_sizes, qk_scale=D ** -0.5)

        # gated compressed branch on the same logical tile geometry
        cs = torch.matmul(pq, pk.transpose(-2, -1)) * (D ** -0.5)
        attn = torch.softmax(cs, dim=-1).to(pv.dtype)
        oc = (attn @ pv).repeat_interleave(tile_elems, dim=2)  # [B,H,L_buf,D]
        out_buf = o_s.to(gb.dtype) + gb * oc.to(gb.dtype)

        out = out_buf[:, :, untile_combined_index].transpose(1, 2)  # [B,L,H,D] packed
        out = out.to(q.dtype)
        if return_sparsity:
            return out, self.topk
        return out

    def forward(self, q, k, v, x, return_sparsity=False, segments=None):
        R'''
        Args:
            q, k, v: [B, L, H, D] POST-RoPE queries/keys/values.
            x:       [B, L, C] attention-input hidden states (drives the compress gate).
            return_sparsity: also return the sparse-branch keep ratio.
        Returns:
            [B, L, H, D] attention output (before the output projection).
        '''
        if segments is not None:
            # Segment-aware tile geometry with a differentiable variable-length block-sparse
            # kernel (forward + backward), so training and inference share one selection
            # semantics. Falls back to the flat 1-D block map only when the caller cannot
            # supply a clean segment layout (segments is None).
            return self.forward_h3(q, k, v, x, segments, return_sparsity=return_sparsity)
        B, L, H, D = q.shape
        qb = q.transpose(1, 2).contiguous()                   # [B,H,L,D]
        kb = k.transpose(1, 2).contiguous()
        vb = v.transpose(1, 2).contiguous()

        o_s = self.sparse(qb, kb, vb)
        if o_s is None:
            # Shorter than one block: exact dense attention (sparsity gives no benefit).
            o_s = F.scaled_dot_product_attention(
                qb.float(), kb.float(), vb.float()).to(qb.dtype)
        gate = self.to_gate_compress(x).view(B, L, H, D).transpose(1, 2)  # [B,H,L,D]
        o_c = self.compressed(qb, kb, vb)
        out = o_s + gate.to(o_s.dtype) * o_c.to(o_s.dtype)
        out = out.transpose(1, 2)                             # [B,L,H,D]
        if return_sparsity:
            return out, self.topk
        return out