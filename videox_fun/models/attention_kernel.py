# Modified from https://github.com/thu-ml/TurboDiffusion/blob/main/turbodiffusion/SLA/kernel.py
""" 
Copyright (c) 2025 by SLA team.

Licensed under the Apache License, Version 2.0 (the "License");

Citation (please cite if you use this code):

@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention}, 
  author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}
"""

import math

import torch
import triton
import triton.language as tl

# ======================================================================================
# This module holds TWO sparse-attention kernel families, kept in contiguous sections.
#
#   SECTION A -- VSA, variable-length block-sparse. This is the path the repo actually runs
#       for MiniMax-H3 and Wan sparse attention (attention_utils.VSASparseAttention.forward_h3
#       -> sparse_linear_attention_varlen). Everything that path needs lives in this section,
#       in call order: the two vendored upstream blocks (the official tile-64 forward kernel,
#       then the on-device mask -> key-list compaction), our own varlen backward kernels, the
#       autograd Function, and its wrapper.
#
#   SECTION B -- the fixed-length SLA original this file was modified from (see the licence
#       header above). Still live, NOT dead code: attention_utils' flat sparse path calls
#       get_block_map + _sparse_linear_attention.apply, and its tile pooling calls mean_pool.
#
# The two families share no kernels. A symbol suffixed `_varlen` belongs to section A.
# ======================================================================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VENDORED FROM UPSTREAM (Apache-2.0) -- BLOCK 1/2 BEGIN >>>>>>>>>>>>>>>>>>>>
# Source: hao-ai-lab/fastvideo `fastvideo_kernel` 0.3.5, file
# python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py, lines 11-166 and 703-752.
# It replaces the hand-written `_attn_fwd_varlen` that used to live further down this module.
#
# Why vendored: upstream's compiled sm90 CUDA kernel (`fastvideo_kernel._C`) ships only cp312 /
# torch==2.12.0 wheels, so it is not installable here (we run torch 2.7.0 / py3.10), whereas its
# pure-Triton tile-64 block-sparse forward runs as-is with no build step. A kernel-level micro-bench
# on the Wan VSA workload (seq 33390, 672 tiles, sparsity 0.8, bf16, H800) measured this forward at
# 6.31 ms/call vs `_attn_fwd_varlen` at 13.48 ms/call (2.14x), with identical output (rel_l2
# 6.2e-5). The wins are @triton.autotune over num_stages/num_warps, tl.make_block_ptr + tl.advance
# loads, a fused tl.dot(p, v, acc) accumulate, and a per-query-tile variable kv-block count (vs a
# uniform-MAXK loop over a dummy-padded tile).
#
# This is deliberately NOT the fastest forward available. That same sm90 CUDA kernel (ThunderKittens)
# IS buildable from source on this box -- only its prebuilt wheels are pinned -- and measures
# 4.66 ms/call on the workload above, i.e. 1.35x over this Triton kernel (step-level VSA-vs-dense
# 1.019x -> 1.115x at 480P, 1.233x -> 1.381x at 720P). It stays out of the production path because
# it adds a build dependency and fails SILENTLY three ways: on a non-sm90a device pass it compiles
# to an empty stub returning uninitialised memory; it reinterpret_casts `variable_block_sizes` to
# int32 with no dtype check, so an int64 tensor yields all-NaN; and its lse is [B,H,L,1] in the
# natural-log domain scaled by -1/qk_scale rather than our [B,H,L] base-2. The build recipe, the
# runtime gate it would need, and the measurement set are archived in
# `_fastvideo_sm90_kernel/README.md`.
#
# Deviations from upstream, listed so a diff against the sdist still audits this block:
#   1. its `import torch / triton / triton.language`, and its SPARSE-ADDITION `import math`, are
#      hoisted to this module's top import block -- same names, no behavioural change, and the
#      two `SPARSE ADDITION BEGIN/END` banner comments upstream wrapped around that import are
#      dropped with it (the one around `@triton.autotune` below is kept);
#   2. the inner dot's `p.to(tl.bfloat16)` is `p.to(v.dtype)` here, because our varlen path and
#      its fp32 oracle tests must not be forced to bf16 (output identical when v IS bf16).
# Everything else between the BEGIN/END markers is byte-identical to upstream. Our own backward
# (`_attn_bwd_*_varlen`) is untouched by this block: it supplies the forward only.

# BLOCK_M / BLOCK_N are fixed at 64 because they are structural, not tunable:
# the kernel indexes the top-k list per BLOCK_M q-tile and addresses keys as
# kv_idx * BLOCK_N, so both must match the granularity q2k_index and
# variable_block_sizes were built at.
#
# num_stages / num_warps ARE free, and the previous {3, 4, 7} was inherited from
# the upstream tutorial rather than tuned here. It skips 5 and 6; on Blackwell
# (sm_121) the optimum is num_stages=5, so the search could not reach it. Both
# block paths independently select 5 once it is available. Autotune still picks
# per architecture, so other GPUs re-tune rather than inheriting this choice.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64]\
    for BN in [64]\
    for s in [2, 3, 4, 5, 6, 7]\
    for w in [4, 8]\
]


# ──────────────────────────── SPARSE ADDITION BEGIN ───────────────────────────
@triton.autotune(configs, key=["N_CTX_Q", "HEAD_DIM"])
@triton.jit
def _attn_fwd_sparse(
        Q,
        K,
        V,
        sm_scale,  #
        q2k_index,
        q2k_num,
        max_kv_blks,  #
        variable_block_sizes,
        M,
        Out,  #
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,
        Z,
        H,
        N_CTX_Q,  #
        N_CTX_KV,  #
        HEAD_DIM: tl.constexpr,  #
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        STAGE: tl.constexpr):
    """
    64×64 **block-sparse** forward kernel. Back-prop kernels remain dense
    (32×64 and 64×32) – memory footprint unchanged.
    """

    # ----- program-id mapping -----
    q_blk = tl.program_id(0)  # Q-tile index
    off_hz = tl.program_id(1)  # fused (batch, head)
    b = off_hz // H
    h = off_hz % H
    q_tiles = N_CTX_Q // BLOCK_M
    meta_base = ((b * H + h) * q_tiles + q_blk)

    kv_blocks = tl.load(q2k_num + meta_base)  # int32
    kv_ptr = q2k_index + meta_base * max_kv_blks  # ptr to list

    # ----- base pointers -----
    # Note: when q and kv have different sequence lengths, their per-(batch,head)
    # strides differ, so we must compute separate base offsets.
    q_off = (b.to(tl.int64) * stride_qz + h.to(tl.int64) * stride_qh)
    k_off = (b.to(tl.int64) * stride_kz + h.to(tl.int64) * stride_kh)
    v_off = (b.to(tl.int64) * stride_vz + h.to(tl.int64) * stride_vh)
    o_off = (b.to(tl.int64) * stride_oz + h.to(tl.int64) * stride_oh)

    Q_ptr = tl.make_block_ptr(base=Q + q_off,
                              shape=(N_CTX_Q, HEAD_DIM),
                              strides=(stride_qm, stride_qk),
                              offsets=(q_blk * BLOCK_M, 0),
                              block_shape=(BLOCK_M, HEAD_DIM),
                              order=(1, 0))

    K_base = tl.make_block_ptr(base=K + k_off,
                               shape=(HEAD_DIM, N_CTX_KV),
                               strides=(stride_kk, stride_kn),
                               offsets=(0, 0),
                               block_shape=(HEAD_DIM, BLOCK_N),
                               order=(0, 1))

    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_base = tl.make_block_ptr(base=V + v_off,
                               shape=(N_CTX_KV, HEAD_DIM),
                               strides=(stride_vk, stride_vn),
                               offsets=(0, 0),
                               block_shape=(BLOCK_N, HEAD_DIM),
                               order=v_order)

    O_ptr = tl.make_block_ptr(base=Out + o_off,
                              shape=(N_CTX_Q, HEAD_DIM),
                              strides=(stride_om, stride_on),
                              offsets=(q_blk * BLOCK_M, 0),
                              block_shape=(BLOCK_M, HEAD_DIM),
                              order=(1, 0))

    # ----- accumulators -----
    offs_m = q_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # 1/ln2
    q = tl.load(Q_ptr)

    # ----- sparse loop over valid K/V tiles -----
    for i in range(0, kv_blocks):
        kv_idx = tl.load(kv_ptr + i).to(tl.int32)
        block_size = tl.load(variable_block_sizes + kv_idx)
        K_ptr = tl.advance(K_base, (0, kv_idx * BLOCK_N))
        V_ptr = tl.advance(V_base, (kv_idx * BLOCK_N, 0))

        k = tl.load(K_ptr)
        qk = tl.dot(q, k)
        # mask out invalid columns
        mask = tl.arange(0, BLOCK_N) < block_size
        qk = tl.where(mask[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(V_ptr)
        acc = tl.dot(p.to(v.dtype), v, acc)  # dtype-agnostic: official hardcodes bf16,
        # but our varlen path/tests also run fp32; p.to(v.dtype) keeps bf16 identical.
        m_i = m_ij

    # ----- epilogue -----
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(M + off_hz * N_CTX_Q + offs_m, m_i)
    tl.store(O_ptr, acc.to(Out.type.element_ty))

def triton_block_sparse_attn_forward(q, k, v, q2k_index, q2k_num, variable_block_sizes):
    B, H, Tq, D = q.shape
    Tkv = k.shape[2]
    sm_scale = 1.0 / math.sqrt(D)
    max_kv_blks = q2k_index.shape[-1]
    assert Tq % 64 == 0, f"q length must be a multiple of 64, but got {Tq}"
    assert Tkv % 64 == 0, f"kv length must be a multiple of 64, but got {Tkv}"
    assert q2k_num.shape[
        -1] == Tq // 64, f"shape mismatch, Tq // 64 = {Tq // 64}, q2k_num.shape[-2] = {q2k_num.shape[-2]}"
    assert variable_block_sizes.numel() == Tkv // 64, (
        f"shape mismatch, variable_block_sizes must have length {Tkv // 64}, "
        f"got {variable_block_sizes.numel()}")
    o = torch.empty_like(q)
    M = torch.empty((B, H, Tq), dtype=torch.float32, device=q.device)

    grid = lambda _: (triton.cdiv(Tq, 64), B * H, 1)
    _attn_fwd_sparse[grid](q,
                           k,
                           v,
                           sm_scale,
                           q2k_index,
                           q2k_num,
                           max_kv_blks,
                           variable_block_sizes,
                           M,
                           o,
                           q.stride(0),
                           q.stride(1),
                           q.stride(2),
                           q.stride(3),
                           k.stride(0),
                           k.stride(1),
                           k.stride(2),
                           k.stride(3),
                           v.stride(0),
                           v.stride(1),
                           v.stride(2),
                           v.stride(3),
                           o.stride(0),
                           o.stride(1),
                           o.stride(2),
                           o.stride(3),
                           B,
                           H,
                           Tq,
                           Tkv,
                           HEAD_DIM=D,
                           STAGE=3)

    return o, M
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VENDORED FROM UPSTREAM (Apache-2.0) -- BLOCK 1/2 END   >>>>>>>>>>>>>>>>>>>>

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VENDORED FROM UPSTREAM (Apache-2.0) -- BLOCK 2/2 BEGIN >>>>>>>>>>>>>>>>>>>>
# Source: hao-ai-lab/fastvideo `fastvideo_kernel` 0.3.5, file
# python/fastvideo_kernel/triton_kernels/index.py, lines 32-61 and 106-144.
#
# Why vendored: same install blocker as block 1/2, and this helper is pure Triton with no
# dependency on the compiled `_C` extension.
#
# What it replaces: our mask -> key-list conversion used `mask.float().topk(int(counts.max()))`,
# whose `int(...)` forces a GPU->CPU sync on every layer of every step and materializes an fp32
# copy of the [B,H,n,n] mask. `map_to_index` compacts each row on-device
# (grid = (bs, h, num_q_blocks)), pads with -1, and returns the per-row counts as an int32 tensor
# -- no host sync, no fp32 temp.
#
# Semantics relied on by the caller: for each query block, the first `index_num` entries of
# `index` are its selected kv-block ids in ASCENDING order; the rest are -1 and are never read
# (both the official forward kernel above and our backward loop bound their iteration by it).
#
# Deviation from upstream: its `import torch / triton / triton.language` are hoisted to this
# module's top, and its unused `from typing import Tuple` (kept upstream only for parity) is
# dropped. Between the BEGIN/END markers the block is otherwise byte-identical.

@triton.jit
def map_to_index_kernel(
    map_ptr,
    index_ptr,
    index_num_ptr,
    map_bs_stride,
    map_h_stride,
    map_q_stride,
    map_kv_stride,
    index_bs_stride,
    index_h_stride,
    index_q_stride,
    index_kv_stride,
    index_num_bs_stride,
    index_num_h_stride,
    index_num_q_stride,
    num_kv_blocks,
):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    index_ptr_base = index_ptr + b * index_bs_stride + h * index_h_stride + q * index_q_stride
    map_ptr_base = map_ptr + b * map_bs_stride + h * map_h_stride + q * map_q_stride

    num = 0
    for i in tl.range(num_kv_blocks):
        map_entry = tl.load(map_ptr_base + i * map_kv_stride)
        if map_entry:
            tl.store(index_ptr_base + num * index_kv_stride, i)
            num += 1

    tl.store(index_num_ptr + b * index_num_bs_stride + h * index_num_h_stride + q * index_num_q_stride, num)


def map_to_index(block_map: torch.Tensor):
    """
    Convert a block map to indices and counts.
    
    Args:
        block_map: [bs, h, num_q_blocks, num_kv_blocks]
            The block map tensor.
    
    Returns:
        index: [bs, h, num_q_blocks, num_kv_blocks]
            The indices of the blocks.
        index_num: [bs, h, num_q_blocks]
            The number of blocks for each q block.
    """
    bs, h, num_q_blocks, num_kv_blocks = block_map.shape

    index = torch.full((block_map.shape), -1, dtype=torch.int32, device=block_map.device)
    index_num = torch.empty((bs, h, num_q_blocks), dtype=torch.int32, device=block_map.device)

    grid = (bs, h, num_q_blocks)
    map_to_index_kernel[grid](
        block_map,
        index,
        index_num,
        block_map.stride(0),
        block_map.stride(1),
        block_map.stride(2),
        block_map.stride(3),
        index.stride(0),
        index.stride(1),
        index.stride(2),
        index.stride(3),
        index_num.stride(0),
        index_num.stride(1),
        index_num.stride(2),
        num_kv_blocks=num_kv_blocks,
    )

    return index, index_num
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VENDORED FROM UPSTREAM (Apache-2.0) -- BLOCK 2/2 END   >>>>>>>>>>>>>>>>>>>>


@triton.jit
def _attn_bwd_preprocess_varlen(
    O, DO, DELTAS, QSIZE,
    L_PAD,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # delta[row] = sum_d o[row,d] * do[row,d], per query tile, over the tile-order buffer.
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)
    qkv_offset = idx_bh * L_PAD * D
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    qsize = tl.load(QSIZE + idx_m)
    q_mask = offs_m < qsize
    ptrs = qkv_offset + idx_m * BLOCK_M * D + offs_m[:, None] * D + offs_d[None, :]
    o = tl.load(O + ptrs, mask=q_mask[:, None], other=0.0)
    do = tl.load(DO + ptrs, mask=q_mask[:, None], other=0.0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(DELTAS + idx_bh * L_PAD + idx_m * BLOCK_M + offs_m, delta, mask=q_mask)


@triton.jit
def _attn_bwd_dq_varlen(
    Q, K, V, LSE, DELTAS, DO, DQ, LUT, LUTN,
    QSIZE, KSIZE,
    qk_scale,
    MAXK,
    N_TILES,
    L_PAD,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # dQ for one query tile: loop over the LUTN[row] key tiles that tile actually selected (ids in
    # LUT, ascending, -1 beyond the count) and recompute p from LSE. The bound is a runtime per-row
    # value, so a row that selected few tiles does no extra work and never touches the -1 pads.
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)
    qkv_offset = idx_bh * L_PAD * D
    lse_offset = idx_bh * L_PAD
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    qsize = tl.load(QSIZE + idx_m)
    q_mask = offs_m < qsize
    m_ptrs = qkv_offset + idx_m * BLOCK_M * D + offs_m[:, None] * D + offs_d[None, :]
    q = tl.load(Q + m_ptrs, mask=q_mask[:, None], other=0.0)
    do = tl.load(DO + m_ptrs, mask=q_mask[:, None], other=0.0)
    delta = tl.load(DELTAS + lse_offset + idx_m * BLOCK_M + offs_m, mask=q_mask, other=0.0)
    lse = tl.load(LSE + lse_offset + idx_m * BLOCK_M + offs_m, mask=q_mask, other=float('inf'))

    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    lut_base = LUT + (idx_bh * N_TILES + idx_m) * MAXK
    kv_blocks = tl.load(LUTN + idx_bh * N_TILES + idx_m)
    for i in tl.range(0, kv_blocks):
        idx_n = tl.load(lut_base + i)
        ksize = tl.load(KSIZE + idx_n)
        n_mask = offs_n < ksize
        n_ptrs = qkv_offset + idx_n * BLOCK_N * D + offs_n[:, None] * D + offs_d[None, :]
        k = tl.load(K + n_ptrs, mask=n_mask[:, None], other=0.0)   # [BLOCK_N, D]
        v = tl.load(V + n_ptrs, mask=n_mask[:, None], other=0.0)   # [BLOCK_N, D]
        qk = tl.dot(q, k.T) * (qk_scale * 1.4426950408889634)     # [BLOCK_M, BLOCK_N]
        p = tl.math.exp2(qk - lse[:, None])
        p = tl.where(n_mask[None, :], p, 0.0)
        dp = tl.dot(do, v.T).to(tl.float32)                        # [BLOCK_M, BLOCK_N]
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)                           # [BLOCK_M, D]
    tl.store(DQ + m_ptrs, (dq * qk_scale).to(DQ.type.element_ty), mask=q_mask[:, None])


@triton.jit
def _attn_bwd_dkdv_varlen(
    Q, K, V, DO, DK, DV, KBID, LSE, DELTAS,
    QSIZE, KSIZE,
    qk_scale,
    N_TILES,
    L_PAD,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # dK/dV for one key tile: loop over ALL query tiles, keep those whose KBID incidence is 1.
    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)
    qkv_offset = idx_bh * L_PAD * D
    lse_offset = idx_bh * L_PAD
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    ksize = tl.load(KSIZE + idx_n)
    n_mask = offs_n < ksize
    n_ptrs = qkv_offset + idx_n * BLOCK_N * D + offs_n[:, None] * D + offs_d[None, :]
    k = tl.load(K + n_ptrs, mask=n_mask[:, None], other=0.0)       # [BLOCK_N, D]
    v = tl.load(V + n_ptrs, mask=n_mask[:, None], other=0.0)       # [BLOCK_N, D]

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    kbid_base = KBID + idx_bh * N_TILES * N_TILES + idx_n
    for idx_m in tl.range(0, N_TILES):
        kbid = tl.load(kbid_base + idx_m * N_TILES)
        if kbid == 1:
            qsize = tl.load(QSIZE + idx_m)
            m_mask = offs_m < qsize
            m_ptrs = qkv_offset + idx_m * BLOCK_M * D + offs_m[:, None] * D + offs_d[None, :]
            q = tl.load(Q + m_ptrs, mask=m_mask[:, None], other=0.0)   # [BLOCK_M, D]
            do = tl.load(DO + m_ptrs, mask=m_mask[:, None], other=0.0)
            lse = tl.load(LSE + lse_offset + idx_m * BLOCK_M + offs_m, mask=m_mask, other=float('inf'))
            delta = tl.load(DELTAS + lse_offset + idx_m * BLOCK_M + offs_m, mask=m_mask, other=0.0)
            qkT = tl.dot(k, q.T) * (qk_scale * 1.4426950408889634)    # [BLOCK_N, BLOCK_M]
            pT = tl.math.exp2(qkT - lse[None, :])
            pT = tl.where(m_mask[None, :], pT, 0.0)
            dv += tl.dot(pT.to(do.dtype), do)                          # [BLOCK_N, D]
            dpT = tl.dot(v, do.T).to(tl.float32)                       # [BLOCK_N, BLOCK_M]
            dsT = pT * (dpT - delta[None, :])
            dk += tl.dot(dsT.to(q.dtype), q)                           # [BLOCK_N, D]
    tl.store(DK + n_ptrs, (dk * qk_scale).to(DK.type.element_ty), mask=n_mask[:, None])
    tl.store(DV + n_ptrs, dv.to(DV.type.element_ty), mask=n_mask[:, None])


class _sparse_linear_attention_varlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_tile, k_tile, v_tile, q2k_idx, q2k_num, block_sizes, kb_id, qk_scale):
        assert q_tile.is_contiguous() and k_tile.is_contiguous() and v_tile.is_contiguous()
        B, H, L_BUF, D = q_tile.shape
        # Forward runs the vendored official Triton block-sparse kernel, which consumes the
        # official (q2k_index, q2k_num) pair map_to_index produced: the first q2k_num[row] entries
        # of each row are that query tile's kv-tile ids and the -1 pads are never read. Its M is
        # [B,H,L_BUF] base-2 lse (m_i + log2 l_i), byte-compatible with what our backward expects,
        # so the save_for_backward contract is unchanged.
        out, lse = triton_block_sparse_attn_forward(
            q_tile, k_tile, v_tile, q2k_idx, q2k_num, block_sizes)
        ctx.save_for_backward(q_tile, k_tile, v_tile, q2k_idx, q2k_num,
                              block_sizes, kb_id, lse, out)
        ctx.qk_scale = qk_scale
        return out

    @staticmethod
    def backward(ctx, do):
        (q_tile, k_tile, v_tile, q2k_idx, q2k_num,
         block_sizes, kb_id, lse, out) = ctx.saved_tensors
        do = do.contiguous()
        B, H, L_BUF, D = q_tile.shape
        num_tiles = L_BUF // 64
        MAXK = q2k_idx.shape[-1]
        dq = torch.zeros_like(q_tile)
        dk = torch.zeros_like(k_tile)
        dv = torch.zeros_like(v_tile)
        delta = torch.empty((B, H, L_BUF), device=q_tile.device, dtype=torch.float32)
        grid = (num_tiles, B * H)
        _attn_bwd_preprocess_varlen[grid](
            out, do, delta, block_sizes, L_BUF, D, 64,
            num_warps=8, num_stages=3,
        )
        _attn_bwd_dq_varlen[grid](
            q_tile, k_tile, v_tile, lse, delta, do, dq, q2k_idx, q2k_num,
            block_sizes, block_sizes, ctx.qk_scale,
            MAXK, num_tiles, L_BUF, D, 64, 64,
            num_warps=8, num_stages=3,
        )
        _attn_bwd_dkdv_varlen[grid](
            q_tile, k_tile, v_tile, do, dk, dv, kb_id, lse, delta,
            block_sizes, block_sizes, ctx.qk_scale,
            num_tiles, L_BUF, D, 64, 64,
            num_warps=8, num_stages=3,
        )
        return dq, dk, dv, None, None, None, None, None


def sparse_linear_attention_varlen(q_tile, k_tile, v_tile, block_mask, block_sizes,
                                   qk_scale=None):
    """Tile-order block-sparse attention for the H3/Wan VSA path (forward + backward).

    q_tile/k_tile/v_tile: [B, H, L_BUF, D] contiguous, rows in tile order, pad slots zero.
    block_mask: [B, H, num_tiles, num_tiles] bool query->key incidence at the PHYSICAL 64-token
        block granularity the kernels run at (a logical 128/256-tile selection is expanded down to
        it by ``_h3_expand_mask_and_sizes_to_64``). It drives both stages: map_to_index compacts it
        into the per-query key list used by the forward and by dQ, and it is the dK/dV reverse-loop
        incidence. block_sizes: [num_tiles] int valid token count per 64-token block; 0 is legal
        (a partial logical tile's trailing sub-blocks) and makes that block contribute nothing.
    num_tiles is derived as L_BUF // 64, so the buffer carries no separate dummy tile.
    Returns a NEW [B, H, L_BUF, D] output buffer (pad rows are garbage; the caller untiles only the
    real rows). Differentiable -- gradients flow to q_tile/k_tile/v_tile.
    """
    assert q_tile.is_contiguous() and k_tile.is_contiguous() and v_tile.is_contiguous()
    B, H, L_BUF, D = q_tile.shape
    assert L_BUF > 0 and L_BUF % 64 == 0, "VSA tile 缓冲长度必须是正的 64 倍数。"
    assert k_tile.shape == q_tile.shape and v_tile.shape == q_tile.shape
    num_tiles = L_BUF // 64
    assert tuple(block_mask.shape) == (B, H, num_tiles, num_tiles), \
        f"block_mask {tuple(block_mask.shape)} vs expected {(B, H, num_tiles, num_tiles)}"
    assert block_sizes.numel() == num_tiles, \
        f"block_sizes has {block_sizes.numel()} entries, expected {num_tiles}"
    default_scale = D ** -0.5
    if qk_scale is not None and not math.isclose(qk_scale, default_scale, rel_tol=1e-7):
        raise ValueError("VSA 前向核仅支持 head_dim**-0.5；不接受自定义 qk_scale。")
    qk_scale = default_scale
    # On-device compaction by the vendored official Triton kernel: no host sync (the topk over
    # int(counts.max()) it replaces synced once per layer per step). `q2k_idx` keeps the full
    # num_tiles width and pads with -1; both kernels bound their loops by q2k_num.
    q2k_idx, q2k_num = map_to_index(block_mask.contiguous())
    block_sizes = block_sizes.to(torch.int32).contiguous()
    kb_id = block_mask.to(torch.int8).contiguous()
    return _sparse_linear_attention_varlen.apply(
        q_tile, k_tile, v_tile, q2k_idx, q2k_num, block_sizes, kb_id, qk_scale)


# ======================================================================================
# SECTION B -- fixed-length SLA original (NOT varlen): mean_pool / get_block_map / _attn_fwd /
# _attn_bwd_{preprocess,dq,dkdv} / class _sparse_linear_attention. Do not confuse these with
# section A's `*_varlen` kernels; nothing in section A calls into this section.
# ======================================================================================

@triton.jit
def compress_kernel(
    X, XM,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_bh * L * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    x = tl.load(X + x_offset + offs_l[:, None] * D + offs_d[None, :], mask=offs_l[:, None] < L)

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))


def mean_pool(x, BLK):
    assert x.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    grid = (L_BLOCKS, B * H)
    compress_kernel[grid](x, x_mean, L, D, BLK)
    return x_mean


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64):
    arg_k = k - torch.mean(k, dim=-2, keepdim=True) # smooth-k technique in SageAttention
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


@triton.jit
def _attn_fwd(
    Q, K, V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT, LSE, OS,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_offset = idx_bh * L * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    lse_offset = idx_bh * L
    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[None, :] * D + offs_d[:, None]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    OS_ptrs = OS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    LUT_ptr = LUT + lut_offset
    LSE_ptrs = LSE + lse_offset + offs_m
    
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L - idx_n * BLOCK_N
        
        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[None, :])
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
        if L - idx_n * BLOCK_N < BLOCK_N:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < L)
    
    m_i += tl.math.log2(l_i)
    tl.store(LSE_ptrs, m_i, mask=offs_m < L)


@triton.jit
def _attn_bwd_preprocess(
    OS, DOS, DELTAS,
    L,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    OS += idx_bh * L * D
    DOS += idx_bh * L * D
    DELTAS += idx_bh * L

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    o_s = tl.load(OS + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L)
    do_s = tl.load(DOS + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L)
    
    delta_s = tl.sum(o_s * do_s, axis=1).to(DELTAS.type.element_ty)
    tl.store(DELTAS + offs_m, delta_s, mask=offs_m < L)


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    Q, K, V, LSE, DELTAS,
    DOS, DQ, LUT,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    qkv_offset = idx_bh * L * D
    lse_offset = idx_bh * L
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    DQ_ptrs = DQ + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    DOS_ptrs = DOS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m
    LUT_ptr = LUT + lut_offset

    # load Q, DOS, DOL, LSE, DELTA, S: they stay in SRAM throughout the inner loop.
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)
    do_s = tl.load(DOS_ptrs, mask=offs_m[:, None] < L)
    delta_s = tl.load(DELTAS_ptrs, mask=offs_m < L)
    lse = tl.load(LSE_ptrs, mask=offs_m < L, other=float("inf"))
    
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for block_idx in tl.range(topk, num_stages=2):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L - idx_n * BLOCK_N
        
        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        qk = tl.dot(q, k.T) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
        p = tl.math.exp2(qk - lse[:, None])
        p = tl.where(n_mask[None, :], p, 0.0)
        
        # Compute dP and dS.
        dp = tl.dot(do_s, v.T).to(tl.float32)
        ds = p * (dp - delta_s[:, None])
        # Compute dQ.
        dq += tl.dot(ds.to(k.dtype), k)
    tl.store(DQ_ptrs, dq * qk_scale, mask=offs_m[:, None] < L)
    

@triton.jit
def _attn_bwd_dkdv(
    Q, K, V, DOS, DK, DV,
    qk_scale, KBID, LSE, DELTAS,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SLICE_FACTOR: tl.constexpr,
):
    BLOCK_M2: tl.constexpr = BLOCK_M // BLOCK_SLICE_FACTOR

    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M2)
    offs_d = tl.arange(0, D)

    qkv_offset = idx_bh * L * D
    kbid_offset = idx_bh * M_BLOCKS * N_BLOCKS
    lse_offset = idx_bh * L

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    DOS_ptrs = DOS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    DK_ptrs = DK + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    DV_ptrs = DV + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m
    KBID_ptr = KBID + kbid_offset + idx_n

    # load K, V and CK: they stay in SRAM throughout the inner loop.
    k = tl.load(K_ptrs, mask=offs_n[:, None] < L)
    v = tl.load(V_ptrs, mask=offs_n[:, None] < L)
        
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    for idx_m in tl.range(0, L, BLOCK_M2):
        kbid = tl.load(KBID_ptr)
        if kbid == 1:
            m_mask = offs_m < L - idx_m
            q = tl.load(Q_ptrs, mask=m_mask[:, None])
            lse = tl.load(LSE_ptrs, mask=m_mask, other=float("inf"))
            qkT = tl.dot(k, q.T) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
            pT = tl.math.exp2(qkT - lse[None, :])
            pT = tl.where(offs_n[:, None] < L, pT, 0.0)

            do = tl.load(DOS_ptrs, mask=m_mask[:, None])
            # Compute dV.
            dv += tl.dot(pT.to(do.dtype), do)
            delta = tl.load(DELTAS_ptrs, mask=m_mask)
            # Compute dP and dS.
            dpT = tl.dot(v, tl.trans(do))
            dsT = pT * (dpT - delta[None, :])
            dk += tl.dot(dsT.to(q.dtype), q)
        
        # Increment pointers
        Q_ptrs += BLOCK_M2 * D
        DOS_ptrs += BLOCK_M2 * D
        LSE_ptrs += BLOCK_M2
        DELTAS_ptrs += BLOCK_M2
        if (idx_m + BLOCK_M2) % BLOCK_M == 0:
            KBID_ptr += N_BLOCKS

    # Write back dK, dV and dCK
    tl.store(DK_ptrs, dk * qk_scale, mask=offs_n[:, None] < L)
    tl.store(DV_ptrs, dv, mask=offs_n[:, None] < L)
    

class _sparse_linear_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert k_block_id.is_contiguous() and lut.is_contiguous()

        # We recommend the following two settings
        assert BLOCK_M == 64 or BLOCK_M == 128
        assert BLOCK_N == 64

        B, H, L, D = q.shape
        if qk_scale is None:
            qk_scale = D**-0.5

        M_BLOCKS = triton.cdiv(L, BLOCK_M)

        o_s = torch.empty_like(v)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)

        grid = (M_BLOCKS, B * H)
        _attn_fwd[grid](
            q, k, v, qk_scale, topk,
            lut, lse, o_s,
            L, M_BLOCKS,
            D, BLOCK_M, BLOCK_N,
            num_warps=4 if q.shape[-1] == 64 else 8,
            num_stages=3
        )
        
        ctx.save_for_backward(q, k, v, k_block_id, lut, lse, o_s)
        ctx.qk_scale = qk_scale
        ctx.topk = topk
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        return o_s

    @staticmethod
    def backward(ctx, do_s):
        q, k, v, k_block_id, lut, lse, o_s = ctx.saved_tensors
        do_s = do_s.contiguous()

        BLOCK_M, BLOCK_N = ctx.BLOCK_M, ctx.BLOCK_N
        B, H, L, D = q.shape

        M_BLOCKS = triton.cdiv(L, BLOCK_M)
        N_BLOCKS = triton.cdiv(L, BLOCK_N)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta_s = torch.empty_like(lse)

        grid = (M_BLOCKS, B * H)
        _attn_bwd_preprocess[grid](
            o_s, do_s, delta_s,
            L, D, BLOCK_M,
        )

        grid = (M_BLOCKS, B * H)
        _attn_bwd_dq[grid](
            q, k, v, lse, delta_s,
            do_s, dq, lut,
            ctx.qk_scale, ctx.topk,
            L, M_BLOCKS,
            D, BLOCK_M, BLOCK_N,
            num_warps=4 if q.shape[-1] == 64 else 8,
            num_stages=4 if q.shape[-1] == 64 else 5
        )

        grid = (N_BLOCKS, B * H)
        _attn_bwd_dkdv[grid](
            q, k, v, do_s, dk, dv,
            ctx.qk_scale, k_block_id, lse, delta_s,
            L, M_BLOCKS, N_BLOCKS,
            D, BLOCK_M, BLOCK_N,
            BLOCK_SLICE_FACTOR=BLOCK_M // 64,
            num_warps=4 if q.shape[-1] == 64 else 8,
            num_stages=4 if q.shape[-1] == 64 else 5
        )

        return dq, dk, dv, None, None, None, None, None, None