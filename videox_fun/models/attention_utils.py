import os
import warnings

import torch

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
    elif major>9:
        from sageattention_sm120 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
except:
    try:
        from sageattention import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    except:
        sageattn = None
        SAGE_ATTENTION_AVAILABLE = False


from .attention_kernel import _sparse_linear_attention, get_block_map


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
    except:
        return q, k, v


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
            deterministic=deterministic)[0].unflatten(0, (b, lq))
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
    if torch.is_grad_enabled() and attention_type == "SAGE_ATTENTION":
        attention_type = "FLASH_ATTENTION"

    if attention_type == "SAGE_ATTENTION" and SAGE_ATTENTION_AVAILABLE:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
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
            q: queries of shape (B, H, L, D).
            k: keys of shape (B, H, L, D).
            v: values of shape (B, H, L, D).
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