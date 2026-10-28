# FastVideo VSA (Video Sparse Attention) student backbone for the Wan2.1 single-expert
# models. Mirrors the SparkDiffusion/RoLa integration in ``spark_transformer3d.py`` and the
# TurboDiffusion/SLA integration in ``turbowan_transformer3d.py``: it swaps the dense softmax
# self-attention of ``WanTransformer3DModel`` for the coarse-to-fine block-sparse VSA kernel
# (reusing the in-repo SLA Triton kernel) plus FastVideo's learned ``to_gate_compress``
# compressed-global compensation branch.
#
# VSA reference: Zhang et al., "VSA: Faster Video Diffusion with Trainable Sparse Attention",
# arXiv:2505.13389 (FastVideo / hao-ai-lab). Official weights: FastVideo/FastWan2.1-T2V-1.3B.
import torch

from .attention_utils import VSASparseAttention, attention
from .wan_transformer3d import (WanSelfAttention, WanTransformer3DModel,
                                rope_apply_qk)


class FastVideoWanSelfAttention(WanSelfAttention):
    """Wan self-attention whose dense softmax kernel is swapped for VSA block-sparse attention
    (sparse fine branch + gated mean-pooled compressed branch)."""

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 topk=0.2):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        # topk: keep-ratio of key blocks per query (0.2 -> 80% sparse, the official FastWan
        # ``--VSA-sparsity 0.8``). >= 1.0 falls back to the original dense attention.
        self.topk = topk
        if topk < 1.0:
            self.attn_op = VSASparseAttention(
                self.head_dim, num_heads, dim, topk=topk)

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads * head_dim]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
            k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
            v = self.v(x.to(dtype)).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # Dense fallback (topk >= 1.0): original full-softmax attention.
        if self.topk >= 1.0:
            q, k = rope_apply_qk(q, k, grid_sizes, freqs)
            x = attention(
                q.to(dtype),
                k.to(dtype),
                v=v.to(dtype),
                k_lens=seq_lens,
                window_size=self.window_size
            )
            x = x.to(dtype).flatten(2)
            return self.o(x)

        q, k = rope_apply_qk(q, k, grid_sizes, freqs)

        g0 = grid_sizes[0]
        if (bool((grid_sizes == g0).all()) and s == int(g0.prod())
                and bool((seq_lens == s).all())):
            segments = ((int(g0[0]), int(g0[1]), int(g0[2])),)
            out = self.attn_op(q.to(dtype), k.to(dtype), v.to(dtype), x.to(dtype),
                               segments=segments)
        else:
            outputs = []
            for i, (grid, length) in enumerate(zip(grid_sizes.tolist(), seq_lens.tolist())):
                grid = tuple(int(axis) for axis in grid)
                length = int(length)
                if min(grid) <= 0 or length != grid[0] * grid[1] * grid[2] or not 0 < length <= s:
                    raise ValueError("Wan VSA 的 seq_lens 必须等于有效网格 token 数且不超过序列长度。")
                oi = self.attn_op(q[i:i + 1, :length].to(dtype), k[i:i + 1, :length].to(dtype),
                                  v[i:i + 1, :length].to(dtype), x[i:i + 1, :length].to(dtype),
                                  segments=(grid,))
                outputs.append(torch.nn.functional.pad(oi, (0, 0, 0, 0, 0, s - length)))
            out = torch.cat(outputs, dim=0)
        x = out.to(dtype).flatten(2)
        return self.o(x)


class FastVideoWanTransformer3DModel(WanTransformer3DModel):
    """Wan2.1 transformer (T2V / I2V, single expert) with a VSA sparse self-attention. Block
    type stays WanAttentionBlock, so the existing
    `--fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock` policy keeps working."""

    # NOTE: diffusers from_pretrained filters config.json keys by __init__ signature, so
    # every parent parameter must be declared explicitly (a *args/**kwargs passthrough would
    # silently drop dim/ffn_dim/... and build a wrongly-sized model).
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        cross_attn_type=None,
        student_topk=0.2,
    ):
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            in_channels=in_channels,
            hidden_size=hidden_size,
            add_control_adapter=add_control_adapter,
            in_dim_control_adapter=in_dim_control_adapter,
            downscale_factor_control_adapter=downscale_factor_control_adapter,
            add_ref_conv=add_ref_conv,
            in_dim_ref_conv=in_dim_ref_conv,
            cross_attn_type=cross_attn_type,
        )
        self.student_topk = student_topk
        self.register_to_config(student_topk=student_topk)
        self._apply_sparse_self_attention(student_topk)

    def _apply_sparse_self_attention(self, topk):
        for block in self.blocks:
            dense_attn = block.self_attn
            sparse_attn = FastVideoWanSelfAttention(
                dense_attn.dim, dense_attn.num_heads, dense_attn.window_size,
                dense_attn.qk_norm, dense_attn.eps, topk=topk,
            )
            # Under low_cpu_mem_usage from_pretrained the replaced params are meta tensors, so
            # init_weights_' zeros carry no data: wan_transformer3d.py's loader re-materialises
            # every key the checkpoint lacks, and `to_gate_compress` is on its zero-init
            # whitelist, so a cold start from a dense base still gets a zero gate. Only upgrade
            # in place (copy the real tensors over) when the weights are concrete.
            if not next(dense_attn.parameters()).is_meta:
                old_sd = {k: v for k, v in dense_attn.state_dict().items() if not v.is_meta}
                # Match the live params' device AND dtype: freshly built VSA params are fp32,
                # and mixing them with bf16 backbone weights breaks FSDP flat-param flattening.
                p0 = next(dense_attn.parameters())
                sparse_attn = sparse_attn.to(device=p0.device, dtype=p0.dtype)
                sparse_attn.load_state_dict(old_sd, strict=False)
            block.self_attn = sparse_attn

    def set_student_topk(self, topk):
        self.student_topk = topk
        self.register_to_config(student_topk=topk)
        if topk < 1.0 and all(hasattr(block.self_attn, "attn_op") and
                              not any(p.is_meta for p in block.self_attn.attn_op.parameters())
                              for block in self.blocks):
            for block in self.blocks:
                block.self_attn.topk = topk
                block.self_attn.attn_op.topk = topk
        else:
            self._apply_sparse_self_attention(topk)

    def enable_multi_gpus_inference(self,):
        """VSA has no sequence-parallel path in this repo.

        The in-repo VSA kernel only implements the single-GPU coarse-to-fine block-sparse
        forward; there is no ulysses head-parallel or ring-attention binding for it (unlike
        the RoLa/SLA ``usp_attn_*`` forwards). The 1.3B student fits comfortably on one GPU,
        so run inference single-device. Fail loudly rather than silently falling back to the
        parent's DENSE ``usp_attn_forward`` (which would bypass VSA and drop the gate params).
        """
        raise NotImplementedError(
            "FastVideoWanTransformer3DModel (VSA) supports single-GPU inference only; the VSA "
            "kernel has no ulysses/ring sequence-parallel path. Set ulysses_degree=ring_degree=1.")

    def sparse_trainable_params(self):
        """Names of the extra VSA-branch parameters, for logging / freezing."""
        return [name for name, _ in self.named_parameters() if "attn_op." in name]
