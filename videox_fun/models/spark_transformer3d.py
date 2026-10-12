# SparkDiffusion sparse student backbone for the Wan2.2 A14B MoE models.
import torch

from .attention_utils import (RoLASparseAttention, SparseLinearAttention,
                              attention)
from .wan_transformer3d import (Wan2_2Transformer3DModel, WanSelfAttention,
                               rope_apply_qk)


class SparkWanSelfAttention(WanSelfAttention):
    """Wan self-attention whose softmax kernel is swapped for RoLa sparse+low-rank attention."""

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 topk=0.03,
                 attn_variant='rola',
                 rola_rank=64):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        # topk: ratio of key blocks kept per query (0.1 -> 90% sparse, 0.05 -> 95%, 0.03 -> 97%).
        # >= 1.0 falls back to the original dense attention.
        self.topk = topk
        self.attn_variant = attn_variant
        if topk < 1.0:
            if attn_variant == 'rola':
                self.attn_op = RoLASparseAttention(
                    self.head_dim, num_heads, topk, rola_rank=rola_rank)
            elif attn_variant == 'pure_sla':
                self.attn_op = SparseLinearAttention(self.head_dim, topk)
            else:
                raise ValueError(
                    f"unknown attn_variant {attn_variant!r}; expected 'rola' or 'pure_sla'")

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
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

        # Keep the PRE-RoPE qk-normed q/k for the RoLa low-rank branch (rope_apply_qk
        # returns new tensors, so q_pre/k_pre stay valid after the rebind below).
        q_pre, k_pre = q, k
        q, k = rope_apply_qk(q, k, grid_sizes, freqs)

        # PureSLA ablation path: fused block-sparse + feature-map linear (proj_l).
        if self.attn_variant == 'pure_sla':
            x = self.attn_op(q.to(dtype), k.to(dtype), v.to(dtype))
            x = x.to(dtype).flatten(2)
            return self.o(x)

        # ---- RoLa: block-sparse branch + gated rank-truncated low-rank branch ----
        v_bhld = v.transpose(1, 2)                                    # [B,H,L,D]
        o_s = self.attn_op.sparse(q.transpose(1, 2), k.transpose(1, 2), v_bhld)
        o_l = self.attn_op.lowrank(
            q_pre.transpose(1, 2), k_pre.transpose(1, 2), v_bhld, grid_sizes, freqs)
        # Gate is computed from the raw hidden states (attention input), per official RoLa.
        x_bhld = x.view(b, s, n, d).transpose(1, 2)                   # [B,H,L,D]
        gate = self.attn_op.gate(x_bhld)                              # [B,H,L,1]
        if o_s is not None:
            scale = (o_s.pow(2).mean(dim=-1, keepdim=True) + 1e-5).sqrt()
            out = ((o_s.float() / scale.float() + gate * o_l.float()) * scale.float()).to(o_l.dtype)
        else:
            out = gate * o_l
        x = out.transpose(1, 2).flatten(2)                           # BHLD -> [B,L,H*D]
        return self.o(x)


class SparkWan2_2Transformer3DModel(Wan2_2Transformer3DModel):
    """Wan2.2 transformer (T2V / I2V, single expert per checkpoint) with a sparse
    student self-attention. Block type stays WanAttentionBlock, so the existing
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
        student_topk=0.03,
        attn_variant='rola',
        rola_rank=64,
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
        )
        self.student_topk = student_topk
        self.attn_variant = attn_variant
        self.rola_rank = rola_rank
        self._apply_sparse_self_attention(student_topk)

    def _apply_sparse_self_attention(self, topk):
        for block in self.blocks:
            dense_attn = block.self_attn
            sparse_attn = SparkWanSelfAttention(
                dense_attn.dim, dense_attn.num_heads, dense_attn.window_size,
                dense_attn.qk_norm, dense_attn.eps, topk=topk,
                attn_variant=self.attn_variant, rola_rank=self.rola_rank,
            )
            # Under low_cpu_mem_usage from_pretrained the replaced params are meta tensors
            # and get filled by the checkpoint load afterwards; only upgrade in place (copy
            # the real tensors, the freshly-init RoLa params always survive) when weights are
            # concrete.
            if not next(dense_attn.parameters()).is_meta:
                old_sd = {k: v for k, v in dense_attn.state_dict().items() if not v.is_meta}
                # Match the live params' device AND dtype: freshly built RoLa params are fp32,
                # and mixing them with bf16 backbone weights breaks FSDP flat-param flattening.
                p0 = next(dense_attn.parameters())
                sparse_attn = sparse_attn.to(device=p0.device, dtype=p0.dtype)
                sparse_attn.load_state_dict(old_sd, strict=False)
            block.self_attn = sparse_attn

    def set_student_topk(self, topk):
        """(Re)build the sparse kernels at the given top-k ratio. Always recreates
        attn_op, so the RoLa params (proj_q/proj_k/gate_proj/gate_bias) are reset to their
        init: call right after from_pretrained and BEFORE loading a stage-1 sparse checkpoint
        (its trained RoLa params would be dropped otherwise)."""
        self.student_topk = topk
        self._apply_sparse_self_attention(topk)

    def enable_multi_gpus_inference(self,):
        """Bind the sparse-aware USP forward instead of the parent's dense one.

        ``Wan2_2Transformer3DModel.enable_multi_gpus_inference`` monkey-patches every
        ``blocks[i].self_attn.forward`` to ``usp_attn_forward`` (dense full-softmax via
        ``xFuserLongContextAttention``), which never reaches ``SparkWanSelfAttention.attn_op``
        and would silently drop the trained RoLa params under multi-GPU. Rebind to
        ``usp_attn_sparkwan_forward`` (ulysses head-parallel + RoLa kernel) so the sparse branch
        stays live; a plain ``shard_model`` FSDP wrap does not help here -- it only shards
        params and shares the same SP branch, so the patch still applies.
        """
        import types

        from ..dist import (get_sequence_parallel_rank,
                            get_sequence_parallel_world_size, get_sp_group,
                            usp_attn_sparkwan_forward)
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather

        # Ulysses head-split needs num_heads divisible by the SP degree.
        n_heads = self.blocks[0].self_attn.num_heads
        if self.sp_world_size > 1 and n_heads % self.sp_world_size != 0:
            raise ValueError(
                f"sparse multi-GPU (ulysses) needs num_heads ({n_heads}) divisible by the "
                f"sequence-parallel degree ({self.sp_world_size}); set ulysses_degree accordingly "
                f"and keep ring_degree=1 (the SLA kernel has no ring-attention path).")

        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_sparkwan_forward, block.self_attn)

        if hasattr(self, 'vace_blocks'):
            for block in self.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_sparkwan_forward, block.self_attn)

    def sparse_trainable_params(self):
        """Names of the extra sparse-branch parameters, for logging / stage-A freezing."""
        return [name for name, _ in self.named_parameters() if "attn_op." in name]
