# FastVideo VSA (Video Sparse Attention) student backbone for MiniMax-H3. Mirrors the Wan-side
# integration in ``fastvideo_wan_transformer3d.py``: it swaps the dense softmax self-attention of
# ``MiniMaxH3Transformer3DModel`` for the coarse-to-fine block-sparse VSA kernel (reusing the
# in-repo SLA Triton kernel) plus FastVideo's learned ``to_gate_compress`` compressed-global
# compensation branch.
#
# MiniMax-H3 runs full self-attention over ONE packed multi-modal sequence (video + text + audio
# rows) through a swappable attention *processor*, so the VSA op is wired in at the processor
# level (``FastVideoMiniMaxH3AttnProcessor``) and the trainable ``attn_op`` gate is attached to
# each main block's ``MiniMaxH3Attention`` module. The token refiner keeps the dense processor.
#
# VSA reference: Zhang et al., "VSA: Faster Video Diffusion with Trainable Sparse Attention",
# arXiv:2505.13389 (FastVideo / hao-ai-lab).
import torch
import torch.nn.functional as F

from .attention_utils import VSASparseAttention, attention
from .minimax_h3_transformer3d import (MiniMaxH3Transformer3DModel,
                                       apply_minimax_h3_rotary_emb)


class FastVideoMiniMaxH3AttnProcessor:
    r"""MiniMax-H3 attention processor whose dense softmax kernel is swapped for VSA block-sparse
    attention (sparse fine branch + gated mean-pooled compressed branch).

    Mirrors ``MiniMaxH3AttnProcessor.__call__`` exactly, except the padless single-document path
    ``attention(q, k, v, causal=False)`` becomes the VSA op ``attn.attn_op(q, k, v, x)``. Two
    cases keep the original DENSE attention:
      - ``attention_mask is not None``: the packed sequence carries padding rows kept in their own
        attention document; the VSA kernel does not accept an arbitrary mask, so fall back to the
        masked SDPA path (identical to the base processor).
      - ``attn.topk >= 1.0``: sparsity disabled -> the base dense full-softmax path.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb=None,
        attention_mask=None,
        valid_length=None,
        vsa_segments=None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if rotary_emb is not None:
            query = apply_minimax_h3_rotary_emb(query, *rotary_emb)
            key = apply_minimax_h3_rotary_emb(key, *rotary_emb)

        # VSA on the padless single-document path only. query/key/value are [B, L, H, D] POST-RoPE
        # and hidden_states is the [B, L, C] attention input that drives the compress gate; the op
        # returns [B, L, H, D], matching the dense paths below so the common tail is unchanged.
        if attention_mask is None and getattr(attn, "topk", 1.0) < 1.0:
            hidden_states = attn.attn_op(
                query, key, value, hidden_states, segments=vsa_segments)
        elif attention_mask is None:
            hidden_states = attention(query, key, value, causal=False)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=attention_mask[None, None],
            ).transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class FastVideoMiniMaxH3Transformer3DModel(MiniMaxH3Transformer3DModel):
    r"""MiniMax-H3 transformer (joint video + audio) with a VSA sparse self-attention on the main
    block stack. Block type stays ``MiniMaxH3TransformerBlock``, so the existing
    ``--fsdp_transformer_layer_cls_to_wrap=MiniMaxH3TransformerBlock`` policy keeps working.

    """

    # NOTE: diffusers from_pretrained filters config.json keys by __init__ signature, so every
    # parent parameter must be declared explicitly (a *args/**kwargs passthrough would silently
    # drop hidden_size/num_layers/... and build a wrongly-sized model).
    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size=(1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        student_topk=0.2,
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            ffn_dim=ffn_dim,
            in_channels=in_channels,
            audio_in_channels=audio_in_channels,
            patch_size=patch_size,
            text_dim=text_dim,
            freq_dim=freq_dim,
            time_embed_hidden_dim=time_embed_hidden_dim,
            time_embed_dim=time_embed_dim,
            rope_freq_dim=rope_freq_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )
        self.student_topk = student_topk
        self.register_to_config(student_topk=student_topk)
        self._apply_sparse_self_attention(student_topk)

    def _apply_sparse_self_attention(self, topk):
        # Sparsify only the main block stack (the heavy video+audio attention); the text token
        # refiner keeps the dense processor. The VSA op is attached to the existing attention
        # module (an nn.Module) so its trainable gate lands in the model's parameter tree, and the
        # processor is swapped to route the padless path through it.
        for block in self.transformer_blocks:
            attn = block.attn
            attn.topk = topk
            attn.set_processor(FastVideoMiniMaxH3AttnProcessor())
            if topk >= 1.0:
                # Dense fallback: drop any residual VSA submodule (e.g. one left on meta by an
                # earlier low_cpu_mem_usage __init__) so no stale meta params reach the offloader.
                if hasattr(attn, "attn_op"):
                    del attn.attn_op
                continue
            # ``dim`` sizes to_gate_compress's input, which is the attention-input hidden states
            # (hidden_size = to_q.in_features), NOT inner_dim (heads * head_dim).
            attn_op = VSASparseAttention(
                attn.head_dim, attn.heads, attn.to_q.in_features, topk=topk)
            # Under low_cpu_mem_usage from_pretrained the backbone params are meta tensors and the
            # freshly built gate is materialized by a later set_student_topk (called after
            # from_pretrained) or the checkpoint load; only align device/dtype when concrete, and
            # match the backbone dtype so bf16 weights and an fp32 gate never mix under FSDP.
            p0 = attn.to_q.weight
            if not p0.is_meta:
                attn_op = attn_op.to(device=p0.device, dtype=p0.dtype)
            attn.attn_op = attn_op

    def set_student_topk(self, topk):
        self.student_topk = topk
        self.register_to_config(student_topk=topk)
        if topk < 1.0 and all(hasattr(block.attn, "attn_op") and
                              not any(p.is_meta for p in block.attn.attn_op.parameters())
                              for block in self.transformer_blocks):
            for block in self.transformer_blocks:
                block.attn.topk = topk
                block.attn.attn_op.topk = topk
        else:
            self._apply_sparse_self_attention(topk)

    def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states,
                timestep, timestep_indices, token_tags, position_ids,
                video_indices, audio_indices, text_indices,
                attention_kwargs=None, return_dict=True):
        attention_kwargs = dict(attention_kwargs or {})
        attention_kwargs["vsa_segments"] = self._compute_h3_segments(token_tags, position_ids)
        return super().forward(
            hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
            timestep_indices, token_tags, position_ids, video_indices, audio_indices,
            text_indices, attention_kwargs=attention_kwargs, return_dict=return_dict)

    @staticmethod
    def _video_grid_from_positions(pos):
        """Recover the (T, H, W) token grid of one contiguous video run from its rotary
        ``position_ids`` [n, 3]. The video rows are frame-major then (h, w) raster with w
        fastest: frames carry a distinct temporal coord, and within a frame the leading run
        of an identical height coord spans exactly one width row. Returns None on mismatch."""
        n = pos.shape[0]
        if n == 0:
            return None
        t_col = pos[:, 0]
        rows_per_frame = int((t_col == t_col[0]).sum())   # frame 0's temporal coord is unique
        if rows_per_frame <= 0 or n % rows_per_frame != 0:
            return None
        h_col = pos[:rows_per_frame, 1]
        W = int((h_col == h_col[0]).to(torch.int32).cumprod(0).sum())  # one height row
        if W <= 0 or rows_per_frame % W != 0:
            return None
        H = rows_per_frame // W
        T = n // rows_per_frame
        if T * H * W != n:
            return None
        return (T, H, W)

    def _compute_h3_segments(self, token_tags, position_ids):
        """Packed-order segment list for ``build_h3_tile_geometry``: dense run lengths
        (int) plus one (t, h, w) triple per video region, in packed order. Supports
        multiple video regions (Ref2VA / PDD reference videos sit as extra tag-0 runs and
        each becomes its own sparse region, exempt top-k per region). Returns None (flat
        fallback) when the sequence is padded or any video run's grid cannot be recovered."""
        if token_tags.numel() == 0 or bool((token_tags < 0).any()):
            return None
        tags = token_tags.detach().to(torch.int64).cpu()
        change = torch.nonzero(tags[1:] != tags[:-1]).flatten() + 1
        bounds = torch.cat([torch.zeros(1, dtype=change.dtype), change,
                            torch.tensor([tags.numel()], dtype=change.dtype)])
        bounds = [int(b) for b in bounds.tolist()]
        segments = []
        video_runs = 0
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            if e <= s:
                continue
            if int(tags[s]) == 0:                          # video run
                video_runs += 1
                grid = self._video_grid_from_positions(position_ids[s:e])
                if grid is None:
                    return None
                segments.append(grid)
            else:                                          # dense (text / audio) run
                segments.append(e - s)
        if video_runs < 1:
            return None
        return tuple(segments)

    def enable_multi_gpus_inference(self,):
        """VSA has no sequence-parallel path in this repo.

        The in-repo VSA kernel only implements the single-GPU coarse-to-fine block-sparse forward;
        there is no ulysses head-parallel or ring-attention binding for it (unlike the base
        ``MiniMaxH3MultiGPUsAttnProcessor``). Fail loudly rather than silently falling back to a
        DENSE multi-GPU processor (which would bypass VSA and drop the gate params).
        """
        raise NotImplementedError(
            "FastVideoMiniMaxH3Transformer3DModel (VSA) supports single-GPU inference only; the "
            "VSA kernel has no ulysses/ring sequence-parallel path. Set ulysses_degree=ring_degree=1.")

    def sparse_trainable_params(self):
        """Names of the extra VSA-branch parameters, for logging / freezing."""
        return [name for name, _ in self.named_parameters() if "attn_op." in name]
