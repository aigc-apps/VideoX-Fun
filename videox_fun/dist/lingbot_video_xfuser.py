import torch

from .fuser import xFuserLongContextAttention


def usp_attn_lingbot_video_forward(
    self,
    x,
    rotary_emb,
    attention_mask=None,
    num_video_tokens=None,
):
    """Multi-GPU replacement for LingBotVideoAttention.forward.

    LingBot-Video runs a single joint self-attention over ``[video; text]``. Only the
    video tokens are sharded across ranks; the text tokens stay replicated on every
    rank and are handed to xFuser as joint tensors. That keeps the result identical
    to single-GPU inference: attention is invariant to key order, every query still
    sees every key exactly once, and no padding token is ever introduced.

    ``x`` and ``rotary_emb`` are the rank-local slices ``[video_local; text]``, and
    ``num_video_tokens`` is the local video length.
    """
    from ..models.lingbot_video_transformer3d import apply_rotary_emb

    if attention_mask is not None:
        raise ValueError(
            "Sequence-parallel LingBot-Video attention cannot apply an attention mask. "
            "Pass unpadded text embeddings (batch size 1) instead."
        )
    if num_video_tokens is None:
        raise ValueError("`num_video_tokens` is required for sequence-parallel attention.")

    seq_len = x.shape[1]
    q = self.to_q(x).unflatten(2, (self.num_heads, self.head_dim))
    k = self.to_k(x).unflatten(2, (self.num_heads, self.head_dim))
    v = self.to_v(x).unflatten(2, (self.num_heads, self.head_dim))
    q = apply_rotary_emb(self.norm_q(q), rotary_emb)
    k = apply_rotary_emb(self.norm_k(k), rotary_emb)

    half_dtypes = (torch.float16, torch.bfloat16)

    def half(tensor):
        return tensor if tensor.dtype in half_dtypes else tensor.to(torch.bfloat16)

    nv = num_video_tokens
    out = xFuserLongContextAttention()(
        None,
        half(q[:, :nv]).contiguous(),
        half(k[:, :nv]).contiguous(),
        half(v[:, :nv]).contiguous(),
        joint_tensor_query=half(q[:, nv:]).contiguous(),
        joint_tensor_key=half(k[:, nv:]).contiguous(),
        joint_tensor_value=half(v[:, nv:]).contiguous(),
        joint_strategy="rear",
    )
    if out.shape[1] != seq_len:
        raise RuntimeError(
            f"Sequence-parallel attention returned {out.shape[1]} tokens, expected {seq_len}."
        )
    return self.to_out(out.flatten(2, 3).type_as(x))
