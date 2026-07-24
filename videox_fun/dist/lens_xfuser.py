from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .fuser import xFuserLongContextAttention


def usp_lens_joint_attention_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-GPU replacement for LensJointAttention.forward using ring/ulysses attention.

    Follows the same pattern as Flux2MultiGPUsAttnProcessor2_0:
    - Image tokens are the main sequence (distributed across GPUs via ring attention).
    - Text tokens are passed as joint_tensor (replicated on all GPUs).

    The caller (LensTransformer2DModel.forward) is expected to have already
    chunked ``hidden_states`` and image-side ``image_rotary_emb`` along the
    image sequence dimension by ``sp_world_size``. attention_mask is ignored
    on this path; text padding contamination is accepted as a tradeoff with
    xFuser's flash-attn backend.
    """
    from ..models.lens_transformer2d import apply_rotary_emb_lens

    bsz, seq_img, _ = hidden_states.shape
    seq_txt = encoder_hidden_states.shape[1]

    # Fused QKV per stream -> split.
    img_qkv = self.img_qkv(hidden_states).view(bsz, seq_img, 3, self.heads, self.dim_head)
    txt_qkv = self.txt_qkv(encoder_hidden_states).view(bsz, seq_txt, 3, self.heads, self.dim_head)
    img_q, img_k, img_v = img_qkv.unbind(dim=2)
    txt_q, txt_k, txt_v = txt_qkv.unbind(dim=2)

    # QK RMSNorm.
    img_q = self.norm_q(img_q)
    img_k = self.norm_k(img_k)
    txt_q = self.norm_added_q(txt_q)
    txt_k = self.norm_added_k(txt_k)

    # RoPE.
    img_freqs, txt_freqs = image_rotary_emb
    if img_freqs.shape[0] < seq_img:
        raise ValueError(
            f"Image RoPE length {img_freqs.shape[0]} is shorter than "
            f"image sequence length {seq_img}."
        )
    img_freqs = img_freqs[:seq_img]
    img_q = apply_rotary_emb_lens(img_q, img_freqs)
    img_k = apply_rotary_emb_lens(img_k, img_freqs)
    if seq_txt > 0:
        if txt_freqs.shape[0] < seq_txt:
            raise ValueError(
                f"Text RoPE length {txt_freqs.shape[0]} is shorter than "
                f"text sequence length {seq_txt}."
            )
        txt_freqs = txt_freqs[:seq_txt]
        txt_q = apply_rotary_emb_lens(txt_q, txt_freqs)
        txt_k = apply_rotary_emb_lens(txt_k, txt_freqs)

    half_dtypes = (torch.float16, torch.bfloat16)
    def half(x):
        return x if x.dtype in half_dtypes else x.to(torch.bfloat16)

    # Use xFuserLongContextAttention with joint_strategy='front'
    # Image tokens are distributed via ring attention, text tokens are replicated (joint).
    out = xFuserLongContextAttention()(
        None,
        half(img_q), half(img_k), half(img_v),
        dropout_p=0.0, causal=False,
        joint_tensor_query=half(txt_q),
        joint_tensor_key=half(txt_k),
        joint_tensor_value=half(txt_v),
        joint_strategy='front',
    )
    out = out.flatten(2, 3)
    out = out.to(img_q.dtype)

    # With joint_strategy='front', output order is [txt, img].
    txt_out_raw, img_out_raw = out.split_with_sizes([seq_txt, out.shape[1] - seq_txt], dim=1)

    img_out = self.to_out[1](self.to_out[0](img_out_raw))
    txt_out = self.to_add_out(txt_out_raw)
    return img_out, txt_out
