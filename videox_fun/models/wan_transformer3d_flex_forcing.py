# Flex-Forcing: Towards a Unified Autoregressive and Bidirectional Video
# Diffusion Model (arXiv 2607.03509)
#
# Built directly on top of the Self-Forcing backbone in
# videox_fun/models/wan_transformer3d_self_forcing.py: the block class, the
# parameter names, the KV-cache bookkeeping, the Forcing-KV hooks and the
# sequence-parallel path are all inherited unchanged, so a Wan2.1 / CausVid /
# Self-Forcing checkpoint loads into this model as-is and only the new
# `flex_kproj.*` tensors are reported missing.
"""Flex-Forcing Wan backbone.

Flex-Forcing generalises Self-Forcing along two axes:

* **Frame axis (§3.1)** - a partition ``a_t`` of the latent frames replaces the
  scalar ``num_frame_per_block``. Attention is bidirectional inside a chunk and
  autoregressive across chunks, so ``[1]*F`` recovers pure Self-Forcing and
  ``[F]`` recovers pure bidirectional diffusion. The partition algebra lives in
  :mod:`videox_fun.utils.flex_chunking`; this module only turns a partition into
  a FlexAttention block mask (training) or into a KV-cache attention window
  (inference).
* **Denoising-timestep axis (§3.2)** - the partition may itself change with the
  noise level (large chunks while planning, small chunks while refining), which
  the pipeline drives through nested partitions.

The one genuinely new piece of network here is the **noise-level aligned
K-Projection** ``Π_{t←0}`` (§3.3): cached keys are produced by a *clean* context
pass (timestep ``context_noise``) but are consumed while denoising at noise
level ``t``. Π maps them into the key space of level ``t``. It is a lightweight
timestep-conditioned linear map, applied on the fly at attention time (the cache
tensors themselves are never modified), and **identity-initialised** so a freshly
converted Self-Forcing checkpoint behaves exactly as before step 0.
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from torch.nn.attention.flex_attention import create_block_mask

from ..utils.flex_chunking import chunk_boundaries, chunk_ends_tensor
from .attention_utils import attention
from .wan_transformer3d import rope_apply, sinusoidal_embedding_1d
from .wan_transformer3d_self_forcing import (CasualWanSelfAttention,
                                             WanTransformer3DModel_SelfForcing,
                                             causal_rope_apply, flex_attention)

__all__ = ["FlexKProjection", "FlexWanSelfAttention",
           "WanTransformer3DModel_FlexForcing"]


def _first_frame_timestep(t: torch.Tensor) -> torch.Tensor:
    """Collapse a ``[B]`` or ``[B, F]`` timestep tensor to ``[B]``.

    One noise level per chunk: every frame of the chunk being denoised shares
    its timestep in the Flex-Forcing rollout, so the first frame's entry is
    the chunk's noise level.
    """
    return t.reshape(t.shape[0], -1)[:, 0]


class FlexKProjection(nn.Module):
    r"""Noise-level aligned key projection :math:`\Pi_{t\leftarrow 0}` (Flex-Forcing §3.3).

    A per-head, timestep-conditioned affine map applied to **cached** keys only:

    .. math::
        \tilde K_t = \mathrm{concat}\big(\Pi_{t\leftarrow 0}(K_0^{<F_{t,k}}),\;
                                          \tilde K_t^{\,F_{t,k}}\big)

    The chunk currently being denoised already produces its keys at level ``t``
    and is therefore passed through untouched; everything else in the visible
    window was written by the clean context pass at ``t = context_noise`` and
    gets projected.

    Parameterisation (deliberately tiny - ``3 * num_heads * head_dim`` outputs
    per layer, i.e. ~0.1% of a Wan-1.3B block):

    .. math::
        \Pi_{t\leftarrow 0}(k) = k \odot (1 + \gamma_t)
                                 + \langle k, u_t\rangle \otimes v_t

    with ``(γ_t, u_t, v_t)`` produced from the sinusoidal embedding of ``t``.
    The output layer is **zero-initialised**, which makes Π the exact identity
    at step 0 - the paper requires this so distillation starts from the
    unmodified Self-Forcing behaviour.

    The map is applied on the fly: no cache tensor is rewritten and no gradient
    flows back into the cache (which was filled under ``no_grad``), so only Π's
    own parameters are trained by this branch. The paper trains it with the same
    small learning rate as the generator (2e-6).
    """

    MODES = ("diag", "diag_rank1")

    def __init__(self,
                 num_heads: int,
                 head_dim: int,
                 freq_dim: int = 256,
                 mode: str = "diag_rank1",
                 hidden_dim: Optional[int] = None,
                 slice_tokens: int = 8192):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(
                f"flex_kproj_mode must be one of {self.MODES}, got {mode!r}")
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.freq_dim = int(freq_dim)
        self.mode = mode
        # Projected key tensors are float32 for numerical stability (γ sits next
        # to 1.0, where bf16 would quantise small corrections away); long
        # contexts are processed in slices to bound the transient footprint.
        self.slice_tokens = max(1024, int(slice_tokens))

        hidden_dim = int(hidden_dim) if hidden_dim is not None else self.freq_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(self.freq_dim, hidden_dim), nn.GELU(approximate='tanh'))
        # Named `proj_out` on purpose: WanTransformer3DModel.from_pretrained's
        # initialize_missing_parameters() zero-fills every missing key containing
        # "proj_out", which preserves the identity init under
        # low_cpu_mem_usage=True (meta device), where __init__'s own
        # nn.init.zeros_ calls are no-ops.
        self.proj_out = nn.Linear(hidden_dim, self.num_heads * self.head_dim * 3)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def _params(self, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """(γ, u, v) for each batch entry, float32, shape [B, 3, heads, dim]."""
        if t.dim() == 0:
            t = t.reshape(1)
        t = _first_frame_timestep(t)
        emb = sinusoidal_embedding_1d(
            self.freq_dim, t.to(ref.device).float()).to(self.time_mlp[0].weight.dtype)
        params = self.proj_out(self.time_mlp(emb))
        return params.view(-1, 3, self.num_heads, self.head_dim).float()

    def forward(self, k: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Project cached keys ``k`` ([B, L, heads, dim]) from level 0 to level ``t``."""
        params = self._params(t, k)
        gamma, u, v = params.unbind(1)
        out = torch.empty_like(k)
        length = k.shape[1]
        step = min(length, self.slice_tokens)
        for start in range(0, length, step):
            end = min(start + step, length)
            k_f = k[:, start:end].float()
            k_t = k_f * (1.0 + gamma.unsqueeze(1))
            if self.mode == "diag_rank1":
                k_t = k_t + (k_f * u.unsqueeze(1)).sum(-1, keepdim=True) * v.unsqueeze(1)
            out[:, start:end] = k_t.to(k.dtype)
        return out


class FlexWanSelfAttention(CasualWanSelfAttention):
    """Self-attention with a flexible chunk window and the K-Projection.

    Self-contained: :meth:`forward` is a full copy of the Self-Forcing
    attention forward with the two Flex-Forcing insertions inlined, so the
    Self-Forcing base class carries no Flex-Forcing seam. QKV projection, RoPE,
    the rolling eviction arithmetic and the cache index bookkeeping are copied
    verbatim from the base; only the clean-half key projection (teacher forcing)
    and the KV-window read (inference) differ. The Flex-Forcing state reaches
    this layer through the existing ``forcing_kv_state`` channel under the
    ``"flex"`` key, which keeps it compatible with gradient checkpointing (the
    same dict object is replayed during recomputation).

    ``forcing_kv_state["flex"]`` may contain:

    * ``timestep`` - ``[B]`` noise level of the chunk being denoised (drives Π).
    * ``attn_window`` - ``(lo_token, hi_token)``; widens the visible window so
      an edit chunk can also attend to *future* clean tokens (§4.2 any-order /
      any-timestep editing). ``None`` keeps the causal window.

    There is no switch for Π: a layer that was built with a projection always
    applies it, and ``flex_kproj_mode='none'`` is the only way to leave it out.

    Forcing-KV (arXiv 2605.09681) and Flex-Forcing are orthogonal papers; when
    Forcing-KV's grouped path is active it owns the window read, so Π is
    bypassed rather than silently applied twice.
    """

    def __init__(self,
                 *args,
                 flex_kproj_mode: str = "diag_rank1",
                 flex_kproj_freq_dim: int = 256,
                 flex_kproj_slice_tokens: int = 8192,
                 **kwargs):
        # The base signature is forwarded verbatim so it cannot drift out of
        # sync; only the Flex-Forcing knobs are declared explicitly.
        super().__init__(*args, **kwargs)
        self.flex_kproj_mode = flex_kproj_mode
        self.flex_kproj = None if flex_kproj_mode == "none" else FlexKProjection(
            self.num_heads, self.head_dim, freq_dim=flex_kproj_freq_dim,
            mode=flex_kproj_mode, slice_tokens=flex_kproj_slice_tokens)

    @staticmethod
    def _flex_of(forcing_kv_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return (forcing_kv_state or {}).get("flex") or {}

    def _flex_window(self, window_start: int, local_start_index: int,
                     local_end_index: int, flex: Dict[str, Any]):
        """Visible token range, optionally widened for any-order editing (§4.2).

        Editing a middle span ``[a, b)`` of an already generated clip needs the
        clean tokens on *both* sides, i.e. the window ``[0, F)``. Attention is
        invariant to the KV ordering (RoPE is already baked into the cached
        keys), so widening the range is enough - no re-permutation is required.
        """
        lo, hi = window_start, local_end_index
        attn_window = flex.get("attn_window")
        if attn_window is not None:
            lo = max(0, min(int(attn_window[0]), local_start_index))
            hi = max(int(attn_window[1]), local_end_index)
        return lo, hi

    def _flex_project(self, cached_k: torch.Tensor, lo: int,
                      local_start_index: int, local_end_index: int,
                      flex: Dict[str, Any]) -> torch.Tensor:
        """Apply Π to every cached key except the chunk currently denoised."""
        if self.flex_kproj is None:
            return cached_k
        timestep = flex.get("timestep")
        if timestep is None:
            return cached_k
        length = cached_k.shape[1]
        # Current chunk sits at [local_start_index, local_end_index) in cache
        # coordinates; translate into window-relative offsets.
        past_end = max(0, min(local_start_index - lo, length))
        cur_end = max(past_end, min(local_end_index - lo, length))
        parts = []
        if past_end > 0:
            parts.append(self.flex_kproj(cached_k[:, :past_end], timestep))
        parts.append(cached_k[:, past_end:cur_end])
        if cur_end < length:
            parts.append(self.flex_kproj(cached_k[:, cur_end:], timestep))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    def forward(
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
        t=0,
        forcing_kv_state=None,
    ):
        r"""Self-Forcing attention forward with the Flex-Forcing insertions.

        Verbatim copy of ``CasualWanSelfAttention.forward`` except for two
        inlined Flex-Forcing hooks (kept here so the Self-Forcing base stays
        pure):

        * teacher-forcing path - the clean half ``roped_key[0]`` goes through
          the noise-level aligned K-Projection Π (§3.3); this is what puts the
          K-Projection into the training graph, since the block-mask path never
          builds a KV cache;
        * KV-cache path - the visible window is optionally widened for
          any-order / any-timestep editing (§4.2) and every cached key outside
          the chunk being denoised is projected by Π.

        With no ``forcing_kv_state["flex"]`` the result is bit-identical to the
        base forward. Note (sequence parallel): the SP forward is bound
        externally by ``usp_attn_self_forcing_forward`` and does not route
        through this method, so Π is not applied on the SP KV-cache path;
        shipped Flex-Forcing inference is single-GPU.
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start
        flex = self._flex_of(forcing_kv_state)

        # Query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        if kv_cache is None:
            # Check if this is teacher forcing training (sequence length is doubled)
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                # Split into clean and noisy parts for teacher forcing
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # Apply same RoPE to both clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                # Flex-Forcing §3.3: `roped_key[0]` is the clean half
                # (`x = cat([clean_x, x])` in the model forward) and the mask
                # lets a noisy chunk read only the clean tokens strictly before
                # it - exactly the K^{<F} that Π projects. Identity when the
                # K-Projection is off or no timestep is supplied.
                if (self.flex_kproj is not None
                        and flex.get("timestep") is not None):
                    roped_key[0] = self.flex_kproj(roped_key[0], flex["timestep"])

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                # Pad to 128 multiple for flex attention
                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                # Apply flex attention with block mask
                if padded_length != 0:
                    x = flex_attention(
                        query=padded_roped_query.transpose(2, 1),
                        key=padded_roped_key.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask
                    )[:, :, :-padded_length].transpose(2, 1)
                else:
                    x = flex_attention(
                        query=padded_roped_query.transpose(2, 1),
                        key=padded_roped_key.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask
                    ).transpose(2, 1)
            else:
                # Standard inference without teacher forcing
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                # Pad to 128 multiple for flex attention
                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                # Apply flex attention with block mask
                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            # Causal inference with KV cache
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            # Apply causal RoPE with frame offset
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]

            # Forcing-KV: per-cache compression state, lazily created on first
            # use. Before the switch step (ar_start) every layer attends over
            # [sink + recent history + current chunk]; after the switch, layer 0
            # keeps that path while layers >= 1 split heads into static/dynamic
            # groups with separate history budgets (official architecture).
            forcing_kv_active = bool(
                self.forcing_kv_enable and self.forcing_kv_static_heads is not None)
            clean_pass = bool(forcing_kv_state is not None and forcing_kv_state.get("clean_pass", False))
            fkv = None
            fkv_switched = False
            if forcing_kv_active:
                if "forcing_kv" not in kv_cache:
                    kv_cache["forcing_kv"] = {
                        "switched_at": None,
                        "dyn_k": None,
                        "dyn_v": None,
                        "dyn_valid": 0,
                    }
                fkv = kv_cache["forcing_kv"]
                fpb = max(1, int(self.num_frame_per_block))
                ar_step = current_start // (frame_seqlen * fpb)
                if clean_pass and ar_step >= self.forcing_kv_ar_start and fkv["switched_at"] is None:
                    # Mirror the official switch: it only takes effect from the
                    # next chunk on, so the switching chunk itself stays
                    # ungrouped.
                    fkv["switched_at"] = ar_step
                fkv_switched = fkv["switched_at"] is not None and ar_step > fkv["switched_at"]

            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v

            # Compute attention with local window
            if self.local_attn_size == -1:
                max_attention_size = local_end_index
            else:
                max_attention_size = self.local_attn_size * frame_seqlen
            window_start = max(0, local_end_index - max_attention_size)
            if forcing_kv_active:
                x = self._forcing_kv_grouped_attention(
                    roped_query, kv_cache, fkv, fkv_switched,
                    local_start_index, local_end_index, frame_seqlen)
                if clean_pass and fkv_switched and self.forcing_kv_layer_idx != 0:
                    self._forcing_kv_dynamic_update(
                        kv_cache, fkv, local_start_index, local_end_index,
                        frame_seqlen, forcing_kv_state=forcing_kv_state)
            else:
                # Flex-Forcing KV-cache path: optionally widen the visible
                # window (§4.2) and project every cached key except the chunk
                # being denoised (§3.3). With no flex state this is the plain
                # causal window read, bit-identical to the base forward.
                lo, hi = self._flex_window(window_start, local_start_index,
                                           local_end_index, flex)
                cached_k = kv_cache["k"][:, lo:hi]
                cached_v = kv_cache["v"][:, lo:hi]
                cached_k = self._flex_project(cached_k, lo, local_start_index,
                                              local_end_index, flex)
                x = attention(roped_query, cached_k, cached_v)
            # Expose the attention window for offline head profiling
            kv_cache["_fkv_last_q"] = roped_query
            kv_cache["_fkv_window_start"] = window_start
            kv_cache["_fkv_local_end"] = local_end_index
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # Output projection
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanTransformer3DModel_FlexForcing(WanTransformer3DModel_SelfForcing):
    r"""
    Wan diffusion backbone for Flex-Forcing (arXiv 2607.03509), supporting both
    text-to-video and image-to-video.

    Inherits everything from ``WanTransformer3DModel_SelfForcing`` and adds:

    * the flexible frame-axis partition used by the FlexAttention training masks
      (:meth:`set_flex_chunk_sizes`);
    * the noise-level aligned K-Projection (:class:`FlexKProjection`), enabled
      per forward through ``forcing_kv_state["flex"]["timestep"]``.

    Blocks stay ``CasualWanAttentionBlock`` instances with unchanged parameter
    names - only ``block.self_attn`` is upgraded to
    :class:`FlexWanSelfAttention`. That keeps FSDP's
    ``--fsdp_transformer_layer_cls_to_wrap=CasualWanAttentionBlock`` working and
    lets existing Self-Forcing / CausVid checkpoints load without remapping.
    """
    _supports_gradient_checkpointing = True

    @register_to_config
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

        # Self-Forcing causal inference parameters (inherited)
        local_attn_size=-1,
        sink_size=0,

        # Forcing-KV hybrid KV cache compression (arXiv 2605.09681, inherited)
        forcing_kv_enable=False,
        forcing_kv_head_profile=None,
        forcing_kv_ar_start=1,
        forcing_kv_spatial_context_length=1,
        forcing_kv_temporal_context_length=1,
        forcing_kv_dynamic_context_length=1,
        forcing_kv_num_frame_patch=6,
        forcing_kv_sim_retention_ratio=0.33,

        # Flex-Forcing noise-level aligned K-Projection (arXiv 2607.03509 §3.3)
        flex_kproj_mode='diag_rank1',
        flex_kproj_freq_dim=256,
        flex_kproj_slice_tokens=8192,
    ):
        r"""
        Initialize the Flex-Forcing diffusion backbone.

        All arguments up to ``forcing_kv_sim_retention_ratio`` are documented on
        :class:`WanTransformer3DModel_SelfForcing`; the Flex-Forcing specific
        ones are:

        Args:
            flex_kproj_mode (`str`, *optional*, defaults to 'diag_rank1'):
                K-Projection parameterisation - ``'none'`` disables it entirely
                (the model then behaves exactly like Self-Forcing with variable
                chunk sizes), ``'diag'`` keeps only the per-head/per-dim scaling
                ``k ⊙ (1 + γ_t)``, ``'diag_rank1'`` adds the rank-1 term
                ``⟨k, u_t⟩ ⊗ v_t`` (the paper's setting).
            flex_kproj_freq_dim (`int`, *optional*, defaults to 256):
                Sinusoidal timestep embedding width feeding the K-Projection.
            flex_kproj_slice_tokens (`int`, *optional*, defaults to 8192):
                Token slice length used when projecting long cached contexts in
                float32, bounding the transient memory footprint.
        """
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
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            forcing_kv_enable=forcing_kv_enable,
            forcing_kv_head_profile=forcing_kv_head_profile,
            forcing_kv_ar_start=forcing_kv_ar_start,
            forcing_kv_spatial_context_length=forcing_kv_spatial_context_length,
            forcing_kv_temporal_context_length=forcing_kv_temporal_context_length,
            forcing_kv_dynamic_context_length=forcing_kv_dynamic_context_length,
            forcing_kv_num_frame_patch=forcing_kv_num_frame_patch,
            forcing_kv_sim_retention_ratio=forcing_kv_sim_retention_ratio,
        )

        # Upgrade each block's self-attention in place via a class swap. The
        # parent __init__ has already built the blocks and run init_weights(),
        # so the initialised q/k/v/o parameters, the Forcing-KV attributes and
        # every other piece of per-layer state carry over untouched - no
        # re-creation, no state_dict round-trip and no attribute allow-list
        # that could silently drop a newly added base attribute. Block class
        # and parameter names stay unchanged on purpose (see the class
        # docstring).
        for block in self.blocks:
            attn = block.self_attn
            attn.__class__ = FlexWanSelfAttention
            attn.flex_kproj_mode = flex_kproj_mode
            attn.flex_kproj = None if flex_kproj_mode == "none" \
                else FlexKProjection(
                    attn.num_heads, attn.head_dim,
                    freq_dim=flex_kproj_freq_dim, mode=flex_kproj_mode,
                    slice_tokens=flex_kproj_slice_tokens)

        if flex_kproj_mode != 'none':
            # Defensive re-zeroing: FlexKProjection.__init__ already starts
            # proj_out at zero, but any init_weights() pass that runs after
            # construction (e.g. diffusers' post_init hooks) would xavier it
            # again and break the exact identity start the paper requires.
            # Under low_cpu_mem_usage these are meta tensors (no-op) and
            # initialize_missing_parameters() zero-fills them instead.
            for block in self.blocks:
                kproj = block.self_attn.flex_kproj
                nn.init.zeros_(kproj.proj_out.weight)
                nn.init.zeros_(kproj.proj_out.bias)

        # Flex-Forcing runtime state
        self.flex_kproj_mode = flex_kproj_mode
        # Frame-axis partition for the FlexAttention (no KV cache) training path.
        # None => fall back to the inherited uniform num_frame_per_block masks.
        self.flex_chunk_sizes = None

    # ------------------------------------------------------------------
    # Frame-axis partition (§3.1)
    # ------------------------------------------------------------------
    def set_flex_chunk_sizes(self, chunk_sizes: Optional[Sequence[int]]):
        """Set the frame-axis partition used by the block-mask training path.

        ``None`` restores the inherited uniform ``num_frame_per_block`` masks.
        The cached-mask guard of the parent forward is invalidated so a new
        random partition per rollout (flexible-chunk training, §3.3) always
        rebuilds its FlexAttention mask.
        """
        self.flex_chunk_sizes = None if chunk_sizes is None else [int(c) for c in chunk_sizes]
        self._block_mask_expected_len = None

    @staticmethod
    def _require_exact_partition(chunk_sizes: Sequence[int],
                                 num_frames: int) -> List[int]:
        """Validate ``chunk_sizes`` as an exact partition of ``num_frames``."""
        chunk_sizes = [int(c) for c in chunk_sizes]
        if sum(chunk_sizes) != num_frames:
            raise ValueError(
                f"chunk_sizes {chunk_sizes} cover {sum(chunk_sizes)} latent "
                f"frames but num_frames={num_frames}.")
        return chunk_sizes

    @staticmethod
    def _padded_length(seq_len: int) -> int:
        """Right padding to a multiple of 128, as FlexAttention requires."""
        return math.ceil(seq_len / 128) * 128 - seq_len

    def _build_block_mask(self, attention_mask, seq_len: int, device):
        """Pad ``seq_len`` to a multiple of 128 and store ``self.block_mask``."""
        padded_length = self._padded_length(seq_len)
        self.block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=seq_len + padded_length,
            KV_LEN=seq_len + padded_length,
            _compile=True,
            device=device
        )

    def create_flexible_block_mask(
        self,
        chunk_sizes: Sequence[int],
        num_frames: int,
        frame_seqlen: int,
        device: Union[torch.device, str] = "cpu",
    ):
        """Block mask for an arbitrary partition (variable-size chunks).

        Generalises :meth:`create_block_mask_for_training` from equal blocks to
        the partition ``chunk_sizes``: a query token sees its whole own chunk
        (bidirectional) plus every token of earlier chunks (autoregressive),
        optionally clipped to ``local_attn_size`` frames.
        """
        chunk_sizes = self._require_exact_partition(chunk_sizes, num_frames)
        total_length = num_frames * frame_seqlen

        # Padded query rows keep ends=0 exactly like the inherited uniform
        # builder, so they only attend to themselves and are sliced away later.
        ends = torch.zeros(
            total_length + self._padded_length(total_length),
            device=device, dtype=torch.long)
        ends[:total_length] = chunk_ends_tensor(chunk_sizes, frame_seqlen, device=device)

        local_attn_size = self.local_attn_size

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                # Global chunk-wise causal: bidirectional inside a chunk, all
                # previous chunks visible.
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            # Local attention: limited window
            return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)

        self._build_block_mask(attention_mask, total_length, device)

        # Consumers that still expect the scalar Self-Forcing knobs (Forcing-KV
        # AR stride / rolling-cache budget) get the conservative largest chunk.
        self.num_frame_per_block = max(chunk_sizes)
        self.independent_first_frame = chunk_sizes[0] == 1

    def create_flexible_teacher_forcing_mask(
        self,
        chunk_sizes: Sequence[int],
        num_frames: int,
        frame_seqlen: int,
        device: Union[torch.device, str] = "cpu",
    ):
        """Teacher-forcing mask for an arbitrary partition.

        Sequence layout is inherited: ``[clean frames..., noisy frames...]``.
        Clean tokens are chunk-causal among themselves; a noisy chunk attends to
        its own noisy tokens plus all clean tokens strictly before it.
        """
        chunk_sizes = self._require_exact_partition(chunk_sizes, num_frames)
        total_length = num_frames * frame_seqlen * 2  # Clean + noisy
        padded_length = self._padded_length(total_length)
        clean_ends = num_frames * frame_seqlen

        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        for start_frame, end_frame in chunk_boundaries(chunk_sizes):
            tok_start = start_frame * frame_seqlen
            tok_end = end_frame * frame_seqlen
            # Clean frames: chunk-wise causal attention
            context_ends[tok_start:tok_end] = tok_end
            # Noisy frames of this chunk
            noisy_slice = slice(clean_ends + tok_start, clean_ends + tok_end)
            noise_noise_starts[noisy_slice] = clean_ends + tok_start
            noise_noise_ends[noisy_slice] = clean_ends + tok_end
            # ... may read every clean token of the previous chunks only
            noise_context_ends[noisy_slice] = tok_start

        def attention_mask(b, h, q_idx, kv_idx):
            # Clean frames mask
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # Noisy frames mask: attend to clean + self
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        self._build_block_mask(attention_mask, total_length, device)

        self.num_frame_per_block = max(chunk_sizes)

    def create_block_mask_for_training(
        self,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
        independent_first_frame: bool = False,
        device: Union[torch.device, str] = "cpu",
    ):
        """Dispatch to the flexible builder when a partition is set."""
        if self.flex_chunk_sizes is None:
            return super().create_block_mask_for_training(
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                num_frame_per_block=num_frame_per_block,
                independent_first_frame=independent_first_frame,
                device=device,
            )
        return self.create_flexible_block_mask(
            self.flex_chunk_sizes, num_frames, frame_seqlen, device)

    def create_teacher_forcing_mask(
        self,
        device: Union[torch.device, str],
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
    ):
        """Dispatch to the flexible builder when a partition is set."""
        if self.flex_chunk_sizes is None:
            return super().create_teacher_forcing_mask(
                device=device,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                num_frame_per_block=num_frame_per_block,
            )
        return self.create_flexible_teacher_forcing_mask(
            self.flex_chunk_sizes, num_frames, frame_seqlen, device)

    # ------------------------------------------------------------------
    # K-Projection plumbing (§3.3) and any-order editing window (§4.2)
    # ------------------------------------------------------------------
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: Optional[dict] = None,
        crossattn_cache: Optional[dict] = None,
        current_start: int = 0,
        cache_start: int = 0,
        clean_x=None,
        aug_t=None,
        forcing_kv_state: Optional[dict] = None,
        flex_state: Optional[dict] = None,
    ):
        r"""
        Run the Flex-Forcing backbone. Identical to the inherited forward plus
        the ``flex_state`` channel; see
        :meth:`WanTransformer3DModel_SelfForcing.forward` for the argument list.

        Args:
            flex_state (Dict, *optional*):
                Flex-Forcing per-forward state forwarded to every self-attention
                layer through ``forcing_kv_state["flex"]``. The only key is
                ``attn_window`` ``(lo_token, hi_token)``, which widens the visible
                KV window for any-order / any-timestep editing (§4.2). The noise
                level Π needs is taken from ``t`` automatically (uniform per
                chunk), so it does not need to be passed.

        Passing no ``flex_state`` on a model built with
        ``flex_kproj_mode='none'`` leaves ``forcing_kv_state`` untouched, i.e.
        bit-identical Self-Forcing behaviour.
        """
        attn_window = (flex_state or {}).get("attn_window")
        # 3.3 has no runtime switch: a transformer built with a projection always
        # applies it, and `flex_kproj_mode == 'none'` is the only way to opt out.
        has_kproj = self.flex_kproj_mode != 'none'
        # `clean_x is not None` is the teacher-forcing / block-mask *training*
        # path (--use_teacher_forcing). Its clean half plays the role the
        # KV-cache's committed clean context plays at inference, so Π has to be
        # armed there as well: the pyramid of 3.2 forces block-mask training
        # (see the --flex_pyramid_levels guard in train_distill.py), and a
        # KV-cache-only arming would leave 3.3 out of the training graph, i.e.
        # silently untrained at its identity initialisation.
        if attn_window is not None or (
                has_kproj and (kv_cache is not None or clean_x is not None)):
            if torch.is_tensor(t):
                # One noise level per chunk: [B] or [B, F] -> [B]
                timestep = _first_frame_timestep(t)
            else:
                timestep = torch.tensor(
                    [float(t)], device=self.patch_embedding.weight.device)
            forcing_kv_state = dict(forcing_kv_state) if forcing_kv_state is not None else {}
            forcing_kv_state["flex"] = {
                "timestep": timestep,
                "attn_window": attn_window,
            }

        return super().forward(
            x,
            t,
            context,
            seq_len,
            clip_fea=clip_fea,
            y=y,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
            clean_x=clean_x,
            aug_t=aug_t,
            forcing_kv_state=forcing_kv_state,
        )
