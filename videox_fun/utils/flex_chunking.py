# Flex-Forcing: Towards a Unified Autoregressive and Bidirectional Video
# Diffusion Model (arXiv 2607.03509) - shared frame-axis chunking algebra.
"""Frame-axis partition helpers shared by the Flex-Forcing model, pipeline and
training scripts.

Flex-Forcing replaces Self-Forcing / CausVid's single scalar
``num_frame_per_block`` with an explicit *partition* of the frame axis

    a_t = (a_{t,0}, ..., a_{t,K_t}),      a_{t,0} = 1,  a_{t,K_t} = F + 1

Frames inside one chunk are denoised with **bidirectional** attention, chunks
are denoised **autoregressively** (history is summarised by the KV cache). Both
classical extremes are special cases of the very same representation:

    a_t = (1, 2, ..., F, F+1)  ->  fully autoregressive (1 frame / block)
    a_t = (1, F + 1)           ->  fully bidirectional  (vanilla diffusion)

Everything below is pure index algebra so that the model (block masks), the
pipeline (rollout schedule) and the trainers (random partition sampling) all
share one source of truth instead of re-deriving boundaries locally.

Conventions used across this module:

* A partition is a **list of chunk sizes** ``[15, 3, 3]`` (paper notation
  ``a_t`` is the equivalent cumulative boundary list ``[1, 16, 19, 22]``).
* ``num_frames`` always denotes **latent** frames (post-VAE), matching
  ``target_shape[1]`` in the trainers and ``num_latent_frames`` in the pipeline.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

ChunkSizes = List[int]
Boundaries = List[Tuple[int, int]]

__all__ = [
    "chunk_boundaries",
    "uniform_chunks",
    "normalize_chunk_spec",
    "sample_flexible_chunks",
    "refine_partition",
    "build_pyramid_partitions",
    "build_full_then_blocks_partitions",
    "validate_nested_partitions",
    "chunk_ends_tensor",
    "broadcast_chunk_sizes",
    "chunk_sizes_to_block_kwargs",
    "FLEX_RANDOM_PROB",
    "sample_flex_level0",
]


def chunk_boundaries(chunk_sizes: Sequence[int]) -> Boundaries:
    """Chunk sizes -> list of half-open frame spans ``[(start, end), ...]``.

    ``sum(sizes)`` is trusted as the frame count; callers validate with
    :func:`normalize_chunk_spec` / :func:`validate_nested_partitions`.
    """
    spans = []
    start = 0
    for size in chunk_sizes:
        size = int(size)
        spans.append((start, start + size))
        start += size
    return spans


def uniform_chunks(num_frames: int,
                   chunk_size: int,
                   independent_first_frame: bool = False) -> ChunkSizes:
    """Fixed-size partition, i.e. the classic Self-Forcing ``[1, N, N, ...]``.

    The trailing chunk absorbs the remainder, so ``num_frames`` need not be
    divisible by ``chunk_size``.
    """
    chunk_size = max(1, int(chunk_size))
    sizes: ChunkSizes = []
    remaining = int(num_frames)
    if independent_first_frame:
        if remaining < 1:
            raise ValueError(
                f"num_frames must be >= 1 for independent_first_frame, got {num_frames}")
        sizes.append(1)
        remaining -= 1
    while remaining > 0:
        step = min(chunk_size, remaining)
        sizes.append(step)
        remaining -= step
    return sizes


def normalize_chunk_spec(spec: Union[None, int, str, Sequence[int]],
                         num_frames: int,
                         chunk_size: Optional[int] = None,
                         independent_first_frame: bool = False) -> ChunkSizes:
    """Parse a user-facing chunk specification into a validated partition.

    Accepted forms (``num_frames`` = latent frames the partition must cover):

    * ``None``                 -> uniform ``chunk_size`` (defaults to 1)
    * ``int``                  -> uniform chunks of that size
    * ``[15, 3, 3]``           -> explicit sizes, must sum to ``num_frames``
    * ``"15-3-3"`` / ``"15,3,3"`` / ``"[15,3,3]"`` -> explicit sizes
    * ``"uniform:3"``          -> uniform chunks of size 3
    * ``"ar"``                 -> fully autoregressive ``[1] * num_frames``
    * ``"bidir"`` / ``"full"`` -> fully bidirectional ``[num_frames]``

    ``independent_first_frame`` is applied to the *uniform* forms only (``None``,
    ``int``, ``"uniform[:N]"``), where it inserts the leading size-1 chunk that
    makes the layout ``[1, N, N, ...]``. Without it those forms produce a
    partition the ``independent_first_frame`` trainer never trained on. The
    keyword forms leave it alone on purpose: ``"ar"`` is already one frame per
    chunk, and prefixing ``"bidir"`` would break the single bidirectional chunk.
    An explicit size list also leaves it alone - a leading 1 there is written by
    the caller, and is read back with ``sizes[0] == 1``.
    """
    if spec is None:
        sizes = uniform_chunks(num_frames, chunk_size if chunk_size is not None else 1,
                               independent_first_frame=independent_first_frame)
    elif isinstance(spec, bool):
        raise TypeError(f"chunk spec must not be a bool, got {spec!r}")
    elif isinstance(spec, int):
        sizes = uniform_chunks(num_frames, spec,
                               independent_first_frame=independent_first_frame)
    elif isinstance(spec, str):
        sizes = _parse_chunk_string(spec, num_frames,
                                    independent_first_frame=independent_first_frame)
    elif isinstance(spec, (list, tuple)):
        sizes = [int(s) for s in spec]
    else:
        raise TypeError(
            f"Unsupported chunk spec type {type(spec)}: {spec!r}. Expected "
            "None, int, list/tuple of ints, or a string such as '15-3-3' / "
            "'uniform:3' / 'ar' / 'bidir'.")
    return _validate_sizes(sizes, num_frames, spec)


def _parse_chunk_string(spec: str, num_frames: int,
                        independent_first_frame: bool = False) -> ChunkSizes:
    text = spec.strip()
    lowered = text.lower()
    if lowered in ("ar", "autoregressive", "causal"):
        return [1] * int(num_frames)
    if lowered in ("bidir", "bidirectional", "full"):
        return [int(num_frames)]
    if lowered.startswith("uniform:"):
        return uniform_chunks(num_frames, int(lowered.split(":", 1)[1]),
                              independent_first_frame=independent_first_frame)
    if lowered.startswith("uniform"):
        return uniform_chunks(num_frames, 1,
                              independent_first_frame=independent_first_frame)
    body = text.strip("[]() ")
    sep = "," if "," in body else ("-" if "-" in body else None)
    if sep is None:
        raise ValueError(
            f"Cannot parse chunk spec {spec!r}. Use '15-3-3', '[15,3,3]', "
            "'uniform:3', 'ar' or 'bidir'.")
    return [int(part) for part in body.split(sep) if part.strip() != ""]


def _validate_sizes(sizes: ChunkSizes, num_frames: int, spec) -> ChunkSizes:
    if len(sizes) == 0:
        raise ValueError(f"Chunk spec {spec!r} produced an empty partition.")
    if any(int(s) < 1 for s in sizes):
        raise ValueError(
            f"Chunk spec {spec!r} contains a non-positive chunk size: {sizes}")
    total = int(sum(sizes))
    if total != int(num_frames):
        raise ValueError(
            f"Chunk spec {spec!r} covers {total} latent frames but the video "
            f"has {num_frames}. Provide sizes summing exactly to num_frames "
            "(a leading 1 encodes independent_first_frame).")
    return [int(s) for s in sizes]


# Share of iterations the level-0 mixture's *random* arm takes; the other arms
# split the remainder equally, so with a pyramid this is 30% coarse / 30%
# uniform / 30% full_then_blocks / 10% random. ``sample_flex_level0`` below is
# the single source of truth for the coarsest layout that **both** trainers
# draw: stage 2 (``train_distill.py``) nests it into the §3.2 ladder, stage 1
# (``train_ode.py``) regresses against it directly because it predicts x0 in one
# forward per chunk and has no ladder to build. Sharing the draw is what keeps
# the two stages on the same layout distribution - stage 1's checkpoint is stage
# 2's initialisation (``--ode_transformer_path``), so a layout stage 2 spends
# most of its iterations on must not be one stage 1 never regressed.
#
# The layouts ``--num_frame_per_block`` names are arms, not something the random
# draw is expected to stumble onto: ``sample_flexible_chunks`` reaches every
# individually legal partition but any *particular* one only by chance. At 21
# latent frames the uniform ``[3] * 7`` has probability 1/183708, and the fully
# causal ``[1] * 21`` is 1/21! - unreachable in practice at any ``max_chunk``,
# which is why no ``--flex_chunk_min/max`` setting can substitute for these arms.
# Pinning them is what makes ``--num_frame_per_block`` name a layout the model
# genuinely trains on, and keeps ``log_validation`` - which pins exactly that
# layout - in distribution. Deliberately not a CLI flag.
#
# Deliberate deviation from the paper, and the reason this constant exists at
# all: arXiv 2607.03509 §5 says "we randomly pick the chunk size in training
# from 2 to 10" and pins no arm of its own, so its level 0 is drawn entirely
# from that random family. That family spans thousands of distinct layouts, so
# the random arm buys exactly what §3.3 asks of it - coverage of "attention
# configurations that range from strictly causal to fully non-causal" - but it
# cannot train any *particular* layout, and the layouts inference actually
# selects (``PAPER_CHUNK_CONFIGS``, plus the one ``log_validation`` pins) are
# the deterministic ones. Holding it to a minority share is what leaves those
# arms enough iterations to be trained at all. Setting this to 1.0 reproduces
# the paper's mixture exactly: every threshold below collapses to 0, so the
# three deterministic arms become unreachable and the uniform draw always lands
# in the random one.
FLEX_RANDOM_PROB = 0.10


def sample_flexible_chunks(num_frames: int,
                           min_chunk: int = 2,
                           max_chunk: int = 10,
                           generator: Optional[torch.Generator] = None,
                           device: Union[str, torch.device] = "cpu",
                           independent_first_frame: bool = False) -> ChunkSizes:
    """Randomly partition the frame axis (paper §3.3 flexible-chunk training).

    Chunk sizes are drawn uniformly from ``[min_chunk, max_chunk]`` at every
    rollout so a single model sees the whole spectrum from strictly causal to
    fully bidirectional attention. The only restriction is that a chunk must
    either take the whole remainder or leave at least ``min_chunk`` frames
    behind, so no sub-``min_chunk`` sliver is ever stranded mid-clip. That
    restriction is applied *uniformly*, including at the tail: closing the tail
    as soon as the remainder fits in ``max_chunk`` would throw away most of its
    randomness and leave all but one uniform layout unreachable.

    ``generator`` must live on ``device`` (mirrors the trainers, which pass
    ``torch_rng`` together with ``accelerator.device``).
    """
    num_frames = int(num_frames)
    min_chunk = max(1, int(min_chunk))
    max_chunk = max(min_chunk, int(max_chunk))
    sizes: ChunkSizes = []
    remaining = num_frames
    if independent_first_frame:
        sizes.append(1)
        remaining -= 1
    while remaining > 0:
        # Two legal moves, offered everywhere along the clip including the tail:
        # take the whole remainder, or leave at least min_chunk frames behind.
        # Anything in between would strand a sub-min_chunk sliver mid-video.
        upper = min(max_chunk, remaining - min_chunk)
        choices = list(range(min_chunk, upper + 1)) if upper >= min_chunk else []
        if remaining <= max_chunk:
            choices.append(remaining)
        if not choices:
            # Degenerate only: min_chunk == max_chunk and the remainder is
            # strictly between them, so nothing legal fits.
            sizes.append(remaining)
            break
        step = choices[int(torch.randint(
            0, len(choices), (1,), generator=generator, device=device).item())]
        sizes.append(step)
        remaining -= step
    return sizes


def refine_partition(chunk_sizes: Sequence[int],
                     min_num_frame_per_block: int = 1) -> ChunkSizes:
    """One nested refinement step (paper §3.2 pyramid timestep chunking).

    Every chunk coarser than ``min_num_frame_per_block`` is split into two
    near-equal parts; chunks already at (or below) that block size are kept
    as-is, which makes the operation converge and idempotent. This realises the
    paper's nestedness requirement
    ``a'_{t,k} = a_{t,k-1} ∪ S_{t,k} ∪ a_{t,k}``: all previous boundaries are
    retained and only new split points are inserted, so a KV cache written at
    the coarse level stays valid at the refined level.

    Binary splitting reproduces the configurations listed in the paper, e.g.
    ``[21] -> [11, 10] -> [6, 5, 5, 5] -> [3, 3, 3, 2, 3, 2, 3, 2]``.
    """
    min_num_frame_per_block = max(1, int(min_num_frame_per_block))
    refined: ChunkSizes = []
    for size in chunk_sizes:
        size = int(size)
        if size <= min_num_frame_per_block or size <= 1:
            refined.append(size)
            continue
        head = (size + 1) // 2
        refined.extend([head, size - head])
    return refined


def build_pyramid_partitions(num_frames: int,
                             num_levels: int = 2,
                             min_num_frame_per_block: int = 1,
                             coarse_chunk: Optional[int] = None,
                             base_chunks: Optional[Sequence[int]] = None,
                             independent_first_frame: bool = False) -> List[ChunkSizes]:
    """Coarse-to-fine nested partition ladder for pyramid denoising (§3.2).

    Level 0 uses the coarsest planning chunks (one chunk per whole clip when
    ``coarse_chunk`` is None, i.e. fully bidirectional at the high-noise
    steps); each subsequent level refines it with :func:`refine_partition`
    until ``min_num_frame_per_block`` or a fixed point is reached. ``base_chunks``
    overrides level 0 outright with a caller-supplied partition, which is how
    the pipeline lets a user pin the planning layout (e.g. ``[18, 3]``) while
    still refining it below. The returned list is ordered coarse -> fine and
    satisfies :func:`validate_nested_partitions`.
    """
    num_levels = max(1, int(num_levels))
    if base_chunks is not None:
        base = _validate_sizes([int(s) for s in base_chunks], num_frames,
                               "pyramid-base")
    elif coarse_chunk is not None:
        base = uniform_chunks(num_frames, coarse_chunk, independent_first_frame)
    else:
        base = ([1, int(num_frames) - 1] if independent_first_frame and num_frames > 1
                else [int(num_frames)])
        base = _validate_sizes(base, num_frames, "pyramid-base")
    partitions: List[ChunkSizes] = [base]
    current = base
    for _ in range(num_levels - 1):
        nxt = refine_partition(current, min_num_frame_per_block)
        if nxt == current:
            break
        partitions.append(nxt)
        current = nxt
    return partitions


def build_full_then_blocks_partitions(num_frames: int,
                                      num_frame_per_block: int,
                                      independent_first_frame: bool = False) -> List[ChunkSizes]:
    """Two-level ladder: one full planning chunk, then block-major refinement.

    Level 0 is a single bidirectional chunk over the whole clip - the *first*
    denoising step plans everything at once. Level 1 is the classic Self-Forcing
    uniform ``num_frame_per_block`` partition, which the walk reuses for *every*
    remaining step, so the rollout turns block-major right after the planning
    pass. This is the coarse-to-fine special case of the pyramid where the fine
    level is a fixed block size instead of a binary split.

    ``[F]`` -> blocks is always a valid refinement (the block partition keeps
    the level-0 boundary at ``F``), so the ladder satisfies
    :func:`validate_nested_partitions` and the KV cache written by the planning
    step stays reusable by the block-major steps.
    """
    num_frames = int(num_frames)
    block = uniform_chunks(num_frames, num_frame_per_block,
                           independent_first_frame)
    return [[num_frames], block]


def validate_nested_partitions(partitions: Sequence[Sequence[int]],
                               num_frames: int) -> Boundaries:
    """Check a pyramid ladder: same coverage + monotonically nested boundaries.

    Returns the boundary set (in frames) of the finest level. Raises
    ``ValueError`` when a level does not cover ``num_frames`` exactly or when a
    later level drops a boundary of an earlier one - either would invalidate the
    KV cache reuse the pyramid schedule relies on.
    """
    if len(partitions) == 0:
        raise ValueError("partitions must contain at least one level.")
    prev_bounds: Optional[set] = None
    finest: Boundaries = []
    for level, sizes in enumerate(partitions):
        sizes = [int(s) for s in sizes]
        if any(s < 1 for s in sizes):
            raise ValueError(f"Level {level} has a non-positive chunk size: {sizes}")
        if sum(sizes) != int(num_frames):
            raise ValueError(
                f"Level {level} {sizes} covers {sum(sizes)} latent frames but "
                f"the video has {num_frames}.")
        bounds = {end for _, end in chunk_boundaries(sizes)}
        if prev_bounds is not None and not prev_bounds.issubset(bounds):
            missing = sorted(prev_bounds - bounds)
            raise ValueError(
                f"Level {level} {sizes} is not a refinement of level "
                f"{level - 1}: boundaries at frames {missing} were dropped. "
                "Pyramid partitions must be nested so cached keys stay valid.")
        prev_bounds = bounds
        finest = chunk_boundaries(sizes)
    return finest


def chunk_ends_tensor(chunk_sizes: Sequence[int],
                      frame_seqlen: int,
                      device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """Token-level exclusive visibility end per query token.

    ``ends[i]`` is the first key token index that query token ``i`` may **not**
    see, i.e. the end of the chunk containing ``i``. Combined with the mask
    ``(kv_idx < ends[q_idx]) | (q_idx == kv_idx)`` this yields exactly the
    Flex-Forcing pattern: bidirectional inside a chunk, causal across chunks.
    This is the variable-chunk generalisation of the ``ends`` array built by
    ``WanTransformer3DModel_SelfForcing.create_block_mask_for_training``.
    """
    frame_seqlen = int(frame_seqlen)
    total_length = int(sum(int(s) for s in chunk_sizes)) * frame_seqlen
    ends = torch.zeros(total_length, dtype=torch.long, device=device)
    for start, end in chunk_boundaries(chunk_sizes):
        ends[start * frame_seqlen:end * frame_seqlen] = end * frame_seqlen
    return ends


def broadcast_chunk_sizes(chunk_sizes: Sequence[int],
                          device: Union[str, torch.device] = "cpu",
                          src: int = 0,
                          max_chunks: int = 512) -> ChunkSizes:
    """Rank-0 broadcast of a sampled partition (mirrors the trainers'
    ``dist.broadcast`` of ``num_generated_blocks``).

    Every rank must roll out with the *same* partition, otherwise the KV-cache
    frame bookkeeping diverges across the SP/FSDP group. Falls back to the
    input unchanged when torch.distributed is not initialised.
    """
    sizes = [int(s) for s in chunk_sizes]
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover - torch always ships distributed
        return sizes
    if not dist.is_available() or not dist.is_initialized():
        return sizes
    payload = torch.zeros(max_chunks + 1, dtype=torch.long, device=device)
    if dist.get_rank() == src:
        if len(sizes) > max_chunks:
            raise ValueError(
                f"Partition has {len(sizes)} chunks, exceeding max_chunks={max_chunks}.")
        payload[0] = len(sizes)
        payload[1:1 + len(sizes)] = torch.tensor(sizes, dtype=torch.long)
    dist.broadcast(payload, src=src)
    count = int(payload[0].item())
    return [int(v) for v in payload[1:1 + count].tolist()]


def chunk_sizes_to_block_kwargs(chunk_sizes: Sequence[int]) -> dict:
    """Backwards-compatible ``(num_frame_per_block, independent_first_frame)``.

    Several consumers still expect the scalar Self-Forcing knobs - Forcing-KV
    derives its AR step and its rolling-cache budget from
    ``num_frame_per_block``. Reporting the largest chunk is the conservative
    choice: it over-provisions the cache budget and never under-estimates the
    AR stride.
    """
    sizes = [int(s) for s in chunk_sizes]
    return {
        "num_frame_per_block": max(sizes) if sizes else 1,
        "independent_first_frame": bool(sizes and sizes[0] == 1),
    }


def sample_flex_level0(num_frames: int,
                       num_frame_per_block: int,
                       chunk_min: int,
                       chunk_max: int,
                       pyramid: bool,
                       generator: Optional[torch.Generator] = None,
                       device: Union[str, torch.device] = "cpu",
                       independent_first_frame: bool = False) -> Dict[str, Any]:
    """Draw one iteration's coarsest Flex-Forcing layout (the four-arm mixture).

    Single source of truth for the level-0 partition **both** trainers draw:
    ``train_distill.py`` nests the result into the §3.2 ladder, ``train_ode.py``
    regresses against it directly. Keeping the draw here is what stops the two
    stages from drifting apart - stage 1's checkpoint is stage 2's
    initialisation, so every layout stage 2 spends iterations on has to be one
    stage 1 actually regressed.

    Four arms, decided by a single uniform draw. ``"random"`` takes
    :data:`FLEX_RANDOM_PROB` (10%) and the other three split what is left
    equally, i.e. 30% each:

    * ``"coarse"`` - one chunk over the whole clip (inference's planning step),
      i.e. fully bidirectional. Only drawn when ``pyramid`` is on: without a
      ladder a ``[F]`` level 0 would be a degenerate single-chunk rollout, so
      the two surviving arms widen to 45% each and ``"random"`` stays at 10%.
    * ``"uniform"`` - a pure Self-Forcing grid (no binary refinement) in one of
      three sub-layouts drawn from ``num_frame_per_block`` and fully causal:
      ``[1]*F`` alone, ``[nfpb]*k`` alone, or ``[nfpb]*k`` planning followed by
      ``[1]*F`` for every remaining step - the coarse-block -> per-frame descent
      the pyramid never samples, because it halves through ``[2, 1x14]`` first.
      The three are equiprobable *within* the arm, so 10% of iterations each.
    * ``"full_then_blocks"`` - ``denoise_mode="full_then_blocks"``: a whole-clip
      planning chunk, then block-major refinement at a width drawn from
      ``{1, num_frame_per_block}``.
    * ``"random"`` - the §3.3 / §5 spectrum via :func:`sample_flexible_chunks`.

    Distributed contract: the arm selector is drawn once and broadcast from
    rank 0, then **every** arm issues exactly one :func:`broadcast_chunk_sizes`.
    All ranks therefore end up with a byte-identical layout *and* the same
    collective count. A per-rank arm draw would leave ranks in different arms
    with different walk forward counts -> FSDP all-gather desync (NCCL hang).

    Args:
        pyramid: whether the caller ladders at all, i.e.
            ``args.flex_pyramid_levels > 1``. Stage 1 has no ladder but must
            still pass stage 2's setting, because this flag decides whether the
            coarse arm exists and therefore what the other three arms weigh.

    Returns:
        ``"sizes"`` - the level-0 partition, already rank-reconciled.
        ``"arm"`` - ``"coarse"`` / ``"uniform_causal"`` / ``"uniform_blocks"`` /
        ``"uniform_plan_then_causal"`` / ``"full_then_blocks"`` / ``"random"``.
        The uniform arm is resolved to its sub-layout here so the caller can
        rebuild the ladder depth without re-deriving the band arithmetic.
        ``"arm_text"`` - human-readable description for the layout log.
        ``"causal"`` - ``[1]*F``, present on ``"uniform_plan_then_causal"``.
        ``"block_level"`` - the block-major level, present on
        ``"full_then_blocks"``.
    """
    num_frames = int(num_frames)
    nfpb = max(1, int(num_frame_per_block))
    # The arm selector must be IDENTICAL on every rank, not just its broadcast
    # count: the arms build their ladder from different tensors, so a per-rank
    # `u` would desync the group. Broadcast rank 0's draw; the arm's own single
    # broadcast then reconciles its within-arm randomness (the random base, the
    # block width) to rank 0.
    u = torch.rand((), generator=generator, device=device)
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover - torch always ships distributed
        dist = None
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.broadcast(u.reshape(1), src=0)
    u = u.item()

    # `random` holds a fixed slice and the deterministic arms divide the rest,
    # so switching the coarse arm off widens the two survivors instead of
    # silently fattening `random`. `ftb_hi` is both the end of the last
    # deterministic arm and the start of the random one, and it is the final
    # threshold tested, so no rounding remainder can fall through unhandled.
    det_arms = 3 if pyramid else 2
    det_w = (1.0 - FLEX_RANDOM_PROB) / det_arms
    coarse_hi = det_w if pyramid else 0.0
    uniform_hi = coarse_hi + det_w
    ftb_hi = uniform_hi + det_w

    if u < coarse_hi:
        # Coarse end of §3.1: reuse the ladder builder's own level-0 rule so the
        # `independent_first_frame` handling cannot drift from inference's.
        sizes = build_pyramid_partitions(
            num_frames, num_levels=1, base_chunks=None,
            independent_first_frame=independent_first_frame)[0]
        return {
            "sizes": broadcast_chunk_sizes(sizes, device=device),
            "arm": "coarse",
            "arm_text": "whole clip, the 3.2 coarse planning layout",
        }

    if u < uniform_hi:
        # The three sub-layouts change the ladder DEPTH (hence stage 2's walk
        # forward count), so the choice must be rank-consistent. It is free here:
        # `u` was already broadcast, so slicing its position within the uniform
        # band gives every rank the same sub-arm with no extra collective. Like
        # every arm this issues EXACTLY ONE broadcast. `uniform_chunks` is called
        # directly rather than through `normalize_chunk_spec`, which only applies
        # `independent_first_frame` to its uniform forms via a keyword - the
        # direct call keeps this arm visibly identical to the bands around it.
        band = (u - coarse_hi) / det_w                 # [0, 1) in the uniform arm
        sub = min(int(band * 3), 2)                     # 0 / 1 / 2, equiprobable
        causal = uniform_chunks(num_frames, 1,
                                independent_first_frame=independent_first_frame)
        if sub == 0:
            return {
                "sizes": broadcast_chunk_sizes(causal, device=device),
                "arm": "uniform_causal",
                "arm_text": "uniform 1-frame blocks, single level (fully causal)",
            }
        sizes = broadcast_chunk_sizes(
            uniform_chunks(num_frames, nfpb,
                           independent_first_frame=independent_first_frame),
            device=device)
        if sub == 1:
            return {"sizes": sizes, "arm": "uniform_blocks",
                    "arm_text": f"uniform {nfpb}-frame blocks, single level"}
        return {
            "sizes": sizes, "arm": "uniform_plan_then_causal",
            "causal": list(causal),
            "arm_text": (f"uniform {nfpb}-frame plan -> fully causal [1x"
                         f"{num_frames}] for the remaining steps"),
        }

    if u < ftb_hi:
        # "First step full, every later step block-major": a fixed two-level
        # ladder that does NOT go through the binary pyramid refinement, so the
        # block-major level is reached at high noise (the 2nd step) instead of as
        # a deep refinement - the one trajectory the pyramid arms never sample.
        # The block width is drawn between fully causal (1 frame / block) and the
        # launcher's `num_frame_per_block`, so both the tight-AR and the
        # coarse-block refinement of the whole-clip plan get trained. Level 0 is
        # just `[F]`, identical on every rank and needing no sync, so the ONE
        # broadcast this arm issues covers the block-major level; rank 0's draw
        # wins and every rank rebuilds the same two-level ladder locally.
        block_choices = sorted({1, nfpb})
        pick = int(torch.rand((), generator=generator, device=device).item()
                   * len(block_choices))
        block = block_choices[min(pick, len(block_choices) - 1)]
        block = min(block, num_frames)
        ftb = build_full_then_blocks_partitions(
            num_frames, block, independent_first_frame=independent_first_frame)
        ftb[1] = broadcast_chunk_sizes(ftb[1], device=device)
        return {
            "sizes": list(ftb[0]), "arm": "full_then_blocks",
            "block_level": list(ftb[1]),
            "arm_text": (f"full clip -> uniform {max(ftb[1])}-frame blocks, "
                         "the full_then_blocks ladder"),
        }

    sizes = sample_flexible_chunks(
        num_frames, min_chunk=chunk_min, max_chunk=chunk_max,
        generator=generator, device=device,
        independent_first_frame=independent_first_frame)
    return {
        "sizes": broadcast_chunk_sizes(sizes, device=device),
        "arm": "random",
        "arm_text": (f"random {chunk_min}..{chunk_max}-frame blocks, "
                     "the 3.3 stochastic-chunk spectrum"),
    }
