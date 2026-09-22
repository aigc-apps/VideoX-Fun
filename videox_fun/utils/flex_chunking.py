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
from typing import List, Optional, Sequence, Tuple, Union

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
    "validate_nested_partitions",
    "chunk_ends_tensor",
    "broadcast_chunk_sizes",
    "chunk_sizes_to_block_kwargs",
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
                         chunk_size: Optional[int] = None) -> ChunkSizes:
    """Parse a user-facing chunk specification into a validated partition.

    Accepted forms (``num_frames`` = latent frames the partition must cover):

    * ``None``                 -> uniform ``chunk_size`` (defaults to 1)
    * ``int``                  -> uniform chunks of that size
    * ``[15, 3, 3]``           -> explicit sizes, must sum to ``num_frames``
    * ``"15-3-3"`` / ``"15,3,3"`` / ``"[15,3,3]"`` -> explicit sizes
    * ``"uniform:3"``          -> uniform chunks of size 3
    * ``"ar"``                 -> fully autoregressive ``[1] * num_frames``
    * ``"bidir"`` / ``"full"`` -> fully bidirectional ``[num_frames]``

    ``independent_first_frame`` needs no separate flag: it is exactly the
    partition whose first chunk has size 1 (read it back with
    ``sizes[0] == 1``).
    """
    if spec is None:
        sizes = uniform_chunks(num_frames, chunk_size if chunk_size is not None else 1)
    elif isinstance(spec, bool):
        raise TypeError(f"chunk spec must not be a bool, got {spec!r}")
    elif isinstance(spec, int):
        sizes = uniform_chunks(num_frames, spec)
    elif isinstance(spec, str):
        sizes = _parse_chunk_string(spec, num_frames)
    elif isinstance(spec, (list, tuple)):
        sizes = [int(s) for s in spec]
    else:
        raise TypeError(
            f"Unsupported chunk spec type {type(spec)}: {spec!r}. Expected "
            "None, int, list/tuple of ints, or a string such as '15-3-3' / "
            "'uniform:3' / 'ar' / 'bidir'.")
    return _validate_sizes(sizes, num_frames, spec)


def _parse_chunk_string(spec: str, num_frames: int) -> ChunkSizes:
    text = spec.strip()
    lowered = text.lower()
    if lowered in ("ar", "autoregressive", "causal"):
        return [1] * int(num_frames)
    if lowered in ("bidir", "bidirectional", "full"):
        return [int(num_frames)]
    if lowered.startswith("uniform:"):
        return uniform_chunks(num_frames, int(lowered.split(":", 1)[1]))
    if lowered.startswith("uniform"):
        return uniform_chunks(num_frames, 1)
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


# Share of iterations that pin the launcher's own uniform ``num_frame_per_block``
# layout instead of drawing a random partition. ``sample_flexible_chunks`` can
# reach every partition whose chunks are individually legal, but it reaches any
# *particular* one only by chance: the uniform ``[3] * 7`` at 21 latent frames
# has probability 1/183708, i.e. 0.11 expected hits in 20000 iterations. Without
# this band ``--num_frame_per_block`` would name a layout the model never trains
# on, and the validation pass - which pins exactly that layout - would render
# out of distribution. Raising the sampler's randomness makes this *worse*, not
# better: more reachable partitions means each one is drawn less often.
# Deliberately not a CLI flag, and shared by both trainers so the two stages
# cannot drift apart.
UNIFORM_BLOCK_PROB = 0.1


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
        #
        # The tail used to be closed unconditionally (`if remaining <=
        # max_chunk: append(remaining); break`), which silently removed most of
        # the sampler's randomness: at 21 frames with 2..10 a run of 3s stopped
        # at `[3, 3, 3, 3, 9]` after four draws, so the uniform `[3] * 7` was
        # unreachable outright - exhaustive enumeration found 679 partitions and
        # none of them was it. Uniform layouts also bias short: the old rule
        # topped out at 7 chunks with 65% of draws at just 3.
        upper = min(max_chunk, remaining - min_chunk)
        choices = list(range(min_chunk, upper + 1)) if upper >= min_chunk else []
        if remaining <= max_chunk:
            choices.append(remaining)
        if not choices:
            # Degenerate only: min_chunk == max_chunk and the remainder is
            # strictly between them, so nothing legal fits. Same as before.
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
