"""3D neighborhood attention (NATTEN ``na3d`` semantics) and absolute RoPE, in MLX.

There is no NATTEN for Metal, and MLX has no neighborhood-attention primitive, so
this module rebuilds both halves of the LTX-2.5 diffusion VAE decoder's attention:

* :func:`na3d` -- local 3-D attention where every query attends to exactly
  ``prod(kernel_size)`` keys. At grid boundaries the window is **shifted inward**
  (not clamped-and-masked-short), which is NATTEN's semantic and therefore the one
  the checkpoint was trained with. Implemented as query tiles + one additive/boolean
  mask per *window geometry*, dispatched through ``mx.fast.scaled_dot_product_attention``.
  Transcribed from the vendor's own CPU fallback
  (``ltx_core/model/video_vae/transformer/fallback_na/eager.py`` @ LTX-2 v1.2.0,
  itself vendored from comfy-kitchen ``backends/eager/na.py``), which is the
  authoritative statement of what NATTEN computes.

* :func:`rope_inv_freqs` / :func:`apply_abs_rope` -- per-axis absolute RoPE.

  **The float64 trap.** LTX builds its RoPE inverse frequencies in double precision
  on purpose (the connector is constructed with ``double_precision_rope=True``), and
  MLX has no float64 at all. ComfyUI issue #15512 is this exact table crashing on
  MPS, and the fix comfyanonymous accepted (PR #15516) *keeps* the double precision
  by computing on CPU and casting once. The vendor's own ltx-core does the same thing
  with numpy (``rope_math.rope_inv_freqs`` builds an ``np.float64`` array and calls
  ``.to(torch.float32)``). So do we: the table is built in ``numpy.float64`` and cast
  to float32 exactly once, at construction. Building it natively in float32 does not
  crash -- it silently returns different frequencies, which is the worst failure mode
  available. ``tests/test_diffusion_decoder.py`` pins the table against a float64
  reference *and* asserts the naive float32 construction is measurably different, so
  nobody can "simplify" this away.

RoPE positions are **local, 0-based, per call**. That is the vendor's choice too
(``det_attn_rope.py`` module docstring): every attention here is a local window with
no cross-tile tokens, and absolute-vs-local RoPE differs by a global phase that
cancels inside the softmax over that window.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import numpy as np

__all__ = [
    "NA_KV_STACK_BUDGET",
    "NA_SCORE_BUDGET",
    "apply_abs_rope",
    "default_rope_dim_split",
    "na3d",
    "rope_inv_freqs",
    "window_bounds",
]

#: Element budget for one query tile's ``[Nq, Nk]`` score/mask block.
NA_SCORE_BUDGET = 2**24

#: Element budget for the stacked K/V copies of one batched SDPA call. This is the
#: knob that trades wall-clock (fewer, larger dispatches) against peak memory.
NA_KV_STACK_BUDGET = 2**27

#: How much redundant attention a query tile may carry, as ``keys_seen / kernel_volume``.
#:
#: A tile of queries attends to the union of their windows -- ``prod(tile + kernel - 1)``
#: keys -- while each query only *needs* ``prod(kernel)`` of them. Everything else is
#: masked out and thrown away, so this ratio IS the multiple of wasted attention FLOPs,
#: and it grows fast: at stage 5 the score budget alone leaves tiles wasting 7.1x, and
#: the (3,5,5) kernels of stages 3-4 waste **74x** because a 75-key kernel sits inside a
#: 5544-key halo. Smaller tiles cut that at the cost of more K/V gather traffic and more
#: dispatches. 2.5 is the knee: it takes stage 5 from 202 TFLOP to 67 and stage 4 from
#: 4.3 to 0.19, while keeping tiles big enough to batch. ``LTX2_NA3D_MAX_WASTE``.
NA_MAX_WINDOW_WASTE = 2.5

#: Floor on tile volume, so the waste target cannot shrink tiles into per-query gathers
#: where dispatch overhead and K/V duplication dominate. ``LTX2_NA3D_MIN_TILE``.
NA_MIN_TILE_TOKENS = 64


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    """Split ``head_dim`` across the (T, H, W) RoPE chunks.

    Verbatim from ``ltx_core.model.video_vae.transformer.rope_math``. For the
    checkpoint's ``head_dim=64`` this is ``(16, 24, 24)``.
    """
    if head_dim % 8 != 0:
        raise ValueError(f"head_dim={head_dim} must be a multiple of 8 for the default split")
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    return (d_t, d_hw, d_hw)


def rope_inv_freqs(dim: int, base: float = 10000.0) -> mx.array:
    """``1 / base**(i/dim)`` for even ``i``, computed in float64 and cast once.

    See the module docstring: the double precision is load-bearing and MLX cannot
    express it, so the table is built off-device in numpy.
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")
    exponents = np.arange(0, dim, 2, dtype=np.float64) / dim
    inv = 1.0 / np.power(np.float64(base), exponents)
    return mx.array(inv.astype(np.float32))


def _rot_axis(chunk: mx.array, positions: mx.array, inv: mx.array, axis: int) -> mx.array:
    """Rotate one axis-chunk ``chunk[..., D]`` (D even) of a ``(B,T,H,W,NH,HD)`` tensor.

    Interleaved-pair convention: the last dim is read as ``(D/2, 2)`` pairs, matching
    ``rope_math.rot_abs_axis_impl``. Rotation math runs in float32 regardless of the
    tensor dtype (the vendor's ``rope_compute_dtype`` default).
    """
    out_dtype = chunk.dtype
    pairs = chunk.reshape(*chunk.shape[:-1], chunk.shape[-1] // 2, 2).astype(mx.float32)
    xe = pairs[..., 0]
    xo = pairs[..., 1]
    shape = [1, 1, 1, 1, 1, inv.shape[0]]
    shape[axis] = positions.shape[0]
    ang = (positions[:, None] * inv[None, :]).reshape(shape)
    c = mx.cos(ang)
    s = mx.sin(ang)
    rotated = mx.stack([xe * c - xo * s, xe * s + xo * c], axis=-1)
    return rotated.reshape(chunk.shape).astype(out_dtype)


def apply_abs_rope(
    x: mx.array,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[mx.array, mx.array, mx.array],
) -> mx.array:
    """Absolute per-axis RoPE on ``(B, T, H, W, NH, HD)``, local 0-based positions."""
    d_t, d_h, _ = rope_split
    t, h, w = x.shape[1], x.shape[2], x.shape[3]
    xt = _rot_axis(x[..., :d_t], mx.arange(t, dtype=mx.float32), inv_freqs[0], axis=1)
    xh = _rot_axis(x[..., d_t : d_t + d_h], mx.arange(h, dtype=mx.float32), inv_freqs[1], axis=2)
    xw = _rot_axis(x[..., d_t + d_h :], mx.arange(w, dtype=mx.float32), inv_freqs[2], axis=3)
    return mx.concatenate([xt, xh, xw], axis=-1)


# ---------------------------------------------------------------------------
# Neighborhood attention
# ---------------------------------------------------------------------------


def window_bounds(length: int, kernel: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Per-index ``(start, end)`` of the attended window along one axis.

    NATTEN semantics: the window is always exactly ``min(kernel, length)`` wide and
    slides *inward* at the borders instead of being truncated. Only the non-causal
    branch of the vendor fallback is reproduced -- the video VAE decoder never
    requests causal neighborhood attention.
    """
    kernel = min(kernel, length)
    lo = length - kernel
    half = kernel // 2
    starts = tuple(min(max(i - half, 0), lo) for i in range(length))
    return starts, tuple(s + kernel for s in starts)


def _pick_tiles(
    dims: tuple[int, int, int],
    kernels: tuple[int, int, int],
    budget: int,
    max_waste: float = NA_MAX_WINDOW_WASTE,
    min_tile_tokens: int = NA_MIN_TILE_TOKENS,
) -> list[int]:
    """Per-axis query-tile lengths.

    Two conditions drive the halving, and they are not the same condition. The score
    budget is about **memory** -- one tile's ``[Nq, Nk]`` mask and score block have to
    fit. The waste target is about **work** -- see :data:`NA_MAX_WINDOW_WASTE`. The
    vendor's CPU fallback only has the first, because on CPU it is a correctness
    fallback and nobody runs a production decode through it. Here it is the production
    path, so the second matters more than the first.
    """
    tiles = list(dims)
    kernel_vol = math.prod(kernels)

    def keys_seen(ts: list[int]) -> int:
        return math.prod(min(d, t + k - 1) for t, k, d in zip(ts, kernels, dims, strict=True))

    while max(tiles) > 1:
        nq, nk = math.prod(tiles), keys_seen(tiles)
        i = max(range(3), key=lambda a: tiles[a] / kernels[a])
        if tiles[i] <= 1:
            break
        next_nq = nq // tiles[i] * max(1, (tiles[i] + 1) // 2)
        over_budget = nq * nk > budget
        # The floor is checked against what the halving would *produce*, so a tile
        # never overshoots it; the budget overrides the floor, since not fitting is
        # not a trade-off.
        wasteful = nk > max_waste * kernel_vol and next_nq >= min_tile_tokens
        if not (over_budget or wasteful):
            break
        tiles[i] = max(1, (tiles[i] + 1) // 2)
    return tiles


def _env_float(name: str, default: float) -> float:
    import os

    return float(os.environ.get(name, default))


@lru_cache(maxsize=256)
def _group_mask(rel_bounds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]) -> mx.array:
    """Boolean ``[1, 1, Nq, Nk]`` visibility mask for one tile geometry.

    Cached: in a full decode the interior geometry repeats for thousands of tiles and
    the corner/edge geometries repeat across every stage-5 chunk.
    """
    bools = []
    for starts, ends in rel_bounds:
        st = np.asarray(starts)[:, None]
        en = np.asarray(ends)[:, None]
        kj = np.arange(int(en.max()))[None, :]
        bools.append((kj >= st) & (kj < en))
    visible = (
        bools[0][:, None, None, :, None, None]
        & bools[1][None, :, None, None, :, None]
        & bools[2][None, None, :, None, None, :]
    )
    nq = visible.shape[0] * visible.shape[1] * visible.shape[2]
    nk = visible.shape[3] * visible.shape[4] * visible.shape[5]
    return mx.array(visible.reshape(1, 1, nq, nk))


def na3d(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    kernel_size: tuple[int, int, int],
    scale: float | None = None,
    score_budget: int = NA_SCORE_BUDGET,
    kv_stack_budget: int = NA_KV_STACK_BUDGET,
) -> mx.array:
    """3D neighborhood attention over ``(B, T, H, W, NH, HD)`` tensors.

    ``scale`` defaults to ``head_dim ** -0.5``; pass ``1.0`` when Q is pre-scaled
    (which is what the LTX attention module does, folding the scale into ``q_norm``).
    """
    batch, t, h, w, nh, hd = q.shape
    dims = (t, h, w)
    kernels = tuple(min(k_, d) for k_, d in zip(kernel_size, dims, strict=True))
    if scale is None:
        scale = hd**-0.5
    if scale != 1.0:
        q = q * scale

    bounds = [window_bounds(d, k_) for d, k_ in zip(dims, kernels, strict=True)]
    tile_t, tile_h, tile_w = _pick_tiles(
        dims,
        kernels,
        score_budget,
        max_waste=_env_float("LTX2_NA3D_MAX_WASTE", NA_MAX_WINDOW_WASTE),
        min_tile_tokens=int(_env_float("LTX2_NA3D_MIN_TILE", NA_MIN_TILE_TOKENS)),
    )

    # Group query tiles by *relative* window geometry so one mask serves many tiles.
    groups: dict[tuple, list[tuple[tuple[int, int, int, int, int, int], tuple[int, int, int, int, int, int]]]] = {}
    for t0 in range(0, t, tile_t):
        t1 = min(t0 + tile_t, t)
        rt0, rt1 = bounds[0][0][t0], bounds[0][1][t1 - 1]
        rel_t = (
            tuple(s - rt0 for s in bounds[0][0][t0:t1]),
            tuple(e - rt0 for e in bounds[0][1][t0:t1]),
        )
        for h0 in range(0, h, tile_h):
            h1 = min(h0 + tile_h, h)
            rh0, rh1 = bounds[1][0][h0], bounds[1][1][h1 - 1]
            rel_h = (
                tuple(s - rh0 for s in bounds[1][0][h0:h1]),
                tuple(e - rh0 for e in bounds[1][1][h0:h1]),
            )
            for w0 in range(0, w, tile_w):
                w1 = min(w0 + tile_w, w)
                rw0, rw1 = bounds[2][0][w0], bounds[2][1][w1 - 1]
                rel_w = (
                    tuple(s - rw0 for s in bounds[2][0][w0:w1]),
                    tuple(e - rw0 for e in bounds[2][1][w0:w1]),
                )
                groups.setdefault((rel_t, rel_h, rel_w), []).append(
                    ((t0, t1, h0, h1, w0, w1), (rt0, rt1, rh0, rh1, rw0, rw1))
                )

    out = mx.zeros((batch, t, h, w, nh, hd), dtype=v.dtype)
    for rel, tiles in groups.items():
        mask = _group_mask(rel)
        nq, nk = mask.shape[2], mask.shape[3]
        (qt0, qt1, qh0, qh1, qw0, qw1), _ = tiles[0]
        tq, th, tw = qt1 - qt0, qh1 - qh0, qw1 - qw0
        g_max = max(1, kv_stack_budget // max(1, batch * nh * nk * hd))
        for c0 in range(0, len(tiles), g_max):
            chunk = tiles[c0 : c0 + g_max]
            g = len(chunk)
            q_s = mx.stack([q[:, a:b, c:d, e:f] for (a, b, c, d, e, f), _ in chunk])
            k_s = mx.stack([k[:, a:b, c:d, e:f] for _, (a, b, c, d, e, f) in chunk])
            v_s = mx.stack([v[:, a:b, c:d, e:f] for _, (a, b, c, d, e, f) in chunk])
            q_s = q_s.transpose(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nq, hd)
            k_s = k_s.transpose(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            v_s = v_s.transpose(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            o = mx.fast.scaled_dot_product_attention(q_s, k_s, v_s, scale=1.0, mask=mask)
            o = o.reshape(g, batch, nh, tq, th, tw, hd).transpose(0, 1, 3, 4, 5, 2, 6)
            for i, ((a, b, c, d, e, f), _) in enumerate(chunk):
                out[:, a:b, c:d, e:f] = o[i]
            mx.eval(out)
    return out
