"""Live TAE preview of a running LTX-2 render, and the early-abort contract that goes with it.

A High-tier LTX-2.5 render is five minutes of GPU time, and its composition — who is in
frame, how they are framed, where the camera sits — is settled in the first forward or two.
Today the only way to learn that a take is wrong is to wait for it. This module turns each
forward's *current denoised estimate* into a thumbnail on disk, so a watcher (a UI, a shell
loop, an eyeball) can stop a bad take at second thirty instead of minute five.

Ported from the proven MiniMax-H3 implementation (``minimax_h3_mlx/live_preview.py``,
``notes/LIVE_PREVIEW_2026-08-11.md``). The file contract is the same one, with an LTX schema
id, so a panel written against either is a small edit away from driving both.

Two properties are load-bearing:

* **It is read-only.** LTX's denoisers wrap an :class:`~ltx_core_mlx.model.transformer.model.X0Model`,
  so the x0 estimate is a tensor the loop *already holds* — ``denoised_v1`` in
  ``res2s_denoise_loop``, ``video_x0`` in ``denoise_loop``/``guided_denoise_loop``. Nothing is
  recomputed, no scheduler is re-entered, no step index advances, no tensor the denoiser owns
  is written or rebound. A render with the preview on must be byte-identical to the same
  render with it off.
* **Every file is written atomically** (write to ``<name>.tmp<pid>``, ``fsync``,
  ``os.replace``). A watcher polling at any frequency either sees the previous complete file
  or the next complete file, never a torn one.

The abort half is a sentinel file, not a signal: the loops check for ``<live-dir>/ABORT``
between forwards and stop cleanly if it is there. A file is the one channel that works across
a UI, a shell, an ssh session and a supervisor without any of them holding the process handle.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

#: ``status.json``'s ``schema`` field. Bump it if a field changes meaning; add fields freely.
#: ``h3-live-preview/1`` is the sibling contract this one is derived from.
SCHEMA = "ltx-live-preview/1"

#: Exit code for a render stopped by the ABORT sentinel. Distinct from 0 (done) and from 1
#: (the CLI's ordinary traceback exit), so a supervisor can tell "the user stopped this" apart
#: from "this crashed" without parsing anything.
ABORT_EXIT_CODE = 75

ABORT_FILENAME = "ABORT"
STATUS_FILENAME = "status.json"
LATEST_FILENAME = "preview_latest.png"

#: The animated companion to LATEST_FILENAME: the warm-up frames this decode already
#: produced, written as one looping WebP. Same decode, so it costs no GPU.
LOOP_FILENAME = "preview_latest.webp"
#: Milliseconds per loop frame. The decoder emits eight pixel frames per latent token
#: at 24 fps, so ~42 ms plays them back at the rate they were generated at.
LOOP_FRAME_MS = 42
#: A monitor, not a deliverable — trade bytes and encode time for fidelity.
LOOP_QUALITY = 60

#: Append-only log, one compact JSON object per published preview. ``status.json`` is the
#: contract a panel polls; this is the record an *analysis* wants afterwards (which sigma and
#: which stage produced ``preview_07.png``), and it is what ``scripts/live_preview_strip.py``
#: labels the evolution strip from. Advisory, not part of the polling contract.
HISTORY_FILENAME = "history.jsonl"

#: Latent frames of causal warm-up handed to the tiny decoder ahead of the previewed frame.
#: Every ``MemBlock`` remembers the previous frame's input, so a frame decoded alone is decoded
#: as if it opened the clip. Measured against the full-sequence decode on a real generated
#: LTX-2.5 latent (``scripts/probe_preview_context.py``); see
#: ``notes/ltx25_perf_exp2.md`` for the table. LTX's decoder has no chunk padding to hide
#: behind (H3's did), so each extra context token costs real time — this default is the knee.
DEFAULT_CONTEXT = 2

#: Latent cells (``latent_h * latent_w``) the preview decode may work on before it starts
#: pooling. **An LTX latent cell is 32x32 pixels** — four times H3's 16x16 — so the same cell
#: count buys four times the pixels and four times the transient working set. 400 cells is
#: 409,600 preview pixels: it leaves the 768x448 draft tier (336 cells) and both half-res
#: Stage-1 grids untouched, and brings the 1024x576 Stage-2 grid (576 cells) under in a single
#: halving (144 cells -> a 512x288 thumbnail).
LATENT_CELL_BUDGET = 400


class LivePreviewAborted(RuntimeError):
    """Raised between forwards when the ABORT sentinel appears."""


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write ``payload`` so a concurrent reader never observes a partial file."""
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def auto_downscale(latent_height: int, latent_width: int, budget: int = LATENT_CELL_BUDGET) -> int:
    """Smallest power-of-two pooling factor that fits the budget and divides both latent axes."""
    factor = 1
    while (latent_height // factor) * (latent_width // factor) > budget:
        nxt = factor * 2
        if latent_height % nxt or latent_width % nxt:
            break
        factor = nxt
    return factor


def pool_latent(latents: mx.array, factor: int) -> mx.array:
    """2x2-style average pool of ``(B, C, T, H, W)`` latents on the spatial axes."""
    if factor <= 1:
        return latents
    b, c, t, h, w = latents.shape
    return latents.reshape(b, c, t, h // factor, factor, w // factor, factor).mean(axis=(4, 6))


def local_output_index(context: int) -> int:
    """Output frame carrying the previewed latent token inside a ``context + 1`` token decode.

    The tiny decoder emits eight raw frames per latent token and drops the seven causal lead-in
    frames of the sequence, so ``T`` tokens yield ``T * 8 - 7`` frames and the *last* of them is
    the last frame of the last token — index ``8 * context``.
    """
    return 8 * context


def approximate_output_frame(latent_frame: int) -> int:
    """The delivered pixel frame the preview actually shows.

    LTX's video VAE groups pixel frames ``(1, 8, 8, 8, ...)`` per latent frame, so latent
    frame ``i`` covers delivered frames ``max(0, 8i - 7) .. 8i``. :func:`local_output_index`
    decodes the **last** of those, so this must name the last one too — ``8i`` — not the first.
    Getting this wrong is invisible in the thumbnail and silently mislabels which moment of
    the clip a viewer is looking at (and made the first context probe compare two different
    frames). ``8i`` is also exactly the final frame index for a full clip: ``(F-1)*8`` with
    ``F`` latent frames and ``(F-1)*8+1`` delivered frames.
    """
    return max(0, int(latent_frame) * 8)


class LivePreviewMonitor:
    """Per-forward TAE thumbnails plus the abort sentinel, for one whole render.

    One monitor spans both stages of a two-stage render, so ``forward``/``total_forwards`` in
    ``status.json`` count the whole job rather than restarting per stage — which is what a
    progress bar wants. ``stage``/``total_stages`` say which half is running.

    A "forward" here is **one x0 estimate**, not one DiT pass. On the HQ path that distinction
    is large and deliberate: ``res2s_denoise_loop`` is second-order (two estimates per outer
    step) and CFG adds an unconditional DiT pass to each, so the High tier's Stage 1 publishes
    ``2 * steps + 1`` previews off roughly ``4 * steps`` DiT forwards. The preview is tied to
    the estimate because the estimate is the thing that has a picture in it.
    """

    def __init__(
        self,
        directory: Path,
        tae_checkpoint: Path,
        *,
        output: Path,
        every: int = 1,
        latent_frame: int | None = None,
        context: int = DEFAULT_CONTEXT,
        downscale: int = 0,
        decoder=None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.abort_path = self.directory / ABORT_FILENAME
        self.status_path = self.directory / STATUS_FILENAME
        self.latest_path = self.directory / LATEST_FILENAME
        self.loop_path = self.directory / LOOP_FILENAME
        self._loop_frames = 0
        self.history_path = self.directory / HISTORY_FILENAME

        self.output = Path(output)
        self.every = max(1, int(every))
        self.requested_latent_frame = latent_frame
        self.context = max(0, int(context))
        self.requested_downscale = max(0, int(downscale))
        self.downscale = 1

        self.total_forwards = 0
        self.total_stages = 0
        self.forward = 0
        self.stage_name = ""
        self.stage_index = 0
        self.stage_forward = 0
        self.stage_forwards = 0
        self.sigma: float | None = None
        self.forward_seconds: list[float] = []
        self.overhead_seconds: list[float] = []
        self.started_at = time.time()
        self._started_perf = time.perf_counter()
        self._tick = time.perf_counter()
        self._geometry: dict | None = None
        self._last_preview: Path | None = None
        self._stale_abort_cleared = False
        self._plan: list[tuple[str, int]] = []

        # A sentinel left behind by a previous job would kill this one before its first
        # forward. Clearing it here is the only sane default: ABORT means "stop the render I
        # am watching", and this render did not exist when that file was made.
        if self.abort_path.exists():
            self.abort_path.unlink()
            self._stale_abort_cleared = True

        load_started = time.perf_counter()
        if decoder is not None:
            # Injection seam for the contract test, which gates the file protocol without a
            # 22 MB checkpoint or a GPU.
            self.tae = decoder
        else:
            from .tiny_video_vae import load_tiny_ltx_video_decoder

            self.tae = load_tiny_ltx_video_decoder(tae_checkpoint)
        self.tae_load_seconds = time.perf_counter() - load_started
        self.tae_checkpoint = Path(tae_checkpoint) if tae_checkpoint is not None else None

        self.write_status("starting")

    # -- plan / geometry ------------------------------------------------------------------

    def plan(self, stages: list[tuple[str, int]]) -> None:
        """Declare the whole render's stages as ``[(name, x0_estimates), ...]``.

        Called once, before Stage 1, so ``total_forwards`` and the ETA are right from the
        first thumbnail instead of climbing as stages are discovered.
        """
        self._plan = [(str(name), int(count)) for name, count in stages]
        self.total_stages = len(self._plan)
        self.total_forwards = sum(count for _, count in self._plan)
        self.write_status("starting")

    def start_stage(
        self,
        name: str,
        *,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        estimates: int | None = None,
    ) -> None:
        """Record the stage's latent geometry and pick the frame to preview."""
        frame = latent_frames // 2 if self.requested_latent_frame is None else int(self.requested_latent_frame)
        if not 0 <= frame < latent_frames:
            raise ValueError(f"--live-preview-latent-frame {frame} is outside this render's 0..{latent_frames - 1}")
        self.downscale = (
            auto_downscale(latent_height, latent_width) if self.requested_downscale == 0 else max(1, self.requested_downscale)
        )
        if (latent_height % self.downscale) != 0 or (latent_width % self.downscale) != 0:
            raise ValueError(
                f"--live-preview-downscale {self.downscale} does not divide this render's "
                f"{latent_width}x{latent_height} latent"
            )
        self.stage_name = str(name)
        self.stage_index += 1
        self.stage_forward = 0
        known = dict(self._plan)
        self.stage_forwards = int(estimates) if estimates is not None else int(known.get(name, 0))
        self._geometry = {
            "latent_frames": int(latent_frames),
            "latent_height": int(latent_height),
            "latent_width": int(latent_width),
            "rows_per_frame": int(latent_height) * int(latent_width),
            "latent_frame": frame,
            "preview_width": latent_width // self.downscale * 32,
            "preview_height": latent_height // self.downscale * 32,
            "approx_output_frame": approximate_output_frame(frame),
        }
        self._tick = time.perf_counter()
        self.write_status("running")

    # -- abort ----------------------------------------------------------------------------

    def check_abort(self, stage: str = "between forwards") -> None:
        if self.abort_path.exists():
            # Consume it: the sentinel means "stop this render", and leaving it behind would
            # abort whatever runs into the same directory next.
            try:
                self.abort_path.unlink()
            except FileNotFoundError:
                pass
            self.write_status("aborted", extra={"aborted_at_stage": stage})
            raise LivePreviewAborted(
                f"live preview ABORT sentinel seen {stage} (forward {self.forward}/{self.total_forwards}); "
                f"stopping before any output is written. Status: {self.status_path}"
            )

    # -- per forward ----------------------------------------------------------------------

    def publish(self, x0_tokens: mx.array, sigma: float) -> float:
        """Decode and publish this estimate's x0 thumbnail. Returns the seconds it cost.

        ``x0_tokens`` is the ``(B, N, 128)`` denoised estimate the loop already computed. It is
        read, never written.
        """
        self.forward += 1
        self.stage_forward += 1
        self.sigma = float(sigma)
        now = time.perf_counter()
        self.forward_seconds.append(max(0.0, now - self._tick))
        self._tick = now
        if self._geometry is None:
            raise RuntimeError("start_stage() must run before the first forward")

        last_of_stage = self.stage_forwards and self.stage_forward == self.stage_forwards
        if self.forward % self.every and not last_of_stage and self.forward != self.total_forwards:
            self.overhead_seconds.append(0.0)
            self.write_status("running", extra={"skipped_preview": True})
            self._tick = time.perf_counter()
            return 0.0

        started = time.perf_counter()
        geometry = self._geometry
        rows_per_frame = geometry["rows_per_frame"]
        target = geometry["latent_frame"]
        first = max(0, target - self.context)
        context = target - first
        tokens = context + 1
        lo = first * rows_per_frame
        hi = (target + 1) * rows_per_frame

        rows = x0_tokens[:, lo:hi, :].astype(mx.float32)
        latents = rows.reshape(
            rows.shape[0], tokens, geometry["latent_height"], geometry["latent_width"], self.tae.latent_channels
        ).transpose(0, 4, 1, 2, 3)
        latents = pool_latent(latents, self.downscale)
        index = local_output_index(context)
        decoded = self.tae.decode(latents, index + 1)
        mx.eval(decoded)
        stack = np.array(decoded)[0]  # (C, T, H, W)
        frame = stack[:, index].transpose(1, 2, 0)
        image = (np.clip(frame, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG", compress_level=1)
        payload = buffer.getvalue()

        path = self.directory / f"preview_{self.forward:02d}.png"
        _atomic_write(path, payload)
        _atomic_write(self.latest_path, payload)
        self._last_preview = path

        # THE LOOP IS ALREADY DECODED — and on this lane it is bigger than
        # H3's. ``context`` tokens of causal warm-up exist so the previewed
        # frame does not open the clip (DEFAULT_CONTEXT above), and this
        # decoder emits EIGHT pixel frames per latent token, so the call above
        # produced ``8 * context + 1`` frames — seventeen at the shipped
        # context — of which the line above used exactly one and dropped
        # sixteen. They are the moments immediately before the previewed one,
        # in order: real motion, already computed, for no extra GPU.
        #
        # Written as one animated WebP beside the PNG. A browser loops an
        # animated WebP by itself, so the consumer needs no player and no
        # timer; the PNG is untouched, so every existing consumer keeps its
        # still. Announced in status.json rather than assumed.
        if stack.shape[1] > 1:
            try:
                seq = np.clip(stack.transpose(1, 2, 3, 0), 0.0, 1.0)
                seq = (seq * 255.0 + 0.5).astype(np.uint8)
                frames = [Image.fromarray(f) for f in seq]
                loop_buffer = io.BytesIO()
                frames[0].save(
                    loop_buffer,
                    format="WEBP",
                    save_all=True,
                    append_images=frames[1:],
                    duration=LOOP_FRAME_MS,
                    loop=0,
                    quality=LOOP_QUALITY,
                    method=0,  # fastest setting: this is a monitor, not a deliverable
                )
                _atomic_write(self.loop_path, loop_buffer.getvalue())
                self._loop_frames = len(frames)
            except Exception:
                # A monitor must never be able to fail the render it watches.
                # The still above already shipped; the consumer falls back.
                self._loop_frames = 0
        with open(self.history_path, "a") as handle:
            handle.write(
                json.dumps(
                    {
                        "forward": self.forward,
                        "stage": self.stage_name,
                        "stage_forward": self.stage_forward,
                        "sigma": self.sigma,
                        "preview": path.name,
                        "latent_frame": geometry["latent_frame"],
                        "approx_output_frame": geometry["approx_output_frame"],
                        "elapsed_seconds": round(time.perf_counter() - self._started_perf, 3),
                    }
                )
                + "\n"
            )

        del rows, latents, decoded
        cost = time.perf_counter() - started
        self.overhead_seconds.append(cost)
        self.write_status("running", extra={"preview_seconds": round(cost, 4)})
        self._tick = time.perf_counter()
        return cost

    # -- status ---------------------------------------------------------------------------

    def write_status(self, status: str, extra: dict | None = None) -> None:
        done = len(self.forward_seconds)
        mean = sum(self.forward_seconds) / done if done else None
        remaining = max(0, self.total_forwards - self.forward)
        payload = {
            "schema": SCHEMA,
            "status": status,
            "aborted": status == "aborted",
            "forward": self.forward,
            "total_forwards": self.total_forwards,
            "stage": self.stage_name or None,
            "stage_index": self.stage_index,
            "total_stages": self.total_stages,
            "stage_forward": self.stage_forward,
            "stage_forwards": self.stage_forwards,
            "sigma": self.sigma,
            "preview": self._last_preview.name if self._last_preview else None,
            # The animated companion, named only once it exists. A consumer that
            # does not know the key keeps using `preview` and sees a still.
            "preview_loop": LOOP_FILENAME if self._loop_frames else None,
            "preview_loop_frames": self._loop_frames or None,
            "preview_path": str(self._last_preview) if self._last_preview else None,
            "preview_latest_path": str(self.latest_path) if self._last_preview else None,
            "abort_sentinel": str(self.abort_path),
            "output": str(self.output),
            "pid": os.getpid(),
            "started_at": round(self.started_at, 3),
            "updated_at": round(time.time(), 3),
            "elapsed_seconds": round(time.perf_counter() - self._started_perf, 3),
            "mean_forward_seconds": round(mean, 3) if mean else None,
            "eta_seconds": round(mean * remaining, 1) if mean else None,
            "preview_overhead_seconds": round(sum(self.overhead_seconds), 3),
            "tae_load_seconds": round(self.tae_load_seconds, 3),
            "tae_checkpoint": str(self.tae_checkpoint) if self.tae_checkpoint else None,
            "stale_abort_cleared": self._stale_abort_cleared,
            "every": self.every,
            "context": self.context,
            "downscale": self.downscale,
        }
        if self._geometry is not None:
            payload.update(
                {
                    "latent_frame": self._geometry["latent_frame"],
                    "latent_frames": self._geometry["latent_frames"],
                    "approx_output_frame": self._geometry["approx_output_frame"],
                    "preview_width": self._geometry["preview_width"],
                    "preview_height": self._geometry["preview_height"],
                }
            )
        if extra:
            payload.update(extra)
        _atomic_write(self.status_path, (json.dumps(payload, indent=2) + "\n").encode())

    def finish(self, status: str = "done", extra: dict | None = None) -> None:
        self.write_status(status, extra=extra)

    def summary(self) -> dict:
        overhead = sum(self.overhead_seconds)
        written = sum(1 for value in self.overhead_seconds if value > 0.0)
        return {
            "directory": str(self.directory),
            "schema": SCHEMA,
            "previews_written": written,
            "forwards": self.forward,
            "total_forwards": self.total_forwards,
            "every": self.every,
            "context": self.context,
            "downscale": self.downscale,
            "latent_frame": self._geometry["latent_frame"] if self._geometry else None,
            "preview_size": (
                [self._geometry["preview_width"], self._geometry["preview_height"]] if self._geometry else None
            ),
            "tae_load_seconds": round(self.tae_load_seconds, 3),
            "overhead_seconds": round(overhead, 3),
            "overhead_seconds_per_preview": (round(overhead / written, 4) if written else None),
            "per_forward_overhead_seconds": [round(value, 4) for value in self.overhead_seconds],
        }


__all__ = [
    "ABORT_EXIT_CODE",
    "ABORT_FILENAME",
    "HISTORY_FILENAME",
    "DEFAULT_CONTEXT",
    "LATENT_CELL_BUDGET",
    "LATEST_FILENAME",
    "SCHEMA",
    "STATUS_FILENAME",
    "LivePreviewAborted",
    "LivePreviewMonitor",
    "approximate_output_frame",
    "auto_downscale",
    "local_output_index",
    "pool_latent",
]
