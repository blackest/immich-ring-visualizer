"""LTX-2 sigma schedules.

`ltx2_schedule` is a thin wrapper over `mlx_arsenal.diffusion.dynamic_shift_schedule`
that preserves LTX's original keyword name (``steps``) and default ``num_tokens``.
The predefined LTX-specific tables (DISTILLED_SIGMAS, STAGE_2_SIGMAS) and the
LTX-only helpers (get_sigma_schedule, sigma_to_timestep) stay local.

**Every schedule this module hands out terminates at sigma 0.0.** A schedule that
stops short does not "run fewer steps" — it returns a latent with residual noise
in it, which then gets attributed to the step count. Asking for fewer steps
therefore *thins* a table (keep both endpoints, drop interior points) rather than
truncating it; a request that cannot be thinned is refused with a message instead
of being quietly served as an unfinished denoise. See :func:`thin_sigmas`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
from mlx_arsenal.diffusion import dynamic_shift_schedule

_MAX_SHIFT_ANCHOR = 4096


def ltx2_schedule(
    steps: int,
    num_tokens: int = _MAX_SHIFT_ANCHOR,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> list[float]:
    """LTX-2 token-count-adaptive flow-matching sigma schedule."""
    return dynamic_shift_schedule(
        steps,
        num_tokens=num_tokens,
        base_shift=base_shift,
        max_shift=max_shift,
        stretch=stretch,
        terminal=terminal,
    )


__all__ = [
    "DISTILLED_MAX_POINTS",
    "DISTILLED_PRESET_NAMES",
    "DISTILLED_SIGMAS",
    "DISTILLED_SIGMAS_LTX25_FAST",
    "STAGE_2_SIGMAS",
    "STAGE_2_SIGMAS_LTX25",
    "STAGE_2_SIGMAS_LTX25_S2",
    "DistilledSchedule",
    "distilled_presets_for",
    "get_sigma_schedule",
    "ltx2_schedule",
    "resolve_distilled_schedule",
    "resolve_stage2_sigmas",
    "sigma_to_timestep",
    "thin_sigmas",
    "validate_sigmas",
]

# Predefined sigma schedule for 8-step distilled model.
# 9 values = 8 steps (iterate consecutive pairs: sigmas[i], sigmas[i+1]).
DISTILLED_SIGMAS: list[float] = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]

# Sigma schedule for stage 2 refinement (two-stage pipeline).
# 4 values = 3 steps.
#
# This is the LTX-2.3 schedule, and it is exactly the tail of DISTILLED_SIGMAS
# (indices 5..8) — stage 2 picks up where stage 1's last few steps were.
STAGE_2_SIGMAS: list[float] = [
    0.909375,
    0.725,
    0.421875,
    0.0,
]

# LTX-2.5's stage-2 schedule, read verbatim from the official ComfyUI template
# `video_ltx2_5_t2v.json` (node 395, `ManualSigmas`): "0.85, 0.7250, 0.4219, 0.0".
#
# **Only the FIRST value differs from 2.3** — 0.85 where we use 0.909375. The
# other two are the same numbers the template prints at four decimal places
# (0.4219 is 0.421875 rounded for display, not a different sigma), which is what
# makes the first value's change deliberate rather than a transcription artifact:
# whoever wrote the template took 2.3's tail and moved one number.
#
# It matters because it is a REFINE pass. Starting stage 2 at 0.909375 re-noises
# the upscaled latent harder than the vendor does, so the refine deviates further
# from the stage-1 result. We shipped the 2.3 value on 2.5 until 2026-08-12.
#
# Stage 1 needs no such entry: our DISTILLED_SIGMAS is byte-identical to the same
# template's node 404, all nine values including the five inside the top 2.5% of
# the noise range that no naive shifted schedule reproduces.
STAGE_2_SIGMAS_LTX25: list[float] = [
    0.85,
    0.725,
    0.421875,
    0.0,
]

# LTX-2.5 distilled lane, stage 2 at TWO steps — experiment 5's arm S2, adopted
# as this lane's default on 2026-08-12 (`notes/ltx25_perf_exp45.md` §B.5, board
# row 5). It is the vendor's list with 0.725 removed and the terminal 0.0 kept.
#
# Measured against the vendor 8+3 at 1024x576x121, q8, seed 774411: **170.2 s ->
# 140.7 s, -29.6 s (-17.4 %)** for one fewer stage-2 forward, at composition
# correlation **0.9988** — i.e. the same take. A stage-2 forward costs 4.6x a
# stage-1 forward on this lane (29.6 s vs 6.40 s), which is why stage 2 is the
# cheapest place to thin.
#
# Do not mistake 0.9988 for "identical output". It is not: dropping a sigma point
# changes the trajectory and the ancestral sampler's RNG draw, so this schedule
# renders a *different file* from the vendor list on the same seed. Owner verdict
# 2026-08-12, judgment 2: "quality is fine" — adopted with that known re-roll.
STAGE_2_SIGMAS_LTX25_S2: list[float] = [
    0.85,
    0.421875,
    0.0,
]

# LTX-2.5 distilled lane, stage 1 front cluster 5 points -> 2 — experiment 5's
# arm F6. Approved 2026-08-12 as the "fast" draft preset (stacked with the S2
# stage 2 above it renders in 120.8 s, **-49.4 s / -29.0 %**).
#
# **This changes the take, and that is inherent rather than a defect.** On a
# rectified-flow model the first updates decide the composition, so removing
# three of the five points inside the top 2.5 % of the noise range rerolls the
# shot: composition correlation 0.920 against the vendor list, a coherent take
# but a different one. Use it for drafts and for exploring takes cheaply, not to
# make a cheaper copy of a render you already like.
DISTILLED_SIGMAS_LTX25_FAST: list[float] = [
    1.0,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]

#: The distilled checkpoint is trained for 8 steps, so 9 points is the ceiling on
#: an explicit schedule for either of its stages. Longer lists are refused rather
#: than silently clamped.
DISTILLED_MAX_POINTS: int = len(DISTILLED_SIGMAS)


def validate_sigmas(
    sigmas,
    *,
    name: str = "schedule",
    max_points: int | None = None,
) -> list[float]:
    """Return ``sigmas`` as a list, or raise ``ValueError`` saying what is wrong.

    A usable sigma schedule is at least two points, strictly decreasing, starts
    no higher than 1.0, and **terminates at exactly 0.0**. The terminal zero is
    the load-bearing one: without it the last step never returns the x0 estimate
    and the stage hands residual noise to whatever comes next.

    Args:
        sigmas: The candidate schedule.
        name: What to call it in error messages (e.g. ``"--stage1-sigmas"``).
        max_points: Optional ceiling on the number of points.

    Returns:
        The schedule as a fresh ``list[float]``.

    Raises:
        ValueError: On any violation, quoting the offending list.
    """
    values = [float(s) for s in sigmas]
    if len(values) < 2:
        raise ValueError(f"{name} needs at least 2 points (1 step), got {values}")
    if any(math.isnan(v) or math.isinf(v) for v in values):
        raise ValueError(f"{name} contains a non-finite sigma: {values}")
    if values[0] > 1.0:
        raise ValueError(f"{name} starts above 1.0, which is outside the noise range: {values}")
    for a, b in zip(values[:-1], values[1:]):
        if not a > b:
            raise ValueError(f"{name} must be strictly decreasing: {values}")
    if values[-1] != 0.0:
        raise ValueError(
            f"{name} must terminate at 0.0 or the stage stops mid-denoise and hands on a "
            f"latent with residual noise in it: {values}"
        )
    if max_points is not None and len(values) > max_points:
        raise ValueError(
            f"{name} has {len(values)} points; the distilled checkpoint is trained for "
            f"{max_points - 1} steps ({max_points} points) and longer schedules are out of "
            f"distribution: {values}"
        )
    return values


def thin_sigmas(sigmas, steps: int | None, *, name: str = "schedule") -> list[float]:
    """Thin a sigma table to ``steps`` steps, **keeping both endpoints**.

    This is what a step count means on a fixed distilled table. The obvious
    alternative — ``sigmas[: steps + 1]`` — is what this codebase shipped until
    2026-08-12, and it is wrong: it drops the terminal 0.0, so ``--stage2-steps
    2`` on LTX-2.5 produced ``[0.85, 0.725, 0.421875]``, an unfinished refine,
    not a two-step one. Anyone who reached for those flags to "run it cheaper"
    was rendering a half-denoised latent and attributing the result to the step
    count (`notes/ltx25_perf_exp45.md` §B.8).

    Interior points are dropped at a uniform index stride. The rule is
    mechanical and carries no quality claim — it does not know which points a
    given checkpoint can spare. For the schedules that *have* been graded, use
    the named presets (:func:`resolve_distilled_schedule`) or pass the list
    explicitly.

    Args:
        sigmas: The table to thin. Assumed already valid.
        steps: How many steps to end up with. ``None`` or ``0`` returns the
            table unchanged.
        name: What to call it in error messages.

    Returns:
        ``steps + 1`` sigmas, starting at ``sigmas[0]`` and ending at
        ``sigmas[-1]``.

    Raises:
        ValueError: If ``steps`` is negative, or exceeds what the table holds.
            Padding a fixed table up to a longer schedule is a different
            operation with a different answer, so it is refused rather than
            guessed at.
    """
    table = list(sigmas)
    if steps is None or steps == 0:
        return table
    available = len(table) - 1
    if steps < 0:
        raise ValueError(f"{name}: step count must be positive, got {steps}")
    if steps > available:
        raise ValueError(
            f"{name}: cannot thin a {len(table)}-point schedule ({available} steps) up to "
            f"{steps} steps. Pass the points you want explicitly (--stage1-sigmas / "
            f"--stage2-sigmas) — padding a fixed distilled table is a guess, not a thinning."
        )
    if steps == available:
        return table
    stride = available / steps
    indices = [int(i * stride + 0.5) for i in range(steps + 1)]
    indices[-1] = available  # exact, rather than trusting float rounding
    return [table[i] for i in indices]


def resolve_stage2_sigmas(model_version: tuple[int, ...], stage2_steps: int | None = None) -> list[float]:
    """Stage-2 schedule for a checkpoint generation, thinned as the caller asks.

    Args:
        model_version: The checkpoint's generation, e.g. ``(2, 5)``. Anything
            below ``(2, 5)`` — including the empty tuple an unreadable
            checkpoint yields — gets the 2.3 schedule, so an unknown checkpoint
            keeps the older behaviour instead of opting into a newer schedule.
        stage2_steps: Number of steps; ``None``/0 means the full schedule.

    Returns:
        The sigma list, always terminating at 0.0. A step count **thins** the
        table (:func:`thin_sigmas`) instead of truncating it; before 2026-08-12
        it truncated, and a truncated stage 2 is an unfinished refine.

    Note:
        This is the schedule for the *dev* two-stage lanes' refine pass. The
        distilled lane resolves its own pair through
        :func:`resolve_distilled_schedule`, because its default was graded
        separately.
    """
    sigmas = STAGE_2_SIGMAS_LTX25 if tuple(model_version) >= (2, 5) else STAGE_2_SIGMAS
    return thin_sigmas(sigmas, stage2_steps, name="stage 2")


@dataclass(frozen=True)
class DistilledSchedule:
    """A named (stage 1, stage 2) sigma pair for the distilled two-stage lane."""

    stage1: tuple[float, ...]
    stage2: tuple[float, ...]
    summary: str

    def as_lists(self) -> tuple[list[float], list[float]]:
        """The pair as fresh mutable lists, so a caller cannot edit the table."""
        return list(self.stage1), list(self.stage2)


# The 2.3 lane, pinned. Its schedules are the vendor's and nothing here moves
# them: experiment 5 ran entirely on the 2.5 distilled checkpoint and says
# nothing about what 2.3 can spare.
_PRESETS_LTX23: dict[str, DistilledSchedule] = {
    "default": DistilledSchedule(
        stage1=tuple(DISTILLED_SIGMAS),
        stage2=tuple(STAGE_2_SIGMAS),
        summary="LTX-2.3 vendor schedule, 8+3 steps",
    ),
}
_PRESETS_LTX23["vendor"] = _PRESETS_LTX23["default"]

# The 2.5 distilled lane. ``default`` is arm S2, adopted 2026-08-12 on the
# owner's verdict; ``vendor`` reproduces what shipped before that date.
_PRESETS_LTX25: dict[str, DistilledSchedule] = {
    "default": DistilledSchedule(
        stage1=tuple(DISTILLED_SIGMAS),
        stage2=tuple(STAGE_2_SIGMAS_LTX25_S2),
        summary="8+2 steps, -17 % wall against the vendor list, same take (composition 0.9988)",
    ),
    "fast": DistilledSchedule(
        stage1=tuple(DISTILLED_SIGMAS_LTX25_FAST),
        stage2=tuple(STAGE_2_SIGMAS_LTX25_S2),
        summary="5+2 steps, -29 % wall, DIFFERENT take (composition 0.920) — drafts",
    ),
    "vendor": DistilledSchedule(
        stage1=tuple(DISTILLED_SIGMAS),
        stage2=tuple(STAGE_2_SIGMAS_LTX25),
        summary="8+3 steps, the vendor template's own lists (this lane's default before 2026-08-12)",
    ),
}

#: Every preset name, for CLI ``choices`` and for the error message when one misses.
DISTILLED_PRESET_NAMES: tuple[str, ...] = ("default", "fast", "vendor")


def distilled_presets_for(model_version: tuple[int, ...]) -> dict[str, DistilledSchedule]:
    """The presets available on a checkpoint generation.

    2.5 and newer get the graded set; everything older gets ``default`` /
    ``vendor`` only, both of which are the 2.3 vendor schedule. The thinned
    presets are deliberately *absent* rather than silently remapped on 2.3 — no
    2.3 render was ever graded on them.
    """
    return _PRESETS_LTX25 if tuple(model_version) >= (2, 5) else _PRESETS_LTX23


def resolve_distilled_schedule(
    model_version: tuple[int, ...],
    *,
    preset: str | None = None,
    stage1_sigmas=None,
    stage2_sigmas=None,
    stage1_steps: int | None = None,
    stage2_steps: int | None = None,
) -> tuple[list[float], list[float]]:
    """Both sigma schedules for one distilled two-stage render.

    Precedence, per stage independently:

    1. an explicit sigma list wins;
    2. else a step count thins **the checkpoint's own full schedule** — the
       ``vendor`` preset for this generation — to that many steps;
    3. else the chosen preset's list is used whole.

    Passing an explicit list *and* a step count for the same stage is a
    contradiction and raises rather than picking one.

    **Step 2 deliberately thins the vendor table rather than the chosen
    preset's.** A step count is a request about the checkpoint ("give me N of
    its steps"), not about a preset, and reading it the other way would make
    ``stage2_steps=3`` — which the Phosphene panel and this package's own
    ic-lora / lipdub / keyframe defaults all pass explicitly — *fail* on
    LTX-2.5, because the adopted default holds only 2. A shipped default must
    not turn an existing caller's arguments into an error.

    Args:
        model_version: The checkpoint's generation, from
            ``utils.sampler_choice.model_version_of``.
        preset: A name from :data:`DISTILLED_PRESET_NAMES`, or ``None`` for
            ``"default"``.
        stage1_sigmas: Explicit stage-1 schedule, validated.
        stage2_sigmas: Explicit stage-2 schedule, validated. Its **first** value
            is not just a starting point: stage 2 re-noises the upscaled latent
            to it (``noise*sigma + stage1*(1-sigma)``), so on LTX-2.5's 0.85 only
            15 % of stage 1 survives into the refine. Moving it changes far more
            than one step.
        stage1_steps: Thin the checkpoint's stage-1 table to this many steps.
        stage2_steps: Thin the checkpoint's stage-2 table to this many steps.
            On LTX-2.5 that table is the vendor's 3-step list, so ``3`` returns
            it whole and ``2`` returns the adopted default.

    Returns:
        ``(sigmas_1, sigmas_2)``, both terminating at 0.0.

    Raises:
        ValueError: On an unknown preset, a contradictory pair of inputs, or a
            schedule that fails :func:`validate_sigmas`.
    """
    presets = distilled_presets_for(model_version)
    chosen = preset or "default"
    if chosen not in presets:
        known = ", ".join(sorted(presets))
        version = ".".join(str(v) for v in model_version) or "unknown"
        raise ValueError(
            f"unknown distilled schedule preset {chosen!r} for a {version} checkpoint. "
            f"Available here: {known}. (The thinned presets were graded on the LTX-2.5 "
            f"distilled checkpoint and are not offered on older ones.)"
        )
    base_1, base_2 = presets[chosen].as_lists()
    full_1, full_2 = presets["vendor"].as_lists()

    def _one(stage: str, base: list[float], full: list[float], explicit, steps: int | None) -> list[float]:
        flag = f"--stage{stage}-sigmas"
        if explicit is not None:
            if steps is not None:
                raise ValueError(
                    f"pass either {flag} or --stage{stage}-steps, not both: an explicit "
                    f"schedule already says how many steps it has."
                )
            return validate_sigmas(explicit, name=flag, max_points=DISTILLED_MAX_POINTS)
        resolved = base if steps is None or steps == 0 else thin_sigmas(full, steps, name=f"stage {stage}")
        return validate_sigmas(resolved, name=f"stage {stage}", max_points=DISTILLED_MAX_POINTS)

    return (
        _one("1", base_1, full_1, stage1_sigmas, stage1_steps),
        _one("2", base_2, full_2, stage2_sigmas, stage2_steps),
    )


def get_sigma_schedule(
    schedule_name: str = "distilled",
    num_steps: int | None = None,
) -> list[float]:
    """Get a sigma schedule by name.

    Args:
        schedule_name: "distilled", "stage_2" (LTX-2.3) or "stage_2_ltx25".
        num_steps: Optional number of **steps**. The schedule is thinned to that
            many steps, keeping both endpoints. Before 2026-08-12 this argument
            was a point count and it sliced (``sigmas[:num_steps]``), which
            returned schedules that did not end at 0.0.

    Returns:
        List of sigma values, always terminating at 0.0.
    """
    if schedule_name == "distilled":
        sigmas = DISTILLED_SIGMAS
    elif schedule_name == "stage_2":
        sigmas = STAGE_2_SIGMAS
    elif schedule_name == "stage_2_ltx25":
        sigmas = STAGE_2_SIGMAS_LTX25
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")

    return thin_sigmas(sigmas, num_steps, name=schedule_name)


def sigma_to_timestep(sigma: float) -> mx.array:
    """Convert sigma to timestep array.

    Args:
        sigma: Noise level.

    Returns:
        Timestep as (1,) array.
    """
    return mx.array([sigma], dtype=mx.bfloat16)
