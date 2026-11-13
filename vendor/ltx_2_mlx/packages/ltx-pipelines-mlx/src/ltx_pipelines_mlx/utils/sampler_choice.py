"""Which diffusion step a checkpoint's generation expects.

LTX-2.5 introduced the ancestral (SDE) Euler sampler and upstream gates it on
the checkpoint's own generation: ``ANCESTRAL_SAMPLER_SINCE_VERSION = (2, 5)``.
:class:`~ltx_core_mlx.components.diffusion_steps.EulerAncestralDiffusionStep`
has been implemented and tested here since the port landed — and, until this
module existed, **nothing instantiated it**. Every 2.5 render this project has
produced ran the 2.3 Euler step.

Three independent sources say that is wrong:

1. upstream's own constant, quoted above;
2. all three official ComfyUI 2.5 templates (`video_ltx2_5_t2v.json`,
   `_i2v.json`, `_flf2v.json`) select ``euler_ancestral``;
3. the first community settings crib sheet names Euler / Euler-ancestral.

**The templates also disagree with each other about ``eta``, and the
disagreement is meaningful rather than sloppy.** t2v and i2v use ComfyUI's
plain ``euler_ancestral`` sampler, whose eta is 1.0. flf2v — first-frame /
last-frame interpolation — instead uses ``SamplerEulerAncestral`` with **eta
explicitly 0**, and at ``eta = 0`` the ancestral step collapses *exactly* onto
the deterministic Euler step (asserted in `tests/test_ltx25_ancestral_sampler.py`).
So the vendor is saying: inject ancestral noise for open-ended generation, and
do not inject it when the result is pinned between two given frames. That maps
onto our keyframe pipeline directly, which is why `resolve_diffusion_step`
takes an explicit ``eta`` rather than assuming one.

Design rules, both load-bearing:

* **Returning ``None`` means "the caller's existing Euler path", not "no
  sampler".** A 2.3 checkpoint must produce byte-identical output to what it
  produced before this module existed, and the cheapest way to guarantee that
  is for the 2.3 branch to change no code path at all.
* **An unreadable version falls back to 2.3.** `LTXModelConfig.model_version`
  already defaults to ``(2, 3)`` and an unparseable version yields ``()``,
  which compares below every real version. An unknown checkpoint therefore gets
  the older, deterministic behaviour rather than opting itself into a newer
  sampler.
"""

from __future__ import annotations

import os

from ltx_core_mlx.components.diffusion_steps import EulerAncestralDiffusionStep

__all__ = [
    "ANCESTRAL_ETA",
    "ANCESTRAL_SAMPLER_SINCE_VERSION",
    "ANCESTRAL_S_NOISE",
    "KEYFRAME_ETA",
    "SAMPLER_ENV_VAR",
    "model_version_of",
    "resolve_diffusion_step",
]

#: Upstream ``ltx_pipelines.utils.constants.ANCESTRAL_SAMPLER_SINCE_VERSION``.
ANCESTRAL_SAMPLER_SINCE_VERSION: tuple[int, int] = (2, 5)

#: ComfyUI's ``euler_ancestral`` sampler node, which the official 2.5 t2v and
#: i2v templates select, runs eta 1.0 / s_noise 1.0.
ANCESTRAL_ETA: float = 1.0
ANCESTRAL_S_NOISE: float = 1.0

#: The official flf2v template pins eta to 0 — deterministic Euler — for
#: generation bounded by given keyframes.
KEYFRAME_ETA: float = 0.0

#: Escape hatch for A/B measurement and for a user whose checkpoint lies about
#: its generation. ``euler`` forces the legacy step, ``euler_ancestral`` forces
#: the new one regardless of version. Mirrors the panel's LTX_FORCE_CAP_TIER
#: precedent: an override that is obvious in the logs and never the default.
SAMPLER_ENV_VAR = "LTX_SAMPLER"

_LEGACY_NAMES = {"euler", "legacy", "off"}
_ANCESTRAL_NAMES = {"euler_ancestral", "ancestral", "euler-ancestral"}


def model_version_of(model) -> tuple[int, ...]:
    """Read a model's checkpoint generation, tolerating every wrapper.

    Pipelines hold the DiT variously as a bare ``LTXModel``, wrapped in
    ``X0Model``, in ``StreamingLTXModel``, or in ``TiledLTXModel``. Rather than
    teach this function about each wrapper, it walks ``.model`` until it finds
    something carrying a config — and returns ``()`` if it never does, which
    compares below every real version and therefore selects the legacy step.
    """
    seen = 0
    while model is not None and seen < 8:
        config = getattr(model, "config", None)
        version = getattr(config, "model_version", None)
        if version is not None:
            return tuple(version)
        model = getattr(model, "model", None)
        seen += 1
    return ()


def resolve_diffusion_step(
    model,
    *,
    eta: float = ANCESTRAL_ETA,
    s_noise: float = ANCESTRAL_S_NOISE,
) -> EulerAncestralDiffusionStep | None:
    """The diffusion step this model's generation expects, or ``None`` for Euler.

    Args:
        model: The DiT, or any wrapper around it (see :func:`model_version_of`).
        eta: Ancestral noise fraction. ``0.0`` makes the ancestral step
            identical to the Euler step, which is what the official flf2v
            template asks for; the default 1.0 is what t2v/i2v ask for.
        s_noise: Scale on the injected noise. Upstream and every official
            template use 1.0.

    Returns:
        An :class:`EulerAncestralDiffusionStep` when the checkpoint is 2.5 or
        newer and ``eta > 0``; otherwise ``None``, meaning the caller should
        take its existing Euler path unchanged.
    """
    override = (os.environ.get(SAMPLER_ENV_VAR) or "").strip().lower()
    if override in _LEGACY_NAMES:
        return None
    if override in _ANCESTRAL_NAMES:
        return EulerAncestralDiffusionStep(eta=eta, s_noise=s_noise) if eta > 0 else None
    if override:
        raise ValueError(
            f"{SAMPLER_ENV_VAR}={override!r} is not a sampler. Use one of: "
            f"{sorted(_LEGACY_NAMES | _ANCESTRAL_NAMES)}, or unset it to follow the checkpoint."
        )

    if model_version_of(model) < ANCESTRAL_SAMPLER_SINCE_VERSION:
        return None
    # eta == 0 is arithmetically the Euler step. Returning None rather than a
    # zero-eta stepper keeps that case on the byte-identical code path and
    # saves a per-step noise tensor nobody would use.
    if eta <= 0:
        return None
    return EulerAncestralDiffusionStep(eta=eta, s_noise=s_noise)
