"""HQ two-stage pipeline — res_2s second-order sampler for Stage 1.

Same architecture as TI2VidTwoStagesPipeline but uses the res_2s second-order sampler
instead of Euler for Stage 1 denoising, producing higher quality at fewer steps.
Supports guidance (CFG/STG) with the res_2s sampler.

Ported from ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py
"""

from __future__ import annotations

import mlx.core as mx
from mlx_arsenal.diffusion import TeaCacheController

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import (
    compute_video_latent_shape,
    snap_output_dimensions,
)
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_pipelines_mlx.scheduler import ltx2_schedule, resolve_stage2_sigmas
from ltx_pipelines_mlx.ti2vid_two_stages import DEFAULT_CFG_SCALE, TI2VidTwoStagesPipeline
from ltx_pipelines_mlx.utils.helpers import create_noised_state
from ltx_pipelines_mlx.utils.sampler_choice import model_version_of, resolve_diffusion_step
from ltx_pipelines_mlx.utils.samplers import (
    denoise_loop,
    euler_loop_estimates,
    res2s_denoise_loop,
    res2s_loop_estimates,
)

# TeaCache calibration constants for the HQ res_2s path (LTX-2 stage 1, 30
# steps, 384x576x65 reference shape, MLX bf16 q8). Calibrated 2026-04-27 from
# a 5-prompt run (145 deltas) via scripts/calibrate_teacache.py --hq. The
# robust fitter (scripts/fit_teacache_poly.py) picked degree 1.
#
# res_2s has fundamentally different per-step dynamics from Euler:
# - SDE noise injection between stage 1 and stage 2 inflates delta_in
#   (HQ median 0.66 vs Euler median 0.08).
# - Pearson(delta_in, delta_out) is 0.62 here vs Euler's 0.41 — the polynomial
#   is more predictive, justifying a more aggressive default threshold.
# - The pol(delta) values cluster around 0.8 because delta_in is large and
#   the slope is ~1.27, which produces a "cliff" in skip-rate vs threshold:
#   below ~0.8 nothing skips; at ~1.0 most steps skip. Default 1.0 lands at
#   the sweet spot (~52% skip per simulation, ~2x speedup expected).
LTX2_HQ_TEACACHE_COEFFICIENTS: list[float] = [
    1.2692083808655041,
    -0.033401134092491416,
]
LTX2_HQ_TEACACHE_THRESH: float = 1.0  # tune per use case

#: The SFT value for isolated-modality guidance — what upstream ships and what
#: ``LTX_2_3_HQ_PARAMS`` carries. Anything other than 1.0 makes
#: :meth:`MultiModalGuider.do_isolated_modality_generation` true, which costs a
#: whole extra DiT forward on **every** prediction in stage 1.
MODALITY_SCALE_SFT: float = 3.0

#: The neutral value. At 1.0 the guider's modality term is
#: ``(1 - 1) * (cond - uncond_modality) == 0`` and ``_predict`` never builds the
#: pass at all, so the arithmetic is unchanged *and* the forward is not run.
MODALITY_SCALE_NEUTRAL: float = 1.0

#: The checkpoint generation from which this path stops running the
#: isolated-modality pass by default. Keyed by generation for the same reason
#: :func:`ltx_pipelines_mlx.scheduler.resolve_stage2_sigmas` is: the HQ pipeline
#: is generation-agnostic — a 2.3 checkpoint reaches this exact code — and 2.3's
#: measured behaviour must not move.
MODALITY_GUIDANCE_OFF_SINCE_VERSION: tuple[int, int] = (2, 5)


def resolve_modality_scale(model_version: tuple[int, ...]) -> float:
    """The HQ path's default ``modality_scale`` for a checkpoint generation.

    **This is a deliberate output change on LTX-2.5, not an optimisation.**
    Isolated-modality guidance is a real guidance term; switching it off changes
    the picture. It is defaulted off here because it was measured and then
    passed by eye, not because it was proved neutral:

    * measured — one guidance pass is one third of every *computed* stage-1
      prediction on this path, so dropping it took the pinned High tier from
      **306.9 s to 246.2 s (−60.7 s, −19.8 %)** at 1024x576x121, seed 774411,
      with **no** memory movement (39.52 vs 39.53 GB peak);
    * gated — the owner graded the two clips side by side on 2026-08-12 and
      passed this arm ("G modality is nice") while failing the CFG arm that
      buys the same 61 s ("D2 changes character and has visual weirdness"),
      which is why *this* pass is the one that goes and CFG is untouched.

    Evidence: ``~/AI/projects/phosphene/notes/ltx25_perf_exp1.md`` (arm
    ``G_modality_off``), board row 1 of ``ltx25_perf_board.md``.

    Args:
        model_version: The checkpoint's generation, e.g. ``(2, 5)``. Anything
            below ``(2, 5)`` — including the empty tuple an unreadable
            checkpoint yields — keeps the SFT value, so LTX-2.3 and any
            unrecognised checkpoint render exactly as they did before.

    Returns:
        ``MODALITY_SCALE_NEUTRAL`` on 2.5 and newer, ``MODALITY_SCALE_SFT``
        otherwise. Callers that want the other value pass their own
        ``video_guider_params`` / ``audio_guider_params``, which this default
        never overrides.
    """
    if tuple(model_version) >= MODALITY_GUIDANCE_OFF_SINCE_VERSION:
        return MODALITY_SCALE_NEUTRAL
    return MODALITY_SCALE_SFT


def build_hq_guider_params(
    model_version: tuple[int, ...],
    *,
    cfg_scale: float,
    stg_scale: float,
    video_guider_params: MultiModalGuiderParams | None = None,
    audio_guider_params: MultiModalGuiderParams | None = None,
) -> tuple[MultiModalGuiderParams, MultiModalGuiderParams]:
    """The HQ path's (video, audio) guider params, or the caller's if given.

    Extracted from ``generate_two_stage`` so the defaults can be asserted
    without loading 26 GB of weights. The rescale scales (0.45 video / 1.0
    audio) and the audio CFG 7.0 are ``LTX_2_3_HQ_PARAMS`` verbatim and are
    **not** version-keyed — only ``modality_scale`` is.

    Args:
        model_version: The checkpoint's generation, as
            :func:`~ltx_pipelines_mlx.utils.sampler_choice.model_version_of`
            reports it.
        cfg_scale: The video-side CFG scale. The audio guider stays at 7.0;
            ``_predict`` ORs the two, which is why lowering only the video one
            removes no pass (experiment 1 §3).
        stg_scale: Passed straight through. With ``stg_blocks=[]``,
            ``MultiModalGuiderParams.__post_init__`` folds it to 0.0.
        video_guider_params: Caller override. Returned untouched when set.
        audio_guider_params: Caller override. Returned untouched when set.

    Returns:
        ``(video_params, audio_params)``.
    """
    modality_scale = resolve_modality_scale(model_version)
    if video_guider_params is None:
        video_guider_params = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=0.45,
            modality_scale=modality_scale,
            stg_blocks=[],
        )
    if audio_guider_params is None:
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=stg_scale,
            rescale_scale=1.0,
            modality_scale=modality_scale,
            stg_blocks=[],
        )
    return video_guider_params, audio_guider_params


def _build_hq_teacache_controller(num_steps: int, thresh: float | None) -> TeaCacheController:
    """Construct an HQ-specific TeaCacheController.

    Mirrors :func:`ltx_pipelines_mlx.ti2vid_two_stages._build_teacache_controller`
    but uses ``LTX2_HQ_TEACACHE_COEFFICIENTS`` / ``LTX2_HQ_TEACACHE_THRESH``
    so res_2s gets coefficients fit on its own dynamics.
    """
    if not LTX2_HQ_TEACACHE_COEFFICIENTS:
        raise RuntimeError(
            "TeaCache coefficients for the LTX-2 HQ path are not calibrated yet — "
            "run scripts/calibrate_teacache.py --hq to generate them, then paste "
            "the values into LTX2_HQ_TEACACHE_COEFFICIENTS in this file."
        )
    return TeaCacheController(
        num_steps=num_steps,
        rel_l1_thresh=thresh if thresh is not None else LTX2_HQ_TEACACHE_THRESH,
        coefficients=LTX2_HQ_TEACACHE_COEFFICIENTS,
    )


class TI2VidTwoStagesHQPipeline(TI2VidTwoStagesPipeline):
    """HQ two-stage generation with res_2s second-order sampler.

    Inherits from TI2VidTwoStagesPipeline and overrides Stage 1 to use the res_2s
    sampler for higher quality at fewer steps. Stage 2 is identical.

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        gemma_model_id: Gemma model for text encoding.
        low_memory: Aggressive memory management.
        dev_transformer: Dev transformer filename.
        distilled_lora: Distilled LoRA filename for Stage 2.
        distilled_lora_strength: LoRA fusion strength.
    """

    def generate_two_stage(
        self,
        prompt: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int = 15,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = 0.0,
        image: str | None = None,
        images=None,
        prompt_relay=None,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        enable_teacache: bool = False,
        teacache_thresh: float | None = None,
        tap: callable | None = None,
        live_preview=None,
        loose_reference: bool = False,
    ) -> tuple[mx.array, mx.array]:
        """Generate video using HQ two-stage pipeline with res_2s sampler.

        Same as TI2VidTwoStagesPipeline.generate_two_stage but uses res_2s sampler
        for Stage 1 instead of Euler. ``enable_teacache`` / ``teacache_thresh``
        / ``tap`` are forwarded to ``res2s_denoise_loop`` exactly as in the
        Euler path.

        ``loose_reference`` (2.5 only) — "Inspire": keep the conditioning
        image as subject/style guidance but let the composition re-imagine
        itself, i.e. deliberately skip the masked-sample re-pin that anchors
        i2v. False (the default) resolves per generation: on >= 2.5 the
        sample is re-pinned each res_2s update so i2v actually animates the
        supplied image (the +ltx25.4 fix, extended to this loop); 2.3 keeps
        its historical bytes untouched.
        """
        # --- Text encoding (Prompt Relay: encode the combined prompt) ---
        encode_prompt, relay_token_ranges = self._prompt_relay_setup(prompt, prompt_relay)
        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(encode_prompt)
        num_text_tokens = video_embeds.shape[1]
        relay_mask = self._prompt_relay_mask_builder(prompt_relay, relay_token_ranges, num_text_tokens)

        # --- Load DiT + VAE encoder + upsampler ---
        if self.dit is None:
            self.dit = self._load_dev_transformer()

        self._load_vae_encoder()
        if self.upsampler is None:
            self._load_upsampler()

        assert self.dit is not None
        assert self.vae_encoder is not None
        assert self.upsampler is not None

        # --- Stage 1: Half resolution with res_2s sampler + guidance ---
        # Snap to the two-stage grid (multiples of 64) and report if it changed.
        height, width = snap_output_dimensions(height, width, two_stage=True)
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        # I2V conditioning at half resolution. ``images`` is the upstream-iso
        # multi-anchor list; ``image`` is the legacy single-image shorthand
        # (frame_idx=0, strength=1.0).
        from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
        from ltx_pipelines_mlx.utils.args import ImageConditioningInput

        enc_h_half = H_half * 32
        enc_w_half = W_half * 32
        resolved_images = list(images) if images else []
        if image is not None and not resolved_images:
            resolved_images = [ImageConditioningInput(path=image, frame_idx=0, strength=1.0)]
        conditionings_1: list = []
        if resolved_images:
            conditionings_1 = combined_image_conditionings(
                resolved_images,
                enc_h=enc_h_half,
                enc_w=enc_w_half,
                spatial_dims=(F, H_half, W_half),
                video_encoder=self.vae_encoder,
                frame_rate=frame_rate,
            )

        # Stage 1 video/audio: legacy_scalar_blend=True for bit-exact match
        # (see ti2vid_two_stages.py for rationale).
        video_state = create_noised_state(
            base_shape=video_shape,
            conditionings=conditionings_1,
            spatial_dims=(F, H_half, W_half),
            positions=video_positions_1,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )
        audio_state = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(F, H_half, W_half),  # unused
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )

        # Stage 1 sigma schedule (dynamic for dev model)
        num_tokens = F * H_half * W_half
        sigmas_1 = ltx2_schedule(stage1_steps, num_tokens=num_tokens)
        x0_model = X0Model(self.dit)

        if live_preview is not None:
            # Stage 1 runs at HALF resolution, so its previews are a half-res composition
            # monitor — which is what you want from a monitor, and cheaper. Stage 2's are
            # full-res, so the owner sees the refine land.
            # res_2s is second-order: 2 estimates per step, +1 for the terminal denoise.
            # ``resolve_stage2_sigmas`` is pure and is called again below for the real
            # stage-2 run — this call decides nothing, it only sizes the progress total.
            live_preview.plan(
                [
                    ("stage1", res2s_loop_estimates(sigmas_1)),
                    ("stage2", euler_loop_estimates(resolve_stage2_sigmas(model_version_of(self.dit), stage2_steps))),
                ]
            )
            live_preview.start_stage("stage1", latent_frames=F, latent_height=H_half, latent_width=W_half)

        # Build guider params (HQ defaults: no STG, lower rescale).
        #
        # ``modality_scale`` is keyed by the checkpoint's generation, exactly as
        # the stage-2 schedule is below: 2.5 drops the isolated-modality pass
        # (owner-passed output change, −60.7 s — see resolve_modality_scale),
        # 2.3 keeps the SFT 3.0 it has always had. A caller-supplied
        # ``*_guider_params`` overrides both branches and is not touched here.
        video_guider_params, audio_guider_params = build_hq_guider_params(
            model_version_of(self.dit),
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
        )

        video_factory = create_multimodal_guider_factory(video_guider_params, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_guider_params, negative_context=neg_audio_embeds)

        # Stage 1: res_2s with guidance
        teacache_controller = None
        if enable_teacache:
            teacache_controller = _build_hq_teacache_controller(stage1_steps, teacache_thresh)
            teacache_controller.reset()
        self._pre_denoise_flush(video_state, audio_state)
        # Masked-sample re-pin, resolved per generation like the stage-2
        # schedule and modality scale: 2.5 anchors i2v for real (the
        # +ltx25.4 class, closed on this loop too), 2.3 keeps its
        # historical bytes. ``loose_reference`` (Inspire) turns it off on
        # purpose — the accidental behavior the owner graded as a feature,
        # now an explicit choice instead of a defect.
        _repin = (model_version_of(self.dit) >= (2, 5)) and not loose_reference
        output_1 = res2s_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_1,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            video_cross_attention_mask=relay_mask(F, H_half, W_half, video_state.latent.shape[1]),
            teacache=teacache_controller,
            tap=tap,
            preview=live_preview,
            repin_masked_sample=_repin,
        )
        if self.low_memory:
            aggressive_cleanup()

        # --- Fuse distilled LoRA for Stage 2 ---
        self._fuse_distilled_lora(self.dit)

        # --- Upscale with denormalize/renormalize ---
        # Strip any appended keyframe tokens (multi-anchor with frame_idx>0
        # appends via VideoConditionByKeyframeIndex; only the base
        # F*H*W tokens are spatial latent we need to unpatchify).
        gen_tokens_1 = output_1.video_latent[:, : F * H_half * W_half, :]
        video_half = self.video_patchifier.unpatchify(gen_tokens_1, (F, H_half, W_half))

        video_mlx = video_half.transpose(0, 2, 3, 4, 1)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
        video_upscaled = self.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)
        # NOTE: mx.eval is MLX graph evaluation, NOT Python eval()
        mx.eval(video_upscaled)

        H_full = H_half * 2
        W_full = W_half * 2

        # I2V conditioning at full resolution for Stage 2 (re-encode at upscaled dims)
        conditionings_2: list = []
        if resolved_images:
            enc_h_full = H_full * 32
            enc_w_full = W_full * 32
            conditionings_2 = combined_image_conditionings(
                resolved_images,
                enc_h=enc_h_full,
                enc_w=enc_w_full,
                spatial_dims=(F, H_full, W_full),
                video_encoder=self.vae_encoder,
                frame_rate=frame_rate,
            )

        if self.low_memory:
            self.image_conditioner.free()
            self.upsampler = None
            aggressive_cleanup()

        # --- Stage 2: Refine at full resolution (no CFG) ---
        video_tokens, _ = self.video_patchifier.patchify(video_upscaled)

        # LTX-2.5 moves stage 2's first sigma 0.909375 -> 0.85 (official
        # template, node 395). 2.3 gets its own list, unchanged.
        sigmas_2 = resolve_stage2_sigmas(model_version_of(self.dit), stage2_steps)
        start_sigma = sigmas_2[0]

        video_positions_2 = compute_video_positions(F, H_full, W_full, frame_rate=frame_rate)

        # Stage 2 video: legacy_scalar_blend=True bit-matches the legacy inline
        # ``noise * sigma + video_tokens * (1 - sigma)`` arithmetic.
        video_state_2 = create_noised_state(
            base_shape=video_tokens.shape,
            conditionings=conditionings_2,
            spatial_dims=(F, H_full, W_full),
            positions=video_positions_2,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens,
            legacy_scalar_blend=True,
        )

        # Stage 2 audio: default (mask path) matches legacy noise_latent_state.
        audio_tokens_1 = output_1.audio_latent
        audio_state_2 = create_noised_state(
            base_shape=audio_tokens_1.shape,
            conditionings=[],
            spatial_dims=(F, H_full, W_full),  # unused
            positions=audio_positions,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=audio_tokens_1,
        )

        # Stage 2: simple denoising (no CFG)
        if live_preview is not None:
            live_preview.start_stage("stage2", latent_frames=F, latent_height=H_full, latent_width=W_full)

        self._pre_denoise_flush(video_state_2, audio_state_2)
        output_2 = denoise_loop(
            model=x0_model,
            video_state=video_state_2,
            audio_state=audio_state_2,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_2,
            video_cross_attention_mask=relay_mask(F, H_full, W_full, video_state_2.latent.shape[1]),
            # Stage 1 above is res_2s (its own stochastic sampler, untouched);
            # this refine pass is the Euler one the 2.5 templates replace.
            diffusion_step=resolve_diffusion_step(self.dit),
            preview=live_preview,
            # Inspire carries through the refine pass too — re-pinning the
            # reference here would anchor stage 2 to a composition stage 1
            # deliberately departed from. The default (True) is +ltx25.4's
            # shipped behavior, unchanged for anchored renders.
            repin_masked_sample=not loose_reference,
        )
        if self.low_memory:
            aggressive_cleanup()

        # Strip appended keyframe tokens before unpatchify (see stage 1).
        gen_tokens_2 = output_2.video_latent[:, : F * H_full * W_full, :]
        video_latent = self.video_patchifier.unpatchify(gen_tokens_2, (F, H_full, W_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)

        return video_latent, audio_latent
