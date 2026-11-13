"""Distilled two-stage video generation pipeline.

Mirrors upstream ``ltx_pipelines.distilled.DistilledPipeline`` 1:1:

  Stage 1: Distilled DiT at **half resolution** (8 steps, no CFG).
  Stage 2: Spatial 2x upscaler + distilled DiT refine at **full resolution**
           (2 steps on LTX-2.5, 3 on 2.3, no CFG).

Both schedules come from :func:`~ltx_pipelines_mlx.scheduler.resolve_distilled_schedule`,
which is keyed on the checkpoint's generation. LTX-2.5's stage-2 default is the
graded two-step list (experiment 5 arm S2, adopted 2026-08-12); 2.3 keeps the
vendor 8+3 it always had. ``schedule_preset="fast"`` and explicit
``stage1_sigmas`` / ``stage2_sigmas`` are the opt-ins on top.

Same distilled checkpoint is used in both stages — no LoRA fusion between
stages (the model is already distilled). Use this pipeline when you want
the speed of the distilled model at higher target resolutions, where
running distilled directly at full res can produce out-of-distribution
artefacts.

For the simpler distilled-at-target one-stage path, see
:class:`BasePipeline`.

For dev model + CFG quality, see :class:`TI2VidTwoStagesPipeline` /
:class:`TI2VidTwoStagesHQPipeline`.
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import (
    compute_video_latent_shape,
    snap_output_dimensions,
)
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import (
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_positions,
)

from .scheduler import resolve_distilled_schedule
from .ti2vid_two_stages import TI2VidTwoStagesPipeline
from .utils.helpers import create_noised_state
from .utils.progress import phase
from .utils.sampler_choice import model_version_of, resolve_diffusion_step
from .utils.samplers import denoise_loop, euler_loop_estimates

_materialize = getattr(mx, "eval")  # noqa: B009 -- security hook flags mx.eval pattern


class DistilledPipeline(TI2VidTwoStagesPipeline):
    """Distilled two-stage T2V/I2V pipeline (half-res → upscale → full-res refine).

    Reuses :class:`TI2VidTwoStagesPipeline`'s upsampler loading and helpers but
    overrides ``generate_two_stage`` to:

    - Skip negative-prompt encoding (no CFG).
    - Load the distilled transformer directly (no dev model, no LoRA fusion).
    - Run simple ``denoise_loop`` on both stages with the generation's schedule
      pair from ``resolve_distilled_schedule`` (overridable by preset, by an
      explicit sigma list, or by a step count that thins rather than truncates).

    Args:
        model_dir: Path to model weights or HuggingFace repo ID. Must
            contain the distilled checkpoint (e.g. ``dgrauet/ltx-2.3-mlx-q8``
            ships ``transformer-distilled.safetensors``).
        gemma_model_id: Gemma model for text encoding.
        low_memory: Aggressive memory management.
        low_ram_streaming: Stream transformer blocks from disk.
        tile_count: Optional modality tiling configuration.
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        tile_count=None,
    ):
        super().__init__(
            model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
            tile_count=tile_count,
        )

    def load(self) -> None:
        """Load distilled DiT + VAE encoder + upsampler (skip decoders).

        Skips reloading the text encoder: ``generate_two_stage`` encodes
        the prompt and frees Gemma BEFORE calling :meth:`load`. Loading
        Gemma again here would just thrash the Metal heap (7.5 GB
        load/mmap + free) right before DiT is loaded — a documented
        cause of macOS GPU watchdog crashes under sustained system
        contention.
        """
        if self._loaded:
            return

        if self.dit is None:
            transformer_path = self.model_dir / "transformer.safetensors"
            if not transformer_path.exists():
                transformer_path = self._resolve_safetensors(self.model_dir, "transformer-distilled")
            self.dit = self._load_transformer_with_optional_streaming(transformer_path)

        self._load_vae_encoder()

        if self.upsampler is None:
            self._load_upsampler()

        self._loaded = True

    def generate_two_stage(  # type: ignore[override]
        self,
        prompt: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        stage1_sigmas=None,
        stage2_sigmas=None,
        schedule_preset: str | None = None,
        loose_reference: bool = False,
        image: str | None = None,
        images=None,
        prompt_relay=None,
        live_preview=None,
        **_unused_kwargs,
    ) -> tuple[mx.array, mx.array]:
        """Generate video using the distilled two-stage pipeline.

        Args:
            prompt: Text prompt.
            height: Final video height.
            width: Final video width.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Stage 1 steps. **Thins** the preset's table, keeping
                its terminal 0.0 (it truncated before 2026-08-12, which left the
                stage unfinished).
            stage2_steps: Stage 2 steps, same semantics.
            stage1_sigmas: Explicit stage-1 schedule, e.g.
                ``[1.0, 0.975, 0.909375, 0.725, 0.421875, 0.0]``. Validated:
                strictly decreasing, terminating at 0.0, at most the distilled
                checkpoint's 9 points.
            stage2_sigmas: Explicit stage-2 schedule, same validation.
            schedule_preset: Named schedule for this lane —  ``"default"``,
                ``"fast"`` or ``"vendor"`` on LTX-2.5. See
                :func:`~ltx_pipelines_mlx.scheduler.resolve_distilled_schedule`.
            loose_reference: "Inspire" — keep the reference image as
                subject/style guidance but skip the masked-sample re-pin, so
                the composition re-imagines itself instead of animating the
                image. Default False = anchored i2v (the +ltx25.4 fix). Only
                meaningful with a conditioning image; inert on t2v.
            image: Optional reference image for I2V conditioning.
            **_unused_kwargs: Accepted (and ignored) for signature compatibility
                with :meth:`TI2VidTwoStagesPipeline.generate_two_stage`. CFG / STG /
                TeaCache flags don't apply to the distilled flow.

        Returns:
            Tuple of (video_latent, audio_latent) at full resolution.
        """
        # --- Prompt Relay setup (temporal prompt gating on video cross-attn) ---
        encode_prompt, relay_token_ranges = self._prompt_relay_setup(prompt, prompt_relay)

        # --- Text encoding (positive only — no CFG) ---
        self._load_text_encoder()
        with phase("Encoding prompt", verbose=self.verbose):
            video_embeds, audio_embeds = self._encode_text(encode_prompt)
            _materialize(video_embeds, audio_embeds)
        if self.low_memory:
            self.prompt_encoder.free()
            aggressive_cleanup()

        # Per-stage Prompt Relay mask builder. Ranges were computed pre-encode;
        # the mask is rebuilt each stage because tokens-per-frame (H*W) differs.
        num_text_tokens = video_embeds.shape[1]
        relay_mask = self._prompt_relay_mask_builder(prompt_relay, relay_token_ranges, num_text_tokens)

        # --- Load distilled DiT + VAE encoder + upsampler ---
        self.load()
        assert self.dit is not None
        assert self.vae_encoder is not None
        assert self.upsampler is not None

        # Both schedules are resolved here, before stage 1 spends a minute of
        # GPU: an invalid schedule should fail at the door, not between stages.
        sigmas_1, sigmas_2 = resolve_distilled_schedule(
            model_version_of(self.dit),
            preset=schedule_preset,
            stage1_sigmas=stage1_sigmas,
            stage2_sigmas=stage2_sigmas,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
        )
        if self.verbose:
            print(
                f"  Schedule: stage 1 {len(sigmas_1) - 1} steps {sigmas_1} | "
                f"stage 2 {len(sigmas_2) - 1} steps {sigmas_2}"
            )

        # --- Stage 1: half resolution ---
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
        # multi-anchor list; ``image`` is the legacy single-image shorthand.
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

        if live_preview is not None:
            # Both schedules were resolved together above, before any GPU was
            # spent, so the preview's total (and its ETA) is right from the
            # first thumbnail. Deliberately reuses `sigmas_1` / `sigmas_2`
            # rather than re-deriving them: a step count thins the checkpoint's
            # table through `resolve_distilled_schedule`, and a second
            # derivation here would be a second place for that to drift.
            live_preview.plan(
                [
                    ("stage1", euler_loop_estimates(sigmas_1)),
                    ("stage2", euler_loop_estimates(sigmas_2)),
                ]
            )
            live_preview.start_stage(
                "stage1", latent_frames=F, latent_height=H_half, latent_width=W_half
            )

        stage1_dit = self.dit
        if self._tile_count is not None:
            from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler

            tiler_1 = VideoModalityTiler(self._tile_count, latent_shape=(F, H_half, W_half))
            stage1_dit = TiledLTXModel(self.dit, tiler_1)

        x0_model = X0Model(stage1_dit)

        self._pre_denoise_flush(video_state, audio_state)
        output_1 = denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_1,
            video_cross_attention_mask=relay_mask(F, H_half, W_half, video_state.latent.shape[1]),
            # LTX-2.5 samples stage 1 ancestrally; 2.3 gets None -> plain Euler.
            diffusion_step=resolve_diffusion_step(self.dit),
            preview=live_preview,
            # Inspire: deliberately skip the anchor re-pin (see the docstring).
            repin_masked_sample=not loose_reference,
        )
        if self.low_memory:
            aggressive_cleanup()

        # --- Upscale (same denorm/upsample/renorm as TI2VidTwoStagesPipeline) ---
        # Strip appended keyframe tokens (multi-anchor with frame_idx>0).
        gen_tokens_1 = output_1.video_latent[:, : F * H_half * W_half, :]
        video_half = self.video_patchifier.unpatchify(gen_tokens_1, (F, H_half, W_half))
        video_mlx = video_half.transpose(0, 2, 3, 4, 1)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
        video_upscaled = self.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)
        _materialize(video_upscaled)

        H_full = H_half * 2
        W_full = W_half * 2

        # I2V conditioning at full resolution (re-encode at upscaled dims)
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

        # --- Stage 2: full resolution refine (no LoRA swap — already distilled) ---
        video_tokens, _ = self.video_patchifier.patchify(video_upscaled)
        # ``sigmas_2`` was resolved before stage 1 (see above). Its first value
        # is also the re-noising level: LTX-2.5 starts stage 2 at 0.85, so 15 %
        # of stage 1 survives into the refine and 85 % is fresh noise.
        start_sigma = sigmas_2[0]

        video_positions_2 = compute_video_positions(F, H_full, W_full, frame_rate=frame_rate)

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

        stage2_x0_model = x0_model
        if self._tile_count is not None:
            from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler

            tiler_2 = VideoModalityTiler(self._tile_count, latent_shape=(F, H_full, W_full))
            stage2_x0_model = X0Model(TiledLTXModel(self.dit, tiler_2))

        if live_preview is not None:
            live_preview.start_stage("stage2", latent_frames=F, latent_height=H_full, latent_width=W_full)

        self._pre_denoise_flush(video_state_2, audio_state_2)
        output_2 = denoise_loop(
            model=stage2_x0_model,
            video_state=video_state_2,
            audio_state=audio_state_2,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_2,
            video_cross_attention_mask=relay_mask(F, H_full, W_full, video_state_2.latent.shape[1]),
            diffusion_step=resolve_diffusion_step(self.dit),
            preview=live_preview,
            repin_masked_sample=not loose_reference,
        )
        if self.low_memory:
            aggressive_cleanup()

        gen_tokens_2 = output_2.video_latent[:, : F * H_full * W_full, :]
        video_latent = self.video_patchifier.unpatchify(gen_tokens_2, (F, H_full, W_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)

        return video_latent, audio_latent


__all__ = ["DistilledPipeline"]
