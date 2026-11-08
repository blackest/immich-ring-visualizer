# soft_finish -- vendored colour pass

`soft_finish.py` is **our** script (mp4 in, mp4 out through ffmpeg pipes). Its maths is a port of two nodes from
**Orion4D_FXMax v1.0.0** -- MIT, https://github.com/orion4d/Orion4D_FXMax -- read from the pack installed at
`/Volumes/AI/ComfyUI/custom_nodes/orion4d_fxmax` on 2026-11-08 (the folder is not a git checkout, so there is no commit
to pin; `pyproject.toml` says `version = "1.0.0"`, `license = "MIT"`).

| ours | theirs |
|---|---|
| `gaussian_blur`, `midtone_mask`, `clarity`, `luma` | `utils.py` `gaussian_blur` / `get_sigma_from_radius` / `gaussian_kernel1d` / `midtone_mask` / `rgb_to_luma`; `Orion4D_FXMax.py` `_clarity` / `_apply_luma_detail` (luma mode) |
| `tone_cineon` | `color_fx_common.py` `_apply_dctl_tone_mapper`, mode "Cineon-ish" (`_tone_map_cineon_ish_np`), exposure 0, contrast 1, no shadow lift or black floor |

What changed in the port: torch -> numpy, and torch's reflect-padded `conv2d` -> `cv2.GaussianBlur` with
`BORDER_REFLECT_101` (the same border rule; the kernel is the same 3-sigma Gaussian, sigma = max(0.15, radius*0.5+0.25)).
Only the two steps the recipe uses are ported; the pack's other effects, its web app and its presets are not.

Checked 2026-11-08 against the ComfyUI route on a 5 s character-LoRA clip: five frames differ by 1.5-1.8 levels on average
(max 12-19) where the recipe itself moves a pixel by about 8.7. The port ends slightly brighter (mean 73.3 against 72.4),
mostly colour-conversion differences in the two decoders.

If the pack's maths changes upstream this copy does not follow; compare before updating.
