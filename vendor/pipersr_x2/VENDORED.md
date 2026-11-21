# pipersr_x2: 2x sharpen of a finished clip on the Apple Neural Engine (ours)

`pipersr_x2_run.py` is **our own** small runner; nothing of upstream is copied. It runs as a subprocess of
`h3_sharpNG.py` on this folder's own env (`.venv`, Python 3.12, built by `build_venv.sh` from `requirements.lock`), so the app
never imports PiperSR. Python 3.12 because `coremltools` has no native Core ML library for 3.14.

**Model:** PiperSR 1.0.0 by Ben Racicot (https://modelpiper.com, https://github.com/ModelPiper/PiperSR), a small image
super-resolution model for the Neural Engine, 128x128 tiles in and 256x256 out, bundled in the `pipersr` wheel.

**Licence:** the `pipersr` code is AGPL-3.0; the model weights are free for personal and academic use with attribution, and
commercial use needs a separate licence from the author (contact in the package metadata). I have not seen the full
`MODEL_LICENSE` text, only this summary from the package README. Attribution: "Powered by PiperSR from ModelPiper -
https://modelpiper.com" (printed by the runner and shown on the Sharp 2x button).

**What it does and does not do:** it sharpens edges and keeps the picture; it does not redraw it (LTX's upscaler does that,
see `upscale_x2NG.py`), so blocks from a low-resolution render stay. Frames are processed one by one, so shimmer between
frames is possible (not yet checked on moving footage). Tiles overlap by 16 px on each side and only each tile's centre is
kept, so there are no grid seams. Measured 2026-11-21 on an M2 Max: 124 frames at 640x384 -> 1280x768 in 18 s, sound kept.
