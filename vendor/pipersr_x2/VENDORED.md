# pipersr_x2: 2x sharpen of a finished clip or a still picture on the Apple Neural Engine (ours)

`pipersr_x2_run.py` is **our own** small runner; nothing of upstream is copied. It runs as a subprocess of
`h3_sharpNG.py` (H3 films) and `klein_sharpNG.py` (Klein pictures, written beside the original as `<name>_sr2x.png`) on this folder's own env (`.venv`, Python 3.12, built by `build_venv.sh` from `requirements.lock`), so the app
never imports PiperSR. Python 3.12 because `coremltools` has no native Core ML library for 3.14.

**Model:** PiperSR 1.0.0 by Ben Racicot (https://modelpiper.com, https://github.com/ModelPiper/PiperSR), a small image
super-resolution model for the Neural Engine, 128x128 tiles in and 256x256 out, bundled in the `pipersr` wheel.

**Licence:** the `pipersr` code is AGPL-3.0; the model weights are free for personal and academic use with attribution, and
commercial use needs a separate licence from the author (contact in the package metadata). I have not seen the full
`MODEL_LICENSE` text, only this summary from the package README. Attribution: "Powered by PiperSR from ModelPiper -
https://modelpiper.com" (printed by the runner and shown on the Sharp 2x button).

**Halo limit (always on, chosen by eye 2026-11-21):** PiperSR's raw output roughly doubles the edge contrast but overshoots it
(halos) and brings out grain in dark areas. Each output pixel is held between the darkest and brightest value of a plain
Lanczos 2x resize within a 5x5 window (`limit_halo`; a separable numpy running min/max, 0.24 s for a 3072x2048 picture, the
same numbers as PIL's rank filters at 4.5 s). On a dark AI picture that removed about 88% of the overshoot and 40% of the
grain and kept about 60% of the edge gain; against the plain resize, the 50/50 blend and the raw output, the user judged it
the best, the fairy's wings clearly sharper. `--strength` (0 to 1, default 1) then blends the result with the plain resize.
Klein saves each strength as its own file, `<name>_sr2x<percent>.png` (`_sr2x100.png` by default).

**What it does and does not do:** it sharpens edges and keeps the picture; it does not redraw it (LTX's upscaler does that,
see `upscale_x2NG.py`), so blocks from a low-resolution render stay. Frames are processed one by one, so shimmer between
frames is possible (not yet checked on moving footage). Tiles overlap by 16 px on each side and only each tile's centre is
kept, so there are no grid seams. Measured 2026-11-21 on an M2 Max: 124 frames at 640x384 -> 1280x768 in 23 s with the halo limit (18 s
without), sound kept.

Stills: a 1024x576 Klein picture became 2048x1152 in 0.8 s (measured 2026-11-21); the edges of a water droplet and its ripples were
visibly cleaner than a plain Lanczos resize of the same picture, nothing else changed.
