# Setting Up LTX-2.5 for Animate

Ring Visualizer's "Animate" task (image-to-video, `ltx_engineNG.py`) doesn't
ship a video model. It calls out to a separate, standalone install: the
**ltx-2-mlx MLX lab**, as a subprocess -- the same lab Phosphene itself uses,
but this app never imports it or talks to a running Phosphene process (see
`PHOSPHENE_DECOUPLING_PLAN.md`).

**You don't need to read this file to get it installed.** When the Animate
task can't find a working install, its rail shows an "Install now (~66 GB)"
button that runs everything below automatically (`ltx_installNG.py`) and
reports progress live. This doc is for anyone who'd rather run the steps by
hand, is on a headless machine, or wants to understand what the button does.

If this isn't installed, the rest of Ring Visualizer works fine -- you just
get a clear error (or the install banner) the moment you try to generate a
video, instead of the feature working.

---

## 0. Read this first: Apple Silicon Mac only

**ltx-2-mlx will not run on Windows, Linux, or an Intel Mac.** It's built on
[MLX](https://github.com/ml-explore/mlx), Apple's own array framework --
Metal-only, Apple Silicon only. The in-app Install button checks for this
and refuses on anything else with a plain explanation; there's no
workaround, because the whole engine is Apple-Silicon-only.

---

## 1. Where it comes from

- **Code:** [mrbizarro/ltx-2-mlx](https://github.com/mrbizarro/ltx-2-mlx),
  pinned to tag `v0.14.19+ltx25.7` (a maintained fork carrying the LTX-2.5
  port -- see `ltx_installNG.py`'s own comments for what that fork adds over
  the base project).
- **Weights:** LTX-2.5's actual base model is gated on Hugging Face, and
  this fork's maintainer publishes their own quantization of it as **GitHub
  release assets** on a release of the public
  [mrbizarro/Phosphene](https://github.com/mrbizarro/Phosphene) repo (tag
  `weights-ltx25-v1`) -- a normal release-binary download, not a dependency
  on Phosphene the running app. There's no other public mirror of this
  specific quantization.

---

## 2. Manual install

Requires macOS 14+ on Apple Silicon and [`uv`](https://docs.astral.sh/uv/).
Budget **~66 GB** of disk (the base engine + text encoder; Ring Visualizer
always renders at LTX's "high" quality tier, fixed, no smaller option).

```bash
# 1. Clone the fork and pin to the tested tag
git clone https://github.com/mrbizarro/ltx-2-mlx.git ~/ltx-2-mlx
cd ~/ltx-2-mlx
git checkout v0.14.19+ltx25.7

# 2. Build the venv (Python 3.11 -- a real pin, not "whatever's newest")
uv venv --python 3.11 --seed env

# 3. Install pinned deps. mlx==0.31.1 specifically -- 0.31.2 measurably
#    regresses LTX's own audio output by ~22 dB.
uv pip install --python env/bin/python \
  'mlx==0.31.1' 'mlx-lm==0.31.1' 'mlx-metal==0.31.1' 'transformers>=5.0.0,<5.13.0'
uv pip install --python env/bin/python \
  --build-constraints pip-build-constraints.txt \
  ./packages/ltx-core-mlx ./packages/ltx-pipelines-mlx ./packages/ltx-trainer
# Reinstall editable, no-deps -- avoids a uv workspace quirk that leaves the
# checkout looking "dirty" against its own tracked source on a fresh install.
uv pip install --python env/bin/python --reinstall --no-deps \
  --build-constraints pip-build-constraints.txt \
  ./packages/ltx-core-mlx ./packages/ltx-pipelines-mlx ./packages/ltx-trainer
# mlx-vlm -- needed for Animate's "Chat with Gemma"/"Enhance" features
# (ltx_vision_chat_helperNG.py and siblings run inside this same venv).
uv pip install --python env/bin/python --no-deps 'mlx-vlm==0.4.4'
```

Then fetch the weights (sharded GitHub release assets, sha256-verified,
resumable -- see `ltx_pack_fetchNG.py` for the exact mechanics, ported from
Phosphene's own `fetch_pack_release.py`):

```bash
mkdir -p ~/mlx_models
python3 - <<'PY'
import sys
sys.path.insert(0, ".")  # run from the immichring repo root
import ltx_pack_fetchNG as f
from pathlib import Path
root = Path.home() / "mlx_models"
for key in ("q8_25", "hq_25", "gemma4_25"):
    f.fetch_pack(key, root, print)
PY
```

This lands:

```
~/mlx_models/
  ltx-2.5-mlx-q8/          <- q8_25 + hq_25 merged (base engine + full-quality DiT)
  gemma4-12b-ltx25-q4/     <- gemma4_25 (the paired text encoder)
```

---

## 3. Where Ring Visualizer looks for it

`ltx_engineNG.py` searches, in order:

1. `$LTX_MLX_DIR` (repo) / `$LTX_MLX_MODELS_DIR` (weights) if set
2. `~/ltx-2-mlx` and `~/AI/ltx-2-mlx`
3. Falls back to `~/ltx-2-mlx` (so a "not found" error points somewhere sensible)

**Recommended:** set `LTX_MLX_DIR`/`LTX_MLX_MODELS_DIR` explicitly rather
than relying on the fallback list.

```
<repo dir>/
  env/bin/python
  env/bin/ltx-2-mlx
<weights dir>/
  ltx-2.5-mlx-q8/
  gemma4-12b-ltx25-q4/
```

---

## 4. Verify it's found

```bash
curl -s http://localhost:5000/api/ng/videogen/status | python3 -m json.tool
```

Reports `binary_ok`/`model_ok`/`gemma_ok` separately and `ready` once all
three are true. `enhance_gemma_ok` gates only the optional Enhance/chat
features, not rendering itself, and isn't installed by the steps above --
that's a separate, plain chat-tuned Gemma-3 checkpoint; see
`ltx_engineNG.py`'s `LTX_DEFAULT_ENHANCE_GEMMA` if you want it.

---

## 5. Notes

- Ring Visualizer never imports MLX or the model in-process -- every render
  is a subprocess call into that venv's own Python.
- Freshly installing via the in-app button? **Restart Ring Visualizer once
  it finishes** -- the install paths above are resolved once when the app
  starts, so a fresh install won't show as `ready` until the process
  restarts.
- Ring Visualizer and Phosphene can share the same install; just don't
  render in both at the same time (GPU contention).
- Not fetched by either path above: Phosphene's live-preview decoder (`tae`)
  and its Remix/Upscale-x2 adapter (`ic_upscale_x2`) -- neither feature is
  wired up in Ring Visualizer, so there's nothing here that needs them.
