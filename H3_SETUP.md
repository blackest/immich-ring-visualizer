# Setting Up MiniMax-H3 for the H3 task

Ring Visualizer's H3 task (prompt/image-to-video with synced dialogue and
sound, `h3_engineNG.py`) doesn't ship a video model. It calls out to a
separate, standalone install -- an MLX port of MiniMax's H3 (FL2VA) model --
as a subprocess. This app never imports it or talks to a running Phosphene
process (see `PHOSPHENE_DECOUPLING_PLAN.md`).

**H3 is a deliberate opt-in, unlike LTX and Music.** It's ~75 GB, needs a
36 GB+ Mac even in its reduced-memory form, and ships under MiniMax's own
Community License (territory restrictions apply -- read it before
installing: see the license file bundled in the `MiniMaxAI/MiniMax-H3`
Hugging Face repo, the upstream model this port is quantized from). The
in-app "Install now" button only appears
after you tick a checkbox acknowledging this; there's no bundled or
default-on path to it.

**You don't need to read this file to get it installed.** The H3 task's
rail shows the license notice + Install button when eligible, and it runs
everything below automatically (`h3_installNG.py`) with live progress. This
doc is for anyone who'd rather run the steps by hand or wants to understand
what the button does.

---

## 0. Read this first: Apple Silicon Mac only, 36 GB+ RAM

**This engine will not run on Windows, Linux, or an Intel Mac** (MLX,
Metal-only, Apple Silicon only). It also needs **at least 36 GB of unified
memory** -- even the reduced-RAM "Q8" engine this app defaults to peaks
around 25.6 GiB while rendering; below 36 GB it swaps and a few seconds of
video takes hours. Both checks run before the in-app button starts
anything.

---

## 1. Where it comes from

- **Code:** [mrbizarro/minimax-h3-mlx](https://github.com/mrbizarro/minimax-h3-mlx),
  branch `codex/h3-engine-v2`.
- **Weights, straight from Hugging Face** (via the cloned repo's own
  `scripts/download_selected.py` -- not anything this app or Phosphene
  re-hosts):
  - `DeepBeepMeep/MiniMax-H3` -- the pruned bf16 DiT (the base model)
  - `ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit` -- the compact text
    encoder/VAE pack
  - `MiniMaxAI/MiniMax-H3` -- a couple of small upstream metadata/config files

There's an extra step past the download, easy to miss: this app's default
model ("h3q8", the reduced-memory engine) needs a **locally quantized DiT**
that nothing downloads pre-built -- see step 2 below.

---

## 2. Manual install

Requires macOS 14+ on Apple Silicon, 36 GB+ RAM, and
[`uv`](https://docs.astral.sh/uv/). Budget **~75 GB** of disk (the bf16
download plus ~22 GB more for the locally-built Q8 pack).

```bash
# 1. Clone and pin to the branch this app is validated against
git clone --branch codex/h3-engine-v2 \
  https://github.com/mrbizarro/minimax-h3-mlx.git ~/minimax-h3-mlx
cd ~/minimax-h3-mlx

# 2. Its own venv (separate from LTX -- the two disagree on which MLX
#    version to run)
uv venv --python 3.11 --seed .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 3. Download the weights -- appends "models/" to --root itself
.venv/bin/python scripts/download_selected.py --root ~/mlx_models/hailuo-h3

# 4. Build the reduced-RAM Q8 engine locally from the downloaded bf16 DiT
#    (~5 min, one time; this is the step step 1 above alone won't give you)
.venv/bin/python scripts/quantize_stream.py \
  --src ~/mlx_models/hailuo-h3/models/deepbeep-pruned-bf16/MiniMax-H3-FL2VA-pruned_bf16.safetensors \
  --out ~/mlx_models/hailuo-h3/models/h3-dit-q8

# 5. Validate it in a FRESH process before trusting it -- the quantizer's
#    own process can hold the bf16 source AND the Q8 output at once and get
#    killed by macOS right after writing every shard correctly, on a 36 GB
#    Mac. A clean load in a separate process is the only real proof it works.
.venv/bin/python -c "
from minimax_h3_mlx.load import load_dit
load_dit('$HOME/mlx_models/hailuo-h3/models/h3-dit-q8', verbose=False)
print('LOAD OK')
"
# Only once that prints LOAD OK:
touch ~/mlx_models/hailuo-h3/models/h3-dit-q8/.built_ok
```

This lands:

```
~/mlx_models/hailuo-h3/models/
  deepbeep-pruned-bf16/    <- the downloaded bf16 DiT
  ddalcu-q8/               <- compact text encoder + VAEs
  upstream-meta/           <- small metadata files
  h3-dit-q8/               <- the LOCALLY BUILT Q8 engine (step 4-5 above)
```

---

## 3. Where Ring Visualizer looks for it

`h3_engineNG.py` searches, in order:

1. `$LTX_H3_ROOT` (repo) / `$LTX_H3_MODELS` (weights, the `models/`
   directory itself) if set
2. `~/minimax-h3-mlx` and `~/AI/minimax-h3-mlx`
3. Falls back to `~/minimax-h3-mlx`

**Recommended:** set `LTX_H3_ROOT`/`LTX_H3_MODELS` explicitly rather than
relying on the fallback list.

```
<repo dir>/
  .venv/bin/python
  scripts/generate_staged.py
<weights dir>/          <- this IS the .../models/ directory, see above
  h3-dit-q8/
  ddalcu-q8/
  upstream-meta/FL2VA/text_encoder/config.json
```

---

## 4. Verify it's found

```bash
curl -s "http://localhost:5000/api/ng/h3/status?model=h3q8" | python3 -m json.tool
```

Reports `python_ok`/`runner_ok`/`dit_ok`/`compact_ok`/`text_config_ok`
separately and `ready` once all are true.

---

## 5. Notes

- Ring Visualizer never imports MLX or the model in-process -- every render
  is a subprocess call into that venv's own Python.
- Freshly installing via the in-app button? **Restart Ring Visualizer once
  it finishes** -- the install paths above are resolved once when the app
  starts, so a fresh install won't show as `ready` until the process
  restarts.
- The full bf16 engine (needs 60 GB+ RAM) isn't built by the steps above --
  only the Q8 engine this app defaults to. The bf16 weights ARE downloaded
  as part of step 3 (they're what the Q8 build quantizes from), so nothing
  extra needs fetching if you later want to wire up the larger tier.
