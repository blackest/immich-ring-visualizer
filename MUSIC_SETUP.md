# Setting Up YuE2 for Music

Ring Visualizer's Music task (style+lyrics-to-song, `music_engineNG.py`)
doesn't ship an audio model. It calls out to a separate, standalone install
-- an MLX port of YuE2 -- as a subprocess. This app never imports it or
talks to a running Phosphene process (see `PHOSPHENE_DECOUPLING_PLAN.md`).

**You don't need to read this file to get it installed.** When the Music
task can't find a working install, its rail shows an "Install now (~11 GB)"
button that runs everything below automatically (`music_installNG.py`) and
reports progress live. This doc is for anyone who'd rather run the steps by
hand, is on a headless machine, or wants to understand what the button does.

If this isn't installed, the rest of Ring Visualizer works fine -- you just
get a clear error (or the install banner) the moment you try to compose a
song, instead of the feature working.

---

## 0. Read this first: Apple Silicon Mac only

**This engine will not run on Windows, Linux, or an Intel Mac.** Like every
model this app talks to, it's an [MLX](https://github.com/ml-explore/mlx)
(Apple Silicon, Metal-only) port. The in-app Install button checks for this
and refuses on anything else with a plain explanation.

Needs a real chunk of unified memory to run at all -- if generation fails
with an out-of-memory error, that's the model, not a bug here.

---

## 1. Where it comes from

- **Code:** [vanch007/mlx-Yue](https://github.com/vanch007/mlx-Yue), pinned
  to commit `9253ed133343406947bde7b67d43c7a63fb39d99`.
- **Weights, both from Hugging Face at pinned revisions:**
  - Generator: `vanch007/mlx-Yue2-3B` @ `fa66d203dd56d7e033e05ee8b32269768984c4cc`
  - VAE decoder: `m-a-p/YuE2-Vae` @ `95535e72a97bc0f09b8ada125d26b4009428c0e8`

Unlike LTX, both are ordinary public Hugging Face repos -- no gated
upstream, no special mirror.

---

## 2. Manual install

Requires macOS 14+ on Apple Silicon and [`uv`](https://docs.astral.sh/uv/).
Budget **~11 GB** of disk. Note: **Python 3.12**, not 3.11 -- a real pin
conflict with LTX/H3's Python 3.11 lane (this engine needs `mlx==0.32.2`,
which those two can't run), which is exactly why it gets its own venv.

```bash
# 1. Clone and pin
git clone --no-checkout https://github.com/vanch007/mlx-Yue ~/yue2-mlx
cd ~/yue2-mlx
git fetch --force origin 9253ed133343406947bde7b67d43c7a63fb39d99
git checkout --force --detach FETCH_HEAD

# 2. Its own venv
uv venv --python 3.12 .venv

# 3. Frozen dependency sync
UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --frozen --no-dev --extra transcription --project .
```

Then fetch the weights (plain `hf_hub_download`, sha256-verified for the
VAE's `model.safetensors`):

```bash
mkdir -p ~/mlx_models/yue2
python3 -c "
from huggingface_hub import hf_hub_download
gen = ('vanch007/mlx-Yue2-3B', 'fa66d203dd56d7e033e05ee8b32269768984c4cc',
       ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'ar-8bit.safetensors', 'ar-bf16.safetensors',
        'config.json', 'licenses/SnakeBeta-NVIDIA-MIT.txt',
        'licenses/stable-audio-tools-MIT.txt', 'nar-bf16.safetensors', 'qwen.tiktoken'))
vae = ('m-a-p/YuE2-Vae', '95535e72a97bc0f09b8ada125d26b4009428c0e8',
       ('model.safetensors', 'config.json', 'weights_manifest.json',
        'LICENSE', 'THIRD_PARTY_NOTICES.md'))
import os
for repo, rev, files in (gen, vae):
    dest = os.path.expanduser('~/mlx_models/yue2/' + ('generator' if repo == gen[0] else 'vae'))
    for name in files:
        hf_hub_download(repo_id=repo, revision=rev, filename=name, local_dir=dest)
"
```

This lands:

```
~/mlx_models/yue2/
  generator/   <- vanch007/mlx-Yue2-3B
  vae/         <- m-a-p/YuE2-Vae
```

---

## 3. Where Ring Visualizer looks for it

`music_engineNG.py` searches, in order:

1. `$LTX_MUSIC_ROOT` (repo) / `$LTX_MUSIC_MODELS` (weights) if set
2. `~/yue2-mlx` and `~/AI/yue2-mlx`
3. Falls back to `~/yue2-mlx`

**Recommended:** set `LTX_MUSIC_ROOT`/`LTX_MUSIC_MODELS` explicitly rather
than relying on the fallback list.

```
<repo dir>/
  .venv/bin/lyra
<weights dir>/
  generator/
  vae/
```

---

## 4. Verify it's found

```bash
curl -s "http://localhost:5000/api/ng/music/status?precision=8bit" | python3 -m json.tool
```

Reports `binary_ok`/`model_ok`/`vae_ok` separately and `ready` once all
three are true.

---

## 5. Notes

- Ring Visualizer never imports MLX or the model in-process -- every render
  is a subprocess call into that venv's own `lyra` console script.
- Freshly installing via the in-app button? **Restart Ring Visualizer once
  it finishes** -- the install paths above are resolved once when the app
  starts, so a fresh install won't show as `ready` until the process
  restarts.
- The `lyra cover` feature (transcribing an existing track) fetches its own
  extra models (SheetSage2, MERT-v2-FullSong) to a separate cache on first
  use -- that one genuinely does need network access at that moment, unlike
  everything else here, which always runs offline once installed.
