# Setting Up HiDream-O1 for Character Sheets

Ring Visualizer's character-sheet feature (the "settings panel" — turnaround
shots, extended pose datasets, custom prompts) doesn't ship an image model.
It calls out to a separate, standalone install: the **HiDream-O1 MLX lab**,
via `hidream_engine.py`, as a subprocess — the same lab Phosphene itself
uses. See `PHOSPHENE_DECOUPLING_PLAN.md` for why it's a subprocess call
rather than an import.

If this isn't installed, the rest of Ring Visualizer works fine — you just
get a clear error the moment you try to generate a sheet (see "Verify it's
found" below) instead of the feature working.

---

## 0. Read this first: Apple Silicon Mac only

**HiDream-O1 MLX will not run on Windows, Linux, or an Intel Mac.** The lab
is built on [MLX](https://github.com/ml-explore/mlx), Apple's own array
framework — it only runs on Apple Silicon (M1 or newer) via Metal, and its
own README says so explicitly: *"Requires macOS on Apple Silicon (M1 or
newer)."*

On any other OS, `/api/phosphene/status` will permanently report
`ready: false` — there's no separate flag or workaround, because the whole
engine this feature calls is Apple-Silicon-only. Running Ring Visualizer
itself (the Flask app) on Windows/Linux is fine for everything else the app
does; only the character-sheet feature needs this Mac-only dependency.

---

## 1. Where to get it

The MLX port (weights + generator scripts) is published on Hugging Face by
its author, Mrbizarro:

- **MLX port (what you actually want):** [mlx-community/HiDream-O1-Image-Dev-mlx-bf16](https://huggingface.co/mlx-community/HiDream-O1-Image-Dev-mlx-bf16)
- Base model card (PyTorch, not what this runs): [HiDream-ai/HiDream-O1-Image-Dev](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev)

**Easiest path if you already run [Phosphene](https://github.com/mrbizarro/phosphene):**
install [Pinokio](https://pinokio.computer), then install Phosphene through
it, and pick "HiDream-O1-Image-Dev BF16" in its Image Studio engine
dropdown. That gives you a working lab with no manual Python/venv setup —
Ring Visualizer's `hidream_engine.py` will find that same install (see
section 3).

---

## 2. Manual install

Requires macOS 14+ on Apple Silicon, Python 3.11, and
[`uv`](https://docs.astral.sh/uv/). The BF16 weights are **~17 GB**
(`model.safetensors`) plus ~73 MB of extra "diffusion head" weights this
fork adds on top of the base HF checkpoint — budget disk space
accordingly.

```bash
# Download the pre-converted MLX weights + generator code
hf download mlx-community/HiDream-O1-Image-Dev-mlx-bf16 --local-dir HIDREAM-O1-MLX-LAB-active
cd HIDREAM-O1-MLX-LAB-active

# Set up the venv
uv venv --python 3.11
uv pip install -r requirements.txt

# Sanity-check it works stand-alone before wiring up Ring Visualizer
.venv/bin/python scripts/hidream_o1/generate_hidream_o1_mlx.py \
  --model-path mlx_models/hidream-o1-dev-bf16 \
  --prompt "a red apple on a wooden table, photorealistic" \
  --output /tmp/hidream_test.png
```

(Converting the weights yourself from the upstream PyTorch checkpoint is
also documented in the lab's own `README.md`, but takes ~50 GB free disk
and isn't necessary — the pre-converted download above is what Ring
Visualizer and Phosphene both expect.)

---

## 3. Where Ring Visualizer looks for it

`hidream_engine.py` searches, in order, for the first directory that
exists:

1. `$HIDREAM_LAB_DIR` (if set)
2. `~/HIDREAM-O1-MLX-LAB-active`
3. `~/AI/HIDREAM-O1-MLX-LAB-active`
4. `~/AI/projects/HIDREAM-O1-MLX-LAB-active`
5. `/Volumes/AI/HIDREAM-O1-MLX-LAB-active`

If none of those exist, it falls back to `~/HIDREAM-O1-MLX-LAB-active`
(mainly so the resulting "not found" error points somewhere sensible).

**Recommended:** set `HIDREAM_LAB_DIR` explicitly rather than relying on
the fallback list — those extra candidates exist for this machine's
current layout and may not match yours:

```bash
export HIDREAM_LAB_DIR="/path/to/HIDREAM-O1-MLX-LAB-active"
```

Whichever directory it resolves to, it expects this layout inside it
(exactly what the Quick Start above produces):

```
<lab dir>/
  .venv/bin/python
  mlx_models/hidream-o1-dev-bf16/
    model.safetensors
    extras/custom_heads.safetensors
  scripts/hidream_o1/generate_hidream_o1_mlx.py
```

---

## 4. Verify it's found

```bash
curl -s http://localhost:5000/api/phosphene/status | python3 -m json.tool
```

This calls `hidream_engine.hidream_health()` directly (no subprocess
launched) and reports each piece separately:

- `lab_dir` — the directory it resolved from the search order above
- `python_ok` — `.venv/bin/python` exists and is executable
- `model_ok` — both `model.safetensors` and `extras/custom_heads.safetensors` are present
- `script_ok` — `generate_hidream_o1_mlx.py` exists
- `ready` — all three of the above are true; only then will sheet generation actually run

If `ready` is `false`, the response tells you exactly which of the three
checks failed and what path it looked at — fix that one thing rather than
re-checking everything.

---

## 5. Notes

- Ring Visualizer never imports MLX or the model in-process — every
  generation is a subprocess call into that venv's own Python, so an
  install/upgrade issue in the lab can't take down the rest of the app.
- Ring Visualizer and Phosphene can both point at the same lab install;
  just don't run a generation in both at the same time (GPU contention —
  see `PHOSPHENE_DECOUPLING_PLAN.md`).
- There's currently no non-MLX fallback engine wired up for Windows/Linux
  users. Adding one would mean a different backend entirely (e.g. a
  CUDA/PyTorch path), which is a separate project, not a config change.
