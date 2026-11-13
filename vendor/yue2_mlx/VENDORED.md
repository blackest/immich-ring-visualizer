# yue2-mlx (the `lyra` music engine), vendored for immichRingNG

`music_engineNG.py` runs this as a subprocess through its `lyra` command-line tool. It lives here so a Phosphene or Pinokio
update cannot change what this app runs. Only the model weights (`mlx_models/yue2`, `yue2-loras`) are still shared.

**Source:** https://github.com/vanch007/mlx-Yue at commit `9253ed1` (2026-09-15, "fix: refresh dependencies and correct Apple
Silicon runtime checks"). Taken from the Phosphene/Pinokio checkout of it, which had no local changes and was 0 commits ahead
of that upstream. Copied byte for byte on 2026-11-13 with `git archive`; **no local changes**.

**Licences:** code Apache-2.0 (`LICENSE`); the nested `vendor/yue` package carries its own `LICENSE`, `MODEL_LICENSE` (this
governs the YuE model weights, not this code), `THIRD_PARTY_NOTICES.md` and `licenses/` (Oobleck VAE and SnakeBeta, both MIT).
Keep all of those together with the code.

**What is here:** `src/lyra`, `vendor/yue` (the `yue2` package), `pyproject.toml`, `uv.lock`, `README.md`, `PORTING.md`, `LICENSE`.
Left out on purpose (not needed to run): `docs` (306 MB), `tests`, `tools`, `skills`, `reports`, `validation`, `oracle`, `examples`.

**Not in upstream, added by us:** `requirements.lock` (50 exact packages for Python 3.14.6), `build_venv.sh` and this file.

## The venv

`.venv/` here is git-ignored. Build it with `./build_venv.sh` (Python 3.14.6 from `requirements.lock`). It was Python 3.12.11
with `numpy==2.2.6` until 2026-11-13. The engine's own `pyproject.toml` still says `requires-python >=3.12,<3.13` and
`numpy==2.2.6`, so it cannot be pip-installed on 3.14: the script puts `src` and `vendor/yue/src` on the path, writes a minimal
`mlx_yue` package record (lyra looks its own version up) and makes the `lyra` / `mlx-yue` console scripts. The vendored
`pyproject.toml` stays byte-identical to upstream.

**Checked 2026-11-13, same request and seed, 3.12 against 3.14.6 + numpy 2.5.4:** every intermediate array (ABC tokens, prefix,
semantic tokens, latent) identical; audio equal to within 119 dB of the signal (largest sample difference 2.4e-7), for an
8 s instrumental piano clip (also through the app) and a 60 s vocal clip. The package pins that matter to the output are held
exactly: `mlx` 0.32.2, `mlx-lm` 0.31.3, `transformers` 5.10.4, `tokenizers` 0.22.2, `safetensors` 0.7.0. Do not float them: in a
test environment with the newest versions the planning pass (a language model writing an ABC score) produced a different score
for the piano request, so a different piece of music, while the vocal request happened to match.

## Two upstream commits we do not have

`ef652df` RealAudio Tokenizer v4 and `c5b5a81` general community-LoRA support (`--lora`, `--lora-scale`). Neither is used by
this app today.
