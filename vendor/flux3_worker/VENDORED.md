# The FLUX 3 / Klein worker env, owned by immichRingNG

Klein (the pane that makes pictures with FLUX.2 klein) runs `klein_workerNG.py` (this repository's own script) on this env's
interpreter. There is no Phosphene code in it: the script imports `mflux`, `mlx` and Pillow from here and reads the weights
from the Hugging Face cache. Only the **env** used to live inside Pinokio.

**What is here:** `requirements.lock` (57 exact packages, Python 3.12: `mflux` 0.23.0, `mlx` 0.32.4, `torch` 2.13.0,
`transformers` 5.19.0 ...) and `build_env.sh`, which builds the git-ignored `env/` from it with `--no-deps` (the lock is the
complete list). `klein_engineNG.worker_python()` prefers `vendor/flux3_worker/env` once it exists, then Phosphene's install.

**Checked 2026-11-13:** the Pinokio env's 57 packages were diffed file by file against a clean build of the same pins and
are identical (no patches). The weights and the Hugging Face cache are still read from Pinokio.
