#!/bin/sh
# Ours, not upstream: builds this folder's `env` (Python 3.11) from requirements.lock and installs the three packages
# editable from ./packages, so the code that runs is the code in this repository (see VENDORED.md).
set -eu
# requirements.lock is the complete, exact list (Phosphene installed some of these with --no-deps), so install it as is.
cd "$(dirname "$0")"
rm -rf env
uv venv --python 3.11 --seed env
uv pip install --python env/bin/python --no-deps -r requirements.lock
uv pip install --python env/bin/python --no-deps --build-constraints build-constraints.txt -e packages/ltx-core-mlx -e packages/ltx-pipelines-mlx -e packages/ltx-trainer
echo "built $PWD/env ($(env/bin/python --version))"
