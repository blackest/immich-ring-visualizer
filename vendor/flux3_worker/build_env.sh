#!/bin/sh
# Ours, not upstream: builds this folder's `env` (Python 3.14) from requirements.lock. The lock is the complete, exact
# list, so it is installed with --no-deps. The worker script itself is ../../klein_workerNG.py.
set -eu
cd "$(dirname "$0")"
rm -rf env
uv venv --python 3.14 --seed env
uv pip install --python env/bin/python --no-deps -r requirements.lock
echo "built $PWD/env ($(env/bin/python --version))"
