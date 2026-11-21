#!/bin/sh
# Ours, not upstream: builds this folder's .venv (Python 3.12) from requirements.lock. The lock is the complete, exact list,
# so it is installed with --no-deps. Python 3.12, not 3.14: coremltools has no native Core ML library for 3.14 (the model
# would not load on the Neural Engine).
set -eu
cd "$(dirname "$0")"
rm -rf .venv
uv venv --python 3.12 --seed .venv
uv pip install --python .venv/bin/python --no-deps -r requirements.lock
echo "built $PWD/.venv ($(.venv/bin/python --version))"
