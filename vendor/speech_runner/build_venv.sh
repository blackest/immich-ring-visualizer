#!/bin/sh
# Ours, not upstream: builds this folder's .venv (Python 3.12) from requirements.lock. The lock is the complete, exact list,
# so it is installed with --no-deps. The runner scripts in this folder run on this interpreter.
set -eu
cd "$(dirname "$0")"
rm -rf .venv
uv venv --python 3.12 --seed .venv
uv pip install --python .venv/bin/python --no-deps -r requirements.lock
echo "built $PWD/.venv ($(.venv/bin/python --version))"
