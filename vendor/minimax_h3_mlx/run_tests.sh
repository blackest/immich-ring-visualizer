#!/usr/bin/env bash
# Run the engine's own logic tests (no model weights needed) with the Hailuo H3 venv's python:
#   vendor/minimax_h3_mlx/run_tests.sh
# H3_PYTHON overrides the interpreter. The parity tests of the original repo (they compare against
# diffusers/transformers reference weights) are not vendored.
set -u
cd "$(dirname "$0")"
PY="${H3_PYTHON:-/Volumes/AI/Pinkio/api/phosphene.git/minimax-h3-mlx/.venv/bin/python}"
fail=0
for t in tests/test_*.py; do
  printf '%-34s ' "$(basename "$t")"
  if PYTHONPATH=. "$PY" "$t" >/tmp/h3_vendor_test.out 2>&1; then echo ok; else echo FAIL; tail -5 /tmp/h3_vendor_test.out; fail=1; fi
done
exit $fail
