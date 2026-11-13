#!/bin/sh
# Ours, not upstream: builds this folder's .venv on Python 3.14 from requirements.lock.
# The engine's own pyproject says requires-python <3.13, so it cannot be pip-installed on 3.14; instead the two source
# folders go on the path and a minimal package record is written (lyra looks its own version up), plus the console scripts.
set -eu
cd "$(dirname "$0")"
rm -rf .venv
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements.lock
SP=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
printf '%s\n%s\n' "$PWD/src" "$PWD/vendor/yue/src" > "$SP/vendored_yue2.pth"
mkdir -p "$SP/mlx_yue-0.1.0.dist-info"
printf 'Metadata-Version: 2.1\nName: mlx-yue\nVersion: 0.1.0\n' > "$SP/mlx_yue-0.1.0.dist-info/METADATA"
for n in lyra mlx-yue; do
  printf '#!%s\nimport sys\nfrom lyra.cli import main\nsys.exit(main())\n' "$PWD/.venv/bin/python" > ".venv/bin/$n"
  chmod +x ".venv/bin/$n"
done
echo "built $PWD/.venv ($(.venv/bin/python --version))"
