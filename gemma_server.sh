#!/bin/bash
# Standalone mlx_vlm.server for the enhance/chat Gemma-3 checkpoint --
# lets Hermes (or anything else OpenAI-compatible) call Gemma directly,
# and lets ltx_engineNG.py's chat helpers skip the per-call subprocess
# reload when this is running (falls back to the subprocess path
# automatically if it's not).
#
# Same venv and path-resolution convention as ltx_engineNG.py:
# LTX_MLX_DIR (repo) / LTX_MLX_MODELS_DIR (weights root), with the same
# conventional fallback locations.
#
# Run this, then before an LTX render acquires the GPU you'd want to
# POST /unload to this server first (not wired up yet -- see
# ltx_engineNG.py's _LTX_SUBPROCESS_LOCK for where that hook belongs).

set -euo pipefail

PORT="${RINGVIZ_GEMMA_SERVER_PORT:-8811}"
# 0.0.0.0 so LAN peers (e.g. the Hermes/Rachel gateway) can reach this
# directly, not just processes on this machine.
HOST="${RINGVIZ_GEMMA_SERVER_HOST:-0.0.0.0}"

resolve_repo_dir() {
    if [ -n "${LTX_MLX_DIR:-}" ]; then
        echo "$LTX_MLX_DIR"
        return
    fi
    for cand in "$HOME/ltx-2-mlx" "$HOME/AI/ltx-2-mlx" "/Volumes/AI/Pinkio/api/phosphene.git/ltx-2-mlx"; do
        if [ -d "$cand" ]; then
            echo "$cand"
            return
        fi
    done
    echo "$HOME/ltx-2-mlx"
}

resolve_weights_root() {
    if [ -n "${LTX_MLX_MODELS_DIR:-}" ]; then
        echo "$LTX_MLX_MODELS_DIR"
        return
    fi
    cand="/Volumes/AI/Pinkio/drive/drives/peers/d1785024377531/mlx_models"
    if [ -d "$cand" ]; then
        echo "$cand"
        return
    fi
    echo "$HOME/mlx_models"
}

REPO_DIR="$(resolve_repo_dir)"
WEIGHTS_ROOT="$(resolve_weights_root)"
PYTHON="$REPO_DIR/env/bin/python"
GEMMA="$WEIGHTS_ROOT/gemma-3-12b-it-4bit"

if [ ! -x "$PYTHON" ]; then
    echo "ltx-2-mlx venv python not found at $PYTHON" >&2
    exit 1
fi
if [ ! -d "$GEMMA" ]; then
    echo "Gemma checkpoint not found at $GEMMA" >&2
    exit 1
fi

echo "Starting Gemma server: $GEMMA on $HOST:$PORT"
exec "$PYTHON" -m mlx_vlm.server --model "$GEMMA" --host "$HOST" --port "$PORT"
