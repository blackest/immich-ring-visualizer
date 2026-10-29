#!/bin/bash
# Start OpenH3-IR (`h3ir serve`) against one of two language models:
#
#   h3ir_serveNG.sh gemma   Gemma 4 26B  (the :8080 launchd server) -> compiler on :8700
#   h3ir_serveNG.sh qwen    Qwen3.8-27B  (mlx_vlm.server, :8082)    -> compiler on :8701
#
# Gemma is the everyday one. Qwen is for briefs with a video attached: Gemma's description of a
# real clip came back in a shape the compiler crashed on, and the 8B Qwen could not write a brief
# that passed its checks; the 27B did. (The compiler sends the model three still frames of a clip,
# not the video itself, so this is instruction-following, not video ability.)
# routes/enhancedh3NG.py sends a brief to :8701 when any attached asset is a video.
#
# Qwen's server is started here only if :8082 is not already answering, and is left running when
# the compiler stops: kill it (pkill -f "mlx_vlm.server.*--port 8082") before a render that needs
# the memory (~15 GB).

set -euo pipefail

H3IR="${RINGVIZ_H3IR_BIN:-/Volumes/AI/ComfyUI/venv/bin/h3ir}"
PYTHON="${RINGVIZ_GEMMA_PYTHON:-$HOME/mlxenv/bin/python}"

case "${1:-}" in
    gemma)
        LLM_PORT=8080
        LLM_MODEL="/Users/john/models/gemma4-26b-mlx"
        H3IR_PORT=8700
        ;;
    qwen)
        LLM_PORT=8082
        LLM_MODEL="/Volumes/AI/models/Qwen3.8-27B-4bit"
        H3IR_PORT=8701
        ;;
    *)
        echo "usage: $0 gemma|qwen" >&2
        exit 2
        ;;
esac

[ -x "$H3IR" ] || { echo "h3ir not found at $H3IR" >&2; exit 1; }
[ -d "$LLM_MODEL" ] || { echo "model not found at $LLM_MODEL (is /Volumes/AI mounted?)" >&2; exit 1; }

if ! curl -s -m 3 -o /dev/null "http://127.0.0.1:$LLM_PORT/v1/models"; then
    if [ "$1" = gemma ]; then
        echo "nothing answering on :$LLM_PORT -- the Gemma server is the com.john.mlx-server launchd job" >&2
        exit 1
    fi
    echo "Starting $LLM_MODEL on :$LLM_PORT"
    nohup "$PYTHON" -m mlx_vlm.server --model "$LLM_MODEL" --host 127.0.0.1 --port "$LLM_PORT" \
        > "${TMPDIR:-/tmp}/h3ir-qwen-llm.log" 2>&1 &
    for _ in $(seq 1 60); do
        curl -s -m 3 -o /dev/null "http://127.0.0.1:$LLM_PORT/v1/models" && break
        sleep 2
    done
fi

echo "h3ir on :$H3IR_PORT using $LLM_MODEL"
export H3IR_LLM_URL="http://127.0.0.1:$LLM_PORT/v1"
export H3IR_LLM_MODEL="$LLM_MODEL"
exec "$H3IR" serve --host 127.0.0.1 --port "$H3IR_PORT"
