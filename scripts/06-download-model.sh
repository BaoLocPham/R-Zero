#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: bash scripts/06-download-model.sh [--model MODEL_ID] [--revision REVISION]'
    echo 'Downloads Qwen/Qwen3-4B-Base by default using the R-Zero environment; no GPU allocation is needed.'
    echo 'Records the resolved snapshot path/revision in locks/base-model.json.'
    echo 'Use that snapshot path as --model when reproducing a specific revision.'
    exit 0
fi
source "$SCRIPT_DIR/common.sh"
python_bin="$RZERO_ENV/bin/python"
if [[ ! -x "$python_bin" ]]; then echo "Missing $python_bin; finish install-envs first." >&2; exit 1; fi
need flock
mkdir -p "$RZERO_ROOT/logs" "$RZERO_ROOT/locks" "$HF_HOME"
exec 9>"$RZERO_ROOT/locks/.model-download.lock"
if ! flock -n 9; then echo 'A model download is already running for this checkout.' >&2; exit 1; fi
log="$RZERO_ROOT/logs/download-model-$(date +%Y%m%d-%H%M%S)-$$.log"
"$python_bin" "$SCRIPT_DIR/download-model.py" --model "${RZERO_BASE_MODEL:-Qwen/Qwen3-4B-Base}" \
    --record "$RZERO_ROOT/locks/base-model.json" "$@" 2>&1 | tee "$log"
