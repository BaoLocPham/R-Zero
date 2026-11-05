#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: bash scripts/training-smoke.sh [SBATCH_OPTIONS...] [-- DRIVER_ARGUMENTS...]'
    echo 'Submits one iteration with one questioner/solver update and two 100-question shards.'
    echo 'Use scripts/rohan.sh training-smoke or scripts/gondor.sh training-smoke for named clusters.'
    exit 0
fi
exec bash "$SCRIPT_DIR/submit.sh" training-smoke "$@"
