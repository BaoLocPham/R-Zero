#!/usr/bin/env bash
# Accept driver arguments; customize resources through submit.sh or cluster.sh.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: bash scripts/14-reproduce.sh [--model MODEL] [--name NAME] [--iterations N] [DRIVER_ARGUMENTS...]'
    echo 'Submits on the login node, or runs directly inside an allocation.'
    echo 'Resume completed iterations using --start-iteration N; Agent0 --resume is not supported.'
    exit 0
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then exec bash "$SCRIPT_DIR/submit.sh" reproduce -- "$@"; fi
source "$SCRIPT_DIR/common.sh"
exec bash "$RZERO_ROOT/slurm/job.sbatch" reproduce "$@"
