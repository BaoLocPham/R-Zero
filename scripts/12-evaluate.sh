#!/usr/bin/env bash
# Accept driver arguments; customize resources through submit.sh or cluster.sh.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: bash scripts/12-evaluate.sh [--model MODEL]'
    echo 'Submits on the login node, or runs directly inside an allocation.'
    exit 0
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then exec bash "$SCRIPT_DIR/submit.sh" evaluate -- "$@"; fi
source "$SCRIPT_DIR/common.sh"
exec bash "$RZERO_ROOT/slurm/job.sbatch" evaluate "$@"
