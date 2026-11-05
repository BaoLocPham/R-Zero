#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: bash scripts/install-envs.sh [SBATCH_OPTIONS...]'
    echo 'On the login node, submits install-envs (2 CPUs, 32G, 4 hours by default).'
    echo 'Inside an allocation, installs directly. Configure cluster.local.sh first.'
    echo 'The current installer requires Ampere or newer; the V100 backend port is pending.'
    exit 0
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec bash "$SCRIPT_DIR/submit.sh" install-envs "$@"
fi
if (( $# )); then echo 'Pass sbatch options from the login node; inside an allocation use no arguments.' >&2; exit 2; fi
source "$SCRIPT_DIR/common.sh"
exec bash "$RZERO_ROOT/slurm/job.sbatch" install-envs
