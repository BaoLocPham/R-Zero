#!/usr/bin/env bash
# Compatibility entry point for the original two positional arguments.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (( $# != 2 )); then
 echo 'Usage: bash scripts/main.sh BASE_MODEL NAME' >&2
 echo 'For configurable runs: bash scripts/submit.sh reproduce -- --help' >&2
 exit 2
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
 export RZERO_ROOT="$root"
 exec bash "$root/slurm/job.sbatch" reproduce --model "$1" --name "$2"
fi
exec bash "$root/scripts/submit.sh" reproduce -- --model "$1" --name "$2"
