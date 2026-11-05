#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: bash scripts/02-allocate-gpus.sh [SRUN_OPTIONS...]'
    echo 'Allocates the configured GPUs with 2 CPUs, 32G and 2 hours by default.'
    echo 'For named clusters: bash scripts/rohan.sh allocate (or gondor.sh).'
    exit 0
fi
source "$SCRIPT_DIR/common.sh"
need srun
options=(--partition="$RZERO_PARTITION" --nodes=1 --ntasks=1 --gres="$RZERO_GRES"
    --cpus-per-task="${RZERO_ALLOCATE_CPUS:-2}" --mem="${RZERO_ALLOCATE_MEM:-32G}"
    --time="${RZERO_ALLOCATE_TIME:-02:00:00}" --chdir="$RZERO_ROOT")
for setting in CONSTRAINT ACCOUNT QOS; do
    variable="RZERO_$setting"
    if [[ -n "${!variable:-}" ]]; then options+=("--${setting,,}=${!variable}"); fi
done
if [[ -n "${RZERO_MAIL_USER:-}" ]]; then options+=(--mail-user="$RZERO_MAIL_USER" --mail-type=ALL); fi
exec srun "${options[@]}" "$@" --pty bash -l
