#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../project-env.sh"
if [[ $# == 0 || "$1" == --help ]]; then
 echo 'Usage: bash scripts/submit.sh {install|preflight|reproduce|evaluate} [SBATCH_OPTIONS] -- [RUN_ARGUMENTS]'
 exit 0
fi
kind=$1; shift
case "$kind" in install|preflight|reproduce|evaluate) ;; *) echo "Unknown job: $kind" >&2; exit 2 ;; esac
options=(); args=()
while (( $# )); do
 if [[ "$1" == -- ]]; then shift; args=("$@"); break; fi
 options+=("$1"); shift
done
mkdir -p "$RZERO_ROOT/logs"
resources=(--parsable --job-name="rzero-$kind" --nodes=1 --ntasks=1
 --partition="$RZERO_PARTITION" --gres="$RZERO_GRES" --cpus-per-task="$RZERO_CPUS"
 --mem="$RZERO_MEM" --time="$RZERO_TIME" --chdir="$RZERO_ROOT"
 --output="$RZERO_ROOT/logs/%x-%j.out" --error="$RZERO_ROOT/logs/%x-%j.err")
for setting in CONSTRAINT ACCOUNT QOS; do
 name="RZERO_$setting"
 if [[ -n "${!name:-}" ]]; then resources+=("--${setting,,}=${!name}"); fi
done
exec sbatch "${resources[@]}" "${options[@]}" "$RZERO_ROOT/slurm/job.sbatch" "$kind" "${args[@]}"
