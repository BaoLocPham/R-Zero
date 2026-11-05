#!/usr/bin/env bash
# Agent0-style submission: resource options before --, driver arguments after it.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# == 0 || "$1" == --help || "$1" == -h ]]; then
    echo 'Usage: bash scripts/submit.sh KIND [SBATCH_OPTIONS...] [-- RUN_ARGUMENTS...]'
    echo 'Kinds: install-envs (alias: install), preflight, training-smoke, reproduce, evaluate'
    echo 'Example: bash scripts/rohan.sh install-envs --test-only --time=04:00:00'
    echo 'Resume uses -- --start-iteration N and requires a completed preceding iteration.'
    echo 'The current training backend still requires Ampere or newer; the Rohan V100 port is pending.'
    exit 0
fi
source "$SCRIPT_DIR/common.sh"
kind=$1; shift
if [[ "$kind" == install ]]; then kind=install-envs; fi
# Installation and kernel checks do not start Ray training workers.
case "$kind" in
    install-envs)
        cpus=${RZERO_INSTALL_CPUS:-2}; memory=${RZERO_INSTALL_MEM:-32G}
        duration=${RZERO_INSTALL_TIME:-04:00:00} ;;
    preflight)
        cpus=${RZERO_PREFLIGHT_CPUS:-2}; memory=${RZERO_PREFLIGHT_MEM:-8G}
        duration=${RZERO_PREFLIGHT_TIME:-00:10:00} ;;
    training-smoke)
        cpus=$RZERO_CPUS; memory=$RZERO_MEM; duration=01:00:00 ;;
    reproduce|evaluate)
        cpus=$RZERO_CPUS; memory=$RZERO_MEM; duration=$RZERO_TIME ;;
    *) echo "Unknown job kind: $kind" >&2; exit 2 ;;
esac
options=(); arguments=()
test_only=false
while (( $# )); do
    if [[ "$1" == -- ]]; then shift; arguments=("$@"); break; fi
    case "$1" in
        --test-only|--test-only=*) test_only=true ;;
        --continue-after|--continue-after=*)
            echo 'R-Zero cannot resume incomplete iterations like Agent0. Use -- --start-iteration N after a completed iteration.' >&2
            exit 2 ;;
    esac
    options+=("$1"); shift
done
for argument in "${arguments[@]}"; do
    case "$argument" in
        --resume|--resume=*|--run-dir|--run-dir=*)
            echo 'R-Zero does not support Agent0 --resume/--run-dir. Use --name and --start-iteration N after a completed iteration.' >&2
            exit 2 ;;
    esac
done
need sbatch
mkdir -p "$RZERO_ROOT/logs"
name="rzero-$kind"
resources=(--parsable --job-name="$name" --nodes=1 --ntasks=1
    --partition="$RZERO_PARTITION" --gres="$RZERO_GRES" --cpus-per-task="$cpus"
    --mem="$memory" --time="$duration" --chdir="$RZERO_ROOT"
    --output="$RZERO_ROOT/logs/%x-%j.out" --error="$RZERO_ROOT/logs/%x-%j.err")
for setting in CONSTRAINT ACCOUNT QOS; do
    variable="RZERO_$setting"
    if [[ -n "${!variable:-}" ]]; then resources+=("--${setting,,}=${!variable}"); fi
done
if [[ -n "${RZERO_MAIL_USER:-}" ]]; then
    resources+=(--mail-user="$RZERO_MAIL_USER" --mail-type=ALL)
fi
if submitted=$(sbatch "${resources[@]}" "${options[@]}" "$RZERO_ROOT/slurm/job.sbatch" "$kind" "${arguments[@]}"); then
    :
else
    status=$?
    echo "Submission settings: partition=$RZERO_PARTITION gres=$RZERO_GRES cpus=$cpus mem=$memory time=$duration (before command-line overrides)" >&2
    echo 'Check cluster.local.sh, your current allocations and QoS limits. No job ID was saved.' >&2
    exit "$status"
fi
if [[ "$test_only" == true ]]; then
    if [[ -n "$submitted" ]]; then printf '%s\n' "$submitted"; fi
    echo 'Slurm submission check passed. No job was submitted or saved; this does not validate the training backend.' >&2
    exit 0
fi
job_id=${submitted%%;*}
if [[ ! "$job_id" =~ ^[1-9][0-9]*$ ]]; then
    echo "Unrecognized sbatch result: $submitted; inspect the queue. No job ID was saved." >&2
    exit 1
fi
record="$RZERO_ROOT/logs/$kind-job-id.txt"
temporary=$(mktemp "$record.XXXXXX")
printf '%s\n' "$job_id" > "$temporary"
mv -- "$temporary" "$record"
# Preserve the old install spelling for users/scripts already using it.
if [[ "$kind" == install-envs ]]; then
    printf '%s\n' "$job_id" > "$RZERO_ROOT/logs/install-job-id.txt"
fi
printf '%s\n' "$job_id"
echo "Submitted $name job $job_id. Logs: $RZERO_ROOT/logs/$name-$job_id.{out,err}" >&2
