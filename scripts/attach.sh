#!/usr/bin/env bash
# Open a shell or GPU monitor in an existing allocation, as in Agent0.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
job_id=''; kind=reproduce; gpu=false; command_args=()
while (( $# )); do
    case "$1" in
        --help|-h)
            echo 'Usage: ./scripts/attach.sh [JOB_ID] [--install | --preflight | --smoke | --eval | --kind KIND] [--gpu | -- COMMAND ...]'
            echo 'Defaults to a shell in the latest reproduction job. --gpu runs watch -n 1 nvidia-smi.'
            echo 'Exit or Ctrl-C leaves the monitoring step; the batch job keeps running.'
            exit 0 ;;
        --install|--install-envs) kind=install-envs ;;
        --preflight) kind=preflight ;;
        --smoke) kind=training-smoke ;;
        --eval) kind=evaluate ;;
        --kind)
            if (( $# < 2 )); then echo '--kind requires a job kind.' >&2; exit 2; fi
            kind=$2; shift ;;
        --gpu) gpu=true ;;
        --)
            shift
            if (( $# == 0 )); then echo 'Expected a command after --.' >&2; exit 2; fi
            command_args=("$@"); break ;;
        *)
            if [[ -n "$job_id" || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
                echo 'Expected one positive numeric JOB_ID. Use --help for options.' >&2; exit 2
            fi
            job_id=$1 ;;
    esac
    shift
done
if [[ "$kind" == install ]]; then kind=install-envs; fi
case "$kind" in install-envs|preflight|training-smoke|reproduce|evaluate) ;; *) echo 'Unknown job kind.' >&2; exit 2 ;; esac
if [[ "$gpu" == true && ${#command_args[@]} -gt 0 ]]; then echo 'Choose --gpu or a command after --.' >&2; exit 2; fi
if [[ -z "$job_id" ]]; then
    record="$ROOT/logs/$kind-job-id.txt"
    if [[ "$kind" == install-envs && ! -r "$record" && -r "$ROOT/logs/install-job-id.txt" ]]; then record="$ROOT/logs/install-job-id.txt"; fi
    if [[ ! -r "$record" ]]; then
        echo "No saved $kind job ID. Supply one: ./scripts/attach.sh JOB_ID" >&2; exit 1
    fi
    job_id=$(tr -d '[:space:]' < "$record")
fi
if [[ ! "$job_id" =~ ^[1-9][0-9]*$ ]]; then echo 'Saved job ID is invalid; supply a numeric JOB_ID.' >&2; exit 2; fi
for program in squeue srun; do
    if ! command -v "$program" >/dev/null 2>&1; then echo "Missing $program; run on the cluster login node." >&2; exit 1; fi
done
if ! state=$(squeue --noheader --jobs="$job_id" --format=%T); then
    echo "Cannot query job $job_id; check the job ID and Slurm connection." >&2; exit 1
fi
state=$(printf '%s' "$state" | tr -d '[:space:]')
if [[ "$state" != RUNNING ]]; then
    echo "Job $job_id is ${state:-not in the active queue}; attach requires RUNNING. Check squeue --me." >&2; exit 1
fi
if [[ "$gpu" == true ]]; then command_args=(watch -n 1 nvidia-smi)
elif (( ${#command_args[@]} == 0 )); then command_args=(bash -l); fi
echo "Attaching to job $job_id. Exit or Ctrl-C leaves this step; the batch job keeps running."
exec srun --jobid="$job_id" --overlap -N1 -n1 -c1 --pty "${command_args[@]}"
