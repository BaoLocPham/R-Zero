#!/usr/bin/env bash
# Follow both logs without requiring a working Slurm controller or environment.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
job_id=''; kind=reproduce; follow=true
while (( $# )); do
    case "$1" in
        --help|-h)
            echo 'Usage: ./scripts/logs.sh [JOB_ID] [--once] [--install | --preflight | --smoke | --eval | --kind KIND]'
            echo 'Defaults to the latest reproduction job. --install selects install-envs.'
            echo 'Follows the last 50 lines of stdout/stderr, waiting for queued jobs. Ctrl-C only stops viewing.'
            exit 0 ;;
        --once) follow=false ;;
        --install|--install-envs) kind=install-envs ;;
        --preflight) kind=preflight ;;
        --smoke) kind=training-smoke ;;
        --eval) kind=evaluate ;;
        --kind)
            if (( $# < 2 )); then echo '--kind requires a job kind.' >&2; exit 2; fi
            kind=$2; shift ;;
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
legacy=false
if [[ -z "$job_id" ]]; then
    record="$ROOT/logs/$kind-job-id.txt"
    if [[ "$kind" == install-envs && ! -r "$record" && -r "$ROOT/logs/install-job-id.txt" ]]; then
        record="$ROOT/logs/install-job-id.txt"; legacy=true
    fi
    if [[ ! -r "$record" ]]; then
        echo "No saved $kind job ID. Supply one: ./scripts/logs.sh JOB_ID --kind $kind" >&2; exit 1
    fi
    job_id=$(tr -d '[:space:]' < "$record")
fi
if [[ ! "$job_id" =~ ^[1-9][0-9]*$ ]]; then echo 'Saved job ID is invalid; supply a numeric JOB_ID.' >&2; exit 2; fi
# Logs from submissions made before the install-envs alias keep their old names.
if [[ "$kind" == install-envs && ! -f "$ROOT/logs/rzero-install-envs-$job_id.out" && -f "$ROOT/logs/rzero-install-$job_id.out" ]]; then legacy=true; fi
log_kind=$kind
if [[ "$legacy" == true ]]; then log_kind=install; fi
output="$ROOT/logs/rzero-$log_kind-$job_id.out"
errors="$ROOT/logs/rzero-$log_kind-$job_id.err"
echo "$kind job $job_id"
if [[ "$follow" == true ]]; then
    echo 'Following stdout/stderr. Ctrl-C stops viewing; the job keeps running.'
    if [[ ! -f "$output" || ! -f "$errors" ]]; then echo 'Waiting for log files; they appear when Slurm starts the job.'; fi
    exec tail -n 50 -F -- "$output" "$errors"
fi
exec tail -n 50 -- "$output" "$errors"
