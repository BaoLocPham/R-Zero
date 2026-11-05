#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cluster=${1:?Use rohan.sh or gondor.sh}; shift
case "$cluster" in rohan|gondor) ;; *) echo 'Unknown cluster.' >&2; exit 2 ;; esac
export RZERO_CLUSTER_OVERRIDE="$cluster"
if [[ "$cluster" == gondor ]]; then
    case "${1:-}" in
        --gpu)
            if (( $# < 2 )); then echo '--gpu requires a100, h100 or h200.' >&2; exit 2; fi
            export RZERO_GONDOR_GPU_OVERRIDE=$2; shift 2 ;;
        --gpu=*) export RZERO_GONDOR_GPU_OVERRIDE=${1#*=}; shift ;;
    esac
    if [[ -n "${RZERO_GONDOR_GPU_OVERRIDE+x}" ]]; then
        case "$RZERO_GONDOR_GPU_OVERRIDE" in
            a100|h100|h200) ;; *) echo '--gpu requires a100, h100 or h200.' >&2; exit 2 ;;
        esac
    fi
fi
if [[ $# == 0 || "$1" == --help || "$1" == -h ]]; then
    echo "Usage: bash scripts/$cluster.sh [--gpu a100|h100|h200 (Gondor)] COMMAND [ARGUMENTS...]"
    echo 'Commands: install-envs, preflight, training-smoke, reproduce, evaluate, download, allocate, logs, attach'
    echo "Example: bash scripts/$cluster.sh install-envs --test-only"
    echo 'Rohan selects V100 resources; its training backend port is still pending.'
    exit 0
fi
command=$1; shift
case "$command" in
    install|install-envs) exec bash "$SCRIPT_DIR/install-envs.sh" "$@" ;;
    download) exec bash "$SCRIPT_DIR/06-download-model.sh" "$@" ;;
    allocate) exec bash "$SCRIPT_DIR/02-allocate-gpus.sh" "$@" ;;
    logs|attach) exec bash "$SCRIPT_DIR/$command.sh" "$@" ;;
    *) exec bash "$SCRIPT_DIR/submit.sh" "$command" "$@" ;;
esac
