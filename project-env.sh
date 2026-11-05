#!/usr/bin/env bash
export RZERO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$RZERO_ROOT/cluster.local.sh" ]]; then source "$RZERO_ROOT/cluster.local.sh"; fi
# Named entrypoints override stale local scheduling settings, including in jobs.
export RZERO_CLUSTER="${RZERO_CLUSTER_OVERRIDE-${RZERO_CLUSTER:-}}"
export RZERO_GONDOR_GPU="${RZERO_GONDOR_GPU_OVERRIDE-${RZERO_GONDOR_GPU:-a100}}"
export REPRO_SCRATCH="${REPRO_SCRATCH:-/scratch/$USER/self-evol}"
export REPRO_CACHE="${REPRO_CACHE:-$REPRO_SCRATCH/cache}"
export REPRO_ENVS="${REPRO_ENVS:-$REPRO_SCRATCH/envs}"
export STORAGE_PATH="${STORAGE_PATH:-$REPRO_SCRATCH/artifacts/rzero}"
export HF_HOME="${HF_HOME:-$REPRO_CACHE/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPRO_CACHE/pip}"
export TORCH_HOME="${TORCH_HOME:-$REPRO_CACHE/torch}"
export RZERO_ENV="${RZERO_ENV:-$REPRO_ENVS/rzero}"
export RZERO_GPUS="${RZERO_GPUS:-2}"
export RZERO_CPUS="${RZERO_CPUS:-16}"
export RZERO_PARTITION="${RZERO_PARTITION:-gpu}"
export RZERO_GRES="${RZERO_GRES:-gpu:$RZERO_GPUS}"
export RZERO_MEM="${RZERO_MEM:-128G}"
export RZERO_TIME="${RZERO_TIME:-24:00:00}"
if [[ -n "${RZERO_CLUSTER:-}" ]]; then
 case "$RZERO_CLUSTER" in
  rohan|gondor) source "$RZERO_ROOT/clusters/$RZERO_CLUSTER.sh" ;;
  *) echo "Unsupported profile: $RZERO_CLUSTER (use custom resource settings instead)" >&2; return 2 ;;
 esac
fi
export PYTHONPATH="$RZERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_DISABLE_COMPILE_CACHE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
