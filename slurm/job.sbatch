#!/usr/bin/env bash
set -euo pipefail
source "${RZERO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/project-env.sh"
cd "$RZERO_ROOT"
: "${SLURM_JOB_ID:?Submit with scripts/submit.sh or run inside a Slurm allocation}"
if [[ -n "${RZERO_MODULE_PATH:-}" ]]; then module use "$RZERO_MODULE_PATH"; fi
if [[ -n "${RZERO_CONDA_MODULE:-}" ]]; then module load "$RZERO_CONDA_MODULE"; fi
if [[ -n "${RZERO_CONDA_INIT:-}" ]]; then source "$RZERO_CONDA_INIT"; fi
kind=$1; shift
mkdir -p "$REPRO_CACHE" "$REPRO_ENVS" "$STORAGE_PATH"/{models,evaluation,generated_question,temp_results,datasets}
export RZERO_SERVICE_PORT="${RZERO_SERVICE_PORT:-$((10000 + SLURM_JOB_ID % 40000))}"
export RAY_TMPDIR="${SLURM_TMPDIR:-/tmp}/rz-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/rzero-$USER-${SLURM_JOB_ID}/triton"
if [[ "$kind" == install ]]; then
 command -v conda >/dev/null
 conda create -y --prefix "$RZERO_ENV" python=3.10
 conda activate "$RZERO_ENV"
 # flash-attn builds against the installed PyTorch/CUDA toolchain.
 python -m pip install --upgrade pip setuptools wheel packaging ninja
 python -m pip install torch==2.7.0
 sed '/^flash_attn==/d' requirements.txt > "$RZERO_ENV/requirements-no-flash.txt"
 python -m pip install -r "$RZERO_ENV/requirements-no-flash.txt" stopit==1.1.2
 python -m pip install flash_attn==2.7.4.post1 --no-build-isolation
 python -m pip check
 exec python -m slurm.preflight
fi
if command -v conda >/dev/null; then conda activate "$RZERO_ENV"
else source "$RZERO_ENV/bin/activate"; fi
python -m slurm.preflight
if [[ "$kind" == preflight ]]; then exit 0; fi
exec python -m slurm.run "$kind" "$@"
