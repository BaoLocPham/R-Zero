#!/usr/bin/env bash
set -euo pipefail
source "${RZERO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/project-env.sh"
cd "$RZERO_ROOT"
: "${SLURM_JOB_ID:?Submit with scripts/submit.sh or run inside a Slurm allocation}"
kind=$1; shift
if [[ "$kind" == install ]]; then kind=install-envs; fi
case "$kind" in install-envs|preflight|training-smoke|reproduce|evaluate) ;; *) echo "Unknown job kind: $kind" >&2; exit 2 ;; esac
# Initialize the Conda shell function as well as loading the site module.
if [[ "$kind" == install-envs || -n "${RZERO_CONDA_INIT:-}${RZERO_CONDA_MODULE:-}" ]] || command -v conda >/dev/null 2>&1; then
 source "$RZERO_ROOT/scripts/03-load-modules.sh"
fi
echo "Started: $(date -Is)"
echo "Job: $SLURM_JOB_ID; kind: $kind; host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}; CPUs=${SLURM_CPUS_PER_TASK:-unset}"
mkdir -p "$REPRO_CACHE" "$REPRO_ENVS" "$STORAGE_PATH"/{models,evaluation,generated_question,temp_results,datasets}
export RZERO_SERVICE_PORT="${RZERO_SERVICE_PORT:-$((10000 + SLURM_JOB_ID % 40000))}"
export RAY_TMPDIR="${SLURM_TMPDIR:-/tmp}/rz-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/rzero-$USER-${SLURM_JOB_ID}/triton"
if [[ "$kind" == install-envs ]]; then
 # Ninja otherwise detects all node CPUs rather than the Slurm CPU allocation.
 export MAX_JOBS="${SLURM_CPUS_PER_TASK:-2}"
 command -v conda >/dev/null
 if [[ -x "$RZERO_ENV/bin/python" ]]; then
  "$RZERO_ENV/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else "R-Zero requires Python 3.10; choose a new RZERO_ENV prefix")'
  echo "Reusing existing Python 3.10 environment: $RZERO_ENV"
 else
  conda create -y --prefix "$RZERO_ENV" python=3.10
 fi
 conda activate "$RZERO_ENV"
 # flash-attn builds against the installed PyTorch/CUDA toolchain.
 python -m pip install --upgrade pip setuptools wheel packaging ninja
 python -m pip install torch==2.7.0
 sed '/^flash_attn==/d' requirements.txt > "$RZERO_ENV/requirements-no-flash.txt"
 # Do not fall back to compiling PyAV without FFmpeg development libraries.
 python -m pip install --only-binary=av -r "$RZERO_ENV/requirements-no-flash.txt" stopit==1.1.2
 python -m pip install flash_attn==2.7.4.post1 --no-build-isolation
 python -m pip check
 exec python -m slurm.preflight
fi
if command -v conda >/dev/null; then conda activate "$RZERO_ENV"
else source "$RZERO_ENV/bin/activate"; fi
python -m slurm.preflight
if [[ "$kind" == preflight ]]; then exit 0; fi
if [[ "$kind" == training-smoke ]]; then
 exec python -m slurm.run reproduce --name "qwen3-smoke-$SLURM_JOB_ID" \
  --iterations 1 --questioner-steps 1 --solver-steps 1 \
  --shards 2 --samples-per-shard 100 --skip-evaluation "$@"
fi
exec python -m slurm.run "$kind" "$@"
