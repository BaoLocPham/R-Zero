#!/usr/bin/env bash
# Source this file so Conda remains initialized in the calling shell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Run: source scripts/03-load-modules.sh' >&2; exit 2
fi
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/project-env.sh" || return 1
rzero_load_modules() {
    local rzero_conda_base
    if [[ -n "${RZERO_CONDA_INIT:-}" ]]; then
        if [[ ! -r "$RZERO_CONDA_INIT" ]]; then
            echo "Conda initialization script is unreadable: $RZERO_CONDA_INIT" >&2; return 1
        fi
        source "$RZERO_CONDA_INIT" || return 1
    elif ! command -v conda >/dev/null 2>&1; then
        if ! command -v module >/dev/null 2>&1; then
            if [[ -r /usr/share/lmod/lmod/init/bash ]]; then source /usr/share/lmod/lmod/init/bash || return 1
            elif [[ -r /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh || return 1
            else echo 'Set RZERO_CONDA_INIT in cluster.local.sh.' >&2; return 1; fi
        fi
        if [[ -n "${RZERO_MODULE_PATH:-}" ]]; then module use "$RZERO_MODULE_PATH" || return 1; fi
        if [[ -z "${RZERO_CONDA_MODULE:-}" ]]; then
            echo 'Set RZERO_CONDA_MODULE or RZERO_CONDA_INIT in cluster.local.sh.' >&2; return 1
        fi
        module load "$RZERO_CONDA_MODULE" || return 1
    fi
    if ! command -v conda >/dev/null 2>&1; then echo 'Conda is unavailable.' >&2; return 1; fi
    rzero_conda_base=$(conda info --base) || return 1
    source "$rzero_conda_base/etc/profile.d/conda.sh" || return 1
    conda --version || return 1
}
if rzero_load_modules; then unset -f rzero_load_modules
else unset -f rzero_load_modules; return 1; fi
