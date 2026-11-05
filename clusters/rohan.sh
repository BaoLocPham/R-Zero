# Rohan scheduling profile. The V100 training backend port is still pending.
export RZERO_PARTITION=gpu
export RZERO_GRES="gpu:v100:${RZERO_GPUS:-2}"
export RZERO_CONSTRAINT=gpu-32G
export RZERO_MODULE_PATH="${RZERO_MODULE_PATH-/sw/modules/all}"
export RZERO_CONDA_MODULE="${RZERO_CONDA_MODULE-Anaconda3/2024.02}"
