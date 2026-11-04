# Copy to cluster.local.sh; shared paths must exist on login and compute nodes.
export REPRO_SCRATCH="/scratch/$USER/self-evol"
export RZERO_CLUSTER=gondor
export RZERO_GONDOR_GPU=a100  # a100, h100, h200 (Ampere or newer required)
export RZERO_GPUS=2
export RZERO_CPUS=16
export RZERO_MEM=128G
export RZERO_TIME=24:00:00
export RZERO_ACCOUNT=""
export RZERO_QOS=""
export RZERO_CONDA_INIT="$HOME/miniconda3/etc/profile.d/conda.sh"
export RZERO_CONDA_MODULE=""
export RZERO_MODULE_PATH=""
# Optional credentials: export HF_TOKEN and WANDB_API_KEY in your shell.
# Console logging is the default; no Hugging Face uploads are performed.
