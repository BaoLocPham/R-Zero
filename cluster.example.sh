# Copy to cluster.local.sh; shared paths must exist on login and compute nodes.
export REPRO_SCRATCH="/scratch/$USER/self-evol"
export RZERO_CLUSTER=""  # rohan, gondor, or empty for custom scheduling
export RZERO_GONDOR_GPU=a100  # Gondor: a100, h100 or h200
export RZERO_GPUS=2
export RZERO_CPUS=16
export RZERO_INSTALL_CPUS=2
export RZERO_INSTALL_MEM=32G
export RZERO_INSTALL_TIME=04:00:00
export RZERO_PREFLIGHT_CPUS=2
export RZERO_PREFLIGHT_MEM=8G
export RZERO_PREFLIGHT_TIME=00:10:00
export RZERO_MEM=128G
export RZERO_TIME=24:00:00
export RZERO_ACCOUNT=""
export RZERO_QOS=""
export RZERO_MAIL_USER=""  # optional Slurm notifications
export RZERO_CONDA_INIT="$HOME/miniconda3/etc/profile.d/conda.sh"
export RZERO_CONDA_MODULE=""
export RZERO_MODULE_PATH=""
# Optional credentials: export HF_TOKEN and WANDB_API_KEY in your shell.
# Console logging is the default; no Hugging Face uploads are performed.
# Named entrypoints override the scheduling profile in this file:
#   ./scripts/rohan.sh install-envs --test-only
#   ./scripts/gondor.sh --gpu a100 install-envs
# Rohan's scheduling profile does not yet provide a V100 training backend.
