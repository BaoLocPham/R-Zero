# Copy to cluster.local.sh on Rohan; edit account/QoS if required.
# Scheduling is supported; V100 training/installation still needs the backend port.
export REPRO_SCRATCH="/scratch/$USER/self-evol"
export STORAGE_PATH="$REPRO_SCRATCH/artifacts/rzero"
export RZERO_CLUSTER=rohan
export RZERO_GPUS=2
export RZERO_CPUS=16  # Current Ray driver needs >=13 for two GPUs, unlike Agent0.
export RZERO_MEM=128G
export RZERO_TIME=08:00:00
export RZERO_INSTALL_CPUS=2
export RZERO_INSTALL_MEM=32G
export RZERO_INSTALL_TIME=04:00:00
export RZERO_PREFLIGHT_CPUS=2
export RZERO_PREFLIGHT_MEM=8G
export RZERO_PREFLIGHT_TIME=00:10:00
export RZERO_ACCOUNT=""
export RZERO_QOS=""
export RZERO_MAIL_USER=""
export RZERO_MODULE_PATH=/sw/modules/all
export RZERO_CONDA_MODULE=Anaconda3/2024.02
export RZERO_CONDA_INIT=""  # Alternatively, set a readable conda.sh path.
