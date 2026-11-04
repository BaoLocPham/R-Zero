case "${RZERO_GONDOR_GPU:-a100}" in
 a100) export RZERO_PARTITION=gpu; export RZERO_CONSTRAINT=gpu-a100 ;;
 h100|h200) export RZERO_PARTITION=gpu-large; export RZERO_CONSTRAINT="gpu-${RZERO_GONDOR_GPU}" ;;
 *) echo 'RZERO_GONDOR_GPU must be a100, h100 or h200' >&2; return 2 ;;
esac
export RZERO_GRES="gpu:${RZERO_GONDOR_GPU:-a100}:${RZERO_GPUS:-2}"
