config=$1
gpu_id=${2:-0}

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu_id kolibrify-dpo $config