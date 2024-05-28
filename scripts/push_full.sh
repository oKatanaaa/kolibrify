config=$1
repo=$2
gpu_id=${3:-0}

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu_id kolibrify-push $config $repo