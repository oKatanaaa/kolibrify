config=$1
repo=$2
quant=${3:-q8_0}
gpu_id=${4:-0}

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu_id kolibrify-push $config $repo --quantize $quant