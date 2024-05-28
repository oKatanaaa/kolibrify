config=$1
dataset=$2
path=$3
gpus=${4:-0}

kolibrify-predict $config $dataset $path --gpus $gpus