config=$1
eval_lang=${2:-en}
gpus=${3:-0}

kolibrify-eval-ifeval $config --eval-lang eval_lang --gpus gpus