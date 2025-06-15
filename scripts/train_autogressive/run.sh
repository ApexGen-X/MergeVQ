export MASTER_ADDR=${1:-localhost}
export MASTER_PORT=${2:-10055}
export NODE_RANK=${3:-0}

export OMP_NUM_THREADS=6
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

echo $MASTER_ADDR
echo $MASTER_PORT

# GPU
NODE_RANK=$NODE_RANK python main.py fit --config configs/gpu_s2/imagenet_cond_llamagen_B_256_G_d64_k1024_8gpu_ep300.yaml
NODE_RANK=$NODE_RANK python main.py fit --config configs/gpu_s2/imagenet_cond_llamagen_B_256_GR_d64_k256_8gpu_ep300.yaml

