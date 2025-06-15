#! /bin/bash

### GPU x 8 train
python main.py fit --config configs/gpu_s1/imagenet_mergevq_256_R_vitb_d64_8gpu_270ep.yaml

### NPU train
# python main.py fit --config configs/gpu_s1/imagenet_mergevq_256_R_vitb_d64_8gpu_270ep.yaml

sleep 10s

### GPU eval
CONFIG_BASE="configs/gpu_s1/eval_256_R/imagenet_mergevq_256_R_vitb_d64_k"
CONFIG_TAIL=".yaml"
RESULT_BASE="results/stage1_pt/imagenet_mergevq_256_R_vitb_d64_8gpu_270ep/test/eval_k"
RESULT_TAIL=".txt"
# list of KEEP values and GPU IDs
KEEP_VALUES=("64" "100" "144" "256")
IMAGE_SIZE="256"
# modify the path of checkpoint and available GPU IDs
CKPT="checkpoints/stage1_pt/imagenet_mergevq_256_R_vitb_d64_8gpu_270ep/epoch=269-step=1351080.ckpt"
GPUS=("0" "1" "2" "3")

# check list of KEEP_VALUES and GPUS
if [ ${#KEEP_VALUES[@]} -ne ${#GPUS[@]} ]; then
    echo "Error: KEEP_VALUES and GPUS arrays must have the same length."
    exit 1
fi

# run evaluation with all KEEP values and GPU IDs parallelly
for i in "${!KEEP_VALUES[@]}"; do
    KEEP=${KEEP_VALUES[$i]}
    GPU=${GPUS[$i]}
    # config path and log output
    CONFIG_FILE="${CONFIG_BASE}${KEEP}${CONFIG_TAIL}"
    LOG_FILE="${RESULT_BASE}${KEEP}${RESULT_TAIL}"
    # run evaluation
    CUDA_VISIBLE_DEVICES="$GPU" python -u evaluation.py --config_file "$CONFIG_FILE" \
        --ckpt_path "$CKPT" --image_size $IMAGE_SIZE |& tee "$LOG_FILE" 2>&1 &
done

# waiting till all tasks finished
wait
