#! /bin/bash

IMAGE_SIZE="256"
CONFIG_TAIL=".yaml"
RESULT_TAIL=".txt"

### GPU x 8 evaluation

# TODO: 替换config文件名(CONFIG_BASE), results目录(RESULT_BASE), checkpoints文件名(CKPT)
KEEP_VALUES=("256" "400" "576" "1024")
GPUS=("0" "1" "2" "3")
CONFIG_BASE="configs/gpu_s1/eval_256_G/imagenet_mergevq_256_G_d64_k"
RESULT_BASE="results/imagenet_mergevq_256_G_d64_b8_8gpu_2acc_200ep/test/eval_k"
CKPT="checkpoints/stage1_pt/imagenet_mergevq_256_G_d64_b8_8gpu_2acc_200ep/epoch=199-step=2002000.ckpt"

# # TODO: 替换config文件名(CONFIG_BASE), results目录(RESULT_BASE), checkpoints文件名(CKPT)
# KEEP_VALUES=("256" "400" "576" "1024")
# GPUS=("0" "1" "2" "3")
# CONFIG_BASE="configs/gpu_s1/eval_256_G/imagenet_mergevq_256_G_d96_k"
# RESULT_BASE="results/imagenet_mergevq_256_G_d96_b8_8gpu_2acc_200ep/test/eval_k"
# CKPT="checkpoints/stage1_pt/imagenet_mergevq_256_G_d96_b8_8gpu_2acc_200ep/epoch=199-step=2002000.ckpt"

# # TODO: 替换config文件名(CONFIG_BASE), results目录(RESULT_BASE), checkpoints文件名(CKPT)
# KEEP_VALUES=("64" "100" "144" "196" "256")
# GPUS=("0" "1" "2" "3" "4")
# CONFIG_BASE=" configs/gpu_s1/eval_256_GR/imagenet_mergevq_256_GR_d64_k"
# RESULT_BASE="results/imagenet_mergevq_256_GR_d64_b16_8gpu_2acc_200ep/test/eval_k"
# CKPT="checkpoints/stage1_pt/imagenet_mergevq_256_GR_d64_b16_8gpu_2acc_200ep/epoch=199-step=2002000.ckpt"

# # TODO: 替换config文件名(CONFIG_BASE), results目录(RESULT_BASE), checkpoints文件名(CKPT)
# KEEP_VALUES=("64" "100" "144" "196" "256")
# GPUS=("0" "1" "2" "3" "4")
# CONFIG_BASE="configs/gpu_s1/eval_256_GR/imagenet_mergevq_256_GR_d96_k"
# RESULT_BASE="results/imagenet_mergevq_256_GR_d96_b16_8gpu_2acc_200ep/test/eval_k"
# CKPT="checkpoints/stage1_pt/imagenet_mergevq_256_GR_d96_b16_8gpu_2acc_200ep/epoch=199-step=2002000.ckpt"

### start evaluation with all KEEP values and GPU IDs parallelly
if [ ${#KEEP_VALUES[@]} -ne ${#GPUS[@]} ]; then
    echo "Error: KEEP_VALUES and GPUS arrays must have the same length."
    exit 1
fi
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
