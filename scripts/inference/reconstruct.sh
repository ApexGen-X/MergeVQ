## NPU
#python reconstruct.py \
#--config_file "configs/npu/imagenet_lfqgan_256_L.yaml" \
#--ckpt_path  ../upload_ckpts/in1k_256_L/imagenet_256_L.ckpt \
#--save_dir "./visualize" \
#--version  "1k" \
#--image_num 50 \
#--image_size 256 \


##GPU
# python reconstruct.py \
# --config_file "configs/gpu/imagenet_lfqgan_256_L.yaml" \
# --ckpt_path  ../upload_ckpts/in1k_256_L/imagenet_256_L.ckpt \
# --save_dir "./visualize" \
# --version  "1k" \
# --image_num 50 \
# --image_size 256 \

# ## NPU
# python reconstruct.py \
# --config_file "configs/npu/imagenet_lfqgan_256_L.yaml" \
# --ckpt_path  ../upload_ckpts/in1k_256_L/imagenet_256_L.ckpt \
# --save_dir "./visualize" \
# --version  "1k" \
# --image_num 50 \
# --image_size 256 \

##GPU
# python reconstruct.py \
# --config_file "configs/gpu/imagenet_lfqgan_256_L.yaml" \
# --ckpt_path  ../upload_ckpts/in1k_256_L/imagenet_256_L.ckpt \
# --save_dir "./visualize" \
# --version  "1k" \
# --image_num 50 \
# --image_size 256 \

# /liuzicheng/zly/OpenMAGVIT2/configs/gpu/lfqgan_256_k2l_L12_res462_d96/imagenet_lfqgan_256_L_k2l_L12_res462_d96_k144_b16_4gpu_4acc_20ep.yaml
# /liuzicheng/zly/OpenMAGVIT2/checkpoints/imagenet_lfqgan_128_L_k2l_L12_res462_d96_k256_b16_4gpu_4acc_20ep/test/epoch=19-step=400380.ckpt
# for keep in "k1024" "k576" "k400" "k324" "k256" "k144"; do
for keep in "k576"; do
    python reconstruct.py \
    --config_file "configs/gpu/lfqgan_128_k2l_L12_res4_d64/imagenet_lfqgan_128_L_k2l_L12_res4_d64_"$keep"_b32_4gpu_2acc_20ep.yaml" \
    --ckpt_path  "/liuzicheng/zly/.cache/Open_MAGVIT2/imagenet_128_L.ckpt" \
    --save_dir "results/imagenet_lfqgan_128_L_k2l_L12_res462_d96_k256_b16_4gpu_4acc_20ep/vis/vis_selected"$keep \
    --version  "1k" \
    --image_num 42 \
    --image_size 256
    # echo $keep
done