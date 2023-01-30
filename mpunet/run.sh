#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=36000M
# we run on the gpu partition and we allocate 4 titanx gpus
#SBATCH -p gpu --gres=gpu:titanx:4
#We expect that our program should not run longer than 6 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=6:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
echo $CUDA_VISIBLE_DEVICES
## cd jaw_small_voxel_size_crop_double_norm
mp train --num_GPUs=3 --overwrite  --init_filters 64 #--force_GPU=0

mp predict --num_GPUs=4 --sum_fusion  --init_filters 64  \
--overwrite --by_radius --no_eval --force_GPU=0 --out_dir predictions_no_arg \
--use_diag_dim
# --fusion_save_to_disk --delete_fusion_after \
