#!/bin/bash
#SBATCH --job-name=reptile_tiered_imagenet
#SBATCH --output=../logs/reptile_tiered_imagenet_%a.out
#SBATCH --error=../logs/reptile_tiered_imagenet_%a.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0

source ../env/bin/activate
cd .. && python -m src.main --exp_name reptile_tiered_imagenet --train --task_sampler $SLURM_ARRAY_TASK_ID --model reptile --runs 1 --folder ./data --dataset tiered_imagenet --num-ways 5 --num-shots 1 --use-cuda --step-size 0.33 --lr 0.01 --batch-size 32 --num-workers 8 --num-epochs 150 --meta-lr 0.001 --output-folder ./config/reptile_tiered_imagenet_try_2/$SLURM_ARRAY_TASK_ID/