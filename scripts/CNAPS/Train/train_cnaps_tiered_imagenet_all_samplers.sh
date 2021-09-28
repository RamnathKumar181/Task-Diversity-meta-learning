#!/bin/bash
#SBATCH --job-name=cnaps_tiered_imagenet
#SBATCH --output=../logs/cnaps_tiered_imagenet_%a.out
#SBATCH --error=../logs/cnaps_tiered_imagenet_%a.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0-7

source ../env/bin/activate
cd .. && python -m src.main --exp_name cnaps_tiered_imagenet --train --model cnaps --runs 1 --folder ./data --task_sampler $SLURM_ARRAY_TASK_ID --dataset tiered_imagenet --meta-lr 0.0005 --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 16 --num-workers 8 --num-epochs 10 --output-folder ./config/cnaps_tiered_imagenet/$SLURM_ARRAY_TASK_ID/
