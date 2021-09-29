#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --job-name=metaoptnet_tiered_imagenet
#SBATCH --output=../logs/metaoptnet_tiered_imagenet_%a.out
#SBATCH --error=../logs/metaoptnet_tiered_imagenet_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0

source ../env/bin/activate
cd .. && python -m src.main --exp_name metaoptnet_tiered_imagenet --train --model metaoptnet --runs 1 --folder ./data --meta-lr 0.01 --task_sampler $SLURM_ARRAY_TASK_ID --dataset tiered_imagenet --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 60 --output-folder ./config/metaoptnet_tiered_imagenet/$SLURM_ARRAY_TASK_ID/
