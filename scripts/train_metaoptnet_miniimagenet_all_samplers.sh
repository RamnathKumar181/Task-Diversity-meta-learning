#!/bin/bash
#SBATCH --job-name=metaoptnet_miniimagenet
#SBATCH --output=../logs/metaoptnet_miniimagenet_%a.out
#SBATCH --error=../logs/metaoptnet_miniimagenet_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0-5

source ../env/bin/activate
cd .. && python -m src.main --exp_name metaoptnet_miniimagenet --train --model metaoptnet --runs 1 ./data --meta-lr 0.1 --task_sampler $SLURM_ARRAY_TASK_ID --dataset miniimagenet --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 60 --output-folder ./config/metaoptnet_miniimagenet/$SLURM_ARRAY_TASK_ID/
