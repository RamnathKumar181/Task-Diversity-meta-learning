#!/bin/bash
#SBATCH --job-name=cnaps_miniimagenet
#SBATCH --output=../logs/cnaps_miniimagenet_%a.out
#SBATCH --error=../logs/cnaps_miniimagenet_%a.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=6

source ../env/bin/activate
cd .. && python -m src.main --exp_name cnaps_augmented_miniimagenet --train --model cnaps --use-augmentations --runs 1 --folder ./data --task_sampler $SLURM_ARRAY_TASK_ID --dataset miniimagenet --meta-lr 0.001 --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 16 --num-workers 8 --num-epochs 10 --output-folder ./config/cnaps_augmented_miniimagenet/$SLURM_ARRAY_TASK_ID/
