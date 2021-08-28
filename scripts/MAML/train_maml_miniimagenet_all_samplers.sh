#!/bin/bash
#SBATCH --job-name=maml_miniimagenet
#SBATCH --output=../logs/maml_miniimagenet_%a.out
#SBATCH --error=../logs/maml_miniimagenet_%a.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0-5

source ../env/bin/activate
cd .. && python -m src.main --exp_name maml_miniimagenet --train --runs 1 --folder ./data --task_sampler $SLURM_ARRAY_TASK_ID --dataset miniimagenet --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 150 --output-folder ./config/maml_miniimagenet/$SLURM_ARRAY_TASK_ID/