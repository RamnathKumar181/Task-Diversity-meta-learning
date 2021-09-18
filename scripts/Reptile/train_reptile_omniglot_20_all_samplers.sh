#!/bin/bash
#SBATCH --job-name=reptile_omniglot_20
#SBATCH --output=../logs/reptile_omniglot_20%a.out
#SBATCH --error=../logs/reptile_omniglot_20%a.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0-7

source ../env/bin/activate
cd .. && python -m src.main --exp_name reptile_omniglot_20 --train --task_sampler $SLURM_ARRAY_TASK_ID --model reptile --runs 1 --image-size 28 --folder ./data --dataset omniglot --num-ways 20 --num-shots 1 --use-cuda --step-size 0.33 --lr 0.01 --batch-size 32 --num-workers 4 --num-epochs 150 --meta-lr 0.001 --output-folder ./config/reptile_omniglot_20/$SLURM_ARRAY_TASK_ID/
