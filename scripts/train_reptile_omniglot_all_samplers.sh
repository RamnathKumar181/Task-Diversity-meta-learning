#!/bin/bash
#SBATCH --job-name=reptile_omniglot
#SBATCH --output=../logs/reptile_omniglot_%a.out
#SBATCH --error=../logs/reptile_omniglot_%a.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0

source ../env/bin/activate
cd .. && python -m src.main --exp_name reptile_omniglot_uniform_sampling --train --model reptile --runs 1 --folder ./data --dataset omniglot --image-size 28 --num-ways 5 --num-shots 1 --use-cuda --step-size 0.33 --lr 0.001 --batch-size 32 --num-workers 8 --num-epochs 100 --output-folder ./config/reptile_omniglot/$SLURM_ARRAY_TASK_ID/
