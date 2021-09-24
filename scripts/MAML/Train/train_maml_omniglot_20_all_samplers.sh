#!/bin/bash
#SBATCH --job-name=maml_omniglot_20
#SBATCH --output=../logs/maml_omniglot_20_%a.out
#SBATCH --error=../logs/maml_omniglot_20_%a.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0-7

source ../env/bin/activate
cd .. && python -m src.main --exp_name maml_omniglot_20 --train --runs 1 --folder ./data --image-size 28 --task_sampler $SLURM_ARRAY_TASK_ID --dataset omniglot --num-ways 20 --num-shots 1 --use-cuda --step-size 0.1 --meta-lr 0.001 --batch-size 16 --num-workers 8 --num-epochs 150 --num-adaptation-steps 5 --output-folder ./config/maml_omniglot_20/$SLURM_ARRAY_TASK_ID/
