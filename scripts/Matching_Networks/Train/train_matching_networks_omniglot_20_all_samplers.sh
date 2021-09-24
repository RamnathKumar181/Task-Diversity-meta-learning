#!/bin/bash
#SBATCH --job-name=matching_networks_omniglot_20
#SBATCH --output=../logs/matching_networks_omniglot_20_%a.out
#SBATCH --error=../logs/matching_networks_omniglot_20_%a.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0-7

source ../env/bin/activate
cd .. && python -m src.main --exp_name matching_networks_omniglot_20 --train --model matching_networks --runs 1 --folder $SLURM_TMPDIR/data --image-size 28 --task_sampler $SLURM_ARRAY_TASK_ID --dataset omniglot --num-ways 20 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 100 --output-folder ./config/matching_networks_omniglot_20/$SLURM_ARRAY_TASK_ID/
