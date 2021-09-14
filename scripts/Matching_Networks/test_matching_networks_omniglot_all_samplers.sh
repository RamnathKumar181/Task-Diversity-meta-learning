#!/bin/bash
#SBATCH --job-name=test_matching_networks_omniglot
#SBATCH --output=../logs/test_matching_networks_omniglot_%a.out
#SBATCH --error=../logs/test_matching_networks_omniglot_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --array=5

source ../env/bin/activate
cd .. && python -m src.main --exp_name matching_networks_omniglot --log-test-tasks --model matching_networks --runs 1 --folder $SLURM_TMPDIR/data --image-size 28 --task_sampler $SLURM_ARRAY_TASK_ID --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 100 --output-folder ./config/matching_networks_omniglot/$SLURM_ARRAY_TASK_ID/