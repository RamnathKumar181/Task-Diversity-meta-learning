#!/bin/bash
#SBATCH --job-name=test_matching_networks_miniimagenet
#SBATCH --output=../logs/test_matching_networks_miniimagenet_%a.out
#SBATCH --error=../logs/test_matching_networks_miniimagenet_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --array=0-7

source ../env/bin/activate
cd .. && python -m src.main --exp_name matching_networks_miniimagenet --log-test-tasks --model matching_networks --runs 1 --folder $SLURM_TMPDIR/data --task_sampler $SLURM_ARRAY_TASK_ID --dataset miniimagenet --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 100 --output-folder ./config/matching_networks_miniimagenet/$SLURM_ARRAY_TASK_ID/
