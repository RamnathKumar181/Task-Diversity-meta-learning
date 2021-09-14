#!/bin/bash
#SBATCH --job-name=test_reptile_miniimagenet
#SBATCH --output=../logs/test_reptile_miniimagenet%a.out
#SBATCH --error=../logs/test_reptile_miniimagenet%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --array=0-7

source ../env/bin/activate
cd .. && python -m src.main --exp_name test_reptile_miniimagenet --log-test-tasks --task_sampler $SLURM_ARRAY_TASK_ID --model reptile --runs 1 --folder ./data --dataset miniimagenet --num-ways 5 --num-shots 1 --use-cuda --step-size 0.33 --lr 0.01 --batch-size 32 --num-workers 4 --num-epochs 150 --meta-lr 0.001 --output-folder ./config/reptile_miniimagenet_try_2/$SLURM_ARRAY_TASK_ID/
