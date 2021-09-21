#!/bin/bash
#SBATCH --job-name=test_protonet_omniglot_20
#SBATCH --output=../logs/test_protonet_omniglot_20_%a.out
#SBATCH --error=../logs/test_protonet_omniglot_20_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --array=0-7

source ../env/bin/activate
cd ..
python -m src.main --exp_name test_protonet_omniglot_20 --log-test-tasks --model protonet --runs 1 --folder ./data --meta-lr 0.001 --image-size 28 --task_sampler $SLURM_ARRAY_TASK_ID --dataset omniglot --num-ways 20 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 100 --output-folder ./config/protonet_omniglot_20/$SLURM_ARRAY_TASK_ID/