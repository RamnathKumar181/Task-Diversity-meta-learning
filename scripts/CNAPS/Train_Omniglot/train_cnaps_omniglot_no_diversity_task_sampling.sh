#!/bin/bash
#SBATCH --job-name=cnaps_omniglot_no_diversity_task_sampling
#SBATCH --output=../logs/cnaps_omniglot_no_diversity_task_sampling.out
#SBATCH --error=../logs/cnaps_omniglot_no_diversity_task_sampling.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

source ../env/bin/activate
cd .. && python -m src.main --exp_name cnaps_omniglot_no_diversity_task_sampling --train --model cnaps --runs 1 ./data --meta-lr 0.001 --task_sampler no_diversity_task --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 16 --num-workers 8 --num-epochs 10 --output-folder ./config/cnaps_omniglot_no_diversity_task_sampling/
