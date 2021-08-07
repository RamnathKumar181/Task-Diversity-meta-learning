#!/bin/bash
#SBATCH --job-name=metaoptnet_omniglot_no_diversity_task_sampling
#SBATCH --output=../logs/metaoptnet_omniglot_no_diversity_task_sampling.out
#SBATCH --error=../logs/metaoptnet_omniglot_no_diversity_task_sampling.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

source ../env/bin/activate
cd .. && python -m src.main --exp_name metaoptnet_omniglot_no_diversity_task_sampling --train --model metaoptnet --runs 1 ./data --meta-lr 0.1 --task_sampler no_diversity_task --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 600 --output-folder ./config/metaoptnet_omniglot_no_diversity_task_sampling/
