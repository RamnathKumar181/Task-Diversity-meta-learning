#!/bin/bash
#SBATCH --job-name=cnaps_omniglot_no_diversity_batch_sampling
#SBATCH --output=../logs/cnaps_omniglot_no_diversity_batch_sampling.out
#SBATCH --error=../logs/cnaps_omniglot_no_diversity_batch_sampling.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

source ../env/bin/activate
cd .. && python -m src.main --train --model cnaps --runs 1 ./data --meta-lr 0.001 --task_sampler no_diversity_batch --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 10 --output-folder ./config/cnaps_omniglot_no_diversity_batch_sampling/
python -m src.main ./data --output-folder ./config/cnaps_omniglot_no_diversity_batch_sampling/
