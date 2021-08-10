#!/bin/bash
#SBATCH --job-name=cnaps_omniglot_single_batch
#SBATCH --output=../logs/cnaps_omniglot_single_batch.out
#SBATCH --error=../logs/cnaps_omniglot_single_batch.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

source ../env/bin/activate
cd .. && python -m src.main --exp_name cnaps_omniglot_single_batch --train --model cnaps --runs 1 ./data --dataset omniglot --meta-lr 0.001 --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 1 --num-workers 8 --num-epochs 10 --output-folder ./config/cnaps_omniglot_single_batch/
