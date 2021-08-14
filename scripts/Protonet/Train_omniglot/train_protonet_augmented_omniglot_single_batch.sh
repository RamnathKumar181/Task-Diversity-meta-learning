#!/bin/bash
#SBATCH --job-name=protonet_augmented_omniglot_single_batch
#SBATCH --output=../logs/protonet_augmented_omniglot_single_batch.out
#SBATCH --error=../logs/protonet_augmented_omniglot_single_batch.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

source ../env/bin/activate
cd .. && python -m src.main --exp_name protonet_augmented_omniglot_single_batch --use-random-crop --use-color-jitter --train --model protonet --runs 1 ./data --dataset omniglot --meta-lr 0.001 --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 1 --num-workers 8 --num-epochs 50 --output-folder ./config/protonet_augmented_omniglot_single_batch/
