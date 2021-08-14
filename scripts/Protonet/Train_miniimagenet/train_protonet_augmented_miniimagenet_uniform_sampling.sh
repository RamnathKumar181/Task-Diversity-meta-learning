#!/bin/bash
#SBATCH --job-name=protonet_augmented_miniimagenet_uniform_sampling
#SBATCH --output=../logs/protonet_augmented_miniimagenet_uniform_sampling.out
#SBATCH --error=../logs/protonet_augmented_miniimagenet_uniform_sampling.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

source ../env/bin/activate
cd .. && python -m src.main --exp_name protonet_augmented_miniimagenet_uniform_sampling --use-random-crop --use-color-jitter --train --model protonet --runs 1 ./data --dataset augmented_miniimagenet --meta-lr 0.001 --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 100 --output-folder ./config/protonet_augmented_miniimagenet_uniform_sampling/
