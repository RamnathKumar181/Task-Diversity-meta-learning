#!/bin/bash
#SBATCH --job-name=protonet_miniimagenet_single_batch
#SBATCH --output=../logs/protonet_miniimagenet_single_batch.out
#SBATCH --error=../logs/protonet_miniimagenet_single_batch.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

source ../env/bin/activate
cd .. && python -m src.main --exp_name protonet_miniimagenet_single_batch --train --model protonet --runs 1 ./data --dataset miniimagenet --meta-lr 0.001 --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 1 --num-workers 8 --num-epochs 100 --output-folder ./config/protonet_miniimagenet_single_batch/
