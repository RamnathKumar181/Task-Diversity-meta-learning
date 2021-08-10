#!/bin/bash
#SBATCH --job-name=maml_miniimagenet_single_batch_to_omniglot
#SBATCH --output=../logs/maml_miniimagenet_single_batch_to_omniglot.out
#SBATCH --error=../logs/maml_miniimagenet_single_batch_to_omniglot.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

source ../env/bin/activate
cd .. && python -m src.main --exp_name maml_miniimagenet_single_batch_to_omniglot --runs 1 ./data --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --batch-size 1 --num-workers 8 --output-folder ./config/maml_miniimagenet_single_batch/
