#!/bin/bash
#SBATCH --job-name=maml_omniglot_uniform_sampling_to_miniimagenet
#SBATCH --output=../logs/maml_omniglot_uniform_sampling_to_miniimagenet.out
#SBATCH --error=../logs/maml_omniglot_uniform_sampling_to_miniimagenet.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

source ../env/bin/activate
cd .. && python -m src.main --exp_name maml_omniglot_uniform_sampling_to_miniimagenet --runs 1 ./data --dataset miniimagenet --num-ways 5 --num-shots 1 --use-cuda --batch-size 32 --num-workers 8 --output-folder ./config/maml_omniglot_uniform_sampling/
