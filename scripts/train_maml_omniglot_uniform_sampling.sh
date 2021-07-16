#!/bin/bash
#SBATCH --job-name=maml_omniglot_uniform_sampling
#SBATCH --output=../logs/maml_omniglot_uniform_sampling_%j.out
#SBATCH --error=../logs/maml_omniglot_uniform_sampling_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=6:59:59
#SBATCH --partition=unkillable

source ../env/bin/activate
cd .. && python -m src.main --train ./data --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 600 --output-folder ./config/
