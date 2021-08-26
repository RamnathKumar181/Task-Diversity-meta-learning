#!/bin/bash
#SBATCH --job-name=reptile_omniglot_uniform_sampling
#SBATCH --output=../logs/reptile_omniglot_uniform_sampling.out
#SBATCH --error=../logs/reptile_omniglot_uniform_sampling.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

source ../env/bin/activate
cd .. && python -m src.main --exp_name reptile_omniglot_uniform_sampling --train --model reptile --runs 1 --folder ./data --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.33 --lr 0.00044 --batch-size 32 --num-workers 8 --num-epochs 100 --output-folder ./config/reptile_omniglot_uniform_sampling/
