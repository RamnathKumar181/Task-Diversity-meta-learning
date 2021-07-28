#!/bin/bash
#SBATCH --job-name=matching_networks_omniglot_uniform_sampling
#SBATCH --output=../logs/matching_networks_omniglot_uniform_sampling.out
#SBATCH --error=../logs/matching_networks_omniglot_uniform_sampling.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

source ../env/bin/activate
cd .. && python -m src.main --train --model matching_networks --runs 1 ./data --dataset omniglot --meta-lr 0.001 --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 100 --output-folder ./config/matching_networks_omniglot_uniform_sampling/
python -m src.main ./data --output-folder ./config/matching_networks_omniglot_uniform_sampling/