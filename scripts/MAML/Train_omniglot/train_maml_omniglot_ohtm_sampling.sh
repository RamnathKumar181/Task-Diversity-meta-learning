#!/bin/bash
#SBATCH --job-name=maml_omniglot_ohtm_sampling
#SBATCH --output=../logs/maml_omniglot_ohtm_sampling.out
#SBATCH --error=../logs/maml_omniglot_ohtm_sampling.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

source ../env/bin/activate
cd .. && python -m src.main --exp_name maml_omniglot_ohtm_sampling --train --task_sampler ohtm --runs 1 ./data --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 150 --output-folder ./config/maml_omniglot_ohtm_sampling/
