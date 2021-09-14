#!/bin/bash
#SBATCH --job-name=reptile_miniimagenet
#SBATCH --output=../logs/reptile_miniimagenet_%a.out
#SBATCH --error=../logs/reptile_miniimagenet_%a.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0

source ../env/bin/activate
cd ../GBML && python main.py --alg=Reptile
