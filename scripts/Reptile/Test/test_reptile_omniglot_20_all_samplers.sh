#!/bin/bash
#SBATCH --job-name=test_reptile_omniglot_20
#SBATCH --output=../logs/test_reptile_omniglot_20_%a.out
#SBATCH --error=../logs/test_reptile_omniglot_20_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --array=0-7

source ../env/bin/activate
cd .. && python -m src.main --exp_name test_reptile_omniglot_20 --log-test-tasks --task_sampler $SLURM_ARRAY_TASK_ID --model reptile --runs 1 --image-size 28 --folder $SLURM_TMPDIR --dataset omniglot --num-ways 20 --num-shots 1 --num-steps 10 --use-cuda --step-size 0.33 --lr 0.01 --batch-size 32 --num-workers 4 --num-epochs 150 --meta-lr 0.0005 --output-folder ./config/reptile_omniglot_20/$SLURM_ARRAY_TASK_ID/
