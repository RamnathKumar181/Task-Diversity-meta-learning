#!/bin/bash
#SBATCH --job-name=test_metaoptnet_omniglot
#SBATCH --output=../logs/test_metaoptnet_omniglot_%a.out
#SBATCH --error=../logs/test_metaoptnet_omniglot_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0-7

source ../env/bin/activate
cd ..
python -m src.main --exp_name test_metaoptnet_omniglot --log-test-tasks --model metaoptnet --runs 1 --folder $SLURM_TMPDIR/data --meta-lr 0.1 --task_sampler $SLURM_ARRAY_TASK_ID --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 8 --num-epochs 60 --output-folder ./config/metaoptnet_omniglot/$SLURM_ARRAY_TASK_ID/
