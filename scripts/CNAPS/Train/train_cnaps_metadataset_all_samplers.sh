#!/bin/bash
#SBATCH --job-name=cnaps_meta_dataset
#SBATCH --output=../logs/cnaps_meta_dataset_%a.out
#SBATCH --error=../logs/cnaps_meta_dataset_%a.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0

source ../env/bin/activate

cp -r /network/projects/r/ramnath.kumar/meta_dataset/records $SLURM_TMPDIR
cp -r ../data/meta_dataset/records $SLURM_TMPDIR

cd .. && python -m src.main --exp_name cnaps_meta_dataset --train --model cnaps --runs 1 --folder $SLURM_TMPDIR/records --task_sampler $SLURM_ARRAY_TASK_ID --dataset meta_dataset --meta-lr 0.001 --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 0 --num-epochs 10 --output-folder ./config/cnaps_meta_dataset/$SLURM_ARRAY_TASK_ID/

rm -rf $SLURM_TMPDIR/records
