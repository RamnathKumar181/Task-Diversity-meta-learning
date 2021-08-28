#!/bin/bash
#SBATCH --job-name=protonet_meta_dataset
#SBATCH --output=../logs/protonet_meta_dataset_%a.out
#SBATCH --error=../logs/protonet_meta_dataset_%a.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0

cp -r /network/projects/r/ramnath.kumar/meta_dataset/records $SLURM_TMPDIR
cp -r ../data/meta_dataset/records $SLURM_TMPDIR

source ../env/bin/activate
cd ..
python -m src.main --exp_name protonet_meta_dataset --train --model protonet --runs 1 --folder $SLURM_TMPDIR/records --meta-lr 0.001 --task_sampler $SLURM_ARRAY_TASK_ID --dataset meta_dataset --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 256 --num-workers 4 --num-epochs 100 --output-folder ./config/protonet_meta_dataset/$SLURM_ARRAY_TASK_ID/
rm -rf $SLURM_TMPDIR/records