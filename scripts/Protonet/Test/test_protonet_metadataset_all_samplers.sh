#!/bin/bash
#SBATCH --job-name=test_protonet_meta_dataset
#SBATCH --output=../logs/test_protonet_meta_dataset_%a.out
#SBATCH --error=../logs/test_protonet_meta_dataset_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --array=0-7

source ../env/bin/activate
ulimit -n 50000

cp -r /network/projects/r/ramnath.kumar/meta_dataset/records $SLURM_TMPDIR
cp -r ../data/meta_dataset/records $SLURM_TMPDIR

echo "Finished moving data"

python -m src.main --exp_name test_protonet_meta_dataset --log-test-tasks --model protonet --runs 1 --folder $SLURM_TMPDIR/records --meta-lr 0.001 --task_sampler $SLURM_ARRAY_TASK_ID --dataset meta_dataset --num-ways 5 --num-shots 1 --use-cuda --step-size 0.4 --batch-size 32 --num-workers 0 --num-epochs 100 --output-folder ./config/protonet_meta_dataset/$SLURM_ARRAY_TASK_ID/

rm -rf $SLURM_TMPDIR/records
