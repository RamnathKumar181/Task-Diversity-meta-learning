#!/bin/bash
#SBATCH --job-name=cnaps_meta
#SBATCH --output=../logs/cnaps_meta.out
#SBATCH --error=../logs/cnaps_meta.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

source ../env/bin/activate
ulimit -n 50000

cp -r /network/projects/r/ramnath.kumar/meta_dataset/records $SLURM_TMPDIR
cp -r ../data/meta_dataset/records $SLURM_TMPDIR

echo "Finished moving data"
cd ../cnaps/src

python run_cnaps.py --data_path $SLURM_TMPDIR/records

rm -rf $SLURM_TMPDIR/records
