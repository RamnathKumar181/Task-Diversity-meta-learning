#!/bin/bash
#SBATCH --job-name=zip_meta_dataset
#SBATCH --output=../logs/zip_meta_dataset_%a.out
#SBATCH --error=../logs/zip_meta_dataset_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G

cp -r /network/projects/r/ramnath.kumar/meta_dataset/records $SLURM_TMPDIR
cp -r ../data/meta_dataset/records $SLURM_TMPDIR

tar -zcvf $SLURM_TMPDIR/records.tar.gz $SLURM_TMPDIR/records

cp -r $SLURM_TMPDIR/records.tar.gz $HOME/scratch-new/
rm -rf $SLURM_TMPDIR/records
rm -rf $SLURM_TMPDIR/records.tar.gz

echo "Finished Compressing file"
