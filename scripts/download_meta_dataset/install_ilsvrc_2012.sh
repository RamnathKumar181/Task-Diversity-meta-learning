#!/bin/bash
#SBATCH --job-name=install_ilsvrc_2021
#SBATCH --output=../logs/install_ilsvrc_2021_%a.out
#SBATCH --error=../logs/install_ilsvrc_2021_%a.err
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G

cd $SLURM_TMPDIR/meta_dataset/
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
cp ILSVRC2012_img_train.tar /network/projects/r/ramnath.kumar/
