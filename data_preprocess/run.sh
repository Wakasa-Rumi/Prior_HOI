#!/bin/bash

#PBS -N prior_hoi
#PBS -l select=1:ncpus=12:mem=24gb:ngpus=1
#PBS -l walltime=01:00:00
#PBS -q workq

date
hostname
source /home/yiyao/miniconda3/etc/profile.d/conda.sh
conda activate pytorch3d
cd /home/yiyao/HOI/HOI/ho
python use_dexycb.py
exit