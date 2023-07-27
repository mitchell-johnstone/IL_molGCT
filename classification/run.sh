#!/bin/bash

#SBATCH --job-name=training
#SBATCH --output=slurm-%j.out
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=72
#SBATCH --cpus-per-gpu=30
#SBATCH --time=1-0:0
# time format: <days>-<hours>:<minutes>

# set up the environment
source /home/johnstonem/anaconda3/bin/activate il_molgct

python -m pip install STOUT-pypi
python testingSTOUT.py

# deactivate env
conda deactivate

