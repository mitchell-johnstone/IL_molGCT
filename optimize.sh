#!/bin/bash

#SBATCH --job-name=optimize
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --cpus-per-gpu=15
#SBATCH --time=1-0:0
# time format: <days>-<hours>:<minutes>

# set up the environment
source /home/johnstonem/anaconda3/bin/activate ilmolgct

# command
python optimize.py

# deactivate env
conda deactivate