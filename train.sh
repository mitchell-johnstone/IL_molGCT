#!/bin/bash

#SBATCH --job-name=training
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=16
#SBATCH --time=1-0:0
# time format: <days>-<hours>:<minutes>

# set up the environment
source /home/johnstonem/anaconda3/bin/activate molgct

# command
# python3 test_frcnn.py -p ../data/test_data --network vgg --write --load models/vgg/boosted.hdf5
# python train.py -epochs 3 -batchsize 1024 -imp_test False -print_model True
python train.py -imp_test False -epochs 4 -batchsize 1024 -print_model True -load_weights weights 

# deactivate env
conda deactivate

