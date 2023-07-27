#!/bin/bash

#SBATCH --job-name=training
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=72
#SBATCH --cpus-per-gpu=30
#SBATCH --time=1-0:0
# time format: <days>-<hours>:<minutes>

# set up the environment
source /home/johnstonem/anaconda3/bin/activate ilmolgct

tokenizer="atom"
lr_scheduler="WarmUpDefault"
lr_WarmUpSteps="8000"
lr="0.0001"
lr_beta1="0.9"
lr_beta2="0.95"
use_KLA="True"
KLA_ini_beta="0.015"
KLA_inc_beta="0.015"
latent_dim="128"
d_model="256"
n_layers="6"
heads="8"
dropout="0.2"
batchsize="256"
epochs="50"

######### full results
python train.py -tokenizer ${tokenizer} -epochs ${epochs} -batchsize ${batchsize} -lr ${lr} -lr_beta1 ${lr_beta1} -lr_beta2 ${lr_beta1} -use_KLA ${use_KLA} -KLA_ini_beta ${KLA_ini_beta} -KLA_inc_beta ${KLA_inc_beta} -latent_dim ${latent_dim} -d_model ${d_model} -n_layers ${n_layers} -heads ${heads} -dropout ${dropout}
echo "s" | python inference.py -load_weights "weights/${tokenizer}_lr=${lr}_drop=${dropout}_e=${epochs}_b=${batchsize}_nl=${n_layers}_h=${heads}_d=${d_model}_lat=${latent_dim}" -tokenizer ${tokenizer} -epochs ${epochs} -batchsize ${batchsize} -latent_dim ${latent_dim} -d_model ${d_model} -n_layers ${n_layers} -heads ${heads} -dropout ${dropout}
echo "m" | python inference.py -load_weights "weights/${tokenizer}_lr=${lr}_drop=${dropout}_e=${epochs}_b=${batchsize}_nl=${n_layers}_h=${heads}_d=${d_model}_lat=${latent_dim}" -tokenizer ${tokenizer} -epochs ${epochs} -batchsize ${batchsize} -latent_dim ${latent_dim} -d_model ${d_model} -n_layers ${n_layers} -heads ${heads} -dropout ${dropout}

# deactivate env
conda deactivate

