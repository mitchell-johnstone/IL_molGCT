import argparse
import time
import torch
from tqdm import tqdm
import numpy as np
from Models import get_model
from Process import *
import torch.nn.functional as F
from torch.nn import DataParallel
from Optim import CosineWithRestarts
from Batch import create_masks
import joblib
import dill as pickle
import pandas as pd
import csv
import timeit
from hyperopt import fmin, tpe, space_eval, hp
from functools import partial


def KLAnnealer(opt, epoch):
    beta = opt.KLA_ini_beta + opt.KLA_inc_beta * ((epoch + 1) - opt.KLA_beg_epoch)
    return beta


def loss_function(opt, beta, preds_prop, preds_mol, ys_cond, ys_mol, mu, log_var):
    RCE_mol = F.cross_entropy(preds_mol.contiguous().view(-1, preds_mol.size(-1)), ys_mol, ignore_index=opt.trg_pad, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    if opt.use_cond2dec == True:
        RCE_prop = F.mse_loss(preds_prop, ys_cond, reduction='sum')
        loss = RCE_mol + RCE_prop + beta * KLD
    else:
        RCE_prop = torch.zeros(1)
        loss = RCE_mol + beta * KLD
    return loss, RCE_mol, RCE_prop, KLD

def run_model(params, opt):

    opt.lr_scheduler = params["lr_scheduler"]
    opt.lr_WarmUpSteps = params["lr_WarmUpSteps"]
    opt.lr = params["lr"]
    opt.lr_beta1 = params["lr_beta1"]
    opt.lr_beta2 = params["lr_beta2"]
    opt.use_KLA = params["use_KLA"]
    opt.KLA_ini_beta = params["KLA_ini_beta"]
    opt.KLA_inc_beta = params["KLA_inc_beta"]
    opt.latent_dim = params["latent_dim"]
    opt.d_model = params["d_model"]
    opt.n_layers = params["n_layers"]
    opt.heads = params["heads"]
    opt.dropout = params["dropout"]
    opt.batchsize = params["batchsize"]

    # Get the model
    model = get_model(opt, len(opt.src_tok.vocab), len(opt.trg_tok.vocab))

    # Set up the optimizer and scheduler
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.lr_beta1, opt.lr_beta2), eps=opt.lr_eps)
    if opt.lr_scheduler == "SGDR":
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if torch.cuda.is_available():
        # Parallelize the model to use multiple GPUS
        model = DataParallel(model)

    # put model and data onto the same device
    model = model.to(opt.device)

    if opt.checkpoint > 0:
        cptime = time.time()

    beta = 0
    current_step = 0
    loss = []
    for epoch in range(opt.epochs):

        total_loss,    RCE_mol_loss,    RCE_prop_loss,    KLD_loss    = 0, 0, 0, 0
        total_loss_te, RCE_mol_loss_te, RCE_prop_loss_te, KLD_loss_te = 0, 0, 0, 0

        # KL annealing
        if opt.use_KLA == True:
            if epoch + 1 >= opt.KLA_beg_epoch and beta < opt.KLA_max_beta:
                beta = KLAnnealer(opt, epoch)
        else:
            beta = 1

        # Training
        model.train()

        for batch in opt.train:
            current_step += 1
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            trg_input = trg[:, :-1]

            cond = torch.stack([vars(batch)[label] for label in opt.cond_labels]).transpose(0,1)
            # cond = torch.stack([batch.Temperature, batch.Pressure, batch.DynViscosity, \
            #                     batch.Density, batch.ElecConductivity]).transpose(0, 1)

            src_mask, trg_mask = create_masks(src, trg_input, cond, opt)

            preds_prop, preds_mol, mu, log_var, z = model(src, trg_input, cond, src_mask, trg_mask)
            ys_mol = trg[:, 1:].contiguous().view(-1).to(opt.device)
            ys_cond = torch.unsqueeze(cond, 2).contiguous().view(-1, opt.cond_dim, 1)

            opt.optimizer.zero_grad()

            loss, RCE_mol, RCE_prop, KLD = loss_function(opt, beta, preds_prop, preds_mol, ys_cond, ys_mol, mu, log_var)

            loss.backward()
            opt.optimizer.step()

            if opt.lr_scheduler == "SGDR":
                opt.sched.step()
            elif opt.lr_scheduler == "WarmUpDefault":
                head = np.float64(np.power(np.float64(current_step), -0.5))
                tail = np.float64(current_step) * np.power(np.float64(opt.lr_WarmUpSteps), -1.5)
                lr = np.float64(np.power(np.float64(opt.d_model), -0.5)) * min(head, tail)
                for param_group in opt.optimizer.param_groups:
                    param_group['lr'] = lr

        # Test
        model.eval()
        with torch.no_grad():
            for batch in opt.test:
                src = batch.src.transpose(0, 1)
                trg = batch.trg.transpose(0, 1)
                trg_input = trg[:, :-1]

                cond = torch.stack([vars(batch)[label] for label in opt.cond_labels]).transpose(0,1)
                # cond = torch.stack([batch.Temperature, batch.Pressure, batch.DynViscosity, \
                #                     batch.Density, batch.ElecConductivity]).transpose(0, 1)

                src_mask, trg_mask = create_masks(src, trg_input, cond, opt)
                preds_prop, preds_mol, mu, log_var, z = model(src, trg_input, cond, src_mask, trg_mask)

                ys_mol = trg[:, 1:].contiguous().view(-1).to(opt.device)
                ys_cond = torch.unsqueeze(cond, 2).contiguous().view(-1, opt.cond_dim, 1)

                loss_te, RCE_mol_te, RCE_prop_te, KLD_te = loss_function(opt, beta, preds_prop, preds_mol, ys_cond, ys_mol, mu, log_var)

                total_loss_te += loss_te.item()
        
        loss.append(total_loss_te / len(opt.test.dataset))
    return loss[-1]


def get_program_arguments():
    """
    Gets the program arguments, setting the defaults if not given
    """
    parser = argparse.ArgumentParser()
    # Data settings
    parser.add_argument('-imp_test', type=bool, default=True)
    parser.add_argument('-data', type=str, default='data/IL/train.csv')
    parser.add_argument('-data_te', type=str, default='data/IL/test.csv')
    parser.add_argument('-lang_format', type=str, default='SMILES')
    calProp = not os.path.isfile("data/moses/prop_temp.csv") or not os.path.isfile("data/moses/prop_temp_te.csv")
    parser.add_argument('-calProp', type=bool, default=calProp) #if prop_temp.csv and prop_temp_te.csv exist, set False
    
    parser.add_argument('-tokenizer', type=str, default="smilespe")

    # Learning hyperparameters
    parser.add_argument('-epochs', type=int, default=5)
    # parser.add_argument('-lr_scheduler', type=str, default="SGDR", help="WarmUpDefault, SGDR")
    parser.add_argument('-lr_scheduler', type=str, default="WarmUpDefault", help="WarmUpDefault, SGDR")
    parser.add_argument('-lr_WarmUpSteps', type=int, default=8000, help="only for WarmUpDefault")
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-lr_beta1', type=float, default=0.9)
    parser.add_argument('-lr_beta2', type=float, default=0.98)
    parser.add_argument('-lr_eps', type=float, default=1e-9)

    # KL Annealing
    parser.add_argument('-use_KLA', type=bool, default=True)
    parser.add_argument('-KLA_ini_beta', type=float, default=0.02)
    parser.add_argument('-KLA_inc_beta', type=float, default=0.02)
    parser.add_argument('-KLA_max_beta', type=float, default=1.0)
    parser.add_argument('-KLA_beg_epoch', type=int, default=1) #KL annealing begin

    # Network sturucture
    parser.add_argument('-use_cond2dec', type=bool, default=False)
    parser.add_argument('-use_cond2lat', type=bool, default=True)
    parser.add_argument('-latent_dim', type=int, default=128)
    parser.add_argument('-cond_dim', type=int, default=4)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.3)
    parser.add_argument('-batchsize', type=int, default=256)
    # parser.add_argument('-batchsize', type=int, default=1024*8)
    parser.add_argument('-max_strlen', type=int, default=80)  # max 80

    # History
    parser.add_argument('-verbose', type=bool, default=False)
    parser.add_argument('-save_folder_name', type=str, default='saved_model')
    parser.add_argument('-print_model', type=bool, default=False)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)

    return parser.parse_args()


def main():
    """
    main program. Triggers the following subprocesses
    - get program arguments
    - loading the data
    - making the model
    - training the model
    - saving the model
    """
    opt = get_program_arguments()
    opt.load_weights = None

    opt.cond_labels = ["Temperature", "DynViscosity", "Density", "ElecConductivity"]
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare the dataset
    prepare_dataset(opt)

    # Define space for bayesian optimization
    space = {
        "lr_scheduler": hp.choice("lr_scheduler", ["WarmUpDefault", "SGDR"]),
        "lr_WarmUpSteps": hp.choice("lr_WarmUpSteps", [6000, 8000, 10000]),
        "lr": hp.choice("lr", [1e-4, 1e-5, 1e-6, 1e-7]),
        "lr_beta1": hp.uniform("lr_beta1", 0.9, 0.99),
        "lr_beta2": hp.uniform("lr_beta2", 0.9, 0.99),
        "use_KLA": hp.choice("use_KLA", [True, False]),
        "KLA_ini_beta": hp.uniform("KLA_ini_beta", 0.01, 0.05),
        "KLA_inc_beta": hp.uniform("KLA_inc_beta", 0.01, 0.05),
        "latent_dim": hp.choice("latent_dim", [2**i for i in range(6,9)]),
        "d_model": hp.choice("d_model", [2**i for i in range(8,11)]),
        "n_layers": hp.choice("n_layers", [4,5,6,7,8,9,10]),
        "heads": hp.choice("heads", [4, 8, 16]),
        "dropout": hp.uniform("dropout", 0, 1),
        "batchsize": hp.choice("batchsize", [2**i for i in range(6,9)])
    }
    run = partial(run_model, opt=opt)
    best = fmin(run,
        space=space,
        algo=tpe.suggest,
        max_evals=50)
    print("Best Params:",hyperopt.space_eval(space, best))


if __name__ == "__main__":
    main()