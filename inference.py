from Batch import create_masks
from Beam import beam_search
from Models import get_model
from dataDistibutionCheck import checkdata
from Optim import CosineWithRestarts
from rand_gen import rand_gen_from_data_distribution, tokenlen_gen_from_data_distribution
from Process import *

from io import StringIO
from nltk.corpus import wordnet
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdDepictor, AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.autograd import Variable
from tqdm import tqdm
import argparse
import argparse
import dill as pickle
import joblib
import math
import moses
import numpy as np
import pdb
import re
import sys
import time
import torch
import torch.nn.functional as F


def gen_mol(cond, model, opt, toklen, z):
    model.eval()

    if opt.conds not in 'msl':
        cond = np.array(cond.split(','))
    cond = cond.reshape(1, -1)

    cond = opt.robust_scaler.transform(cond)
    cond = Variable(torch.Tensor(cond))

    sentence = beam_search(cond, model, opt.src_tok, opt.trg_tok, toklen, opt, z)
    return sentence


# def single_moses_step(toklen, idx, model, opt, conds):
#     print("Got here!")
#     z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
#     molecule_tmp = gen_mol(conds[idx], model, opt, toklen, z)
#     toklen_gen = molecule_tmp.count(" ") + 1
# 
#     molecule_tmp = ''.join(molecule_tmp).replace(" ", "")
#     
#     conds_trg = conds[idx]
#     toklen_check = toklen-opt.cond_dim # toklen - cond_dim: due to cond size
# 
#     m = Chem.MolFromSmiles(molecule_tmp)
#     if m is None:
#         val_check = 0
#         conds_rdkit = np.array([None, None, None])
#     else:
#         val_check = 1
#         conds_rdkit = np.array([Descriptors.MolLogP(m), Descriptors.TPSA(m), QED.qed(m)])
#     return molecule_tmp, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen


# def moses_benchmark(model, opt):
#     molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
#     print("\nGenerating molecules for MOSES benchmarking...")
#     n_samples = 30000
#     nBins = [1000, 1000, 1000]
# 
#     data = pd.read_csv(opt.load_traindata)
#     toklen_data = pd.read_csv(opt.load_toklendata)
# 
#     conds = rand_gen_from_data_distribution(data, size=n_samples, nBins=nBins)
#     toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples)
#     
#     model = model.to(opt.device)
# 
#     start = time.time()
#     for idx in tqdm(range(n_samples), desc="Generated:", total=n_samples):
#         toklen = int(toklen_data[idx]) + opt.cond_dim  # + cond_dim due to cond2enc
#         
#         z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
#         molecule_tmp = gen_mol(conds[idx], model, opt, toklen, z)
#         toklen_gen.append(molecule_tmp.count(" ")+1)
#         molecule_tmp = "".join(molecule_tmp).replace(" ", "")
#         
#         molecules.append(molecule_tmp)
#         conds_trg.append(conds[idx])
#         # toklen - cond_dim: due to cond dim
#         toklen_check.append(toklen-opt.cond_dim)
#         m = Chem.MolFromSmiles(molecule_tmp)
#         if m is None:
#             val_check.append(0)
#             conds_rdkit.append([None, None, None])
#         else:
#             val_check.append(1)
#             conds_rdkit.append(np.array([Descriptors.MolLogP(m), Descriptors.TPSA(m), QED.qed(m)]))
#         
#     np_conds_trg, np_conds_rdkit = np.array(conds_trg), np.array(conds_rdkit)
#     gen_list = pd.DataFrame(
#         {"mol": molecules, "val_check": val_check, "trg(logP)": np_conds_trg[:, 0], "trg(tPSA)": np_conds_trg[:, 1], "trg(QED)": np_conds_trg[:, 2], "rdkit(logP)": np_conds_rdkit[:, 0], "rdkit(tPSA)": np_conds_rdkit[:, 1], "rdkit(QED)": np_conds_rdkit[:, 2], "toklen": toklen_check, "toklen_gen": toklen_gen})
#     gen_list.to_csv('moses_bench2_lat={}_epo={}_k={}_{}.csv'.format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")), index=True)
# 
#     print("Please check the file: 'moses_bench2_lat={}_epo={}_k={}_{}.csv'".format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")))
# 
# 
# def ten_condition_test(model, opt):
#     molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
#     print("\nGenerating molecules for 10 condition sets...")
#     n_samples = 10
#     n_per_samples = 200
#     nBins = [1000, 1000, 1000]
# 
#     data = pd.read_csv(opt.load_traindata)
#     toklen_data = pd.read_csv(opt.load_toklendata)
# 
#     conds = rand_gen_from_data_distribution(data, size=n_samples, nBins=nBins)
#     toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples*n_per_samples)
# 
#     print("conds:\n", conds)
#     model = model.to(opt.device)
#     
#     start = time.time()
#     for idx in tqdm(range(n_samples), desc="Samples:", total=n_samples):
#         for i in tqdm(range(n_per_samples), desc="Sub-Samples:", total=n_per_samples):
#             toklen = int(toklen_data[idx*n_per_samples + i]) + opt.cond_dim  # + cond_dim due to cond2enc
#             z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
#             molecule_tmp = gen_mol(conds[idx], model, opt, toklen, z)
#             toklen_gen.append(molecule_tmp.count(" ") + 1)
#             molecule_tmp = "".join(molecule_tmp).replace(" ", "")
# 
#             molecules.append(molecule_tmp)
#             conds_trg.append(conds[idx])
# 
#             toklen_check.append(toklen-opt.cond_dim) # toklen - cond_dim: due to cond size
#             m = Chem.MolFromSmiles(molecule_tmp)
#             if m is None:
#                 val_check.append(0)
#                 conds_rdkit.append([None, None, None])
#             else:
#                 val_check.append(1)
#                 conds_rdkit.append(np.array([Descriptors.MolLogP(m), Descriptors.TPSA(m), QED.qed(m)]))
# 
#     np_conds_trg, np_conds_rdkit = np.array(conds_trg), np.array(conds_rdkit)
#     gen_list = pd.DataFrame(
#         {"set_idx": idx, "mol": molecules, "val_check": val_check, "trg(logP)": np_conds_trg[:, 0], "trg(tPSA)": np_conds_trg[:, 1], "trg(QED)": np_conds_trg[:, 2], "rdkit(logP)": np_conds_rdkit[:, 0], "rdkit(tPSA)": np_conds_rdkit[:, 1], "rdkit(QED)": np_conds_rdkit[:, 2], "toklen": toklen_check, "toklen_gen": toklen_gen})
#     gen_list.to_csv('moses_bench2_10conds_lat={}_epo={}_k={}_{}.csv'.format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")), index=True)
# 
#     print("Please check the file: 'moses_bench2_10conds_lat={}_epo={}_k={}_{}.csv'".format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")))


def single_molecule(model, opt):
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
    conds = opt.conds.split(';')
    toklen_data = pd.read_csv(opt.load_toklendata)
    toklen= int(tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max() - toklen_data.min()), size=1)) + opt.cond_dim  # + cond_dim due to cond2enc

    z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))

    model = model.to(opt.device)

    for cond in conds:
        molecules.append(gen_mol(cond, model, opt, toklen, z))
    toklen_gen = molecules[0].count(" ") + 1
    molecules = ''.join(molecules).replace(" ", "")
    m = Chem.MolFromSmiles(molecules)
    target_cond = conds[0].split(',')

    validity = "Invalid" if m is None else "Valid"
    print(f"   --[{validity}]:", molecules)
    print("   --Target:", end="")
    for label, value in zip(opt.cond_labels, target_cond):
        print(f" {label}={value},", end="")
    print(f" LatentToklen={toklen-opt.cond_dim}") #toklen - cond_dim: due to cond dim


def get_attention(model, opt):
    # put model and data onto the same device
    model = model.to(opt.device)
    model.train()
    for batch in opt.train:
        src = batch.src.transpose(0, 1).to(opt.device)
        trg = batch.trg.transpose(0, 1).to(opt.device)
        trg_input = trg[:, :-1].to(opt.device)
        cond = torch.stack([vars(batch)[label] for label in opt.cond_labels]).transpose(0,1).to(opt.device)
        src_mask, trg_mask = create_masks(src, trg_input, cond, opt)
        src_mask = src_mask.to(opt.device)
        trg_mask = trg_mask.to(opt.device)
        preds_prop, preds_mol, mu, log_var, z, attentions = model(src, trg_input, cond, src_mask, trg_mask)
        
        if len(src[0]) < 18: # Get small chemicals
            break
        # samples = [[opt.src_tok.vocab.itos[i] for i in item] for item in src]
        # for n, sample in enumerate(samples):
            # for i, val in enumerate(sample):
                # if val == '.':
                    # period = i
            # # Need 0-period to only go to 0-period
            # eps = 0 # + is a connection, - is no connection
            # dim = len(sample)
            # attention = [a[:, :, :dim, :dim] for a in attentions]
# 
            # for layer in range(len(attention)):
                # for head in range(len(attention[0][0])):
                    # if torch.any(attention[layer][n, head, 0:period, period+1:len(sample)] > eps):
                        # continue
                    # if torch.any(attention[layer][n, head, period+1:len(sample), 0:period] > eps):
                        # continue
                    # 
                    # with open("figures/attention_scores.pkl", "wb") as f:
                        # pickle.dump(attentions, f)
                # 
                    # with open("figures/attention_strings.pkl", "wb") as f:
                        # pickle.dump([[opt.src_tok.vocab.itos[i] for i in item] for item in src], f)
# 
                    # print(layer, head)
                    # break

    print("Done")
        
    with open("figures/attention_scores.pkl", "wb") as f:
        pickle.dump(attentions, f)

    with open("figures/attention_strings.pkl", "wb") as f:
        pickle.dump([[opt.src_tok.vocab.itos[i] for i in item] for item in src], f)

    for item in src:
        #print(len(item))
        print([opt.src_tok.vocab.itos[i] for i in item])
        #print(''.join([opt.src_tok.vocab.itos[i] for i in item]))


def get_program_arguments():
    """
    Gets the program arguments, setting the defaults if not given
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', type=str, default="weights")
    parser.add_argument('-load_traindata', type=str, default="data/IL/train.csv")
    parser.add_argument('-data', type=str, default='data/IL/train.csv')
    parser.add_argument('-data_te', type=str, default='data/IL/test.csv')
    parser.add_argument('-load_toklendata', type=str, default='toklen_list.csv')
    parser.add_argument('-k', type=int, default=4)
    
    parser.add_argument('-lang_format', type=str, default='SMILES')
    parser.add_argument('-max_strlen', type=int, default=80) #max 80
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-use_cond2dec', type=bool, default=False)
    parser.add_argument('-use_cond2lat', type=bool, default=True)
    parser.add_argument('-cond_dim', type=int, default=5)
    parser.add_argument('-latent_dim', type=int, default=128)
    parser.add_argument('-epochs', type=int, default=0)
    parser.add_argument('-batchsize', type=int, default=256)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-lr_beta1', type=int, default=0.9)
    parser.add_argument('-lr_beta2', type=int, default=0.98)
    
    parser.add_argument('-print_model', type=bool, default=False)
    parser.add_argument('-verbose', type=bool, default=False)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    return parser.parse_args()


def main():
    opt = get_program_arguments()
    opt.cond_labels = ["Temperature", "Pressure", "DynViscosity", "Density", "ElecConductivity"]

    print("Number of GPUS to use: ", torch.cuda.device_count())
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert opt.k > 0
    assert opt.max_strlen > 10

    opt.robust_scaler = joblib.load(opt.load_weights + '/scaler.pkl')
    create_fields(opt)
    model = get_model(opt, len(opt.src_tok.vocab), len(opt.trg_tok.vocab), return_encoder_attention=True)
    prepare_dataset(opt)

    min_max = checkdata(opt)

    while True:
        print("\nEnter properties to generate molecules (refer the pop-up data distribution)")
        print("* Current trained ranges:")
        for label, (min_, max_) in zip(opt.cond_labels, min_max):
            print("\t--", label, ":", min_, "~", max_)
        opt.conds = input("* Enter the properties (Or type m: MOSES benchmark, s: 10-Condition set test, q: quit):")

        if opt.conds=="q":
            break
        elif opt.conds=="a":
            get_attention(model, opt)
        #elif opt.conds == 'm':
        #    moses_benchmark(model, opt)
        #elif opt.conds == 's':
        #    ten_condition_test(model, opt)
        else:
            single_molecule(model, opt)


if __name__ == '__main__':
    main()
