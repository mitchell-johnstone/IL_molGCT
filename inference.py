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

#def get_synonym(word, SRC):
#    syns = wordnet.synsets(word)
#    for s in syns:
#        for l in s.lemmas():
#            if SRC.vocab.stoi[l.name()] != 0:
#                return SRC.vocab.stoi[l.name()]
#            
#    return 0

def gen_mol(cond, model, toklen, opt, z):
    model.eval()

    if opt.conds not in 'msl':
        cond = np.array(cond.split(',')[:-1])
    cond = cond.reshape(1, -1)

    cond = opt.robust_scaler.transform(cond)
    cond = Variable(torch.Tensor(cond))

    sentence = beam_search(cond, model, opt.src_tok, opt.trg_tok, toklen, opt, z)
    return sentence


# def single_moses_step(toklen, idx, model, opt, conds):
#     print("Got here!")
#     z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
#     molecule_tmp = gen_mol(conds[idx], model, toklen, opt, z)
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


def ten_condition_test(model, opt):
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
    print("\nGenerating molecules for 10 condition sets...")
    n_samples = 10
    n_per_samples = 200
    nBins = [1000, 1000, 1000]

    data = pd.read_csv(opt.load_traindata)
    toklen_data = pd.read_csv(opt.load_toklendata)

    conds = rand_gen_from_data_distribution(data, size=n_samples, nBins=nBins)
    toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples*n_per_samples)

    print("conds:\n", conds)
    model = model.to(opt.device)
    
    start = time.time()
    for idx in tqdm(range(n_samples), desc="Samples:", total=n_samples):
        for i in tqdm(range(n_per_samples), desc="Sub-Samples:", total=n_per_samples):
            toklen = int(toklen_data[idx*n_per_samples + i]) + opt.cond_dim  # + cond_dim due to cond2enc
            z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
            molecule_tmp = gen_mol(conds[idx], model, toklen, opt, z)
            toklen_gen.append(molecule_tmp.count(" ") + 1)
            molecule_tmp = ''.join(molecule_tmp).replace(" ", "")

            molecules.append(molecule_tmp)
            conds_trg.append(conds[idx])

            toklen_check.append(toklen-opt.cond_dim) # toklen - cond_dim: due to cond size
            m = Chem.MolFromSmiles(molecule_tmp)
            if m is None:
                val_check.append(0)
                conds_rdkit.append([None, None, None])
            else:
                val_check.append(1)
                conds_rdkit.append(np.array([Descriptors.MolLogP(m), Descriptors.TPSA(m), QED.qed(m)]))

            if (idx*n_per_samples+i+1) % 200 == 0:
                np_conds_trg, np_conds_rdkit = np.array(conds_trg), np.array(conds_rdkit)
                gen_list = pd.DataFrame(
                    {"set_idx": idx, "mol": molecules, "val_check": val_check, "trg(logP)": np_conds_trg[:, 0], "trg(tPSA)": np_conds_trg[:, 1], "trg(QED)": np_conds_trg[:, 2], "rdkit(logP)": np_conds_rdkit[:, 0], "rdkit(tPSA)": np_conds_rdkit[:, 1], "rdkit(QED)": np_conds_rdkit[:, 2], "toklen": toklen_check, "toklen_gen": toklen_gen})
                gen_list.to_csv('moses_bench2_10conds_lat={}_epo={}_k={}_{}.csv'.format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")), index=True)

    print("Please check the file: 'moses_bench2_10conds_lat={}_epo={}_k={}_{}.csv'".format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")))


def moses_benchmark(model, opt):
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
    print("\nGenerating molecules for MOSES benchmarking...")
    n_samples = 30000
    nBins = [1000, 1000, 1000]

    data = pd.read_csv(opt.load_traindata)
    toklen_data = pd.read_csv(opt.load_toklendata)

    conds = rand_gen_from_data_distribution(data, size=n_samples, nBins=nBins)
    toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples)
    
    model = model.to(opt.device)

    start = time.time()
    for idx in tqdm(range(n_samples), desc="Generated:", total=n_samples):
        toklen = int(toklen_data[idx]) + opt.cond_dim  # + cond_dim due to cond2enc
        
        z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
        molecule_tmp = gen_mol(conds[idx], model, toklen, opt, z)
        toklen_gen.append(molecule_tmp.count(' ')+1)
        molecule_tmp = ''.join(molecule_tmp).replace(" ", "")
        
        molecules.append(molecule_tmp)
        conds_trg.append(conds[idx])
        # toklen - cond_dim: due to cond dim
        toklen_check.append(toklen-opt.cond_dim)
        m = Chem.MolFromSmiles(molecule_tmp)
        if m is None:
            val_check.append(0)
            conds_rdkit.append([None, None, None])
        else:
            val_check.append(1)
            conds_rdkit.append(np.array([Descriptors.MolLogP(m), Descriptors.TPSA(m), QED.qed(m)]))
        
        if (idx+1) % 2000 == 0:
            np_conds_trg, np_conds_rdkit = np.array(conds_trg), np.array(conds_rdkit)
            gen_list = pd.DataFrame(
                {"mol": molecules, "val_check": val_check, "trg(logP)": np_conds_trg[:, 0], "trg(tPSA)": np_conds_trg[:, 1], "trg(QED)": np_conds_trg[:, 2], "rdkit(logP)": np_conds_rdkit[:, 0], "rdkit(tPSA)": np_conds_rdkit[:, 1], "rdkit(QED)": np_conds_rdkit[:, 2], "toklen": toklen_check, "toklen_gen": toklen_gen})
            gen_list.to_csv('moses_bench2_lat={}_epo={}_k={}_{}.csv'.format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")), index=True)

    print("Please check the file: 'moses_bench2_lat={}_epo={}_k={}_{}.csv'".format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")))


def single_molecule(model, opt):
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
    conds = opt.conds.split(';')
    toklen_data = pd.read_csv(opt.load_toklendata)
    toklen= int(tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max() - toklen_data.min()), size=1)) + opt.cond_dim  # + cond_dim due to cond2enc

    z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))

    model = model.to(opt.device)

    for cond in conds:
        molecules.append(gen_mol(cond + ',', model, toklen, opt, z))
    toklen_gen = molecules[0].count(" ") + 1
    molecules = ''.join(molecules).replace(" ", "")
    m = Chem.MolFromSmiles(molecules)
    target_cond = conds[0].split(',')
    if m is None:
        #toklen - cond_dim: due to cond dim
        print("   --[Invalid]: {}".format(molecules))
        print("   --Target: logP={}, tPSA={}, QED={}, LatentToklen={}".format(target_cond[0], target_cond[1], target_cond[2], toklen-opt.cond_dim))
    else:
        logP_v, tPSA_v, QED_v = Descriptors.MolLogP(m), Descriptors.TPSA(m), QED.qed(m)
        print("   --[Valid]: {}".format(molecules))
        print("   --Target: logP={}, tPSA={}, QED={}, LatentToklen={}".format(target_cond[0], target_cond[1], target_cond[2], toklen-opt.cond_dim))
        print("   --From RDKit: logP={:,.4f}, tPSA={:,.4f}, QED={:,.4f}, GenToklen={}".format(logP_v, tPSA_v, QED_v, toklen_gen))


def get_program_arguments():
    """
    Gets the program arguments, setting the defaults if not given
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', type=str, default="weights")
    parser.add_argument('-load_traindata', type=str, default="data/moses/prop_temp.csv")
    parser.add_argument('-load_toklendata', type=str, default='toklen_list.csv')
    parser.add_argument('-k', type=int, default=4)
    
    parser.add_argument('-lang_format', type=str, default='SMILES')
    parser.add_argument('-max_strlen', type=int, default=80) #max 80
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-use_cond2dec', type=bool, default=False)
    parser.add_argument('-use_cond2lat', type=bool, default=True)
    parser.add_argument('-cond_dim', type=int, default=3)
    parser.add_argument('-latent_dim', type=int, default=128)
    parser.add_argument('-epochs', type=int, default=1111111111111)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-lr_beta1', type=int, default=0.9)
    parser.add_argument('-lr_beta2', type=int, default=0.98)
    
    parser.add_argument('-print_model', type=bool, default=False)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    return parser.parse_args()


def main():
    opt = get_program_arguments()

    print("Number of GPUS to use: ", torch.cuda.device_count())
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert opt.k > 0
    assert opt.max_strlen > 10

    opt.robust_scaler = joblib.load(opt.load_weights + '/scaler.pkl')
    create_fields(opt)
    model = get_model(opt, len(opt.src_tok.vocab), len(opt.trg_tok.vocab))

    opt.max_logP, opt.min_logP, opt.max_tPSA, opt.min_tPSA, opt.max_QED, opt.min_QED = checkdata(opt.load_traindata)

    while True:
        opt.conds =input("\nEnter logP, tPSA, QED to generate molecules (refer the pop-up data distribution)\
        \n* logP: {:.2f} ~ {:.2f}; tPSA: {:.2f} ~ {:.2f}; QED: {:.2f} ~ {:.2f} is recommended.\
        \n* Typing sample: 2.2, 85.3, 0.8\n* Enter the properties (Or type m: MOSES benchmark, s: 10-Condition set test, q: quit):".format(opt.min_logP, opt.max_logP, opt.min_tPSA, opt.max_tPSA, opt.min_QED, opt.max_QED))

        if opt.conds=="q":
            break
        if opt.conds == 'm':
            moses_benchmark(model, opt)
        elif opt.conds == 's':
            ten_condition_test(model, opt)
        else:
            single_molecule(model, opt)


if __name__ == '__main__':
    main()
