import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp


def calcForDataset(dataset, filename):
    """
    Calculate the properties and save the properties for one dataset
    """
    count = 0
    logP_list, tPSA_list, QED_list = [], [], []

    for smi in tqdm(dataset, desc="Properties Calculated: "):
        mol = Chem.MolFromSmiles(smi)
        logP_list.append(Descriptors.MolLogP(mol)), tPSA_list.append(Descriptors.TPSA(mol)), QED_list.append(QED.qed(mol))

    prop_df = pd.DataFrame({'logP': logP_list, 'tPSA': tPSA_list, 'QED': QED_list})
    prop_df.to_csv(filename, index=False)


def calcForSMILES(smi):
    """
    Calculate the properties for one SMILES codes
    """
    mol = Chem.MolFromSmiles(smi)
    return Descriptors.MolLogP(mol), Descriptors.TPSA(mol), QED.qed(mol)


def calcProperty(opt):
    """
    Calculate the properties for the training and testing set.
    """
    num_proc = mp.cpu_count()
    print("Number of CPUS for multithreading:", num_proc)

    print("Calculating properties for {} train molecules: logP, tPSA, QED".format(len(opt.src_data)))
    with mp.Pool(processes=num_proc) as pool:
        res = pool.map(calcForSMILES, tqdm(opt.src_data))
    res = np.array(res)
    prop_df = pd.DataFrame({'logP': res[:,0], 'tPSA': res[:,1], 'QED': res[:,2]})
    prop_df.to_csv("data/moses/prop_temp.csv", index=False)

    print("Calculating properties for {} test molecules: logP, tPSA, QED".format(len(opt.src_data_te)))
    with mp.Pool(processes=num_proc) as pool:
        res = pool.map(calcForSMILES, tqdm(opt.src_data_te))
    res = np.array(res)
    prop_df_te = pd.DataFrame({'logP': res[:,0], 'tPSA': res[:,1], 'QED': res[:,2]})
    prop_df_te.to_csv("data/moses/prop_temp_te.csv", index=False)

    return prop_df, prop_df_te
