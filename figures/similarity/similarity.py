#!/usr/bin/env python
# coding: utf-8

# In[2]:


# !pip install atomInSmiles


# In[15]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#!pip install rdkit
from rdkit import Chem
from atomInSmiles import similarity, encode, decode
from functools import partial


# In[4]:


output_files = {
    'atom: 30000':'atom_IL_moses_bench2_lat=128_epo=50_k=4_20230723.csv',
    'atom: 10 cond':'atom_moses_bench2_10conds_lat=128_epo=50_k=4_20230722.csv',
    'smilespe: 30000':'smilespe_IL_moses_bench2_lat=128_epo=50_k=4_20230723.csv',
    'smilespe: 10 cond':'smilespe_moses_bench2_10conds_lat=128_epo=50_k=4_20230722.csv'
}


# In[5]:


def getDetails(name, file):
    out = pd.read_csv("../outputs/"+file)
    return out[out.Valid==1]

output_dfs = dict(zip(output_files.keys(), (getDetails(name, output_files[name]) for name in output_files.keys())))


# In[10]:


inp = pd.read_csv("../data/IL/train.csv")
inp['atomInSmiles'] = inp['src'].apply(encode)


# In[19]:


def get_snn(out):
    similar_to_out = partial(similarity, ais2=out)
    return inp["atomInSmiles"].apply(similar_to_out).max()


# In[ ]:


for name, df in output_dfs.items():
    print(name)
    df['atomInSmiles'] = df['SMILES'].apply(encode)
    print("SNN: ", df['atomInSmiles'].apply(get_snn).mean())


# In[ ]:




