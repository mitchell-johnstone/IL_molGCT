import pandas as pd
import torch
from torchtext import data
from Tokenize import moltokenize
from Batch import MyIterator, batch_size_fn
from sklearn.preprocessing import RobustScaler, StandardScaler
import os
import dill as pickle
import numpy as np
import joblib


def read_data(opt):
    """
    Read each set of data, to get all of the data
    """

    # Read the test and train data
    opt.data = pd.read_csv(opt.data)
    opt.data_te = pd.read_csv(opt.data_te)

    # Set up scalar
    opt.robust_scaler = RobustScaler()
    opt.robust_scaler.fit(opt.data[opt.cond_labels].values)

    # Scale the data
    opt.data[opt.cond_labels] = pd.DataFrame(opt.robust_scaler.transform(opt.data[opt.cond_labels].values))
    opt.data_te[opt.cond_labels] = pd.DataFrame(opt.robust_scaler.transform(opt.data_te[opt.cond_labels].values))


def get_data_df(opt, train):
    """
    Get the data as a Pandas DataFrame
    """
    ret = opt.data if train else opt.data_te
    return ret


def create_fields(opt):
    lang_formats = ['SMILES', 'SELFIES']
    if opt.lang_format not in lang_formats:
        print('invalid src language: ' + opt.lang_forma + 'supported languages : ' + lang_formats)

    print("loading molecule tokenizers...")

    t_src = moltokenize()
    t_trg = moltokenize()

    opt.src_tok = data.Field(tokenize=t_src.tokenizer)
    opt.trg_tok = data.Field(tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')

    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            opt.src_tok = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            opt.trg_tok = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()

    cond_fields =  [(label, data.Field(use_vocab=False, sequential=False, dtype=torch.float)) for label in opt.cond_labels]
    opt.data_fields = [('src', opt.src_tok), ('trg', opt.trg_tok)] + cond_fields
    print("Data Fields:", opt.data_fields)


class ChemicalDataset(data.Dataset):
    def __init__(self, df, fields, **kwargs):
        """
        Arguments:
            df (DataFrame): Pandas DataFrame that has the data
            fields: The fields to process the DataFrame
        """
        examples = df.apply(lambda x: data.Example.fromlist(x.values.flatten().tolist(), fields), axis=1)
        super().__init__(examples, fields, **kwargs)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples.iloc[idx]


def create_dataset(opt, train):
    # masking data longer than max_strlen
    label = "train" if train else "test"
    print(f'\n* creating [{label}] dataset and iterator... ')
    df = get_data_df(opt, train)

    if opt.lang_format == 'SMILES':
        mask = (df['src'].str.len() + opt.cond_dim < opt.max_strlen) & (df['trg'].str.len() + opt.cond_dim < opt.max_strlen)
    # if opt.lang_format == 'SELFIES':
    #     mask = (df['src'].str.count('][') + opt.cond_dim < opt.max_strlen) & (df['trg'].str.count('][') + opt.cond_dim < opt.max_strlen)

    df = df.loc[mask]
    dataset = ChemicalDataset(df, opt.data_fields)
    print(f"     - # of {label} samples:", len(df.index))

    if train:
        toklenList = [len(vars(item)['src']) for item in dataset]
        df_toklenList = pd.DataFrame(toklenList, columns=["toklen"])
        df_toklenList.to_csv("toklen_list.csv", index=False)

    if opt.verbose == True:
        print("     - tokenized {label} sample 0:", vars(dataset[0]))

    # put our data into an batching iterator
    iterator = MyIterator(dataset, batch_size=opt.batchsize, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg), x.Temperature, x.Pressure, x.DynViscosity, x.Density, x.ElecConductivity), batch_size_fn=batch_size_fn, train=True, shuffle=True)
    #iterator = data.DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, num_workers=torch.get_num_threads())

    if train:
        if opt.load_weights is None:
            print("     - building vocab from train data...")
            opt.src_tok.build_vocab(dataset)
            if opt.verbose == True:
                print(f'     - vocab size of SRC: {len(opt.src_tok.vocab)}\n        -> {opt.src_tok.vocab.stoi}')
            opt.trg_tok.build_vocab(dataset)
            if opt.verbose == True:
                print(f'     - vocab size of TRG: {len(opt.trg_tok.vocab)}\n        -> {opt.trg_tok.vocab.stoi}')
            if opt.checkpoint > 0:
                try:
                    os.mkdir("weights")
                except:
                    print("weights folder already exists, run program with -load_weights weights to load them")
                    quit()
                pickle.dump(opt.src_tok, open('weights/SRC.pkl', 'wb'))
                pickle.dump(opt.trg_tok, open('weights/TRG.pkl', 'wb')) 

        opt.src_pad = opt.src_tok.vocab.stoi['<pad>']
        opt.trg_pad = opt.trg_tok.vocab.stoi['<pad>']

        opt.train_len = len(iterator)
    else:
        opt.test_len = len(iterator)
    
    return iterator


def prepare_dataset(opt):
    """
    Prepare the Dataset to use
    """
    # Prep the data
    read_data(opt)
    create_fields(opt)

    # Create the datasets
    opt.train = create_dataset(opt, train=True)
    opt.test = create_dataset(opt, train=False)
