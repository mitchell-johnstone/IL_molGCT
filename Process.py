import pandas as pd
import torch
import torchtext
from torchtext import data
from Tokenize import moltokenize
from Batch import MyIterator, batch_size_fn
from sklearn.preprocessing import RobustScaler, StandardScaler
from CalcProperty import calcProperty
import os
import dill as pickle
import numpy as np
import joblib


def try_read_data(filename):
    """
    Try to read the given filename
    """
    if filename is not None:
        try:
            return open(filename, 'rt', encoding='UTF8').read().strip().split('\n')
        except:
            print("error: '" + filename + "' file not found")
            quit()
    return None


def read_data(opt):
    """
    Read each set of data, to get all of the data
    """
    opt.src_data = try_read_data(opt.src_data)
    opt.trg_data = try_read_data(opt.trg_data)
    opt.src_data_te = try_read_data(opt.src_data_te)
    opt.trg_data_te = try_read_data(opt.trg_data_te)

    # Property calculation: logP, tPSA, QED
    if opt.calProp:
        PROP, PROP_te = calcProperty(opt)
    else:
        PROP, PROP_te = pd.read_csv("data/moses/prop_temp.csv"), pd.read_csv("data/moses/prop_temp_te.csv")
    opt.max_logP = PROP["logP"].max()
    opt.min_logP = PROP["logP"].min()
    opt.max_tPSA = PROP["tPSA"].max()
    opt.min_tPSA = PROP["tPSA"].min()
    opt.max_QED = PROP_te["QED"].max()
    opt.min_QED = PROP_te["QED"].min()

    opt.robust_scaler = RobustScaler()
    opt.robust_scaler.fit(PROP.values)
    # if not os.path.isdir('{}'.format(opt.save_folder_name)):
    #     os.mkdir('{}'.format(opt.save_folder_name))
    # joblib.dump(opt.robust_scaler, 'weights/scaler.pkl')
    # opt.robust_scaler = joblib.load('scaler.pkl')

    # Scale the data
    opt.PROP = pd.DataFrame(opt.robust_scaler.transform(PROP.values), columns=["logP", "tPSA", "QED"])
    opt.PROP_te = pd.DataFrame(opt.robust_scaler.transform(PROP_te.values), columns=["logP", "tPSA", "QED"])


def get_data_df(opt, train):
    """
    Get the data as a Pandas DataFrame
    """
    raw_data = {'src': list(opt.src_data) if train else list(opt.src_data_te),\
                'trg': list(opt.trg_data) if train else list(opt.trg_data_te)}
    data_df = pd.DataFrame(raw_data, columns=["src", "trg"])
    prop_df = pd.DataFrame(opt.PROP if train else opt.PROP_te)
    return pd.concat([data_df, prop_df], axis=1)


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

    logP = data.Field(use_vocab=False, sequential=False, dtype=torch.float)
    tPSA = data.Field(use_vocab=False, sequential=False, dtype=torch.float)
    QED = data.Field(use_vocab=False, sequential=False, dtype=torch.float)

    opt.data_fields = [('src', opt.src_tok), ('trg', opt.trg_tok), ('logP', logP), ('tPSA', tPSA), ('QED', QED)]


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
    # print(df.info())
    # print(df.head())

    ################ TESTING THE CODE #####################
    # if train:  #for code test
    #     df = df[:30000]
    # else:
    #     df = df[:3000]

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
    iterator = MyIterator(dataset, batch_size=opt.batchsize, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg), x.logP, x.tPSA, x.QED), batch_size_fn=batch_size_fn, train=True, shuffle=True)
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
