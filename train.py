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
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import dill as pickle
import pandas as pd
from calProp import calcProperty
import csv
import timeit


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


def train_model(model, opt):
    global robustScaler
    print("training model...")
    model.train()

    if opt.checkpoint > 0:
        cptime = time.time()

    if opt.imp_test:
        history_df = pd.DataFrame(columns=["epoch", "beta", "lr", "total_loss", "total_loss_te", "RCE_mol_loss", "RCE_mol_loss_te", "RCE_prop_loss", "RCE_prop_loss_te", "KLD_loss", "KLD_loss_te"])
    else:
        history_df = pd.DataFrame(columns=["epoch", "beta", "lr", "total_loss", "RCE_mol_loss", "RCE_prop_loss", "KLD_loss"])
    history_dict = {}

    beta = 0
    current_step = 0
    for epoch in tqdm(range(opt.epochs), desc="Epochs", position=0):
        total_loss,    RCE_mol_loss,    RCE_prop_loss,    KLD_loss    = 0, 0, 0, 0
        total_loss_te, RCE_mol_loss_te, RCE_prop_loss_te, KLD_loss_te = 0, 0, 0, 0
        accum_train_printevery_n, accum_test_n, accum_test_printevery_n = 0, 0, 0

        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')

        # KL annealing
        if opt.use_KLA == True:
            if epoch + 1 >= opt.KLA_beg_epoch and beta < opt.KLA_max_beta:
                beta = KLAnnealer(opt, epoch)
        else:
            beta = 1

        # Training
        for batch in tqdm(opt.train, desc="Training Loop", total=opt.train_len, leave=False):
            current_step += 1
            src = batch.src.transpose(0, 1).to(opt.device)
            trg = batch.trg.transpose(0, 1).to(opt.device)
            trg_input = trg[:, :-1]

            cond = torch.stack([batch.logP, batch.tPSA, batch.QED]).transpose(0, 1).to(opt.device)

            src_mask, trg_mask = create_masks(src, trg_input, cond, opt)
            preds_prop, preds_mol, mu, log_var, z = model(src, trg_input, cond, src_mask, trg_mask)
            ys_mol = trg[:, 1:].contiguous().view(-1)
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

            for param_group in opt.optimizer.param_groups:
                current_lr = param_group['lr']

            total_loss += loss.item()
            RCE_mol_loss += RCE_mol.item()
            RCE_prop_loss += RCE_prop.item()
            KLD_loss += KLD.item()

            if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()

        history_dict['epoch'] = epoch+1
        history_dict['beta'] = beta
        history_dict['lr'] = current_lr
        history_dict['total_loss'] = total_loss / len(opt.train.dataset)
        history_dict['RCE_mol_loss'] = RCE_mol_loss / len(opt.train.dataset)
        history_dict['RCE_prop_loss'] = RCE_prop_loss / len(opt.train.dataset)
        history_dict['KLD_loss'] = KLD_loss / len(opt.train.dataset)

        # Test
        if opt.imp_test == True:
            model.eval()

            with torch.no_grad():
                for batch in tqdm(opt.test, desc="Testing Loop", total=opt.test_len, leave=False):
                    src = batch.src.transpose(0, 1).to(opt.device)
                    trg = batch.trg.transpose(0, 1).to(opt.device)
                    trg_input = trg[:, :-1]
                    cond = torch.stack([batch.logP, batch.tPSA, batch.QED]).transpose(0, 1).to(opt.device)

                    src_mask, trg_mask = create_masks(src, trg_input, cond, opt)
                    preds_prop, preds_mol, mu, log_var, z = model(src, trg_input, cond, src_mask, trg_mask)
                    ys_mol = trg[:, 1:].contiguous().view(-1)
                    ys_cond = torch.unsqueeze(cond, 2).contiguous().view(-1, opt.cond_dim, 1)

                    loss_te, RCE_mol_te, RCE_prop_te, KLD_te = loss_function(opt, beta, preds_prop, preds_mol, ys_cond, ys_mol, mu, log_var)

                    total_loss_te += loss_te.item()
                    RCE_mol_loss_te += RCE_mol_te.item()
                    RCE_prop_loss_te += RCE_prop_te.item()
                    KLD_loss_te += KLD_te.item()

            history_dict['total_loss_te'] = total_loss_te / len(opt.test.dataset)
            history_dict['RCE_mol_loss_te'] = RCE_mol_loss_te / len(opt.test.dataset)
            history_dict['RCE_prop_loss_te'] = RCE_prop_loss_te / len(opt.test.dataset)
            history_dict['KLD_loss_te'] = KLD_loss_te / len(opt.test.dataset)
            
        history_df = history_df.append(history_dict.copy(), ignore_index=True)

        # Export weights every epoch
        # if not os.path.isdir('{}'.format(opt.save_folder_name)):
        #     os.mkdir('{}'.format(opt.save_folder_name))
        # if not os.path.isdir('{}/epo{}'.format(opt.save_folder_name, epoch + 1)):
        #     os.mkdir('{}/epo{}'.format(opt.save_folder_name, epoch + 1))
        # torch.save(model.state_dict(), f'{opt.save_folder_name}/epo{epoch+1}/model_weights')
        # joblib.dump(robustScaler, f'{opt.save_folder_name}/epo{epoch+1}/scaler.pkl')

    # Export train/test history
    history_df.to_csv(f'trHist_lat={opt.latent_dim}_{time.strftime("%Y%m%d")}.csv', index=False)


def get_program_arguments():
    """
    Gets the program arguments, setting the defaults if not given
    """
    parser = argparse.ArgumentParser()
    # Data settings
    parser.add_argument('-imp_test', type=bool, default=True)
    parser.add_argument('-src_data', type=str, default='data/moses/train.txt')
    parser.add_argument('-src_data_te', type=str, default='data/moses/test.txt')
    parser.add_argument('-trg_data', type=str, default='data/moses/train.txt')
    parser.add_argument('-trg_data_te', type=str, default='data/moses/test.txt')
#     parser.add_argument('-src_data', type=str, default='data/moses/tmp_train.txt')
#     parser.add_argument('-src_data_te', type=str, default='data/moses/tmp_test.txt')
#     parser.add_argument('-trg_data', type=str, default='data/moses/tmp_train.txt')
#     parser.add_argument('-trg_data_te', type=str, default='data/moses/tmp_test.txt')
    parser.add_argument('-lang_format', type=str, default='SMILES')
    calProp = not os.path.isfile("data/moses/prop_temp.csv") or not os.path.isfile("data/moses/prop_temp_te.csv")
    parser.add_argument('-calProp', type=bool, default=calProp) #if prop_temp.csv and prop_temp_te.csv exist, set False

    # Learning hyperparameters
    parser.add_argument('-epochs', type=int, default=1)
    parser.add_argument('-no_cuda', type=str, default=False)
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
    parser.add_argument('-cond_dim', type=int, default=3)
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
    opt = get_program_arguments()

#     num_gpus = torch.cuda.device_count()
#     print("NUM GPUS: ", num_gpus)
#     opt.gpus = list(range(num_gpus))

    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()

    read_data(opt)

    # Property calculation: logP, tPSA, QED
    if opt.calProp:
        PROP, PROP_te = calcProperty(opt)
    else:
        PROP, PROP_te = pd.read_csv("data/moses/prop_temp.csv"), pd.read_csv("data/moses/prop_temp_te.csv")

    SRC, TRG = create_fields(opt)
    opt.max_logP = PROP["logP"].max()
    opt.min_logP = PROP["logP"].min()
    opt.max_tPSA = PROP["tPSA"].max()
    opt.min_tPSA = PROP["tPSA"].min()
    opt.max_QED = PROP_te["QED"].max()
    opt.min_QED = PROP_te["QED"].min()

    robustScaler = RobustScaler()
    robustScaler.fit(PROP)
    # if not os.path.isdir('{}'.format(opt.save_folder_name)):
    #     os.mkdir('{}'.format(opt.save_folder_name))
    # joblib.dump(robustScaler, 'scaler.pkl')
    # robustScaler = joblib.load('scaler.pkl')

    # Scale the data
    PROP, PROP_te = pd.DataFrame(robustScaler.transform(PROP)), pd.DataFrame(robustScaler.transform(PROP_te))

    opt.train = create_dataset(opt, SRC, TRG, PROP, tr_te='tr')
    opt.test = create_dataset(opt, SRC, TRG, PROP_te, tr_te='te')

    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# of trainable parameters: {}".format(total_trainable_params))


    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.lr_beta1, opt.lr_beta2), eps=opt.lr_eps)
    if opt.lr_scheduler == "SGDR":
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))
    
    if opt.load_weights is not None and opt.floyd is not None:
        os.makedirs('weights', exist_ok=True)
        pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    
    train_model(model, opt)

    if opt.floyd is False:
        saveModel(model, opt, SRC, TRG, robustScaler)


def saveModel(model, opt, SRC, TRG, robustScaler):
    """
    Save the model, including the weights and the robust scaler
    """
    if opt.load_weights is not None:
        dst = opt.load_weights
    else:
        dst = 'weights'

    os.makedirs(dst, exist_ok=True)
    print(f'saving weights to {dst}/...')
    torch.save(model.state_dict(), f'{dst}/model_weights')
    pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
    pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
    joblib.dump(robustScaler, open(f'{dst}/scaler.pkl', 'wb'))
    
    print(f'weights and field pickles saved to {dst}')


if __name__ == "__main__":
    main()

