import yaml
import os
import math, random, sys
import module
from module.molgene_unit_net import Hierdecoder_self
from module.molgene_3_net import molgene_3_net, molgene_3_kl_net, molgene_3_kl_l1000_net
from module.data_loader_csv import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import logging
from datetime import datetime
from module.hgraph2graph.hgraph import MolGraph, common_atom_vocab, PairVocab
import torch.optim.lr_scheduler as lr_scheduler
from functools import partial
import rdkit
from module.hgraph2graph.preprocess import tensorize
from rdkit import DataStructs
from rdkit.Chem import AllChem
from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter

with open("./config/config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


with open(config["unit_3_train"]["vocab"]) as f:
    vocab = [x.strip("\r\n ").split() for x in f]
vocab_ = PairVocab(vocab, cuda=False)



@contextmanager
def freeze_module(module: nn.Module):

    original_requires_grad = {name: param.requires_grad for name, param in module.named_parameters()}
    
    try:
        for param in module.parameters():
            param.requires_grad = False
        yield 
    finally:
        for name, param in module.named_parameters():
            param.requires_grad = original_requires_grad[name]


def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c

in_TOTAL_PSEUDO_CELLS = config["data_load"]["TOTAL_PSEUDO_CELLS"]
in_N_CELLS_PER_PSEUDO = config["data_load"]["N_CELLS_PER_PSEUDO"]

log_dir = config["log_path"]

os.makedirs(log_dir, exist_ok=True)  

datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

log_file = os.path.join(log_dir, f'logging_unit_3_train_{datetime_str}.log')
tb_log_dir = os.path.join(log_dir, datetime_str)
print(tb_log_dir)
writer = SummaryWriter(tb_log_dir)

logging.basicConfig(
    filename=log_file,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  
)


logging.info("===== OVER =====")

random.seed(1)


def detailed_analysis(model):
    print("{:<30} {:<20} {:<15}".format('Module', 'Total Params', 'Trainable'))
    print("-" * 65)
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logging.info("{:<30} {:<20,} {:<15,}".format(name, params, trainable))
        print("{:<30} {:<20,} {:<15,}".format(name, params, trainable))


def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)

func = partial(tensorize, vocab=vocab_)

def drug_graph_collate(batch):
    try:
        print(batch[0][0].shape)
        org_img = torch.stack([meta[0] for meta in batch], dim=0)
        print(org_img.shape)
        drug_img = torch.stack([meta[1] for meta in batch], dim=0)
        smiles_string = [meta[2] for meta in batch]
        smiles_data = func(smiles_string)
        is_train = batch[0][3]
        print(len(smiles_string))
        return org_img, drug_img, smiles_data, is_train, smiles_string
    except KeyboardInterrupt:
        exit()
    except Exception as e:  
        smiles_string = [meta[2] for meta in batch]
        print("error:", smiles_string)
    

def Smile_Similarity(org_smiles, trained_smiles, axis=None):
    mol1 = AllChem.MolFromSmiles(org_smiles)  
    mol2 = AllChem.MolFromSmiles(trained_smiles)  
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=1024)

    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity


def load_checkpoint(model, model_state):

    pretrained_dict = model_state
    
    model_dict = model.state_dict()
    filtered_dict = {
        k: v for k, v in pretrained_dict.items() 
        if k in model_dict and v.shape == model_dict[k].shape
    }
    
    missing_keys, _ = model.load_state_dict(filtered_dict, strict=False)
    print(filtered_dict.keys())
    print("#####################################")
    print(missing_keys)
    
    for name, param in model.named_parameters():
        if name in missing_keys or name not in pretrained_dict:
            if param.dim() >= 2: 
                nn.init.xavier_normal_(param)
                logging.info(f"Init {name} with Xavier (shape: {param.shape})")
            elif param.dim() == 1: 
                nn.init.constant_(param, 0)
                logging.info(f"Init {name} with Zeros (shape: {param.shape})")
    
    return model

if __name__ == "__main__":


    num_epochs = config["unit_3_train"]["num_epochs"]
    batch_size = config["unit_3_train"]["batch_size"]
    learning_rate = float(config["unit_3_train"]["learning_rate"])

    device = torch.device("cuda")
    cp_path = config["unit_3_train"]["cp_path"]
    ctl_path = config["unit_3_train"]["ctl_path"]

    train_dataset = CpCtlDataset_solo(cp_path, ctl_path)
    train_batch_sampler = GroupedBatchSampler_l1000(train_dataset, batch_size=batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=drug_graph_collate,
        pin_memory=True,
        num_workers=32,
        persistent_workers=True  
    )

    model = molgene_3_kl_l1000_net(config=config).to(device)


    feat_checkpoint = torch.load("")["model_state_dict"] # replace with your path
    del feat_checkpoint["fc.weight"]
    del feat_checkpoint["fc.bias"]
    model.feat.load_state_dict(feat_checkpoint, strict=False)

    beta = 0.8

    for param in model.parameters():
        param.requires_grad = True


    optimizer = Adam(model.parameters(), lr=float(config["unit_3_train"]["learning_rate"]))

    scheduler = lr_scheduler.ExponentialLR(optimizer, config["unit_3_train"]["anneal_rate"])

    detailed_analysis(model)

    meters = np.zeros(6)
    total_step = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()  
        
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader):
            total_step += 1
            x_org, x_drug, smiles_info, is_train, smiles_string = batch  
            x_org = x_org.float().to(device)
            x_drug = x_drug.float().to(device)
            
            optimizer.zero_grad()
            is_train = 1
            module_to_freeze = model.encoder
            with freeze_module(module_to_freeze):
                loss, kl_div, wacc, iacc, tacc, sacc, mean_mse_loss, var_mse_loss, feat_hgraph_kl_loss, reconstruct_loss = model(x_org, x_drug, 
                        smiles_info[0], smiles_info[1], smiles_info[2], beta, is_train=is_train)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["unit_3_train"]["clip_norm"])
            optimizer.step()
            batch_loss = loss.item()

            epoch_loss += batch_loss

            with torch.no_grad():
                sync_meters = torch.tensor([kl_div, batch_loss, wacc*100, iacc*100, tacc*100, sacc*100], device='cuda')
                meters += sync_meters.cpu().numpy()
            
            if is_train == 1:
                writer.add_scalar('Loss/is_train_1', batch_loss, total_step)
                writer.add_scalar('Loss/is_train_1_mean_mse_loss', mean_mse_loss, total_step)
                writer.add_scalar('Loss/is_train_1_var_mse_loss', var_mse_loss, total_step)
                writer.add_scalar('Loss/is_train_1_reconstruct_loss', reconstruct_loss, total_step)
            else:
                writer.add_scalar('Loss/is_train_0', batch_loss, total_step)
                writer.add_scalar('Loss/is_train_0_reconstruct_loss', reconstruct_loss, total_step)
            writer.flush()
            if total_step % config["unit_3_train"]["print_iter"] == 0:
                meters /= config["unit_3_train"]["print_iter"]
                logging.info(
                    f"[{total_step}] , beta: {beta:.2f}, KL: {meters[0]:.2f}, loss: {meters[1]:.3f}, "
                    f"Word: {meters[2]:.2f}, {meters[3]:.2f}, Topo: {meters[4]:.2f}, Assm: {meters[5]:.2f}"
                )
                meters = np.zeros(6)

            if total_step % config["unit_3_train"]["anneal_iter"] == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

            if total_step % config["unit_3_train"]["log_iter"] == 0:
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], item: {total_step}, batch Loss: {batch_loss:.10f}, "
                        f"kl_div:{kl_div:.4f}, mean_mse_loss:{mean_mse_loss:.6f}, var_mse_loss:{var_mse_loss:.6f}, "
                        f"feat_hgraph_kl_loss:{feat_hgraph_kl_loss:.6f}, reconstruct_loss: {reconstruct_loss:.6f}")

        avg_train_loss = epoch_loss / len(train_loader)

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        ckpt_path = config["unit_3_train"]["ckpt_path_unit"]
        os.makedirs(os.path.join(ckpt_path, datetime_str), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_path, datetime_str, "epoch_unit_"+str(epoch)+"_.pth"))



 