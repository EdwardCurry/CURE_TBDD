import yaml
import os
import math, random, sys
import module
from module.molgene_unit_net import Hierdecoder_self
from module.molgene_3_net import molgene_3_net
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

logging.basicConfig(
    filename=log_file,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True 
)

logging.info("===== Over =====")

random.seed(42)


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
        org_img = torch.stack([torch.from_numpy(meta[0]) for meta in batch], dim=0)
        drug_img = torch.stack([torch.from_numpy(meta[1]) for meta in batch], dim=0)
        smiles_string = [meta[2] for meta in batch]
        smiles_data = func(smiles_string)
        is_train = batch[0][3]
        return org_img, drug_img, smiles_data, is_train, smiles_string
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        smiles_string = [meta[2] for meta in batch]
        print(smiles_string)
    

def Smile_Similarity(org_smiles, trained_smiles, axis=None):
    mol1 = AllChem.MolFromSmiles(org_smiles)  
    mol2 = AllChem.MolFromSmiles(trained_smiles)  
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=1024)

    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity


if __name__ == "__main__":


    num_epochs = config["unit_3_train"]["num_epochs"]
    batch_size = config["unit_3_train"]["batch_size"]
    learning_rate = float(config["unit_3_train"]["learning_rate"])

    device = torch.device("cuda")

    train_dataset = module_3_Dataset(config, is_train_data=True)
    train_batch_sampler = GroupedBatchSampler(train_dataset, batch_size=batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=drug_graph_collate,
        pin_memory=True,
        num_workers=48, 
        persistent_workers=True  
    )

    val_dataset = module_3_Dataset(config, is_train_data=False)
    val_batch_sampler = GroupedBatchSampler(val_dataset, batch_size=batch_size)
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        collate_fn=drug_graph_collate,
        pin_memory=True,
        num_workers=16, 
        persistent_workers=True  
    )


    model, beta = molgene_3_net.from_pretrained(
        config=config,
        ckpt_path_components=("path1", "path2"), # replace with yours
        device="cuda"
    )

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
            
            if is_train == 1:
                module_to_freeze = model.hgraph_encoder
                with freeze_module(module_to_freeze):
                    loss, kl_div, wacc, iacc, tacc, sacc, mse_loss, loss_kl_vec = model(x_org, x_drug, 
                            smiles_info[0], smiles_info[1], smiles_info[2], beta, is_train=is_train)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config["unit_3_train"]["clip_norm"])
            else:
                loss, kl_div, wacc, iacc, tacc, sacc, mse_loss, loss_kl_vec = model(x_org, x_drug, 
                                smiles_info[0], smiles_info[1], smiles_info[2], beta, is_train=is_train)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["unit_3_train"]["clip_norm"])
            optimizer.step()
            batch_loss = loss.item()

            epoch_loss += batch_loss

            with torch.no_grad():
                sync_meters = torch.tensor([kl_div, batch_loss, wacc*100, iacc*100, tacc*100, sacc*100], device='cuda')
                meters += sync_meters.cpu().numpy()
            
            if total_step % config["unit_3_train"]["print_iter"] == 0:
                meters /= config["unit_3_train"]["print_iter"]
                logging.info(
                    f"[{total_step}] , beta: {beta:.2f}, KL: {meters[0]:.2f}, loss: {meters[1]:.3f}, "
                    f"Word: {meters[2]:.2f}, {meters[3]:.2f}, Topo: {meters[4]:.2f}, Assm: {meters[5]:.2f}, mse_loss:{mse_loss:.6f}"
                )
                meters = np.zeros(6)

            if total_step % config["unit_3_train"]["anneal_iter"] == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

            if total_step % config["unit_3_train"]["log_iter"] == 0:
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], item: {total_step}, batch Loss: {batch_loss:.10f}, kl_div:{kl_div:.4f}, mse_loss:{mse_loss:.6f}, loss_kl_vec:{loss_kl_vec:.6f}")

        
            if total_step % config["unit_3_train"]["val_iter"] == 0:
                logging.info(f"################ val #################")
                avg_val_loss = 9999
                model.eval()
                epoch_val_loss = 0.0
                all_val_sim = 0
                all_train_sim = 0
                all_hgraph_sim = 0
                num_val = 0
                num_val_low = 0
                num_train = 0
                num_train_low = 0
                num_h = 0
                with torch.no_grad(): 
                    for batch in val_loader:
                        try:
                            org_img, drug_img, smiles_info, is_train, smiles_string = batch
                            infer_smiles, hgraph_smiles, org_root_vecs, mse_dst = model.infer(org_img, drug_img, tensors=smiles_info[1])
                            for i in range(len(smiles_string)):
                                logging.info(f"is_train: {is_train:4f}")
                                logging.info(f"org_smile: {smiles_string[i]} , infer_smile:{infer_smiles[i]} , hgraph_smiles:{hgraph_smiles[i]}")
                                org_sim = Smile_Similarity(smiles_string[i], infer_smiles[i])
                                logging.info(f"trained sim: {org_sim:4f}")
                                if is_train == 1:
                                    all_train_sim += org_sim
                                    num_train += 1
                                    if org_sim < 0.3
                                        num_train_low += 1
                                else:
                                    all_val_sim += org_sim
                                    num_val += 1
                                    if org_sim < 0.3
                                        num_val_low += 1
                                h_sim = Smile_Similarity(smiles_string[i], hgraph_smiles[i])
                                all_hgraph_sim += h_sim
                                num_h += 1
                                logging.info(f"hgraph sim: {h_sim:4f}")
                                logging.info(f"mse_dst: {mse_dst:4f}")
                        except:
                            continue

                if num_val != 0:
                    logging.info(f"avg val sim: {all_val_sim / num_val}")
                    logging.info(f"low val radio: {num_val_low / num_val}")
                if num_train != 0:
                    logging.info(f"avg train sim: {all_train_sim / num_train}")
                    logging.info(f"low train radio: {num_train_low / num_train}")
                if num_h != 0:
                    logging.info(f"avg hgraph sim: {all_hgraph_sim / num_h}")
                logging.info(f"################ val #################")

        avg_train_loss = epoch_loss / len(train_loader)

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        ckpt_path = config["unit_3_train"]["ckpt_path_unit"]
        os.makedirs(os.path.join(ckpt_path, datetime_str), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_path, datetime_str, "epoch_unit_"+str(epoch)+"_.pth"))



 