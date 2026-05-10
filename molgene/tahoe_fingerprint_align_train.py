import yaml
import os
import math, random, sys
import module
import torch.nn.functional as F
import torch.optim as optim
from module.molgene_unit_net import Hierdecoder_self
from module.molgene_3_net import molgene_3_net
from module.data_loader_csv import *
from module.mol_dit_feat import mol_dit_feat_org_hgraph_model
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
    # breakpoint()
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

logging.info("===== Over =====")

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


def alignment_loss(A, B, alpha=0.4, beta=0.1, weight_init=1): # 1

    with torch.no_grad():
        weights = torch.zeros_like(B)
        non_zero_mask = (B > 0)
        weights[non_zero_mask] = torch.log(weight_init + B[non_zero_mask].float())

        pos_mask = (B > 0).float() 
        neg_mask = (B == 0).float() 

    loss_pos = torch.sum(weights * (A - B) ** 2 * pos_mask) / (pos_mask.sum() + 1e-8)
    
    loss_neg = torch.sum(A ** 2 * neg_mask) / (neg_mask.sum() + 1e-8)
        
    total_loss = loss_pos + alpha * loss_neg 
    return total_loss

def aligned_infonce_loss_mix_mask(A, B, str_labels_list, temperature=0.1, sparse_weight=0.15):

    A = F.normalize(A, dim=-1)
    B = F.normalize(B, dim=-1)
    
    similarity_matrix = torch.matmul(A, B.t())  # (b, b)
    similarity_matrix /= temperature
    
    b = A.size(0)
    labels = torch.arange(b, device=A.device)  # (0, 1, ..., b-1)
    
    str_labels_np = np.array(str_labels_list)
    same_label_mask = torch.tensor(str_labels_np[:, None] == str_labels_np[None, :], 
                                   device=A.device, dtype=torch.bool)
    
    mask = same_label_mask.clone()
    mask.fill_diagonal_(False)  
    similarity_matrix.masked_fill_(mask, -1e10) 
    
    loss = F.cross_entropy(similarity_matrix, labels)
    
    if sparse_weight > 0:
        with torch.no_grad():
            zero_mask = (B == 0).float()
        sparse_loss = torch.mean((A * zero_mask)**2)
        loss += sparse_weight * sparse_loss
    
    return loss

def focal_loss(pred_logits, targets, gamma=2.0, alpha=0.75):
    pos_mask = targets > 0
    neg_mask = ~pos_mask
    
    mse_loss = (pred_logits - targets)**2
    
    final_loss = torch.zeros_like(mse_loss)
    
    final_loss[neg_mask] = mse_loss[neg_mask]
    
    if pos_mask.any():
        pos_errors = mse_loss[pos_mask]
        pt = torch.exp(-pos_errors) 
        focus_weights = (1 - pt) ** gamma 
        final_loss[pos_mask] = focus_weights * pos_errors
    
    loss_weights = torch.where(pos_mask, alpha, 1 - alpha)
    weighted_loss = final_loss * loss_weights
    
    return weighted_loss.sum()


def aligned_infonce_loss_mix_multi(A, B, str_labels_list, temperature=0.1, sparse_weight=0.15):

    A = F.normalize(A, dim=-1)
    B = F.normalize(B, dim=-1)
    
    similarity_matrix = torch.matmul(A, B.t())  # (b, b)
    similarity_matrix /= temperature
    
    unique_labels = np.unique(str_labels_list)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    label_indices = torch.tensor([label_to_idx[label] for label in str_labels_list], 
                                device=A.device)
    
    positive_mask = label_indices.unsqueeze(1) == label_indices.unsqueeze(0)
    
    exp_sim = torch.exp(similarity_matrix)
    
    positive_sum = torch.sum(exp_sim * positive_mask, dim=1, keepdim=True)
    
    negative_sum = torch.sum(exp_sim * (~positive_mask), dim=1, keepdim=True)
    
    loss_per_sample = -torch.log(positive_sum / (positive_sum + negative_sum))
    loss = torch.mean(loss_per_sample)
    
    if sparse_weight > 0:
        with torch.no_grad():
            zero_mask = (B == 0).float()
        sparse_loss = torch.mean((A * zero_mask)**2)
        loss += sparse_weight * sparse_loss
    
    return loss


def sparse_contrastive_loss(a, b, temperature=0.1, reduction='sum'):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    
    similarity = a @ b.T
    logits = similarity / temperature
    pos_logits = torch.diag(logits)
    
    logsumexp = torch.logsumexp(logits, dim=1)
    log_prob = pos_logits - logsumexp
    loss = -log_prob
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def aligned_infonce_loss(A, B, temperature=0.1, sparse_weight=0.15):

    A = F.normalize(A, dim=-1)
    B = F.normalize(B, dim=-1)
    similarity_matrix = torch.matmul(A, B.t()) 
    
    similarity_matrix /= temperature
    
    labels = torch.arange(A.size(0), device=A.device)  # (0, 1, ..., b-1)
    
    loss = F.cross_entropy(similarity_matrix, labels)
    
    if sparse_weight > 0:
        with torch.no_grad():
            zero_mask = (B == 0).float()
        sparse_loss = torch.mean((A * zero_mask)**2)
        loss += sparse_weight * sparse_loss
    
    return loss


def masked_mse(a, b):

    mask = b != 0
    
    if not mask.any():
        return torch.tensor(0.0, device=a.device)
    
    squared_diff = (a - b)**2
    
    masked_diff = squared_diff[mask]
    
    return masked_diff.mean()

if __name__ == "__main__":

    log_path = "./log"
    tb_log_dir = os.path.join(log_path, datetime_str)
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(tb_log_dir)
    print(f"TensorBoard lod at: {tb_log_dir}")


    num_epochs = config["unit_3_train"]["num_epochs"]
    batch_size = config["unit_3_train"]["batch_size"]
    learning_rate = float(config["unit_3_train"]["learning_rate"])

    device = torch.device("cuda")

    train_dataset = org_info_Dataset(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True  
    )

    model = mol_dit_feat_org_hgraph_model(config, embed_dim=32).to(device)
    checkpoint = torch.load("") # replace with yours 
    model.load_state_dict(checkpoint, strict=False)

    for param in model.parameters():
        param.requires_grad = True
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.feat.parameters():
        param.requires_grad = False


    optimizer = Adam(model.parameters(), lr=float(config["unit_3_train"]["learning_rate"]))

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.005,
        total_steps=len(train_loader) * 1000,
        pct_start=0.3
    )

    detailed_analysis(model)

    meters = np.zeros(6)
    total_step = 0
    for epoch in tqdm(range(num_epochs)):
        model.train() 
        
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader):
            total_step += 1
            x_org, x_drug, smiles_string = batch  
            x_org = x_org.float().to(device)
            x_drug = x_drug.float().to(device)
            
            optimizer.zero_grad()

            pre_vec, tar_vec = model(x_org, x_drug, smiles_string)


            classification_loss = focal_loss(   
                pre_vec, tar_vec, alpha=0.5, gamma=2.0     
            )
            alignment_loss_ = alignment_loss(pre_vec, tar_vec)  
            mse_dist = torch.nn.functional.mse_loss(pre_vec, tar_vec)
            masked_mse_ = masked_mse(pre_vec, tar_vec) 

            contrastive_loss = sparse_contrastive_loss(pre_vec, tar_vec) 
            aligned_infonce_loss_ = aligned_infonce_loss(pre_vec, tar_vec, sparse_weight=0)  
            aligned_infonce_loss_mix_mask_ = aligned_infonce_loss_mix_mask(pre_vec, tar_vec, smiles_string, sparse_weight=0)  
            aligned_infonce_loss_mix_multi_ = aligned_infonce_loss_mix_multi(pre_vec, tar_vec, smiles_string, sparse_weight=0) 

        

            loss = alignment_loss_ + aligned_infonce_loss_mix_mask_ 

            # 参数更新
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 主进程打印日志
            if total_step % config["unit_3_train"]["log_interval"] == 0:
                # 整体
                writer.add_scalar('Loss/Loss_all', loss.item(), total_step)
                # mse
                writer.add_scalar('Loss/classification_loss', classification_loss.item(), total_step)  
                writer.add_scalar('Loss/alignment_loss_', alignment_loss_, total_step)  
                writer.add_scalar('Loss/mse_dist', mse_dist, total_step)  
                writer.add_scalar('Loss/masked_mse_', masked_mse_, total_step)
                # 对比学习
                writer.add_scalar('Loss/contrastive_loss', contrastive_loss.item(), total_step)
                writer.add_scalar('Loss/aligned_infonce_loss_', aligned_infonce_loss_.item(), total_step)
                writer.add_scalar('Loss/aligned_infonce_loss_mix_mask_', aligned_infonce_loss_mix_mask_, total_step)
                writer.add_scalar('Loss/aligned_infonce_loss_mix_multi_', aligned_infonce_loss_mix_multi_, total_step)
                
                writer.flush()

        if epoch % config["unit_3_train"]["log_epoch"] == 0:
            ckpt_path = "./ckpt"
            os.makedirs(os.path.join(ckpt_path, datetime_str), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_path, datetime_str, "epoch_unit_"+str(epoch)+"_.pth"))



 