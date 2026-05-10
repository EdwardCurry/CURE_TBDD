import yaml
import os
import math, random, sys
import module
import torch.nn.functional as F
import torch.optim as optim
from module.molgene_unit_net import Hierdecoder_self
from module.molgene_3_net import molgene_3_net, molgene_3_kl_l1000_net
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
import numpy as np
import csv
import numpy as np
import matplotlib.pyplot as plt
import umap
import random
from collections import defaultdict

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


def append_tensor_and_smiles_to_csv(tensor, smiles_list, output_path):

    assert tensor.shape[0] == len(smiles_list), \
        f"not match"
    
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    tensor_np = tensor.detach().numpy()  
    
    file_exists = os.path.exists(output_path)
    
    mode = 'a' if file_exists else 'w'
    
    with open(output_path, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            feature_columns = [f'feat_{i}' for i in range(tensor.shape[1])]
            headers = ['smiles'] + feature_columns
            writer.writerow(headers)
        
        for i, smile in enumerate(smiles_list):
            row = [smile] + [f"{x:.6f}" for x in tensor_np[i]]
            writer.writerow(row)
    

def drug_graph_collate(batch):
    try:
        org_img = torch.stack([meta[0] for meta in batch], dim=0)
        drug_img = torch.stack([meta[1] for meta in batch], dim=0)
        smiles_string = [meta[2] for meta in batch]
        smiles_data = func(smiles_string)
        is_train = batch[0][3]
        return org_img, drug_img, smiles_data, is_train, smiles_string
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        smiles_string = [meta[2] for meta in batch]
        print(smiles_string)




def visualize_smiles_features(data_list, num_smiles=3):
    smiles_dict = defaultdict(list)
    for item in data_list:
        smiles = item["smiles"]
        feature = item["feat"]
        smiles_dict[smiles].append(feature)
    
    unique_smiles = list(smiles_dict.keys())
    if len(unique_smiles) > num_smiles:
        selected_smiles = random.sample(unique_smiles, num_smiles)
    else:
        selected_smiles = unique_smiles
    
    features_list = []
    color_labels = []
    color_map = {}
    
    for idx, smiles in enumerate(selected_smiles):
        color_map[smiles] = np.random.rand(3,)  
        
        for feat in smiles_dict[smiles]:
            features_list.append(feat)
            color_labels.append(color_map[smiles])
    
    breakpoint()
    features_array = np.vstack(features_list)
    colors_array = np.vstack(color_labels)
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(features_array)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], 
                c=colors_array,  
                s=5, alpha=0.6)
    
    plt.title('UMAP Projection of Chemical Features')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(alpha=0.3)
    plt.savefig('smiles_umap_hgraph.png', dpi=300, bbox_inches='tight')

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

    # 初始化模型
    model = molgene_3_kl_l1000_net(config=config).to(device)
    checkpoint = torch.load("") # replace with yours
    model.load_state_dict(checkpoint[0], strict=False)
    model.eval()


    for param in model.parameters():
        param.requires_grad = False
    
    dict_umap = []
    with torch.no_grad():
        for batch in tqdm(train_loader):
            x_org, x_drug, smiles_info, is_train, smiles_string = batch  
            x_org = x_org.float().to(device)
            x_drug = x_drug.float().to(device)
            tar_vec = model.get_embed_hgraph(smiles_info[1])
            append_tensor_and_smiles_to_csv(tar_vec, smiles_string, "./l1000_hgraph.csv")
            for i in range(len(smiles_string)):
                dict_umap.append({"smiles": smiles_string[i], "feat": tar_vec[i].to("cpu").numpy()})

    visualize_smiles_features(dict_umap, num_smiles=1)
    breakpoint()

