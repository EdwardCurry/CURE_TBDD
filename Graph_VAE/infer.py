import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import rdkit
import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm

from hgraph import *

from datetime import datetime
import logging

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', default="./Graph_VAE/l1000_train_processed") 
parser.add_argument('--vocab', default="./Graph_VAE/data/l1000/test.txt") 
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--load_model', default=None)
parser.add_argument('--seed', type=int, default=7)

# 模型参数
parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--step_beta', type=float, default=0.001)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)

parser.add_argument('--epoch', type=int, default=20000)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=10000)
parser.add_argument('--print_iter', type=int, default=10)
parser.add_argument('--save_iter', type=int, default=300)




os.makedirs('./log_l1000', exist_ok=True)

args = parser.parse_args()



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




# dataset
dataset = DataFolder_muti(args.train, shuffle=True)
# sampler = DistributedSampler(dataset, shuffle=True)
dataloader = DataLoader(
    dataset,
    batch_size=1, 
    # sampler=sampler,
    collate_fn=lambda x: x[0]  
)

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
args.vocab = PairVocab(vocab)
model = HierVAE(args).to("cuda")
model.load_state_dict(torch.load("")[0]) # replace with yours



pre_vecs = None
for batch in tqdm(dataloader):
    root_vecs = model.infer_250(*batch, beta=0)

