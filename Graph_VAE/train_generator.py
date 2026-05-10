import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

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

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_model', default=None)
parser.add_argument('--seed', type=int, default=7)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=32)
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

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=25000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=3000)

log_dir = "./log"

os.makedirs(log_dir, exist_ok=True)  
datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

log_file = os.path.join(log_dir, f'logging_{datetime_str}.log')

logging.basicConfig(
    filename=log_file,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  
)



logging.basicConfig(
    filename=os.path.join("./log", f'logging_{datetime_str}.log'),       
    filemode='w',            
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def print_model_params(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f"{name}")
            logging.info(f"{param.shape}")
            logging.info(f"{param.dtype}")
            logging.info("-" * 50)
            total_params += param.numel()
    logging.info(f" {total_params:,}")


def load_checkpoint(model, model_state):

    pretrained_dict = model_state
    
    model_dict = model.state_dict()
    filtered_dict = {
        k: v for k, v in pretrained_dict.items() 
        if k in model_dict and v.shape == model_dict[k].shape
    }
    
    missing_keys, _ = model.load_state_dict(filtered_dict, strict=False)
    
    for name, param in model.named_parameters():
        if name in missing_keys or name not in pretrained_dict:
            if param.dim() >= 2:  
                nn.init.xavier_normal_(param)
                logging.info(f"Init {name} with Xavier (shape: {param.shape})")
            elif param.dim() == 1: 
                nn.init.constant_(param, 0)
                logging.info(f"Init {name} with Zeros (shape: {param.shape})")
    
    return model

args = parser.parse_args()
logging.info(args)

torch.manual_seed(args.seed)
random.seed(args.seed)

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
args.vocab = PairVocab(vocab)

model = HierVAE(args).cuda()
logging.info("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))
print_model_params(model)


optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

if args.load_model:
    logging.info('continuing from checkpoint ' + args.load_model)
    model_state, optimizer_state, total_step_load, beta = torch.load(args.load_model)
    
    model = load_checkpoint(model, model_state)

else:
    total_step = beta = 0

total_step = 0
param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

meters = np.zeros(6)
for epoch in range(args.epoch):
    dataset = DataFolder(args.train, args.batch_size)

    for batch in tqdm(dataset):
        # breakpoint()
        total_step += 1
        model.zero_grad()
        loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        meters = meters + np.array([kl_div, loss.item(), wacc.detach().cpu().clone() * 100, 
                                                         iacc.detach().cpu().clone() * 100, 
                                                         tacc.detach().cpu().clone() * 100, 
                                                         sacc.detach().cpu().clone() * 100])
        
        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            logging.info("[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model), grad_norm(model)))
            sys.stdout.flush()
            meters *= 0

        if total_step % args.anneal_iter == 0:
            scheduler.step()
            logging.info("learning rate: %.6f" % scheduler.get_lr()[0])

        if total_step >= args.warmup and total_step % args.kl_anneal_iter == 0:
            beta = min(args.max_beta, beta + args.step_beta)

        if total_step % args.save_iter == 0:
            ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta)
            torch.save(ckpt, os.path.join(args.save_dir, f"model_epoch_{epoch}_{total_step}.ckpt"))
