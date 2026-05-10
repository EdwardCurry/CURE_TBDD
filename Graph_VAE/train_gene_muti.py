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
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_model', default=None)
parser.add_argument('--seed', type=int, default=7)

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
local_rank = int(os.environ["LOCAL_RANK"])
args = parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', init_method='env://', world_size=int(os.environ["WORLD_SIZE"]), rank=int(os.environ["RANK"]))   
world_size = dist.get_world_size()


def print_model_params(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f"{name}")
            logging.info(f"{param.shape}")
            logging.info(f"{param.dtype}")
            logging.info("-" * 50)
            total_params += param.numel()
    logging.info(f"{total_params:,}")


def load_checkpoint(model, model_state):

    pretrained_dict = model_state
    
    model_dict = model.state_dict()
    filtered_dict = {
        k: v for k, v in pretrained_dict.items() 
        if k in model_dict and v.shape == model_dict[k].shape
    }
    
    missing_keys, _ = model.load_state_dict(filtered_dict, strict=False)
    print(filtered_dict.keys())
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

if local_rank == 0:
    log_dir = "./log_l1000"
    os.makedirs(log_dir, exist_ok=True)
    datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    tb_log_dir = os.path.join(log_dir, datetime_str)
    writer = SummaryWriter(tb_log_dir)
    print(f"TensorBoard log at: {tb_log_dir}")

    log_file = os.path.join(log_dir, f'muti_logging_{datetime_str}.log')
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
else:
    logging.basicConfig(level=logging.WARNING)  

logging.info(f"World Size: {world_size}, Local Rank: {local_rank}")

seed = args.seed + local_rank
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# dataset
dataset = DataFolder_muti(args.train, shuffle=True)
sampler = DistributedSampler(dataset, shuffle=True)
dataloader = DataLoader(
    dataset,
    batch_size=1,  
    sampler=sampler,
    collate_fn=lambda x: x[0]  
)


vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
args.vocab = PairVocab(vocab)
model = HierVAE(args).cuda(local_rank)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

if args.load_model:
    if local_rank == 0:
        model_state, optimizer_state, total_step_load, beta = torch.load(args.load_model)
    else:
        model_state = optimizer_state = None
        total_step_load = beta = None

    model_state_list = [model_state]
    dist.broadcast_object_list(model_state_list, src=0)  
    model_state = model_state_list[0]

    optimizer_state_list = [optimizer_state]
    dist.broadcast_object_list(optimizer_state_list, src=0)
    optimizer_state = optimizer_state_list[0]

    total_step_tensor = torch.tensor([total_step_load] if local_rank == 0 else [0], dtype=torch.int64).cuda()
    beta_tensor = torch.tensor([beta] if local_rank == 0 else [0.0], dtype=torch.float32).cuda()

    dist.broadcast(total_step_tensor, src=0)
    dist.broadcast(beta_tensor, src=0)

    total_step_load = total_step_tensor.item()  
    beta = beta_tensor.item()                   
    model.module = load_checkpoint(model.module, model_state)
else:
    total_step = 0
    beta = 0.0

total_step = 0

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

meters = np.zeros(6)
avg_loss = 0
for epoch in range(args.epoch):
    sampler.set_epoch(epoch)
    model.train()
    for batch in tqdm(dataloader):
        error = torch.tensor(0, dtype=torch.int, device=local_rank)
        
        try:
            loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)
        except Exception as e:
            print(f"Rank {local_rank} error: {e}")
            logging.info(f"Rank {local_rank} error: {e}")
            error = torch.tensor(1, dtype=torch.int, device=local_rank)
        dist.all_reduce(error, op=dist.ReduceOp.MAX)

        if error.item() == 1:
            logging.info(f"jump batch")
            optimizer.zero_grad()  
            continue
        else:
            optimizer.zero_grad() 
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            total_step += 1

        item_loss = loss.item()
        avg_loss = avg_loss + item_loss
        
        with torch.no_grad():
            sync_meters = torch.tensor([kl_div, item_loss, wacc*100, iacc*100, tacc*100, sacc*100], device='cuda')
            dist.all_reduce(sync_meters, op=dist.ReduceOp.SUM)
            sync_meters /= world_size
            meters += sync_meters.cpu().numpy()
        
        if total_step % args.print_iter == 0 and local_rank == 0:
            meters /= args.print_iter
            writer.add_scalar('Loss/loss', item_loss, total_step)
            writer.add_scalar('Loss/beta', beta, total_step)
            writer.add_scalar('Loss/KL', meters[0], total_step)
            writer.add_scalar('Loss/Word', meters[2], total_step)
            writer.add_scalar('Loss/Topo', meters[4], total_step)
            writer.add_scalar('Loss/Assm', meters[5], total_step)
            writer.add_scalar('Loss/PNorm', param_norm(model), total_step)
            writer.add_scalar('Loss/GNorm', grad_norm(model), total_step)
            writer.flush()
            logging.info(
                f"[{total_step}] Beta: {beta:.3f}, KL: {meters[0]:.2f}, loss: {meters[1]:.3f}, "
                f"Word: {meters[2]:.2f}, {meters[3]:.2f}, Topo: {meters[4]:.2f}, Assm: {meters[5]:.2f}, "
                f"PNorm: {param_norm(model):.2f}, GNorm: {grad_norm(model):.2f}"
            )
            meters = np.zeros(6)
        
        if total_step % args.anneal_iter == 0:
            scheduler.step()
            if local_rank == 0:
                logging.info(f"learning rate: {scheduler.get_lr()[0]:.6f}")

        if total_step % 400 == 0:
            
            avg_loss = avg_loss / 400
            writer.add_scalar('Loss/avg', avg_loss, total_step)
            logging.info(f"avg_loss: {avg_loss:.3f}")
            avg_loss = 0

        
        if total_step >= args.warmup and total_step % args.kl_anneal_iter == 0:
            beta = min(args.max_beta, beta + args.step_beta)
        
        if total_step % args.save_iter == 0 and local_rank == 0:
            ckpt = (model.module.state_dict(), optimizer.state_dict(), total_step, beta)
            torch.save(ckpt, os.path.join(args.save_dir, f"model_epoch_{epoch}_{total_step}.ckpt"))

dist.destroy_process_group()