import os
from sklearn import metrics

import torch
import torch.nn as nn
from torch.autograd import Variable, grad
from torch.nn import functional as F
from torch.distributions import NegativeBinomial, normal

import math
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error


# from metrics.module.Dataset import  DrugDoseAnnDataset
from metrics.module.PRnet import PRnet

from metrics.module.utils import Drug_dose_encoder



    
# def _nan2inf(x):
#     return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

class PRnet_test:
    """
    This class contains the implementation of the PRnetTrainer Trainer
    Parameters
    ----------
    model: PRnet
    adata: : `~anndata.AnnData`
        Annotated Data Matrix for training PRnet.
    batch_size: integer
        size of each batch to be fed to network.
    comb_num: int
        Number of combined compounds.
    shuffle: bool
        if `True` shuffles the training dataset.
    split_key: string
        Attributes of data split.
    model_save_dir: string
        Save dir of model. 
    x_dimension: int
        Dimention of x
    hidden_layer_sizes: list
        A list of hidden layer sizes
    z_dimension: int
        Dimention of latent space
    adaptor_layer_sizes: list
        A list of adaptor layer sizes
    comb_dimension: int
        Dimention of perturbation latent space
    drug_dimension: int
        Dimention of rFCGP
    n_genes: int
        Dimention of different expressed gene
    n_epochs: int
        Number of epochs to iterate and optimize network weights.
    train_frac: Float
        Defines the fraction of data that is used for training and data that is used for validation.
    dr_rate: float
        dropout_rate
    loss: list
        Loss of model, subset of 'NB', 'GUSS', 'KL', 'MSE'
    obs_key:
        observation key of data
    """
    def __init__(self, adata, batch_size = 32, comb_num = 2, shuffle = True, split_key='random_split', model_save_dir = './checkpoint/', x_dimension = 5000, hidden_layer_sizes = [128], z_dimension = 64, adaptor_layer_sizes = [128], comb_dimension = 64, drug_dimension = 1031, n_genes=20,  dr_rate = 0.05, loss = ['guss'], obs_key = 'cov_drug_name', model_path="./Perturbation-Response-Prediction-PRnet-0847146/checkpoint/lincs_best_epoch_all.pt", **kwargs): # maybe add more parameters
        
        assert set(loss).issubset(['NB', 'GUSS', 'KL', 'MSE']), "loss should be subset of ['NB', 'GUSS', 'KL', 'MSE']"

        self.x_dim = x_dimension
        self.split_key = split_key
        self.z_dimension = z_dimension
        self.comb_dimension = comb_dimension

        self.model = PRnet(adata, x_dimension=self.x_dim, hidden_layer_sizes=hidden_layer_sizes, z_dimension=z_dimension, adaptor_layer_sizes=adaptor_layer_sizes, comb_dimension=comb_dimension, comb_num=comb_num, drug_dimension=drug_dimension,dr_rate=dr_rate)
        
        self.model_save_dir = model_save_dir
        self.loss = loss
        self.modelPGM = self.model.get_PGM()
        


        self.seed = kwargs.get("seed", 2024)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            if(torch.cuda.device_count() > 1):
                self.modelPGM = nn.DataParallel(self.modelPGM, device_ids=[i for i in range(torch.cuda.device_count())])         
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelPGM = self.modelPGM.to(self.device)

        # self.modelPGM.apply(self.weight_init)
        # print(self.modelPGM)


        self.adata = adata
        #self.adata_deg_list = adata.uns['rank_genes_groups_cov']
        self.de_n_genes = n_genes
        self.adata_var_names = adata.var_names
        # self.train_data, self.valid_data, self.test_data = train_valid_test(self.adata, split_key = split_key)

        
        # if self.train_data is not None:
        #     self.train_dataset = DrugDoseAnnDataset(self.train_data, dtype='train', obs_key=obs_key, comb_num=comb_num)     
        #     self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        # if self.valid_data is not None:
        #     self.valid_dataset = DrugDoseAnnDataset(self.valid_data, dtype='valid', obs_key=obs_key, comb_num=comb_num)
        #     self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True)
        # if self.test_data is not None:
        #     self.test_dataset = DrugDoseAnnDataset(self.test_data, dtype='test', obs_key=obs_key, comb_num=comb_num)
        #     self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

        if set(['NB']).issubset(loss):
            self.criterion = NBLoss()
        if set(['GUSS']).issubset(loss):
            self.criterion = nn.GaussianNLLLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        self.shuffle = shuffle
        self.batch_size = batch_size

        # Optimization attributes

        self.epoch = -1  # epoch = self.epoch + 1 in compute metrics
        self.best_state_dictPGM = None


        self.PGM_losses = []
        self.r2_score_mean = []
        self.r2_score_var = []
        self.mse_score = []
        self.r2_score_mean_de = []
        self.r2_score_var_de = []
        self.mse_score_de = []
        self.best_mse = np.inf
        self.patient = 0
        # breakpoint()
        self.modelPGM.load_state_dict(torch.load(model_path))

    
    def make_noise(self, batch_size, shape, volatile=False):
        tensor = torch.randn(batch_size, shape)
        noise = Variable(tensor, volatile)
        noise = noise.to(self.device, dtype=torch.float32)
        return noise
       

    def infer(self, smiles, dose, control, return_dict = False):

        self.modelPGM.eval()

        # breakpoint()
        x_true_array = np.zeros((0, self.x_dim))
        y_true_array = np.zeros((0, self.x_dim))
        y_pre_array = np.zeros((0, self.x_dim))
        cov_drug_list = []

        control = torch.tensor(control)
        control = control.to(self.device, dtype=torch.float32)
        if set(['NB']).issubset(self.loss):
                control = torch.log1p(control)

        encode_label = Drug_dose_encoder([smiles], [dose])
        encode_label = torch.tensor(encode_label)
        encode_label = encode_label.to(self.device, dtype=torch.float32)
        b_size = control.size(0)
        
        noise = self.make_noise(b_size, 10)
        gene_reconstructions = self.modelPGM(control, encode_label, noise).detach()
        dim = gene_reconstructions.size(1) // 2
        gene_means = gene_reconstructions[:, :dim]
        gene_vars = gene_reconstructions[:, dim:]
        gene_vars = F.softplus(gene_vars)
            
        if set(['GUSS']).issubset(self.loss):
            dist = normal.Normal(
                torch.clamp(
                    torch.Tensor(gene_means),
                    min=1e-3,
                    max=1e3,
                ),
                torch.clamp(
                    torch.Tensor(gene_vars.sqrt()),
                    min=1e-3,
                    max=1e3,
                )           
            )
            
        nb_sample = dist.sample().cpu().numpy()

        return nb_sample


 

    








