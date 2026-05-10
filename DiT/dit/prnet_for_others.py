import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import os

from models.transformer import Denoiser
from diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalTransition

from diffusion import diffusion_utils
from metrics.train_loss import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from metrics.molecular_metrics_sampling import compare_smiles_lists
from analysis.rdkit_functions import costom_compute_relaxed_validity
import numpy as np
import utils
from tqdm import tqdm
from metrics.property_metric import calculateSAS, calculateSA, calculateQED, calculatePass_ro5, calculatelogP, calculateMW, calculateHBA, calculateTPSA, calculatenRot, calculateHBD
from metrics.prnet_model import PRnet_test
from metrics.check import check_drug_dose_consistency_with_mode
from metrics.PRnet_sys import PRnet_metrics, prnet_2_csv, PRnet_metrics_for_others
import scanpy as sc

def main(data_type, file_path):

    config_kwargs = {
            'batch_size' : 512,
            'comb_num' : 1,
            'save_dir' : './checkpoint/',
            'results_dir' : './results/lincs/',
            'n_epochs' : 100,
            'split_key' : "drug_split_4",
            'x_dimension' : 978,
            'hidden_layer_sizes' : [128],
            'z_dimension' : 64,
            'adaptor_layer_sizes' : [128],
            'comb_dimension' : 64, 
            'drug_dimension': 1024,
            'dr_rate' : 0.05,
            'n_epochs' : 100,
            'lr' : 1e-3, 
            'weight_decay' : 1e-8,
            'scheduler_factor' : 0.5,
            'scheduler_patience' : 5,
            'n_genes' : 20,
            'loss' : ['GUSS'], 
            'obs_key' : 'cov_drug_dose_name'
        }  
    print(os.getcwd())

    h5ad_file = "DiT/Perturbation-Response-Prediction-PRnet-0847146/dataset/Lincs_L1000.h5ad" # replace with your path
    adata = sc.read(h5ad_file)
    PRnet = PRnet_test(
                            adata,
                            batch_size=config_kwargs['batch_size'],
                            comb_num=config_kwargs['comb_num'],
                            split_key=config_kwargs['split_key'],
                            model_save_dir=config_kwargs['save_dir'],
                            x_dimension=config_kwargs['x_dimension'],
                            hidden_layer_sizes=config_kwargs['hidden_layer_sizes'],
                            z_dimension=config_kwargs['z_dimension'],
                            adaptor_layer_sizes=config_kwargs['adaptor_layer_sizes'],
                            comb_dimension=config_kwargs['comb_dimension'],
                            drug_dimension=config_kwargs['drug_dimension'],
                            dr_rate=config_kwargs['dr_rate'],
                            n_genes=config_kwargs['n_genes'],
                            loss = config_kwargs['loss'],
                            obs_key = config_kwargs['obs_key']
                    )
    res, dose_dict = check_drug_dose_consistency_with_mode(h5ad_file, file_path, data_type)

    siginfo_beta = "DiT/graph_dit/compare_data/L1000_result/siginfo_beta.csv"  # replace with your path
    result_list = PRnet_metrics_for_others(file_path, dose_dict, PRnet, data_type) # feat_true_csv, dose_dict, PRnet, data_type="ours"
    prnet_2_csv(result_list, f"DiT/graph_dit/compare_data/result/{os.path.basename(file_path)}") # replace with your path



if __name__ == "__main__":
    main(data_type="gx2mol", file_path="DiT/graph_dit/compare_data/L1000_result/Gx2Mol_mask_cell_level3.csv")  # replace with your path 
    main(data_type="gx2mol", file_path="DiT/graph_dit/compare_data/L1000_result/Gx2Mol_mask_drugs_level3.csv")  # replace with your path
    main(data_type="triomphe", file_path="DiT/graph_dit/compare_data/L1000_result/TRIOMPHE_mask_cell_level3.csv")  # replace with your path
    main(data_type="triomphe", file_path="DiT/graph_dit/compare_data/L1000_result/TRIOMPHE_mask_drug_level3.csv")  # replace with your path
