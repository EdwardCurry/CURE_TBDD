
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import anndata as ad
import numpy as np
import pandas as pd
import sys
import os
import glob
import time
import multiprocessing # For parallel processing
import logging # For better error logging
import yaml
import random
from torch.utils.data.sampler import Sampler
import math

def pair_data(csv_paths):
    pair_img_dict = {}
    full_imgs = os.listdir(csv_paths)
    for img_name in full_imgs:
        if "(" not in img_name and ")" not in img_name:
            drug_img = img_name.replace("DMSO_TF", "Trametinib (DMSO_TF solvate)")
            pair_img_dict[f"{img_name}"] = drug_img
    return pair_img_dict



def process_h5ad_file(in_TOTAL_PSEUDO_CELLS, in_N_CELLS_PER_PSEUDO, h5ad_path):
    """
    Reads a single .h5ad file, generates pseudo-cells 5 times,
    and saves the results to CSV files.
    Designed to be run in a separate process.
    """
    base_name = os.path.splitext(os.path.basename(h5ad_path))[0]
    # output_csv_base_name = os.path.join(self.OUTPUT_DIR, base_name)
    # process_id = os.getpid() # Get process ID for logging
    # logging.info(f"[PID {process_id}] Processing file: {h5ad_path}")

    try:
        # --- 1. Read Input AnnData File ---
        try:
            adata = ad.read_h5ad(h5ad_path)
            # Basic check for required data
            if 'phase' not in adata.obs.columns:
                return f"Skipped: Missing 'phase' column in {h5ad_path}"
        except FileNotFoundError:
            return f"Skipped: File not found {h5ad_path}"
        except Exception as e:
            return f"Skipped: Error reading H5AD {h5ad_path} - {e}"

        # --- 2. Calculate Phase Proportions and Target Counts ---
        try:
            phase_counts = adata.obs['phase'].value_counts()
            total_real_cells = phase_counts.sum()
            if total_real_cells == 0:
                return f"Skipped: No cells in {h5ad_path}"

            proportions = phase_counts / total_real_cells
            target_counts_float = proportions * in_TOTAL_PSEUDO_CELLS
            target_counts = target_counts_float.round().astype(int)
            diff = in_TOTAL_PSEUDO_CELLS - target_counts.sum()

            if diff != 0:
                if not target_counts.empty:
                    idx_to_adjust = target_counts.idxmax()
                    target_counts[idx_to_adjust] += diff

        except Exception as e:
            return f"Skipped: Count calculation error {h5ad_path} - {e}"

        # --- Loop for Repetitions ---
        files_generated_count = 0
        pseudo_cell_list = []
        pseudo_cell_phases = []
        generation_successful_rep = True

        # --- 3. Generate Pseudo-cells (Sampling/Averaging) ---
        try:
            for phase in target_counts.index:
                num_pseudo_for_phase = target_counts.get(phase, 0) # Use .get for safety
                if num_pseudo_for_phase <= 0:
                    continue

                real_cell_indices_phase = adata.obs.index[adata.obs['phase'] == phase]
                n_real_cells_in_phase = len(real_cell_indices_phase)

                if n_real_cells_in_phase < in_N_CELLS_PER_PSEUDO:
                    dynamic_N_CELLS_PER_PSEUDO = n_real_cells_in_phase
                else:
                    dynamic_N_CELLS_PER_PSEUDO = in_N_CELLS_PER_PSEUDO
                    

                for i in range(num_pseudo_for_phase):
                    chosen_indices = np.random.choice(real_cell_indices_phase,
                                                    size=dynamic_N_CELLS_PER_PSEUDO,
                                                    replace=False)
                    pseudo_expression = adata[chosen_indices, :].X.mean(axis=0)

                    if hasattr(pseudo_expression, "A"):
                        pseudo_expression = pseudo_expression.A.flatten()
                    elif isinstance(pseudo_expression, np.matrix):
                        pseudo_expression = np.array(pseudo_expression).flatten()
                    elif not isinstance(pseudo_expression, np.ndarray):
                        pseudo_expression = np.array(pseudo_expression)
                    if pseudo_expression.ndim > 1:
                        pseudo_expression = pseudo_expression.flatten()

                    pseudo_cell_list.append(pseudo_expression)
                    pseudo_cell_phases.append(phase)

        except Exception as e:
            generation_successful_rep = False

        # --- 4. Prepare Data for Saving ---
        if not generation_successful_rep or not pseudo_cell_list:
            return "not generation_successful_rep or not pseudo_cell_list"

        try:
            pseudo_cell_matrix = np.vstack(pseudo_cell_list)
            pseudo_cell_matrix = np.expand_dims(pseudo_cell_matrix, axis=0) 
            pseudo_cell_matrix = np.repeat(pseudo_cell_matrix, repeats=3, axis=0)
            # pseudo_cell_matrix = np.transpose(pseudo_cell_matrix, (0,2,1))
            return pseudo_cell_matrix

            # n_generated = pseudo_cell_matrix.shape[0]

            # pseudo_df = pd.DataFrame(pseudo_cell_matrix, columns=adata.var_names)
            # pseudo_df.insert(0, 'phase', pseudo_cell_phases[:n_generated])
            # pseudo_df.index = [f"pseudo_{i+1}" for i in range(n_generated)]
            # pseudo_df.index.name = "PseudoCellID"

        except Exception as e:
            return "error"

    except Exception as e:
        return f"Failed: Unexpected error in {h5ad_path} - {e}"



class MatrixDataset(Dataset):
    def __init__(self, csv_paths, drug_metadata_path):
        self.pair_img_dict = pair_data(csv_paths)
        self.csv_paths = csv_paths
        self.org_img_keys = list(self.pair_img_dict.keys())
        self.drug_metadata_path = drug_metadata_path

        print(f"Attempting to read Parquet file: {self.drug_metadata_path}")

        self.drug_metadata_df = pd.read_parquet(self.drug_metadata_path)

        print("File read successfully!")
        print("\nDataFrame Info:")
        self.drug_metadata_df.info() 

        
    def __len__(self):
        return len(self.org_img_keys)

    def load_img(self, img_path):
        df = pd.read_csv(img_path, header=None, skiprows=1)
        
        matrix = df.iloc[:, 2:].to_numpy() 
        
        matrix = np.expand_dims(matrix, axis=0) 
        matrix = np.repeat(matrix, repeats=3, axis=0)
        # matrix = np.transpose(matrix, (1, 2, 0))
        # matrix = ((matrix + 1.0) * 127.5).clip(0, 255)
        return matrix


    def __getitem__(self, idx):
        # 读取CSV文件
        org_path = os.path.join(self.csv_paths, self.org_img_keys[idx])
        drug_path = os.path.join(self.csv_paths, self.pair_img_dict[f"{self.org_img_keys[idx]}"])

        org_img = self.load_img(org_path)
        drug_img = self.load_img(drug_path)

        smiles = 'C=Cc1cccc(C(=O)N2CC(c3ccc(F)cc3)C(C)(C)C2)c1'
        self.drug_metadata_df

        return org_img, drug_img, smiles


class org_info_Dataset(Dataset):
    def __init__(self, config, smiles_dot=False):
        self.smiles_dot = smiles_dot
        self.config = config
        self.INPUT_DIR = self.config["data_load"]["INPUT_DIR"]
        self.OUTPUT_DIR = self.config["data_load"]["OUTPUT_DIR"]
        self.LOG_FILE = self.config["data_load"]["LOG_FILE"]
        self.TOTAL_PSEUDO_CELLS = self.config["data_load"]["TOTAL_PSEUDO_CELLS"]
        self.N_CELLS_PER_PSEUDO = self.config["data_load"]["N_CELLS_PER_PSEUDO"]
        self.NUM_REPETITIONS = self.config["data_load"]["NUM_REPETITIONS"]
        self.h5ad_files = glob.glob(os.path.join(self.INPUT_DIR, '*.h5ad'))
        # df
        self.df = pd.read_parquet(config["drug_metadata_path"])

        with open(config["yes_smiles"]["path"], 'r', encoding='utf-8') as f:
            self.yes_smiles = [line.strip() for line in f]

        

        self.h5ad_files_pair = {}
        for h5ad_file in self.h5ad_files:
            if h5ad_file.split("+")[-1] != "DMSO_TF.h5ad":
                drug_string = os.path.basename(h5ad_file).split("+")[-1].split(".")[0]
                drug_smiles = self.lookup_smiles(drug_string)
                if drug_smiles is None:
                    continue
                # self.h5ad_files_pair[f"{h5ad_file.split('+')[0]}+DMSO_TF.h5ad"] = {"drug_path":h5ad_file, "drug_smiles": drug_smiles}
                self.h5ad_files_pair[h5ad_file] = {"org_path":f"{h5ad_file.split('+')[0]}+DMSO_TF.h5ad", "drug_smiles": drug_smiles}
        self.h5ad_files_pair_keys = list(self.h5ad_files_pair.keys())

        print("dataset len: ", len(self.h5ad_files_pair_keys) * self.NUM_REPETITIONS)

        


    def __len__(self):
        return len(self.h5ad_files_pair_keys) * self.NUM_REPETITIONS
    
    def __getitem__(self, index):
        key_index = index // self.NUM_REPETITIONS
        drug_img_path = self.h5ad_files_pair_keys[key_index]
        org_img_path = self.h5ad_files_pair[drug_img_path]["org_path"]
        smiles = self.h5ad_files_pair[drug_img_path]["drug_smiles"]

        org_img = self.process_h5ad_file(org_img_path)
        drug_img = self.process_h5ad_file(drug_img_path)

        return org_img, drug_img, smiles
    
    def lookup_smiles(self, drug_string):
        matches = self.df[self.df['drug'] == drug_string]['canonical_smiles']
        
        if not matches.empty and matches.iloc[0] not in ["", None]:
            smiles = matches.iloc[0].strip()
            if self.smiles_dot:
                return smiles

            if '.' in smiles:
                parts = [p for p in smiles.split('.') if p] 
                if parts:
                    max_part = max(parts, key=len)
                    return max_part
                else:
                    return None 
            elif "As" in smiles or "none" in smiles or "None" in smiles:
                return None
            else:
                return smiles  
        else:
            return None  


    def process_h5ad_file(self, h5ad_path):  # pseudoimage
        """
        Reads a single .h5ad file, generates pseudo-cells 5 times,
        and saves the results to CSV files.
        Designed to be run in a separate process.
        """
        base_name = os.path.splitext(os.path.basename(h5ad_path))[0]
        # output_csv_base_name = os.path.join(self.OUTPUT_DIR, base_name)
        # process_id = os.getpid() # Get process ID for logging
        # logging.info(f"[PID {process_id}] Processing file: {h5ad_path}")

        try:
            # --- 1. Read Input AnnData File ---
            try:
                adata = ad.read_h5ad(h5ad_path)
                # Basic check for required data
                if 'phase' not in adata.obs.columns:
                    return f"Skipped: Missing 'phase' column in {h5ad_path}"
            except FileNotFoundError:
                return f"Skipped: File not found {h5ad_path}"
            except Exception as e:
                return f"Skipped: Error reading H5AD {h5ad_path} - {e}"

            # --- 2. Calculate Phase Proportions and Target Counts ---
            try:
                phase_counts = adata.obs['phase'].value_counts()
                total_real_cells = phase_counts.sum()
                if total_real_cells == 0:
                    return f"Skipped: No cells in {h5ad_path}"

                proportions = phase_counts / total_real_cells
                target_counts_float = proportions * self.TOTAL_PSEUDO_CELLS
                target_counts = target_counts_float.round().astype(int)
                diff = self.TOTAL_PSEUDO_CELLS - target_counts.sum()

                if diff != 0:
                    if not target_counts.empty:
                        idx_to_adjust = target_counts.idxmax()
                        target_counts[idx_to_adjust] += diff

            except Exception as e: 
                return f"Skipped: Count calculation error {h5ad_path} - {e}"

            # --- Loop for Repetitions ---
            files_generated_count = 0
            # for n in range(1, self.NUM_REPETITIONS + 1):
            pseudo_cell_list = []
            pseudo_cell_phases = []
            generation_successful_rep = True

            # --- 3. Generate Pseudo-cells (Sampling/Averaging) ---
            try:
                for phase in target_counts.index:
                    num_pseudo_for_phase = target_counts.get(phase, 0) # Use .get for safety
                    if num_pseudo_for_phase <= 0:
                        continue

                    real_cell_indices_phase = adata.obs.index[adata.obs['phase'] == phase]
                    n_real_cells_in_phase = len(real_cell_indices_phase)

                    if n_real_cells_in_phase < self.N_CELLS_PER_PSEUDO:
                        dynamic_N_CELLS_PER_PSEUDO = n_real_cells_in_phase
                    else:
                        dynamic_N_CELLS_PER_PSEUDO = self.N_CELLS_PER_PSEUDO
                        

                    for i in range(num_pseudo_for_phase):
                        chosen_indices = np.random.choice(real_cell_indices_phase,
                                                        size=dynamic_N_CELLS_PER_PSEUDO,
                                                        replace=False)
                        pseudo_expression = adata[chosen_indices, :].X.mean(axis=0)

                        if hasattr(pseudo_expression, "A"):
                            pseudo_expression = pseudo_expression.A.flatten()
                        elif isinstance(pseudo_expression, np.matrix):
                            pseudo_expression = np.array(pseudo_expression).flatten()
                        elif not isinstance(pseudo_expression, np.ndarray):
                            pseudo_expression = np.array(pseudo_expression)
                        if pseudo_expression.ndim > 1:
                            pseudo_expression = pseudo_expression.flatten()

                        pseudo_cell_list.append(pseudo_expression)
                        pseudo_cell_phases.append(phase)

            except Exception as e:
                generation_successful_rep = False

            # --- 4. Prepare Data for Saving ---
            if not generation_successful_rep or not pseudo_cell_list:
                return "not generation_successful_rep or not pseudo_cell_list"

            try:
                pseudo_cell_matrix = np.vstack(pseudo_cell_list)
                pseudo_cell_matrix = np.expand_dims(pseudo_cell_matrix, axis=0) 
                pseudo_cell_matrix = np.repeat(pseudo_cell_matrix, repeats=3, axis=0)
                # pseudo_cell_matrix = np.transpose(pseudo_cell_matrix, (0,2,1))
                return pseudo_cell_matrix

                # n_generated = pseudo_cell_matrix.shape[0]

                # pseudo_df = pd.DataFrame(pseudo_cell_matrix, columns=adata.var_names)
                # pseudo_df.insert(0, 'phase', pseudo_cell_phases[:n_generated])
                # pseudo_df.index = [f"pseudo_{i+1}" for i in range(n_generated)]
                # pseudo_df.index.name = "PseudoCellID"

            except Exception as e:
                return "error"

        except Exception as e:
            return f"Failed: Unexpected error in {h5ad_path} - {e}"


class bulk_org_info_Dataset(Dataset):
    def __init__(self, config, smiles_dot=False):
        self.smiles_dot = smiles_dot
        self.config = config
        self.INPUT_DIR = self.config["data_load"]["INPUT_DIR"]
        self.OUTPUT_DIR = self.config["data_load"]["OUTPUT_DIR"]
        self.LOG_FILE = self.config["data_load"]["LOG_FILE"]
        self.TOTAL_PSEUDO_CELLS = self.config["data_load"]["TOTAL_PSEUDO_CELLS"]
        self.N_CELLS_PER_PSEUDO = self.config["data_load"]["N_CELLS_PER_PSEUDO"]
        self.NUM_REPETITIONS = self.config["data_load"]["NUM_REPETITIONS"]
        self.h5ad_files = glob.glob(os.path.join(self.INPUT_DIR, '*.h5ad'))
        self.df = pd.read_parquet(config["drug_metadata_path"])

        with open(config["yes_smiles"]["path"], 'r', encoding='utf-8') as f:
            self.yes_smiles = [line.strip() for line in f]

        

        self.h5ad_files_pair = {}
        for h5ad_file in self.h5ad_files:
            if h5ad_file.split("+")[-1] != "DMSO_TF.h5ad":
                drug_string = os.path.basename(h5ad_file).split("+")[-1].split(".")[0]
                drug_smiles = self.lookup_smiles(drug_string)
                if drug_smiles is None:
                    continue
                # self.h5ad_files_pair[f"{h5ad_file.split('+')[0]}+DMSO_TF.h5ad"] = {"drug_path":h5ad_file, "drug_smiles": drug_smiles}
                self.h5ad_files_pair[h5ad_file] = {"org_path":f"{h5ad_file.split('+')[0]}+DMSO_TF.h5ad", "drug_smiles": drug_smiles}
        self.h5ad_files_pair_keys = list(self.h5ad_files_pair.keys())

        print("dataset len: ", len(self.h5ad_files_pair_keys) * self.NUM_REPETITIONS)

        


    def __len__(self):
        return len(self.h5ad_files_pair_keys) * self.NUM_REPETITIONS
    
    def __getitem__(self, index):
        key_index = index // self.NUM_REPETITIONS
        drug_img_path = self.h5ad_files_pair_keys[key_index]
        org_img_path = self.h5ad_files_pair[drug_img_path]["org_path"]
        smiles = self.h5ad_files_pair[drug_img_path]["drug_smiles"]

        org_img = self.bulk_process_h5ad_file(org_img_path)
        drug_img = self.bulk_process_h5ad_file(drug_img_path)

        # if org_img.shape != (3, 128, 128):
        #     print("org_img", org_img.shape, "  ", org_img_path)
        # elif drug_img.shape != (3, 128, 128):
        #     print("drug_img", drug_img.shape, "  ", drug_img_path)

        return org_img, drug_img, smiles
    
    def lookup_smiles(self, drug_string):
        matches = self.df[self.df['drug'] == drug_string]['canonical_smiles']
        
        if not matches.empty and matches.iloc[0] not in ["", None]:
            smiles = matches.iloc[0].strip()
            if self.smiles_dot:
                return smiles

            if '.' in smiles:
                parts = [p for p in smiles.split('.') if p]  
                if parts:
                    max_part = max(parts, key=len)
                    return max_part
                else:
                    return None  
            elif "As" in smiles or "none" in smiles or "None" in smiles:
                return None
            else:
                return smiles  
        else:
            return None  


    def bulk_process_h5ad_file(self, h5ad_path):
        """
        Reads a single .h5ad file, computes the global average of all cells,
        and returns the result as a (1, G) vector.
        """
        base_name = os.path.splitext(os.path.basename(h5ad_path))[0]

        try:
            # --- 1. Read Input AnnData File ---
            try:
                adata = ad.read_h5ad(h5ad_path)
                if adata.n_obs == 0:
                    return f"Skipped: No cells in {h5ad_path}"
            except FileNotFoundError:
                return f"Skipped: File not found {h5ad_path}"
            except Exception as e:
                return f"Skipped: Error reading H5AD {h5ad_path} - {e}"

            # --- 2. Compute Global Average Expression ---
            try:
                # Compute mean across all cells (axis=0)
                global_avg = adata.X.mean(axis=0)
                
                # Handle different data formats
                if hasattr(global_avg, "A"):  # Sparse matrices
                    global_avg = global_avg.A.flatten()
                elif isinstance(global_avg, np.matrix):
                    global_avg = np.array(global_avg).flatten()
                elif not isinstance(global_avg, np.ndarray):
                    global_avg = np.array(global_avg)
                
                # Ensure we have a 1D vector
                global_avg = global_avg.flatten()
                # print("global_avg:", global_avg.shape)
                
                # Reshape to (1, G)
                return global_avg  # .reshape(1, -1)

            except Exception as e:
                return f"Skipped: Error computing global average {h5ad_path} - {e}"

        except Exception as e:
            return f"Failed: Unexpected error in {h5ad_path} - {e}"


class CpCtlDataset_solo_mask(Dataset):
    def __init__(self, cp_path, ctl_path, data_type, seed=42): # data_type: other,smiles,cell
        self.data_type = data_type
        self.seed = seed 
        np.random.seed(seed)
        
        self.cp_df = pd.read_csv(cp_path)
        self.ctl_df = pd.read_csv(ctl_path) 
        
        if 'look_id' not in self.cp_df.columns or 'look_id' not in self.ctl_df.columns:
            raise ValueError("Both dataframes must contain 'look_id' column")
        
        look_id_to_ctl_features = {}
        for look_id, group in self.ctl_df.groupby('look_id'):
            features = group.iloc[:, -978:].values.astype(np.float32)
            look_id_to_ctl_features[look_id] = features
        
        self.look_id_mean_ctl = {}
        for look_id, features_list in look_id_to_ctl_features.items():
            self.look_id_mean_ctl[look_id] = torch.tensor(
                np.mean(features_list, axis=0), 
                dtype=torch.float32
            )
        
        del look_id_to_ctl_features
        
        valid_look_ids = set(self.look_id_mean_ctl.keys())
        has_match = self.cp_df['look_id'].isin(valid_look_ids)
        no_match_count = len(self.cp_df) - has_match.sum()
        
        if no_match_count > 0:
            print(f"Warning: {no_match_count} rows in CP file have no matching look_id in CTL file and will be removed")
        
        self.cp_df_filtered = self.cp_df[has_match]
        
        if len(self.cp_df_filtered) == 0:
            raise ValueError("No matching look_id found between CP and CTL files after filtering")


        np.random.seed(42)
    
        df_cell = self.cp_df_filtered[self.cp_df_filtered['cell_id'] == "A375"].copy()
        
        unique_smiles = self.cp_df_filtered['smiles'].unique()
        sample_size = max(1, int(np.ceil(len(unique_smiles) * 0.01)))
        sampled_smiles = np.random.choice(unique_smiles, sample_size, replace=False)
        df_smiles = self.cp_df_filtered[self.cp_df_filtered['smiles'].isin(sampled_smiles)].copy()
        
        merged = pd.merge(df_cell, df_smiles, 
                        on=list(self.cp_df_filtered.columns),
                        how='inner')
        dup_idx = merged.index
        
        df_cell_clean = df_cell.drop(dup_idx)
        df_smiles_clean = df_smiles.drop(dup_idx)
        
        other_idx = self.cp_df_filtered.index.difference(
            df_cell.index.union(df_smiles.index)
        )
        df_other = self.cp_df_filtered.loc[other_idx]

        self.df_cell_clean = df_cell_clean
        self.df_smiles_clean = df_smiles_clean
        self.df_other = df_other
        
        self.cp_shuffled_other = self.df_other.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.cp_features_other = torch.tensor(
            self.cp_shuffled_other.iloc[:, -978:].values.astype(np.float32),
            dtype=torch.float32
        )

        self.cp_shuffled_cell = self.df_cell_clean.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.cp_features_cell = torch.tensor(
            self.cp_shuffled_cell.iloc[:, -978:].values.astype(np.float32),
            dtype=torch.float32
        )

        self.cp_shuffled_smiles = self.df_smiles_clean.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.cp_features_smiles = torch.tensor(
            self.cp_shuffled_smiles.iloc[:, -978:].values.astype(np.float32),
            dtype=torch.float32
        )
        
    
    def __len__(self):
        if self.data_type == "other":
            return len(self.cp_shuffled_other)
        elif self.data_type == "cell":
            return len(self.cp_shuffled_cell)
        elif self.data_type == "smiles":
            return len(self.cp_shuffled_smiles)

    def __getitem__(self, idx_real):
        if self.data_type == "other":
            look_id = self.cp_shuffled_other.iloc[idx_real]['look_id']
            mean_ctl_vector = self.look_id_mean_ctl[look_id]
            smiles = self.cp_shuffled_other.iloc[idx_real]["smiles"].strip()
            return mean_ctl_vector, self.cp_features_other[idx_real], smiles
        elif self.data_type == "cell":
            look_id = self.cp_shuffled_cell.iloc[idx_real]['look_id']
            mean_ctl_vector = self.look_id_mean_ctl[look_id]
            smiles = self.cp_shuffled_cell.iloc[idx_real]["smiles"].strip()
            
            return mean_ctl_vector, self.cp_features_cell[idx_real], smiles
        elif self.data_type == "smiles":
            look_id = self.cp_shuffled_smiles.iloc[idx_real]['look_id']
            mean_ctl_vector = self.look_id_mean_ctl[look_id]
            smiles = self.cp_shuffled_smiles.iloc[idx_real]["smiles"].strip()
            
            return mean_ctl_vector, self.cp_features_smiles[idx_real], smiles



class CpCtlDataset_solo(Dataset):
    def __init__(self, cp_path, ctl_path, seed=42):
        self.seed = seed 
        np.random.seed(seed)
        
        self.cp_df = pd.read_csv(cp_path)
        self.ctl_df = pd.read_csv(ctl_path) 
        
        if 'look_id' not in self.cp_df.columns or 'look_id' not in self.ctl_df.columns:
            raise ValueError("Both dataframes must contain 'look_id' column")
        
        look_id_to_ctl_features = {}
        for look_id, group in self.ctl_df.groupby('look_id'):
            features = group.iloc[:, -978:].values.astype(np.float32)
            look_id_to_ctl_features[look_id] = features
        
        self.look_id_mean_ctl = {}
        for look_id, features_list in look_id_to_ctl_features.items():
            self.look_id_mean_ctl[look_id] = torch.tensor(
                np.mean(features_list, axis=0), 
                dtype=torch.float32
            )
        
        del look_id_to_ctl_features
        
        valid_look_ids = set(self.look_id_mean_ctl.keys())
        has_match = self.cp_df['look_id'].isin(valid_look_ids)
        no_match_count = len(self.cp_df) - has_match.sum()
        
        if no_match_count > 0:
            print(f"Warning: {no_match_count} rows in CP file have no matching look_id in CTL file and will be removed")
        
        self.cp_df_filtered = self.cp_df[has_match]
        
        if len(self.cp_df_filtered) == 0:
            raise ValueError("No matching look_id found between CP and CTL files after filtering")
        
        self.cp_shuffled = self.cp_df_filtered.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        self.cp_features = torch.tensor(
            self.cp_shuffled.iloc[:, -978:].values.astype(np.float32),
            dtype=torch.float32
        )
        
        self.smiles_list = self.cp_shuffled['smiles'].str.strip().tolist()
    
    def __len__(self):
        return len(self.cp_shuffled) * 3
    
    def __getitem__(self, idx):
        if idx < len(self.cp_shuffled) * 2:
            is_train = 1
        else:
            is_train = 0
        idx_real = idx % len(self.cp_shuffled)
        look_id = self.cp_shuffled.iloc[idx_real]['look_id']
        mean_ctl_vector = self.look_id_mean_ctl[look_id]
        smiles = self.cp_shuffled.iloc[idx_real]["smiles"].strip()
        
        return mean_ctl_vector, self.cp_features[idx_real], smiles, is_train

class GroupedBatchSampler_l1000(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.type_indices = {0: list(range(len(dataset.cp_features)*2, len(dataset.cp_features)*3)), 1: list(range(0, len(dataset.cp_features)*2))}
    
    def __iter__(self):
        for t in self.type_indices:
            random.shuffle(self.type_indices[t])
        
        batches = []
        for data_type, indices in self.type_indices.items():
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if batch:
                    batches.append(batch)
        
        random.shuffle(batches)
        for batch in batches:
            yield batch
    
    def __len__(self):
        total = 0
        for indices in self.type_indices.values():
            total += int(math.ceil(len(indices) / self.batch_size))
        return total

class module_3_Dataset(Dataset):
    def __init__(self, config, is_train_data):
        self.config = config
        self.INPUT_DIR = self.config["data_load"]["INPUT_DIR"] 
        self.OUTPUT_DIR = self.config["data_load"]["OUTPUT_DIR"]
        self.LOG_FILE = self.config["data_load"]["LOG_FILE"]
        self.TOTAL_PSEUDO_CELLS = self.config["data_load"]["TOTAL_PSEUDO_CELLS"]
        self.N_CELLS_PER_PSEUDO = self.config["data_load"]["N_CELLS_PER_PSEUDO"]
        self.NUM_REPETITIONS = self.config["data_load"]["NUM_REPETITIONS"]
        self.h5ad_files = glob.glob(os.path.join(self.INPUT_DIR, '*.h5ad'))
        # df
        self.df = pd.read_parquet(config["drug_metadata_path"])

        self.all_legal_smiles = self.get_legal_smiles(self.config["unit_3_train"]["all_smiles_path"])
        self.t_legal_smiles = self.get_legal_smiles(self.config["unit_3_train"]["t_smiles_path"])
        self.v_legal_smiles = self.get_legal_smiles(self.config["unit_3_train"]["v_smiles_path"])


        self.h5ad_files_pair = []
        self.val_smiles = []
        for h5ad_file in self.h5ad_files:
            if h5ad_file.split("+")[-1] != "DMSO_TF.h5ad":
                drug_string = os.path.basename(h5ad_file).split("+")[-1].split(".")[0]
                drug_smiles = self.lookup_smiles(drug_string)
                if is_train_data:
                    if drug_smiles is None:
                        continue
                    if drug_smiles in self.all_legal_smiles and not self.config["unit_3_train"]["is_frozen"]: # dataset只有一次
                        self.h5ad_files_pair.append({"drug_path": h5ad_file, "org_path":f"{h5ad_file.split('+')[0]}+DMSO_TF.h5ad", "drug_smiles": drug_smiles, "is_train": 0})
                    if drug_smiles in self.t_legal_smiles: # dataset只有一次
                        self.h5ad_files_pair.append({"drug_path": h5ad_file, "org_path":f"{h5ad_file.split('+')[0]}+DMSO_TF.h5ad", "drug_smiles": drug_smiles, "is_train": 1})
                        # self.h5ad_files_pair.append({"drug_path": h5ad_file, "org_path":f"{h5ad_file.split('+')[0]}+DMSO_TF.h5ad", "drug_smiles": drug_smiles, "is_train": 1})
                        # self.h5ad_files_pair.append({"drug_path": h5ad_file, "org_path":f"{h5ad_file.split('+')[0]}+DMSO_TF.h5ad", "drug_smiles": drug_smiles, "is_train": 1})
                else:
                    if drug_smiles not in self.val_smiles:
                        self.val_smiles.append(drug_smiles)
                        if drug_smiles in self.v_legal_smiles: 
                            self.h5ad_files_pair.append({"drug_path": h5ad_file, "org_path":f"{h5ad_file.split('+')[0]}+DMSO_TF.h5ad", "drug_smiles": drug_smiles, "is_train": 0})
                        elif drug_smiles in self.t_legal_smiles: 
                            self.h5ad_files_pair.append({"drug_path": h5ad_file, "org_path":f"{h5ad_file.split('+')[0]}+DMSO_TF.h5ad", "drug_smiles": drug_smiles, "is_train": 1})
        if is_train_data:
            self.h5ad_files_pair = self.h5ad_files_pair * self.NUM_REPETITIONS

        print("dataset len: ", len(self.h5ad_files_pair))


    def __len__(self):
        return len(self.h5ad_files_pair)
    
    def __getitem__(self, index):
        key_index = index 
        drug_img_path = self.h5ad_files_pair[key_index]["drug_path"]
        org_img_path = self.h5ad_files_pair[key_index]["org_path"]
        smiles = self.h5ad_files_pair[key_index]["drug_smiles"]
        is_train = self.h5ad_files_pair[key_index]["is_train"]

        org_img = self.process_h5ad_file(org_img_path)
        drug_img = self.process_h5ad_file(drug_img_path)

        return org_img, drug_img, smiles, is_train

    def get_legal_smiles(self, txt_path):
        with open(txt_path, 'r') as file:
            return [line.strip() for line in file]
    
    def lookup_smiles(self, drug_string):
        matches = self.df[self.df['drug'] == drug_string]['canonical_smiles']
        
        if not matches.empty and matches.iloc[0] not in ["", None]:
            smiles = matches.iloc[0].strip()
            
            if '.' in smiles:
                parts = [p for p in smiles.split('.') if p]  
                if parts:
                    max_part = max(parts, key=len)
                    if "none" not in max_part and "None" not in max_part:
                        return max_part
                    else:
                        return None
                else:
                    return None  
            elif "As" in smiles or "none" in smiles or "None" in smiles:
                return None
            else:
                return smiles  
        else:
            return None 


    def process_h5ad_file(self, h5ad_path):
        """
        Reads a single .h5ad file, generates pseudo-cells 5 times,
        and saves the results to CSV files.
        Designed to be run in a separate process.
        """
        base_name = os.path.splitext(os.path.basename(h5ad_path))[0]
        # output_csv_base_name = os.path.join(self.OUTPUT_DIR, base_name)
        # process_id = os.getpid() # Get process ID for logging
        # logging.info(f"[PID {process_id}] Processing file: {h5ad_path}")

        try:
            # --- 1. Read Input AnnData File ---
            try:
                adata = ad.read_h5ad(h5ad_path)
                # Basic check for required data
                if 'phase' not in adata.obs.columns:
                    return f"Skipped: Missing 'phase' column in {h5ad_path}"
            except FileNotFoundError:
                return f"Skipped: File not found {h5ad_path}"
            except Exception as e:
                return f"Skipped: Error reading H5AD {h5ad_path} - {e}"

            # --- 2. Calculate Phase Proportions and Target Counts ---
            try:
                phase_counts = adata.obs['phase'].value_counts()
                total_real_cells = phase_counts.sum()
                if total_real_cells == 0:
                    return f"Skipped: No cells in {h5ad_path}"

                proportions = phase_counts / total_real_cells
                target_counts_float = proportions * self.TOTAL_PSEUDO_CELLS
                target_counts = target_counts_float.round().astype(int)
                diff = self.TOTAL_PSEUDO_CELLS - target_counts.sum()

                if diff != 0:
                    if not target_counts.empty:
                        idx_to_adjust = target_counts.idxmax()
                        target_counts[idx_to_adjust] += diff

            except Exception as e:
                return f"Skipped: Count calculation error {h5ad_path} - {e}"

            # --- Loop for Repetitions ---
            files_generated_count = 0
            pseudo_cell_list = []
            pseudo_cell_phases = []
            generation_successful_rep = True

            # --- 3. Generate Pseudo-cells (Sampling/Averaging) ---
            try:
                for phase in target_counts.index:
                    num_pseudo_for_phase = target_counts.get(phase, 0) # Use .get for safety
                    if num_pseudo_for_phase <= 0:
                        continue

                    real_cell_indices_phase = adata.obs.index[adata.obs['phase'] == phase]
                    n_real_cells_in_phase = len(real_cell_indices_phase)

                    if n_real_cells_in_phase < self.N_CELLS_PER_PSEUDO:
                        dynamic_N_CELLS_PER_PSEUDO = n_real_cells_in_phase
                    else:
                        dynamic_N_CELLS_PER_PSEUDO = self.N_CELLS_PER_PSEUDO
                        

                    for i in range(num_pseudo_for_phase):
                        chosen_indices = np.random.choice(real_cell_indices_phase,
                                                        size=dynamic_N_CELLS_PER_PSEUDO,
                                                        replace=False)
                        pseudo_expression = adata[chosen_indices, :].X.mean(axis=0)

                        if hasattr(pseudo_expression, "A"):
                            pseudo_expression = pseudo_expression.A.flatten()
                        elif isinstance(pseudo_expression, np.matrix):
                            pseudo_expression = np.array(pseudo_expression).flatten()
                        elif not isinstance(pseudo_expression, np.ndarray):
                            pseudo_expression = np.array(pseudo_expression)
                        if pseudo_expression.ndim > 1:
                            pseudo_expression = pseudo_expression.flatten()

                        pseudo_cell_list.append(pseudo_expression)
                        pseudo_cell_phases.append(phase)

            except Exception as e:
                generation_successful_rep = False

            # --- 4. Prepare Data for Saving ---
            if not generation_successful_rep or not pseudo_cell_list:
                return "not generation_successful_rep or not pseudo_cell_list"

            try:
                pseudo_cell_matrix = np.vstack(pseudo_cell_list)
                pseudo_cell_matrix = np.expand_dims(pseudo_cell_matrix, axis=0) 
                pseudo_cell_matrix = np.repeat(pseudo_cell_matrix, repeats=3, axis=0)
                # pseudo_cell_matrix = np.transpose(pseudo_cell_matrix, (0,2,1))
                return pseudo_cell_matrix

                # n_generated = pseudo_cell_matrix.shape[0]

                # pseudo_df = pd.DataFrame(pseudo_cell_matrix, columns=adata.var_names)
                # pseudo_df.insert(0, 'phase', pseudo_cell_phases[:n_generated])
                # pseudo_df.index = [f"pseudo_{i+1}" for i in range(n_generated)]
                # pseudo_df.index.name = "PseudoCellID"

            except Exception as e:
                return "error"

        except Exception as e:
            return f"Failed: Unexpected error in {h5ad_path} - {e}"




class GroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.type_indices = {0: [], 1: []}
        for idx in range(len(dataset.h5ad_files_pair)):
            data_type = dataset.h5ad_files_pair[idx]["is_train"]
            self.type_indices[data_type].append(idx)
    
    def __iter__(self):
        for t in self.type_indices:
            random.shuffle(self.type_indices[t])
        
        batches = []
        for data_type, indices in self.type_indices.items():
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if batch:
                    batches.append(batch)
        
        random.shuffle(batches)
        for batch in batches:
            yield batch
    
    def __len__(self):
        total = 0
        for indices in self.type_indices.values():
            total += int(math.ceil(len(indices) / self.batch_size))
        return total


def test_csv():
    csv_files = [
        "molgene/data/cell_img/testImage/CVCL_C466+Trametinib (DMSO_TF solvate)+1.csv", # replace with yours
        "molgene/data/cell_img/testImage/CVCL_C466+Trametinib (DMSO_TF solvate)+2.csv", # replace with yours
        "molgene/data/cell_img/testImage/CVCL_C466+Trametinib (DMSO_TF solvate)+3.csv"  # replace with yours
    ]
    
    csv_files = "molgene/data/cell_img/testImage"  # replace with yours
    dataset = MatrixDataset(csv_files)

    test = dataset[0]
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    for batch in dataloader:
        print(f"batch shape: {batch.shape}")  
        print(f"data type: {batch.dtype}") 
        break


def test_org():
    with open("./config/config.yaml", "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)
    dataset = org_info_Dataset(config_dict)
    test = dataset[0]


if __name__ == "__main__":
    pass