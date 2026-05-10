import os
import csv
import anndata
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime


def check_drug_dose_consistency_with_mode(h5ad_path, csv_path):
    try:
        adata = anndata.read_h5ad(h5ad_path)
        csv_df = pd.read_csv(csv_path)
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return {"error": error_msg}, {}
    
    if 'canonical_smiles' not in adata.obs.columns:
        return {"error": "The pert_id column is missing from the obs file in the h5ad file"}, {}
    if 'pert_dose' not in adata.obs.columns:
        return {"error": "The pert_dose column is missing from the obs file in the h5ad file"}, {}
    if 'smiles' not in csv_df.columns:
        return {"error": "drug_id column is missing in CSV file"}, {}
    
    unique_drug_ids = csv_df['smiles'].unique()
    h5ad_pert_ids = set(adata.obs['canonical_smiles'].unique())
    
    result_dict = {
        'consistent_drugs': [],
        'inconsistent_drugs': [],
        'missing_drugs': [],
        'summary': {
            'total_drugs': len(unique_drug_ids),
            'consistent_count': 0,
            'inconsistent_count': 0,
            'missing_count': 0
        }
    }
    
    dose_dict = {}
    
    for smiles in tqdm(unique_drug_ids):
        if smiles not in h5ad_pert_ids:
            result_dict['missing_drugs'].append(smiles)
            result_dict['summary']['missing_count'] += 1
            dose_dict[smiles] = None  
            continue
            
        drug_records = adata.obs[adata.obs['canonical_smiles'] == smiles]
        
        dose_counts = drug_records['pert_dose'].value_counts()
        most_common_dose = dose_counts.idxmax()
        max_count = dose_counts.max()
        
        dose_dict[smiles] = most_common_dose
        
        unique_doses = drug_records['pert_dose'].unique()
        
        if len(unique_doses) == 1:
            result_dict['consistent_drugs'].append(smiles)
            result_dict['summary']['consistent_count'] += 1
        else:
            result_dict['inconsistent_drugs'].append({
                'smiles': smiles,
                'unique_doses': list(unique_doses),
                'most_common_dose': most_common_dose,
                'most_common_count': max_count,
                'sample_count': len(drug_records),
                'dose_distribution': dose_counts.to_dict()
            })
            result_dict['summary']['inconsistent_count'] += 1
    
    return result_dict, dose_dict


def PRnet_metrics(org_index, feat_true_csv, dose_dict, all_smiles, PRnet, org_smiles, siginfo_beta=None, data_type="ours"):

    try:
        feat_df = pd.read_csv(feat_true_csv)
        print(f"{feat_true_csv}")
    except Exception as e:
        print(f"Error {feat_true_csv}")
        print(e)
        return []
    
    if len(org_index) != len(all_smiles):
        print(f"Error")
        return []
    
    if data_type=="ours":
        ctl_columns = [col for col in feat_df.columns if col.startswith('ctl_')]
        if not ctl_columns:
            print("Error: No column starting with 'ctl_' found in CSV file")
            return []
        cp_columns = [col for col in feat_df.columns if col.startswith('cp_')]
        if not cp_columns:
            print("Error: No column starting with 'ctl_' found in CSV file")
            return []
    elif data_type=="gexmolgen":
        columns = feat_df.columns.tolist()
        if len(columns) < 982 + 977:
            raise ValueError("Error")
        new_columns = columns[:3]
        for i in range(3, 981):
            new_columns.append(f"cp_{i-3}")
        for i in range(981, 981 + 978):
            new_columns.append(f"ctl_{i-981}")
        feat_df.columns = new_columns
        ctl_columns = [col for col in feat_df.columns if col.startswith('ctl_')]
        if not ctl_columns:
            print("Error: No column starting with 'ctl_' found in CSV file")
            return []
        cp_columns = [col for col in feat_df.columns if col.startswith('cp_')]
        if not cp_columns:
            print("Error: No column starting with 'ctl_' found in CSV file")
            return []
    elif data_type=="gx2mol":
        siginfo_beta_df = pd.read_csv(siginfo_beta)

    
    pred_list = []
    cp_true_list = []
    result_list = []

    for idx, smiles, smiles_org in tqdm(zip(org_index, all_smiles, org_smiles), 
                                total=len(org_index), 
                                desc="Processing"):

        row = feat_df.iloc[idx]
        if row.empty:
            print(f"warning")
            continue
            
        ctl_true = row[ctl_columns].values.flatten().astype(float)
        ctl_true = np.expand_dims(ctl_true, axis=0)  

        cp_true = row[cp_columns].values.flatten().astype(float)
        cp_true = np.expand_dims(cp_true, axis=0)  
        
        dose = dose_dict[smiles_org]
        if dose and smiles is not None:
            pred_pre_vector = PRnet.infer(smiles, dose, ctl_true)
            pred_org_vector = PRnet.infer(smiles_org, dose, ctl_true)
            comp_t_org = compare_vec(cp_true, pred_org_vector)
            comp_t_pre = compare_vec(cp_true, pred_pre_vector)
            comp_org_pre = compare_vec(pred_org_vector, pred_pre_vector)

            result_list.append({"smiles_org": smiles_org, "smiles_p":smiles ,"pred_pre_vector":pred_pre_vector, "pred_org_vector":pred_org_vector, "cp_true":cp_true, "comp_t_org":comp_t_org, "comp_t_pre":comp_t_pre, "comp_org_pre":comp_org_pre})
        else:
            pred_vector = None

            result_list.append({"smiles_org": smiles_org, "smiles_p":smiles ,"pred_pre_vector":None, "pred_org_vector":None, "cp_true":None, "comp_t_org":None, "comp_t_pre":None, "comp_org_pre":None})
        
    
    return result_list


def PRnet_metrics_top_k(org_index, feat_true_csv, dose_dict, top_k_result_list, PRnet, org_smiles, siginfo_beta=None, data_type="ours"):
    try:
        feat_df = pd.read_csv(feat_true_csv)
    except Exception as e:
        print(e)
        return []
    
    if len(org_index) != len(top_k_result_list):
        return []
    
    if data_type=="ours":
        ctl_columns = [col for col in feat_df.columns if col.startswith('ctl_')]
        if not ctl_columns:
            return []
        cp_columns = [col for col in feat_df.columns if col.startswith('cp_')]
        if not cp_columns:
            return []
    elif data_type=="gexmolgen":
        columns = feat_df.columns.tolist()
        if len(columns) < 982 + 977:
            raise ValueError(f"Error")
        new_columns = columns[:3]
        for i in range(3, 981):
            new_columns.append(f"cp_{i-3}")
        for i in range(981, 981 + 978):
            new_columns.append(f"ctl_{i-981}")
        feat_df.columns = new_columns
        ctl_columns = [col for col in feat_df.columns if col.startswith('ctl_')]
        if not ctl_columns:
            return []
        cp_columns = [col for col in feat_df.columns if col.startswith('cp_')]
        if not cp_columns:
            return []
    elif data_type=="gx2mol":
        siginfo_beta_df = pd.read_csv(siginfo_beta)

    pred_list = []
    cp_true_list = []
    result_list = []

    for idx, meta_smiles, smiles_org in tqdm(zip(org_index, top_k_result_list, org_smiles), 
                                total=len(org_index), 
                                desc="processing"):

        row = feat_df.iloc[idx]
        if row.empty:
            print(f"warning")
            continue
        dose = dose_dict[smiles_org]
        ctl_true = row[ctl_columns].values.flatten().astype(float)
        ctl_true = np.expand_dims(ctl_true, axis=0)  

        cp_true = row[cp_columns].values.flatten().astype(float)
        cp_true = np.expand_dims(cp_true, axis=0)  
        mse_min = 999999
        temp_res = {"smiles_org": smiles_org, "smiles_p":"None" ,"pred_pre_vector":None, "pred_org_vector":None, "cp_true":None, "comp_t_org":None, "comp_t_pre":None, "comp_org_pre":None}
        if meta_smiles is not None:
            for meta in meta_smiles:
                smiles = meta["smi"]

                if dose and smiles is not None:
                    pred_pre_vector = PRnet.infer(smiles, dose, ctl_true)
                    pred_org_vector = PRnet.infer(smiles_org, dose, ctl_true)
                    comp_t_org = compare_vec(cp_true, pred_org_vector)
                    comp_t_pre = compare_vec(cp_true, pred_pre_vector)
                    comp_org_pre = compare_vec(pred_org_vector, pred_pre_vector)
                    if comp_org_pre["mse_score"] < mse_min:
                        mse_min = comp_org_pre["mse_score"]
                        temp_res = {"smiles_org": smiles_org, "smiles_p":smiles ,"pred_pre_vector":pred_pre_vector, "pred_org_vector":pred_org_vector, "cp_true":cp_true, "comp_t_org":comp_t_org, "comp_t_pre":comp_t_pre, "comp_org_pre":comp_org_pre}
                else:
                    pred_vector = None

        result_list.append(temp_res)
        
    
    return result_list


def PRnet_metrics_for_others(feat_true_csv, dose_dict, PRnet, data_type="ours"):

    try:
        feat_df = pd.read_csv(feat_true_csv)
    except Exception as e:
        print(e)
        return []
    
    if data_type=="ours" or data_type=="gx2mol" or data_type=="triomphe":
        ctl_columns = [col for col in feat_df.columns if col.startswith('ctl_')]
        if not ctl_columns:
            return []
        cp_columns = [col for col in feat_df.columns if col.startswith('cp_')]
        if not cp_columns:
            return []
    elif data_type=="gexmolgen":
        columns = feat_df.columns.tolist()
        if len(columns) < 982 + 977:
            raise ValueError(f"Error")
        new_columns = ["index", "smiles", "pre_smiles"]
        for i in range(3, 981):
            new_columns.append(f"cp_{i-3}")
        new_columns.append("index")
        for i in range(981, 981 + 978):
            new_columns.append(f"ctl_{i-981}")
        feat_df.columns = new_columns
        ctl_columns = [col for col in feat_df.columns if col.startswith('ctl_')]
        if not ctl_columns:
            return []
        cp_columns = [col for col in feat_df.columns if col.startswith('cp_')]
        if not cp_columns:
            return []



    pred_list = []
    cp_true_list = []
    result_list = []


    for i, (index, row) in tqdm(enumerate(feat_df.iterrows())):
        smiles = row["pre_smiles"]
        smiles_org = row["smiles"]
            
        ctl_true = row[ctl_columns].values.flatten().astype(float)
        ctl_true = np.expand_dims(ctl_true, axis=0)  

        cp_true = row[cp_columns].values.flatten().astype(float)
        cp_true = np.expand_dims(cp_true, axis=0)  
        
        dose = dose_dict[smiles_org]
        if dose and smiles is not None:
            pred_pre_vector = PRnet.infer(smiles, dose, ctl_true)
            pred_org_vector = PRnet.infer(smiles_org, dose, ctl_true)
            comp_t_org = compare_vec(cp_true, pred_org_vector)
            comp_t_pre = compare_vec(cp_true, pred_pre_vector)
            comp_org_pre = compare_vec(pred_org_vector, pred_pre_vector)

            result_list.append({"smiles_org": smiles_org, "smiles_p":smiles ,"pred_pre_vector":pred_pre_vector, "pred_org_vector":pred_org_vector, "cp_true":cp_true, "comp_t_org":comp_t_org, "comp_t_pre":comp_t_pre, "comp_org_pre":comp_org_pre})
        else:
            pred_vector = None

            result_list.append({"smiles_org": smiles_org, "smiles_p":smiles ,"pred_pre_vector":None, "pred_org_vector":None, "cp_true":None, "comp_t_org":None, "comp_t_pre":None, "comp_org_pre":None})
        
    
    return result_list


def compare_vec(vec_1, vec_2):
    yp_m = vec_1.mean(axis=0)
    yp_v = vec_1.var(axis=0)

    yt_m = vec_2.mean(axis=0)
    yt_v = vec_2.var(axis=0)


    mse_score =  mean_squared_error(vec_1, vec_2)
    r2_score_ = r2_score(vec_1[0], vec_2[0])
    pearson_score_ , _ = pearsonr(vec_1.flatten(), vec_2.flatten())
    r2_score_mean = r2_score(yt_m, yp_m)
    r2_score_var = r2_score(yt_v, yp_v)

    return {"mse_score":mse_score, "r2_score": r2_score_, "pearson_score_": pearson_score_, "r2_score_mean": r2_score_mean, "r2_score_var": r2_score_var}





def prnet_2_csv(dict_list, output_file, metadata=None):

    headers = ['comp_org_pre_mse_score']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        skipped_count = 0 
        valid_rows = 0   
        
        for idx, data_dict in enumerate(dict_list):
          
            row = {}
            
            comp_dict = data_dict.get('comp_org_pre')

            if comp_dict is None:
                skipped_count += 1
                continue
            elif not isinstance(comp_dict, dict):
                print(f"Warning")
                skipped_count += 1
                continue
            else:
                mse_score = comp_dict.get('mse_score')
                
                if mse_score is None or mse_score == '':
                    skipped_count += 1
                    continue
                
                row['comp_org_pre_mse_score'] = mse_score
            
            writer.writerow(row)
            valid_rows += 1
    return output_file




if __name__ == "__main__":
    pass
