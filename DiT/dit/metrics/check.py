import anndata
import pandas as pd
import numpy as np
from tqdm import tqdm

def check_drug_dose_consistency_with_mode(h5ad_path, csv_path, data_type="ours"):
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

    if data_type == "ours":
        if 'smiles' not in csv_df.columns:
            return {"error": "drug_id column is missing in CSV file"}, {}
        unique_drug_ids = csv_df['smiles'].unique()
    elif data_type == "gexmolgen":
        if 'label' not in csv_df.columns:
            return {"error": "drug_id column is missing in CSV file"}, {}
        unique_drug_ids = csv_df['label'].unique()
    elif data_type == "gx2mol":
        if 'smiles' not in csv_df.columns:
            return {"error": "drug_id column is missing in CSV file"}, {}
        unique_drug_ids = csv_df['smiles'].unique()
    elif data_type == "triomphe":
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

if __name__ == "__main__":
    pass
    



