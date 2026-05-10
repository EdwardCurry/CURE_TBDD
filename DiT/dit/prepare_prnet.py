import pandas as pd
import anndata
import numpy as np
import re
from tqdm import tqdm

def process_files(gxmol_path, siginfo_path, ctl_path, h5ad_path, output_path):
    gxmol_df = pd.read_csv(gxmol_path)
    siginfo_df = pd.read_csv(siginfo_path)
    ctl_df = pd.read_csv(ctl_path)
    adata = anndata.read_h5ad(h5ad_path) 
    
    results = []
    

    for _, row in tqdm(gxmol_df.iterrows()):
        smiles = row['label']
        pre_smiles = row['valid']
        sig_nature = row['sig_nature']

        siginfo_match = siginfo_df[siginfo_df['sig_id'] == sig_nature]
        
        if siginfo_match.empty:
            continue
            
        plates_mate = siginfo_match.iloc[0]['det_plates']
        plates_list = plates_mate.split('|') if isinstance(plates_mate, str) else []
        plates_mate_1 = plates_list[0].strip() if plates_list else None
        
        if not plates_mate_1:
            continue
            
        matched_index = None
        for idx in adata.obs.index:
            if plates_mate_1 in idx:
                matched_index = idx
                break

        if matched_index is None:
            continue
            

        row_idx = adata.obs.index.get_loc(matched_index)
        sample_features = adata.X[row_idx]  
        sample_features = [meta for meta in sample_features]
        

        ctl_match = None
        

        if "index" in ctl_df.columns:
            pattern = re.compile(re.escape(plates_mate_1), re.IGNORECASE)
            
            for idx, value in ctl_df["index"].items():
                if pd.isna(value):
                    continue
                if pattern.search(str(value)):
                    ctl_match = [meta for meta in ctl_df.iloc[idx][1:]]
                    break

        if ctl_match is None:

            continue
        else:
            dict_mate = {'smiles': smiles, 'pre_smiles': pre_smiles}
            for idx, cp_meta in enumerate(sample_features):
                dict_mate[f"cp_{idx}"] = cp_meta
            for idx, ctl_meta in enumerate(ctl_match):
                dict_mate[f"ctl_{idx}"] = ctl_meta
            results.append(dict_mate)

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

