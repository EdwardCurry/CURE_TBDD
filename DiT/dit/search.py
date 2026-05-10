import os
import pandas as pd
from FPSim2 import FPSim2Engine
from FPSim2.io import create_db_file
from rdkit import Chem
from collections import Counter
import sys

def get_top_k_sim_smiles_list(query_smiles_list, k=20):
    top_k_result_list = []
    sim_avg_all = 0
    none_num = 0
    for smiles in query_smiles_list:
        if smiles is not None:
            top_k_result, sim_avg = get_top_k_sim_smiles(smiles, k=k)
            top_k_result_list.append(top_k_result)
            sim_avg_all = sim_avg + sim_avg_all
        else:
            top_k_result_list.append(None)
            none_num = none_num + 1
    return top_k_result_list, sim_avg_all / (len(query_smiles_list) - none_num)



def get_top_k_sim_smiles(query_smiles, k=20):
    file_path = os.path.join(os.path.dirname(sys.path[0]), f"data/raw/smiles_feat_dit-mse-prnet_uni.csv") 
    db_file = os.path.join(os.path.dirname(sys.path[0]), f"data/search/smiles_feat_dit-mse-prnet_uni.h5") 
    df = pd.read_csv(file_path)
    fpe = FPSim2Engine(db_file)

    results_topk = fpe.top_k(
        query_smiles,
        k=k,
        threshold=0.0,
        metric='tanimoto',
        n_workers=4
    )

    top_k_result = []
    sim_all = 0
    if len(results_topk) > 0:
        for mol_id, sim in results_topk:
            sim_all = sim_all + sim
            smi = df.loc[mol_id, 'smiles']
            top_k_result.append({"smi": smi, "sim": sim})
    else:
        print("None")
    if len(results_topk) > 0:
        sim_avg = sim_all / len(results_topk)
    else:
        sim_avg = 0

    return top_k_result, sim_avg


def get_goal_radio(top_k_result_list, org_smiles):
    goal_list = []
    none_num = 0
    for top_k_result, org_sim in zip(top_k_result_list, org_smiles):
        if top_k_result is not None:
            top_k_result_all_smiles = [meta["smi"] for meta in top_k_result]
            if org_sim in top_k_result_all_smiles:
                goal_list.append(True)
            else:
                goal_list.append(False)
        else:
            goal_list.append(True)
            none_num = none_num + 1
    counts = Counter(goal_list)
    total = len(org_smiles) - none_num
    return (total - counts[False]) / total, goal_list

