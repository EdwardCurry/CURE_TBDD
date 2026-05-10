from analysis.rdkit_functions import compute_molecular_metrics
from mini_moses.metrics.metrics import compute_intermediate_statistics
from metrics.property_metric import TaskModel

import torch
import torch.nn as nn

import os
import csv
import time

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Draw
from rdkit.Chem.Fraggle.FraggleSim import GetFraggleSimilarity

from rdkit.Chem import DataStructs  
from rdkit import RDLogger


RDLogger.DisableLog('rdApp.*')

def calculate_similarity(smiles1, smiles2):

    fraggle_sim, morgan_sim, maccs_sim = 0.0, 0.0, 0.0
    
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return (fraggle_sim, morgan_sim, maccs_sim)
        
        fraggle_sim = GetFraggleSimilarity(mol1, mol2)[0]
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        morgan_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        
        fp1_maccs = MACCSkeys.GenMACCSKeys(mol1)
        fp2_maccs = MACCSkeys.GenMACCSKeys(mol2)
        maccs_sim = DataStructs.TanimotoSimilarity(fp1_maccs, fp2_maccs)
    
    except:
        pass
        
    return (fraggle_sim, morgan_sim, maccs_sim)

def compare_smiles_lists(smiles_list1, smiles_list2, output_csv='similarity_results.csv'):


    if len(smiles_list1) != len(smiles_list2):
        raise ValueError("Error")
    
    results = []
    
    for i, (s1, s2) in enumerate(zip(smiles_list1, smiles_list2)):
        if s2 is None or s2 == "":
            continue
        fraggle, morgan, maccs = calculate_similarity(s1, s2)
        
        results.append({
            'Index': i+1,
            'SMILES_1': s1,
            'SMILES_2': s2,
            'Fraggle_Similarity': round(fraggle, 4),
            'Morgan_Similarity': round(morgan, 4),
            'MACCS_Similarity': round(maccs, 4)
        })
        
        if (i+1) % 10 == 0:
            print(f'Processed {i+1}/{len(smiles_list1)} pairs')
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(compare_smiles_lists)
    avg_fraggle = df['Fraggle_Similarity'].mean()
    avg_morgan = df['Morgan_Similarity'].mean()
    avg_maccs = df['MACCS_Similarity'].mean()
    
    print('\n' + '='*50)
    print(f'Average Fraggle Similarity: {avg_fraggle:.4f}')
    print(f'Average Morgan Similarity: {avg_morgan:.4f}')
    print(f'Average MACCS Similarity: {avg_maccs:.4f}')
    print('='*50)
    
    return df

def result_to_csv(path, dict_data):
    file_exists = os.path.exists(path)
    log_name = dict_data.pop("log_name", None)
    if log_name is None:
        raise ValueError("The provided dictionary must contain a 'log_name' key.")
    field_names = ["log_name"] + list(dict_data.keys())
    dict_data["log_name"] = log_name
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()
        writer.writerow(dict_data)


class SamplingMolecularMetrics(nn.Module):
    def __init__(
        self,
        dataset_infos,
        train_smiles,
        reference_smiles,
        n_jobs=1,
        device="cpu",
        batch_size=512,
    ):
        super().__init__()
        self.task_name = dataset_infos.task
        self.dataset_infos = dataset_infos
        self.active_atoms = dataset_infos.active_atoms
        self.train_smiles = train_smiles

        if reference_smiles is not None:
            print(
                f"--- Computing intermediate statistics for training for #{len(reference_smiles)} smiles ---"
            )
            start_time = time.time()
            self.stat_ref = compute_intermediate_statistics(
                reference_smiles, n_jobs=n_jobs, device=device, batch_size=batch_size
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f"--- End computing intermediate statistics: using {elapsed_time:.2f}s ---"
            )
        else:
            self.stat_ref = None
    
        self.comput_config = {
            "n_jobs": n_jobs,
            "device": device,
            "batch_size": batch_size,
        }

        self.task_evaluator = {'meta_taskname': dataset_infos.task, 'sas': None, 'scs': None}
        # Temporarily remove the evaluation model, and add the subsequent evaluation model here
        if "smiles" in dataset_infos.task:
            self.task_evaluator["QED_score"] = None
            self.task_evaluator["pass_ro5"] = None
            self.task_evaluator["logP"] = None
            self.task_evaluator["MW"] = None
            self.task_evaluator["HBD"] = None
            self.task_evaluator["HBA"] = None
            self.task_evaluator["TPSA"] = None
            self.task_evaluator["nRot"] = None
            self.task_evaluator["unique"] = None
        else:
            for cur_task in dataset_infos.task.split("-")[:]:
                model_path = os.path.join(
                    dataset_infos.base_path, "data/evaluator", f"{cur_task}.joblib"
                )
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                evaluator = TaskModel(model_path, cur_task)
                self.task_evaluator[cur_task] = evaluator

    def forward(self, molecules, targets, name, current_epoch, val_counter, test=False):  
        if isinstance(targets, list):
            targets_cat = torch.cat(targets, dim=0)
            targets_np = targets_cat.detach().cpu().numpy()
        else:
            targets_np = targets.detach().cpu().numpy()

        if "smiles" in self.task_name:
            unique_smiles, all_smiles, all_metrics, targets_log = compute_molecular_metrics(
                molecules,
                targets_np,
                self.train_smiles,
                self.stat_ref,
                self.dataset_infos,
                self.task_evaluator,
                self.comput_config,
            )
        else:
            unique_smiles, all_smiles, all_metrics, targets_log = compute_molecular_metrics(
                molecules,
                targets_np,
                self.train_smiles,
                self.stat_ref,
                self.dataset_infos,
                self.task_evaluator,
                self.comput_config,
            )

        if test:
            file_name = "final_smiles.txt"
            with open(file_name, "w") as fp:
                all_tasks_name = list(self.task_evaluator.keys())
                all_tasks_name = all_tasks_name.copy()
                if 'meta_taskname' in all_tasks_name:
                    all_tasks_name.remove('meta_taskname')
                if 'scs' in all_tasks_name:
                    all_tasks_name.remove('scs')

                all_tasks_str = "smiles, " + ", ".join([f"input_{task}" for task in all_tasks_name] + [f"output_{task}" for task in all_tasks_name])
                fp.write(all_tasks_str + "\n")
                for i, smiles in enumerate(all_smiles):
                    if targets_log is not None:
                        all_result_str = f"{smiles}, " + ", ".join([f"{targets_log['input_'+task][i]}" for task in all_tasks_name] + [f"{targets_log['output_'+task][i]}" for task in all_tasks_name])
                        fp.write(all_result_str + "\n")
                    else:
                        fp.write("%s\n" % smiles)
                print("All smiles saved")
        else:
            result_path = os.path.join(os.getcwd(), f"graphs/{name}")
            os.makedirs(result_path, exist_ok=True)
            text_path = os.path.join(
                result_path,
                f"valid_unique_molecules_e{current_epoch}_b{val_counter}.txt",
            )
            textfile = open(text_path, "w")
            for smiles in unique_smiles:
                textfile.write(smiles + "\n")
            textfile.close()

        all_logs = all_metrics
        if test:
            all_logs["log_name"] = "test"
        else:
            all_logs["log_name"] = (
                "epoch" + str(current_epoch) + "_batch" + str(val_counter)
            )
    
        return all_smiles

    def reset(self):
        pass

if __name__ == "__main__":
    pass