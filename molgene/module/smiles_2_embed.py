from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
import torch
import numpy as np

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

def smiles_list_to_fp_tensor(smiles_list: list, 
                           fp_type: str = 'countvect', 
                           radius: int = 2, 
                           n_bits: int = 2048,
                           show_progress: bool = False) -> torch.Tensor:

    if not isinstance(smiles_list, list):
        raise TypeError("should be smiles list")
    
    if len(smiles_list) == 0:
        return torch.empty((0, n_bits), dtype=torch.float)
    
    batch_size = len(smiles_list)
    fp_tensors = []
    invalid_indices = []
    valid_count = 0
    
    iterator = enumerate(smiles_list)
    if show_progress:
        iterator = tqdm(iterator, total=len(smiles_list), desc="SMILES")
    
    for idx, smiles in iterator:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("unvalid SMILES")
                
            if fp_type == 'bitvect':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                array = np.zeros((n_bits,), dtype=np.int8)
                AllChem.ConvertToNumpyArray(fp, array)
                tensor = torch.tensor(array, dtype=torch.float)
                
            elif fp_type == 'countvect':
                fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=n_bits)
                array = np.zeros((n_bits,), dtype=np.int32)
                for index, count in fp.GetNonzeroElements().items():
                    if index < n_bits:
                        array[index] = count
                tensor = torch.tensor(array, dtype=torch.float)
                
            else:
                raise ValueError("not support")
                
            valid_count += 1
            fp_tensors.append(tensor)
                
        except Exception as e:
            if show_progress:
                tqdm.write(f"warning: '{smiles}' error: {str(e)}")
            fp_tensors.append(torch.zeros(n_bits, dtype=torch.float))
            invalid_indices.append(idx)
    
    batch_tensor = torch.stack(fp_tensors, dim=0)
    
    
    return batch_tensor