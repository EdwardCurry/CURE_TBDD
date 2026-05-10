import csv
from rdkit import DataStructs
from rdkit.Chem import AllChem

def Smile_Similarity(org_smiles, trained_smiles, axis=None):
    mol1 = AllChem.MolFromSmiles(org_smiles)  
    mol2 = AllChem.MolFromSmiles(trained_smiles)  
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=1024)

    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity

def process_csv(filename, func):
    with open(filename, 'r', newline='') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames  
        rows = list(reader)  

    for row in rows:
        canonical_smiles = row['canonical_smiles'].strip()
        gex_molgen = row['GexMolGen'].strip()
        row['sim'] = Smile_Similarity(canonical_smiles, gex_molgen)
    
    updated_fieldnames = list(fieldnames)
    if 'sim' not in updated_fieldnames:
        updated_fieldnames.append('sim')
    
    with open(filename, 'w', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=updated_fieldnames)
        writer.writeheader()  
        writer.writerows(rows)  

if __name__ == "__main__":
    pass
