from rdkit import Chem

def clean_smiles(input_path: str, output_path: str) -> None:
    valid_smiles = []
    with open(input_path, 'r') as f:
        for line in f:
            s = line.strip()
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                valid_smiles.append(s)
            else:
                print(s)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(valid_smiles))

# 使用示例
clean_smiles(
    input_path='data/my_smiles/all.txt',
    output_path='data/my_smiles/cleaned_all.txt'
)