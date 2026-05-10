import pandas as pd

def process_smiles(input_parquet_path: str, output_txt_path: str) -> None:

    try:
        df = pd.read_parquet(input_parquet_path)
        
        if 'canonical_smiles' not in df.columns:
            raise ValueError(f"canonical_smiles not exist.")
        
        smiles_series = df['canonical_smiles'].astype(str)
        unique_smiles = smiles_series.unique()
        
        split_smiles = []
        for s in unique_smiles:
            stripped = s.strip()
            if not stripped:
                continue
            parts = stripped.split('.')
            for part in parts:
                cleaned_part = part.strip()
                if cleaned_part: 
                    split_smiles.append(cleaned_part)
        
        unique_split_smiles = sorted(list(set(split_smiles)))
        
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(unique_split_smiles))
                    
    except Exception as e:
        print(f"error: {str(e)}")


def process_smiles_lookup(input_parquet_path: str, output_txt_path: str) -> None:
    try:
        df = pd.read_parquet(input_parquet_path)
        
        if 'canonical_smiles' not in df.columns:
            raise ValueError(f"'canonical_smiles' not exist.")
        
        smiles_series = df['canonical_smiles'].astype(str)
        unique_smiles = smiles_series.unique()
        
        split_smiles = []
        for s in unique_smiles:
            stripped = s.strip()
            if not stripped:
                continue
            parts = stripped.split('.')
            max_part = max(parts, key=len)
            if max_part:
                split_smiles.append(max_part)
        
        unique_split_smiles = sorted(list(set(split_smiles)))
        
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(unique_split_smiles))
            
        
    except Exception as e:
        print(f"error: {str(e)}")