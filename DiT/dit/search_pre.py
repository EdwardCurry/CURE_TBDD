import pandas as pd
import argparse

def process_smiles(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        df.columns = df.columns.str.strip().str.lower()
        
        if 'smiles' not in df.columns:
            possible_columns = [col for col in df.columns if 'smile' in col]
            
            if not possible_columns:
                raise ValueError("No columns containing 'smile' were found in the file")
                
            df.rename(columns={possible_columns[0]: 'smiles'}, inplace=True)
            print(f"use '{possible_columns[0]}' as SMILES")
        
        unique_smiles = df['smiles'].drop_duplicates().reset_index(drop=True)
        
        result_df = pd.DataFrame(unique_smiles, columns=['smiles'])
        result_df.to_csv(output_file, index=False)
        
        print(f"Saved at: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    pass