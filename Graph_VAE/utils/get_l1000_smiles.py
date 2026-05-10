import pandas as pd
import sys

# 检查命令行参数
if len(sys.argv) < 2:
    print("python script.py input.csv [output.txt]")
    sys.exit(1)

input_csv = sys.argv[1]
output_txt = sys.argv[2] if len(sys.argv) >= 3 else 'output.txt'

try:
    df = pd.read_csv(input_csv)
    
    if 'smiles' not in df.columns:
        raise ValueError("'smiles' not found")
    
    unique_smiles = df['smiles'].drop_duplicates().tolist()
    
    with open(output_txt, 'w') as f:
        f.write('\n'.join(map(str, unique_smiles)))
    
    print(f"{output_txt} | {len(unique_smiles)}")

except Exception as e:
    print(f"error: {e}")
    sys.exit(1)

