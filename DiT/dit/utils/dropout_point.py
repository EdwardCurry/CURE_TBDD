import csv

def filter_smiles(input_file, output_file):

    with open(input_file, 'r', newline='') as infile, \
         open(output_file, 'w', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        headers = next(reader)
        try:
            smiles_idx = next(
                i for i, h in enumerate(headers) 
                if h.strip().lower() == "smiles"
            )
        except StopIteration:
            raise ValueError("Column 'smiles' not found in CSV file")
        
        writer.writerow(headers)
        
        for row in reader:
            if len(row) == 0:
                continue
            
            if len(row) <= smiles_idx:
                continue
                
            smiles_str = row[smiles_idx].strip()
            if '.' not in smiles_str:
                writer.writerow(row)
