import os
import re
import csv

def process_out_file(out_filepath):
    with open(out_filepath, 'r') as f:
        text = f.read()

    # 1. Parse coordinates to map H to C
    atoms = []
    in_coords = False
    for line in text.split('\n'):
        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
            in_coords = True
            continue
        if in_coords:
            if '---------------------------------' in line:
                continue
            if line.strip() == '':
                break
            parts = line.strip().split()
            if len(parts) == 4:
                atoms.append(parts[0].upper())

    if not atoms:
        return None

    # Map H index to C index
    mapping = {} # H_idx -> C_idx
    num_carbons = 0
    current_carbon_idx = -1
    for atom_idx, symbol in enumerate(atoms):
        if symbol == 'C':
            current_carbon_idx += 1
            num_carbons += 1
        elif symbol == 'H':
            if current_carbon_idx >= 0:
                mapping[atom_idx] = current_carbon_idx

    # 2. Extract A(iso) for protons
    hfccs = {}
    nuclei_blocks = text.split('Nucleus ')[1:]
    for block in nuclei_blocks:
        lines = block.split('\n')
        if 'H :' in lines[0]:
            nuc_str = lines[0].split('H :')[0].strip()
            try:
                nuc_idx = int(nuc_str) - 1 # ORCA might use 1-based in printing but 0-based in logic? 
                # Wait, earlier nuc_str was "1", "2", "4", "6". 
                # Is "1H" index 1 or atom #1 (which is index 0)? 
                # Let's check: atom list was C, H, H -> indices 0, 1, 2. ORCA printed 1H, 2H. So the number is the 0-based index! 
                # "18H" means atom index 18.
                nuc_idx = int(nuc_str)
            except ValueError:
                continue
            
            aiso_match = re.search(r'A\(iso\)=\s*([-\d\.]+)', block)
            if aiso_match:
                hfccs[nuc_idx] = float(aiso_match.group(1))

    # 3. Average H values for each C
    carbon_hfccs = {i: [] for i in range(num_carbons)}
    for h_idx, a_mhz in hfccs.items():
        if h_idx in mapping:
            c_idx = mapping[h_idx]
            carbon_hfccs[c_idx].append(a_mhz)

    hfcc_string_list = []
    for c_idx in range(num_carbons):
        vals = carbon_hfccs[c_idx]
        if vals:
            avg = sum(vals) / len(vals)
            # Remove trailing zeros, up to 4 decimals
            val_str = str(round(avg, 4))
            hfcc_string_list.append(val_str)
        else:
            hfcc_string_list.append("0") # or perhaps "0.0"

    hfcc_string = "[" + " ".join(hfcc_string_list) + "]"
    return hfcc_string

def main():
    results = []
    with open('dataset_master.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_id = row['ID']
            smiles = row['SMILES']
            out_filepath = os.path.join('epr_alkyl267', f'{job_id}_epr.out')
            if os.path.exists(out_filepath):
                hfcc_str = process_out_file(out_filepath)
                if hfcc_str:
                    results.append({'ID': job_id, 'SMILES': smiles, 'A(iso)': hfcc_str})
                    if job_id == 'rad_0001':
                        print(f"Test {job_id}: {hfcc_str}")
    
    if results:
        with open('epr_dataset.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['ID', 'SMILES', 'A(iso)'])
            writer.writeheader()
            writer.writerows(results)
        print(f"Processed {len(results)} files successfully. Database written to epr_dataset.csv")
    else:
        print("No files processed.")

if __name__ == '__main__':
    main()
