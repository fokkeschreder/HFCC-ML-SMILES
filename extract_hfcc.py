import os
import csv

def parse_inp_file_for_mapping(inp_filepath):
    """
    Parses the ORCA .inp file to determine which H atoms belong to which C atom.
    Returns a mapping from atom_index to carbon_index (0-indexed in SMILES order)
    and the total number of carbon atoms.
    """
    mapping = {}
    current_carbon_idx = -1
    atom_idx = 0
    with open(inp_filepath, 'r') as f:
        in_xyz = False
        for line in f:
            line = line.strip()
            if line.startswith('* xyz'):
                in_xyz = True
                continue
            if in_xyz:
                if line == '*' or line.startswith('* xyzfile') or line.startswith('$'):
                    break
                parts = line.split()
                if len(parts) >= 4:
                    symbol = parts[0].upper()
                    if symbol == 'C':
                        current_carbon_idx += 1
                        atom_idx += 1
                    elif symbol == 'H':
                        mapping[atom_idx] = current_carbon_idx
                        atom_idx += 1
                    else:
                        atom_idx += 1
    
    num_carbons = current_carbon_idx + 1
    return mapping, num_carbons

def parse_orca_out_hfcc(out_filepath):
    """
    Parses the ORCA .out file to extract isotropic HFCC values for H atoms.
    Returns a dictionary mapping atom_index to A(MHz) value.
    """
    hfccs = {}
    in_hfcc_block = False
    with open(out_filepath, 'r') as f:
        for line in f:
            if "ISOTROPIC HYPERFINE COUPLING PARAMETERS" in line:
                in_hfcc_block = True
                continue
            if in_hfcc_block:
                if "--------------------------------" in line or "Nucleus" in line or line.strip() == "":
                    continue
                if "DIPOLAR HYPERFINE" in line or "TENSOR" in line:
                    break
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        idx = int(parts[0])
                        symbol = parts[1]
                        a_mhz = float(parts[2])
                        if symbol == 'H':
                            hfccs[idx] = a_mhz
                    except ValueError:
                        pass
    return hfccs

def process_job(job_id, smiles, inp_filepath, out_filepath):
    """
    Processes a single job: computes averaged HFCCs per carbon and returns the string.
    """
    mapping, num_carbons = parse_inp_file_for_mapping(inp_filepath)
    hfccs = parse_orca_out_hfcc(out_filepath)
    
    if not hfccs:
        return None
        
    carbon_hfccs = {i: [] for i in range(num_carbons)}
    
    for h_idx, a_mhz in hfccs.items():
        if h_idx in mapping:
            c_idx = mapping[h_idx]
            carbon_hfccs[c_idx].append(a_mhz)
            
    hfcc_string_list = []
    for c_idx in range(num_carbons):
        vals = carbon_hfccs[c_idx]
        if vals:
            avg_hfcc = sum(vals) / len(vals)
            hfcc_string_list.append(round(avg_hfcc))
        else:
            hfcc_string_list.append(0)
            
    hfcc_string = "[" + " ".join(map(str, hfcc_string_list)) + "]"
    return hfcc_string

def main():
    if not os.path.exists('dataset_smiles.txt'):
        print("dataset_smiles.txt not found. Run generate_dataset.py first.")
        return
        
    results = []
    
    with open('dataset_smiles.txt', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) == 2:
                job_id, smiles = row
                
                inp_filepath = os.path.join('orca_inputs', f"{job_id}.inp")
                out_filepath = os.path.join('orca_inputs', f"{job_id}.out")
                
                if os.path.exists(inp_filepath) and os.path.exists(out_filepath):
                    hfcc_str = process_job(job_id, smiles, inp_filepath, out_filepath)
                    if hfcc_str:
                        results.append({"job_id": job_id, "smiles": smiles, "hfcc_string": hfcc_str})
                    else:
                        print(f"Warning: Could not parse HFCCs for {job_id}")
                else:
                    # We do not print skipping messages to avoid flooding the console
                    # if the user only ran a subset of ORCA jobs.
                    pass
                    
    if results:
        with open('final_dataset.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["job_id", "smiles", "hfcc_string"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Successfully processed {len(results)} jobs and saved to final_dataset.csv.")
    else:
        print("No valid ORCA outputs found to process.")

if __name__ == '__main__':
    main()
