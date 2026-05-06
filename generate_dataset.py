import os
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_random_alkane(max_atoms=8):
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(6))
    for i in range(1, max_atoms):
        valid_targets = [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() < 4]
        mol.AddAtom(Chem.Atom(6))
        if valid_targets:
            target = random.choice(valid_targets)
            mol.AddBond(target, i, Chem.BondType.SINGLE)
    
    if random.random() < 0.3 and mol.GetNumAtoms() >= 4:
        valid_targets = [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() < 4]
        if len(valid_targets) >= 2:
            t1, t2 = random.sample(valid_targets, 2)
            if not mol.GetBondBetweenAtoms(t1, t2):
                mol.AddBond(t1, t2, Chem.BondType.SINGLE)
                
    Chem.SanitizeMol(mol)
    return mol

def get_unique_alkanes(n=100):
    smiles_set = set()
    mols = []
    attempts = 0
    while len(smiles_set) < n and attempts < 10000:
        attempts += 1
        num_atoms = random.randint(3, 7)
        try:
            mol = generate_random_alkane(num_atoms)
            smi = Chem.MolToSmiles(mol)
            if smi not in smiles_set:
                smiles_set.add(smi)
                mols.append(mol)
        except Exception:
            pass
    return mols

def get_unique_radicals(alkane_mols):
    radical_smiles_set = set()
    radical_mols = []
    
    for mol in alkane_mols:
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        unique_ranks = set()
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != 'C': continue
            if atom.GetTotalNumHs() == 0: continue
            
            rank = ranks[atom.GetIdx()]
            if rank in unique_ranks:
                continue
            unique_ranks.add(rank)
            
            rwmol = Chem.RWMol(mol)
            ratom = rwmol.GetAtomWithIdx(atom.GetIdx())
            ratom.SetNumRadicalElectrons(1)
            Chem.SanitizeMol(rwmol)
            
            smi = Chem.MolToSmiles(rwmol)
            if smi not in radical_smiles_set:
                radical_smiles_set.add(smi)
                radical_mols.append((smi, rwmol))
                
    return radical_mols

def write_orca_inputs(smi, rad_id, output_dir='orca_inputs'):
    canonical_mol = Chem.MolFromSmiles(smi)
    canonical_mol = Chem.AddHs(canonical_mol)
    
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    res = AllChem.EmbedMolecule(canonical_mol, params)
    if res == -1:
        params.useRandomCoords = True
        res = AllChem.EmbedMolecule(canonical_mol, params)
        if res == -1:
            print(f"Failed to embed {smi}")
            return False
            
    conf = canonical_mol.GetConformer()
    
    xyz_lines = []
    for atom in canonical_mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            pos = conf.GetAtomPosition(atom.GetIdx())
            xyz_lines.append(f"C {pos.x:10.6f} {pos.y:10.6f} {pos.z:10.6f}")
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == 'H':
                    pos_h = conf.GetAtomPosition(nbr.GetIdx())
                    xyz_lines.append(f"H {pos_h.x:10.6f} {pos_h.y:10.6f} {pos_h.z:10.6f}")
                    
    opt_inp = f"""! UKS wB97X-D3 6-31G* Opt
%maxcore 1500
%geom
  MaxIter 150
end

* xyz 0 2
{chr(10).join(xyz_lines)}
*
"""
    
    epr_inp = f"""! UKS wB97X-D3 IGLO-II
%maxcore 3000


* xyzfile 0 2 {rad_id}_opt.xyz

%eprnmr
  Nuclei = all H {{aiso, adip}}
end
"""
    with open(os.path.join(output_dir, f"{rad_id}_opt.inp"), 'w') as f:
        f.write(opt_inp)
    with open(os.path.join(output_dir, f"{rad_id}_epr.inp"), 'w') as f:
        f.write(epr_inp)
        
    return True

def main():
    os.makedirs('orca_inputs', exist_ok=True)
    print("Generating base alkanes...")
    alkanes = get_unique_alkanes(100)
    print(f"Generated {len(alkanes)} alkanes.")
    
    print("Generating unique radicals...")
    radicals = get_unique_radicals(alkanes)
    print(f"Generated {len(radicals)} unique radicals.")
    
    success_count = 0
    records = []
    
    for idx, (smi, mol) in enumerate(radicals):
        rad_id = f"rad_{idx+1:04d}"
        if write_orca_inputs(smi, rad_id, 'orca_inputs'):
            records.append({'ID': rad_id, 'SMILES': smi, 'Status': 'Pending'})
            success_count += 1
            
    df = pd.DataFrame(records)
    df.to_csv('dataset_master.csv', index=False)
    print(f"Successfully wrote {success_count} pairs of ORCA input files and dataset_master.csv.")

if __name__ == '__main__':
    main()
