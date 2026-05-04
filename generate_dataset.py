import os
import random
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
    
    # 30% chance to add a ring if enough atoms
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
        # Use canonical rank to find unique atoms (symmetry classes)
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        unique_ranks = set()
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != 'C': continue
            if atom.GetTotalNumHs() == 0: continue # Quaternary carbons have no H to lose
            
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

def write_orca_input(smi, mol, filename):
    canonical_mol = Chem.MolFromSmiles(smi)
    canonical_mol = Chem.AddHs(canonical_mol)
    
    # 3D Embed
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
    
    # Build custom xyz string with C followed immediately by its attached H's
    xyz_lines = []
    for atom in canonical_mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            pos = conf.GetAtomPosition(atom.GetIdx())
            xyz_lines.append(f"C {pos.x:10.6f} {pos.y:10.6f} {pos.z:10.6f}")
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == 'H':
                    pos_h = conf.GetAtomPosition(nbr.GetIdx())
                    xyz_lines.append(f"H {pos_h.x:10.6f} {pos_h.y:10.6f} {pos_h.z:10.6f}")
                    
    basename = os.path.basename(filename).replace('.inp', '')
    orca_inp = f"""! wB97X-D 6-31G* Opt
* xyz 0 2
{chr(10).join(xyz_lines)}
*

$new_job
! wB97X-D3 IGLO-II EPRNMR
%mdci
  EPR_Nuc = all H
end
* xyzfile 0 2 {basename}.xyz
"""
    with open(filename, 'w') as f:
        f.write(orca_inp)
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
    with open('dataset_smiles.txt', 'w') as f:
        f.write("job_id,smiles\n")
        for idx, (smi, mol) in enumerate(radicals):
            safe_smi = smi.replace('/', '_').replace('\\', '_')
            filename = f"orca_inputs/{safe_smi}.inp"
            if write_orca_input(smi, mol, filename):
                f.write(f"{safe_smi},{smi}\n")
                success_count += 1
                
    print(f"Successfully wrote {success_count} ORCA input files.")

if __name__ == '__main__':
    main()
