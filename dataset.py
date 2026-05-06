import os
import torch
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset

def get_node_features(atom):
    atomic_num = atom.GetAtomicNum()
    degree = atom.GetDegree()
    num_hs = atom.GetTotalNumHs()
    imp_val = atom.GetImplicitValence()
    charge = atom.GetFormalCharge()
    num_radicals = atom.GetNumRadicalElectrons()
    
    hyb = atom.GetHybridization()
    hyb_type = 0
    if hyb == Chem.rdchem.HybridizationType.SP: hyb_type = 1
    elif hyb == Chem.rdchem.HybridizationType.SP2: hyb_type = 2
    elif hyb == Chem.rdchem.HybridizationType.SP3: hyb_type = 3
    
    is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    is_in_ring = 1.0 if atom.IsInRing() else 0.0

    return [
        float(atomic_num),
        float(degree),
        float(num_hs),
        float(imp_val),
        float(charge),
        float(num_radicals),
        float(hyb_type),
        is_aromatic,
        is_in_ring
    ]

def get_edge_features(bond):
    bond_type = bond.GetBondType()
    b_type = 0
    if bond_type == Chem.rdchem.BondType.SINGLE: b_type = 1
    elif bond_type == Chem.rdchem.BondType.DOUBLE: b_type = 2
    elif bond_type == Chem.rdchem.BondType.TRIPLE: b_type = 3
    elif bond_type == Chem.rdchem.BondType.AROMATIC: b_type = 4
    
    is_conjugated = 1.0 if bond.GetIsConjugated() else 0.0
    is_in_ring = 1.0 if bond.IsInRing() else 0.0
    
    return [
        float(b_type),
        is_conjugated,
        is_in_ring
    ]

class EPRDataset(InMemoryDataset):
    def __init__(self, root, csv_path, transform=None, pre_transform=None):
        self.csv_path = csv_path
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['epr_data.pt']

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.csv_path)
        data_list = []

        for idx, row in df.iterrows():
            smiles = row['SMILES']
            val_str = row['A(iso)'].strip('[]')
            target_vals = [float(x) for x in val_str.split()]

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            node_features = []
            carbon_indices = []
            
            for i, atom in enumerate(mol.GetAtoms()):
                node_features.append(get_node_features(atom))
                if atom.GetSymbol() == 'C':
                    carbon_indices.append(i)

            if len(carbon_indices) != len(target_vals):
                print(f"Skipping {smiles}: {len(carbon_indices)} carbons vs {len(target_vals)} targets")
                continue

            x = torch.tensor(node_features, dtype=torch.float)

            edge_indices = []
            edge_attrs = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                e_features = get_edge_features(bond)
                
                # Undirected graph, add both directions
                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e_features, e_features]

            if len(edge_indices) > 0:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            else:
                # E.g. single atom (methane radical)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 3), dtype=torch.float)

            # Targets and mask
            # We assign target to all nodes, but mask is True only for Carbon nodes
            y = torch.zeros(x.size(0), dtype=torch.float)
            mask = torch.zeros(x.size(0), dtype=torch.bool)

            for i, c_idx in enumerate(carbon_indices):
                y[c_idx] = target_vals[i]
                mask[c_idx] = True

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, mask=mask)
            data.smiles = smiles
            data.job_id = row['ID']
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
        print(f"Processed and saved {len(data_list)} graphs.")

if __name__ == '__main__':
    dataset = EPRDataset(root='data', csv_path='epr_dataset.csv')
    print(f"Dataset length: {len(dataset)}")
    print(f"Sample data: {dataset[0]}")
    print(f"Node feature shape: {dataset[0].x.shape}")
    print(f"Edge index shape: {dataset[0].edge_index.shape}")
