import argparse
import torch
from rdkit import Chem
from dataset import get_node_features, get_edge_features
from model import EPRPredictor

def predict(smiles, model_path='best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error parsing SMILES: {smiles}")
        return
        
    node_features = []
    carbon_indices = []
    
    for i, atom in enumerate(mol.GetAtoms()):
        node_features.append(get_node_features(atom))
        if atom.GetSymbol() == 'C':
            carbon_indices.append(i)
            
    x = torch.tensor(node_features, dtype=torch.float)
    
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        e_features = get_edge_features(bond)
        
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e_features, e_features]
        
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
        
    x = x.to(device)
    edge_index = edge_index.to(device)
    
    model = EPRPredictor(in_channels=x.shape[1], hidden_channels=64, num_layers=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        out = model(x, edge_index)
        
    print(f"Predictions for {smiles}:")
    for i, c_idx in enumerate(carbon_indices):
        pred_val = out[c_idx].item()
        print(f"Carbon {i} (Atom index {c_idx}): {pred_val:.4f} MHz")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict A(iso) for Carbon atoms in a SMILES string.')
    parser.add_argument('smiles', type=str, help='SMILES string of the radical')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to trained model weights')
    args = parser.parse_args()
    
    predict(args.smiles, args.model)
