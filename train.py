import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from dataset import EPRDataset
from model import EPRPredictor

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        loss = F.mse_loss(out[data.mask], data.y[data.mask])
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(model, loader, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    num_nodes = 0
    
    predictions = []
    targets = []
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        
        mask = data.mask
        pred_c = out[mask]
        y_c = data.y[mask]
        
        loss = F.mse_loss(pred_c, y_c)
        total_loss += loss.item() * data.num_graphs
        
        mae = F.l1_loss(pred_c, y_c, reduction='sum')
        total_mae += mae.item()
        num_nodes += mask.sum().item()
        
        predictions.extend(pred_c.cpu().numpy())
        targets.extend(y_c.cpu().numpy())
        
    mse = total_loss / len(loader.dataset)
    mae = total_mae / num_nodes
    return mse, mae, predictions, targets

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    dataset = EPRDataset(root='data', csv_path='epr_dataset.csv')
    
    indices = np.random.permutation(len(dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    train_dataset = dataset[train_idx.tolist()]
    val_dataset = dataset[val_idx.tolist()]
    test_dataset = dataset[test_idx.tolist()]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    in_channels = dataset[0].x.shape[1]
    model = EPRPredictor(in_channels=in_channels, hidden_channels=64, num_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    print(f"Training on {len(train_dataset)} graphs, Validation on {len(val_dataset)}, Test on {len(test_dataset)}")
    
    epochs = 300
    train_losses = []
    val_losses = []
    
    best_val_mae = float('inf')
    
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_mse, val_mae, _, _ = test(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_mse)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), 'best_model.pth')
            
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Train Loss (MSE): {train_loss:.4f}, Val Loss (MSE): {val_mse:.4f}, Val MAE: {val_mae:.4f}')
            
    print("Training finished.")
    
    model.load_state_dict(torch.load('best_model.pth'))
    test_mse, test_mae, preds, targs = test(model, test_loader, device)
    print(f"Final Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f} MHz")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train MSE')
    plt.plot(val_losses, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_curve.png')
    print("Saved training_curve.png")
    
    plt.figure(figsize=(6, 6))
    plt.scatter(targs, preds, alpha=0.6)
    
    min_val = min(min(targs), min(preds))
    max_val = max(max(targs), max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True A(iso) (MHz)')
    plt.ylabel('Predicted A(iso) (MHz)')
    plt.title(f'Test Set Predictions (MAE: {test_mae:.2f} MHz)')
    plt.savefig('test_scatter.png')
    print("Saved test_scatter.png")

if __name__ == '__main__':
    main()
