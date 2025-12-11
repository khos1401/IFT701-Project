import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from typing import Dict
import torch
import torch.optim as optim
import torch.nn as nn
from quantum_models.qnn.cudaq.model import QuantumNeuralNetwork
from torch.utils.data import TensorDataset, DataLoader
import time
import cudaq

def train_quantum_model(model: QuantumNeuralNetwork, 
                        x_train, y_train, x_val, y_val, 
                        epochs: int=50, 
                        batch_size: int=32, 
                        shots: int=512, 
                        lr: float=0.01
                        ) -> Dict:

    # Create TensorDatasets and DataLoaders for training and validation
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    print(f'Training quantum model with:')
    print(f'- Shots: {shots}')
    print(f'- Total Parameters: {sum(p.numel() for p in model.parameters())}')
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        start_time = time.time()
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data, shots=shots)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 5 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}')
            

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data, shots=shots)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s):')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 60)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
    }