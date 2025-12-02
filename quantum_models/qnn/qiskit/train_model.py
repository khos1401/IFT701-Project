import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from typing import Dict
import torch
import torch.optim as optim
import torch.nn as nn
from quantum_models.qnn.cudaq.model import QuantumNeuralNetwork
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def train_quantum_model(model : QuantumNeuralNetwork,
                        x_train, y_train, x_test, y_test, x_val, y_val,
                        epochs=16,
                        batch_size=32,
                        lr=0.01,
                        device='cpu',
                        num_workers=None,
                        persistent_workers=False):
    """
    Trains a quantum model using the provided training data.
    """

    if num_workers is None:
        num_workers = 0 if os.name == 'nt' else min(4, os.cpu_count() or 2)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=False,
                              persistent_workers=(persistent_workers and num_workers > 0),
                              drop_last=True,
                              )
    try:
        _ = next(iter(train_loader))
    except Exception as e:
        raise RuntimeError(f"Dataset/collate failed when fetching the first batch: {e}")


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)  # Higher LR, better optimizer
    
    train_losses = []
    test_accuracies = []
    val_losses = []
    
    print("Starting optimized training...")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0
        train_losses = []

        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            # Convert back to float32 for quantum computation
            batch_X = batch_X.float()
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        model.eval()
        with torch.no_grad():
            test_subset = x_test.float()
            test_labels_subset = y_test
            
            test_outputs = model(test_subset)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = (predicted == test_labels_subset).sum().item() / len(test_labels_subset)
            test_accuracies.append(accuracy)

            val_outputs = model(x_val.float())
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')
    
    return train_losses, val_losses, test_accuracies
    