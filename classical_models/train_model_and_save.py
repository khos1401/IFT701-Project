import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from utils import training_loop, testing_loop

from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

def train_classical_model(
                    model: nn.Module,
                    X_train: torch.Tensor,
                    y_train: torch.Tensor,
                    X_val: torch.Tensor,
                    y_val: torch.Tensor,
                    X_test: torch.Tensor,
                    y_test: torch.Tensor,
                    epochs: int = 100,
                    batch_size: int = 32,
                    learning_rate: float = 1e-3,
                    device: torch.device = None,
    ):
    model.to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # Loss, optimizer, schedule
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print(f"- Batch Size: {batch_size}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Total Parameters: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_acc = training_loop(
            dataloader=train_loader,
            model=model,
            loss_fn=criterion,
            optimizer=optimizer,
            device=device
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")


        # ---- Validation loop (optional) ----
        val_loss, val_acc = testing_loop(
            dataloader=val_loader,
            model=model,
            loss_fn=criterion,
            device=device
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        scheduler.step(val_loss)
        print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)

    test_loss, test_acc = testing_loop(
        dataloader=test_loader,
        model=model,
        loss_fn=criterion,
        device=device
    )
    print("Final Test Performance:")
    print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }
    return history
