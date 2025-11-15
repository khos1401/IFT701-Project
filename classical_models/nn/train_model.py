import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def train_classical_model(
                    model: nn.Module,
                    x_train: torch.Tensor,
                    y_train: torch.Tensor,
                    x_test: torch.Tensor,
                    y_test: torch.Tensor,
                    x_val: torch.Tensor = None,
                    y_val: torch.Tensor = None,
                    epochs: int = 100,
                    batch_size: int = 32,
                    learning_rate: float = 1e-3,
                    device: torch.device = None,
):

    # Device setup
    if device is None:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Datasets
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if x_val is not None and y_val is not None:
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Loss, optimizer, schedule
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True
    )

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print(f"- Batch Size: {batch_size}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Total Parameters: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(epochs):
        start_time = time.time()

        # Training loop
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            if batch_idx % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, "
                    f"Batch {batch_idx+1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = running_train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # ---- Validation loop (optional) ----
        avg_val_loss = None
        val_acc = None

        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(device)
                    target = target.to(device)

                    logits = model(data)
                    loss = criterion(logits, target)

                    running_val_loss += loss.item()
                    _, predicted = torch.max(logits, dim=1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()

            avg_val_loss = running_val_loss / len(val_loader)
            val_acc = 100.0 * val_correct / val_total

            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)

            # Step scheduler based on validation loss
            scheduler.step(avg_val_loss)
        else:
            # No validation: step with train loss
            scheduler.step(avg_train_loss)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        if val_loader is not None:
            print(f"  Val   Loss: {avg_val_loss:.4f}, Val   Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)

    # Test set evaluation
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            logits = model(data)
            loss = criterion(logits, target)

            test_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100.0 * test_correct / test_total

    print("Final Test Performance:")
    print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses if val_loader is not None else None,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies if val_loader is not None else None,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }

    return history
