from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import torch
import json
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime as dt
from tqdm import tqdm

from dataset.load_dataset import NPZDataLoader


def training_loop(dataloader: torch.utils.data.DataLoader,
                  model: torch.nn.Module,
                  loss_fn: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  device: torch.device = 'cpu'
                  ) -> None:
    """
    One epoch training loop
    """
    size = len(dataloader.dataset)
    train_loss = 0.0
    train_accuracy = 0.0

    model.train()
    pbar = tqdm(dataloader, desc="Training")
    for batch, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_accuracy += (y_pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            pbar.set_postfix({'loss': loss, 'current': f'{current}/{size}'})

    train_loss /= len(dataloader)
    train_accuracy /= size
    return train_loss, train_accuracy


def testing_loop(dataloader,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 device: torch.device = 'cpu'
                 ) -> None:
    """
    Testing loop
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    with torch.inference_mode():
        test_loss, accuracy = 0, 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y).item()
            accuracy += (y_pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        accuracy /= size
    return test_loss, accuracy


def get_data_tensors(file_path):
    # Creat dataset and dataloader
    loader = NPZDataLoader(file_path)
    loader.normalize()
    X = torch.tensor(loader.X, dtype=torch.float32).permute(0, 3, 1, 2)
    y = torch.tensor(loader.y, dtype=torch.long)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.70, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, train_size=0.50, random_state=42, stratify=y_test
    )
    
    return X_train, X_test, X_val, y_train, y_test, y_val



def evaluate_model(model, x_test, y_test, label_names: list):
    """
    Evaluate the model on the test set and return accuracy, classification report and confusion matrix.
    """

    # Get whatever device the model is currently on
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        x_test_float = x_test.to(device).float()
        outputs = model(x_test_float)
        _, predicted = torch.max(outputs.data, 1)

    # Move back to CPU for sklearn / numpy
    y_true = y_test.cpu().numpy()
    y_pred = predicted.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=label_names)
    conf_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_ = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return accuracy, report, conf_matrix, precision, recall_, f1

def plot_training_history(history: dict, title: str, output_path: str='training_history.svg'):
    """
    Plots and saves the training loss and test accuracy history.

    """

    epochs = range(1, len(history['train_losses']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history['train_losses'], label='Train Loss')
    ax1.plot(epochs, history['val_losses'], label='Validation Loss')
    ax1.set_title(f'Training Loss - {title}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_accuracies'], label='Train Accuracy')
    ax2.plot(epochs, history['val_accuracies'], label='Validation Accuracy')
    ax2.set_title(f'Accuracy - {title}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/Train_Loss_History.svg')
    plt.close()


def save_training_history(train_loss, test_accuracies, filepath):
    """
    Saves the training history of a quantum machine learning model to a file.
    
    """

    history = {
        'train_losses': train_loss,
        'test_accuracies': test_accuracies,
        'best_accuracy': max(test_accuracies),
        'best_epoch': test_accuracies.index(max(test_accuracies)) + 1,
        'final_loss': train_loss[-1],
        'final_accuracy': test_accuracies[-1]
    }
    
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training history saved to: {filepath}")


def compute_multiclass_roc(model,
                            x_test: torch.Tensor,
                            y_test: torch.Tensor,
                            class_names: list,
                            strategy: str = "ovr"):
    """
    Compute per-class, micro- and macro-averaged ROC curves & AUCs.
    Works for both binary (2 classes) and multiclass (K > 2).
    """

    model.eval()
    with torch.no_grad():
        x_test_f = x_test.float()                  # stay on CPU
        logits = model(x_test_f)                   # [N, num_classes]
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    y_true = y_test.cpu().numpy()
    n_classes = probs.shape[1]

    # --- Manually one-hot encode y_true for all classes ---
    y_true_bin = np.zeros((len(y_true), n_classes), dtype=int)
    for i in range(n_classes):
        y_true_bin[:, i] = (y_true == i).astype(int)

    fpr, tpr, roc_auc = {}, {}, {}

    # --- Per-class ROC (one-vs-rest) ---
    for i in range(n_classes):
        fi, ti, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        fpr[i], tpr[i] = fi, ti
        roc_auc[i] = auc(fi, ti)

    # --- Micro-average ROC ---
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # --- Macro-average ROC ---
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # --- Overall AUC ---
    if n_classes == 2:
        # standard binary AUC: positive class = 1
        roc_auc["overall"] = roc_auc_score(y_true, probs[:, 1])
    else:
        roc_auc["overall"] = roc_auc_score(
            y_true_bin, probs, multi_class=strategy, average="macro"
        )

    return {
        "fpr": fpr,        # false positive rates
        "tpr": tpr,        # true positive rates
        "roc_auc": roc_auc,  # AUC values
        "n_classes": n_classes,
    }

def plot_multiclass_roc(roc_info: dict, class_names: list, title: str = "Multiclass ROC", save_path: str = None):
    """
    Plot per-class ROC curves plus micro/macro averages.

    """

    fpr, tpr, roc_auc = roc_info["fpr"], roc_info["tpr"], roc_info["roc_auc"]
    n_classes = roc_info["n_classes"]

    plt.figure(figsize=(8, 6))

    # Micro
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        linestyle="--",
        linewidth=2,
        label=f"micro-average (AUC = {roc_auc['micro']:.3f})",
    )

    # Macro
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        linestyle="--",
        linewidth=2,
        label=f"macro-average (AUC = {roc_auc['macro']:.3f})",
    )

    # Per-class
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            linewidth=1.5,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})",
        )

    # Chance line
    plt.plot([0, 0, 1], [0, 1, 1], linewidth=1, alpha=0.3)
    plt.plot([0, 1], [0, 1], linestyle=":", linewidth=1) 

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.close()
    return roc_auc

def save_results(accuracy, report, conf_matrix,
                additional_info: dict = None, 
                filename='results',
                dataset_path: str='Unknown', 
                output_dir: str='../results/',
                ):
    """
    Save model evaluation results to a text file.

    """

    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{filename}_{timestamp}.txt'

    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/{filename}', 'w') as f:
        f.write(f'Model Path: {output_dir}\n')
        f.write(f'Dataset: {dataset_path}\n')
        f.write("Accuracy: {:.2f}%\n".format(accuracy * 100))
        f.write("Classification Report:\n{}\n".format(report))
        f.write("Confusion Matrix:\n{}\n".format(conf_matrix))
        f.write("\n")
        f.write('Additional Information:\n')
        f.write(f'Timestamp: {timestamp}\n')
        if additional_info:
            for key, value in additional_info.items():
                f.write(f'{key}: {value}\n')


    print(f"Results saved to: {os.path.abspath(f'{output_dir}/{filename}')}")