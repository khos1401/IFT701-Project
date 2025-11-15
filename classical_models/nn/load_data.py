import numpy as np
import torch
from sklearn.model_selection import train_test_split


def prepare_data(
    dataset_filename: str,
    seed: int,
):

    # Load dataset
    data = np.load(f'{dataset_filename}', allow_pickle=True)
    X = data['X']
    y = data['y']
    crop_labels = ['cat', 'dog']

    X = X.astype(np.float32)
    max_val = X.max() if X.size > 0 else 1.0
    if max_val > 1.5:      # likely 0–255 images
        X = X / 255.0
    
    X = np.transpose(X, (0, 3, 1, 2))

    # split dataset: 70% train, 15% val, 15% test
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        X, y, train_size=0.70, random_state=seed, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )

    # Convert to tensors for CUDA
    x_train_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.as_tensor(x_test, dtype=torch.float32)
    x_val_tensor = torch.as_tensor(x_val, dtype=torch.float32)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.as_tensor(y_test, dtype=torch.long)
    y_val_tensor = torch.as_tensor(y_val, dtype=torch.long)

    return (
        x_train_tensor, x_test_tensor, x_val_tensor,
        y_train_tensor, y_test_tensor, y_val_tensor,
        crop_labels
    )