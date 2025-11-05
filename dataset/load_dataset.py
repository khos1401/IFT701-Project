from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
from sklearn.model_selection import train_test_split

# Example usage:

# from dataset_loader import NPZDataLoader

# loader = NPZDataLoader("cats_dogs_128.npz")
# loader.normalize("0-1")  # optional normalization
# splits = loader.split(test_size=0.3, val_size=0.5) # 30% test, then 50% of remaining for val

# X_train, y_train = splits["X_train"], splits["y_train"]
# X_val, y_val = splits["X_val"], splits["y_val"]
# X_test, y_test = splits["X_test"], splits["y_test"]

# print(X_train.shape, y_train.shape)

class NPZDataLoader:
    """
    Load an .npz dataset with X (images) and y (labels),
    optionally normalize, and split into train/val/test sets.
    """

    def __init__(self, npz_path: str):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Dataset not found: {npz_path}")

        self.data = np.load(npz_path, allow_pickle=True)
        self.X = self.data["X"]
        self.y = self.data["y"]
        self.classes = (
            self.data["classes"].tolist()
            if "classes" in self.data
            else list(range(len(np.unique(self.y))))
        )
        print(f"Loaded {npz_path.name}")
        print(f"X: {self.X.shape}, dtype={self.X.dtype}")
        print(f"y: {self.y.shape}, classes: {self.classes}")

    def normalize(self, method: str = "0-1") -> None:
        """
        Normalize images.
        - "0-1": scale to [0, 1] float32
        - "standardize": mean 0, std 1
        """
        if method == "0-1":
            self.X = self.X.astype(np.float32) / 255.0
        elif method == "standardize":
            self.X = self.X.astype(np.float32)
            mean = np.mean(self.X, axis=(0, 1, 2), keepdims=True)
            std = np.std(self.X, axis=(0, 1, 2), keepdims=True) + 1e-8
            self.X = (self.X - mean) / std
        else:
            raise ValueError("method must be '0-1' or 'standardize'")
        print(f"Normalized dataset using '{method}' method.")

    def split(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split dataset into train/val/test sets.

        Args:
            test_size: proportion of test data (from total).
            val_size: proportion of validation data (from training portion).
            random_state: reproducibility seed.
            stratify: maintain label proportions.
        Returns:
            dict with keys: X_train, X_val, X_test, y_train, y_val, y_test
        """
        stratify_labels = self.y if stratify else None

        # First split train/test
        X_train, X_test_val, y_train, y_test_val = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=stratify_labels
        )

        # Then split train/val
        stratify_labels_val = y_train if stratify else None
        X_test, X_val, y_test, y_val = train_test_split(
            X_test_val, y_test_val, test_size=val_size, random_state=random_state, stratify=stratify_labels_val
        )

        print(f"Split dataset:")
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Val:   {X_val.shape[0]} samples")
        print(f"Test:  {X_test.shape[0]} samples")

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

