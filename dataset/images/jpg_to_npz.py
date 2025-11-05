# save as png_to_npz.py
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image  # pip install pillow


def load_images_from_class(
    folder: Path,
    target_size: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """
    Load all PNGs from a folder, convert to RGB, optionally resize,
    and return as a list of uint8 arrays (H, W, 3).
    """
    arrays = []
    img_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".png"])
    for p in img_paths:
        try:
            im = Image.open(p)
            im = im.convert("RGB")
            if target_size is not None:
                im = im.resize(target_size, Image.Resampling.BILINEAR)
            arrays.append(np.array(im, dtype=np.uint8))
        except Exception as e:
            print(f"⚠️ Skipping {p} ({e})")
    return arrays


def build_dataset(
    root: Path,
    classes: List[str],
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    From a root directory with subfolders for each class, build X and y.
    - X: (N, H, W, 3), dtype=uint8
    - y: (N,), dtype=int, labels correspond to index in `classes`
    """
    X_list = []
    y_list = []
    for label, cls in enumerate(classes):
        folder = root / cls
        if not folder.exists():
            print(f"⚠️ Missing folder: {folder}, skipping.")
            continue
        imgs = load_images_from_class(folder, target_size=target_size)
        X_list.extend(imgs)
        y_list.extend([label] * len(imgs))
        print(f"✅ {cls}: {len(imgs)} images")

    if len(X_list) == 0:
        raise RuntimeError("No images found. Check your root and class names.")

    # Stack into arrays
    X = np.stack(X_list, axis=0)  # (N, H, W, 3)
    y = np.array(y_list, dtype=np.int64)
    return X, y, classes


def main():
    parser = argparse.ArgumentParser(
        description="Create an NPZ (X,y) from PNG images in class folders."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing class subfolders (e.g., cats/, dogs/).",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        required=True,
        help="List of class folder names under root (e.g., cats dogs). Order defines labels.",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional target size to resize all images (e.g., --size 128 128).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="dataset.npz",
        help="Output NPZ filename (default: dataset.npz)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    size = tuple(args.size) if args.size is not None else None

    X, y, classes = build_dataset(root, args.classes, target_size=size)

    # Save NPZ with metadata
    np.savez_compressed(
        args.out,
        X=X,          # uint8 RGB
        y=y,          # int labels
        classes=np.array(classes),  # class names in label order
        size=np.array(size if size is not None else (), dtype=np.int64),
    )
    print(f"\nSaved {args.out}")
    print(f"  X: {X.shape}, dtype={X.dtype}")
    print(f"  y: {y.shape}, labels: {dict(enumerate(classes))}")
    if size:
        print(f"  resized to: {size[0]}x{size[1]} (WxH)")
    else:
        print("  no resizing (images may vary in size)")


if __name__ == "__main__":
    main()
