import sys
import os
import argparse
import datetime as dt
import json
import csv
import random
import itertools
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from utils import (
    get_data_tensors,
    evaluate_model,
    save_results,
    save_training_history,
    plot_training_history,
)

from train_model_and_save import train_model
from classical_models import ClassicalNN
from quantum_models import QuantumNN


MODEL_REGISTRY = {
    "ClassicalNN": ClassicalNN,
    "QuantumNN": QuantumNN,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_csv(filepath, fieldnames, row):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def update_averages(output_dir, metrics):
    avg_path = os.path.join(output_dir, "averages.json")

    if os.path.exists(avg_path):
        with open(avg_path, "r") as f:
            data = json.load(f)
    else:
        data = {
            "num_trials": 0,
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1score": [],
        }

    for k in ["accuracy", "precision", "recall", "f1score"]:
        data[k].append(metrics[k])

    data["num_trials"] += 1

    averaged = {
        "num_trials": data["num_trials"],
        "accuracy_mean": float(np.mean(data["accuracy"])),
        "accuracy_std": float(np.std(data["accuracy"])),
        "precision_mean": float(np.mean(data["precision"])),
        "recall_mean": float(np.mean(data["recall"])),
        "f1score_mean": float(np.mean(data["f1score"])),
    }

    with open(avg_path, "w") as f:
        json.dump({**data, **averaged}, f, indent=4)


def main(args):
    # Load dataset once to get class labels
    data = np.load(args.dataset_path)
    all_classes = np.unique(data["y"])

    class_pairs = list(itertools.combinations(all_classes, 2))

    base_out = args.output_dir
    os.makedirs(base_out, exist_ok=True)

    for model_name, ModelClass in MODEL_REGISTRY.items():
        print(f"\n==============================")
        print(f" Running {model_name}")
        print(f"==============================")

        model_root = os.path.join(base_out, model_name)
        os.makedirs(model_root, exist_ok=True)

        for class_a, class_b in class_pairs:
            print(f"\n>>> Class pair: {class_a} vs {class_b}")

            pair_dir = os.path.join(
                model_root, f"class_{class_a}_vs_{class_b}"
            )
            os.makedirs(pair_dir, exist_ok=True)

            trials_csv = os.path.join(pair_dir, "trials.csv")

            for trial in range(args.num_of_trials):
                seed = (
                    args.seed
                    + 1000 * list(MODEL_REGISTRY.keys()).index(model_name)
                    + 100 * class_a
                    + 10 * class_b
                    + trial
                )
                set_seed(seed)

                print(f"  Trial {trial + 1}/{args.num_of_trials} | Seed {seed}")

                X_train, X_test, X_val, y_train, y_test, y_val, class_names = \
                    get_data_tensors(
                        args.dataset_path,
                        class_to_keep=[class_a, class_b],
                    )

                start_time = dt.datetime.now()

                model = ModelClass(input_size=X_train.shape[1:])

                history = train_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                )

                duration = dt.datetime.now() - start_time

                accuracy, report, conf_matrix, precision, recall, f1score = \
                    evaluate_model(model, X_test, y_test, label_names=class_names)

                trial_dir = os.path.join(pair_dir, f"trial_{trial + 1}")
                os.makedirs(trial_dir, exist_ok=True)

                save_training_history(
                    train_loss=history["train_losses"],
                    test_accuracies=history["val_accuracies"],
                    filepath=os.path.join(trial_dir, "training_history.json"),
                )

                trial_metrics = {
                    "model": model_name,
                    "class_a": int(class_a),
                    "class_b": int(class_b),
                    "trial": trial + 1,
                    "seed": seed,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1score": f1score,
                    "duration_seconds": duration.total_seconds(),
                }

                append_csv(
                    trials_csv,
                    fieldnames=list(trial_metrics.keys()),
                    row=trial_metrics,
                )

                update_averages(pair_dir, trial_metrics)

                print(f"    Finished in {duration}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pairwise Class Experiment Runner")

    parser.add_argument("--dataset_path", type=str, default="dataset/mnist_8x8.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--num_of_trials", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results/experiments_pairwise")

    args = parser.parse_args()
    main(args)
