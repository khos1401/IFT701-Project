import sys
import os
import argparse
import datetime as dt
import json
import csv
import random
import numpy as np
import torch

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from utils import (
    get_data_tensors,
    evaluate_model,
    save_results,
    save_training_history,
    plot_training_history,
    compute_multiclass_roc,
    plot_multiclass_roc,
)

from train_model_and_save import train_model
from classical_models import ClassicalNN, ClassicalCNN
from quantum_models import QuantumNN, QuantumCNN

# Model registry
MODEL_REGISTRY = {
    "ClassicalNN": ClassicalNN,
    "ClassicalCNN": ClassicalCNN,
    "QuantumNN": QuantumNN,
    "QuantumCNN": QuantumCNN,
}

# Reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# CSV
def append_csv(filepath, fieldnames, row):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# Averaging
def update_averages(model_dir, metrics):
    avg_json = os.path.join(model_dir, "averages.json")
    avg_csv = os.path.join(model_dir, "averages.csv")

    if os.path.exists(avg_json):
        with open(avg_json, "r") as f:
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

    with open(avg_json, "w") as f:
        json.dump({**data, **averaged}, f, indent=4)

    append_csv(
        avg_csv,
        fieldnames=list(averaged.keys()),
        row=averaged,
    )


def main(args):
    X_train, X_test, X_val, y_train, y_test, y_val, class_names = \
        get_data_tensors(args.dataset_path, args.class_to_keep)

    base_out = args.output_dir
    os.makedirs(base_out, exist_ok=True)

    for model_name, ModelClass in MODEL_REGISTRY.items():
        print(f"\n==============================")
        print(f" Running {model_name}")
        print(f"==============================")

        model_dir = os.path.join(base_out, model_name)
        os.makedirs(model_dir, exist_ok=True)

        trials_csv = os.path.join(model_dir, "trials.csv")

        for trial in range(args.num_of_trials):
            seed = args.seed + 1000 * list(MODEL_REGISTRY.keys()).index(model_name) + trial
            set_seed(seed)

            print(f"\n--- Trial {trial + 1}/{args.num_of_trials} | Seed {seed} ---")

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

            accuracy, report, conf_matrix, precision, recall, f1score = evaluate_model(
                model, X_test, y_test, label_names=class_names
            )

            trial_dir = os.path.join(model_dir, f"trial_{trial + 1}")
            os.makedirs(trial_dir, exist_ok=True)

            additional_info = {
                "training_duration": str(duration),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": seed,
                "model_info": model.get_model_info(),
            }

            plot_training_history(
                history=history,
                title=f"{model_name} Training History",
                output_path=trial_dir,
            )

            save_results(
                accuracy=accuracy,
                report=report,
                conf_matrix=conf_matrix,
                additional_info=additional_info,
                filename=f"{model_name}_Results",
                dataset_path=args.dataset_path,
                output_dir=trial_dir,
            )

            save_training_history(
                train_loss=history["train_losses"],
                test_accuracies=history["val_accuracies"],
                filepath=os.path.join(trial_dir, "training_history.json"),
            )

            roc_info = compute_multiclass_roc(
                model=model,
                x_test=X_test,
                y_test=y_test,
                class_names=class_names,
                strategy="ovr",
            )

            plot_multiclass_roc(
                roc_info=roc_info,
                class_names=class_names,
                title=f"{model_name} ROC Curves",
                save_path=os.path.join(trial_dir, "roc_curves.svg"),
            )

            trial_metrics = {
                "model": model_name,
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

            update_averages(
                model_dir=model_dir,
                metrics=trial_metrics,
            )

            print(f"Trial finished in {duration}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Neural Network Experiment Runner")

    parser.add_argument("--dataset_path", type=str, default="dataset/mnist_8x8.npz")
    parser.add_argument("--class_to_keep", type=int, nargs="+", default=[8, 9])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--num_of_trials", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results/experiments")

    args = parser.parse_args()
    main(args)
