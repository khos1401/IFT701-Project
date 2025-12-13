import sys
import os
import datetime as dt
import argparse
import torch
import numpy as np
import re

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from quantum_models.qnn.cudaq.circuits import QuantumCircuitConfig
from quantum_models.qnn.cudaq.model import QuantumNeuralNetwork
from utils import (
    get_data_tensors,
    evaluate_model,
    save_results,
    save_training_history,
    plot_training_history,
    compute_multiclass_roc,
    plot_multiclass_roc,
)
from quantum_models.qnn.cudaq.train_model import train_quantum_model


def main(args):
    for trial in range(args.num_of_trials):
        print(f"\n=== Trial {trial + 1}/{args.num_of_trials} ===")
        start_time = dt.datetime.now()

        X_train, X_test, X_val, y_train, y_test, y_val, class_names = get_data_tensors(
            args.dataset_path, args.class_to_keep
        )

        flat_dim = int(np.prod(X_train.shape[1:]))
        pos_pixel = int(np.log2(flat_dim))

        print('Initializing QuantumCircuit Configuration...')
        config = QuantumCircuitConfig(
            n_qubits=pos_pixel,
            num_reps=2,
            feature_map_type=0,
            ansatz_type=0,
            num_attention_params=225,
        )

        print('Creating quantum neural network model...')
        quantum_model = QuantumNeuralNetwork(
            config=config,
            num_classes=2,
            use_classical_head=True,
        )

        print('Training quantum neural network...')
        history = train_quantum_model(
            model=quantum_model,
            x_train=X_train,
            y_train=y_train,
            x_val=X_val,
            y_val=y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            shots=args.shots,
            lr=args.learning_rate,
        )

        end_time = dt.datetime.now()
        duration = end_time - start_time
        print(f"Trial {trial + 1} completed in: {duration}")

        print('Evaluating quantum neural network model...')
        accuracy, report, conf_matrix, precision, recall, f1score = evaluate_model(
            quantum_model, X_test, y_test, label_names=class_names
        )

        trial_save_dir = os.path.join(args.output_dir, f'{trial}')
        os.makedirs(trial_save_dir, exist_ok=True)

        print('Saving results...')
        additional_info = {
            'training_duration': str(duration),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'seed': args.seed + trial,
        }

        plot_training_history(
            history=history,
            title='Quantum NN Training History',
            output_path=trial_save_dir
        )

        save_results(
            accuracy=accuracy,
            report=report,
            conf_matrix=conf_matrix,
            additional_info=additional_info,
            filename='Quantum_NN_Results',
            dataset_path=args.dataset_path,
            output_dir=trial_save_dir
        )

        save_training_history(
            train_loss=history['train_losses'],
            test_accuracies=history['val_accuracies'],
            filepath=os.path.join(trial_save_dir, 'quantum_nn_training_history.json')
        )

        roc_info = compute_multiclass_roc(
            model=quantum_model,
            x_test=X_test,
            y_test=y_test,
            class_names=class_names,
            strategy="ovr"
        )

        plot_multiclass_roc(
            roc_info=roc_info,
            class_names=class_names,
            title='Quantum NN ROC Curves',
            save_path=os.path.join(trial_save_dir, 'quantum_nn_ROC_curves.svg')
        )

        print('Quantum neural network trial completed. All results saved.\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Neural Network Training Script")
    parser.add_argument('--dataset_path', type=str, default="dataset/mnist_8x8.npz", help='Path to the dataset file (npz format)')
    parser.add_argument('--class_to_keep', type=int, nargs='+', default=[0, 1], help='Classes to keep for binary classification')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--output_dir', type=str, default='results/quantum/nn/', help='Directory to save results')
    parser.add_argument('--num_of_trials', type=int, default=1, help='Number of training trials to run')
    parser.add_argument('--shots', type=int, default=100000, help='Number of shots for quantum sampling')

    args = parser.parse_args()
    main(args)
