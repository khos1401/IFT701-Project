import sys
import os
import datetime as dt
import argparse
import torch
import numpy as np
import re

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from classical_models.nn.model import ClassicalNN
from utils import get_data_tensors, evaluate_model, save_results, save_training_history, plot_training_history, compute_multiclass_roc, plot_multiclass_roc
from classical_models.train_model_and_save import train_classical_model
from classical_models.models import ClassicalNN, ClassicalCNN

def main(args):
    for trial in range(args.num_of_trials):
        print(f"\n=== Trial {trial + 1}/{args.num_of_trials} ===")
        start_time = dt.datetime.now()

        X_train, X_test, X_val, y_train, y_test, y_val, class_names = get_data_tensors(args.dataset_path, args.class_to_keep)

        classical_model = ClassicalCNN(
            input_size=X_train.shape[1:],
            num_classes=2,
        )

        print('Training classical neural network...')
        history = train_classical_model(
            model=classical_model,
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
        
        end_time = dt.datetime.now()
        duration = end_time - start_time
        print(f"Trial {trial + 1} completed in: {duration}")

        print('Evaluating classical neural network model...')
        accuracy, report, conf_matrix, precision, recall, f1score = evaluate_model(
            classical_model, X_test, y_test, label_names=class_names
        )

        trial_save_dir = os.path.join(args.output_dir, f'{trial}')
        os.makedirs(trial_save_dir, exist_ok=True)

        print('Saving results...')
        model_info  = classical_model.get_model_info()

        additional_info = {
            'training_duration': str(duration),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'seed': args.seed + trial,
            'model_info': model_info,
        }

        plot_training_history(
            history=history,
            title='Classical NN Training History',
            output_path=trial_save_dir
        )

        save_results(
            accuracy=accuracy,
            report=report,
            conf_matrix=conf_matrix,
            additional_info=additional_info,
            filename='Classical_NN_Results',
            dataset_path=args.dataset_path,
            output_dir=trial_save_dir
        )

        save_training_history(
            train_loss=history['train_losses'],
            test_accuracies=history['val_accuracies'],
            filepath=os.path.join(trial_save_dir, 'classical_nn_training_history.json')
        )

        roc_info = compute_multiclass_roc(
            model=classical_model,
            x_test=X_test,
            y_test=y_test,
            class_names=class_names,
            strategy="ovr"
        )

        plot_multiclass_roc(
            roc_info=roc_info,
            class_names=class_names,
            title='Classical NN ROC Curves',
            save_path=os.path.join(trial_save_dir, 'classical_nn_ROC_curves.svg')
        )

        print('Classical neural network trial completed. All results saved.\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classical Neural Network Training Script")
    parser.add_argument('--dataset_path', type=str, default="dataset/mnist_8x8.npz", help='Path to the dataset file (npz format)')
    parser.add_argument('--class_to_keep', type=int, default=[8, 9], help='Classes to keep for binary classification')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--output_dir', type=str, default='results/classical/nn/', help='Directory to save results')
    parser.add_argument('--num_of_trials', type=int, default=1, help='Number of training trials to run')
    
    args = parser.parse_args()
    main(args)
