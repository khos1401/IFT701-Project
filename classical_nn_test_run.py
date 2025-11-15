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
from classical_models.nn.load_data import prepare_data
from classical_models.nn.train_model import train_classical_model
from classical_models.nn.utils import (
    evaluate_model, save_results, save_training_history, 
    plot_training_history, compute_multiclass_roc, 
    plot_multiclass_roc
)

def main(args):

    for trial in range(args.num_of_trials):
        print(f"\n=== Trial {trial + 1}/{args.num_of_trials} ===")

        start_time = dt.datetime.now()

        # Prepare data
        (x_train, x_test, x_val, y_train, y_test, y_val, class_names) = prepare_data(dataset_filename=args.dataset_path, seed=args.seed + trial)

        input_size = int(np.prod(x_train.shape[1:]))
        
        print('Intializing classical neural network model...')
        classical_model = ClassicalNN(
            input_size=input_size,
            num_classes=2,
        )

        print('Training classical neural network...')
        history = train_classical_model(
            model=classical_model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            x_val=x_val,
            y_val=y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

        unique, counts = np.unique(y_train.numpy(), return_counts=True)
        print(dict(zip(unique, counts)))
        
        end_time = dt.datetime.now()
        duration = end_time - start_time
        print(f"Trial {trial + 1} completed in: {duration}")

        print('Evaluating classical neural network model...')
        accuracy, report, conf_matrix, precision, recall, f1score = evaluate_model(
            classical_model, x_test, y_test, label_names=class_names
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
            x_test=x_test,
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
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file (npz format)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate for optimizer')
    parser.add_argument('--output_dir', type=str, default='results/classical/nn/', help='Directory to save results')
    parser.add_argument('--num_of_trials', type=int, default=3, help='Number of training trials to run')
    
    args = parser.parse_args()
    main(args)
