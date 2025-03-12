"""
File main_samples_with_smt.py
Author: Riad Ibadulla
Date: 06-Mar-2025
Description: This file contains the method similar to protection but instead of masking the original dataset,
we also pass counter-examples (perturbed samples) to the next delegate classifier. However, a new model is
only created every 5 iterations. Within these 5 iterations, the model trains on both the remaining (masked)
samples and any counter-examples produced. When delegating (starting a new model), only the masked samples
are used.
"""

import train
from dataset import load_and_preprocess_data, CustomDataset
from models import MLPModel_likeMands, MLPModel_likeMands_2
from train import train_model, train_model_with_penalty
from filtering import add_data_with_marabou, filter_data_delegate, add_data_with_ab_crown
from evaluate import plot_histogram
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch
import argparse

# Set random seeds for reproducibility
torch.manual_seed(1997)
np.random.seed(1997)

# Choose the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device = torch.device("cpu")  # Overwritten device
print(f"Using device: {device}")

output_dir = "figures"

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train model with custom parameters.')
parser.add_argument('--using_smt', type=bool, default=True, help='Whether to use SMT filtering (default: True)')
parser.add_argument('--low_threshold', type=float, default=0.4, help='Low threshold for filtering (default: 0.4)')
parser.add_argument('--high_threshold', type=float, default=0.6, help='High threshold for filtering (default: 0.6)')
parser.add_argument('--perturbation', type=float, default=0.01, help='Perturbation value (default: 0.09)')
args = parser.parse_args()

# Set parameters from command-line arguments
USING_SMT = args.using_smt
LOW_THRESHOLD = args.low_threshold
HIGH_THRESHOLD = args.high_threshold
PERTURBATION = args.perturbation

# Constants
NUMBER_OF_EPOCHS = 50
LR = 0.01
DELEGATION_INTERVAL = 5  # New model is created every 5 iterations

# Load and preprocess the data
X_train, X_test, y_train, y_test = load_and_preprocess_data('Datasets/merged_shuffled_dataset.csv')

# Initialize masked (remaining) training data and counter-examples separately.
masked_train_data = X_train
masked_train_labels = y_train

# Initialize empty arrays for counter-examples (perturbed samples).
counter_examples_data = np.empty((0, X_train.shape[1]))
counter_examples_labels = np.empty((0,))

iteration = 0
current_model = None

while len(masked_train_data) > 10:
    iteration += 1
    print(f"\n### Iteration {iteration} ###")
    print(f"Masked Training Samples: {len(masked_train_data)}")

    # On the first iteration of each 5-iteration cycle, delegate to a new model
    if iteration % DELEGATION_INTERVAL == 1:
        print("Delegating to a new model.")
        current_model = MLPModel_likeMands_2(masked_train_data.shape[1]).to(device)
        # When delegating, reset any counter-examples
        counter_examples_data = np.empty((0, masked_train_data.shape[1]))
        counter_examples_labels = np.empty((0,))

    # Build the current training set:
    # For iterations within the same model cycle, combine masked data with counter-examples.
    if counter_examples_data.size > 0:
        current_train_data = np.concatenate([masked_train_data, counter_examples_data], axis=0)
        current_train_labels = np.concatenate([masked_train_labels, counter_examples_labels], axis=0)
    else:
        current_train_data = masked_train_data
        current_train_labels = masked_train_labels

    # Prepare data loader for the current training set
    current_train_dataset = CustomDataset(current_train_data, current_train_labels)
    current_train_loader = DataLoader(current_train_dataset, batch_size=32, shuffle=True)

    # Train the current model on the current training set
    criterion = nn.BCELoss()
    # train_model(current_model, current_train_loader, criterion, lr=LR, epochs=NUMBER_OF_EPOCHS)
    train_model_with_penalty(current_model,
                current_train_loader,
                criterion,
                lr=LR,
                epochs=NUMBER_OF_EPOCHS,
                penalty_lambda=0.01,
                lower=LOW_THRESHOLD,
                upper=HIGH_THRESHOLD)

    # Evaluate the model and plot histogram for current training set predictions
    current_model.eval()
    train_predictions = []
    with torch.no_grad():
        for features, _ in current_train_loader:
            features = features.to(device)
            proba = current_model(features).squeeze().cpu().numpy()
            train_predictions.extend(proba.flatten())
    plot_histogram(train_predictions, f"Iteration {iteration}: Training Set Predictions, Per={PERTURBATION}",
                   iteration=iteration, perturbation=PERTURBATION)

    # Apply filtering to separate ambiguous samples from confident ones.
    if USING_SMT:
        # train_mask, new_samples, new_labels = add_data_with_marabou(
        #     current_model, current_train_loader, low_thresh=LOW_THRESHOLD, high_thresh=HIGH_THRESHOLD, perturbation=PERTURBATION
        # )
        train_mask, new_samples, new_labels = add_data_with_ab_crown(
            current_model, current_train_loader, low_thresh=LOW_THRESHOLD, high_thresh=HIGH_THRESHOLD,
            perturbation=PERTURBATION
        )
    else:
        train_mask = filter_data_delegate(current_model, current_train_loader, low_thresh=LOW_THRESHOLD,
                                          high_thresh=HIGH_THRESHOLD)
        new_samples = np.empty((0, current_train_data.shape[1]))
        new_labels = np.empty((0,))

    # The train_mask is for the entire current training set (combined data).
    # We know how many samples came from the original masked data:
    n_mask = len(masked_train_data)

    # Update masked data: keep only the portion of the original masked data that passed filtering.
    # (Assumes train_mask is a boolean array of length len(current_train_data))
    if len(train_mask) >= n_mask:
        filtered_masked = train_mask[:n_mask]
        masked_train_data = current_train_data[:n_mask][filtered_masked]
        masked_train_labels = current_train_labels[:n_mask][filtered_masked]
    else:
        # Fallback if dimensions do not match (should rarely occur)
        masked_train_data = current_train_data[train_mask]
        masked_train_labels = current_train_labels[train_mask]

    # Update counter-examples with the newly generated perturbed samples.
    # Note: These new counter-examples are used only within the current cycle.
    counter_examples_data = new_samples
    counter_examples_labels = new_labels

    # Optionally, if no ambiguous samples remain in the masked dataset, exit the loop.
    if len(masked_train_data) == 0:
        print("No more ambiguous samples left for training. Exiting the loop.")
        break

    # Save the model state for this iteration
    torch.save(current_model.state_dict(), f"model_epsilon={PERTURBATION}_iteration_{iteration}.pth")