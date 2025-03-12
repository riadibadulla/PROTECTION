"""
File main_samples_with_smt.py
Author: Riad Ibadulla
Date: 06-Mar-2025
Description: This file contains the method similar to protection but instead of masking original dataset we also pass
counter-examples (perturbed samples) to the next delegate classifier. However, we do not train new models, instead we
load protection weights and fine tune on them.
"""

import train
from dataset import load_and_preprocess_data, CustomDataset
from models import MLPModel_likeMands
from train import train_model
from filtering import add_data_with_marabou, filter_data_delegate
from evaluate import plot_histogram
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch
import argparse
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import time

# Set random seeds for reproducibility
torch.manual_seed(1997)
np.random.seed(1997)

# Choose the device (here overwritten to CPU for consistency)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

output_dir = "figures"

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train model with custom parameters.')
parser.add_argument('--using_smt', type=bool, default=True, help='Whether to use SMT filtering (default: True)')
parser.add_argument('--low_threshold', type=float, default=0.4, help='Low threshold for filtering (default: 0.4)')
parser.add_argument('--high_threshold', type=float, default=0.6, help='High threshold for filtering (default: 0.6)')
parser.add_argument('--perturbation', type=float, default=0.09, help='Perturbation value (default: 0.09)')
args = parser.parse_args()

# Set parameters from command-line arguments
USING_SMT = args.using_smt
LOW_THRESHOLD = args.low_threshold
HIGH_THRESHOLD = args.high_threshold
PERTURBATION = args.perturbation

# Constants for training
NUMBER_OF_EPOCHS = 30
LR = 0.01

# Load and preprocess the data
X_train, X_test, y_train, y_test = load_and_preprocess_data('Datasets/merged_shuffled_dataset.csv')

# Initialize the training dataset (this could be the original training set or an already filtered version)
remaining_train_data = X_train
remaining_train_labels = y_train

model_list = []
iteration = 1  # Starting iteration

# Loop over saved model files
while True:
    model_filename = f"model_epsilon={PERTURBATION}_iteration_{iteration}.pth"
    if not os.path.exists(model_filename):
        print(f"No saved model found for iteration {iteration}. Ending training loop.")
        break

    print(f"\n### Iteration {iteration} ###")
    print(f"Training samples count: {len(remaining_train_data)}")

    # Create a new model instance and load the saved weights
    model = MLPModel_likeMands(remaining_train_data.shape[1]).to(device)
    model.load_state_dict(torch.load(model_filename))

    # Set up loss criterion and data loader
    criterion = nn.BCELoss()
    current_train_dataset = CustomDataset(remaining_train_data, remaining_train_labels)
    current_train_loader = DataLoader(current_train_dataset, batch_size=32, shuffle=True)

    # Continue training the loaded model
    train_model(model, current_train_loader, criterion, lr=LR, epochs=NUMBER_OF_EPOCHS)
    model_list.append(model)

    # Evaluate the model and collect predictions for histogram plotting
    model.eval()
    train_predictions = []
    with torch.no_grad():
        for features, _ in current_train_loader:
            features = features.to(device)
            proba = model(features).squeeze().cpu().numpy()
            train_predictions.extend(proba.flatten())
    plot_histogram(train_predictions,
                   f"Iteration {iteration}: Training Set Predictions, Per={PERTURBATION}",
                   iteration=iteration, perturbation=PERTURBATION)

    # Use filtering to update the training data
    if USING_SMT:
        train_mask, new_samples, new_labels = add_data_with_marabou(
            model, current_train_loader,
            low_thresh=LOW_THRESHOLD, high_thresh=HIGH_THRESHOLD,
            perturbation=PERTURBATION
        )
    else:
        train_mask = filter_data_delegate(
            model, current_train_loader,
            low_thresh=LOW_THRESHOLD, high_thresh=HIGH_THRESHOLD
        )
        new_samples = np.empty((0, remaining_train_data.shape[1]))
        new_labels = np.empty((0,))

    # Update remaining training data (filtering out some samples and adding new perturbed ones)
    remaining_train_data = remaining_train_data[train_mask]
    remaining_train_labels = remaining_train_labels[train_mask]
    if new_samples.size > 0:
        remaining_train_data = np.concatenate([remaining_train_data, new_samples], axis=0)
        remaining_train_labels = np.concatenate([remaining_train_labels, new_labels], axis=0)

    # Save the updated model back to file
    torch.save(model.state_dict(), model_filename)

    iteration += 1