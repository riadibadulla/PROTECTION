import train
from dataset import load_and_preprocess_data, CustomDataset
from models import MLPModel_likeMands, MLPModel_likeMands_2
from train import train_model, train_model_with_penalty
from filtering import add_data_with_marabou, filter_data_delegate
from filtering_abcrown import add_data_with_abcrown
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
DELEGATION_INTERVAL = 5  # New delegate model is created every 5 iterations

# Load and preprocess the data
X_train, X_test, y_train, y_test = load_and_preprocess_data('Datasets/merged_shuffled_dataset.csv')

# Initialize masked (remaining) training data and counter-examples separately.
masked_train_data = X_train
masked_train_labels = y_train

# Initialize empty arrays for counter-examples (perturbed samples).
counter_examples_data = np.empty((0, X_train.shape[1]))
counter_examples_labels = np.empty((0,))

iteration = 0
delegate_counter = 0
current_model = None

while len(masked_train_data) > 0:
    iteration += 1
    print(f"\n### Iteration {iteration} ###")
    print(f"Masked Training Samples: {len(masked_train_data)}")

    # Start a new delegate cycle every DELEGATION_INTERVAL iterations
    if iteration % DELEGATION_INTERVAL == 1:
        # Save the previous delegate model if it exists (i.e. not the very first delegate)
        if current_model is not None:
            save_name = f"model_epsilon={PERTURBATION}_iteration_{delegate_counter}.pth"
            torch.save(current_model.state_dict(), save_name)
            print(f"Saved delegate model {delegate_counter} as {save_name}.")
        delegate_counter += 1
        print(f"Delegating to a new model. Starting delegate {delegate_counter}.")
        current_model = MLPModel_likeMands_2(masked_train_data.shape[1]).to(device)
        # current_model = MLPModel_likeMands(masked_train_data.shape[1]).to(device)
        # Reset counter-examples for the new delegate cycle.
        counter_examples_data = np.empty((0, masked_train_data.shape[1]))
        counter_examples_labels = np.empty((0,))

    # Build the current training set:
    # Within the same delegate cycle, combine masked data with any counter-examples.
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
    plot_histogram(train_predictions,
                   f"Iteration {iteration}: Training Set Predictions, Per={PERTURBATION}",
                   iteration=iteration,
                   perturbation=PERTURBATION)

    # Apply filtering to separate ambiguous samples from confident ones.
    if USING_SMT:
        train_mask, new_samples, new_labels = add_data_with_marabou(
            current_model, current_train_loader, low_thresh=LOW_THRESHOLD, high_thresh=HIGH_THRESHOLD,
            perturbation=PERTURBATION
        )
        # train_mask, new_samples, new_labels = add_data_with_abcrown(
        #     current_model, current_train_loader, low_thresh=LOW_THRESHOLD, high_thresh=HIGH_THRESHOLD,
        #     perturbation=PERTURBATION
        # )
    else:
        train_mask = filter_data_delegate(current_model, current_train_loader,
                                          low_thresh=LOW_THRESHOLD, high_thresh=HIGH_THRESHOLD)
        new_samples = np.empty((0, current_train_data.shape[1]))
        new_labels = np.empty((0,))

    # Determine how many samples in the current training set came from the original masked data.
    n_mask = len(masked_train_data)

    # Update masked data: keep only the portion of the original masked data that passed filtering.
    if len(train_mask) >= n_mask:
        filtered_masked = train_mask[:n_mask]
        masked_train_data = current_train_data[:n_mask][filtered_masked]
        masked_train_labels = current_train_labels[:n_mask][filtered_masked]
    else:
        # Fallback if dimensions do not match (should rarely occur)
        masked_train_data = current_train_data[train_mask]
        masked_train_labels = current_train_labels[train_mask]

    # Update counter-examples with the newly generated perturbed samples.
    counter_examples_data = new_samples
    counter_examples_labels = new_labels

    # Optionally, if no ambiguous samples remain in the masked dataset, exit the loop.
    if len(masked_train_data) == 0:
        print("No more ambiguous samples left for training. Exiting the loop.")
        break

# After the loop, save the last delegate model (even if its cycle wasn't completed)
if current_model is not None:
    save_name = f"model_epsilon={PERTURBATION}_delegate_{delegate_counter}.pth"
    torch.save(current_model.state_dict(), save_name)
    print(f"Saved final delegate model {delegate_counter} as {save_name}.")