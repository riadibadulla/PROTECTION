import train
from dataset import load_and_preprocess_data, CustomDataset
from models import MLPModel_simple
from train import train_model
from filtering import filter_data_by_model_with_marabou, filter_data_delegate
from evaluate import evaluate_model, plot_histogram
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch

# Determine the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device = torch.device("cpu")  # Overwritten device
print(f"Using device: {device}")

# Constants
NUMBER_OF_EPOCHS = 20
LR = 0.001
USING_SMT = True
LOW_THRESHOLD = 0.4
HIGH_THRESHOLD = 0.6


# Load and preprocess the data
X_train, X_test, y_train, y_test = load_and_preprocess_data('Datasets/merged_shuffled_dataset.csv')

# Initial datasets
remaining_train_data = X_train
remaining_train_labels = y_train

remaining_test_data = X_test
remaining_test_labels = y_test

# Initialize predictions storage
final_predictions = np.zeros(len(remaining_test_data))
test_indices = np.arange(len(remaining_test_data))  # Keep track of test indices
model_list = []
iteration = 0

while len(remaining_train_data) > 0:
    iteration += 1
    print(f"\n### Iteration {iteration} ###")
    print(f"Remaining Training Samples: {len(remaining_train_data)}")
    print(f"Remaining Test Samples: {len(remaining_test_data)}")

    # Create a new model
    model = MLPModel_simple(remaining_train_data.shape[1]).to(device)
    criterion = nn.BCELoss()

    # Prepare data loaders for the current filtered data
    current_train_dataset = CustomDataset(remaining_train_data, remaining_train_labels)
    current_train_loader = DataLoader(current_train_dataset, batch_size=32, shuffle=True)

    current_test_dataset = CustomDataset(remaining_test_data, remaining_test_labels)
    current_test_loader = DataLoader(current_test_dataset, batch_size=32)

    # Train the model
    train_model(model, current_train_loader, criterion, lr=LR, epochs=NUMBER_OF_EPOCHS)
    model_list.append(model)

    # Evaluate the model and filter test data
    model.eval()
    test_mask = []
    predictions = []
    train_predictions = []

    with torch.no_grad():
        # Predictions for current test set
        for i, (features, _) in enumerate(current_test_dataset):
            features = features.to(device)  # Removed unnecessary unsqueeze
            proba = model(features).item()  # Directly use features as input
            predictions.append(proba)

            # Determine which samples to exclude (filter for the next iteration)
            if LOW_THRESHOLD <= proba <= HIGH_THRESHOLD:
                test_mask.append(True)  # Keep for next iteration
            else:
                test_mask.append(False)  # Exclude from next iteration

        # Predictions for current training set
        for features, _ in current_train_loader:
            features = features.to(device)
            proba = model(features).squeeze().cpu().numpy()
            train_predictions.extend(proba)

    # Plot histograms for current iteration
    plot_histogram(predictions, f"Iteration {iteration}: Test Set Predictions")
    plot_histogram(train_predictions, f"Iteration {iteration}: Training Set Predictions")

    # Store predictions for test samples not filtered for the next iteration
    for i, include in enumerate(test_mask):
        if not include:  # Only store confident predictions
            final_predictions[test_indices[i]] = predictions[i]

    # Update test dataset for the next iteration
    test_mask = np.array(test_mask)
    remaining_test_data = remaining_test_data[test_mask]
    remaining_test_labels = remaining_test_labels[test_mask]
    test_indices = test_indices[test_mask]

    # Filter training data
    if USING_SMT:
        train_mask = filter_data_by_model_with_marabou(model, current_train_loader, low_thresh=LOW_THRESHOLD, high_thresh=HIGH_THRESHOLD)
    else:
        train_mask = filter_data_delegate(model, current_train_loader, low_thresh=LOW_THRESHOLD, high_thresh=HIGH_THRESHOLD)

    remaining_train_data = remaining_train_data[train_mask]
    remaining_train_labels = remaining_train_labels[train_mask]

    if len(remaining_train_data) == 0:
        print("No more ambiguous samples left for training. Exiting the loop.")

# Final histogram and accuracy
plot_histogram(final_predictions[final_predictions > 0], "Final Combined Predictions (Confident Test Samples)")

# Evaluate combined predictions
final_labels = y_test
combined_accuracy = np.mean((final_predictions > 0.5) == final_labels)
print(f"Final Combined Model Accuracy: {combined_accuracy * 100:.2f}%")