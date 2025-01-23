import train
from dataset import load_and_preprocess_data, CustomDataset
from models import MLPModel_likeMands
from train import train_model
from filtering import filter_data_by_model_with_marabou, filter_data_delegate
from evaluate import plot_histogram
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import time
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

# Constants
NUMBER_OF_EPOCHS = 10
LR = 0.01
USING_SMT = False
LOW_THRESHOLD = 0.4
HIGH_THRESHOLD = 0.6
PERTURBATION = 0.001

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

# Initialize for ROC-AUC calculation
all_predictions = []  # Store all probabilities
all_true_labels = []  # Store all true labels

while len(remaining_train_data) > 10:
    iteration += 1
    print(f"\n### Iteration {iteration} ###")
    print(f"Remaining Training Samples: {len(remaining_train_data)}")
    print(f"Remaining Test Samples: {len(remaining_test_data)}")

    # Create a new model
    model = MLPModel_likeMands(remaining_train_data.shape[1]).to(device)
    criterion = nn.BCELoss()

    # Prepare data loaders for the current filtered data
    current_train_dataset = CustomDataset(remaining_train_data, remaining_train_labels)
    current_train_loader = DataLoader(current_train_dataset, batch_size=32, shuffle=True)

    current_test_dataset = CustomDataset(remaining_test_data, remaining_test_labels)
    current_test_loader = DataLoader(current_test_dataset, batch_size=32, shuffle=False)

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
        for i, (features, label) in enumerate(current_test_dataset):  # Add label
            features = features.to(device)
            proba = model(features).item()
            predictions.append(proba)

            # Store for ROC and AUC
            all_predictions.append(proba)
            all_true_labels.append(label.item())

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
    plot_histogram(predictions, f"Iteration {iteration}: Test Set Predictions, Per={PERTURBATION}", iteration=iteration, perturbation=PERTURBATION)
    plot_histogram(train_predictions, f"Iteration {iteration}: Training Set Predictions, Per={PERTURBATION}", iteration=iteration, perturbation=PERTURBATION)

    # Store predictions for test samples not filtered for the next iteration
    for i, include in enumerate(test_mask):
        if not include:  # Only store confident predictions
            final_predictions[test_indices[i]] = predictions[i]

    # Update test dataset for the next iteration
    test_mask = np.array(test_mask)
    if test_mask.size > 0:
        remaining_test_data = remaining_test_data[test_mask]
        remaining_test_labels = remaining_test_labels[test_mask]
        test_indices = test_indices[test_mask]

    # Filter training data
    if USING_SMT:
        train_mask = filter_data_by_model_with_marabou(model, current_train_loader, low_thresh=LOW_THRESHOLD, high_thresh=HIGH_THRESHOLD, perturbation=PERTURBATION)
    else:
        train_mask = filter_data_delegate(model, current_train_loader, low_thresh=LOW_THRESHOLD, high_thresh=HIGH_THRESHOLD)

    remaining_train_data = remaining_train_data[train_mask]
    remaining_train_labels = remaining_train_labels[train_mask]

    if len(remaining_train_data) == 0:
        print("No more ambiguous samples left for training. Exiting the loop.")
    model.eval()
    torch.save(model.state_dict(), f"model_epsilon={PERTURBATION}_iteration_{iteration}.pth")

# Final histogram and accuracy
plot_histogram(final_predictions[final_predictions > 0], f"Final Combined Predictions (Confident Test Samples) per={PERTURBATION}", perturbation=PERTURBATION)

# Evaluate combined predictions
final_labels = y_test
combined_accuracy = np.mean((final_predictions > 0.5) == final_labels)
print(f"Final Combined Model Accuracy: {combined_accuracy * 100:.2f}%")

# Compute and Plot ROC-AUC
fpr, tpr, thresholds = roc_curve(all_true_labels, all_predictions)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic (ROC) Curve, for Perturbations={PERTURBATION}')
plt.legend(loc="lower right")
plt.show()

# Create a unique filename
unique_id = f"iter_{iteration}" if iteration is not None else f"ts_{int(time.time())}"
filename = f"roc_curve_{unique_id}_per={PERTURBATION}.png"
plt.savefig(os.path.join(output_dir, filename))  # Save figure to 'figures' folder

# Print the AUC score
print(f"AUC Score: {roc_auc:.2f}")