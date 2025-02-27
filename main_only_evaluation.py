
import os
import torch
import numpy as np
import argparse
from dataset import load_and_preprocess_data, CustomDataset
from models import MLPModel_likeMands, TransformerIDS
from filtering import filter_data_by_model_with_marabou, filter_data_delegate
from evaluate import plot_histogram
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
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

torch.manual_seed(1997)
np.random.seed(1997)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train model with custom parameters.')
parser.add_argument('--using_smt', type=bool, default=True, help='Whether to use SMT filtering (default: True)')
parser.add_argument('--low_threshold', type=float, default=0.4, help='Low threshold for filtering (default: 0.4)')
parser.add_argument('--high_threshold', type=float, default=0.6, help='High threshold for filtering (default: 0.6)')
parser.add_argument('--perturbation', type=float, default=0.09, help='Perturbation value (default: 0.09)')
parser.add_argument('--trained_perturbation', type=float, default=0.09, help='Perturbation value (default: 0.09)')
args = parser.parse_args()

# Constants
USING_SMT = args.using_smt
LOW_THRESHOLD = args.low_threshold
HIGH_THRESHOLD = args.high_threshold
PERTURBATION = args.perturbation
TRAINED_WITH_PERTURBATION = args.trained_perturbation

# Load and preprocess the data
X_train, X_test, y_train, y_test = load_and_preprocess_data('Datasets/merged_shuffled_dataset.csv')

# Ensure test labels are binary (0 or 1)
y_test = (y_test > 0.5).astype(int)

# Initial datasets
remaining_test_data = X_test
remaining_test_labels = y_test

# Initialize predictions storage
final_predictions = np.zeros(len(remaining_test_data))
test_indices = np.arange(len(remaining_test_data))  # Keep track of test indices
iteration = 0

# Initialize for ROC-AUC calculation
all_predictions = []  # Store all probabilities
all_true_labels = []  # Store all true labels

while True:
    iteration += 1
    model_path = f"model_epsilon={TRAINED_WITH_PERTURBATION}_iteration_{iteration}.pth"
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"No more models found. Exiting after {iteration - 1} iterations.")
        break

    print(f"\n### Iteration {iteration} ###")
    print(f"Remaining Test Samples: {len(remaining_test_data)}")

    # Load the model
    model = MLPModel_likeMands(remaining_test_data.shape[1]).to(device)
    # model = TransformerIDS().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Wrap the test data in a DataLoader
    current_test_dataset = CustomDataset(remaining_test_data, remaining_test_labels)
    current_test_loader = DataLoader(current_test_dataset, batch_size=32, shuffle=False)

    # Evaluate the model and filter test data
    test_mask = []
    predictions = []

    with torch.no_grad():
        # Predictions for current test set
        for i, (features, label) in enumerate(current_test_dataset):  # Add label
            features = features.to(device)
            proba = model(features).item()
            # outputs = model(features).squeeze().cpu().numpy()
            predictions.append(proba)

            # Store for ROC and AUC
            all_predictions.append(proba)
            all_true_labels.append(label.item())

            # Determine which samples to exclude (filter for the next iteration)
            if not USING_SMT:
                # Filter training data
                if LOW_THRESHOLD <= proba <= HIGH_THRESHOLD:
                    test_mask.append(True)  # Keep for next iteration
                else:
                    test_mask.append(False)  # Exclude from next iteration

        # Plot histograms for current iteration
        plot_histogram(predictions, f"Iteration {iteration}: Test Set Predictions, Train_per = {TRAINED_WITH_PERTURBATION}, Tes_Per={PERTURBATION}", iteration=iteration, perturbation=PERTURBATION)
        if USING_SMT:
            test_mask = filter_data_by_model_with_marabou(model, current_test_loader, low_thresh=LOW_THRESHOLD,
                                                          high_thresh=HIGH_THRESHOLD, perturbation=PERTURBATION)
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

# Final histogram and accuracy
if np.any(final_predictions > 0):
    plot_histogram(final_predictions[final_predictions > 0], f"Final Combined Predictions (Confident Test Samples) per={PERTURBATION}", perturbation=PERTURBATION)

    # Evaluate combined predictions
    final_labels = y_test
    combined_accuracy = np.mean((final_predictions > 0.5) == final_labels)
    print(f"Final Combined Model Accuracy: {combined_accuracy * 100:.2f}%")

    # Compute and Plot ROC-AUC
    if all_predictions and all_true_labels:
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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # Ensure the directory exists

        roc_curve_path = os.path.join(output_dir, f"roc_curve_train_per_{TRAINED_WITH_PERTURBATION} test_perturbation_{PERTURBATION}_SMT_USED={USING_SMT}.png")
        plt.savefig(roc_curve_path)
        # Print the AUC score
        print(f"AUC Score: {roc_auc:.2f}")
    else:
        print("No predictions or true labels available for ROC curve calculation.")
else:
    print("No confident predictions available for evaluation.")
