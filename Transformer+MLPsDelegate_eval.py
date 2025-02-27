import os
import torch
import numpy as np
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

# Constants
USING_SMT = False
LOW_THRESHOLD = 0.48
HIGH_THRESHOLD = 0.53
PERTURBATION = 0.01
TRAINED_WITH_PERTURBATION = 0.09
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
    model_path = f"model_trans_epsilon={TRAINED_WITH_PERTURBATION}_iteration_{iteration}.pth"

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"No more models found. Exiting after {iteration - 1} iterations.")
        break

    print(f"\n### Iteration {iteration} ###")
    print(f"Remaining Test Samples: {len(remaining_test_data)}")

    # Load the model
    if iteration == 1:
        model = TransformerIDS().to(device)
        LOW_THRESHOLD = 0.05
        HIGH_THRESHOLD = 0.95
        # USING_SMT = True
    else:
        model = MLPModel_likeMands(remaining_test_data.shape[1]).to(device)
        LOW_THRESHOLD = 0.48
        HIGH_THRESHOLD = 0.53
        # USING_SMT = True
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
            features = features.unsqueeze(0).unsqueeze(0)
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
        plot_histogram(predictions,
                       f"Iteration {iteration}: Test Set Predictions, Train_per = {TRAINED_WITH_PERTURBATION}, Tes_Per={PERTURBATION}",
                       iteration=iteration, perturbation=PERTURBATION)
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
    plot_histogram(final_predictions[final_predictions > 0],
                   f"Final Combined Predictions (Confident Test Samples) per={PERTURBATION}", perturbation=PERTURBATION)

    # Evaluate combined predictions
    final_labels = y_test
    combined_accuracy = np.mean((final_predictions > 0.5) == final_labels)
    print(f"Final Combined Model Accuracy: {combined_accuracy * 100:.2f}%")

    # Compute and Plot ROC-AUC
    if all_predictions and all_true_labels:
        fpr, tpr, thresholds = roc_curve(all_true_labels, all_predictions)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve, for Perturbations={PERTURBATION}')
        plt.legend(loc="lower right")
        plt.show()

        print(f"AUC Score: {roc_auc:.2f}")

        fpr_5_idx = np.argmin(np.abs(fpr - 0.05))
        fpr_15_idx = np.argmin(np.abs(fpr - 0.15))
        print(f"TPR at FPR 5%: {tpr[fpr_5_idx]:.4f}")
        print(f"TPR at FPR 15%: {tpr[fpr_15_idx]:.4f}")

        final_predictions_binary = (final_predictions > 0.5).astype(int)
        fpr_final = np.sum((final_predictions_binary == 1) & (y_test == 0)) / np.sum(y_test == 0)
        fnr_final = np.sum((final_predictions_binary == 0) & (y_test == 1)) / np.sum(y_test == 1)
        print(f"Final FPR at threshold 0.5: {fpr_final:.4f}")
        print(f"Final FNR at threshold 0.5: {fnr_final:.4f}")
