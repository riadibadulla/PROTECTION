import os
import torch
import numpy as np
from dataset import load_and_preprocess_data, CustomDataset
from models import MLPModel_likeMands
from filtering import filter_data_by_model_with_marabou, filter_data_delegate
from evaluate import plot_histogram
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import multiprocessing

# Choose device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

torch.manual_seed(1997)
np.random.seed(1997)

# Constants
LOW_THRESHOLD = 0.48
HIGH_THRESHOLD = 0.52
TRAINED_WITH_PERTURBATION = 0.09
adversarial_epsilon = 0.01
output_dir = "figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data('Datasets/merged_shuffled_dataset.csv')
y_test = (y_test > 0.5).astype(int)

# Save original test set for adversarial generation
full_test_data = X_test.copy()
full_test_labels = y_test.copy()

# Initial datasets for clean predictions (filtered iteratively)
remaining_test_data = X_test
remaining_test_labels = y_test

final_predictions = np.zeros(len(remaining_test_data))
test_indices = np.arange(len(remaining_test_data))
iteration = 0

# For ROC/AUC on clean predictions
all_predictions = []
all_true_labels = []

# For adversarial predictions (we accumulate predictions on the full test set)
adv_prediction_sum = np.zeros(len(full_test_data))
adv_model_count = 0

# --- Define a simple FGSM adversarial attack ---
def fgsm_attack(model, data, label, epsilon, device):
    data.requires_grad = True
    model.eval()
    criterion = torch.nn.BCELoss()
    output = model(data)
    loss = criterion(output, label)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    return perturbed_data.detach()

# --- Main loop over models ---
while True:
    iteration += 1
    model_path = f"model_epsilon={TRAINED_WITH_PERTURBATION}_iteration_{iteration}.pth"
    if not os.path.exists(model_path):
        print(f"No more models found. Exiting after {iteration-1} iterations.")
        break

    print(f"\n### Iteration {iteration} ###")
    print(f"Remaining Test Samples: {len(remaining_test_data)}")

    model = MLPModel_likeMands(remaining_test_data.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    current_test_dataset = CustomDataset(remaining_test_data, remaining_test_labels)
    current_test_loader = DataLoader(current_test_dataset, batch_size=32, shuffle=False)

    test_mask = []
    predictions = []

    with torch.no_grad():
        for i, (features, label) in enumerate(current_test_dataset):
            features = features.to(device)
            proba = model(features).item()
            predictions.append(proba)
            all_predictions.append(proba)
            all_true_labels.append(label.item())
            if LOW_THRESHOLD <= proba <= HIGH_THRESHOLD:
                test_mask.append(True)
            else:
                test_mask.append(False)
        for i, include in enumerate(test_mask):
            if not include:
                final_predictions[test_indices[i]] = predictions[i]

    test_mask = np.array(test_mask)
    if test_mask.size > 0:
        remaining_test_data = remaining_test_data[test_mask]
        remaining_test_labels = remaining_test_labels[test_mask]
        test_indices = test_indices[test_mask]

    # ---- Generate adversarial predictions on the full test set ----
    full_dataset = CustomDataset(full_test_data, full_test_labels)
    for i, (features, label) in enumerate(full_dataset):
        features = features.to(device)
        # Create label tensor matching output shape
        label_tensor = torch.tensor([label], dtype=torch.float32, device=device)
        adv_features = fgsm_attack(model, features, label_tensor, adversarial_epsilon, device)
        adv_pred = model(adv_features).item()
        adv_prediction_sum[i] += adv_pred
    adv_model_count += 1

# --- Clean evaluation ---
if np.any(final_predictions > 0):
    final_labels = y_test
    combined_accuracy = np.mean((final_predictions > 0.5) == final_labels)
    print(f"Final Combined Model Accuracy: {combined_accuracy*100:.2f}%")

    if all_predictions and all_true_labels:
        fpr, tpr, thresholds = roc_curve(all_true_labels, all_predictions)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
        roc_curve_path = os.path.join(output_dir, f"roc_curve_train_per_{TRAINED_WITH_PERTURBATION}.png")
        plt.savefig(roc_curve_path)
        print(f"AUC Score: {roc_auc:.2f}")
    else:
        print("No predictions or true labels available for ROC curve calculation.")
else:
    print("No confident predictions available for evaluation.")

# --- Adversarial evaluation ---
if adv_model_count > 0:
    ensemble_adv_predictions = adv_prediction_sum / adv_model_count
    combined_adv_accuracy = np.mean((ensemble_adv_predictions > 0.5) == full_test_labels)
    print(f"Final Ensemble Adversarial Accuracy: {combined_adv_accuracy*100:.2f}%")
    fpr_adv, tpr_adv, _ = roc_curve(full_test_labels, ensemble_adv_predictions)
    roc_auc_adv = auc(fpr_adv, tpr_adv)
    plt.figure()
    plt.plot(fpr_adv, tpr_adv, color='darkorange', lw=2, label=f'Adv ROC curve (area = {roc_auc_adv:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Adversarial ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    adv_roc_curve_path = os.path.join(output_dir, f"roc_curve_adv_train_per_{TRAINED_WITH_PERTURBATION}.png")
    plt.savefig(adv_roc_curve_path)
    print(f"Adversarial AUC Score: {roc_auc_adv:.2f}")

    # Plot histogram of adversarial predictions (assuming plot_histogram accepts two arguments)
    plot_histogram(ensemble_adv_predictions, title='Adversarial Predictions Histogram')
else:
    print("No adversarial predictions available for evaluation.")