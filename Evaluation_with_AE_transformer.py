import os
import torch
import numpy as np
import glob
import random
from dataset import load_and_preprocess_data, CustomDataset
from models import MLPModel_likeMands, TransformerIDS
from filtering import filter_data_delegate
from evaluate import plot_histogram
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from attacks import fgsm_attack_l2, bim_attack_l2

# Device and seed settings
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

output_dir = "figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# torch.manual_seed(1997)
# np.random.seed(1997)

# Constants
LOW_THRESHOLD = 0.4
HIGH_THRESHOLD = 0.5
TRAINED_WITH_PERTURBATION = 0.09
# TRAINED_WITH_PERTURBATION_LIST = [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
adversarial_epsilon = 0.01
N_RUNS = 1  # Number of times to generate and evaluate AE

print(f"Trained with perturbation: {TRAINED_WITH_PERTURBATION}")
# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data('Datasets/merged_shuffled_dataset.csv')
y_test = (y_test > 0.5).astype(int)
full_test_data = X_test.copy()
full_test_labels = y_test.copy()

# Get model paths
model_paths = sorted(glob.glob(f"model_trans_epsilon={TRAINED_WITH_PERTURBATION}_iteration_*.pth"))

# --- Run AE generation & evaluation multiple times ---
ae_accuracies = []

for run in range(N_RUNS):
    print(f"\n### Running Adversarial Evaluation {run+1}/{N_RUNS} ###")
    # Generate adversarial examples using one random model per sample
    adv_examples = np.zeros_like(full_test_data)

    for i in range(len(full_test_data)):
        chosen_model_path =  f"model_trans_epsilon={TRAINED_WITH_PERTURBATION}_iteration_1.pth"

        # model = MLPModel_likeMands(full_test_data.shape[1]).to(device)
        model = TransformerIDS().to(device)
        model.load_state_dict(torch.load(chosen_model_path, map_location=device))
        model.eval()

        sample = torch.tensor(full_test_data[i], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        label_tensor = torch.tensor([[full_test_labels[i]]], dtype=torch.float32).to(device)

        adv_sample = fgsm_attack_l2(model, sample, label_tensor, adversarial_epsilon, device)
        # adv_sample = bim_attack_l2(model,sample,label_tensor,epsilon=adversarial_epsilon,
        #                              alpha=adversarial_epsilon/5,num_iter=5,device=device)
        adv_examples[i] = adv_sample.squeeze(0).squeeze(0).cpu().numpy()

    # --- Pass AE through **full sequential model pipeline** (same as clean samples) ---
    remaining_adv_data = adv_examples
    remaining_adv_labels = full_test_labels.copy()
    test_indices = np.arange(len(remaining_adv_data))
    final_adv_predictions = np.zeros(len(remaining_adv_data))

    iteration = 0
    while True:
        iteration += 1
        if iteration == 1:
            LOW_THRESHOLD = 0.05
            HIGH_THRESHOLD = 0.95
        else:
            LOW_THRESHOLD = 0.48
            HIGH_THRESHOLD = 0.53
        model_path = f"model_trans_epsilon={TRAINED_WITH_PERTURBATION}_iteration_{iteration}.pth"
        if not os.path.exists(model_path):
            print(f"No more models found. Exiting after {iteration-1} iterations.")
            break
        if iteration ==1:
            model = TransformerIDS().to(device)
        else:
            model = MLPModel_likeMands(remaining_adv_data.shape[1]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Remaining Test Samples: {len(remaining_adv_data)}")
        current_test_dataset = CustomDataset(remaining_adv_data, remaining_adv_labels)
        test_mask = []
        predictions = []

        with torch.no_grad():
            for i, (features, label) in enumerate(current_test_dataset):
                features = features.to(device)
                features = features.unsqueeze(0).unsqueeze(0)
                proba = model(features).item()
                predictions.append(proba)
                if LOW_THRESHOLD <= proba <= HIGH_THRESHOLD:
                    test_mask.append(True)  # Uncertain, pass to next model
                else:
                    test_mask.append(False)  # Confident, assign prediction

            for i, include in enumerate(test_mask):
                if not include:  # Store confident predictions
                    final_adv_predictions[test_indices[i]] = predictions[i]

        # Update remaining AE data for next iteration
        test_mask = np.array(test_mask)
        if test_mask.size > 0:
            remaining_adv_data = remaining_adv_data[test_mask]
            remaining_adv_labels = remaining_adv_labels[test_mask]
            test_indices = test_indices[test_mask]

    # Compute AE accuracy for this run
    ae_accuracy = np.mean((final_adv_predictions > 0.5) == full_test_labels)
    ae_accuracies.append(ae_accuracy)
    print(f"Run {run+1}: Adversarial Accuracy = {ae_accuracy * 100:.2f}%")

# --- Compute Final Averaged AE Accuracy ---
mean_ae_accuracy = np.mean(ae_accuracies)
std_ae_accuracy = np.std(ae_accuracies)
print(f"\nFinal Adversarial Accuracy Over {N_RUNS} Runs: {mean_ae_accuracy * 100:.2f}% ± {std_ae_accuracy * 100:.2f}%")

# # --- Final AE ROC Curve ---
# fpr_adv, tpr_adv, _ = roc_curve(full_test_labels, final_adv_predictions)
# roc_auc_adv = auc(fpr_adv, tpr_adv)
# plt.figure()
# plt.plot(fpr_adv, tpr_adv, color='darkorange', lw=2, label=f'Adv ROC curve (area = {roc_auc_adv:.2f})')
# plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Adversarial ROC Curve')
# plt.legend(loc="lower right")
# plt.show()
# print(f"Adversarial AUC Score: {roc_auc_adv:.2f}")
# plot_histogram(final_adv_predictions, title='Adversarial Predictions Histogram')

# --- Final AE ROC Curve ---
fpr_adv, tpr_adv, _ = roc_curve(full_test_labels, final_adv_predictions)
roc_auc_adv = auc(fpr_adv, tpr_adv)
plt.figure()
plt.plot(fpr_adv, tpr_adv, color='darkorange', lw=2, label=f'Adv ROC curve (area = {roc_auc_adv:.2f})')
plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Adversarial ROC Curve')
plt.legend(loc="lower right")
plt.show()
print(f"Adversarial AUC Score: {roc_auc_adv:.2f}")
plot_histogram(final_adv_predictions, title='Adversarial Predictions Histogram')

# --- Compute and Print TPR at FPR 5% and 15% ---
fpr_5_idx = np.argmin(np.abs(fpr_adv - 0.05))
fpr_15_idx = np.argmin(np.abs(fpr_adv - 0.15))
tpr_5 = tpr_adv[fpr_5_idx]
tpr_15 = tpr_adv[fpr_15_idx]
print(f"TPR at FPR 5%: {tpr_5:.4f}")
print(f"TPR at FPR 15%: {tpr_15:.4f}")

# --- Compute and Print FPR and FNR at 0.5 Threshold ---
final_predictions_binary = (final_adv_predictions > 0.5).astype(int)
false_positives = np.sum((final_predictions_binary == 1) & (full_test_labels == 0))
false_negatives = np.sum((final_predictions_binary == 0) & (full_test_labels == 1))
true_positives = np.sum((final_predictions_binary == 1) & (full_test_labels == 1))
true_negatives = np.sum((final_predictions_binary == 0) & (full_test_labels == 0))

fpr_final = false_positives / (false_positives + true_negatives)
fnr_final = false_negatives / (false_negatives + true_positives)

print(f"Final FPR at threshold 0.5: {fpr_final:.4f}")
print(f"Final FNR at threshold 0.5: {fnr_final:.4f}")
