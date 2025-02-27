from dataset import load_and_preprocess_data, CustomDataset
from models import MLPModel_likeMands, TransformerIDS
from train import train_model
from filtering import filter_data_by_model_with_marabou, filter_data_delegate
from evaluate import plot_histogram
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch

# Set seeds for reproducibility
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
NUMBER_OF_EPOCHS = 1
LR = 0.01
LOW_THRESHOLD = 0.49
HIGH_THRESHOLD = 0.51
PERTURBATION = 0.01

# Load and preprocess the data
X_train, X_test, y_train, y_test = load_and_preprocess_data('Datasets/merged_shuffled_dataset.csv')

# Initial datasets
remaining_train_data = X_train
remaining_train_labels = y_train

model_list = []
iteration = 0

# Initialize for ROC-AUC calculation
all_predictions = []  # Store all probabilities
all_true_labels = []  # Store all true labels

while len(remaining_train_data) > 10:
    iteration += 1
    print(f"\n### Iteration {iteration} ###")
    print(f"Remaining Training Samples: {len(remaining_train_data)}")

    # Choose model and filtering strategy based on iteration
    if iteration == 1:
        # On first iteration, use TransformerIDS and do not use SMT filtering
        model = TransformerIDS().to(device)
        current_using_smt = False
        LR = 0.001
    else:
        LR = 0.01
        # On subsequent iterations, use MLPModel_likeMands and SMT filtering
        model = MLPModel_likeMands(remaining_train_data.shape[1]).to(device)
        current_using_smt = True

    criterion = nn.BCELoss()

    # Prepare data loaders for the current filtered data
    current_train_dataset = CustomDataset(remaining_train_data, remaining_train_labels)
    current_train_loader = DataLoader(current_train_dataset, batch_size=32, shuffle=True)

    # Train the model
    train_model(model, current_train_loader, criterion, lr=LR, epochs=NUMBER_OF_EPOCHS)
    model_list.append(model)

    # Evaluate the model: Get predictions for the current training set
    model.eval()
    train_predictions = []
    with torch.no_grad():
        for features, _ in current_train_loader:
            features = features.to(device)
            proba = model(features).squeeze().cpu().numpy()
            # Flatten predictions if needed
            train_predictions.extend(proba.flatten())

    # Plot histograms for current iteration
    plot_histogram(train_predictions, f"Iteration {iteration}: Training Set Predictions, Per={PERTURBATION}",
                   iteration=iteration, perturbation=PERTURBATION)

    # Filter training data based on the chosen strategy
    if current_using_smt:
        train_mask = filter_data_by_model_with_marabou(model, current_train_loader,
                                                       low_thresh=LOW_THRESHOLD,
                                                       high_thresh=HIGH_THRESHOLD,
                                                       perturbation=PERTURBATION)
    else:
        train_mask = filter_data_delegate(model, current_train_loader,
                                          low_thresh=LOW_THRESHOLD,
                                          high_thresh=HIGH_THRESHOLD)

    # Update remaining training data based on the mask
    remaining_train_data = remaining_train_data[train_mask]
    remaining_train_labels = remaining_train_labels[train_mask]

    if len(remaining_train_data) == 0:
        print("No more ambiguous samples left for training. Exiting the loop.")

    # Save the model state
    torch.save(model.state_dict(), f"model_trans_epsilon={PERTURBATION}_iteration_{iteration}.pth")