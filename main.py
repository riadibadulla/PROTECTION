from dataset import load_and_preprocess_data, CustomDataset
from models import MLPModel_small, MLPModel2_large, Conv1DModel, PureCNN, smallCNN, MLPModel_simple,  MLPModel_thin
from train import train_model, filter_data_by_model
from SMT import filter_data_by_model_with_marabou
from evaluate import evaluate_model, combine_predictions, plot_histogram
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
# Load and preprocess the data
import torch

# Determine the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device = torch.device("cpu") # overwritten device

print(f"Using device: {device}")

NUMBER_OF_EPOCHS = 20
LR = 0.001

X_train, X_test, y_train, y_test = load_and_preprocess_data('Datasets/merged_shuffled_dataset.csv')

# Create datasets and data loaders
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Train the first model
print(f'Input features shape after the encoding is equal to {X_train.shape[1]}')
# model1 = MLPModel2_large(X_train.shape[1])
# model1 = Conv1DModel(1,X_train.shape[1])
model1 = MLPModel_simple(X_train.shape[1]).to(device)
criterion = nn.BCELoss()
train_model(model1, train_loader, criterion, lr=LR, epochs=NUMBER_OF_EPOCHS)

# Evaluate the first model
accuracy1, y_test_proba = evaluate_model(model1, test_loader)
print(f"First Model Accuracy: {accuracy1 * 100:.2f}%")
plot_histogram(y_test_proba, "Histogram of Model 1 Predictions")

# Filter data for the second model
train_mask = filter_data_by_model_with_marabou(model1, train_loader, low_thresh=0.48, high_thresh=0.52)
test_mask = filter_data_by_model_with_marabou(model1, test_loader, low_thresh=0.48, high_thresh=0.52)

X_filtered_train = X_train[train_mask]
y_filtered_train = y_train[train_mask]

filtered_train_dataset = CustomDataset(X_filtered_train, y_filtered_train)
filtered_train_loader = DataLoader(filtered_train_dataset, batch_size=32, shuffle=True)

# Train the second model
# model2 = MLPModel_small(X_train.shape[1])
model2 = MLPModel_simple(X_train.shape[1]).to(device)
train_model(model2, filtered_train_loader, criterion, lr=LR, epochs=NUMBER_OF_EPOCHS)

# Combine predictions and evaluate
final_predictions = combine_predictions(model1, model2, test_loader.dataset)
plot_histogram(final_predictions, "Combined Predictions from Both Models")

# Evaluate combined predictions
final_labels = [label.numpy() for _, label in test_loader.dataset]
final_labels = np.array(final_labels)

combined_accuracy = np.mean((final_predictions > 0.5) == final_labels)
print(f"Combined Model Accuracy: {combined_accuracy * 100:.2f}%")