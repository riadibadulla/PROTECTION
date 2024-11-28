import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models import MLPModel_small,MLPModel2_large

# Load and preprocess data
data = pd.read_csv('merged_shuffled_dataset.csv')
data = data.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Attack Name'])

# Define features and labels
X = data.drop(columns=['Label'])
y = data['Label'].values

# Encode categorical variable
protocol_encoder = OneHotEncoder()
protocol_encoded = protocol_encoder.fit_transform(X[['Protocol']]).toarray()
X_numeric = X.drop(columns=['Protocol']).values
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Combine preprocessed features
X_processed = np.hstack([X_numeric_scaled, protocol_encoded])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Custom PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]




# Prepare datasets
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Train the first model
model1 = MLPModel_small(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model1.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model1(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the first model
model1.eval()
y_test_proba = []
y_test_labels = []
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model1(features).squeeze()
        y_test_proba.extend(outputs.numpy())
        y_test_labels.extend(labels.numpy())

accuracy1 = np.mean(((np.array(y_test_proba) > 0.5) == y_test_labels))
print(f"First Model Accuracy: {accuracy1 * 100:.2f}%")

# Plot histogram of predicted probabilities
plt.figure(figsize=(10, 6))
plt.hist(y_test_proba, bins=[i * 0.05 for i in range(21)], edgecolor='black', alpha=0.7)
plt.xlabel('Predicted Probability of Class 1 (Attack)')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Probabilities Model1')
plt.xticks([i * 0.05 for i in range(21)])
plt.show()

# Filter the data for the second model
# Generate probabilities for the training set
y_train_proba = []
model1.eval()
with torch.no_grad():
    for features, _ in train_loader:
        outputs = model1(features).squeeze()
        y_train_proba.extend(outputs.numpy())

# Convert to numpy array for filtering
y_train_proba = np.array(y_train_proba)

# Create the mask for filtering training data
test_mask = (np.array(y_test_proba) >= 0.25) & (np.array(y_test_proba) <= 0.65)
train_mask = (np.array(y_train_proba) >= 0.25) & (np.array(y_train_proba) <= 0.65)

X_filtered_train = X_train[train_mask]
y_filtered_train = y_train[train_mask]

# Prepare filtered dataset for the second model
filtered_train_dataset = CustomDataset(X_filtered_train, y_filtered_train)
filtered_train_loader = DataLoader(filtered_train_dataset, batch_size=32, shuffle=True)
filtered_test_dataset = CustomDataset(X_test[test_mask], y_test[test_mask])
filtered_test_loader = DataLoader(filtered_test_dataset, batch_size=32)

model2 = MLPModel_small(X_train.shape[1])
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
for epoch in range(20):
    model2.train()
    for features, labels in filtered_train_loader:
        optimizer2.zero_grad()
        outputs = model2(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer2.step()


# Evaluate on filtered test data
filtered_accuracy = 0
total = 0
with torch.no_grad():
    for features, labels in filtered_test_loader:
        outputs = model2(features).squeeze()
        predictions = (outputs > 0.5).numpy()
        filtered_accuracy += (predictions == labels.numpy()).sum()
        total += labels.size(0)

filtered_accuracy = filtered_accuracy / total
print(f"Second Model Accuracy on Filtered Test Data: {filtered_accuracy * 100:.2f}%")

# Evaluate combined predictions for the test set
final_predictions = []
model1.eval()
model2.eval()

with torch.no_grad():
    for i, (features, _) in enumerate(test_loader.dataset):
        # Get the probability prediction from model1
        proba_model1 = model1(features.unsqueeze(0)).item()

        if 0.25 <= proba_model1 <= 0.65:
            # If within range, use model2
            proba_model2 = model2(features.unsqueeze(0)).item()
            final_predictions.append(proba_model2)
        else:
            # Otherwise, retain model1's prediction
            final_predictions.append(proba_model1)

# Convert final_predictions to numpy for histogram
final_predictions = np.array(final_predictions)

# Plot histogram of combined predicted probabilities
plt.figure(figsize=(10, 6))
plt.hist(final_predictions, bins=[i * 0.05 for i in range(21)], edgecolor='black', alpha=0.7)
plt.xlabel('Predicted Probability of Class 1 (Attack)')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Probabilities (Combined Models)')
plt.xticks([i * 0.05 for i in range(21)])
plt.show()