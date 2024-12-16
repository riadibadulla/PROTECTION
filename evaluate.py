import numpy as np
import matplotlib.pyplot as plt
import torch

# Determine the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device = torch.device("cpu") # overwritten device

def evaluate_model(model, data_loader):
    predictions, labels = [], []
    model.eval()
    with torch.no_grad():
        for features, label in data_loader:
            outputs = model(features.unsqueeze(1).to(device)).squeeze()
            predictions.extend(outputs.to(torch.device("cpu")).numpy())
            labels.extend(label.numpy())

    accuracy = np.mean(((np.array(predictions) > 0.5) == labels))
    return accuracy, predictions

def combine_predictions(model1, model2, test_dataset, low_thresh=0.25, high_thresh=0.65):
    final_predictions = []
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for features, _ in test_dataset:
            proba_model1 = model1(features.unsqueeze(0).unsqueeze(0).to(device)).item()
            if low_thresh <= proba_model1 <= high_thresh:
                proba_model2 = model2(features.unsqueeze(0).unsqueeze(0).to(device)).item()
                final_predictions.append(proba_model2)
            else:
                final_predictions.append(proba_model1)

    return np.array(final_predictions)

def plot_histogram(predictions, title):
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=[i * 0.05 for i in range(21)], edgecolor='black', alpha=0.7)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()
