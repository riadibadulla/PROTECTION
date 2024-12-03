import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bars


def train_model(model, train_loader, criterion, lr=0.001, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for features, labels in progress_bar:
            optimizer.zero_grad()
            features = features.unsqueeze(1).to(torch.device("mps"))
            # features = features.unsqueeze(1)
            labels = labels.to(torch.device("mps"))
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(train_loader):.4f}")


def filter_data_by_model(model, data_loader, low_thresh=0.25, high_thresh=0.65):
    probabilities = []
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Filtering data", leave=False)
        for features, _ in progress_bar:
            outputs = model(features.unsqueeze(1).to(torch.device("mps"))).squeeze()
            probabilities.extend(outputs.to(torch.device("cpu")).numpy())

    probabilities = np.array(probabilities)
    mask = (probabilities >= low_thresh) & (probabilities <= high_thresh)
    return mask

# import torch
# import torch.optim as optim
# import numpy as np
#
# def train_model(model, train_loader, criterion, lr=0.001, epochs=10):
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     for epoch in range(epochs):
#         model.train()
#         for features, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(features).squeeze()
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
# def filter_data_by_model(model, data_loader, low_thresh=0.25, high_thresh=0.65):
#     probabilities = []
#     model.eval()
#     with torch.no_grad():
#         for features, _ in data_loader:
#             outputs = model(features).squeeze()
#             probabilities.extend(outputs.numpy())
#
#     probabilities = np.array(probabilities)
#     mask = (probabilities >= low_thresh) & (probabilities <= high_thresh)
#     return mask
