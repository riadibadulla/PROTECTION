import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bars

# Determine the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device = torch.device("cpu")  # Overwritten device


def train_model(model, train_loader, criterion, lr=0.001, epochs=10, T_max=5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for features, labels in progress_bar:
            optimizer.zero_grad()
            features = features.unsqueeze(1).to(device)  # Add channel dimension
            labels = labels.to(device).float().view(-1, 1)  # Ensure shape (batch_size, 1)

            outputs = model(features)
            outputs = outputs.view(-1, 1)  # Ensure output shape matches labels
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Update the learning rate
        scheduler.step()

        # Print the current learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(train_loader):.4f}, Learning rate: {current_lr:.6f}")



def penalty_function(p, lower=0.05, upper=0.95):
    """
    Computes a penalty that is zero outside the [lower, upper] region and
    grows when p is inside that region.

    For a piecewise-quadratic-like penalty:
        P(p) = (p - lower) * (upper - p) if p in [lower, upper], else 0
    """
    # clamp(...) ensures values do not go below 0
    left = torch.clamp(p - lower, min=0.0)
    right = torch.clamp(upper - p, min=0.0)
    return left * right  # elementwise multiplication


def train_model_with_penalty(model, train_loader, criterion, lr=0.001, epochs=10, T_max=5, penalty_lambda=0.1, lower=0.4, upper=0.6):
    """
    Trains the given model using the provided DataLoader, criterion, and hyperparameters.
    Adds a penalty term to the loss function to discourage outputs in the abstention region.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for features, labels in progress_bar:
            optimizer.zero_grad()

            # features shape: (batch_size, num_features)
            # for a 1D MLP, we don't necessarily need an extra channel dimension,
            # but if your model expects it, keep .unsqueeze(1). Otherwise remove it:
            features = features.unsqueeze(1).to(device)

            labels = labels.to(device).float().view(-1, 1)  # shape: (batch_size, 1)

            outputs = model(features)
            outputs = outputs.view(-1, 1)
            # outputs shape: (batch_size, 1) after model forward pass

            # 1) Base loss (e.g., BCE)
            base_loss = criterion(outputs, labels)

            # 2) Penalty for p in [0.05, 0.95]
            #    outputs is the sigmoid output => p in [0,1]
            #    penalty_function returns a tensor of shape (batch_size, 1)
            penalty = penalty_function(outputs, lower, upper)

            # Mean penalty over the batch
            penalty_mean = penalty.mean()

            # Combine the losses
            loss = base_loss + penalty_lambda * penalty_mean

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Update the learning rate
        scheduler.step()

        # Print the current learning rate and average loss
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1} completed. "
              f"Average loss: {epoch_loss / len(train_loader):.4f}, "
              f"Learning rate: {current_lr:.6f}")