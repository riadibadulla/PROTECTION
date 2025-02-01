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
            print(f"Output shape: {outputs.shape}, Label shape: {labels.shape}")
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