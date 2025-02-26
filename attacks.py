"""
File Name:attacks.py
Author: Riad Ibadulla
Date: 26-Feb-2025
Description: This file contains the attack function, which can be called outside. Functions accepts
modes, data, and epsilon to generate AEs.
"""

import torch

def fgsm_attack(model, data, label, epsilon, device):
    """
    Generates AE using FGSM
    Args:
        model (nn.Module): The trained model
        data (Tensor): data X only
        label (Tensor): Labels Y only
        epsilon (float): the perturbation level of attack

    Returns:
        Tensor: The generated AE
    """
    data.requires_grad = True
    model.eval()
    criterion = torch.nn.BCELoss()
    output = model(data)
    output = output.view(-1, 1)  # Ensure shape [1,1]
    loss = criterion(output, label)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data.detach()


def fgsm_attack_l2(model, data, label, epsilon, device):
    """
    Generates AE using FGSM L2 version
    Args:
        model (nn.Module): The trained model
        data (Tensor): data X only
        label (Tensor): Labels Y only
        epsilon (float): the perturbation level of attack

    Returns:
        Tensor: The generated AE
    """
    # Ensure the input tensor requires gradient
    data.requires_grad = True
    model.eval()
    criterion = torch.nn.BCELoss()
    output = model(data)
    output = output.view(-1, 1)  # Ensure shape [batch_size, 1]
    loss = criterion(output, label)

    # zero all existing gradients
    model.zero_grad()

    loss.backward()
    # get grads
    data_grad = data.grad.data

    # for batch inputs, compute the L2 norm per sample
    # reshape the gradient so each gradient is flattened
    grad_view = data_grad.view(data_grad.shape[0], -1)
    grad_norm = torch.norm(grad_view, p=2, dim=1, keepdim=True)  # shape: [batch_size, 1]

    # Reshape grad_norm to broadcast correctly over the data dimensions
    while len(grad_norm.shape) < len(data_grad.shape):
        grad_norm = grad_norm.unsqueeze(-1)

    # epsilon for division by zero avoding
    normalized_grad = data_grad / (grad_norm + 1e-10)

    # Create the perturbed data by adding the normalized gradient scaled by epsilon
    perturbed_data = data + epsilon * normalized_grad

    # Clamp the perturbed data to be within valid data range (e.g., [0,1])
    perturbed_data = torch.clamp(perturbed_data, 0, 1)

    return perturbed_data.detach()


def bim_attack_l2(model, data, label, epsilon, alpha, num_iter, device):
    """
    BIM attack with an L2 norm constraint.

    Args:
        model: The neural network model.
        data: The input data tensor.
        label: The true label tensor.
        epsilon: The maximum L2 perturbation allowed.
        alpha: The step size for each iteration.
        num_iter: Number of iterations.
        device: The device (CPU/GPU) for computations.

    Returns:
        perturbed_data: The adversarially perturbed data.
    """
    model.eval()

    # Clone the original data so that we can compute the perturbation relative to it
    original_data = data.clone().detach().to(device)
    perturbed_data = original_data.clone().detach()

    criterion = torch.nn.BCELoss()

    for i in range(num_iter):
        # Enable gradient computation on the current perturbed data
        perturbed_data.requires_grad = True

        # Forward pass
        output = model(perturbed_data)
        output = output.view(-1, 1)  # Ensure output shape is [batch_size, 1]
        loss = criterion(output, label)

        # Zero out gradients from previous iterations
        model.zero_grad()
        loss.backward()

        # Retrieve the gradient of the loss w.r.t. the input data
        data_grad = perturbed_data.grad.data

        # Normalize the gradient with L2 norm for each sample in the batch
        grad_view = data_grad.view(data_grad.shape[0], -1)
        grad_norm = torch.norm(grad_view, p=2, dim=1, keepdim=True)
        # Reshape grad_norm to enable broadcasting over the input dimensions
        while len(grad_norm.shape) < len(data_grad.shape):
            grad_norm = grad_norm.unsqueeze(-1)
        normalized_grad = data_grad / (grad_norm + 1e-10)

        # Take an update step in the direction of the normalized gradient
        perturbed_data = perturbed_data + alpha * normalized_grad

        # Compute the overall perturbation introduced so far
        perturbation = perturbed_data - original_data
        perturbation_view = perturbation.view(perturbation.shape[0], -1)
        perturb_norm = torch.norm(perturbation_view, p=2, dim=1, keepdim=True)
        # Reshape perturb_norm to enable broadcasting
        while len(perturb_norm.shape) < len(perturbation.shape):
            perturb_norm = perturb_norm.unsqueeze(-1)

        # Project the perturbation onto the L2 ball of radius epsilon for each sample
        # (If the norm is below epsilon, the factor will be 1)
        factor = torch.clamp(epsilon / (perturb_norm + 1e-10), max=1.0)
        perturbed_data = original_data + perturbation * factor

        # Clamp the perturbed data to be within valid data range (e.g., [0,1])
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        # Detach for the next iteration to avoid accumulating gradients
        perturbed_data = perturbed_data.detach()

    return perturbed_data