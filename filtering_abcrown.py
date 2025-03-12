import os
import sys
import importlib.util
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import copy
import time


# -------------------------------
# Dynamic module loader
# -------------------------------
def load_local_module(module_name, module_file_path):
    """
    Dynamically load a module given its fully qualified name and file path.
    """
    spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -------------------------------
# Set absolute paths for the complete_verifier package.
# -------------------------------
# Here we assume that complete_verifier is located at:
# /Users/riadibadulla/PycharmProjects/PROTECTION/complete_verifier/
package_init_path = "/Users/riadibadulla/PycharmProjects/PROTECTION/complete_verifier/__init__.py"
if not os.path.exists(package_init_path):
    raise FileNotFoundError(f"Missing __init__.py at {package_init_path}")

abcrown_path = "/Users/riadibadulla/PycharmProjects/PROTECTION/complete_verifier/abcrown.py"
if not os.path.exists(abcrown_path):
    raise FileNotFoundError(f"Missing abcrown.py at {abcrown_path}")

# Use a safe package name (avoid hyphens)
package_name = "complete_verifier"

# Load and register the package.
package = load_local_module(package_name, package_init_path)
sys.modules[package_name] = package

# Load the abcrown module as part of the package.
abcrown = load_local_module(f"{package_name}.abcrown", abcrown_path)


# -------------------------------
# Define our filtering function using ABCROWN.incomplete_verifier
# -------------------------------
def add_data_with_abcrown(model, data_loader, low_thresh=0.25, high_thresh=0.65, perturbation=0.001):
    """
    For each sample in data_loader, build a simple specification (vnnlib)
    that enforces (for example) the network output >= low_thresh.
    Then, call the ABCROWN.incomplete_verifier method to verify that sample.

    This function mimics your original filter:
      - It extracts each sample,
      - Constructs data_ub and data_lb by adding/subtracting a small perturbation,
      - Constructs a dummy vnnlib: a list with one tuple: (input_x, specs)
        where input_x is the sample (a tensor) and specs is a list containing a
        single clause. Here we set the clause as ([1], low_thresh) meaning:
        1 * output >= low_thresh.
      - It calls verifier.incomplete_verifier(model, data, data_ub, data_lb, vnnlib)
      - If the returned status is 'safe-incomplete', we consider that sample verified ("sat")
        and attempt to retrieve the computed output from arguments.Globals['out']['pred'].

    Returns:
      mask: a boolean numpy array (True for verified samples),
      perturbed_samples: a numpy array of outputs (if available),
      perturbed_labels: the corresponding labels from data_loader.
    """
    # Instantiate the ABCROWN verifier.
    # (Pass an empty list for args; in practice, you might need to pass proper arguments.)
    verifier = abcrown.ABCROWN(args=["--device=cpu"])

    mask = []
    perturbed_samples = []
    perturbed_labels = []

    # Loop over the data loader (assumed to yield (features, labels))
    model.eval()
    with torch.no_grad():
        for features, labels in tqdm(data_loader, desc="Filtering samples"):
            # Process each sample individually (assuming features is a batch tensor)
            for idx in range(features.size(0)):
                sample = features[idx:idx + 1]  # shape: (1, input_dim)
                label = labels[idx]

                # Define upper and lower bounds with the perturbation
                data_ub = sample + perturbation
                data_lb = sample - perturbation

                # Build a dummy vnnlib specification.
                # The specification format expected by incomplete_verifier is:
                #    vnnlib = [(input_x, specs)]
                # Here, input_x is our sample.
                # We set specs to be a single clause: ([1], low_thresh)
                # meaning that we require: 1 * (network output) >= low_thresh.
                # (In a real scenario, you might have multiple clauses and more complex c.)
                vnnlib = [(sample, [([1], low_thresh)])]

                # Call the incomplete verifier.
                try:
                    status, ret, _ = verifier.incomplete_verifier(
                        model_ori=model,
                        data=sample,
                        data_ub=data_ub,
                        data_lb=data_lb,
                        vnnlib=vnnlib,
                        interm_bounds=None
                    )
                except Exception as e:
                    print(f"Verification failed for sample {idx}: {e}")
                    status = "unknown"
                    ret = {}

                # If the verifier returns 'safe-incomplete', we consider the sample verified.
                if status == "safe-incomplete":
                    mask.append(True)
                    # If the configuration is set to save output, it may be stored in arguments.Globals.
                    # We attempt to retrieve it here.
                    try:
                        from complete_verifier import arguments
                        perturbed_output = arguments.Globals.get('out', {}).get('pred', None)
                        if perturbed_output is not None:
                            perturbed_samples.append(perturbed_output.numpy())
                        else:
                            perturbed_samples.append(None)
                    except Exception as e:
                        print(f"Could not retrieve perturbed output for sample {idx}: {e}")
                        perturbed_samples.append(None)
                    perturbed_labels.append(label.item() if hasattr(label, "item") else label)
                else:
                    mask.append(False)
                    perturbed_samples.append(None)
                    perturbed_labels.append(label.item() if hasattr(label, "item") else label)

    return np.array(mask), np.array(perturbed_samples), np.array(perturbed_labels)


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Create a dummy model and dummy data.
    input_dim = 61
    model = torch.nn.Linear(input_dim, 1)

    # Create 10 random samples.
    dummy_features = torch.rand(10, input_dim)
    dummy_labels = torch.randint(0, 2, (10,))
    dataset = TensorDataset(dummy_features, dummy_labels)
    data_loader = DataLoader(dataset, batch_size=2)

    mask, perturbed_samples, perturbed_labels = add_data_with_abcrown(
        model, data_loader, low_thresh=0.25, high_thresh=0.65, perturbation=0.001
    )

    print("Mask:", mask)
    print("Perturbed Samples:", perturbed_samples)
    print("Perturbed Labels:", perturbed_labels)