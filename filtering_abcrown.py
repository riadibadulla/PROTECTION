import os
import sys
import importlib.util
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
import builtins


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
package_init_path = "/Users/riadibadulla/PycharmProjects/PROTECTION/complete_verifier/__init__.py"
if not os.path.exists(package_init_path):
    raise FileNotFoundError(f"Missing __init__.py at {package_init_path}")

abcrown_path = "/Users/riadibadulla/PycharmProjects/PROTECTION/complete_verifier/abcrown.py"
if not os.path.exists(abcrown_path):
    raise FileNotFoundError(f"Missing abcrown.py at {abcrown_path}")

package_name = "complete_verifier"

# Load and register the package.
package = load_local_module(package_name, package_init_path)
sys.modules[package_name] = package

# Load the abcrown module as part of the package.
abcrown = load_local_module(f"{package_name}.abcrown", abcrown_path)

# -------------------------------
# Load the 'loading' module to ensure Customized is defined.
# -------------------------------
loading_path = os.path.join(os.path.dirname(abcrown_path), "loading.py")
if not os.path.exists(loading_path):
    raise FileNotFoundError(f"Missing loading.py at {loading_path}")
loading = load_local_module(f"{package_name}.loading", loading_path)


# Define dummy_Customized so that it returns a callable optimizer.
def dummy_Customized(optimizer_name, default_optimizer):
    def optimizer(*args, **kwargs):
        print(f"Dummy optimizer called with {optimizer_name} and {default_optimizer}.")
        return None

    return optimizer


# Inject our callable dummy_Customized into the loading module if not present.
if not hasattr(loading, "Customized"):
    loading.Customized = dummy_Customized
# Also inject into builtins so that configuration evals (e.g. graph_optimizer) find it.
builtins.Customized = loading.Customized

# -------------------------------
# Force configuration to use CPU.
# -------------------------------
arguments = load_local_module(f"{package_name}.arguments", os.path.join(os.path.dirname(abcrown_path), "arguments.py"))
# Instead of "if 'general' not in arguments.Config", we use hasattr.
if not hasattr(arguments.Config, "general"):
    setattr(arguments.Config, "general", {})
arguments.Config.general["device"] = "cpu"
print("Device set to:", arguments.Config.general["device"])


# -------------------------------
# Define our filtering function using ABCROWN.incomplete_verifier
# -------------------------------
def add_data_with_abcrown(model, data_loader, low_thresh=0.25, high_thresh=0.65, perturbation=0.001):
    """
    For each sample in data_loader, build a simple specification (vnnlib)
    that enforces (for example) that the network output >= low_thresh.
    Then, call ABCROWN.incomplete_verifier to verify the sample.

    Returns:
      mask: a boolean numpy array (True for verified samples),
      perturbed_samples: numpy array of outputs (if available),
      perturbed_labels: corresponding labels.
    """
    # Instantiate the ABCROWN verifier with device forced to CPU.
    verifier = abcrown.ABCROWN(args=["--device=cpu"])

    mask = []
    perturbed_samples = []
    perturbed_labels = []

    model.eval()
    with torch.no_grad():
        for features, labels in tqdm(data_loader, desc="Filtering samples"):
            # Process each sample individually (assuming features is a batch tensor)
            for idx in range(features.size(0)):
                sample = features[idx:idx + 1]  # shape: (1, input_dim)
                label = labels[idx]

                # Define upper and lower bounds with the perturbation.
                data_ub = sample + perturbation
                data_lb = sample - perturbation

                # Build a dummy vnnlib specification.
                # Format: vnnlib = [(input_x, specs)]
                # Here, specs is a list with one clause: ([1], low_thresh)
                vnnlib = [(sample, [([1], low_thresh)])]

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

                if status == "safe-incomplete":
                    mask.append(True)
                    try:
                        from complete_verifier import arguments
                        perturbed_output = arguments.Globals.get("out", {}).get("pred", None)
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
    input_dim = 61
    # Create a dummy model.
    model = torch.nn.Linear(input_dim, 1)

    # Create 32 random samples.
    dummy_features = torch.rand(32, input_dim)
    dummy_labels = torch.randint(0, 2, (32,))
    dataset = TensorDataset(dummy_features, dummy_labels)
    data_loader = DataLoader(dataset, batch_size=4)

    mask, perturbed_samples, perturbed_labels = add_data_with_abcrown(
        model, data_loader, low_thresh=0.25, high_thresh=0.65, perturbation=0.001
    )

    print("Mask:", mask)
    print("Perturbed Samples:", perturbed_samples)
    print("Perturbed Labels:", perturbed_labels)