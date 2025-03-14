import os
import sys
import importlib.util
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
import builtins
from complete_verifier import arguments  # For retrieving the saved output probability

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



def add_data_with_abcrown(model, data_loader, low_thresh=0.25, high_thresh=0.65, perturbation=0.01):
    """
    For each sample in data_loader, build a vnnlib specification that enforces that under
    an input perturbation of ±1% of its features, the network output remains outside the
    abstention range [low_thresh, high_thresh]. In other words, we want to verify that:

         output < low_thresh  OR  output > high_thresh

    Using ABCROWN's incomplete verifier, we use the returned status to determine the result:
      - If the status indicates unsafety (e.g. starts with "unsafe"), then the property is
        not satisfied (i.e. there exists a perturbation causing the output to fall within
        [low_thresh, high_thresh]). In that case, we set the mask to True and return a counterexample.
      - Otherwise (e.g. status starts with "safe" or is unknown), we set the mask to False.

    Returns:
      mask: a boolean numpy array, where True indicates the property is violated (counterexample found),
      perturbed_samples: a numpy array of counterexample outputs (if available),
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
            # Process each sample in the batch individually.
            for idx in range(features.size(0)):
                sample = features[idx:idx + 1]  # shape: (1, input_dim)
                label = labels[idx]

                # Create perturbation bounds ±1% (default perturbation value)
                data_ub = sample + perturbation
                data_lb = sample - perturbation

                # Build a vnnlib specification for the property:
                #    output < low_thresh OR output > high_thresh
                # We express "output < low_thresh" as -output ≥ -low_thresh.
                vnnlib = [(sample, [([-1], -low_thresh), ([1], high_thresh)])]

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

                # Use the status to decide:
                # If the status indicates unsafety (property violated), then the output
                # falls within [low_thresh, high_thresh] for some perturbation.
                if status.startswith("safe"):
                    mask.append(True)
                    # Try to extract a counterexample.
                    counterexample = None
                    if "attack_images" in ret and ret["attack_images"] is not None:
                        ce = ret["attack_images"]
                        if isinstance(ce, torch.Tensor):
                            counterexample = ce.cpu().numpy()
                        else:
                            counterexample = ce
                    else:
                        # Fallback: try to get the output from the global arguments.
                        try:
                            from complete_verifier import arguments
                            perturbed_output = arguments.Globals.get("out", {}).get("pred", None)
                            if perturbed_output is not None:
                                counterexample = perturbed_output.numpy().item()
                        except Exception as e:
                            print(f"Could not retrieve perturbed output for sample {idx}: {e}")
                            counterexample = None
                    perturbed_samples.append(counterexample)
                else:
                    # If status indicates safe (property holds) or is unknown, we set mask to False.
                    mask.append(False)
                    # Optionally, store the verified output.
                    try:
                        from complete_verifier import arguments
                        perturbed_output = arguments.Globals.get("out", {}).get("pred", None)
                        if perturbed_output is not None:
                            counterexample = perturbed_output.numpy().item()
                        else:
                            counterexample = None
                    except Exception as e:
                        counterexample = None
                    perturbed_samples.append(counterexample)

                perturbed_labels.append(label.item() if hasattr(label, "item") else label)

    return np.array(mask), np.array(perturbed_samples), np.array(perturbed_labels)

#
# def add_data_with_abcrown(model, data_loader, low_thresh=0.2, high_thresh=0.8, perturbation=0.01):
#     """
#     Uses the complete verifier exclusively to check if a sample is unsafe.
#
#     The safe property is: for all allowed perturbations,
#         output ≤ low_thresh  OR  output ≥ high_thresh.
#     Thus, if there is any perturbation producing an output in (low_thresh, high_thresh),
#     the sample is unsafe.
#
#     For each sample in the data loader, this function:
#       1. Constructs perturbation bounds (data_lb, data_ub).
#       2. Builds a vnnlib specification for the safe property.
#       3. Calls the complete verifier.
#       4. Extracts the output probability (the network's prediction under perturbation)
#          from a global variable.
#       5. If the output falls in [low_thresh, high_thresh] (or the verifier’s status indicates unsafety),
#          marks the sample as unsafe.
#
#     Returns:
#       mask: Boolean numpy array with one entry per sample (True = unsafe).
#       unsafe_outputs: Array of perturbed output probabilities (counterexamples) for unsafe samples.
#       unsafe_labels: Array of labels for unsafe samples.
#     """
#     verifier = abcrown.ABCROWN(args=["--device=cpu"])
#
#     mask = []  # One Boolean per sample.
#     unsafe_outputs = []  # To store the output probability (counterexample) for unsafe samples.
#     unsafe_labels = []  # To store the label for unsafe samples.
#
#     model.eval()
#     with torch.no_grad():
#         for features, labels in tqdm(data_loader, desc="Filtering samples (complete verifier)"):
#             for idx in range(features.size(0)):
#                 sample = features[idx:idx + 1]  # Shape: (1, input_dim)
#                 label = labels[idx]
#
#                 # Define the allowed perturbation bounds (± perturbation)
#                 data_ub = sample + perturbation
#                 data_lb = sample - perturbation
#
#                 # Build vnnlib specification for the safe property:
#                 # Safe if: output ≤ low_thresh OR output ≥ high_thresh.
#                 # That is, if any perturbation makes output fall into [low_thresh, high_thresh],
#                 # then the sample is unsafe.
#                 vnnlib = [(sample, [([-1], -low_thresh), ([1], high_thresh)])]
#
#                 try:
#                     complete_status = verifier.complete_verifier(
#                         model_ori=model,
#                         model_incomplete=None,  # We are not using an incomplete model here.
#                         vnnlib=vnnlib,
#                         batched_vnnlib=[vnnlib[0]],  # Batched version of the specification.
#                         vnnlib_shape=sample.shape[1:],  # The shape of a single sample's input.
#                         index=idx,
#                         timeout_threshold=10.0,  # Set an appropriate timeout.
#                         results={}  # Pass an empty dictionary to satisfy parameter requirements.
#                     )
#                 except Exception as e:
#                     print(f"Complete verifier failed for sample {idx}: {e}")
#                     complete_status = "unknown"
#
#                 # Extract the output probability from the global variable.
#                 try:
#                     perturbed_output = arguments.Globals.get("out", {}).get("pred", None)
#                     if perturbed_output is not None:
#                         prob_val = perturbed_output.numpy().item()
#                     else:
#                         prob_val = None
#                 except Exception as e:
#                     print(f"Error retrieving output for sample {idx}: {e}")
#                     prob_val = None
#
#                 # Final decision: mark as unsafe if either the complete verifier status
#                 # indicates unsafety (e.g. "unsafe-bab" or "unsafe-pgd")
#                 # OR if the extracted output probability falls within [low_thresh, high_thresh].
#                 if (prob_val is not None and (low_thresh <= prob_val <= high_thresh)) or \
#                         (isinstance(complete_status, str) and complete_status.startswith("unsafe")):
#                     mask.append(True)
#                     unsafe_outputs.append(prob_val)
#                     unsafe_labels.append(label.item() if hasattr(label, "item") else label)
#                 else:
#                     mask.append(False)
#
#     return np.array(mask), np.array(unsafe_outputs), np.array(unsafe_labels)
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