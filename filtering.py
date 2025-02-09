import torch
import torch.nn as nn
import numpy as np
from maraboupy import Marabou,MarabouCore
from tqdm import tqdm
import multiprocessing

# Determine the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
device = torch.device("cpu") # overwritten device
num_cores = multiprocessing.cpu_count()
torch.set_num_threads(num_cores)

def filter_data_by_model_with_marabou(model, data_loader, low_thresh=0.25, high_thresh=0.65, perturbation=0.001):
    options = Marabou.createOptions(numWorkers=num_cores,verbosity=0)
    # onnx save
    # dummy_input = next(iter(data_loader))[0][0].unsqueeze(0)  # extract one sample to infer shape
    try:
        dummy_input = next(iter(data_loader))[0][0].unsqueeze(0)  # Extract one sample to infer shape
    except StopIteration:
        print("Warning: Data loader is empty. Using default dummy input.")
        # When there are two empty iterations for test
        dummy_input = torch.zeros(1, 61)  # Default tensor of shape [1, 61]

    onnx_filename = f"model_perturbations={perturbation}.onnx"

    torch.onnx.export(model, dummy_input, onnx_filename, input_names=["input"], output_names=["output"])

    # onnx load
    network = Marabou.read_onnx(onnx_filename)
    input_vars = network.inputVars[0][0]  # input variables
    output_var = network.outputVars[0][0][0]  # single output variable

    # mask of filtered data
    mask = []
    model.eval()

    # iterate through data loader
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Filtering data with perturbation", leave=False)
        for features, _ in progress_bar:
            for data_point in features:
                # Perturb inputs in Marabou
                data_point = data_point.numpy()
                for i, var in enumerate(input_vars):
                    network.setLowerBound(var, max(0.0, data_point[i] - perturbation))
                    network.setUpperBound(var, min(1.0, data_point[i] + perturbation))
                # output bounds
                network.setLowerBound(output_var, low_thresh)
                network.setUpperBound(output_var, high_thresh)

                # Solve smt
                solve_result = network.solve(options=options)
                mask.append(bool(solve_result[0]=="sat"))  # True if valid

    return np.array(mask)

def filter_data_delegate(model, data_loader, low_thresh=0.25, high_thresh=0.65):
    probabilities = []
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Filtering data", leave=False)
        for features, _ in progress_bar:
            outputs = model(features.unsqueeze(1).to(device)).squeeze()

            # have to flatten as now the shape is [batch,1]
            probabilities.extend(outputs.to(torch.device("cpu")).numpy().flatten())

    probabilities = np.array(probabilities)
    mask = (probabilities >= low_thresh) & (probabilities <= high_thresh)
    return mask