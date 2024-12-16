from fileinput import filename

import torch
import torch.nn as nn
import numpy as np
from maraboupy import Marabou
from tqdm import tqdm

options = Marabou.createOptions(verbosity=0, numWorkers=6)

def filter_data_by_model_with_marabou(model, data_loader, low_thresh=0.25, high_thresh=0.65, perturbation=0.001):
    # onnx save
    dummy_input = next(iter(data_loader))[0][0].unsqueeze(0)  # extract one sample to infer shape
    torch.onnx.export(model, dummy_input, "model.onnx", input_names=["input"], output_names=["output"])

    # onnx load
    network = Marabou.read_onnx("model.onnx")
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
                    if i==3:
                        network.setLowerBound(var, max(0.0, data_point[i] - perturbation))
                        network.setUpperBound(var, min(1.0, data_point[i] + perturbation))
                    else:
                        network.setLowerBound(var, data_point[i])
                        network.setUpperBound(var, data_point[i])
                    network.setLowerBound(var, data_point[i] - perturbation)
                    network.setUpperBound(var, data_point[i] + perturbation)

                # output bounds
                network.setLowerBound(output_var, low_thresh)
                network.setUpperBound(output_var, high_thresh)

                # Solve smt
                solve_result = network.solve(options=options)
                mask.append(bool(solve_result[0]=="sat"))  # True if valid

    return np.array(mask)