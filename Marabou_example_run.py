import torch
import torch.nn as nn
import numpy as np
from maraboupy import Marabou


# Step 1: Define a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 1)  # 2 inputs, 1 output

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


# Instantiate and save the model
model = SimpleModel()
model.fc.weight.data = torch.tensor([[1.0, -1.0]])  # Set custom weights
model.fc.bias.data = torch.tensor([0.0])  # Set custom bias

# Export to ONNX format for Marabou
dummy_input = torch.tensor([[0.0, 0.0]])  # Example input shape
torch.onnx.export(model, dummy_input, "simple_model.onnx", input_names=["input"], output_names=["output"])

# Step 2: Load the ONNX model into Marabou
network = Marabou.read_onnx("simple_model.onnx")

# Step 3: Create a dummy dataset
dummy_data = np.random.rand(100, 2)  # 100 data points, each with 2 features
valid_points = []  # Store data points that meet the condition

# Iterate through dummy dataset
for data_point in dummy_data:
    # Flatten inputVars to handle nested structure
    input_vars = network.inputVars[0]  # Access the first input layer's variable indices
    for i, var in enumerate(input_vars[0]):  # Extract individual variable indices (int)
        network.setLowerBound(int(var), max(0.0, data_point[i] - 0.1))  # Small perturbation
        network.setUpperBound(int(var), min(1.0, data_point[i] + 0.1))

    # Set output bounds for SMT testing
    output_var = network.outputVars[0][0]  # Single output variable index
    network.setLowerBound(int(output_var), 0.45)
    network.setUpperBound(int(output_var), 0.65)

    # Solve the query
    solve_result = network.solve()
    if solve_result[0]:  # Check if a solution exists
        valid_points.append(data_point)  # Save the valid data point

# Step 4: Save valid data points
valid_points = np.array(valid_points)
np.savetxt("valid_points.csv", valid_points, delimiter=",")
print(f"Saved {len(valid_points)} valid data points to 'valid_points.csv'.")