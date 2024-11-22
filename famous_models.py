import torch
from torchvision import models
from torch_operation_counter import OperationsCounterMode
import matplotlib.pyplot as plt
from tqdm import tqdm

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower()
    and not name.startswith("__")  # and "inception" in name
    and callable(models.__dict__[name])
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Lists to store data for plotting
model_list = []
param_list = []
flop_list = []

for name in tqdm(model_names):
    try:
        model = models.__dict__[name]().to(device)
        dsize = (1, 3, 224, 224)
        if "inception" in name:
            dsize = (1, 3, 299, 299)
        inputs = torch.randn(dsize).to(device)
        with OperationsCounterMode(model) as op_counter:
            model(inputs)
        total_ops = op_counter.total_operations
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Append data for plotting
        model_list.append(name)
        param_list.append(total_params)  # Convert to millions
        flop_list.append(total_ops)  # Convert to billions

    except Exception as e:
        pass  # Ignore models that fail to process

import csv
with open("famous_models.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Model Name", "Parameters", "FLOPs"])
    for row in zip(model_list, param_list, flop_list):
        writer.writerow(row)

# Plot FLOPs vs Parameters
plt.figure(figsize=(10, 6))
plt.scatter(param_list, flop_list, alpha=0.7)
for i, model_name in enumerate(model_list):
    plt.text(param_list[i], flop_list[i], model_name, fontsize=8)

plt.title("FLOPs vs Parameters for Torchvision Models")
plt.xlabel("Parameters")
plt.ylabel("FLOPs")
plt.grid(True)
plt.show()