import torch
import torch.nn as nn
import torch.optim as optim
import datasets
from data.uci_data import get_data
import models
import models.generic
import models.feedforward
import models.layers
import utils
from torch.utils.data import TensorDataset, DataLoader

# X, y = datasets.generate_concentric_circles(factor=1, n_classes=2, noise=0.3)
# X = torch.Tensor(X)
# y = torch.Tensor(y).unsqueeze(1)
X, y = datasets.generate_concentric_circles(n_classes=2)
#X = X[:,1:].astype('float')
X = torch.Tensor(X) + 15
y = torch.Tensor(y)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

# model = models.feedforward.FeedForwardNetwork(
#     input_dim=X.shape[1] if len(X.shape) > 1 else 1, 
#     output_dim=y.shape[1] if len(y.shape) > 1 else 1,
#     layer_spec=[100, 30, 256, 30]
# )
model = models.generic.GenericNetwork()
model.hidden_layers.extend([
    models.layers.LinearTransformLayer(X.shape[1] if len(X.shape) > 1 else 1),
    nn.Linear(2, 5),
    models.layers.CompositeActivation([models.layers.AbsActivation(), nn.ReLU()]),
    nn.Linear(5, 10),
    models.layers.CompositeActivation([models.layers.AbsActivation(), nn.ReLU()]),
    nn.Linear(10, 5),
    models.layers.CompositeActivation([models.layers.AbsActivation(), nn.ReLU()]),
    nn.Linear(5, 2),
    nn.Sigmoid()
])
# model = models.generic.GenericNetwork()
# model.hidden_layers.extend([
#     models.layers.LinearTransformLayer(X.shape[1] if len(X.shape) > 1 else 1),
#     models.layers.CompositeLayer(2, [
#         (5, models.layers.SquareActivation()), 
#     ]),
#     nn.Linear(5, y.shape[1] if len(y.shape) > 1 else 1),
#     nn.Sigmoid()
# ])

from torch_operation_counter import counters
from torch_operation_counter import OperationsCounterMode
from torch import ops
counters.operations_mapping[ops.aten.pow] = counters.basic_ops

with OperationsCounterMode(model) as op_counter:
    model(X)

ops = op_counter.total_operations / X.shape[0]
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#params_all = sum(p.numel() for p in model.parameters())

print(f"Ops: {ops}")
print(F"Params: {params}")
#print(F"Params + Non-Trainable: {params_all}")
print(f"Ops/Param = {ops / params:.2f}")