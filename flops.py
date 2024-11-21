import torch
from thop import profile
import models.autoencoder
from thop.profile import register_hooks

import models.layers
import models.generic

import torchvision.models as tmodels

def interaction_flops(module, input, output):
    # Example: Outer product FLOPs
    input_dim = input[0].numel()
    flops = input_dim * input_dim  # Outer product FLOPs
    module.total_ops += flops
    
def abs_activation_flops(module, input, output):
    num_features = input[0].numel()
    module.total_ops += num_features

def gauss_activation_flops(module, input, output):
    num_features = input[0].numel()
    # (x - m)
    # z / s (add + mul; 1 flop)
    # z^2 (mul; 1 flop)
    # exp(-z^2) (mul; 1 flop) + (exp; 15 flops)
    module.total_ops += num_features * 18

def composite_layer_flops(module: models.layers.CompositeLayer, input, output):
    x = input[0]
    total_flops = 0
    for layer in module.sub_layers:
        # Call custom handlers for specific layers
        layer_flops = register_hooks[layer.__class__](layer, x, output)
        total_flops += layer.total_ops
    module.total_ops += total_flops

def sequential_flops(module: torch.nn.Sequential, input, output):
    x = input[0]
    total_flops = 0
    for layer in module._modules.values():
        # Call custom handlers for specific layers
        o = layer(x)
        layer_flops = register_hooks[layer.__class__](layer, x, o)
        total_flops += layer.total_ops
        x = o  # Forward pass to update x
    module.total_ops += total_flops

def generic_network_flops(module: models.generic.GenericNetwork, input, output):
    x = input[0]
    total_flops = 0
    for layer in module.hidden_layers:
        # Call custom handlers for specific layers
        o = layer(x)
        layer_flops = register_hooks[layer.__class__](layer, x, o)
        total_flops += layer.total_ops
        x = o  # Forward pass to update x
    module.total_ops += total_flops

# Register the custom handler with thop
register_hooks[models.layers.InteractionLayer] = interaction_flops
# Square and abs are equivalent here
register_hooks[models.layers.SquareActivation] = abs_activation_flops
register_hooks[models.layers.AbsActivation] = abs_activation_flops
register_hooks[models.layers.LinearTransformLayer] = abs_activation_flops
register_hooks[models.layers.GaussActivation] = gauss_activation_flops
register_hooks[models.layers.CompositeLayer] = composite_layer_flops
register_hooks[models.generic.GenericNetwork] = generic_network_flops
register_hooks[torch.nn.Sequential] = sequential_flops


# Define your model (e.g., ResNet18)
# model = models.autoencoder.Autoencoder(
#     input_dim=2, 
#     latent_dim=2,
#     encoder_spec=[3, 'interact', 5]
# )
import torch
import torch.nn as nn
import torch.optim as optim
import datasets
from data.uci_data import get_data
import models
import models.generic
import models.feedforward
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
model = models.feedforward.FeedForwardNetwork(
    input_dim=X.shape[1] if len(X.shape) > 1 else 1, 
    output_dim=y.shape[1] if len(y.shape) > 1 else 1,
    layer_spec=[5]
)
model.hidden_layers.append(nn.Sigmoid())
# model = models.generic.GenericNetwork()
# model.hidden_layers.extend([
#     models.layers.LinearTransformLayer(X.shape[1] if len(X.shape) > 1 else 1),
#     models.layers.CompositeLayer(2, [
#         (5, nn.ReLU(True)), 
#         (5, models.layers.SquareActivation()), 
#         (5, models.layers.AbsActivation()), 
#         (5, models.layers.GaussActivation(5))
#     ]),
#     nn.Linear(20, y.shape[1] if len(y.shape) > 1 else 1),
#     nn.Sigmoid()
# ])

# Create a dummy input tensor matching the input size
input = torch.randn(1000, 2).to(next(model.parameters()).device)

# Profile the model
macs, params = profile(model, inputs=(input, ), verbose=True)

# Convert MACs to FLOPs
flops = macs * 2

print(f"FLOPs: {flops}")
print(f"Parameters: {params}")
print(f"FLOPs/Parameters: {flops / params:.2f}")
