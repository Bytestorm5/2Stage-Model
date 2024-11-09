import torch
from thop import profile
import models.autoencoder
from thop.profile import register_hooks

import models.layers

import torchvision.models as tmodels

def interactionlayer_flops(module, input, output):
    """
    Calculate FLOPs for InteractionLayer.
    
    Args:
        module: The InteractionLayer instance.
        input: Input tensor to the layer.
        output: Output tensor from the layer.
    
    Returns:
        (macs, params): Tuple containing MACs and params.
    """
    # Extract input dimensions
    batch_size, input_dim = input[0].size()
    
    # Compute FLOPs for torch.bmm(x.unsqueeze(2), x.unsqueeze(1))
    # torch.bmm: For each batch, multiply (input_dim x 1) with (1 x input_dim)
    # FLOPs per batch = 2 * input_dim * input_dim (1 multiply and 1 add per element)
    # Total FLOPs = batch_size * 2 * input_dim^2
    flops = 2 * input_dim * input_dim * batch_size
    
    # No parameters in InteractionLayer
    # module.total_params += 0
    
    module.total_ops += flops
    
    #return flops, params

# Register the custom handler with thop
register_hooks[models.layers.InteractionLayer] = interactionlayer_flops


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
import models.feedforward
import utils
from torch.utils.data import TensorDataset, DataLoader

# X, y = datasets.generate_concentric_circles(factor=1, n_classes=2, noise=0.3)
# X = torch.Tensor(X)
# y = torch.Tensor(y).unsqueeze(1)
X, y = get_data(464)
#X = X[:,1:].astype('float')
#datasets.plot_dataset(X, y)
X = torch.Tensor(X)
y = torch.Tensor(y)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

model = models.feedforward.FeedForwardNetwork(
    input_dim=X.shape[1] if len(X.shape) > 1 else 1, 
    output_dim=y.shape[1] if len(y.shape) > 1 else 1,
    layer_spec=[100, 30, 256, 30]
)

# Create a dummy input tensor matching the input size
input = torch.randn(1, 81).to(next(model.parameters()).device)

# Profile the model
macs, params = profile(model, inputs=(input, ), verbose=True)

# Convert MACs to FLOPs
flops = macs * 2

print(f"FLOPs: {flops}")
print(f"Parameters: {params}")
print(f"FLOPs/Parameters: {flops / params:.2f}")
