from . import layers
import torch
import torch.nn as nn


class GenericNetwork(nn.Module):
    def __init__(self):
        """
        GenericNetwork for general-purpose feedforward tasks.
        Arguments:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output features.
            layer_spec (list, optional): Specification for hidden layers.
        """
        super(GenericNetwork, self).__init__()
                
        # Parse layer_spec to create the hidden layers
        self.hidden_layers = nn.ModuleList()
        
        # Final output layer
        #self.output_layer = nn.Linear(self.hidden_layers[-2].out_features, output_dim)

    def forward(self, x):
        # Pass input through hidden layers
        for layer in self.hidden_layers:
            if isinstance(layer, layers.InteractionLayer):
                x = torch.cat((x, layer(x)), dim=1)
            else:
                x = layer(x)
        
        return x
    
    def get_layer_outputs(self, x):
        """
        Collects and returns outputs at each layer in the network for the input dataset.
        
        Arguments:
            x (torch.Tensor): The input dataset as a tensor.
        
        Returns:
            list[torch.Tensor]: A list of outputs at each layer in the network.
        """
        layer_outputs = []
        for layer in self.hidden_layers:
            if isinstance(layer, layers.InteractionLayer):
                x = torch.cat((x, layer(x)), dim=1)
            else:
                x = layer(x)
            layer_outputs.append(x)  # Store the output of the current layer
        return layer_outputs