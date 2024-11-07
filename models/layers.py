import torch
import torch.nn as nn

def layer_spec(spec, input_dim, final_dim=None):
    """
    Parse a specification list into a sequence of layers.
    Arguments:
        spec (list): The list specifying layer types and dimensions.
        input_dim (int): The initial input dimension.
        final_dim (int, optional): The final output dimension, if applicable.
    Returns:
        nn.ModuleList: A list of PyTorch layers based on the specification.
    """
    layers_list = nn.ModuleList()
    current_dim = input_dim
    
    for item in spec:
        if isinstance(item, int):
            # Add a linear layer with ReLU
            layers_list.append(nn.Linear(current_dim, item))
            layers_list.append(nn.ReLU(True))
            current_dim = item
        elif isinstance(item, str) and item == 'interact':
            # Add an interaction layer
            interaction_layer = InteractionLayer(current_dim)
            layers_list.append(interaction_layer)
            # Update current_dim based on interaction layer output
            current_dim = int(current_dim * (current_dim + 1) / 2) + current_dim
    
    # Optionally add a final output layer
    if final_dim is not None:
        layers_list.append(nn.Linear(current_dim, final_dim))
    
    return layers_list

class InteractionLayer(nn.Module):
    """
    Output Size: N(N+1)/2
    """
    def __init__(self, input_dim):
        super(InteractionLayer, self).__init__()
        self.input_dim = input_dim

    def forward(self, x):
        # x has shape (batch_size, input_dim)
        
        # Compute the outer product to get all pairwise interactions
        outer_product = torch.bmm(x.unsqueeze(2), x.unsqueeze(1))
        
        # Extract the upper triangular elements, including the diagonal
        # Indices of elements in the upper triangle (including the diagonal)
        indices = torch.triu_indices(self.input_dim, self.input_dim, offset=0)
        
        # Use the indices to gather both self-interactions and unique 2-pair interactions
        interactions = outer_product[:, indices[0], indices[1]]
        
        # `interactions` will have shape (batch_size, num_interactions)
        return interactions