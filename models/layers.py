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
        elif isinstance(item, str):
            params = item.split('|')
            if params[0] == 'interact':
                # Add an interaction layer
                interaction_layer = InteractionLayer(current_dim)
                layers_list.append(interaction_layer)
                # Update current_dim based on interaction layer output
                current_dim = int(current_dim * (current_dim + 1) / 2) + current_dim
            elif params[0] == 'linear_transform':
                lintran_layer = LinearTransformLayer(current_dim)
                layers_list.append(lintran_layer)
                current_dim = current_dim
            elif params[0] == 'square':
                if len(params) < 2:
                    raise ValueError("Square layers must define an output! e.g. 'square|10'")
                square_layer = SquareLayer(current_dim, int(params[1]))
                layers_list.append(square_layer)
                current_dim = int(params[1])
    
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
    

class SquareLayer(nn.Module):
    """
    A layer that applies weights and biases to the input,
    then computes squared and interaction terms.
    Output Size: N(N+1)/2
    """
    def __init__(self, input_dim, output_dim):
        super(SquareLayer, self).__init__()
        self.input_dim = input_dim
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x has shape (batch_size, input_dim)
        
        # Apply weights and biases to the input
        weighted_input = torch.pow(self.lin(x), 2)
        return weighted_input
    
class LinearTransformLayer(nn.Module):
    def __init__(self, num_features):
        super(LinearTransformLayer, self).__init__()
        # Define parameters m and b for each feature
        self.m = nn.Parameter(torch.ones(num_features))  # Initialize slopes to 1
        self.b = nn.Parameter(torch.zeros(num_features))  # Initialize intercepts to 0
        self.initialized = False

    def initialize_parameters(self, x):
        # Compute mean and std along the batch dimension
        mean = x.mean(dim=0, keepdim=False)
        std = x.std(dim=0, keepdim=False)

        # Avoid division by zero by setting std to a small value where it's zero
        std[std == 0] = 1e-6

        # Initialize m and b
        with torch.no_grad():  # Ensure this doesn't compute gradients
            self.m.copy_(1.0 / std)
            self.b.copy_(-mean / std)

        self.initialized = True  # Mark as initialized

    def forward(self, x):
        # Initialize parameters on the first forward pass
        if not self.initialized:
            self.initialize_parameters(x)

        # Apply the linear transformation mx + b
        return self.m * x + self.b