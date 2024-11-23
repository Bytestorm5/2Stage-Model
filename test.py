import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# My stuff
from data.uci_data import get_data
import datasets
import models
import models.layers
import models.feedforward
import models.generic
import utils

def custom_equation(tensor):
    # x^(1/3) - y^(2/3)
    # x^2 + y^2
    # e^-(x^2y^2)
    # sin(sqrt(x^2 + y^2))
    # x/y
    x, y = tensor[:, 0], tensor[:, 1]

    return torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))

X, y = datasets.generate_equation_dataset(custom_equation)
X = torch.Tensor(X)
y = torch.Tensor(y)

datasets.plot_dataset(X, y)

# y = torch.stack((y == 0, y == 1), dim=1).float()

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

model = models.generic.GenericNetwork()
def get_activation():
    return models.layers.CompositeActivation([models.layers.AbsActivation(), nn.ReLU(), models.layers.SquareActivation()])
model.hidden_layers.extend([
    models.layers.LinearTransformLayer(X.shape[1] if len(X.shape) > 1 else 1),
    nn.Linear(2, 6),
    get_activation(),
    nn.Linear(6, 12),
    get_activation(),
    nn.Linear(12, 6),
    get_activation(),
    nn.Linear(6, 1)
])

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

utils.train_model(
    model=model,
    train_loader=dataloader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=500,
    l1_weight=1e-5,
    device='cpu'
)
model.eval()

for name, param in model.named_parameters():
    print(name, param.shape, param)

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 80
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import math

# y_pred = model(X)
# plt.plot(X, c=y_pred, cmap='viridis', alpha=0.6)
# plt.show()
def plot_layer_outputs(model, X, y):
    """
    Plots the outputs from each layer in the network:
        - Uses PCA if output dimensions > 2.
        - Plots directly if output dimensions == 2.
        - Uses overlaid histograms if output dimensions == 1.
    
    Arguments:
        layer_outputs (list[torch.Tensor]): A list of outputs at each layer in the network.
        y (torch.Tensor or np.ndarray): Labels (0 or 1) to color the points or histograms accordingly.
    """
    y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    #y = y[:, 0]
    # List of outputs at each layer
    layer_outputs = model.get_layer_outputs(X)
    
    num_layers = len(layer_outputs)
    cols = 3  # Number of columns in the grid
    rows = math.ceil(num_layers / cols)  # Determine rows based on number of layers
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()  # Flatten the axes array for easy iteration
    
    for idx, output in enumerate(layer_outputs):
        layer_class_name = type(model.hidden_layers[idx]).__name__
        # Convert the tensor to a NumPy array
        output_np = output.detach().cpu().numpy()
        output_dim = output_np.shape[1]  # Get the dimensionality of the output
        
        ax = axes[idx]
        
        if output_dim > 2:
            # Apply PCA to reduce the dimensionality to 2D
            pca = PCA(n_components=3)
            transformed_output = pca.fit_transform(output_np)
            explained_variance = sum(pca.explained_variance_ratio_) * 100
            
            # Scatter plot with PCA
            scatter = ax.scatter(
                transformed_output[:, 0], 
                transformed_output[:, 1], 
                c=y, 
                cmap='viridis', 
                alpha=0.6
            )
            ax.set_title(f"Layer {idx} ({layer_class_name}) - PCA (Expl. Var.: {explained_variance:.2f}%)")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.grid(True)
        
        elif output_dim == 2:
            # Direct scatter plot without PCA
            scatter = ax.scatter(
                output_np[:, 0], 
                output_np[:, 1], 
                c=y, 
                cmap='viridis', 
                alpha=0.6
            )
            ax.set_title(f"Layer {idx} ({layer_class_name}) - 2D Output")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.grid(True)
        
        elif output_dim == 1:
            # Overlaid histograms for 1D output
            scatter = ax.scatter(
                y, 
                output_np, 
                c=y, 
                cmap='viridis', 
                alpha=0.6
            )
            ax.set_title(f"Layer {idx} ({layer_class_name}) - 2D Output")
            ax.set_xlabel("Ground Truth")
            ax.set_ylabel("Predicted")
            ax.grid(True)

            # Overlay a red dashed line along x = y
            min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])  # Get the lower bound
            max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])  # Get the upper bound
            #ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="x = y")

            # Ensure the limits are consistent for better visualization
            # ax.set_xlim(min_val, max_val)
            # ax.set_ylim(min_val, max_val)

            # Optional: Add a legend
            ax.legend()

            # Display the plot
            plt.show()
            # output_class_0 = y
            # output_class_1 = output_np
            
            # ax.hist(output_class_0.squeeze(), bins=30, alpha=0.7, color='blue', label='Class 0', edgecolor='black')
            # ax.hist(output_class_1.squeeze(), bins=30, alpha=0.7, color='orange', label='Class 1', edgecolor='black')
            # ax.set_title(f"Layer {idx} ({layer_class_name}) - 1D Output")
            # ax.set_xlabel("Output Value")
            # ax.set_ylabel("Frequency")
            # ax.legend()
        
        else:
            ax.text(0.5, 0.5, "Unsupported Dim", horizontalalignment='center', verticalalignment='center')
        
    # Hide any unused subplots
    for ax in axes[len(layer_outputs):]:
        ax.axis('off')
    
    # Add a single color bar for scatter plots if at least one scatter exists
    # scatter_exists = any(output.detach().cpu().numpy().shape[1] >= 2 for output in layer_outputs)
    # if scatter_exists:
    #     cbar = fig.colorbar(scatter, ax=axes, location='right', shrink=0.8, pad=0.1)
    #     cbar.set_label('Class Label')
    
    plt.tight_layout()
    plt.show()


        
#plot_layer_outputs(model, X, y)


def plot_decision_boundary(model, X, y):
    """
    Plots the decision boundary of the model against the dataset.
    """
    # Create a grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Flatten the grid and predict
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.Tensor(grid)
    with torch.no_grad():
        Z = model(grid_tensor).detach().cpu().numpy()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y.numpy(), edgecolor='k', cmap='viridis')
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Plot the layer outputs
plot_layer_outputs(model, X, y)

# Plot the decision boundary
if X.shape[1] == 2:  # Ensure data is 2D for decision boundary visualization
    plot_decision_boundary(model, X, y)
else:
    print("Decision boundary plot only supported for 2D data.")