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
X, y = datasets.generate_concentric_circles(n_classes=2)
#X = X[:,1:].astype('float')
datasets.plot_dataset(X, y)
X = torch.Tensor(X)
y = torch.Tensor(y)
y = torch.stack((y == 0, y == 1), dim=1).float()

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

model = models.feedforward.FeedForwardNetwork(
    input_dim=X.shape[1] if len(X.shape) > 1 else 1, 
    output_dim=y.shape[1] if len(y.shape) > 1 else 1,
    layer_spec=['linear_transform', 'interact', 5]
)
model.hidden_layers.append(nn.Sigmoid())

criterion = nn.CrossEntropyLoss()

# Define Adam optimizer with the model parameters
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model using train_model function
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

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 80

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import math

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
    y = y[:, 0]
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
            pca = PCA(n_components=2)
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
            output_class_0 = output_np[y == 0]
            output_class_1 = output_np[y == 1]
            
            ax.hist(output_class_0.squeeze(), bins=30, alpha=0.7, color='blue', label='Class 0', edgecolor='black')
            ax.hist(output_class_1.squeeze(), bins=30, alpha=0.7, color='orange', label='Class 1', edgecolor='black')
            ax.set_title(f"Layer {idx} ({layer_class_name}) - 1D Output")
            ax.set_xlabel("Output Value")
            ax.set_ylabel("Frequency")
            ax.legend()
        
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


        
plot_layer_outputs(model, X, y)

# y_pred = y_pred.detach().numpy().reshape((y_pred.shape[0], ))

# import numpy as np
# import matplotlib.pyplot as plt
# x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
# y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
# xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# # Convert the grid to a tensor for prediction
# grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
# with torch.no_grad():
#     z = model(grid).numpy().reshape(xx.shape)

# # Plotting
# plt.figure(figsize=(8, 6))
# # Plot the predictions as a contour plot (background)
# plt.contourf(xx, yy, z, levels=50, cmap="RdYlGn", alpha=0.8)

# # Overlay the original data points
# plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap="RdYlGn", s=40, marker="o", label="True labels")

# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title("Model Predictions Background with True Labels")
# plt.colorbar(label="Model Prediction")
# plt.legend()
# plt.show()