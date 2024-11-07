import torch
import torch.nn as nn
import torch.optim as optim
import datasets
import models
import models.feedforward
import utils
from torch.utils.data import TensorDataset, DataLoader

X, y = datasets.generate_concentric_circles(factor=1, n_classes=2, noise=0.3)
X = torch.Tensor(X)
y = torch.Tensor(y).unsqueeze(1)
datasets.plot_dataset(X, y)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

model = models.feedforward.FeedForwardNetwork(
    input_dim=X.shape[1] if len(X.shape) > 1 else 1, 
    output_dim=y.shape[1] if len(y.shape) > 1 else 1,
    layer_spec=['interact']
)

criterion = nn.L1Loss()

# Define Adam optimizer with the model parameters
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
y_pred = model(X)
y_pred = y_pred.detach().numpy().reshape((y_pred.shape[0], ))

import numpy as np
import matplotlib.pyplot as plt
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Convert the grid to a tensor for prediction
grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
with torch.no_grad():
    z = model(grid).numpy().reshape(xx.shape)

# Plotting
plt.figure(figsize=(8, 6))
# Plot the predictions as a contour plot (background)
plt.contourf(xx, yy, z, levels=50, cmap="RdYlGn", alpha=0.8)

# Overlay the original data points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap="RdYlGn", s=40, marker="o", label="True labels")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Model Predictions Background with True Labels")
plt.colorbar(label="Model Prediction")
plt.legend()
plt.show()