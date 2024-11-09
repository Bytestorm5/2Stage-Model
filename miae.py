import torch
import torch.nn as nn
import torch.optim as optim
import datasets
import models
import models.autoencoder
import models.feedforward
import utils
from torch.utils.data import TensorDataset, DataLoader

X, y = datasets.generate_concentric_circles(factor=1, n_classes=2, noise=0.3, n_samples=1000)
X = torch.Tensor(X)
y = torch.Tensor(y).unsqueeze(1)
datasets.plot_dataset(X, y)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

model = models.autoencoder.Autoencoder(
    input_dim=X.shape[1] if len(X.shape) > 1 else 1, 
    latent_dim=2,
    encoder_spec=[3, 'interact', 5]
)

criterion = nn.L1Loss()

# Define Adam optimizer with the model parameters
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model using train_model function
utils.train_autoencoder(
    model=model,
    train_loader=dataloader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=1000,
    mi_weight=0,
    l1_weight=1e-5,
    device='cpu'
)
model.eval()
y_pred = model(X)[1]
y_pred = y_pred.detach().numpy()

datasets.plot_dataset(y_pred, y)

y_pred = model(X)[0]
y_pred = y_pred.detach().numpy()

datasets.plot_dataset(y_pred, y)