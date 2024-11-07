#import datasets
import numpy as np
from itertools import combinations
from sklearn.feature_selection import mutual_info_regression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
print(f'USING DEVICE: {device}')

def sum_mutual_information(X, num_bins=10, eps=1e-10, device='cpu'):
    """
    Computes the sum of mutual information for all unique pairs of features in X using PyTorch.

    Args:
        X (torch.Tensor): Input tensor of shape (n_samples, n_features).
        num_bins (int): Number of bins to discretize continuous features.
        eps (float): Small value to avoid log(0).
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        float: Sum of mutual information across all unique feature pairs.
    """
    X = X.to(device)
    n_samples, n_features = X.shape

    # Step 1: Binning
    # Compute bin edges for each feature
    bin_edges = []
    for i in range(n_features):
        min_val = torch.min(X[:, i])
        max_val = torch.max(X[:, i])
        edges = torch.linspace(min_val, max_val, steps=num_bins + 1).to(device)
        bin_edges.append(edges)
    
    # Assign each feature value to a bin
    binned_X = torch.zeros_like(X, dtype=torch.long)
    for i in range(n_features):
        # torch.bucketize returns indices in [1, num_bins], so subtract 1 for 0-based
        binned_X[:, i] = torch.bucketize(X[:, i], bin_edges[i][1:-1], right=False)
    
    # Step 2: Compute marginal histograms
    # Shape: (n_features, num_bins)
    marginals = torch.zeros(n_features, num_bins, device=device)
    for i in range(n_features):
        marginals[i].scatter_add_(0, binned_X[:, i], torch.ones(n_samples, device=device))
    marginals = marginals / n_samples  # Convert to probabilities

    # Step 3: Compute joint histograms for all unique pairs
    total_mi = torch.tensor(0.0, device=device)
    pair_count = 0

    # To optimize, process in batches if n_features is large
    for i, j in combinations(range(n_features), 2):
        # Compute joint histogram
        joint_hist = torch.zeros(num_bins, num_bins, device=device)
        indices = binned_X[:, [i, j]]
        # Convert pair of bins to single indices for bincount
        joint_indices = indices[:, 0] * num_bins + indices[:, 1]
        joint_counts = torch.bincount(joint_indices, minlength=num_bins*num_bins)
        joint_hist = joint_counts.reshape(num_bins, num_bins).float()
        joint_prob = joint_hist / n_samples  # Joint probability

        # Compute mutual information
        # P(X,Y) * log(P(X,Y) / (P(X)P(Y)))
        px = marginals[i].unsqueeze(1)  # Shape: (num_bins, 1)
        py = marginals[j].unsqueeze(0)  # Shape: (1, num_bins)
        pxy = joint_prob

        # Avoid division by zero and log(0) by adding eps
        ratio = pxy / (px * py + eps)
        log_ratio = torch.log(ratio + eps)
        mi_matrix = pxy * log_ratio

        mi = torch.sum(mi_matrix)
        total_mi += mi
        pair_count += 1

    # Optionally print the number of pairs processed
    # print(f"Computed mutual information for {pair_count} pairs.")

    return total_mi.item()

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, latent_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            # For reconstruction, depending on data, activation can vary
            # For example, use sigmoid for normalized data
            # nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# Hyperparameters
input_dim = 784  # Example for MNIST
latent_dim = 32
num_epochs = 50
batch_size = 256
learning_rate = 1e-3
mi_weight = 1e-3  # Weight for mutual information loss

# Prepare your dataset
# Replace the following lines with your dataset loading mechanism
# Example with random data
# X = np.random.rand(10000, input_dim).astype(np.float32)
# dataset = TensorDataset(torch.from_numpy(X))
# loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example using MNIST
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
criterion_recon = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#progress = tqdm(total=len(batch_size))

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_mi_loss = 0.0
    for data in loader:
        inputs = data[0].to(device)  # Assuming data is a tuple (input, target)
        inputs = inputs.to(torch.float32)

        # Forward pass
        reconstructed, latent = model(inputs)
        
        # Compute reconstruction loss
        recon_loss = criterion_recon(reconstructed, inputs)
        
        # Compute mutual information loss
        mi_total = sum_mutual_information(latent, device=device)
        
        # Convert mutual information to a tensor
        mi_loss = mi_weight * torch.tensor(mi_total, dtype=torch.float32, device=inputs.device)
        
        # Total loss: reconstruction + mi_weight * mutual information
        # Since we aim to maximize mutual information, but optimizers minimize loss,
        # we subtract the MI term
        loss = recon_loss + mi_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        total_recon_loss += recon_loss.item() * inputs.size(0)
        total_mi_loss += mi_loss.item() * inputs.size(0)
        #progress.update(1)
    
    avg_loss = total_loss / len(loader.dataset)
    avg_recon = total_recon_loss / len(loader.dataset)
    avg_mi = total_mi_loss / len(loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Recon Loss: {avg_recon:.4f}, MI Loss: {avg_mi:.4f}")

print("Training complete.")