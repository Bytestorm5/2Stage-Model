import torch
import torch.nn as nn
from itertools import combinations

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

def schedule_or_const(x):
    """
    Convert a schedule or constant value into a callable function that returns the mi_weight for a given epoch.

    Args:
        x (float, int, list, tuple, callable, _LRScheduler): 
            - float or int: constant mi_weight
            - list or tuple: mi_weight per epoch
            - callable: function that takes epoch (int) and returns mi_weight (float)
            - PyTorch Scheduler: instance of torch.optim.lr_scheduler._LRScheduler

    Returns:
        A callable that takes epoch (int) and returns mi_weight (float)

    Raises:
        ValueError: If `x` is not one of the supported types.
    """
    if isinstance(x, (float, int)):
        # Constant mi_weight
        return lambda epoch: x

    elif isinstance(x, (list, tuple)):
        # List or tuple schedule
        def get_from_list(epoch):
            if epoch < len(x):
                return x[epoch]
            else:
                # If epoch exceeds the schedule length, return the last value
                return x[-1]
        return get_from_list

    elif callable(x):
        # Callable schedule
        return x
    else:
        raise ValueError("Unsupported type for schedule_or_const. Supported types are: float, int, list, tuple, callable, and torch.optim.lr_scheduler._LRScheduler.")

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, l1_weight=1e-5, device='cpu'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            l1_loss = l1_weight * sum(p.abs().sum() for p in model.parameters())
            loss += l1_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

def train_autoencoder(model, train_loader, criterion, optimizer, num_epochs=50, mi_weight=1e-3, l1_weight=1e-5, device='cpu'):
    """
    Train an autoencoder with mutual information and reconstruction loss.

    Args:
        model (nn.Module): Autoencoder model to be trained.
        loader (DataLoader): DataLoader for the dataset.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        mi_weight (float): Weight for the mutual information loss component.
        device (str): Device to run the training on ('cpu' or 'cuda').
    """
    # Move model to device
    model = model.to(device)

    mi_weight_func = schedule_or_const(mi_weight)
    l1_weight_func = schedule_or_const(l1_weight)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_mi_loss = 0.0

        for data in train_loader:
            inputs = data[0].to(device)  # Assuming data is a tuple (input, target)
            inputs = inputs.to(torch.float32)

            # Forward pass
            reconstructed, latent = model(inputs)

            # Compute reconstruction loss
            recon_loss = criterion(reconstructed, inputs)

            # Compute mutual information loss
            mi_total = sum_mutual_information(latent, device=device)
            mi_loss = mi_weight_func(epoch) * torch.tensor(mi_total, dtype=torch.float32, device=device)

            l1_loss = l1_weight_func(epoch) * sum(p.abs().sum() for p in model.parameters())
            # Total loss: reconstruction + mi_weight * mutual information
            loss = recon_loss - mi_loss + l1_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses
            total_loss += loss.item() * inputs.size(0)
            total_recon_loss += recon_loss.item() * inputs.size(0)
            total_mi_loss += mi_loss.item() * inputs.size(0)

        # Calculate average losses per epoch
        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_mi_loss = total_mi_loss / len(train_loader.dataset)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, "
              f"Recon Loss: {avg_recon_loss:.4f}, MI Loss: {avg_mi_loss:.4f}")

    print("Training complete.")
