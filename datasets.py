import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_s_curve, make_swiss_roll
from sklearn.decomposition import PCA
import torch

def plot_dataset(X, y, title='Dataset Visualization', cmap='viridis', edgecolor='k', s=25, explained_variance=True):
    """
    Plot the dataset X and y. If X has more than 2 features, apply PCA to reduce to 2D.
    If y has multiple features, generate separate subplots for each y feature.
    
    Parameters:
    - X: Feature matrix (numpy array or similar), shape (n_samples, n_features).
    - y: Label matrix/vector (numpy array or similar), shape (n_samples,) or (n_samples, n_outputs).
    - title: Base title of the plot.
    - cmap: Colormap for the scatter plot.
    - edgecolor: Edge color for scatter points.
    - s: Size of scatter points.
    - explained_variance: If True and PCA is applied, display explained variance.
    
    Returns:
    - None
    """
    # Validate X
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array. Got {X.ndim}D array instead.")
    
    n_samples, n_features = X.shape
    
    if n_features == 2:
        X_vis = X
        variance_info = ""
        xlabel, ylabel = 'X1', 'X2'
    elif n_features > 2:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
        if explained_variance:
            variance = pca.explained_variance_ratio_
            variance_info = f' (PCA Explained Variance: {variance[0]:.2f}, {variance[1]:.2f})'
        else:
            variance_info = ""
        xlabel, ylabel = 'Component 1', 'Component 2'
    else:
        raise ValueError(f"X must have at least 2 features for visualization. Got {n_features} feature(s).")
    
    # Handle y dimensionality
    if y.ndim == 1:
        y = y.reshape(-1, 1)  # Convert to 2D for uniform processing
    elif y.ndim != 2:
        raise ValueError(f"y must be a 1D or 2D array. Got {y.ndim}D array instead.")
    
    n_outputs = y.shape[1]
    
    # Determine subplot grid size
    n_cols = 2
    n_rows = int(np.ceil(n_outputs / n_cols))
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case of single row/column
    
    for idx in range(n_outputs):
        ax = axes[idx]
        scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], 
                             c=y[:, idx], cmap=cmap, edgecolor=edgecolor, s=s)
        
        # Construct title for each subplot
        subplot_title = f"{title} - y{idx + 1}{variance_info}"
        ax.set_title(subplot_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar_label = 'Class Label' if y.shape[1] == 1 else f'y{idx + 1}'
        cbar.set_label(cbar_label)
    
    # Remove any unused subplots
    for idx in range(n_outputs, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

def plot_model_predictions(model, X, y, title='Model Predictions with Dataset Overlay', 
                           cmap_contour='RdYlGn', cmap_scatter='viridis', 
                           edgecolor='k', s=40, grid_resolution=100, explained_variance=True):
    """
    Plot model predictions as background contours with dataset points overlaid.
    Handles multi-dimensional y by generating multiple subplots.

    Parameters:
    - model: Trained PyTorch model. Should accept inputs of shape (n_samples, n_features).
    - X: Feature matrix (numpy array or similar), shape (n_samples, n_features).
    - y: Label matrix/vector (numpy array or similar), shape (n_samples,) or (n_samples, n_outputs).
    - title: Base title of the plots.
    - cmap_contour: Colormap for the contour plots.
    - cmap_scatter: Colormap for the scatter plots.
    - edgecolor: Edge color for scatter points.
    - s: Size of scatter points.
    - grid_resolution: Number of points along each axis for the grid.
    - explained_variance: If True and PCA is applied, display explained variance.

    Returns:
    - None
    """
    # Validate X
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array. Got {X.ndim}D array instead.")
    
    n_samples, n_features = X.shape
    
    # Apply PCA if necessary
    if n_features > 2:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
        if explained_variance:
            variance = pca.explained_variance_ratio_
            variance_info = f' (PCA Explained Variance: {variance[0]:.2f}, {variance[1]:.2f})'
        else:
            variance_info = ""
        xlabel, ylabel_axis = 'Component 1', 'Component 2'
    elif n_features == 2:
        X_vis = X
        variance_info = ""
        xlabel, ylabel_axis = 'Feature 1', 'Feature 2'
    else:
        raise ValueError(f"X must have at least 2 features for visualization. Got {n_features} feature(s).")
    
    # Handle y dimensionality
    if y.ndim == 1:
        y = y.reshape(-1, 1)  # Convert to 2D for uniform processing
    elif y.ndim != 2:
        raise ValueError(f"y must be a 1D or 2D array. Got {y.ndim}D array instead.")
    
    n_outputs = y.shape[1]
    
    # Determine subplot grid size
    n_cols = 2
    n_rows = int(np.ceil(n_outputs / n_cols))
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case of single row/column
    
    # Prepare grid for contour plots
    if n_features > 2:
        # Generate grid in PCA space
        x_min, x_max = X_vis[:, 0].min() - 1.0, X_vis[:, 0].max() + 1.0
        y_min, y_max = X_vis[:, 1].min() - 1.0, X_vis[:, 1].max() + 1.0
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution), 
                             np.linspace(y_min, y_max, grid_resolution))
        # Flatten the grid and inverse transform to original space
        grid_pca = np.c_[xx.ravel(), yy.ravel()]
        grid_original = pca.inverse_transform(grid_pca)
    else:
        # Generate grid in original X space
        x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
        y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution), 
                             np.linspace(y_min, y_max, grid_resolution))
        grid_original = np.c_[xx.ravel(), yy.ravel()]
    
    # Convert the grid to a tensor for prediction
    grid_tensor = torch.Tensor(grid_original)
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    with torch.no_grad():
        z_pred = model(grid_tensor).numpy()
    
    # Handle multiple outputs
    if n_outputs > 1:
        if z_pred.ndim == 1:
            z_pred = z_pred.reshape(-1, 1)
        elif z_pred.ndim == 2 and z_pred.shape[1] != n_outputs:
            raise ValueError(f"Model output shape {z_pred.shape} does not match number of y features {n_outputs}.")
    else:
        z_pred = z_pred.reshape(-1, 1)
    
    for idx in range(n_outputs):
        ax = axes[idx]
        
        # Get predictions for current y feature
        z = z_pred[:, idx].reshape(xx.shape)
        
        # Plot the predictions as a contour plot (background)
        contour = ax.contourf(xx, yy, z, levels=50, cmap=cmap_contour, alpha=0.8)
        
        # Overlay the original data points
        scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y[:, idx], 
                             cmap=cmap_scatter, edgecolor=edgecolor, s=s, label="Data Points")
        
        # Construct title for each subplot
        subplot_title = f"{title} - y{idx + 1}{variance_info}"
        ax.set_title(subplot_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel_axis)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add colorbar for contour
        cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        cbar_label = 'Model Prediction'
        cbar.set_label(cbar_label)
        
        # Add colorbar for scatter
        scbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        scbar_label = f'y{idx + 1} Label'
        scbar.set_label(scbar_label)
        
        ax.legend(loc='upper right')
    
    # Remove any unused subplots
    for idx in range(n_outputs, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

def generate_nested_spirals(n_points, n_classes, noise=0.5):
    """
    Generate nested spirals dataset.

    Parameters:
    - n_points: Total number of points.
    - n_classes: Number of spiral classes.
    - noise: Standard deviation of Gaussian noise added to the data.

    Returns:
    - X: Feature matrix.
    - y: Labels.
    """
    X = []
    y = []
    n_points_per_class = n_points // n_classes
    for class_num in range(n_classes):
        theta = np.linspace(0, 4 * np.pi, n_points_per_class) + np.random.randn(n_points_per_class) * noise
        r = theta
        x = r * np.cos(theta) + np.random.randn(n_points_per_class) * noise
        y_coord = r * np.sin(theta) + np.random.randn(n_points_per_class) * noise
        X.extend(np.c_[x, y_coord])
        y.extend([class_num] * n_points_per_class)
    X = np.array(X)
    y = np.array(y)
    return X, y



def generate_concentric_circles(n_samples=1000, n_classes=3, noise=0.05, factor=0.5):
    """
    Generate concentric circles dataset.

    Parameters:
    - n_samples: Total number of samples.
    - n_classes: Number of concentric circles.
    - noise: Standard deviation of Gaussian noise.
    - factor: Scale factor between circles.

    Returns:
    - X: Feature matrix.
    - y: Labels.
    """
    X = []
    y = []
    n_samples_per_class = n_samples // n_classes
    for class_num in range(n_classes):
        # Adjust factor for each class to create concentric circles
        current_factor = factor * (class_num + 1) / n_classes
        thetas = np.random.rand(n_samples_per_class) * 2 * np.pi
        
        X_class = np.column_stack((
            current_factor * np.cos(thetas),
            current_factor * np.sin(thetas)
        ))
        X_class += noise * np.random.rand(*X_class.shape)
        
        X.extend(X_class)
        y.extend([class_num] * n_samples_per_class)
    X = np.array(X)
    y = np.array(y)
    return X, y

def generate_swiss_roll(n_samples=1000, noise=0.2):
    """
    Generate Swiss Roll dataset.

    Parameters:
    - n_samples: Number of samples.
    - noise: Standard deviation of Gaussian noise.

    Returns:
    - X: Feature matrix (2D projection).
    - y: Labels.
    """
    X, y = make_swiss_roll(n_samples=n_samples, noise=noise)
    # Project to 2D by taking the first and second dimensions
    X_2D = X[:, [0, 2]]
    return X_2D, y

def generate_high_dimensional_s_shape(n_samples=1000, noise=0.1, n_features=5):
    """
    Generate high-dimensional S-curve dataset.

    Parameters:
    - n_samples: Number of samples.
    - noise: Standard deviation of Gaussian noise.
    - n_features: Number of dimensions (must be >=3).

    Returns:
    - X: Feature matrix.
    - y: Labels.
    """
    if n_features < 3:
        raise ValueError("n_features must be at least 3 for S-curve.")
    X, y = make_s_curve(n_samples=n_samples, noise=noise)
    # Add additional random dimensions
    additional_dims = np.random.randn(n_samples, n_features - 3)
    X_high_dim = np.hstack((X, additional_dims))
    return X_high_dim, y

def generate_intersecting_rings(n_samples, n_classes, noise=0.05, radius_increment=1.0):
    """
    Generate intersecting rings dataset.

    Parameters:
    - n_samples: Total number of samples.
    - n_classes: Number of rings.
    - noise: Standard deviation of Gaussian noise.
    - radius_increment: Increment in radius for each subsequent ring.

    Returns:
    - X: Feature matrix.
    - y: Labels.
    """
    X = []
    y = []
    n_samples_per_class = n_samples // n_classes
    for class_num in range(n_classes):
        radius = radius_increment * (class_num + 1)
        theta = np.linspace(0, 2 * np.pi, n_samples_per_class) + np.random.randn(n_samples_per_class) * noise
        x = radius * np.cos(theta) + np.random.randn(n_samples_per_class) * noise
        y_coord = radius * np.sin(theta) + np.random.randn(n_samples_per_class) * noise
        X.extend(np.c_[x, y_coord])
        y.extend([class_num] * n_samples_per_class)
    X = np.array(X)
    y = np.array(y)
    return X, y

def generate_taylor_series_data(n_samples=1000, x_range=(-2 * np.pi, 2 * np.pi), order=1, noise=0.1):
    """
    Generate dataset based on the Taylor series expansion of sine function.

    Parameters:
    - n_samples: Number of samples to generate.
    - x_range: Tuple (min, max) defining the range of x values.
    - order: Maximum order of the Taylor series expansion.
    - noise: Standard deviation of Gaussian noise added to the data.

    Returns:
    - X: Feature matrix, where each row is [x, x^2, x^3, ..., x^order].
    - y: Labels corresponding to the sine values of x (with optional noise).
    """
    # Generate random x values within the specified range
    x_values = np.random.uniform(x_range[0], x_range[1], n_samples)

    # Compute true sine values
    y_true = np.sin(x_values)

    # Add noise to sine values
    y_noisy = y_true + np.random.normal(0, noise, n_samples)

    # Create feature matrix X with powers of x
    X = np.column_stack([x_values**i for i in range(1, order + 1)])
    
    return X, y_noisy