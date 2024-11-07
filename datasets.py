import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_s_curve, make_swiss_roll
from sklearn.decomposition import PCA

def plot_dataset(X, y, title='Dataset Visualization', cmap='viridis', edgecolor='k', s=25, explained_variance=False):
    """
    Plot the dataset X and y. If X has more than 2 features, apply PCA to reduce to 2D.

    Parameters:
    - X: Feature matrix (numpy array or similar), shape (n_samples, n_features).
    - y: Label vector (numpy array or similar), shape (n_samples,).
    - title: Title of the plot.
    - cmap: Colormap for the scatter plot.
    - edgecolor: Edge color for scatter points.
    - s: Size of scatter points.
    - explained_variance: If True and PCA is applied, display explained variance.

    Returns:
    - None
    """
    # Check the dimensionality of X
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array. Got {X.ndim}D array instead.")
    
    n_features = X.shape[1]
    
    if n_features == 2:
        X_vis = X
        variance_info = ""
    elif n_features > 2:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
        if explained_variance:
            variance = pca.explained_variance_ratio_
            variance_info = f' (PCA Explained Variance: {variance[0]:.2f}, {variance[1]:.2f})'
        else:
            variance_info = ""
    else:
        raise ValueError(f"X must have at least 2 features for visualization. Got {n_features} feature(s).")
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap=cmap, edgecolor=edgecolor, s=s)
    plt.title(title + variance_info)
    plt.xlabel('Component 1' if n_features > 2 else 'X1')
    plt.ylabel('Component 2' if n_features > 2 else 'X2')
    plt.colorbar(scatter, label='Class Label')
    plt.grid(True, linestyle='--', alpha=0.5)
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
        X_class, _ = make_circles(n_samples=n_samples_per_class, noise=noise, factor=current_factor)
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