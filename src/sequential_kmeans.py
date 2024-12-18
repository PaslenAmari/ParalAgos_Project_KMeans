# Import frameworks
import numpy as np

def kmeans_sequential(X, n_clusters=8, max_iter=100, tol=1e-4):
    """Sequential implementation of k-means clustering.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training instances to cluster
    n_clusters : int, default=8
        Number of clusters to form
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance to declare convergence
        
    Returns:
    --------
    labels : ndarray of shape (n_samples,)
        Labels of each point
    centroids : ndarray of shape (n_clusters, n_features)
        Final cluster centroids
    """
    n_samples, n_features = X.shape
    
    # Randomly initialize centroids
    rng = np.random.RandomState(42)
    centroid_indices = rng.choice(n_samples, n_clusters, replace=False)
    centroids = X[centroid_indices]
    
    prev_centroids = np.zeros_like(centroids)
    labels = np.zeros(n_samples, dtype=np.int32)
    
    for iteration in range(max_iter):
        # Assign points to nearest centroid
        for i in range(n_samples):
            min_dist = float('inf')
            for j in range(n_clusters):
                dist = np.sum((X[i] - centroids[j]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    labels[i] = j
        
        # Update centroids
        prev_centroids[:] = centroids[:]
        for j in range(n_clusters):
            mask = labels == j
            if np.any(mask):
                centroids[j] = X[mask].mean(axis=0)
        
        # Check convergence
        if np.sum((centroids - prev_centroids) ** 2) < tol:
            break
            
    return labels, centroids

if __name__ == "__main__":
    # Simple test
    from data_loader import load_handoutlines_dataset
    
    # Load data
    X = load_handoutlines_dataset()
    
    # Run k-means
    print("Running k-means clustering...")
    labels, centroids = kmeans_sequential(X)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of clusters: {len(np.unique(labels))}")
    print(f"Centroids shape: {centroids.shape}")
