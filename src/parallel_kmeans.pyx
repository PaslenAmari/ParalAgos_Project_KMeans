# Import frameworks
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt
import openmp

def kmeans_parallel(double[:, :] X, int n_clusters=8, int max_iter=100, double tol=1e-4, int n_threads=4):
    """Parallel implementation of k-means clustering using OpenMP.
    
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
    n_threads : int, default=4
        Number of OpenMP threads to use
        
    Returns:
    --------
    labels : ndarray of shape (n_samples,)
        Labels of each point
    centroids : ndarray of shape (n_clusters, n_features)
        Final cluster centroids
    """
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    
    # Initialize centroids
    rng = np.random.RandomState(42)
    centroid_indices = rng.choice(n_samples, n_clusters, replace=False)
    centroids = np.array(X[centroid_indices], dtype=np.float64)
    cdef double[:, :] centroids_view = centroids
    
    cdef double[:, :] prev_centroids = np.zeros_like(centroids)
    cdef int[:] labels = np.zeros(n_samples, dtype=np.int32)
    
    cdef int i, j, k, iteration
    cdef double min_dist, dist
    cdef double[:] counts = np.zeros(n_clusters, dtype=np.float64)
    
    for iteration in range(max_iter):
        # Parallel assignment of points to clusters
        with nogil:
            for i in prange(n_samples, num_threads=n_threads):
                min_dist = float('inf')
                for j in range(n_clusters):
                    dist = 0.0
                    for k in range(n_features):
                        dist += (X[i, k] - centroids_view[j, k]) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        labels[i] = j
        
        # Update centroids (parallel reduction)
        prev_centroids[:] = centroids_view[:]
        centroids[:] = 0
        counts[:] = 0
        
        # Parallel accumulation
        with nogil:
            for i in prange(n_samples, num_threads=n_threads):
                for k in range(n_features):
                    centroids_view[labels[i], k] += X[i, k]
                counts[labels[i]] += 1
        
        # Finalize centroid computation
        for j in range(n_clusters):
            if counts[j] > 0:
                for k in range(n_features):
                    centroids_view[j, k] /= counts[j]
        
        # Check convergence
        cdef double change = 0.0
        for j in range(n_clusters):
            for k in range(n_features):
                change += (centroids_view[j, k] - prev_centroids[j, k]) ** 2
        if change < tol:
            break
    
    return np.asarray(labels), np.asarray(centroids_view)

if __name__ == "__main__":
    # Simple test
    from data_loader import load_handoutlines_dataset
    
    # Load data
    X = load_handoutlines_dataset()
    
    # Run parallel k-means
    print("Running parallel k-means clustering...")
    labels, centroids = kmeans_parallel(X, n_threads=4)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of clusters: {len(np.unique(labels))}")
    print(f"Centroids shape: {centroids.shape}")
