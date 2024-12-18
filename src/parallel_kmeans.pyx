# Import frameworks
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt, INFINITY

def kmeans_parallel(double[:, :] X, int n_clusters=8, int max_iter=100, double tol=1e-4, int n_threads=4):
    """Parallel implementation of k-means clustering using OpenMP."""
    # Define variables
    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]
    cdef Py_ssize_t i, j, k, iteration
    cdef double dist, change
    cdef double[:] distances = np.zeros(n_clusters, dtype=np.float64)
    
    # Initialize centroids
    rng = np.random.RandomState(42)
    # Fix the centroid initialization
    indices = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = np.zeros((n_clusters, n_features), dtype=np.float64)
    for i in range(n_clusters):
        for k in range(n_features):
            centroids[i, k] = X[indices[i], k]
    
    cdef double[:, :] centroids_view = centroids
    cdef double[:, :] prev_centroids = np.zeros_like(centroids)
    cdef int[:] labels = np.zeros(n_samples, dtype=np.int32)
    cdef double[:] counts = np.zeros(n_clusters, dtype=np.float64)
    
    for iteration in range(max_iter):
        # Parallel assignment of points to clusters
        with nogil:
            for i in prange(n_samples, num_threads=n_threads):
                # Calculate distances to all centroids
                for j in range(n_clusters):
                    distances[j] = 0.0
                    for k in range(n_features):
                        distances[j] += (X[i, k] - centroids_view[j, k]) ** 2
                
                # Find nearest centroid
                labels[i] = 0
                for j in range(1, n_clusters):
                    if distances[j] < distances[labels[i]]:
                        labels[i] = j
        
        # Store previous centroids
        prev_centroids[:] = centroids_view[:]
        
        # Reset accumulators
        for j in range(n_clusters):
            counts[j] = 0
            for k in range(n_features):
                centroids_view[j, k] = 0
        
        # Accumulate sum for new centroids
        with nogil:
            for i in prange(n_samples, num_threads=n_threads):
                for k in range(n_features):
                    centroids_view[labels[i], k] += X[i, k]
                counts[labels[i]] += 1
        
        # Calculate new centroids and check convergence
        change = 0.0
        for j in range(n_clusters):
            if counts[j] > 0:
                for k in range(n_features):
                    centroids_view[j, k] /= counts[j]
                    change += (centroids_view[j, k] - prev_centroids[j, k]) ** 2
        
        if change < tol:
            break
    
    return np.asarray(labels), np.asarray(centroids_view)