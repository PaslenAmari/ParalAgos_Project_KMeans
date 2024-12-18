# ParalAgos_Project_KMeans - Parallel K-means Clustering Implementation
The Project for optimization purposes for K-Means clustering algotithm (with benchmarks - use dataset with at least 70000 samples)

This repository contains sequential and parallel implementations of the k-means clustering algorithm. 
The parallel version uses OpenMP via Cython to accelerate computations.

## Requirements

- Python 3.7+
- NumPy
- Cython
- A C compiler with OpenMP support (gcc recommended)
- scikit-learn (for dataset generation)

## Installation

1. Clone the repository:
```bash
.....
```
2. Install dependencies:
```bash
pip install numpy cython scikit-learn
```

4. Compile the Cython extension:
```bash
python setup.py build_ext --inplace
```

## Dataset

The implementation uses a synthetic dataset generated using scikit-learn's `make_blobs` function. The default configuration generates:
- 70,000 samples
- 10 features per sample
- 8 clusters
- Fixed random seed (42) for reproducibility

## Running the Benchmark

```bash
python benchmark.py
```
This will run both sequential and parallel implementations and output timing results and speedup calculations.

## Parallelization Strategy

The parallel implementation accelerates two main parts of the k-means algorithm:

1. **Point Assignment Phase**: The computation of distances between points and centroids is parallelized using OpenMP's parallel for loop (`prange`). Each thread processes a subset of the data points independently.

2. **Centroid Update Phase**: The accumulation of points for centroid updates is parallelized using a reduction pattern. Each thread maintains local sums which are then combined to compute the final centroids.

Key optimizations include:
- Using `nogil` contexts to release the GIL during parallel sections
- Memory views for efficient array access
- Parallel reduction for centroid updates

## Performance Analysis

The implementation shows significant speedup with multiple threads:

1. The speedup is approximately linear up to 4 threads
2. Beyond 4 threads, the speedup may plateau due to:
   - Memory bandwidth limitations
   - Overhead from thread management
   - Hardware thread count limitations

Actual speedup values will depend on:
- Hardware configuration (number of cores, memory bandwidth)
- Dataset size and dimensionality
- Number of clusters
- Number of iterations until convergence

## Running with Custom Data

To use your own dataset:

1. Prepare your data as a NumPy array of shape (n_samples, n_features)
2. Save it in a format of your choice (e.g., .npy, .csv)
3. Modify the benchmark script to load your data instead of generating synthetic data

Example with custom data:
```python
# Load custom data
X = np.load('your_data.npy')  # or pd.read_csv('your_data.csv').values

# Run clustering
labels, centroids = kmeans_parallel(X, n_clusters=8, n_threads=4)
```
