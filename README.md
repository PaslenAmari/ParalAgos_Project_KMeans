# Parallel K-means Clustering Implementation
The Project for optimization purposes for K-Means clustering algotithm (with benchmarks - use dataset with at least 70000 samples)

## 1. Algorithm and Parallelization Method

### Algorithm
The implementation uses the K-means clustering algorithm with the following steps:
1. Initialize k centroids randomly
2. Assign each data point to the nearest centroid
3. Update centroids by calculating mean of assigned points
4. Repeat steps 2-3 until convergence

### Parallelization Method
- Implementation uses OpenMP via Cython for parallel processing
- Focus on parallelizing the most computationally intensive parts of the algorithm
- Uses parallel for loops with thread-based parallelism

## 2. Instructions for Reproduction

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/parallel-kmeans.git
cd parallel-kmeans

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Compile Cython code
python setup.py build_ext --inplace
```

### Data Preparation
The implementation uses the HandOutlines dataset from the UCR Time Series Archive:
- Dataset is automatically downloaded and preprocessed when running the code
- Format: 1,370 samples with 2,709 features each
- Data is normalized during preprocessing

### Running the Code
```bash
# Run benchmark
python src/benchmark.py
```

## 3. Parallelized Components

The following parts of the algorithm were parallelized:

1. **Point Assignment Phase**:
   - Parallel computation of distances between points and centroids
   - Each thread processes a subset of data points independently

2. **Centroid Update Phase**:
   - Parallel accumulation of point coordinates for new centroids
   - Uses parallel reduction pattern for summing coordinates

## 4. Speedup Analysis

### Benchmark Results
- Sequential implementation time: 6.74 seconds
- Parallel implementation times:
  * 1 thread: 20.08 seconds (speedup: 0.34x)
  * 2 threads: 20.17 seconds (speedup: 0.33x)
  * 4 threads: 20.07 seconds (speedup: 0.34x)
  * 8 threads: 20.09 seconds (speedup: 0.34x)

### Performance Analysis
- Current implementation shows suboptimal performance
- Overhead from thread management exceeds parallel processing benefits
- Memory access patterns may be limiting performance
- Further optimization required for better parallel efficiency

### Speedup Graph
The speedup graph is automatically generated in `results/speedup_plot.png` showing the relationship between number of threads and speedup factor.

## System Requirements
- Python 3.7+
- C++ compiler with OpenMP support
- Required Python packages:
  * NumPy
  * Cython
  * scikit-learn
  * matplotlib

## Possible Improvements
1. Optimize memory access patterns
2. Implement alternative parallelization methods (CUDA, MPI)
3. Increase granularity of parallel tasks
4. Optimize data structure layout for better cache utilization

## Repository Structure
```
parallel-kmeans/
├── src/                 # Source code
│   ├── sequential_kmeans.py
│   ├── parallel_kmeans.pyx
│   ├── data_loader.py
│   └── benchmark.py
├── data/                # Dataset (downloaded automatically)
├── results/             # Benchmark results and plots
└── tests/               # Unit tests
```