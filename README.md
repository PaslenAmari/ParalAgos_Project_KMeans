# ParalAgos_Project_KMeans - Parallel K-means Clustering Implementation
The Project for optimization purposes for K-Means clustering algotithm (with benchmarks - use dataset with at least 70000 samples)

# Parallel K-means Clustering Implementation

This repository contains sequential and parallel implementations of the k-means clustering algorithm using the HandOutlines dataset from the UCR Time Series Archive.

## Project Structure
```
parallel-kmeans/
├── src/            # Source code
├── data/           # Dataset (downloaded automatically)
├── results/        # Benchmark results
└── tests/          # Unit tests
```

## Requirements
- Python 3.7+
- NumPy
- Cython
- scikit-learn
- matplotlib
- OpenMP-compatible C compiler

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/parallel-kmeans
cd parallel-kmeans
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Compile the Cython extension:
```bash
python setup.py build_ext --inplace
```

## Dataset

The implementation uses the HandOutlines dataset from the UCR Time Series Archive:
- 370,500 samples
- 2,709 features per sample
- Automatically downloaded on first run

## Running the Benchmark

```bash
python src/benchmark.py
```

This will:
1. Download the dataset (if not already present)
2. Run both sequential and parallel implementations
3. Generate a speedup plot in the results directory

## Results

The benchmark will create:
- Execution time comparisons
- Speedup measurements
- A plot showing speedup vs number of threads

Results are saved in the `results/` directory.

## Contributing

Feel free to open issues or submit pull requests.

## License

[Your chosen license]
