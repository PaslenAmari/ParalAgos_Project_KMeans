# Import frameworks
import numpy as np
import time
import matplotlib.pyplot as plt
from sequential_kmeans import kmeans_sequential
from parallel_kmeans import kmeans_parallel
from data_loader import load_handoutlines_dataset

def run_benchmark(n_threads_list=[1, 2, 4, 8], n_clusters=8):
    """Run benchmark comparing sequential and parallel implementations."""
    # Load real dataset
    print("Loading dataset...")
    X = load_handoutlines_dataset()
    
    results = {
        'n_threads': n_threads_list,
        'times': [],
        'speedups': []
    }
    
    # Run sequential version
    print("\nRunning sequential version...")
    start_time = time.time()
    labels_seq, centroids_seq = kmeans_sequential(X, n_clusters=n_clusters)
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.2f} seconds")
    
    # Run parallel versions
    print("\nRunning parallel versions...")
    for n_threads in n_threads_list:
        print(f"\nTesting with {n_threads} threads...")
        start_time = time.time()
        labels_par, centroids_par = kmeans_parallel(X, n_clusters=n_clusters, n_threads=n_threads)
        parallel_time = time.time() - start_time
        speedup = sequential_time / parallel_time
        
        results['times'].append(parallel_time)
        results['speedups'].append(speedup)
        
        print(f"Parallel time: {parallel_time:.2f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        
        # Verify results match between sequential and parallel
        if n_threads == n_threads_list[0]:  # Only check first parallel run
            centroids_match = np.allclose(np.sort(centroids_seq, axis=0), 
                                        np.sort(centroids_par, axis=0), 
                                        rtol=1e-3)
            print(f"Results match sequential version: {centroids_match}")
    
    plot_results(results)
    return results

def plot_results(results):
    """Plot speedup vs number of threads."""
    plt.figure(figsize=(10, 6))
    plt.plot(results['n_threads'], results['speedups'], 'bo-', label='Actual speedup')
    plt.plot(results['n_threads'], results['n_threads'], 'r--', label='Linear speedup')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.title('K-means Clustering Speedup vs Number of Threads\nHandOutlines Dataset')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/speedup_plot.png')
    plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Run benchmark
    results = run_benchmark()
