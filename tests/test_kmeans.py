# Import frameworks
import numpy as np
import unittest
from src.sequential_kmeans import kmeans_sequential
from src.parallel_kmeans import kmeans_parallel

class TestKMeans(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create a simple dataset with obvious clusters
        np.random.seed(42)
        n_samples = 300
        
        # Create three distinct clusters
        cluster1 = np.random.normal(0, 0.1, (n_samples, 2))
        cluster2 = np.random.normal(2, 0.1, (n_samples, 2))
        cluster3 = np.random.normal(-2, 0.1, (n_samples, 2))
        
        self.X = np.vstack([cluster1, cluster2, cluster3])
        self.n_clusters = 3

    def test_sequential_kmeans(self):
        """Test sequential k-means implementation"""
        labels, centroids = kmeans_sequential(self.X, n_clusters=self.n_clusters)
        
        # Check shapes
        self.assertEqual(labels.shape[0], self.X.shape[0])
        self.assertEqual(centroids.shape, (self.n_clusters, self.X.shape[1]))
        
        # Check if all cluster labels are present
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), self.n_clusters)
        
        # Check if centroids are different
        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                dist = np.sum((centroids[i] - centroids[j]) ** 2)
                self.assertGreater(dist, 0)

    def test_parallel_kmeans(self):
        """Test parallel k-means implementation"""
        labels, centroids = kmeans_parallel(self.X, n_clusters=self.n_clusters)
        
        # Check shapes
        self.assertEqual(labels.shape[0], self.X.shape[0])
        self.assertEqual(centroids.shape, (self.n_clusters, self.X.shape[1]))
        
        # Check if all cluster labels are present
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), self.n_clusters)
        
        # Check if centroids are different
        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                dist = np.sum((centroids[i] - centroids[j]) ** 2)
                self.assertGreater(dist, 0)

    def test_results_consistency(self):
        """Test if sequential and parallel versions give similar results"""
        # Run both versions
        labels_seq, centroids_seq = kmeans_sequential(self.X, n_clusters=self.n_clusters)
        labels_par, centroids_par = kmeans_parallel(self.X, n_clusters=self.n_clusters)
        
        # Sort centroids for comparison (as cluster order might differ)
        centroids_seq = centroids_seq[np.argsort(centroids_seq[:, 0])]
        centroids_par = centroids_par[np.argsort(centroids_par[:, 0])]
        
        # Check if centroids are similar (allowing for some numerical differences)
        self.assertTrue(np.allclose(centroids_seq, centroids_par, rtol=1e-2, atol=1e-2))

if __name__ == '__main__':
    unittest.main()
