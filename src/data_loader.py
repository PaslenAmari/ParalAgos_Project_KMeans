# Import libraries
import numpy as np
import os
from urllib.request import urlretrieve
import zipfile

def download_handoutlines_dataset(data_dir='data'):
    """
    Download HandOutlines dataset from UCR archive if not already present
    """
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # URLs for the dataset UCR archive
    base_url = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
    dataset_url = base_url + "HandOutlines.zip"
    
    # Local paths
    zip_path = os.path.join(data_dir, "HandOutlines.zip")
    dataset_dir = os.path.join(data_dir, "HandOutlines")
    
    # Download and extract if not already present
    if not os.path.exists(dataset_dir):
        print("Downloading HandOutlines dataset...")
        urlretrieve(dataset_url, zip_path)
        
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Remove zip file after extraction
        os.remove(zip_path)
        print("Dataset ready!")

def load_handoutlines_dataset(data_dir='data'):
    """
    Load and preprocess HandOutlines dataset
    
    Returns:
    --------
    X : ndarray of shape (n_samples, n_features)
        The preprocessed dataset
    """
    # Ensure dataset is downloaded
    download_handoutlines_dataset(data_dir)
    
    # Load train and test files
    train_path = os.path.join(data_dir, "HandOutlines/HandOutlines_TRAIN.tsv")
    test_path = os.path.join(data_dir, "HandOutlines/HandOutlines_TEST.tsv")
    
    # Load data, assuming tab-separated values
    train_data = np.loadtxt(train_path, delimiter='\t')
    test_data = np.loadtxt(test_path, delimiter='\t')
    
    # Combine train and test, remove class labels (first column)
    X = np.vstack([train_data[:, 1:], test_data[:, 1:]])
    
    # Normalize the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X

if __name__ == "__main__":
    # Test the data loading
    X = load_handoutlines_dataset()
    print("Dataset statistics:")
    print(f"Mean: {X.mean():.3f}")
    print(f"Std: {X.std():.3f}")
    print(f"Min: {X.min():.3f}")
    print(f"Max: {X.max():.3f}")
