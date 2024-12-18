# Import libraries
import os
import zipfile
from urllib.request import urlretrieve
import shutil
import numpy as np
import pandas as pd
from urllib.error import URLError

def download_handoutlines_dataset(data_dir='data'):
    """
    Download HandOutlines dataset from UCR archive if not already present
    """
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # URLs for the dataset UCR archive
    archive_url = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"
    
    # Local paths
    archive_zip_path = os.path.join(data_dir, "UCRArchive_2018.zip")
    dataset_dir = os.path.join(data_dir, "HandOutlines")
    
    # Download and extract if not already present
    if not os.path.exists(dataset_dir):
        try:
            print("Downloading UCR Archive...")
            urlretrieve(archive_url, archive_zip_path)
        except URLError as e:
            print(f"Error downloading archive: {str(e)}")
            if os.path.exists(archive_zip_path):
                os.remove(archive_zip_path)
            return False
            
        try:
            print("Extracting files...")
            with zipfile.ZipFile(archive_zip_path, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    if file.startswith('UCRArchive_2018/HandOutlines/'):
                        zip_ref.extract(file, data_dir, pwd=b'someone')
        except Exception as e:
            print(f"Error during extraction: {str(e)}")
            if os.path.exists(archive_zip_path):
                os.remove(archive_zip_path)
            return False

        try:
            nested_dir = os.path.join(data_dir, 'UCRArchive_2018', 'HandOutlines')
            if os.path.exists(nested_dir):
                if os.path.exists(dataset_dir):
                    shutil.rmtree(dataset_dir)
                shutil.move(nested_dir, data_dir)
                shutil.rmtree(os.path.join(data_dir, 'UCRArchive_2018'))
        except Exception as e:
            print(f"Error organizing files: {str(e)}")
            return False

        try:
            os.remove(archive_zip_path)
            print("Dataset ready!")
        except Exception as e:
            print(f"Warning: Could not remove zip file: {str(e)}")
    
    return True

def load_handoutlines_dataset(data_dir='data'):
    """
    Load HandOutlines dataset from disk and convert to proper format
    Returns:
        X: numpy array of shape (n_samples, n_features) containing the time series data
        None: if there was an error loading the dataset
    """
    print("Loading dataset...")
    
    # Download dataset if not present
    if not download_handoutlines_dataset(data_dir):
        return None
    
    # Paths to data files
    dataset_dir = os.path.join(data_dir, "HandOutlines")
    train_path = os.path.join(dataset_dir, "HandOutlines_TRAIN.tsv")
    test_path = os.path.join(dataset_dir, "HandOutlines_TEST.tsv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Dataset files not found")
        return None

    try:
        # Load data using pandas
        train_data = pd.read_csv(train_path, header=None, delimiter='\t')
        test_data = pd.read_csv(test_path, header=None, delimiter='\t')
        
        # Convert to numpy arrays, excluding the class labels (first column)
        X_train = train_data.iloc[:, 1:].values
        X_test = test_data.iloc[:, 1:].values
        
        # Combine train and test data
        X = np.vstack([X_train, X_test])
        
        # Convert to float64 and normalize if needed
        X = X.astype(np.float64)
        
        # Optionally normalize the data
        # X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

if __name__ == "__main__":
    # Test the loader
    X = load_handoutlines_dataset()
    if X is not None:
        print("\nDataset statistics:")
        print(f"Mean: {X.mean():.2f}")
        print(f"Std: {X.std():.2f}")
        print(f"Min: {X.min():.2f}")
        print(f"Max: {X.max():.2f}")
        print(f"Data type: {X.dtype}")