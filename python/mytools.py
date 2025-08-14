import h5py
import numpy as np

def read_h5_to_float(file_path, dataset_name=None):
    with h5py.File(file_path, 'r') as f:
        # If dataset_name is not provided, try to get the first dataset
        if dataset_name is None:
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"HDF5 file {file_path} is empty, no datasets available")
            dataset_name = keys[0]
        
        # Read the dataset
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in HDF5 file")
        
        data = f[dataset_name][()]
        
        # Ensure data is 3D
        if len(data.shape) != 3:
            raise ValueError(f"Data is not 3D. Actual dimensions: {data.shape}")
        
        # Convert data to float type
        return data.astype(np.float32)