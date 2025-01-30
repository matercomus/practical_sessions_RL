import h5py

def explore_h5_file(filepath):
    # Open the file
    with h5py.File(filepath, 'r') as f:
        # Print all groups and datasets
        print("\nFile Structure:")
        f.visit(lambda x: print(x))
        
        # Print detailed info for each dataset
        print("\nDataset Details:")
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                dataset = f[key]
                print(f"\nDataset: {key}")
                print(f"Shape: {dataset.shape}")
                print(f"Type: {dataset.dtype}")
                print(f"Data preview: \n{dataset[:]}")
            
            elif isinstance(f[key], h5py.Group):
                print(f"\nGroup: {key}")
                for subkey in f[key].keys():
                    print(f" └── {subkey}")

if __name__ == "__main__":
    file_path = "/Users/floppie/Documents/Project - Reinforcement Learning/practical_sessions_RL/tutorial_wg_1/run_2025-01-30_13-09-32/model.h5"  # Replace with your file path
    explore_h5_file(file_path)