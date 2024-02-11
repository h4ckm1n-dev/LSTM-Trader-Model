import h5py

# Function to print section headers with blue color
def print_header(header_text, color='blue'):
    color_code = {
        'blue': '\033[34m',  # Setting color to blue
    }
    reset = '\033[0m'
    print(f"{color_code[color]}{'='*60}\n{header_text}\n{'='*60}{reset}")

# Open the HDF5 file for reading
with h5py.File('/home/h4ckm1n/OpenInterpreter/data.h5', 'r') as hf:
    # Print the keys (dataset names) present in the HDF5 file
    print_header("Keys in the HDF5 file")
    print(list(hf.keys()))
    print("-" * 60)  # Add separation line

    # Inspect each dataset
    for key in hf.keys():
        print("\nDataset:", key)
        dataset = hf[key]
        
        # Print basic information about the dataset
        print_header("Basic Information")
        print("Shape:", dataset.shape)
        print("Datatype:", dataset.dtype)
        
        # If the dataset is small, print its content
        if dataset.size < 100:
            print_header("Content")
            print(dataset[:])
